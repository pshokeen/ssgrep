"""Re-vectorization: re-embed stored chunks without re-parsing transcripts.

Swapping the embedding model requires re-computing vectors for all chunks
already stored in the index, without re-reading or re-parsing any transcripts.

The operation reads chunk text from the database, re-encodes all chunks using
the configured embedding model, truncates and rewrites the vector matrix, and
updates the model id and dimension in meta. Chunk ids and episode metadata
remain byte-identical.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ssgrep import embed, store, vectors
from ssgrep.types import IndexNotFoundError, IndexNotReadyError, IndexStats


def _counts(db_path: Path) -> tuple[int, int, int]:
    """(sessions, episodes, chunks) read through the live database's WAL."""
    conn = store.sqlite3.connect(str(db_path))
    try:
        counts = [
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("sessions", "episodes", "chunks")
        ]
        return counts[0], counts[1], counts[2]
    finally:
        conn.close()


def revectorize(
    project_dir: Path,
    *,
    quiet: bool = False,
    index_dir: Path | None = None,
) -> IndexStats:
    """Re-embed all stored chunks under the current embedding model.

    Reads chunk text from the index database, re-encodes all chunks, and
    rewrites the vector matrix. Chunk ids and episode metadata are
    byte-identical afterwards. The parse stage does not run.

    Mirrors ssgrep.api.index()'s signature for consistency; index_dir is an
    extra keyword-only override defaulting to project_dir/".ssgrep" that lets
    callers keep index state separate from the scanned project, mainly for
    tests.

    Args:
        project_dir: Path to the project directory.
        quiet: If True, suppress progress output.
        index_dir: Optional override for the .ssgrep directory location.

    Returns:
        IndexStats with updated model binding and chunk counts.

    Raises:
        IndexNotFoundError: If no index exists for the project.
    """
    index_dir = index_dir or (project_dir / ".ssgrep")
    gen_store = store.GenerationalStore(index_dir)
    live_db = gen_store.get_index_path()

    if not live_db.exists():
        raise IndexNotFoundError(f"No index found at {live_db}")

    model_id, dimension = embed.get_model_info()

    # Stage a new generation for atomic commit
    next_gen = gen_store.current_generation + 1
    db_path, vec_path = gen_store.stage_generation(next_gen)

    # Copy the database to the new generation. Must read through the WAL --
    # a plain file copy silently drops every commit since the last
    # checkpoint, and commit_generation() below then unlinks the original.
    # See store.copy_database().
    store.copy_database(live_db, db_path)
    live_counts = _counts(live_db)

    # Open the copied database for modification
    conn = store.init_db(db_path)

    # Read all chunks from the database, sorted by vec_row to preserve order
    chunk_rows = conn.execute(
        "SELECT chunk_id, text, vec_row FROM chunks ORDER BY vec_row ASC"
    ).fetchall()

    if not quiet:
        print(f"ssgrep: re-vectorizing {len(chunk_rows)} chunk(s)")

    # Re-encode all chunks
    if chunk_rows:
        texts = [row[1] for row in chunk_rows]
        vecs = embed.encode(texts)

        # Clear the vector store and write new vectors
        vec_store = vectors.open_vectors(vec_path, dimension=dimension)

        # Append the new vectors and record their row numbers
        new_vec_rows = vectors.append(vec_store, vecs)

        # Update chunk rows with their new vec_row values
        for (chunk_id, _, _), new_vec_row in zip(chunk_rows, new_vec_rows, strict=True):
            conn.execute(
                "UPDATE chunks SET vec_row = ? WHERE chunk_id = ?",
                (new_vec_row, chunk_id),
            )

        vectors.close(vec_store)

    # Update model binding in meta
    store.set_meta(conn, "model_id", model_id)
    store.set_meta(conn, "vector_dimension", str(dimension))

    now = datetime.now(UTC)
    store.set_meta(conn, "last_index_time", now.isoformat())
    conn.commit()

    # Read final stats
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    tombstoned_sources, tombstoned_chunks = store.get_tombstone_stats(conn)

    conn.close()

    # Backstop: re-vectorizing re-embeds text that is already stored, so it
    # must not add or drop a single row. Any inequality means the staged
    # generation is not a faithful copy of the live one -- and committing it
    # unlinks the live one, so this is the last moment the difference can be
    # noticed. rebuild_guard gates indexer.index()'s rebuild but has never
    # covered this path, which replaces the live generation just as
    # completely. Exact equality, not a ratio: unlike a rebuild there is no
    # legitimate shrink to tolerate here.
    staged_counts = (session_count, episode_count, chunk_count)
    if staged_counts != live_counts:
        gen_store.discard_generation(next_gen)
        raise IndexNotReadyError(
            "Refusing to commit a re-vectorized index that does not match the "
            f"existing one: {live_counts[0]} sessions, {live_counts[1]} episodes, "
            f"{live_counts[2]} chunks became {staged_counts[0]}, {staged_counts[1]}, "
            f"{staged_counts[2]}. Re-vectorizing re-embeds stored text and must "
            "not change any count. Your existing index has NOT been modified.",
            command=None,
        )

    # Atomically commit the new generation
    gen_store.commit_generation(next_gen)

    return IndexStats(
        session_count=session_count,
        episode_count=episode_count,
        chunk_count=chunk_count,
        index_size_bytes=_dir_size(index_dir),
        last_index_time=now,
        model_id=model_id,
        vector_dimension=dimension,
        skipped_records=0,
        malformed_records=0,
        schema_version=store.SCHEMA_VERSION,
        tombstoned_source_count=tombstoned_sources,
        tombstoned_chunk_count=tombstoned_chunks,
        index_exists=True,
    )


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
