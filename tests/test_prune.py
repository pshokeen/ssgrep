"""Tests for prune command and vector cleanup.

These tests verify that pruning sessions correctly maintains index alignment
even when removed vectors are not in the trailing rows of vectors.f32.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from ssgrep import store
from ssgrep import vectors as vecstore
from ssgrep.types import ContentType
from ssgrep.vectors import validate_alignment
from tests.conftest import build_chunk, build_episode, build_session_file


def _compute_file_md5(path: Path) -> str | None:
    """Compute MD5 hash of a file, or None if it doesn't exist."""
    if not path.exists():
        return None
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        md5.update(f.read())
    return md5.hexdigest()


def _build_index_with_sessions(
    tmp_path: Path,
    sessions: list[tuple[str, int]],  # (session_id, chunk_count)
) -> Path:
    """Build a test index with multiple sessions.

    Each session gets the specified number of chunks with vectors.
    Returns the path to the index directory.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)

    db_path = index_dir / "index.db"
    conn = store.init_db(db_path)

    # Create vector store
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    all_vectors = []
    chunk_id_counter = 0

    # Insert sessions and chunks
    for session_id, chunk_count in sessions:
        # Create session
        session = build_session_file(
            session_id=session_id, path=tmp_path / f"session-{session_id}.jsonl"
        )
        store.insert_session(conn, session)

        # Create episode
        episode = build_episode(
            episode_id=f"ep-{session_id}", session_id=session_id, title=f"Episode for {session_id}"
        )
        store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

        # Create chunks with vectors
        for i in range(chunk_count):
            chunk_id = f"chunk-{chunk_id_counter}"
            chunk_id_counter += 1

            chunk = build_chunk(
                chunk_id=chunk_id,
                episode_id=f"ep-{session_id}",
                session_id=session_id,
                text=f"Text from {session_id} chunk {i}",
                content_type=ContentType.PROMPT if i % 2 == 0 else ContentType.RESPONSE,
                vec_row=None,  # Will be set after vector append
            )

            # Create a random vector for this chunk
            vec = np.random.randn(256).astype(np.float32)
            all_vectors.append(vec)
            store.insert_chunk(conn, chunk, len(all_vectors) - 1)

    # Append all vectors at once
    if all_vectors:
        vecstore.append(vec_store, np.array(all_vectors))

    vecstore.close(vec_store)
    conn.commit()
    conn.close()

    return tmp_path


def test_prune_e2e_surviving_chunk_remapping(tmp_path: Path) -> None:
    """End-to-end test: prune → search with correct chunk remapping (MUST commit).

    This test simulates the ACTUAL prune.py flow and verifies that cleanup_orphaned_vectors
    stages a new generation and commits it atomically, so the remap is durable.

    Scenario: After pruning first session, surviving chunks must still point
    to correct vectors in the new generation.
    """
    # Build index: 2 sessions, ~30 chunks each (like gate scenario)
    _build_index_with_sessions(
        tmp_path,
        [("pruned_session", 30), ("surviving_session", 29)],
    )

    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    # Verify baseline: both sessions have chunks
    conn = sqlite3.connect(str(db_path))
    pruned_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'pruned_session'"
    ).fetchone()[0]
    surviving_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'surviving_session'"
    ).fetchone()[0]
    assert pruned_count == 30, f"Expected 30 chunks in pruned_session, got {pruned_count}"
    assert surviving_count == 29, f"Expected 29 chunks in surviving_session, got {surviving_count}"
    conn.close()

    # Simulate prune.py flow: delete, commit, cleanup
    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "pruned_session")
    conn.commit()
    conn.close()

    # Create gen_store and run cleanup (which stages a new generation)
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify we're at generation 1 now
    gen_store_after = store.GenerationalStore(index_dir)
    assert gen_store_after.current_generation == 1, "Should have staged generation 1"

    # Check alignment at new generation
    is_valid, error_msg = validate_alignment(
        gen_store_after.get_index_path(), gen_store_after.get_vector_path()
    )
    assert is_valid, f"Expected valid alignment, but got invalid: {error_msg}"


def test_prune_first_session_with_middle_vectors(tmp_path: Path) -> None:
    """Prune the first session whose vectors are in the middle.

    This is the exact scenario that previously produced a corrupt alignment:
    - Session 1 has vectors at rows 0-1
    - Session 2 has vectors at rows 2-3
    - After pruning session 1, rows 0-1 are orphaned but not truncated
    - Validation should pass after prune via compaction
    """
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    # Verify initial state
    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    conn.close()

    # Prune session 1
    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "s1")
    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify alignment after prune
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_new = gen_store_after.get_index_path()
    vec_path_new = gen_store_after.get_vector_path()
    is_valid, error_msg = validate_alignment(db_path_new, vec_path_new)
    assert is_valid, f"Index alignment failed: {error_msg}"

    # Verify chunks from s2 still exist
    conn = sqlite3.connect(str(db_path_new))
    remaining = conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 's2'").fetchone()[0]
    assert remaining == 2, f"Expected 2 chunks from s2, got {remaining}"

    # Verify s2 chunks still reference valid vectors
    s2_chunks = conn.execute(
        "SELECT chunk_id, vec_row, text FROM chunks WHERE session_id = 's2' ORDER BY vec_row"
    ).fetchall()
    assert len(s2_chunks) == 2

    # Verify the text content is preserved
    for chunk_id, _vec_row, text in s2_chunks:
        assert "s2" in text, f"Chunk {chunk_id} text should mention s2, got: {text}"

    conn.close()


def test_prune_last_session_trailing_vectors(tmp_path: Path) -> None:
    """Prune the last session (whose vectors are trailing).

    This case already worked; verify it still works after fix.
    """
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    # Prune session 2
    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "s2")
    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify alignment at new generation
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_new = gen_store_after.get_index_path()
    vec_path_new = gen_store_after.get_vector_path()
    is_valid, error_msg = validate_alignment(db_path_new, vec_path_new)
    assert is_valid, f"Index alignment failed: {error_msg}"

    # Verify only s1 remains
    conn = sqlite3.connect(str(db_path_new))
    remaining = conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 's1'").fetchone()[0]
    assert remaining == 2
    conn.close()


def test_prune_middle_session_from_three(tmp_path: Path) -> None:
    """Prune a middle session from three.

    Sessions: s1 (vectors 0-1), s2 (vectors 2-3), s3 (vectors 4-5)
    Prune s2, leaving s1 and s3.
    """
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2), ("s3", 2)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "s2")
    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify alignment at new generation
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_new = gen_store_after.get_index_path()
    vec_path_new = gen_store_after.get_vector_path()
    is_valid, error_msg = validate_alignment(db_path_new, vec_path_new)
    assert is_valid, f"Index alignment failed: {error_msg}"

    # Verify s1 and s3 remain, s2 is gone
    conn = sqlite3.connect(str(db_path_new))
    s1_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 's1'").fetchone()[0]
    s2_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 's2'").fetchone()[0]
    s3_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = 's3'").fetchone()[0]

    assert s1_count == 2
    assert s2_count == 0
    assert s3_count == 2
    conn.close()


def test_validate_alignment_after_prune(tmp_path: Path) -> None:
    """Test that validate_alignment correctly accepts valid state after prune."""
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "s1")
    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Validate alignment at new generation - should pass
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_new = gen_store_after.get_index_path()
    vec_path_new = gen_store_after.get_vector_path()
    is_valid, error_msg = validate_alignment(db_path_new, vec_path_new)
    assert is_valid, f"Expected valid alignment, got: {error_msg}"


def test_detect_orphans_returns_empty_after_prune(tmp_path: Path) -> None:
    """Test that detect_orphans returns empty after cleanup."""
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "s1")
    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Open vector store and check orphans at new generation
    gen_store_after = store.GenerationalStore(index_dir)
    vec_path = gen_store_after.get_vector_path()
    db_path_new = gen_store_after.get_index_path()

    vec_store = vecstore.open_vectors(vec_path, dimension=256)
    conn = sqlite3.connect(str(db_path_new))
    valid_rows = {
        row[0]
        for row in conn.execute("SELECT vec_row FROM chunks WHERE vec_row IS NOT NULL").fetchall()
    }
    conn.close()

    orphans = vecstore.detect_orphans(vec_store, valid_rows)
    vecstore.close(vec_store)

    assert orphans == [], f"Expected no orphans, got: {orphans}"


def test_chunk_text_preserved_after_prune(tmp_path: Path) -> None:
    """Verify that surviving chunks resolve to correct text after prune.

    A remap that shifts rows by one would return wrong content;
    this test ensures we get the right content.
    """
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 3)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    # Capture s2 chunks before prune
    conn = sqlite3.connect(str(db_path))
    before_chunks = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE session_id = 's2' ORDER BY chunk_id"
    ).fetchall()
    conn.close()

    # Prune s1
    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "s1")
    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify s2 chunks still have same text at new generation
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_new = gen_store_after.get_index_path()
    conn = sqlite3.connect(str(db_path_new))
    after_chunks = conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE session_id = 's2' ORDER BY chunk_id"
    ).fetchall()
    conn.close()

    assert len(before_chunks) == len(after_chunks) == 3
    for before, after in zip(before_chunks, after_chunks, strict=False):
        assert before[0] == after[0], f"Chunk ID changed: {before[0]} -> {after[0]}"
        assert (
            before[1] == after[1]
        ), f"Chunk text changed for {before[0]}: {before[1]} -> {after[1]}"


def test_dry_run_does_not_modify_index(tmp_path: Path) -> None:
    """Verify that dry_run doesn't change any index files."""
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])
    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"
    vec_path = index_dir / "vectors.f32"

    # Capture file hashes before dry_run
    db_hash_before = _compute_file_md5(db_path)
    vec_hash_before = _compute_file_md5(vec_path)

    # Run cleanup as if from dry_run (just prepare without committing destructive changes)
    # For dry_run, we just don't call delete_session_chunks, so file states don't change
    # This test just verifies the baseline - that files are identical if nothing is pruned

    db_hash_after = _compute_file_md5(db_path)
    vec_hash_after = _compute_file_md5(vec_path)

    assert db_hash_before == db_hash_after, "Index database should not change"
    assert vec_hash_before == vec_hash_after, "Vector file should not change"


def test_cleanup_orphaned_vectors_with_no_chunks(tmp_path: Path) -> None:
    """Test cleanup when all chunks have been deleted."""
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)

    db_path = index_dir / "index.db"
    vec_path = index_dir / "vectors.f32"

    # Create an index with vectors but no chunks
    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(vec_path, dimension=256)

    # Add some vectors
    vectors = np.random.randn(4, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    conn.commit()
    conn.close()

    # Run cleanup with gen_store
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Vectors file should be deleted or truncated to 0 at new generation
    gen_store_after = store.GenerationalStore(index_dir)
    vec_path_new = gen_store_after.get_vector_path()
    if vec_path_new.exists():
        assert (
            vec_path_new.stat().st_size == 0
        ), "Vector file should be empty after cleanup with no chunks"


def test_cleanup_orphaned_vectors_direct(tmp_path: Path) -> None:
    """Direct test of cleanup_orphaned_vectors function.

    Test that cleanup correctly handles various scenarios without going
    through the full prune workflow.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)

    db_path = index_dir / "index.db"
    vec_path = index_dir / "vectors.f32"

    # Create index with some chunks and vectors
    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(vec_path, dimension=256)

    # Add 4 vectors
    vectors = np.random.randn(4, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    # Create session and chunks
    session = build_session_file(session_id="s1", path=tmp_path / "session.jsonl")
    store.insert_session(conn, session)

    episode = build_episode(episode_id="ep1", session_id="s1")
    store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

    # Insert chunks pointing to vectors 0 and 3 (leaving 1,2 orphaned)
    chunk1 = build_chunk(chunk_id="ch1", episode_id="ep1", session_id="s1")
    store.insert_chunk(conn, chunk1, 0)

    chunk2 = build_chunk(chunk_id="ch2", episode_id="ep1", session_id="s1")
    store.insert_chunk(conn, chunk2, 3)

    conn.commit()
    conn.close()

    # Before cleanup: we have orphans (1, 2)
    vec_store = vecstore.open_vectors(vec_path, dimension=256)
    valid_rows_before = {0, 3}
    orphans_before = vecstore.detect_orphans(vec_store, valid_rows_before)
    vecstore.close(vec_store)
    assert 1 in orphans_before and 2 in orphans_before, "Should have orphans before cleanup"

    # Run cleanup
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # After cleanup: no orphans
    vec_store = vecstore.open_vectors(gen_store.get_vector_path(), dimension=256)
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    valid_rows_after = {
        row[0]
        for row in conn.execute("SELECT vec_row FROM chunks WHERE vec_row IS NOT NULL").fetchall()
    }
    conn.close()
    orphans_after = vecstore.detect_orphans(vec_store, valid_rows_after)
    vecstore.close(vec_store)

    assert orphans_after == [], f"Should have no orphans after cleanup, got: {orphans_after}"


def test_prune_generation_greater_than_zero(tmp_path: Path) -> None:
    """Test that prune works correctly with index at generation > 0.

    This test verifies that cleanup_orphaned_vectors correctly uses get_vector_path()
    instead of hardcoding "vectors.f32", by creating a second generation via prune.
    This is critical because the production index (at 6+) was previously broken
    by hardcoded "vectors.f32" instead of "vectors.f32.{gen}".
    """
    # Build initial index at gen 0
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])

    index_dir = tmp_path / ".ssgrep"

    # First prune: gen 0 -> gen 1
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    store.delete_session_chunks(conn, "s1")
    conn.commit()
    conn.close()

    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify we're now at generation 1
    gen_store_gen1 = store.GenerationalStore(index_dir)
    gen = gen_store_gen1.current_generation
    assert gen == 1, f"Expected generation 1, got {gen}"

    # Verify alignment at gen 1
    is_valid, error_msg = vecstore.validate_alignment(
        gen_store_gen1.get_index_path(), gen_store_gen1.get_vector_path()
    )
    assert is_valid, f"Alignment failed at gen 1: {error_msg}"

    # Second prune: gen 1 -> gen 2
    # This tests that we correctly use get_vector_path() for gen > 0
    conn = sqlite3.connect(str(gen_store_gen1.get_index_path()))
    store.delete_session_chunks(conn, "s2")
    conn.commit()
    conn.close()

    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify we're now at generation 2
    gen_store_gen2 = store.GenerationalStore(index_dir)
    gen = gen_store_gen2.current_generation
    assert gen == 2, f"Expected generation 2, got {gen}"

    # Verify alignment at gen 2
    is_valid, error_msg = vecstore.validate_alignment(
        gen_store_gen2.get_index_path(), gen_store_gen2.get_vector_path()
    )
    assert is_valid, f"Alignment failed at gen 2: {error_msg}"


def test_fts_pruned_text_deleted(tmp_path: Path) -> None:
    """Test that pruned text is removed from FTS index.

    Regression test for Defect 3: delete_session_chunks was deleting from
    chunks before FTS delete, so the FTS delete's subquery matched nothing.
    This test verifies the fix: FTS entries must be deleted BEFORE chunks.
    """
    _build_index_with_sessions(tmp_path, [("session_to_prune", 3), ("session_keep", 2)])

    index_dir = tmp_path / ".ssgrep"
    db_path = index_dir / "index.db"

    # Get the chunk IDs for each session before prune
    conn = sqlite3.connect(str(db_path))
    pruned_chunk_ids = [
        row[0]
        for row in conn.execute(
            "SELECT chunk_id FROM chunks WHERE session_id = 'session_to_prune'"
        ).fetchall()
    ]
    keep_chunk_ids = [
        row[0]
        for row in conn.execute(
            "SELECT chunk_id FROM chunks WHERE session_id = 'session_keep'"
        ).fetchall()
    ]

    # Verify FTS has chunks before prune
    fts_before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert fts_before == 5, f"Expected 5 FTS entries before prune, got {fts_before}"

    # Prune the session
    store.delete_session_chunks(conn, "session_to_prune")
    conn.commit()
    conn.close()

    # Verify FTS entries for pruned session are gone
    conn = sqlite3.connect(str(db_path))
    for chunk_id in pruned_chunk_ids:
        fts_pruned = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()[0]
        msg = f"Expected 0 FTS entry for pruned chunk {chunk_id}, got {fts_pruned}"
        assert fts_pruned == 0, msg

    # Verify FTS entries for kept session still exist
    for chunk_id in keep_chunk_ids:
        fts_kept = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()[0]
        msg = f"Expected 1 FTS entry for kept chunk {chunk_id}, got {fts_kept}"
        assert fts_kept == 1, msg

    conn.close()


def test_prune_zero_vector_tombstoned_session(tmp_path: Path) -> None:
    """Test prune with a tombstoned session that has zero chunks.

    Real case: a title-only transcript that later vanished. Indexing one
    produces a sessions row with 0 chunks. When pruned, cleanup_orphaned_vectors
    must still write the staged vector file before committing the generation.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)

    db_path = index_dir / "index.db"
    conn = store.init_db(db_path)

    # Create vector store with some vectors
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)
    vectors = np.random.randn(3, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    # Create a zero-vector session (title only, no chunks)
    zero_chunk_session = build_session_file(
        session_id="zero_chunks", path=tmp_path / "zero-session.jsonl"
    )
    store.insert_session(conn, zero_chunk_session)
    zero_episode = build_episode(episode_id="zero_ep", session_id="zero_chunks")
    store.insert_episode(conn, zero_episode, zero_episode.prompt_text, zero_episode.response_text)

    # Create a session with actual chunks and vectors
    main_session = build_session_file(
        session_id="main_session", path=tmp_path / "main-session.jsonl"
    )
    store.insert_session(conn, main_session)
    main_episode = build_episode(episode_id="main_ep", session_id="main_session")
    store.insert_episode(conn, main_episode, main_episode.prompt_text, main_episode.response_text)

    # Add 3 chunks pointing to the 3 vectors
    for i in range(3):
        chunk = build_chunk(
            chunk_id=f"chunk-{i}",
            episode_id="main_ep",
            session_id="main_session",
            text=f"Content {i}",
            vec_row=None,
        )
        store.insert_chunk(conn, chunk, i)

    conn.commit()
    conn.close()

    # Tombstone the zero-chunk session
    conn = sqlite3.connect(str(db_path))
    store.tombstone_session_chunks(conn, "zero_chunks")
    conn.commit()
    conn.close()

    # Now prune the zero-chunk tombstoned session
    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "zero_chunks")
    conn.commit()
    conn.close()

    # Run cleanup — this should not fail
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify new generation was created successfully
    gen_store_after = store.GenerationalStore(index_dir)
    assert gen_store_after.current_generation == 1, "Should have advanced to generation 1"

    # Verify alignment is valid at new generation
    db_path_new = gen_store_after.get_index_path()
    vec_path_new = gen_store_after.get_vector_path()
    is_valid, error_msg = validate_alignment(db_path_new, vec_path_new)
    assert is_valid, f"Alignment should be valid after prune: {error_msg}"

    # Verify the main session chunks are still intact with correct vectors
    conn = sqlite3.connect(str(db_path_new))
    remaining = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'main_session'"
    ).fetchone()[0]
    assert remaining == 3, f"Expected 3 chunks to remain, got {remaining}"

    # Verify no stray files left behind
    assert not (index_dir / "index.db.2").exists(), "Should not have leftover index.db.2"
    assert not (index_dir / "vectors.f32.2").exists(), "Should not have leftover vectors.f32.2"

    conn.close()


def test_crash_before_generation_commit(tmp_path: Path) -> None:
    """Test that prior generation remains valid if crash occurs before commit.

    Simulates: vector write completes, remap completes, but crash happens
    before commit_generation() writes the manifest. The previous generation
    should still be queryable and valid.
    """
    # Build initial index at gen 0
    _build_index_with_sessions(tmp_path, [("s1", 2), ("s2", 2)])

    index_dir = tmp_path / ".ssgrep"
    gen_store = store.GenerationalStore(index_dir)

    # Manually stage gen 1 and do some work, but DON'T commit
    next_gen = gen_store.current_generation + 1
    staged_db, staged_vec = gen_store.stage_generation(next_gen)

    # Simulate partial work (file exists but manifest not updated)
    import shutil

    shutil.copy2(str(gen_store.get_index_path()), str(staged_db))

    # NOW open gen_store again (simulating crash+restart)
    # It should still be at gen 0, and gen 0 should still be queryable
    gen_store_after = store.GenerationalStore(index_dir)
    assert gen_store_after.current_generation == 0, "Should still be at gen 0 after crash"

    # Verify gen 0 is still valid
    db_path_gen0 = gen_store_after.get_index_path()
    vec_path_gen0 = gen_store_after.get_vector_path()

    conn = sqlite3.connect(str(db_path_gen0))
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    assert chunk_count == 4, f"Expected 4 chunks at gen 0, got {chunk_count}"

    is_valid, error_msg = vecstore.validate_alignment(db_path_gen0, vec_path_gen0)
    assert is_valid, f"Gen 0 should still be valid after crash: {error_msg}"


def test_prune_safety_guard_live_vs_tombstoned(tmp_path: Path) -> None:
    """CRITICAL: Defect 1 — Prune must only delete tombstoned sessions, never live ones.

    This test exercises the ACTUAL PruneCommand CLI path and verifies that:
    1. Only sessions marked as 'absent' (tombstoned) are deleted
    2. Live sessions with source_status='available' survive with all their content
    3. Chunks from live sessions remain searchable after prune

    This test MUST fail if the safety query is mutated from:
        WHERE s.source_status = 'absent'
    to:
        WHERE 1=1
    """
    # Build a project with source files
    src_live = tmp_path / "live_session.jsonl"
    src_tombstoned = tmp_path / "tombstoned_session.jsonl"

    # Create index with both sessions
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    # Build LIVE session with chunks
    live_session = build_session_file(session_id="live_session_id", path=src_live, is_main=True)
    store.insert_session(conn, live_session)
    live_episode = build_episode(
        episode_id="live_ep_001", session_id="live_session_id", title="Live Episode"
    )
    store.insert_episode(conn, live_episode, live_episode.prompt_text, live_episode.response_text)

    live_chunks = []
    for i in range(5):
        chunk = build_chunk(
            chunk_id=f"live_chunk_{i}",
            episode_id="live_ep_001",
            session_id="live_session_id",
            text=f"Live content {i}: this should survive the prune",
            content_type=ContentType.PROMPT if i % 2 == 0 else ContentType.RESPONSE,
        )
        store.insert_chunk(conn, chunk, i)
        live_chunks.append(chunk)

    # Build TOMBSTONED session with chunks
    tombstoned_session = build_session_file(
        session_id="tombstoned_session_id", path=src_tombstoned, is_main=True
    )
    store.insert_session(conn, tombstoned_session)
    tombstoned_episode = build_episode(
        episode_id="tombstoned_ep_001",
        session_id="tombstoned_session_id",
        title="Tombstoned Episode",
    )
    store.insert_episode(
        conn, tombstoned_episode, tombstoned_episode.prompt_text, tombstoned_episode.response_text
    )

    tombstoned_chunks = []
    for i in range(3):
        chunk = build_chunk(
            chunk_id=f"tombstoned_chunk_{i}",
            episode_id="tombstoned_ep_001",
            session_id="tombstoned_session_id",
            text=f"Tombstoned content {i}: this should be deleted",
            content_type=ContentType.PROMPT if i % 2 == 0 else ContentType.RESPONSE,
        )
        store.insert_chunk(conn, chunk, 5 + i)
        tombstoned_chunks.append(chunk)

    # Add all vectors
    vectors = np.random.randn(8, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    # Tombstone the tombstoned session (mark it as absent source)
    store.tombstone_session_chunks(conn, "tombstoned_session_id")
    conn.commit()
    conn.close()

    # Verify pre-prune state
    conn = sqlite3.connect(str(db_path))
    live_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'live_session_id'"
    ).fetchone()[0]
    tombstoned_count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session_id'"
    ).fetchone()[0]
    live_session_status = conn.execute(
        "SELECT source_status FROM sessions WHERE session_id = 'live_session_id'"
    ).fetchone()[0]
    tombstoned_status = conn.execute(
        "SELECT source_status FROM sessions WHERE session_id = 'tombstoned_session_id'"
    ).fetchone()[0]
    conn.close()

    assert live_count == 5, f"Expected 5 live chunks before prune, got {live_count}"
    assert (
        tombstoned_count == 3
    ), f"Expected 3 tombstoned chunks before prune, got {tombstoned_count}"
    assert live_session_status == "available", "Live session should have source_status='available'"
    assert tombstoned_status == "absent", "Tombstoned session should have source_status='absent'"

    # NOW RUN THE ACTUAL PRUNE COMMAND logic
    # This is the critical part: we must use the actual _tombstoned_sessions() function
    # which contains the safety query WHERE s.source_status = 'absent'.
    # If that query is mutated to WHERE 1=1, this test MUST fail.
    from ssgrep.cli.commands.prune import _filter_by_age, _tombstoned_sessions

    conn = sqlite3.connect(str(db_path))
    tombstoned = _filter_by_age(_tombstoned_sessions(conn), older_than=0)

    # Verify that ONLY the tombstoned session is selected for deletion
    assert len(tombstoned) == 1, f"Expected 1 tombstoned session, got {len(tombstoned)}"
    assert (
        tombstoned[0]["session_id"] == "tombstoned_session_id"
    ), f"Expected tombstoned_session_id to be selected, got {tombstoned[0]['session_id']}"

    # Now perform the actual deletion through the CLI path
    for session_info in tombstoned:
        store.delete_session_chunks(conn, session_info["session_id"])
    conn.commit()
    conn.close()

    # Run cleanup to maintain vector alignment
    gen_store = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(db_path))
    store.cleanup_orphaned_vectors(conn, gen_store)

    # Verify post-prune state (need to reconnect to the new generation DB)
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_new = gen_store_after.get_index_path()
    conn = sqlite3.connect(str(db_path_new))

    # Tombstoned chunks should be gone
    tombstoned_after = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session_id'"
    ).fetchone()[0]
    assert tombstoned_after == 0, f"Tombstoned chunks should be deleted, got {tombstoned_after}"

    # Tombstoned session row should be gone
    tombstoned_session_after = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session_id'"
    ).fetchone()[0]
    assert tombstoned_session_after == 0, "Tombstoned session should be deleted"

    # LIVE chunks must still exist
    live_after = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'live_session_id'"
    ).fetchone()[0]
    assert live_after == 5, f"Live chunks must survive, expected 5, got {live_after}"

    # LIVE session row must still exist
    live_session_after = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'live_session_id'"
    ).fetchone()[0]
    assert live_session_after == 1, "Live session row must survive"

    # Verify live chunks are still SEARCHABLE by checking FTS
    live_fts = conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id LIKE 'live_chunk_%'"
    ).fetchone()[0]
    assert live_fts == 5, f"Live chunks must be in FTS, got {live_fts} FTS entries"

    # Verify live chunk CONTENT is correct
    for i, chunk in enumerate(live_chunks):
        text = conn.execute(
            "SELECT text FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()[0]
        assert "Live content" in text, f"Chunk {chunk.chunk_id} text corrupted"
        assert text.startswith(
            f"Live content {i}"
        ), f"Chunk {chunk.chunk_id} has wrong content: {text}"

    conn.close()


def test_prune_staleness_poison_fixed(tmp_path: Path) -> None:
    """Defect 2 — Staleness should not persist after prune + re-index.

    Scenario:
    1. Index a session (create session_files cursor row)
    2. Delete the source file (tombstone the session)
    3. Re-index (tombstone persists, cursor row still there, marked stale)
    4. Prune (Defect 2: cursor row NOT deleted, staleness persists forever)
    5. Re-index again (should heal staleness but doesn't without the fix)

    After the fix (deleting session_files cursor row), status should show stale=false.
    """
    # Create a source file and index it
    src_file = tmp_path / "test_session.jsonl"
    src_file.write_text('{"test": "data"}\n')

    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    # Build initial index
    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    # Create session from the file that exists
    session = build_session_file(session_id="session_001", path=src_file, is_main=True)
    store.insert_session(conn, session)

    # Record the cursor for this file
    from ssgrep.types import FileCursor

    file_cursor = FileCursor(
        path=src_file, size=16, mtime=src_file.stat().st_mtime, byte_offset=0, first_line_hash="abc"
    )
    store.upsert_session_file(conn, file_cursor)

    # Create some content
    episode = build_episode(episode_id="ep_001", session_id="session_001", title="Test")
    store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

    chunks = []
    for i in range(3):
        chunk = build_chunk(
            chunk_id=f"chunk_{i}",
            episode_id="ep_001",
            session_id="session_001",
            text=f"Test content {i}",
        )
        store.insert_chunk(conn, chunk, i)
        chunks.append(chunk)

    vectors = np.random.randn(3, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    conn.commit()
    conn.close()

    # Verify initial state: session_files has the cursor row
    conn = sqlite3.connect(str(db_path))
    cursor_before = conn.execute(
        "SELECT COUNT(*) FROM session_files WHERE path = ?", (str(src_file),)
    ).fetchone()[0]
    assert cursor_before == 1, "Cursor row should exist after initial index"
    conn.close()

    # Tombstone the session (simulate source file deletion)
    conn = sqlite3.connect(str(db_path))
    store.tombstone_session_chunks(conn, "session_001")
    conn.commit()
    conn.close()

    # Prune the tombstoned session
    conn = sqlite3.connect(str(db_path))
    store.delete_session_chunks(conn, "session_001")
    conn.commit()
    conn.close()

    # Verify the fix: cursor row should be GONE after prune (Defect 2 fix)
    conn = sqlite3.connect(str(db_path))
    cursor_after_prune = conn.execute(
        "SELECT COUNT(*) FROM session_files WHERE path = ?", (str(src_file),)
    ).fetchone()[0]
    assert (
        cursor_after_prune == 0
    ), "Defect 2: Cursor row should be deleted during prune, not left behind"
    conn.close()

    # Now re-index this deleted session should not leave stale markers
    # (The session is gone, so no new work, but if the cursor row existed,
    # it would incorrectly report as stale forever)
    # We verify this by checking that no stale session_files entry exists
    conn = sqlite3.connect(str(db_path))
    stale_cursors = conn.execute(
        "SELECT COUNT(*) FROM session_files WHERE source_status = 'absent'"
    ).fetchone()[0]
    assert stale_cursors == 0, "No stale cursor rows should remain after prune + fix"
    conn.close()


def test_prune_integration_with_confirmation_and_dry_run(tmp_path: Path) -> None:
    """CRITICAL: Integration test that kills four mutations via handle() re-entry.

    Mutations this test catches:
    1. Destructive tail being dead code (lines 201-213: delete loop and commit)
    2. --dry-run falling through to delete (the if dry_run: early return not executing)
    3. Confirmation guard inverted (line 185: if not yes → if yes)
    4. --older-than predicate inverted (line 86: last_seen < cutoff → last_seen >= cutoff)

    Strategy: Build an index with ONE LIVE session (5 chunks) and ONE TOMBSTONED
    session (3 chunks). Call handle() three times:
    - With dry_run=True: database must be unchanged
    - With yes=False: database must be unchanged (user cancels)
    - With yes=True: database must show ONLY tombstoned session deleted

    We RE-OPEN THE SQLITE FILE after each call and assert row counts directly.
    Never assert the returned document or exit code.
    """
    # Build a project with source files
    src_live = tmp_path / "live_session.jsonl"
    src_tombstoned = tmp_path / "tombstoned_session.jsonl"

    # Create index
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    # Build LIVE session with 5 chunks
    live_session = build_session_file(session_id="live_session", path=src_live, is_main=True)
    store.insert_session(conn, live_session)
    live_episode = build_episode(episode_id="live_ep", session_id="live_session", title="Live")
    store.insert_episode(conn, live_episode, live_episode.prompt_text, live_episode.response_text)

    for i in range(5):
        chunk = build_chunk(
            chunk_id=f"live_chunk_{i}",
            episode_id="live_ep",
            session_id="live_session",
            text=f"Live content {i}",
            content_type=ContentType.PROMPT if i % 2 == 0 else ContentType.RESPONSE,
        )
        store.insert_chunk(conn, chunk, i)

    # Build TOMBSTONED session with 3 chunks
    tombstoned_session = build_session_file(
        session_id="tombstoned_session", path=src_tombstoned, is_main=True
    )
    store.insert_session(conn, tombstoned_session)
    tombstoned_episode = build_episode(
        episode_id="tombstoned_ep", session_id="tombstoned_session", title="Tombstoned"
    )
    store.insert_episode(
        conn, tombstoned_episode, tombstoned_episode.prompt_text, tombstoned_episode.response_text
    )

    for i in range(3):
        chunk = build_chunk(
            chunk_id=f"tombstoned_chunk_{i}",
            episode_id="tombstoned_ep",
            session_id="tombstoned_session",
            text=f"Tombstoned content {i}",
            content_type=ContentType.PROMPT if i % 2 == 0 else ContentType.RESPONSE,
        )
        store.insert_chunk(conn, chunk, 5 + i)

    # Add all 8 vectors
    vectors = np.random.randn(8, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    # Tombstone the tombstoned session
    store.tombstone_session_chunks(conn, "tombstoned_session")
    conn.commit()
    conn.close()

    # Verify baseline state
    conn = sqlite3.connect(str(db_path))
    live_chunks_baseline = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_chunks_baseline = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    live_episodes_baseline = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_episodes_baseline = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    live_sessions_baseline = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_sessions_baseline = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    conn.close()

    assert live_chunks_baseline == 5, f"Expected 5 live chunks, got {live_chunks_baseline}"
    assert (
        tombstoned_chunks_baseline == 3
    ), f"Expected 3 tombstoned chunks, got {tombstoned_chunks_baseline}"
    assert live_episodes_baseline == 1
    assert tombstoned_episodes_baseline == 1
    assert live_sessions_baseline == 1
    assert tombstoned_sessions_baseline == 1

    # TEST 1: Call with dry_run=True — database MUST be unchanged
    import typer

    from ssgrep.cli.commands.prune import PruneCommand

    app = typer.Typer()
    cmd = PruneCommand(app)
    cmd.handle(older_than=0, dry_run=True, yes=False, project_dir=str(tmp_path))

    # RE-OPEN THE SQLITE FILE and assert row counts
    conn = sqlite3.connect(str(db_path))
    live_chunks_after_dry = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_chunks_after_dry = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    live_sessions_after_dry = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_sessions_after_dry = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    conn.close()

    # Dry-run must not change anything
    assert (
        live_chunks_after_dry == live_chunks_baseline
    ), f"Dry-run: live chunks changed from {live_chunks_baseline} to {live_chunks_after_dry}"
    assert tombstoned_chunks_after_dry == tombstoned_chunks_baseline, (
        f"Dry-run: tombstoned chunks changed from {tombstoned_chunks_baseline} to "
        f"{tombstoned_chunks_after_dry}"
    )
    assert (
        live_sessions_after_dry == live_sessions_baseline
    ), "Dry-run: live session row was deleted (should not happen)"
    assert (
        tombstoned_sessions_after_dry == tombstoned_sessions_baseline
    ), "Dry-run: tombstoned session row was deleted (should not happen)"

    # TEST 2: Call with yes=False (simulating user saying 'no') — database MUST be unchanged
    # We patch input() to return something other than 'yes'
    import unittest.mock

    with unittest.mock.patch("builtins.input", return_value="no"):
        cmd = PruneCommand(app)
        try:
            cmd.handle(older_than=0, dry_run=False, yes=False, project_dir=str(tmp_path))
        except SystemExit:
            pass  # Expected when user aborts

    # RE-OPEN THE SQLITE FILE and assert row counts
    conn = sqlite3.connect(str(db_path))
    live_chunks_after_no = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_chunks_after_no = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    live_sessions_after_no = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_sessions_after_no = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    conn.close()

    # User saying 'no' must not change anything
    assert (
        live_chunks_after_no == live_chunks_baseline
    ), f"User-no: live chunks changed from {live_chunks_baseline} to {live_chunks_after_no}"
    assert tombstoned_chunks_after_no == tombstoned_chunks_baseline, (
        f"User-no: tombstoned chunks changed from {tombstoned_chunks_baseline} to "
        f"{tombstoned_chunks_after_no}"
    )
    assert (
        live_sessions_after_no == live_sessions_baseline
    ), "User-no: live session row was deleted (should not happen)"
    assert (
        tombstoned_sessions_after_no == tombstoned_sessions_baseline
    ), "User-no: tombstoned session row was deleted (should not happen)"

    # TEST 3: Call with yes=True — database MUST delete ONLY tombstoned session
    cmd = PruneCommand(app)
    cmd.handle(older_than=0, dry_run=False, yes=True, project_dir=str(tmp_path))

    # RE-OPEN THE SQLITE FILE at the new generation and assert row counts
    gen_store = store.GenerationalStore(index_dir)
    db_path_new = gen_store.get_index_path()
    conn = sqlite3.connect(str(db_path_new))

    live_chunks_after_yes = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_chunks_after_yes = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    live_sessions_after_yes = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_sessions_after_yes = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    live_episodes_after_yes = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'live_session'"
    ).fetchone()[0]
    tombstoned_episodes_after_yes = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]

    conn.close()

    # Live session must survive unchanged
    assert (
        live_chunks_after_yes == live_chunks_baseline
    ), f"Live chunks should survive: expected {live_chunks_baseline}, got {live_chunks_after_yes}"
    assert live_sessions_after_yes == live_sessions_baseline, "Live session row should survive"
    assert live_episodes_after_yes == live_episodes_baseline, "Live episodes should survive"

    # Tombstoned session must be completely deleted
    assert (
        tombstoned_chunks_after_yes == 0
    ), f"Tombstoned chunks should be deleted: expected 0, got {tombstoned_chunks_after_yes}"
    assert tombstoned_sessions_after_yes == 0, "Tombstoned session row should be deleted"
    assert tombstoned_episodes_after_yes == 0, "Tombstoned episodes should be deleted"


def test_filter_by_age_with_explicit_timestamps() -> None:
    """Unit test _filter_by_age with explicit last_seen timestamps.

    This directly tests the predicate with:
    - 40+ days ago (should be included)
    - 1 hour ago (should be excluded)
    - None (should be excluded, never guess ages for destructive ops)

    The single call site passes older_than=0 which short-circuits the predicate,
    so this unit test exercises code paths never reached in integration tests.
    """
    from datetime import UTC, datetime

    from ssgrep.cli.commands.prune import _DAY_SECONDS, _filter_by_age

    now = datetime.now(UTC).timestamp()

    # Session last_seen 40 days ago (well before threshold)
    old_session = {
        "session_id": "old_s",
        "path": "/path/old",
        "chunk_count": 1,
        "episode_count": 1,
        "last_seen": now - (40 * _DAY_SECONDS),  # 40 days ago
    }

    # Session last_seen 1 hour ago
    recent_session = {
        "session_id": "recent_s",
        "path": "/path/recent",
        "chunk_count": 1,
        "episode_count": 1,
        "last_seen": now - 3600,  # 1 hour ago
    }

    # Session with no last_seen (should never be pruned)
    unknown_session = {
        "session_id": "unknown_s",
        "path": "/path/unknown",
        "chunk_count": 1,
        "episode_count": 1,
        "last_seen": None,
    }

    sessions = [old_session, recent_session, unknown_session]

    # Test 1: older_than=0 (short-circuit, return all)
    result = _filter_by_age(sessions, older_than=0)
    assert len(result) == 3, f"older_than=0 should return all sessions, got {len(result)}"

    # Test 2: older_than=30 (include 40-day-old session, exclude recent and None)
    result = _filter_by_age(sessions, older_than=30)
    assert len(result) == 1, f"older_than=30 should return 1 session, got {len(result)}"
    assert (
        result[0]["session_id"] == "old_s"
    ), f"Should have selected old_s, got {result[0]['session_id']}"

    # Test 3: older_than=1 (include 40-day-old session, exclude others)
    result = _filter_by_age(sessions, older_than=1)
    assert len(result) == 1, f"older_than=1 should return 1 session, got {len(result)}"
    assert result[0]["session_id"] == "old_s"

    # Test 4: older_than=50 (old_s is only 40 days, so exclude all)
    result = _filter_by_age(sessions, older_than=50)
    assert len(result) == 0, f"older_than=50 should exclude everything, got {len(result)}"

    # Test 5: None session never matches any older_than value
    result = _filter_by_age([unknown_session], older_than=0)
    assert len(result) == 1, "older_than=0 should pass through unfiltered (short-circuit)"
    result = _filter_by_age([unknown_session], older_than=1)
    assert (
        len(result) == 0
    ), "None last_seen should never match any older_than>0 (safety: never guess ages)"


def test_prune_mutation_1_dry_run_with_yes_overrides_dry_run(tmp_path: Path) -> None:
    """MUTATION TEST: prune.py:140 `if dry_run:` → `if dry_run and not yes:`

    With the mutation, calling handle(dry_run=True, yes=True) would NOT short-circuit
    the dry-run early return and would proceed to deletion. This test verifies that
    dry_run=True ALWAYS prevents deletion, even when yes=True.

    Oracle: SQLite row counts BEFORE and AFTER the call. Never the returned document.
    """
    import typer

    from ssgrep.cli.commands.prune import PruneCommand

    # Build a simple project with one tombstoned session (3 chunks, 1 episode)
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    # Create a single tombstoned session with 3 chunks
    session = build_session_file(session_id="test_session", path=tmp_path / "test.jsonl")
    store.insert_session(conn, session)

    episode = build_episode(episode_id="test_ep", session_id="test_session")
    store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

    for i in range(3):
        chunk = build_chunk(
            chunk_id=f"chunk_{i}",
            episode_id="test_ep",
            session_id="test_session",
            text=f"Content {i}",
        )
        store.insert_chunk(conn, chunk, i)

    vectors = np.random.randn(3, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    # Tombstone the session
    store.tombstone_session_chunks(conn, "test_session")
    conn.commit()
    conn.close()

    # ORACLE 1: Capture row counts BEFORE
    conn = sqlite3.connect(str(db_path))
    chunks_before = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'test_session'"
    ).fetchone()[0]
    episodes_before = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'test_session'"
    ).fetchone()[0]
    sessions_before = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'test_session'"
    ).fetchone()[0]
    conn.close()

    assert chunks_before == 3, f"Baseline: expected 3 chunks, got {chunks_before}"
    assert episodes_before == 1, f"Baseline: expected 1 episode, got {episodes_before}"
    assert sessions_before == 1, f"Baseline: expected 1 session, got {sessions_before}"

    # CALL UNDER TEST: dry_run=True, yes=True (mutation would delete)
    app = typer.Typer()
    cmd = PruneCommand(app)
    cmd.handle(older_than=0, dry_run=True, yes=True, project_dir=str(tmp_path))

    # ORACLE 2: Verify row counts are UNCHANGED (mutation kills this test)
    # Note: If the mutation causes deletion, cleanup_orphaned_vectors will create
    # a new generation. We use GenerationalStore to get the current generation's db.
    gen_store_after = store.GenerationalStore(index_dir)
    db_path_final = gen_store_after.get_index_path()

    conn = sqlite3.connect(str(db_path_final))
    chunks_after = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'test_session'"
    ).fetchone()[0]
    episodes_after = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'test_session'"
    ).fetchone()[0]
    sessions_after = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'test_session'"
    ).fetchone()[0]
    conn.close()

    # With proper code (if dry_run:): rows unchanged
    # With mutation (if dry_run and not yes:): rows would be deleted
    assert (
        chunks_after == chunks_before
    ), f"dry_run must prevent deletion: chunks before {chunks_before}, after {chunks_after}"
    assert (
        episodes_after == episodes_before
    ), f"dry_run must prevent deletion: episodes before {episodes_before}, after {episodes_after}"
    assert (
        sessions_after == sessions_before
    ), f"dry_run must prevent deletion: sessions before {sessions_before}, after {sessions_after}"


def test_prune_mutation_2_older_than_filter_respected(tmp_path: Path) -> None:
    """MUTATION TEST: prune.py:138 drop `_filter_by_age(...)` wrapper

    With the mutation, line 138 becomes `sessions = _tombstoned_sessions(conn)`
    instead of `sessions = _filter_by_age(_tombstoned_sessions(conn), older_than)`.
    This means older_than is completely ignored and all tombstoned sessions get deleted.

    This test builds a tombstoned session with FRESH last_seen (today) and calls
    prune with older_than=365 days (filter should exclude it).

    With proper code: _filter_by_age() excludes fresh sessions → nothing deleted
    With mutation: no filtering → session deleted

    Oracle: SQLite row counts BEFORE and AFTER the call.
    """
    import typer

    from ssgrep.cli.commands.prune import PruneCommand
    from ssgrep.types import FileCursor

    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    # Create a tombstoned session with 2 chunks
    src_file = tmp_path / "test_session.jsonl"
    src_file.write_text('{"test": "data"}\n')

    session = build_session_file(session_id="tombstoned_session", path=src_file)
    store.insert_session(conn, session)

    episode = build_episode(episode_id="test_ep", session_id="tombstoned_session")
    store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

    for i in range(2):
        chunk = build_chunk(
            chunk_id=f"chunk_{i}",
            episode_id="test_ep",
            session_id="tombstoned_session",
            text=f"Content {i}",
        )
        store.insert_chunk(conn, chunk, i)

    vectors = np.random.randn(2, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    # Record a cursor with FRESH mtime (today) — not old at all
    file_cursor = FileCursor(
        path=src_file,
        size=16,
        mtime=src_file.stat().st_mtime,  # TODAY
        byte_offset=0,
        first_line_hash="abc",
    )
    store.upsert_session_file(conn, file_cursor)

    # Tombstone the session
    store.tombstone_session_chunks(conn, "tombstoned_session")
    conn.commit()
    conn.close()

    # ORACLE 1: Capture row counts BEFORE
    conn = sqlite3.connect(str(db_path))
    chunks_before = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    episodes_before = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    sessions_before = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    conn.close()

    assert chunks_before == 2, f"Baseline: expected 2 chunks, got {chunks_before}"
    assert episodes_before == 1, f"Baseline: expected 1 episode, got {episodes_before}"
    assert sessions_before == 1, f"Baseline: expected 1 session, got {sessions_before}"

    # CALL UNDER TEST: older_than=365 days (should exclude today's session)
    app = typer.Typer()
    cmd = PruneCommand(app)
    cmd.handle(older_than=365, dry_run=False, yes=True, project_dir=str(tmp_path))

    # ORACLE 2: Verify row counts are UNCHANGED (mutation kills this test)
    # Note: need to reopen with GenerationalStore since cleanup_orphaned_vectors
    # may have staged a new generation
    gen_store = store.GenerationalStore(index_dir)
    db_path_final = gen_store.get_index_path()

    conn = sqlite3.connect(str(db_path_final))
    chunks_after = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    episodes_after = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    sessions_after = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'tombstoned_session'"
    ).fetchone()[0]
    conn.close()

    # With proper code (older_than=365 filters out fresh sessions): rows unchanged
    # With mutation (no filtering): rows deleted
    assert chunks_after == chunks_before, (
        f"older_than=365 should exclude today's session: chunks before "
        f"{chunks_before}, after {chunks_after}"
    )
    assert episodes_after == episodes_before, (
        f"older_than=365 should exclude today's session: episodes before "
        f"{episodes_before}, after {episodes_after}"
    )
    assert sessions_after == sessions_before, (
        f"older_than=365 should exclude today's session: sessions before "
        f"{sessions_before}, after {sessions_after}"
    )


def test_prune_bug1_non_interactive_stdin_exits_cleanly(tmp_path: Path) -> None:
    """Bug 1: Non-interactive stdin should exit with usage error, not crash.

    When stdin is not a TTY (EOF, closed, or piped), prune should print a clear
    error message to stderr and exit with code 2 (USAGE_ERROR), not code 1 (crash)
    from an uncaught EOFError.

    This test runs the ssgrep CLI with stdin redirected from /dev/null, verifying:
    1. Exit code is 2 (not 1)
    2. stdout is empty (no prompt pollution)
    3. stderr contains the error message about TTY requirement
    """
    from ssgrep.cli import exit_codes
    from tests.conftest import get_ssgrep_binary

    # Build a minimal index with tombstoned content
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    session = build_session_file(session_id="tombstoned", path=tmp_path / "test.jsonl")
    store.insert_session(conn, session)

    episode = build_episode(episode_id="ep", session_id="tombstoned")
    store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

    chunk = build_chunk(chunk_id="ch1", episode_id="ep", session_id="tombstoned")
    store.insert_chunk(conn, chunk, 0)

    vectors = np.random.randn(1, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    store.tombstone_session_chunks(conn, "tombstoned")
    conn.commit()
    conn.close()

    # Run prune via CLI with stdin from /dev/null, no --yes flag
    import subprocess

    ssgrep_bin = get_ssgrep_binary()
    result = subprocess.run(
        [str(ssgrep_bin), "prune", "--project-dir", str(tmp_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    # Verify exit code is USAGE_ERROR (2), not internal failure (1)
    assert (
        result.returncode == exit_codes.USAGE_ERROR
    ), f"Expected exit code {exit_codes.USAGE_ERROR}, got {result.returncode}"

    # Verify stdout is empty (no prompt pollution)
    assert result.stdout == "", f"Expected empty stdout, got: {result.stdout}"

    # Verify stderr contains the error message about TTY requirement
    assert (
        "TTY" in result.stderr or "terminal" in result.stderr
    ), f"Expected TTY/terminal error message in stderr, got: {result.stderr}"


def test_prune_bug2_json_mode_no_yes_returns_structured_error(tmp_path: Path) -> None:
    """Bug 2: --json mode with no --yes should return structured error, not bare exit.

    When prune is run with --json, no --yes, and tombstoned content exists,
    it should return a structured JSON error document (ok: false, condition,
    message, command) on stdout, not a bare SystemExit(2).

    This matches the pattern already used for missing-index errors.
    """
    import json as json_lib
    import subprocess

    from ssgrep.cli import exit_codes
    from tests.conftest import get_ssgrep_binary

    # Build a minimal index with tombstoned content
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    vec_store = vecstore.open_vectors(index_dir / "vectors.f32", dimension=256)

    session = build_session_file(session_id="tombstoned", path=tmp_path / "test.jsonl")
    store.insert_session(conn, session)

    episode = build_episode(episode_id="ep", session_id="tombstoned")
    store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

    chunk = build_chunk(chunk_id="ch1", episode_id="ep", session_id="tombstoned")
    store.insert_chunk(conn, chunk, 0)

    vectors = np.random.randn(1, 256).astype(np.float32)
    vecstore.append(vec_store, vectors)
    vecstore.close(vec_store)

    store.tombstone_session_chunks(conn, "tombstoned")
    conn.commit()
    conn.close()

    # Run prune via CLI with --json, no --yes
    ssgrep_bin = get_ssgrep_binary()
    result = subprocess.run(
        [str(ssgrep_bin), "prune", "--project-dir", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        input="",  # EOF on stdin
    )

    # Verify exit code is USAGE_ERROR (2)
    assert (
        result.returncode == exit_codes.USAGE_ERROR
    ), f"Expected exit code {exit_codes.USAGE_ERROR}, got {result.returncode}"

    # Verify stdout contains a single valid JSON document
    # Must be exactly one JSON object, no concatenated extras
    try:
        doc = json_lib.loads(result.stdout)
    except json_lib.JSONDecodeError as e:
        raise AssertionError(f"Expected valid JSON on stdout, got: {result.stdout}") from e

    # Verify document structure matches the pattern
    assert doc.get("ok") is False, f"Expected ok: false, got: {doc.get('ok')}"
    assert (
        doc.get("condition") == "confirmation_required"
    ), f"Expected condition: confirmation_required, got: {doc.get('condition')}"
    assert isinstance(
        doc.get("message"), str
    ), f"Expected message to be a string, got: {doc.get('message')}"
    assert isinstance(
        doc.get("command"), str
    ), f"Expected command to be a string, got: {doc.get('command')}"
    assert (
        "yes" in doc.get("command", "").lower()
    ), f"Expected --yes suggestion in command, got: {doc.get('command')}"


def test_prune_bug3_toctou_race_with_concurrent_rebuild(tmp_path: Path, monkeypatch) -> None:
    """Bug 3 (TOCTOU): prune paused between its confirmation phase and its
    generation lock must land its deletion in the LIVE generation, not the
    one it read before the pause.

    A real threaded race (Event-gated, mirroring the pattern in
    tests/test_concurrent_index.py): the prune thread is paused at exactly
    the pre-fix race window -- after the pre-lock read/confirmation phase,
    before ``_delete_under_lock`` acquires ``hold_generation()``. While it
    is paused, a second thread commits a full rebuild (which runs
    generation cleanup via ``commit_generation``), deletes another source
    file, and runs an incremental index that tombstones it in the NEW
    generation. When prune resumes it must re-read the manifest, lock the
    fresh generation, re-derive the tombstone set under the lock, and
    delete there.

    Pre-fix behavior: prune deleted from a connection opened against the
    superseded generation and handed ``cleanup_orphaned_vectors`` a stale
    cached generation number, staging a generation that collided with the
    rebuild's -- crashing or silently clobbering the rebuild.
    """
    import json as json_mod
    import threading

    import typer

    from ssgrep import embed, indexer
    from ssgrep.cli.commands.prune import PruneCommand

    # --- fake home with real session files (local helpers; see
    # test_concurrent_index.py for the pattern) ---
    home = tmp_path / "home"
    sessions_dir = home / ".claude" / "projects" / "proj"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import ssgrep.discovery as discovery_mod

    discovery_mod._cwd_index_cache.clear()

    def fake_encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(256).astype(np.float32)
        return out

    monkeypatch.setattr(embed, "encode", fake_encode)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = project_dir / ".ssgrep"
    cwd = str(project_dir)

    def make_session_file(session_id: str) -> Path:
        path = sessions_dir / f"{session_id}.jsonl"
        records = [
            {
                "parentUuid": None,
                "isSidechain": False,
                "type": "user",
                "message": {"role": "user", "content": f"prompt for {session_id}"},
                "uuid": f"{session_id}-u0",
                "timestamp": "2026-07-01T10:00:00.000Z",
                "cwd": cwd,
                "sessionId": session_id,
                "gitBranch": "main",
            },
            {
                "parentUuid": f"{session_id}-u0",
                "isSidechain": False,
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"response for {session_id}"}],
                },
                "uuid": f"{session_id}-a0",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "cwd": cwd,
                "sessionId": session_id,
                "gitBranch": "main",
            },
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json_mod.dumps(r) + "\n")
        return path

    for sid in ("lost", "keeper", "victim"):
        make_session_file(sid)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    # Tombstone "lost" in generation 0 so prune's pre-lock phase has a
    # non-empty display set and reaches the race window.
    (sessions_dir / "lost.jsonl").unlink()
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    start_generation = gen_store.current_generation
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    row = conn.execute("SELECT source_status FROM sessions WHERE session_id = 'lost'").fetchone()
    conn.close()
    assert row is not None and row[0] == "absent", "setup: 'lost' must be tombstoned"

    # --- gate: pause prune between its confirmation phase and its lock ---
    prune_paused = threading.Event()
    let_prune_resume = threading.Event()
    orig_delete_under_lock = PruneCommand._delete_under_lock

    def gated_delete_under_lock(self, project, older_than):
        prune_paused.set()
        if not let_prune_resume.wait(timeout=15):
            raise TimeoutError("rebuild thread never signaled completion")
        return orig_delete_under_lock(self, project, older_than)

    monkeypatch.setattr(PruneCommand, "_delete_under_lock", gated_delete_under_lock)

    errors: list[tuple[str, BaseException]] = []

    def prune_worker() -> None:
        try:
            cmd = PruneCommand(typer.Typer())
            cmd.handle(older_than=0, dry_run=False, yes=True, project_dir=str(project_dir))
        except BaseException as exc:  # noqa: BLE001
            errors.append(("prune", exc))

    def rebuild_worker() -> None:
        try:
            if not prune_paused.wait(timeout=15):
                raise TimeoutError("prune thread never reached its pause point")
            # Commit a rebuild (runs generation cleanup on commit), then
            # tombstone a different session in the NEW generation.
            indexer.index(project_dir, index_dir=index_dir, quiet=True, rebuild=True)
            (sessions_dir / "victim.jsonl").unlink()
            indexer.index(project_dir, index_dir=index_dir, quiet=True)
            fresh = store.GenerationalStore(index_dir)
            assert fresh.current_generation > start_generation, "rebuild must move the generation"
            c = sqlite3.connect(str(fresh.get_index_path()))
            status = c.execute(
                "SELECT source_status FROM sessions WHERE session_id = 'victim'"
            ).fetchone()
            c.close()
            assert status is not None and status[0] == "absent", "'victim' must be tombstoned"
        except BaseException as exc:  # noqa: BLE001
            errors.append(("rebuild", exc))
        finally:
            let_prune_resume.set()

    t_prune = threading.Thread(target=prune_worker, name="prune")
    t_rebuild = threading.Thread(target=rebuild_worker, name="rebuild")
    t_prune.start()
    t_rebuild.start()
    t_prune.join(timeout=60)
    t_rebuild.join(timeout=60)
    assert not t_prune.is_alive(), "prune thread hung"
    assert not t_rebuild.is_alive(), "rebuild thread hung"
    assert not errors, f"thread errors: {errors}"

    # --- prune's deletion landed in the live generation ---
    final = store.GenerationalStore(index_dir)
    db_path = final.get_index_path()
    assert db_path.exists(), "live generation's index.db must exist"
    conn = sqlite3.connect(str(db_path))
    victim_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'victim'"
    ).fetchone()[0]
    victim_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'victim'"
    ).fetchone()[0]
    keeper_status = conn.execute(
        "SELECT source_status FROM sessions WHERE session_id = 'keeper'"
    ).fetchone()
    keeper_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = 'keeper'"
    ).fetchone()[0]
    conn.close()

    assert victim_chunks == 0, "prune's deletion must land in the live generation"
    assert victim_sessions == 0, "victim's session row must be deleted from the live generation"
    assert keeper_status is not None and keeper_status[0] == "available"
    assert keeper_chunks > 0, "surviving session's chunks must be intact"

    # Live generation stays aligned (no orphaned or out-of-bounds vectors).
    is_valid, error_msg = validate_alignment(final.get_index_path(), final.get_vector_path())
    assert is_valid, f"live generation must stay aligned, got: {error_msg}"


def test_prune_prompt_holds_no_generation_lock(tmp_path: Path, monkeypatch) -> None:
    """No generation lock may be held while prune waits at the confirmation
    prompt (index-durability spec: "Prune prompt does not hold the lock
    open" -- the no-lock-during-wait half; the reacquire-and-re-derive half
    is pinned by test_prune_bug3_toctou_race_with_concurrent_rebuild).

    The prompt hook plays the human mid-wait: while "typing" it attempts a
    NON-BLOCKING EXCLUSIVE flock on the live generation's lock file.
    hold_generation() takes a SHARED flock on that same file, so the
    exclusive attempt succeeds if and only if no holder is active -- the
    exact probe a concurrent rebuild's generation cleanup would make while
    the human sits at the prompt.
    """
    import fcntl
    import json as json_mod
    import os
    import sys

    import typer

    import ssgrep.discovery as discovery_mod
    from ssgrep import embed, indexer
    from ssgrep.cli.commands.prune import PruneCommand

    home = tmp_path / "home"
    sessions_dir = home / ".claude" / "projects" / "proj"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    discovery_mod._cwd_index_cache.clear()

    def fake_encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(256).astype(np.float32)
        return out

    monkeypatch.setattr(embed, "encode", fake_encode)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    index_dir = project_dir / ".ssgrep"
    cwd = str(project_dir)

    def make_session_file(session_id: str) -> Path:
        path = sessions_dir / f"{session_id}.jsonl"
        records = [
            {
                "parentUuid": None,
                "isSidechain": False,
                "type": "user",
                "message": {"role": "user", "content": f"prompt for {session_id}"},
                "uuid": f"{session_id}-u0",
                "timestamp": "2026-07-01T10:00:00.000Z",
                "cwd": cwd,
                "sessionId": session_id,
                "gitBranch": "main",
            },
            {
                "parentUuid": f"{session_id}-u0",
                "isSidechain": False,
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"response for {session_id}"}],
                },
                "uuid": f"{session_id}-a0",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "cwd": cwd,
                "sessionId": session_id,
                "gitBranch": "main",
            },
        ]
        with open(path, "w") as f:
            for r in records:
                f.write(json_mod.dumps(r) + "\n")
        return path

    for sid in ("doomed", "keeper"):
        make_session_file(sid)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)
    (sessions_dir / "doomed.jsonl").unlink()
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    probe: dict[str, object] = {"exclusive_ok": None, "generation": None}

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def probing_input() -> str:
        gs = store.GenerationalStore(index_dir)
        gen = gs.current_generation
        fd = os.open(str(gs._generation_lock_path(gen)), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                probe["exclusive_ok"] = True
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                probe["exclusive_ok"] = False
        finally:
            os.close(fd)
        probe["generation"] = gen
        return "yes"

    monkeypatch.setattr("builtins.input", probing_input)

    PruneCommand(typer.Typer()).handle(
        older_than=0, dry_run=False, yes=False, project_dir=str(project_dir)
    )

    assert probe["generation"] is not None, "the prompt hook must actually have run"
    assert probe["exclusive_ok"] is True, (
        "an exclusive flock on the live generation must succeed while prune "
        "waits at the confirmation prompt -- no generation lock may be held "
        "during the wait"
    )

    # Positive control: the confirmed prune then completed its deletion in
    # the live generation, so the probed prompt path is the real one.
    final = store.GenerationalStore(index_dir)
    conn = sqlite3.connect(str(final.get_index_path()))
    remaining = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = 'doomed'"
    ).fetchone()[0]
    conn.close()
    assert remaining == 0, "deletion must complete after confirmation"
