"""Vector compaction and orphan cleanup."""

from __future__ import annotations

import os
import shutil
import sqlite3

import numpy as np

from ssgrep.store.generations import GenerationalStore
from ssgrep.store.schema import copy_database


def cleanup_orphaned_vectors(
    conn: sqlite3.Connection,
    gen_store: GenerationalStore,
    generation: int | None = None,
) -> None:
    """Compact vectors to remove orphaned rows after prune.

    After deleting chunks, vector rows may be orphaned (no chunk references
    them). This stages a new generation with compacted vectors, remaps all
    chunk.vec_row references to their new positions, then atomically commits
    the staged generation. This maintains absolute alignment with no crash
    windows: every chunk references a valid vector and no orphaned rows exist.

    Args:
        conn: Open database connection with remaining chunks (will be closed).
        gen_store: GenerationalStore for atomic staged commits.
        generation: The generation the caller is operating on (e.g. the one it
            holds via ``hold_generation()``). When None, falls back to the
            store's cached ``current_generation``. Callers that read the live
            generation under a lock must pass it explicitly so a concurrent
            rebuild committing mid-flight cannot redirect the compaction onto
            a stale cached value.
    """
    current_gen = generation if generation is not None else gen_store.current_generation
    next_gen = current_gen + 1

    # Stage the next generation
    staged_db_path, staged_vec_path = gen_store.stage_generation(next_gen)

    # Copy the current generation's index.db to the staged one
    current_db_path = gen_store.get_index_path(current_gen)
    current_vec_path = gen_store.get_vector_path(current_gen)

    # The caller's deletions have been committed on `conn` but, under
    # journal_mode=WAL, may still live only in index.db-wal along with every
    # other commit since the last checkpoint. A plain file copy takes the
    # main file alone and drops all of it; commit_generation() below then
    # unlinks the original and orphans the WAL. Measured: `ssgrep prune`
    # reporting "Pruned 1 tombstoned sessions" while destroying six others.
    # copy_database() reads through the WAL. See store.copy_database().
    try:
        copy_database(current_db_path, staged_db_path)
    except (OSError, sqlite3.Error):
        # If copy fails, bail silently — the live generation is still intact
        return

    # Close the connection to the live DB so we can open the staged one
    conn.close()

    # Open the staged database and perform compaction there
    staged_conn = sqlite3.connect(str(staged_db_path))
    try:
        # Derive vector_dimension from VECTOR_ROW_BYTES to avoid duplication
        vector_dimension = gen_store.VECTOR_ROW_BYTES // 4

        if not current_vec_path.exists():
            # No vectors to compact; still need to commit the DB copy
            gen_store.commit_generation(next_gen, vector_row_count=0)
            return

        # Get all chunks with their current vec_row from the staged DB
        chunks_with_rows = staged_conn.execute(
            "SELECT chunk_id, vec_row FROM chunks WHERE vec_row IS NOT NULL ORDER BY vec_row"
        ).fetchall()

        if not chunks_with_rows:
            # No chunks left; delete the vector file
            try:
                staged_vec_path.unlink()
            except OSError:
                pass
            staged_conn.commit()
            gen_store.commit_generation(next_gen, vector_row_count=0)
            return

        # Read the current (live) vector file
        current_vec_count = current_vec_path.stat().st_size // (vector_dimension * 4)

        # Build the set of rows we need to keep
        rows_to_keep = {row for _, row in chunks_with_rows}
        rows_to_keep_sorted = sorted(rows_to_keep)

        # Calculate the required new size
        if rows_to_keep_sorted:
            new_max_row = max(rows_to_keep_sorted)
        else:
            new_max_row = -1

        # Check if file needs to be truncated (trailing orphans)
        current_max_row = current_vec_count - 1
        if new_max_row == current_max_row and rows_to_keep_sorted == list(
            range(len(rows_to_keep_sorted))
        ):
            # Vectors are already compact and no truncation needed.
            # Still must write (or delete) the staged vector file before committing.
            if rows_to_keep_sorted and current_vec_path.exists():
                # Non-empty vector set: copy to staged path
                try:
                    shutil.copy2(str(current_vec_path), str(staged_vec_path))
                except (OSError, shutil.Error):
                    pass
            elif not rows_to_keep_sorted:
                # Empty vector set: delete staged vector file
                staged_vec_path.unlink(missing_ok=True)
            staged_conn.commit()
            gen_store.commit_generation(next_gen, vector_row_count=len(rows_to_keep_sorted))
            return

        # If all kept rows are contiguous from 0, just copy the leading portion
        if rows_to_keep_sorted == list(range(len(rows_to_keep_sorted))):
            # Copy only the needed vectors to the staged file
            new_size = (len(rows_to_keep_sorted)) * vector_dimension * 4
            try:
                with open(current_vec_path, "rb") as f:
                    kept_vectors = f.read(new_size)
                with open(staged_vec_path, "wb") as f:
                    f.write(kept_vectors)
                    f.flush()
                    os.fsync(f.fileno())
                staged_conn.commit()
                gen_store.commit_generation(next_gen, vector_row_count=len(rows_to_keep_sorted))
            except OSError:
                pass
            return

        # Read vectors we're keeping from the live file
        vectors_to_keep = []

        try:
            with open(current_vec_path, "rb") as f:
                for old_row in rows_to_keep_sorted:
                    if old_row < current_vec_count:
                        f.seek(old_row * vector_dimension * 4)
                        vec_bytes = f.read(vector_dimension * 4)
                        if len(vec_bytes) == vector_dimension * 4:
                            vec_array = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                            vectors_to_keep.append(vec_array)
        except (OSError, ValueError):
            # Cannot read vectors; leave file as-is and let validation catch it
            return

        if not vectors_to_keep:
            # Couldn't read any vectors; delete the file
            try:
                staged_vec_path.unlink()
            except OSError:
                pass
            staged_conn.commit()
            gen_store.commit_generation(next_gen, vector_row_count=0)
            return

        # Write the compacted vectors to the staged file
        try:
            with open(staged_vec_path, "wb") as f:
                for vec in vectors_to_keep:
                    f.write(vec.astype(np.float32).tobytes())
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            return  # Best-effort; corruption is caught on next search

        # Remap chunk.vec_row in the staged DB: old_row_index -> new_row_index
        old_to_new_row = {old_row: new_idx for new_idx, old_row in enumerate(rows_to_keep_sorted)}

        for chunk_id, old_row in chunks_with_rows:
            new_row = old_to_new_row.get(old_row)
            if new_row is not None:
                query = "UPDATE chunks SET vec_row = ? WHERE chunk_id = ?"
                staged_conn.execute(query, (new_row, chunk_id))

        # Commit changes to staged DB
        staged_conn.commit()

        # Atomically commit the staged generation
        gen_store.commit_generation(next_gen, vector_row_count=len(vectors_to_keep))
    finally:
        staged_conn.close()
