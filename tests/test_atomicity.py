"""Two-store crash atomicity: index.db and vectors.f32 cannot be committed as
one file, so a crash between writing one and the other must never leave a
chunk row that references a vector row which does not exist, and must never
leave a permanently unreferenced vector row lying around blocking the
alignment check. See store.GenerationalStore for the mechanism.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ssgrep.store import GenerationalStore, init_db, insert_chunk
from ssgrep.types import Chunk, ContentType
from ssgrep.vectors import append, close, open_vectors, validate_alignment

DIM = 256


def _chunk(chunk_id: str, session_id: str = "s1", text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        episode_id=f"{chunk_id}-ep",
        session_id=session_id,
        text=text,
        content_type=ContentType.PROMPT,
    )


# ---------------------------------------------------------------------------
# The core guarantee: recovery after a crash between the SQLite commit and
# the vector append leaves a consistent store, in both possible orderings.
# ---------------------------------------------------------------------------


def test_recover_drops_chunks_that_reference_missing_vector_rows(tmp_path: Path) -> None:
    """DONE WHEN: crash between the SQLite commit and the vector append must
    leave a consistent store -- no chunk referencing a missing vector row.

    Simulated crash: three chunks are durably committed to index.db,
    referencing vec_rows 0, 1, 2 (matching the real indexer's append-before
    -insert ordering). The process then dies before the third vector's
    bytes are durable, so the vectors.f32 that survives the crash is one
    row short of what the already-committed chunks expect -- the "far
    worse" case named in the task, and the one a naive truncate-only
    recovery would get wrong (there is nothing left of row 2 to un-lose).
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    conn = init_db(gen_store.get_index_path())
    vec_store = open_vectors(gen_store.get_vector_path(), dimension=DIM)

    vecs = np.random.randn(3, DIM).astype(np.float32)
    vec_rows = append(vec_store, vecs)  # rows 0, 1, 2 -- append() has already returned
    for i, row in enumerate(vec_rows):
        insert_chunk(conn, _chunk(f"c{i}", text=f"text{i}"), row)
    conn.commit()  # durable: c0/c1/c2 now reference rows 0/1/2
    conn.close()
    close(vec_store)

    # The crash: row 2's bytes never reached physical disk.
    vec_path = gen_store.get_vector_path()
    with open(vec_path, "r+b") as f:
        f.truncate(2 * GenerationalStore.VECTOR_ROW_BYTES)

    # Process restart: a fresh store instance, then the explicit recovery
    # step a caller takes when opening the store to index.
    reopened = GenerationalStore(index_dir)
    truncated, dropped = reopened.recover()

    assert dropped == 1, "the chunk referencing the lost row must be dropped"
    assert truncated == 0, "rows 0 and 1 are both still needed; nothing to trim"

    is_valid, reason = validate_alignment(reopened.get_index_path(), vec_path)
    assert is_valid, f"store must be internally consistent after recovery: {reason}"

    check_conn = sqlite3.connect(str(reopened.get_index_path()))
    try:
        remaining = {row[0] for row in check_conn.execute("SELECT chunk_id FROM chunks").fetchall()}
        fts_remaining = {
            row[0] for row in check_conn.execute("SELECT chunk_id FROM chunks_fts").fetchall()
        }
        tri_remaining = {
            row[0] for row in check_conn.execute("SELECT chunk_id FROM chunks_fts_tri").fetchall()
        }
    finally:
        check_conn.close()
    assert remaining == {"c0", "c1"}, "c2 referenced a row that no longer exists"
    assert fts_remaining == {"c0", "c1"}, "the FTS shadow row must not dangle either"
    # The trigram mirror (schema v4) is a second FTS shadow of the same
    # chunk text; a dangling row here would let the dropped chunk keep
    # matching through the trigram leg with nothing behind it to excerpt.
    assert tri_remaining == {"c0", "c1"}, "the trigram shadow row must not dangle either"


def test_recover_truncates_vector_rows_no_committed_chunk_references(tmp_path: Path) -> None:
    """The other ordering: a vector is appended but the crash lands before
    the chunk that would reference it is ever inserted and committed. The
    extra row is harmless (nothing points at it) but must still be trimmed,
    or the store is stuck failing the alignment check forever even though
    nothing meaningful was lost.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    conn = init_db(gen_store.get_index_path())
    vec_store = open_vectors(gen_store.get_vector_path(), dimension=DIM)

    vecs = np.random.randn(2, DIM).astype(np.float32)
    vec_rows = append(vec_store, vecs)
    insert_chunk(conn, _chunk("c0", text="only one committed"), vec_rows[0])
    conn.commit()  # crash lands here: row 1 was appended but never referenced

    conn.close()
    close(vec_store)

    vec_path = gen_store.get_vector_path()
    assert vec_path.stat().st_size == 2 * GenerationalStore.VECTOR_ROW_BYTES

    reopened = GenerationalStore(index_dir)
    truncated, dropped = reopened.recover()

    assert truncated == 1
    assert dropped == 0
    assert vec_path.stat().st_size == 1 * GenerationalStore.VECTOR_ROW_BYTES

    is_valid, reason = validate_alignment(reopened.get_index_path(), vec_path)
    assert is_valid, reason


def test_recover_on_fresh_or_empty_store_is_a_harmless_noop(tmp_path: Path) -> None:
    index_dir = tmp_path / ".ssgrep"
    assert GenerationalStore(index_dir).recover() == (0, 0)  # no index.db at all yet

    init_db(index_dir / "index.db").close()
    open_vectors(index_dir / "vectors.f32", dimension=DIM)
    assert GenerationalStore(index_dir).recover() == (0, 0)  # index.db exists, nothing in it


# ---------------------------------------------------------------------------
# checkpoint() is the cheap per-append journal (D12/defect d): it must not
# require copying the store into a new generation, and recover() must trust
# it to skip work only when it actually agrees with the file on disk.
# ---------------------------------------------------------------------------


def test_checkpoint_and_recover_operate_in_place_without_a_new_generation(tmp_path: Path) -> None:
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    conn = init_db(gen_store.get_index_path())
    vec_store = open_vectors(gen_store.get_vector_path(), dimension=DIM)
    rows = append(vec_store, np.random.randn(4, DIM).astype(np.float32))
    for i, row in enumerate(rows):
        insert_chunk(conn, _chunk(f"c{i}"), row)
    conn.commit()
    gen_store.checkpoint(vec_store.row_count)
    conn.close()
    close(vec_store)

    assert gen_store.current_generation == 0, "routine appends never bump the generation"
    assert not (index_dir / "index.db.1").exists()
    assert not (index_dir / "vectors.f32.1").exists()

    reopened = GenerationalStore(index_dir)
    assert reopened.committed_vector_rows == 4
    assert reopened.recover() == (0, 0), "a correctly checkpointed store has nothing to repair"


def test_recover_skips_full_check_when_journal_already_matches_file_size(tmp_path: Path) -> None:
    """recover()'s fast path trusts the journal only to *skip* the expensive
    chunks-table cross-check, and only when the journaled row count agrees
    with vectors.f32's actual size. Proven here by constructing a chunks
    table the full check would flag (a dangling vec_row=999), then showing
    recover() leaves it alone specifically because the journal matches the
    file -- i.e. the skip is really happening, not just returning zero
    because there was nothing to find.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    conn = init_db(gen_store.get_index_path())
    vec_store = open_vectors(gen_store.get_vector_path(), dimension=DIM)
    rows = append(vec_store, np.random.randn(2, DIM).astype(np.float32))
    for i, row in enumerate(rows):
        insert_chunk(conn, _chunk(f"ok{i}"), row)
    # A chunk inserted directly, bypassing the append-then-insert protocol
    # entirely -- vectors.f32's size never reflects it, only the row count
    # journal (deliberately, below) claims everything is fine.
    insert_chunk(conn, _chunk("dangling"), 999)
    conn.commit()
    conn.close()
    close(vec_store)

    gen_store.checkpoint(2)  # journal says "2 rows are known-good", matching the live file

    reopened = GenerationalStore(index_dir)
    assert reopened.recover() == (0, 0), "fast path must trust a journal that matches the file"

    # The full check (triggered by any mismatch) still catches it, proving
    # the dangling row was really there and the above was a genuine skip.
    # Truncating to 1 live row makes both "ok1" (vec_row=1) and "dangling"
    # (vec_row=999) fall at-or-past it.
    with open(gen_store.get_vector_path(), "r+b") as f:
        f.truncate(1 * GenerationalStore.VECTOR_ROW_BYTES)  # now live_rows != journal
    truncated, dropped = GenerationalStore(index_dir).recover()
    assert dropped == 2


# ---------------------------------------------------------------------------
# Manifest durability: a real timestamp, and an atomic write that can never
# leave a torn manifest or destroy the previous one on failure.
# ---------------------------------------------------------------------------


def test_manifest_timestamp_is_a_real_timestamp_not_a_dot(tmp_path: Path) -> None:
    """Regression guard: the original manifest wrote `"timestamp": str(Path())`,
    which always serializes to the literal string ".".
    """
    index_dir = tmp_path / ".ssgrep"
    GenerationalStore(index_dir).checkpoint(0)

    manifest = json.loads((index_dir / ".manifest").read_text())
    assert manifest["timestamp"] != "."
    parsed = datetime.fromisoformat(manifest["timestamp"])
    assert parsed.tzinfo is not None, "must be a timezone-aware, unambiguous timestamp"


def test_manifest_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    index_dir = tmp_path / ".ssgrep"
    GenerationalStore(index_dir).checkpoint(5)

    assert list(index_dir.glob(".manifest.*.tmp")) == []
    assert (index_dir / ".manifest").exists()


def test_manifest_write_failure_leaves_previous_manifest_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash partway through _write_manifest (simulated: os.fsync raises)
    must never corrupt or partially overwrite the previous, valid manifest
    -- the entire point of write-to-temp-then-rename instead of the
    original plain write_text().
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    gen_store.checkpoint(1)
    original_bytes = (index_dir / ".manifest").read_bytes()

    def failing_fsync(_fd: int) -> None:
        raise OSError("simulated crash mid-manifest-write")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        gen_store.checkpoint(999)
    monkeypatch.undo()

    assert (index_dir / ".manifest").read_bytes() == original_bytes
    assert list(index_dir.glob(".manifest.*.tmp")) == [], "the failed temp file must be cleaned up"


# ---------------------------------------------------------------------------
# commit_generation() must validate before it swaps -- and only cleans up a
# superseded generation once the new one is confirmed valid and durable.
# ---------------------------------------------------------------------------


def test_commit_generation_refuses_broken_staged_generation_and_preserves_live_store(
    tmp_path: Path,
) -> None:
    """Regression guard for the original defect: commit_generation()
    unconditionally deleted the generation it replaced, with no check that
    the new one was actually valid. A broken staged generation must be
    refused, leaving generation 0 -- the live store -- untouched.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    conn0 = init_db(gen_store.get_index_path(0))
    vec0 = open_vectors(gen_store.get_vector_path(0), dimension=DIM)
    rows0 = append(vec0, np.random.randn(2, DIM).astype(np.float32))
    insert_chunk(conn0, _chunk("live0", text="the only good copy"), rows0[0])
    conn0.commit()
    conn0.close()
    close(vec0)

    # Stage a broken generation 1: a chunk referencing a vec_row that does
    # not exist, as if a rebuild crashed mid-way through.
    index_1, vec_1 = gen_store.stage_generation(1)
    conn1 = init_db(index_1)
    vec1 = open_vectors(vec_1, dimension=DIM)
    append(vec1, np.random.randn(1, DIM).astype(np.float32))  # only row 0 is real
    insert_chunk(conn1, _chunk("broken", text="dangling"), 5)  # no such row
    conn1.commit()
    conn1.close()
    close(vec1)

    with pytest.raises(ValueError):
        gen_store.commit_generation(1)

    assert gen_store.current_generation == 0
    assert gen_store.get_index_path(0).exists(), "the live generation must not be deleted"
    assert gen_store.get_vector_path(0).exists(), "the live generation must not be deleted"

    check_conn = sqlite3.connect(str(gen_store.get_index_path(0)))
    try:
        row = check_conn.execute("SELECT text FROM chunks WHERE chunk_id = 'live0'").fetchone()
    finally:
        check_conn.close()
    assert row == ("the only good copy",), "generation 0's data must still be intact"


def test_commit_generation_with_valid_generation_supersedes_and_cleans_up(
    tmp_path: Path,
) -> None:
    """The positive-path companion: a genuinely valid staged generation IS
    swapped in, and the superseded generation 0 files ARE reclaimed --
    commit_generation() must not become overly conservative.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    init_db(gen_store.get_index_path(0)).close()
    open_vectors(gen_store.get_vector_path(0), dimension=DIM)  # empty gen-0 files

    index_1, vec_1 = gen_store.stage_generation(1)
    conn1 = init_db(index_1)
    vec1 = open_vectors(vec_1, dimension=DIM)
    rows1 = append(vec1, np.random.randn(1, DIM).astype(np.float32))
    insert_chunk(conn1, _chunk("c1", text="new generation"), rows1[0])
    conn1.commit()
    conn1.close()
    close(vec1)

    gen_store.commit_generation(1)

    assert gen_store.current_generation == 1
    assert not gen_store.get_index_path(0).exists()
    assert not gen_store.get_vector_path(0).exists()
    assert gen_store.get_index_path(1).exists()
    assert gen_store.get_vector_path(1).exists()


def test_validate_generations_rejects_missing_vectors_file_with_existing_chunks(
    tmp_path: Path,
) -> None:
    """Regression guard: the original validate_generations() returned True
    (valid) whenever vec_path did not exist at all, regardless of whether
    any chunk referenced a vec_row -- silently approving a generation with
    no vector backing whatsoever.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)
    conn = init_db(index_dir / "index.db")
    insert_chunk(conn, _chunk("c0"), 0)
    conn.commit()
    conn.close()

    missing_vec = index_dir / "vectors.f32"
    assert not missing_vec.exists()
    assert gen_store.validate_generations(index_dir / "index.db", missing_vec) is False


def test_validate_generations_rejects_orphaned_vector_rows(tmp_path: Path) -> None:
    """validate_generations() has two independent rejection conditions: a
    chunk's vec_row falling out of bounds (covered above and by
    test_commit_generation_refuses_broken_staged_generation_and_preserves_live_store),
    and a vector row that is in bounds but referenced by no chunk at all.
    This covers the second condition directly.

    Every other test that exercises orphaned vectors goes through
    cleanup_orphaned_vectors(), which by construction always produces a
    compacted, orphan-free staged generation -- so it can never stage the
    one shape that exercises this branch as a rejection. Stage it by hand
    instead: 3 vectors are appended (rows 0, 1, 2) but only rows 0 and 2
    are ever referenced by a chunk, leaving row 1 orphaned-but-in-bounds.
    """
    index_dir = tmp_path / ".ssgrep"
    gen_store = GenerationalStore(index_dir)

    conn = init_db(gen_store.get_index_path())
    vec_store = open_vectors(gen_store.get_vector_path(), dimension=DIM)
    rows = append(vec_store, np.random.randn(3, DIM).astype(np.float32))
    assert list(rows) == [0, 1, 2], "sanity: row 1 must really be in-bounds, not dangling"
    insert_chunk(conn, _chunk("c0"), rows[0])
    insert_chunk(conn, _chunk("c2"), rows[2])  # row 1 is never referenced by any chunk
    conn.commit()
    conn.close()
    close(vec_store)

    assert (
        gen_store.validate_generations(gen_store.get_index_path(), gen_store.get_vector_path())
        is False
    ), "row 1 is orphaned -- in bounds but referenced by no chunk -- and must be rejected"
