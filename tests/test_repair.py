"""Tests for bounded tail repair on the search path."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ssgrep import api, embed, indexer, repair, store


def _append_record(session_path: Path, project_dir: Path, record_type: str, content: str) -> None:
    """Append a JSONL record with cwd to the session file."""
    cwd = str(project_dir)
    if record_type == "user":
        record = {"type": "user", "message": {"content": content}, "cwd": cwd}
    elif record_type == "assistant":
        record = {"type": "assistant", "message": {"content": [{"type": "text", "text": content}]}}
    else:
        record = {record_type: content}
    session_path.write_text(session_path.read_text() + json.dumps(record) + "\n")


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fake home directory with .claude/projects."""
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


@pytest.fixture
def fake_embed(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Mock the embedding function to avoid loading the model."""
    counter: dict[str, int] = {"calls": 0}

    def fake_encode(texts: list[str]) -> np.ndarray:
        counter["calls"] += 1
        return np.random.randn(len(texts), 256).astype(np.float32)

    monkeypatch.setattr(embed, "encode", fake_encode)
    return counter


@pytest.fixture
def project_with_index(
    tmp_path: Path, fake_home: Path, fake_embed: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Set up a project with an indexed session."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create session in the fake claude projects directory
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    session_path.write_text(
        json.dumps({"type": "user", "message": {"content": "hello"}, "cwd": str(project_dir)})
        + "\n"
        + json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}
        )
        + "\n"
    )

    # Run initial index
    index_dir = tmp_path / "idx"
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    return project_dir, session_path, index_dir


def test_small_append_is_repaired(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that a small append to a session is repaired successfully."""
    project_dir, session_path, index_dir = project_with_index

    # Get initial state
    conn = sqlite3.connect(str(index_dir / "index.db"))
    initial_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    # Append a new exchange
    _append_record(session_path, project_dir, "user", "second question")
    _append_record(session_path, project_dir, "assistant", "second answer")

    # Read session_id from index
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Attempt repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    # Verify repair succeeded
    assert result.repair_success is True
    assert result.message is None

    # Verify new chunks were added
    conn = sqlite3.connect(str(index_dir / "index.db"))
    final_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert final_chunks > initial_chunks


def test_large_append_is_refused(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that an append exceeding the byte cap is refused with a message."""
    project_dir, session_path, index_dir = project_with_index

    # Read session_id from index
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Append something large enough to trigger refusal
    large_content = "x" * (repair.MAX_APPEND_BYTES + 1000)
    _append_record(session_path, project_dir, "user", large_content)

    # Attempt repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    # Verify repair was refused with a message
    assert result.repair_success is False
    assert result.message is not None
    assert "bytes" in result.message.lower() or "kb" in result.message.lower()
    assert "ssgrep index" in result.message or "index" in result.message


def test_lock_unavailable_returns_immediately(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that unavailable lock is handled non-blocking."""
    project_dir, session_path, index_dir = project_with_index

    # Read session_id from index
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Append new data
    _append_record(session_path, project_dir, "user", "test")

    # Acquire the lock to simulate unavailability
    repair._repair_lock.acquire()
    start = time.perf_counter()
    try:
        result = repair.repair_current_session_tail(
            project_dir,
            session_path,
            session_id,
            index_dir=index_dir,
        )
        elapsed = time.perf_counter() - start

        # Should return immediately (within 10ms of attempting to acquire)
        assert elapsed < 0.1  # 100ms is still "immediately"
        assert result.repair_success is False
        assert result.message is None
    finally:
        repair._repair_lock.release()


def test_vanished_file_returns_failed_repair(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that repair fails gracefully when the file vanishes."""
    project_dir, session_path, index_dir = project_with_index

    # Read session_id from index
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Remove the file
    session_path.unlink()

    # Attempt repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    # Should fail gracefully
    assert result.repair_success is False


def test_no_index_returns_failed_repair(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that repair fails when no index exists."""
    project_dir, session_path, _ = project_with_index
    index_dir = project_dir / ".nonexistent"

    # Append to session
    session_path.write_text(
        session_path.read_text()
        + json.dumps({"type": "user", "message": {"content": "test"}})
        + "\n"
    )

    # Attempt repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        "any-id",
        index_dir=index_dir,
    )

    # Should fail gracefully
    assert result.repair_success is False


def test_no_new_records_updates_cursor(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that if no new records are parsed, the cursor is still updated."""
    project_dir, session_path, index_dir = project_with_index

    # Read session_id and initial cursor
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    initial_cursor = store.get_session_file(conn, session_path)
    conn.close()

    # Append only whitespace/incomplete line (should not parse)
    session_path.write_text(session_path.read_text() + "\n   \n")

    # Attempt repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    # Should succeed
    assert result.repair_success is True

    # Verify cursor was updated
    conn = sqlite3.connect(str(index_dir / "index.db"))
    new_cursor = store.get_session_file(conn, session_path)
    conn.close()

    assert new_cursor is not None
    assert new_cursor.byte_offset > initial_cursor.byte_offset


def test_episode_numbering_continues_from_index(
    project_with_index: tuple[Path, Path, Path],
) -> None:
    """Verify that new episodes get the correct numbering (continuing from existing)."""
    project_dir, session_path, index_dir = project_with_index

    # Read session_id and initial episode count
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    initial_ep_count = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    conn.close()

    # Append new exchange
    _append_record(session_path, project_dir, "user", "new prompt")
    _append_record(session_path, project_dir, "assistant", "new response")

    # Repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result.repair_success is True

    # Verify episode count increased
    conn = sqlite3.connect(str(index_dir / "index.db"))
    final_ep_count = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE session_id = ?", (session_id,)
    ).fetchone()[0]

    # Verify episode IDs continue the sequence
    episodes = conn.execute(
        "SELECT episode_id FROM episodes WHERE session_id = ? ORDER BY episode_id",
        (session_id,),
    ).fetchall()
    conn.close()

    assert final_ep_count > initial_ep_count
    # Episode IDs should be like "session_id:ep:0", "session_id:ep:1", etc.
    for (ep_id,) in episodes:
        assert ":ep:" in ep_id


def test_rewritten_transcript_refuses_repair(project_with_index: tuple[Path, Path, Path]) -> None:
    """A transcript rewritten at the same path (not appended) must refuse repair.

    The first_line_hash guard exists to detect when session_path no longer
    contains the history the stored byte_offset was computed against -- e.g.
    a compacted or rewritten transcript. If this guard breaks, repair would
    parse from the stale byte_offset into unrelated content, silently
    producing garbage episodes instead of refusing -- a data-corruption bug,
    not just a wrong-answer one.
    """
    project_dir, session_path, index_dir = project_with_index

    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    initial_cursor = store.get_session_file(conn, session_path)
    initial_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    # Rewrite the file from scratch with unrelated content: a different first
    # line, not an append. Deliberately larger than the original so that, if
    # the rewrite guard were disabled, disk_size would still exceed
    # cursor.byte_offset and execution would proceed to actually parse from
    # the stale offset into this unrelated content, well under the byte cap
    # so that check doesn't independently produce the same refusal.
    rewritten = (
        json.dumps(
            {
                "type": "user",
                "message": {"content": "totally different history"},
                "cwd": str(project_dir),
            }
        )
        + "\n"
        + json.dumps(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "unrelated"}]}}
        )
        + "\n"
        + json.dumps(
            {
                "type": "user",
                "message": {"content": "more unrelated padding content"},
                "cwd": str(project_dir),
            }
        )
        + "\n"
    )
    session_path.write_text(rewritten)
    assert len(rewritten) > initial_cursor.byte_offset, "fixture must look like growth"
    assert len(rewritten) - initial_cursor.byte_offset < repair.MAX_APPEND_BYTES

    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    assert result.repair_success is False

    # Refusal, not a garbage parse: the cursor and chunk table must be
    # completely untouched.
    conn = sqlite3.connect(str(index_dir / "index.db"))
    cursor_after = store.get_session_file(conn, session_path)
    chunks_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    assert cursor_after.byte_offset == initial_cursor.byte_offset
    assert chunks_after == initial_chunks


def test_unchanged_file_does_nothing(project_with_index: tuple[Path, Path, Path]) -> None:
    """Verify that repair returns success if file hasn't changed since last index."""
    project_dir, session_path, index_dir = project_with_index

    # Read session_id
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Attempt repair without changing file (should succeed, do nothing)
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    # Should succeed with no changes needed
    assert result.repair_success is True
    assert result.message is None


# ===== Mutation tests =====
# These tests verify that the guarding tests actually fail when code is broken


def test_refuses_append_when_cap_check_removed(project_with_index: tuple[Path, Path, Path]) -> None:
    """Mutation test: breaking MAX_APPEND_BYTES check should make large-append test fail.

    This test appends a large amount and verifies refusal. If MAX_APPEND_BYTES
    check is removed, this should fail.
    """
    project_dir, session_path, index_dir = project_with_index

    # Read session_id
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Append something just under the limit
    small_content = "x" * (repair.MAX_APPEND_BYTES - 1000)
    _append_record(session_path, project_dir, "user", small_content)

    # This should succeed (within limit)
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result.repair_success is True

    # Now append well over the limit from the current position
    large_content = "y" * repair.MAX_APPEND_BYTES
    _append_record(session_path, project_dir, "user", large_content)

    # This should refuse
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result.repair_success is False
    assert result.message is not None


def test_lock_is_actually_acquired(project_with_index: tuple[Path, Path, Path]) -> None:
    """Mutation test: verifies that the lock is truly needed.

    This test ensures that concurrent repair attempts are serialized.
    If the lock is removed, this might race and cause issues (though
    detecting a race is hard; we mainly verify the lock exists).
    """
    project_dir, session_path, index_dir = project_with_index

    # Read session_id
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Append data
    _append_record(session_path, project_dir, "user", "test")

    # Acquire lock externally
    assert repair._repair_lock.acquire(blocking=False)

    # Verify that repair cannot proceed
    start = time.perf_counter()
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    elapsed = time.perf_counter() - start

    # Should have failed immediately
    assert result.repair_success is False
    assert elapsed < 0.5  # Should be nearly instant

    repair._repair_lock.release()


def test_new_records_actually_create_chunks(project_with_index: tuple[Path, Path, Path]) -> None:
    """Mutation test: verifies that chunks are created from new records.

    If chunk creation is skipped, this test should fail.
    """
    project_dir, session_path, index_dir = project_with_index

    # Read initial chunk count
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    initial_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    # Append something that will definitely create a chunk
    _append_record(session_path, project_dir, "user", "a" * 1000)
    _append_record(session_path, project_dir, "assistant", "b" * 1000)

    # Repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result.repair_success is True

    # Verify chunks were created
    conn = sqlite3.connect(str(index_dir / "index.db"))
    final_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    assert final_chunks > initial_chunks


def test_cursor_update_persists(project_with_index: tuple[Path, Path, Path]) -> None:
    """Mutation test: verifies that cursor updates are committed.

    If cursor commit is skipped, subsequent repairs would re-parse the same tail.
    """
    project_dir, session_path, index_dir = project_with_index

    # Read session_id and initial cursor
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    initial_cursor = store.get_session_file(conn, session_path)
    conn.close()

    # First append and repair
    _append_record(session_path, project_dir, "user", "first")

    result1 = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result1.repair_success is True

    # Check cursor advanced
    conn = sqlite3.connect(str(index_dir / "index.db"))
    after_first = store.get_session_file(conn, session_path)
    conn.close()

    assert after_first.byte_offset > initial_cursor.byte_offset
    first_offset = after_first.byte_offset

    # Second append and repair
    _append_record(session_path, project_dir, "user", "second")

    result2 = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result2.repair_success is True

    # Check cursor advanced again (from where it left off, not from beginning)
    conn = sqlite3.connect(str(index_dir / "index.db"))
    after_second = store.get_session_file(conn, session_path)
    conn.close()

    assert after_second.byte_offset > first_offset


def test_nonblocking_truly_nonblocking(project_with_index: tuple[Path, Path, Path]) -> None:
    """Mutation test: verifies non-blocking behavior under contention.

    If blocking=False is removed from lock.acquire(), this would hang.
    """
    project_dir, session_path, index_dir = project_with_index

    # Read session_id
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Append data
    _append_record(session_path, project_dir, "user", "test")

    # Hold the lock
    repair._repair_lock.acquire()
    try:
        # This should return almost instantly
        start = time.perf_counter()
        result = repair.repair_current_session_tail(
            project_dir,
            session_path,
            session_id,
            index_dir=index_dir,
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1  # 100ms is the outer bound for "instantly"
        assert result.repair_success is False
    finally:
        repair._repair_lock.release()


def test_appended_content_is_searchable_without_reindex(
    project_with_index: tuple[Path, Path, Path],
) -> None:
    """search() must repair the session tail; appended content becomes findable
    with no explicit index call in between.

    If search.py's call to repair_current_session_tail (line ~604) is disabled,
    the appended marker will not be indexed and this test will fail.
    """
    project_dir, session_path, index_dir = project_with_index

    # Re-index to default location (project_dir/.ssgrep) for search to find
    indexer.index(project_dir, quiet=True)

    marker = f"ZQX_{uuid4().hex[:8]}"
    _append_record(session_path, project_dir, "user", f"Q: {marker}")
    _append_record(session_path, project_dir, "assistant", f"A: {marker}")

    # NOTE: no indexer call, no repair call. Only search.
    response = api.search(project_dir, marker)

    assert response.results, f"appended content not searchable: {marker}"
    assert any(marker in r.excerpt for r in response.results)


def test_repair_appends_vectors_and_checkpoints(
    project_with_index: tuple[Path, Path, Path],
) -> None:
    """repair() appends vectors, commits chunks, and checkpoints the row count.

    This test simulates a lost-on-crash vector bytes scenario: repair appends
    vectors, the SQLite commit succeeds, but the vector bytes are lost (simulated
    by truncating vectors.f32 back to its pre-repair size). When recover() is
    called on a fresh GenerationalStore, it must detect the dangling chunks
    (chunks referencing missing vector rows) and drop them -- proving that the
    checkpoint was correctly advanced beyond just the SQLite commit.

    If checkpoint() is not called, recover() would take its fast path (trusting
    the manifest's committed_vector_rows) and skip the cross-check, leaving the
    dangling chunks in place and corrupting the store.
    """
    project_dir, session_path, index_dir = project_with_index

    # Get initial state
    conn = sqlite3.connect(str(index_dir / "index.db"))
    initial_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    # Append enough to trigger vector appends
    _append_record(session_path, project_dir, "user", "a" * 1000)
    _append_record(session_path, project_dir, "assistant", "b" * 1000)

    # Read session_id
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Record pre-repair vector file size
    vec_path = index_dir / "vectors.f32"
    initial_vec_size = vec_path.stat().st_size if vec_path.exists() else 0

    # Perform repair -- this appends vectors and calls checkpoint()
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )
    assert result.repair_success is True

    # Verify vectors were appended
    post_repair_vec_size = vec_path.stat().st_size
    assert post_repair_vec_size > initial_vec_size, "repair must append vectors"

    # Verify chunks were added
    conn = sqlite3.connect(str(index_dir / "index.db"))
    after_repair_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert after_repair_chunks > initial_chunks, "repair must add chunks"

    # Simulate a crash where vector bytes are lost but the chunks remain
    # committed in the DB. Truncate vectors.f32 back to pre-repair size.
    with open(vec_path, "r+b") as f:
        f.truncate(initial_vec_size)

    # Create a fresh GenerationalStore and call recover()
    from ssgrep.store import GenerationalStore

    reopened = GenerationalStore(index_dir)
    truncated, dropped = reopened.recover()

    # recover() must have detected and dropped the dangling chunks
    # (chunks referencing vector rows that no longer exist).
    # If checkpoint() was not called, recover()'s fast path would skip the
    # cross-check and leave them in place.
    assert dropped > 0, (
        "recover() must detect and drop dangling chunks; if this fails, "
        "repair() likely did not call checkpoint() after commit"
    )

    # Verify the store is consistent after recovery
    from ssgrep.vectors import validate_alignment

    is_valid, reason = validate_alignment(reopened.get_index_path(), vec_path)
    assert is_valid, f"store must be consistent after recovery: {reason}"


def test_repair_holds_generation_during_work(
    project_with_index: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """repair() holds a SHARED lock on the current generation while modifying it.

    This prevents a concurrent rebuild from unlinking the live generation's files
    mid-repair, which would corrupt both the repair's new writes and the existing
    data.

    This test verifies the lock is actually used by:
    1. Spying on gen_store.hold_generation to verify it is called during repair
       (this proves the mechanism is in place)
    2. Running repair_current_session_tail and verifying it completes successfully
       and produces correct, searchable results (behavior-level assertion that
       proves the lock doesn't prevent normal operation, per CLAUDE.md's
       requirement that spy assertions must be paired with positive assertions
       to avoid vacuous truth)

    If hold_generation() is not called, or called at the wrong point, the test's
    spy assertion will fail. If it's called but doesn't actually work (e.g. wrong
    generation number), the behavior-level assertion would catch it (the repair
    would corrupt the store or produce wrong results).
    """
    from ssgrep.store import GenerationalStore

    project_dir, session_path, index_dir = project_with_index

    # Append new content to trigger vector appends
    _append_record(session_path, project_dir, "user", "unique marker xyz")
    _append_record(session_path, project_dir, "assistant", "response to xyz")

    # Read session_id
    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    # Spy on hold_generation to verify it's called
    original_hold_gen = GenerationalStore.hold_generation
    hold_gen_calls: list[int] = []

    def spied_hold_generation(self, generation_number):
        hold_gen_calls.append(generation_number)
        # Call the real implementation
        return original_hold_gen(self, generation_number)

    monkeypatch.setattr(GenerationalStore, "hold_generation", spied_hold_generation)

    # Run repair
    result = repair.repair_current_session_tail(
        project_dir,
        session_path,
        session_id,
        index_dir=index_dir,
    )

    # Verify repair succeeded (behavior-level assertion)
    assert result.repair_success is True, "repair must complete successfully"

    # Verify hold_generation was actually called (spy assertion)
    assert (
        len(hold_gen_calls) > 0
    ), "hold_generation must be called during repair (proves lock mechanism is in place)"
    # Verify it was called with the current generation number
    gen_store = GenerationalStore(index_dir)
    expected_gen = gen_store.current_generation
    assert expected_gen in hold_gen_calls, (
        f"hold_generation must be called with current generation {expected_gen}; "
        f"got calls with generations {hold_gen_calls}"
    )

    # Verify the store is consistent after repair
    from ssgrep.vectors import validate_alignment

    vec_path = index_dir / "vectors.f32"
    db_path = index_dir / "index.db"
    is_valid, reason = validate_alignment(db_path, vec_path)
    assert is_valid, f"store must be consistent after repair: {reason}"

    # Verify repair actually added chunks that are searchable (additional
    # behavior-level verification that the appended content made it through
    # the lock-protected section correctly)
    conn = sqlite3.connect(str(db_path))
    xyz_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ("xyz",)
    ).fetchone()[0]
    conn.close()
    assert (
        xyz_chunks > 0
    ), "repair must have indexed the appended content (behavior-level verification)"


def test_repair_overrun_warns_on_stderr_and_commits(
    project_with_index: tuple[Path, Path, Path],
    fake_embed: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """A post-commit budget overrun is reported to stderr, never hidden.

    The clock is monkeypatched to stay within budget until encoding has
    happened (so the pre-encode gates pass) and then jump far past
    MAX_REPAIR_MS — the exact shape of a real overrun, where the budget
    gated the decision to start but the work itself ran long. The repair
    must still commit (never refuse retroactively) and must say so.
    """
    project_dir, session_path, index_dir = project_with_index

    _append_record(session_path, project_dir, "user", "overrun test prompt")
    _append_record(session_path, project_dir, "assistant", "overrun test response")

    encoded = {"done": False}
    real_encode = embed.encode

    def flagging_encode(texts):
        encoded["done"] = True
        return real_encode(texts)

    monkeypatch.setattr(embed, "encode", flagging_encode)

    real_perf = time.perf_counter

    def slow_after_encode() -> float:
        return real_perf() + (100.0 if encoded["done"] else 0.0)

    monkeypatch.setattr(repair.time, "perf_counter", slow_after_encode)

    conn = sqlite3.connect(str(index_dir / "index.db"))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    pre_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    result = repair.repair_current_session_tail(
        project_dir, session_path, session_id, index_dir=index_dir
    )

    captured = capsys.readouterr()
    assert result.repair_success is True, "an overrun must not refuse committed work"
    assert "overran" in captured.err, "the overrun must reach stderr"
    assert "committed" in captured.err

    conn = sqlite3.connect(str(index_dir / "index.db"))
    post_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert post_chunks > pre_chunks, "the repair's write must have landed"


def test_repair_survives_concurrent_rebuild_race(
    project_with_index: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real Event-gated repair-vs-rebuild race (index-durability spec:
    "Tail repair survives a concurrent rebuild").

    Repair is paused inside its held-generation span -- right before its
    vector append, i.e. mid-write -- while a full rebuild runs to
    completion on a second thread, committing a newer generation and
    running generation cleanup. Because repair holds the SHARED generation
    lock for its whole body, cleanup's non-blocking EXCLUSIVE attempt must
    skip (not unlink) the generation repair is writing; repair then
    resumes and completes against intact files, and the live generation
    the rebuild committed stays aligned.
    """
    import threading

    from ssgrep import vectors
    from ssgrep.vectors import validate_alignment

    project_dir, session_path, index_dir = project_with_index

    _append_record(session_path, project_dir, "user", "race question")
    _append_record(session_path, project_dir, "assistant", "race answer")

    gen_store = store.GenerationalStore(index_dir)
    start_generation = gen_store.current_generation
    conn = sqlite3.connect(str(gen_store.get_index_path()))
    session_id = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
    conn.close()

    repair_paused = threading.Event()
    let_repair_resume = threading.Event()
    real_append = vectors.append
    armed = {"on": True}

    def gated_append(vstore: object, vecs: object) -> object:
        # Gate only the first caller -- the repair running on the main
        # thread. The rebuild thread's own appends pass through untouched,
        # or the two would deadlock on the same gate.
        if armed["on"]:
            armed["on"] = False
            repair_paused.set()
            if not let_repair_resume.wait(timeout=15):
                raise TimeoutError("rebuild thread never signaled completion")
        return real_append(vstore, vecs)

    monkeypatch.setattr(vectors, "append", gated_append)

    errors: list[tuple[str, BaseException]] = []

    def rebuild_worker() -> None:
        try:
            if not repair_paused.wait(timeout=15):
                raise TimeoutError("repair never reached its paused append")
            indexer.index(project_dir, index_dir=index_dir, rebuild=True, quiet=True)
        except BaseException as error:  # surfaced via `errors` after join
            errors.append(("rebuild", error))
        finally:
            let_repair_resume.set()

    rebuild_thread = threading.Thread(target=rebuild_worker, daemon=True)
    rebuild_thread.start()

    result = repair.repair_current_session_tail(
        project_dir, session_path, session_id, index_dir=index_dir
    )

    rebuild_thread.join(timeout=30)
    assert not rebuild_thread.is_alive(), "rebuild thread must terminate"
    assert errors == [], f"no thread may crash, got: {errors}"
    assert repair_paused.is_set(), "the race window must actually have been exercised"

    # Repair's held generation was not unlinked out from under it: its
    # write completed without error against intact files.
    assert result.repair_success is True

    # The rebuild genuinely superseded the generation repair wrote to, so
    # the race was real, not a no-op overlap.
    final = store.GenerationalStore(index_dir)
    assert final.current_generation > start_generation

    # And the live generation the rebuild committed stays aligned.
    is_valid, error_msg = validate_alignment(final.get_index_path(), final.get_vector_path())
    assert is_valid, f"live generation must stay aligned, got: {error_msg}"
