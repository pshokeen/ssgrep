"""Tests for crash-safe work queue."""

from __future__ import annotations

import time

from ssgrep.workqueue import WorkQueue


def test_workqueue_create_and_open(tmp_path):
    """Test that WorkQueue creates schema on first open."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # Verify the database and table exist
    assert (tmp_path / "workqueue.db").exists()

    # Verify schema
    tables = [
        r[0]
        for r in queue.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "work_items" in tables

    queue.close()


def test_workqueue_enqueue_single_item(tmp_path):
    """Test enqueueing a single work item."""
    queue = WorkQueue(tmp_path)
    queue.open()

    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="SessionEnd hook",
    )

    # Verify the item is in the queue
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].transcript_path == "/path/to/transcript.jsonl"
    assert pending[0].session_id == "session-001"
    assert pending[0].reason == "SessionEnd hook"
    assert pending[0].claimed_at is None

    queue.close()


def test_workqueue_deduplication(tmp_path):
    """Test that repeated hints for the same session deduplicate."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue the same item twice with different reasons
    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook attempt 1",
    )
    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook attempt 2",
    )

    # Should have only one item in queue
    pending = queue.pending()
    assert len(pending) == 1

    # The most recent reason should be stored
    assert pending[0].reason == "Hook attempt 2"

    queue.close()


def test_workqueue_different_sessions_not_deduplicated(tmp_path):
    """Test that items for different sessions are not deduplicated."""
    queue = WorkQueue(tmp_path)
    queue.open()

    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook 1",
    )
    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-002",
        reason="Hook 2",
    )

    pending = queue.pending()
    assert len(pending) == 2

    queue.close()


def test_workqueue_claim_and_complete(tmp_path):
    """Test claiming and completing a work item."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue an item
    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook",
    )

    # Claim it
    item = queue.claim()
    assert item is not None
    assert item.transcript_path == "/path/to/transcript.jsonl"
    assert item.claimed_at is not None

    # Verify it's no longer in pending
    pending = queue.pending()
    assert len(pending) == 0

    # Complete it
    queue.complete(item.item_id)

    # Verify it's removed from queue
    pending = queue.pending()
    abandoned = queue.abandoned()
    assert len(pending) == 0
    assert len(abandoned) == 0

    queue.close()


def test_workqueue_multiple_items_fifo_order(tmp_path):
    """Test that items are claimed in FIFO order (oldest first)."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue items with slight delays
    queue.enqueue(
        transcript_path="/path/to/one.jsonl",
        session_id="s1",
        reason="First",
    )
    time.sleep(0.01)
    queue.enqueue(
        transcript_path="/path/to/two.jsonl",
        session_id="s2",
        reason="Second",
    )
    time.sleep(0.01)
    queue.enqueue(
        transcript_path="/path/to/three.jsonl",
        session_id="s3",
        reason="Third",
    )

    # Claim in order
    item1 = queue.claim()
    assert item1.reason == "First"

    item2 = queue.claim()
    assert item2.reason == "Second"

    item3 = queue.claim()
    assert item3.reason == "Third"

    # No more items
    item4 = queue.claim()
    assert item4 is None

    queue.close()


def test_workqueue_crash_recovery_abandoned_items(tmp_path):
    """Test crash recovery: claimed items remain replayable after process death.

    Scenario:
    1. Enqueue a work item
    2. Claim it (process now has it)
    3. "Crash" by raising an exception
    4. Close the queue without completing the item
    5. Reopen in a new process
    6. Verify the item is still there and can be re-claimed

    This tests the claim-then-delete semantics: the item was never deleted,
    so it survives the crash.
    """
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue an item
    queue.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook",
    )

    # Claim it (simulating start of indexing)
    item = queue.claim()
    assert item is not None
    item_id = item.item_id

    # Verify item is no longer pending
    assert len(queue.pending()) == 0

    # Verify item is abandoned (claimed but not completed)
    abandoned = queue.abandoned()
    assert len(abandoned) == 1
    assert abandoned[0].item_id == item_id

    # Crash simulation: close without completing
    queue.close()

    # Recovery: reopen in new process
    queue2 = WorkQueue(tmp_path)
    queue2.open()

    # Verify the item is still abandoned
    abandoned = queue2.abandoned()
    assert len(abandoned) == 1
    assert abandoned[0].transcript_path == "/path/to/transcript.jsonl"
    assert abandoned[0].session_id == "session-001"

    # Verify no pending items (only abandoned ones)
    pending = queue2.pending()
    assert len(pending) == 0

    # Complete the abandoned item
    queue2.complete(abandoned[0].item_id)

    # Verify it's gone
    assert len(queue2.abandoned()) == 0

    queue2.close()


def test_workqueue_enqueue_never_raises(tmp_path):
    """Test that enqueue() never raises even if database is corrupted.

    A hook that fails must not break the user's session exit, so enqueue()
    catches all exceptions and proceeds silently.
    """
    queue = WorkQueue(tmp_path)
    queue.open()

    # Simulate database corruption by closing and deleting the file
    queue.close()
    (tmp_path / "workqueue.db").unlink()

    # Enqueue should not raise even though DB is gone
    try:
        queue.enqueue(
            transcript_path="/path/to/transcript.jsonl",
            session_id="session-001",
            reason="Hook",
        )
    except Exception as e:
        raise AssertionError(f"enqueue() raised {e}: should never raise") from e


def test_workqueue_concurrent_enqueue_same_item(tmp_path):
    """Test concurrent enqueue of the same item deduplicates.

    Two processes enqueueing the same (path, session_id) simultaneously
    should result in one item in the queue, not a race condition.
    """
    queue1 = WorkQueue(tmp_path)
    queue1.open()

    queue2 = WorkQueue(tmp_path)
    queue2.open()

    # Both queues enqueue the same item
    queue1.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook from process 1",
    )
    queue2.enqueue(
        transcript_path="/path/to/transcript.jsonl",
        session_id="session-001",
        reason="Hook from process 2",
    )

    # Only one item should be in the queue
    pending = queue1.pending()
    assert len(pending) == 1

    queue1.close()
    queue2.close()


def test_workqueue_concurrent_claim(tmp_path):
    """Test concurrent claim operations don't return the same item twice.

    Two processes concurrently claiming should each get different items,
    or one gets None. No item should be returned to both.
    """
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue two items
    queue.enqueue("/path/one.jsonl", "s1", "First")
    queue.enqueue("/path/two.jsonl", "s2", "Second")
    queue.close()

    # Two separate processes open and claim
    q1 = WorkQueue(tmp_path)
    q1.open()
    q2 = WorkQueue(tmp_path)
    q2.open()

    item1 = q1.claim()
    item2 = q2.claim()

    # Both should succeed (different items)
    assert item1 is not None
    assert item2 is not None
    assert item1.item_id != item2.item_id

    # Third claim should get None
    item3 = q1.claim()
    assert item3 is None

    q1.close()
    q2.close()


def test_workqueue_abandoned_reset_on_new_enqueue(tmp_path):
    """Test that a new enqueue of an abandoned item resets claimed_at.

    If a hook re-enqueues an item that was previously claimed but never
    completed, the new enqueue should make it pending again.
    """
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue and claim
    queue.enqueue("/path/transcript.jsonl", "s1", "First attempt")
    item = queue.claim()
    assert item.claimed_at is not None

    # Re-enqueue the same item (hook fires again)
    queue.enqueue("/path/transcript.jsonl", "s1", "Second attempt")

    # Check if it's back in pending
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].claimed_at is None

    # No abandoned items
    abandoned = queue.abandoned()
    assert len(abandoned) == 0

    queue.close()


def test_mutation_test_claimed_replay(tmp_path):
    """Mutation test: prove that the replay mechanism is necessary.

    This test MUST fail if the replay mechanism (storing claimed_at)
    is removed. It demonstrates that without it, a crashed process
    would lose work items.

    Scenario:
    1. Enqueue an item
    2. Claim it (mark as being processed)
    3. Close without completing (simulating crash)
    4. Reopen and verify abandoned() finds it
    5. If abandoned() didn't work, the item would be lost

    To mutate: remove the claimed_at column or the abandoned() query.
    The test should then fail at step 4.
    """
    queue = WorkQueue(tmp_path)
    queue.open()

    # Step 1: Enqueue
    queue.enqueue("/path/transcript.jsonl", "session-1", "Hook")

    # Step 2: Claim
    item = queue.claim()
    assert item is not None
    claimed_item_id = item.item_id
    claimed_time = item.claimed_at
    assert claimed_time is not None  # Must be set

    # Step 3: Simulate crash (close without completing)
    queue.close()

    # Step 4: Recovery (reopen)
    queue2 = WorkQueue(tmp_path)
    queue2.open()

    # Step 5: Verify abandoned() finds it
    abandoned = queue2.abandoned()
    assert len(abandoned) == 1, "Replay mechanism failed: item not recovered"
    assert abandoned[0].item_id == claimed_item_id
    assert abandoned[0].claimed_at == claimed_time

    # Verify it's NOT in pending (mutation test: if we deleted on claim, it'd be here)
    pending = queue2.pending()
    assert len(pending) == 0, "Mutation test: claimed item should not be in pending"

    queue2.close()


def test_mutation_test_delete_semantics(tmp_path):
    """Mutation test: prove that complete() deletes ONLY the completed item.

    An earlier version of this test enqueued, claimed, and completed a
    single item, then verified it was gone after a simulated crash
    (close/reopen). With only one row ever in the table, that scenario
    can't tell an unscoped `DELETE FROM work_items` apart from a correctly
    scoped `DELETE FROM work_items WHERE item_id = ?` -- both empty the
    table either way. This version carries a second, untouched item
    throughout so the blast radius of complete() is actually observable:
    if complete() (or claim()) ever deletes unscoped, the second item
    would vanish too, not just the completed one.

    To mutate: change complete() to delete on claim, add a delete in
    claim(), or drop the `WHERE item_id = ?` scoping from complete()'s
    DELETE. The test should then fail at the final assertion.
    """
    queue = WorkQueue(tmp_path)
    queue.open()

    # Enqueue two distinct items.
    queue.enqueue("/path/one.jsonl", "s1", "Hook 1")
    queue.enqueue("/path/two.jsonl", "s2", "Hook 2")
    assert {item.session_id for item in queue.pending()} == {"s1", "s2"}

    # Claim and complete only one of them (whichever claim() returns --
    # don't assume FIFO ordering); the other is left merely enqueued.
    claimed = queue.claim()
    assert claimed is not None
    other_session_id = "s2" if claimed.session_id == "s1" else "s1"

    queue.complete(claimed.item_id)

    # Simulate crash: close without touching the other item further.
    queue.close()

    # Reopen and verify: the completed item is gone everywhere...
    queue2 = WorkQueue(tmp_path)
    queue2.open()

    abandoned = queue2.abandoned()
    assert len(abandoned) == 0, "Completed item should not be abandoned"

    # ...and the untouched item survived complete()'s DELETE. This is the
    # actual scoping property under test; the two assertions above alone
    # would pass even with an unscoped delete.
    remaining = queue2.pending()
    assert len(remaining) == 1, "complete() must not delete the other item"
    assert remaining[0].session_id == other_session_id

    queue2.close()


def test_workqueue_empty_claim_returns_none(tmp_path):
    """Test that claiming from an empty queue returns None."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # No enqueues, so queue is empty
    item = queue.claim()
    assert item is None

    queue.close()


def test_workqueue_wal_mode_enabled(tmp_path):
    """Test that WAL mode is enabled for durability."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # Check PRAGMA journal_mode
    mode = queue.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal", "WAL mode must be enabled for durability"

    queue.close()


def test_workqueue_multiple_paths_same_session(tmp_path):
    """Test that different paths for the same session are separate items."""
    queue = WorkQueue(tmp_path)
    queue.open()

    # Same session ID, different paths
    queue.enqueue("/path/one.jsonl", "s1", "Hook 1")
    queue.enqueue("/path/two.jsonl", "s1", "Hook 2")

    pending = queue.pending()
    assert len(pending) == 2

    queue.close()


def test_workqueue_item_timestamps(tmp_path):
    """Test that enqueue and claim timestamps are recorded correctly."""
    queue = WorkQueue(tmp_path)
    queue.open()

    before = time.time()
    queue.enqueue("/path/transcript.jsonl", "s1", "Hook")
    after = time.time()

    item = queue.claim()
    assert before <= item.enqueue_time <= after
    assert item.enqueue_time <= item.claimed_at


def test_workqueue_clear(tmp_path):
    """Test the clear() method removes all items (used in tests)."""
    queue = WorkQueue(tmp_path)
    queue.open()

    queue.enqueue("/path/one.jsonl", "s1", "Hook 1")
    queue.enqueue("/path/two.jsonl", "s2", "Hook 2")

    assert len(queue.pending()) == 2

    queue.clear()

    assert len(queue.pending()) == 0
    assert len(queue.abandoned()) == 0

    queue.close()
