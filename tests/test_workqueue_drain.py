"""Tests for workqueue drain functionality (D13 reconciliation).

Tests cover:
1. Drain processes queued items and empties queue.
2. Empty queue is silent and idempotent.
3. Crashed drain leaves items replayable.
4. Corrupt queue does not break index().
5. MCP startup drains without blocking or raising.
6. Subagent-shaped transcripts are correctly classified during drain.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from ssgrep import indexer
from ssgrep.mcp_server import get_mcp_server
from ssgrep.workqueue import WorkQueue


@pytest.fixture
def project_with_transcripts(tmp_path: Path) -> tuple[Path, Path]:
    """Create a project with one transcript file.

    Returns (project_dir, transcript_path).
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create a minimal valid transcript
    transcript_path = project_dir / "transcript.jsonl"
    records = [
        {
            "type": "system",
            "subtype": "session_start",
            "timestamp": "2026-07-25T10:00:00",
            "session_id": "test-001",
        },
        {
            "type": "user",
            "timestamp": "2026-07-25T10:00:05",
            "message": {"content": "What is Python?"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-07-25T10:00:10",
            "message": {"content": "Python is a programming language."},
        },
    ]
    with open(transcript_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return project_dir, transcript_path


class TestDrainProcessesQueuedItems:
    """Test 1: index() with items queued processes them and leaves queue empty."""

    def test_drain_processes_items_and_empties_queue(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """Queued items are processed and removed from queue."""
        project_dir, transcript_path = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Enqueue a work item
        queue = WorkQueue(index_dir)
        queue.open()
        queue.enqueue(
            transcript_path=str(transcript_path),
            session_id="test-001",
            reason="Test",
        )
        queue.close()

        # Verify item is in queue
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        assert len(pending) == 1
        queue.close()

        # Run index, which should drain the queue
        stats = indexer.index(project_dir, quiet=True)
        assert stats.index_exists

        # Verify queue is now empty
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        abandoned = queue.abandoned()
        assert len(pending) == 0, "Pending items should be drained"
        assert len(abandoned) == 0, "No abandoned items should exist"
        queue.close()


class TestEmptyQueueIsSilent:
    """Test 2: index() with empty queue behaves like today, no noise."""

    def test_empty_queue_is_silent(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """Empty queue should not produce output or affect index."""
        project_dir, _ = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Ensure queue exists and is empty
        queue = WorkQueue(index_dir)
        queue.open()
        queue.close()

        # Run index with empty queue
        stats = indexer.index(project_dir, quiet=True)
        assert stats.index_exists

        # Verify queue is still empty
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        assert len(pending) == 0, "Empty queue should stay empty"
        queue.close()

    def test_index_without_queue_is_idempotent(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """Running index without a queue should work fine."""
        project_dir, _ = project_with_transcripts

        # First run: creates everything including queue
        stats1 = indexer.index(project_dir, quiet=True)
        chunks1 = stats1.chunk_count

        # Second run: queue exists but is empty (already drained)
        stats2 = indexer.index(project_dir, quiet=True)
        chunks2 = stats2.chunk_count

        # Results should be identical
        assert chunks1 == chunks2, "Idempotent index should produce same results"


class TestCrashedDrainerLeavesItemsReplayable:
    """Test 3: Drainer crash leaves items replayable."""

    def test_crashed_drain_leaves_item_replayable(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """A crash after claim but before complete leaves item in claimed state."""
        project_dir, transcript_path = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Enqueue an item
        queue = WorkQueue(index_dir)
        queue.open()
        queue.enqueue(
            transcript_path=str(transcript_path),
            session_id="test-001",
            reason="Test",
        )
        queue.close()

        # Simulate a crashed indexer: claim the item but don't complete it
        queue = WorkQueue(index_dir)
        queue.open()
        item = queue.claim()
        assert item is not None
        claimed_item_id = item.item_id
        queue.close()

        # Verify the item is now claimed (abandoned)
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        abandoned = queue.abandoned()
        assert len(pending) == 0, "Claimed item should not appear in pending"
        assert len(abandoned) == 1, "Claimed item should appear in abandoned"
        assert abandoned[0].item_id == claimed_item_id
        queue.close()

        # Now run index, which should process the abandoned item
        stats = indexer.index(project_dir, quiet=True)
        assert stats.index_exists

        # Verify the item is now gone (completed)
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        abandoned = queue.abandoned()
        assert len(pending) == 0, "All items should be cleared"
        assert len(abandoned) == 0, "No abandoned items should remain"
        queue.close()

    def test_partial_crash_scenario(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """Multiple items; one crashes mid-process, others complete."""
        project_dir, transcript_path = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Create a second transcript for a different session
        transcript_path_2 = project_dir / "transcript2.jsonl"
        records = [
            {
                "type": "system",
                "subtype": "session_start",
                "timestamp": "2026-07-25T11:00:00",
                "session_id": "test-002",
            },
            {
                "type": "user",
                "timestamp": "2026-07-25T11:00:05",
                "message": {"content": "Another question?"},
            },
            {
                "type": "assistant",
                "timestamp": "2026-07-25T11:00:10",
                "message": {"content": "Another answer."},
            },
        ]
        with open(transcript_path_2, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Enqueue two items
        queue = WorkQueue(index_dir)
        queue.open()
        queue.enqueue(
            transcript_path=str(transcript_path),
            session_id="test-001",
            reason="First",
        )
        queue.enqueue(
            transcript_path=str(transcript_path_2),
            session_id="test-002",
            reason="Second",
        )
        queue.close()

        # Claim the first one (simulating a crash before completion)
        queue = WorkQueue(index_dir)
        queue.open()
        item1 = queue.claim()
        item1_id = item1.item_id if item1 else None
        queue.close()

        # Claim and complete the second
        queue = WorkQueue(index_dir)
        queue.open()
        item2 = queue.claim()
        if item2:
            queue.complete(item2.item_id)
        queue.close()

        # Verify state: one claimed, one completed
        queue = WorkQueue(index_dir)
        queue.open()
        abandoned = queue.abandoned()
        assert len(abandoned) == 1
        assert abandoned[0].item_id == item1_id
        queue.close()

        # Run index: should complete the abandoned item
        indexer.index(project_dir, quiet=True)

        # Verify both are gone
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        abandoned = queue.abandoned()
        assert len(pending) == 0
        assert len(abandoned) == 0
        queue.close()


class TestCorruptQueueDegradation:
    """Test 4: Corrupt or unreadable queue does not break index()."""

    def test_corrupt_queue_file_is_silent_noop(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """A corrupt queue file should be silently skipped."""
        project_dir, _ = project_with_transcripts
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Create a corrupt workqueue.db
        queue_path = index_dir / "workqueue.db"
        queue_path.write_text("this is not sqlite")

        # index() should not crash
        stats = indexer.index(project_dir, quiet=True)
        assert stats.index_exists, "Index should be built despite corrupt queue"

    def test_unreadable_queue_is_noop(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """An unreadable queue (permission denied) is a no-op."""
        project_dir, _ = project_with_transcripts
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Create a queue and then make it unreadable (if running as non-root)
        queue_path = index_dir / "workqueue.db"
        queue = WorkQueue(index_dir)
        queue.open()
        queue.close()

        try:
            queue_path.chmod(0o000)
            # index() should not crash despite unreadable queue
            stats = indexer.index(project_dir, quiet=True)
            assert stats.index_exists
        finally:
            # Restore permissions for cleanup
            queue_path.chmod(0o644)

    def test_missing_queue_is_normal(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """A missing queue directory is completely normal."""
        project_dir, _ = project_with_transcripts
        # Don't create index_dir; let index() handle it

        # index() should work fine with no queue
        stats = indexer.index(project_dir, quiet=True)
        assert stats.index_exists


class TestMCPStartupDrain:
    """Test 5: MCP startup drains, and failures don't prevent serving."""

    @pytest.fixture(autouse=True)
    def reset_mcp_global(self) -> None:
        """Reset the module-level _mcp global before each test for isolation."""
        import ssgrep.mcp_server as mcp_server

        mcp_server._mcp = None
        yield
        mcp_server._mcp = None

    def test_mcp_startup_drains_queue(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """MCP server startup should drain queued items."""
        project_dir, transcript_path = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Change to project directory so MCP sees it
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)

            # Enqueue an item
            queue = WorkQueue(index_dir)
            queue.open()
            queue.enqueue(
                transcript_path=str(transcript_path),
                session_id="test-001",
                reason="MCP test",
            )
            queue.close()

            # Verify it's queued
            queue = WorkQueue(index_dir)
            queue.open()
            pending = queue.pending()
            assert len(pending) == 1
            queue.close()

            # Get MCP server (which calls _drain_startup_workqueue)
            server = get_mcp_server()
            assert server is not None

            # Verify queue was drained
            queue = WorkQueue(index_dir)
            queue.open()
            pending = queue.pending()
            abandoned = queue.abandoned()
            # Queue should be empty after drain; both pending and abandoned lists
            # should have the item removed (FAILS if drain doesn't complete items)
            assert len(pending) == 0, "Pending items should be drained"
            assert len(abandoned) == 0, "Abandoned items should be drained"
            queue.close()
        finally:
            os.chdir(old_cwd)

    def test_mcp_startup_handles_missing_queue(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """MCP startup should not crash if queue doesn't exist."""
        project_dir, _ = project_with_transcripts

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)

            # No queue exists; MCP should not crash
            server = get_mcp_server()
            assert server is not None
        finally:
            os.chdir(old_cwd)

    def test_mcp_startup_handles_corrupt_queue(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """MCP startup should handle corrupt queue gracefully."""
        project_dir, _ = project_with_transcripts
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        # Create corrupt queue
        (index_dir / "workqueue.db").write_text("corrupt")

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)

            # MCP should not crash
            server = get_mcp_server()
            assert server is not None
        finally:
            os.chdir(old_cwd)


class TestSubagentDrainClassification:
    """Test 6: Subagent-shaped transcripts are correctly classified during drain."""

    def test_drain_classifies_subagent_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_home: tuple[Path, Path]
    ) -> None:
        """Verify that drain correctly identifies subagent transcripts.

        Creates a subagent-shaped path (under ~/.claude/projects/*/subagents/),
        enqueues it, drains, and verifies the indexed row has is_main=0 and
        a non-null parent_session_id.
        """
        # Set up a fake home with .claude/projects
        fake_home = tmp_path / "home"
        (fake_home / ".claude" / "projects" / "test-project").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        # Create a subagent-shaped transcript path
        session_id = str(uuid4())
        subagent_session_id = str(uuid4())
        agent_dir = f"agent-{uuid4()}"
        subagents_dir = (
            fake_home
            / ".claude"
            / "projects"
            / "test-project"
            / session_id
            / "subagents"
            / agent_dir
        )
        subagents_dir.mkdir(parents=True)
        transcript_path = subagents_dir / f"{subagent_session_id}.jsonl"

        # Create a minimal valid transcript
        # cwd is recorded like every real transcript's records: the drain's
        # scope filter (2026-08-07) fails CLOSED on cwd-less files -- exactly
        # as discovery itself rejects them under any scope -- so a fixture
        # without cwd would be dropped as unattributable, which is the scope
        # filter's job, not this test's subject (classification).
        records = [
            {
                "type": "system",
                "subtype": "session_start",
                "timestamp": "2026-07-25T10:00:00",
                "session_id": subagent_session_id,
                "cwd": str(tmp_path / "project"),
            },
            {
                "type": "user",
                "timestamp": "2026-07-25T10:00:05",
                "message": {"content": "Subagent question"},
                "cwd": str(tmp_path / "project"),
            },
            {
                "type": "assistant",
                "timestamp": "2026-07-25T10:00:10",
                "message": {"content": "Subagent answer"},
                "cwd": str(tmp_path / "project"),
            },
        ]
        with open(transcript_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Create project_dir for indexing (can be anywhere, doesn't need to exist)
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        index_dir = project_dir / ".ssgrep"

        # Enqueue the subagent transcript
        queue = WorkQueue(index_dir)
        queue.open()
        queue.enqueue(
            transcript_path=str(transcript_path),
            session_id=subagent_session_id,
            reason="Subagent test",
        )
        queue.close()

        # Run index, which should drain and classify correctly
        stats = indexer.index(project_dir, quiet=True)
        assert stats.index_exists

        # Verify queue is empty
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        assert len(pending) == 0
        queue.close()

        # Verify the indexed row has is_main=0 and parent_session_id set
        conn = sqlite3.connect(str(index_dir / "index.db"))
        row = conn.execute(
            "SELECT is_main, parent_session_id FROM sessions WHERE session_id = ?",
            (subagent_session_id,),
        ).fetchone()
        conn.close()

        assert row is not None, f"Session {subagent_session_id} should be in database"
        is_main, parent_session_id = row
        assert is_main == 0, f"Subagent should have is_main=0, got {is_main}"
        assert (
            parent_session_id is not None
        ), f"Subagent should have non-null parent_session_id, got {parent_session_id}"
        assert (
            parent_session_id == session_id
        ), f"parent_session_id should be {session_id}, got {parent_session_id}"


class TestMutationResistance:
    """Mutation tests: prove these tests actually guard the implementation."""

    def test_mutation_removing_drain_call_fails(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """If drain is disabled, this test would fail (mutation guard).

        Verifies that index() actually drains the queue by checking that
        items enqueued before index() are gone afterward. If _drain_workqueue()
        were removed/disabled, this would fail.
        """
        project_dir, transcript_path = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Enqueue an item before indexing
        queue = WorkQueue(index_dir)
        queue.open()
        queue.enqueue(
            transcript_path=str(transcript_path),
            session_id="test-001",
            reason="Mutation test",
        )
        queue.close()

        # Verify item is queued
        queue = WorkQueue(index_dir)
        queue.open()
        before = len(queue.pending())
        queue.close()
        assert before == 1, "Item should be queued before index()"

        # Call index, which should drain the queue
        indexer.index(project_dir, quiet=True)

        # Verify queue is now empty (FAILS if drain is disabled/removed)
        queue = WorkQueue(index_dir)
        queue.open()
        after = len(queue.pending())
        queue.close()
        assert after == 0, "Queue should be drained by index()"

    def test_mutation_breaking_claimthendelete_order_fails(
        self, project_with_transcripts: tuple[Path, Path], isolated_home: tuple[Path, Path]
    ) -> None:
        """If items are deleted on claim instead of complete, this test fails.

        Verifies crash-safety by confirming that items claimed but not
        completed remain in abandoned state for retry. If delete happened on
        claim instead of complete, claimed items would be lost.
        """
        project_dir, transcript_path = project_with_transcripts
        index_dir = project_dir / ".ssgrep"

        # Enqueue an item
        queue = WorkQueue(index_dir)
        queue.open()
        queue.enqueue(
            transcript_path=str(transcript_path),
            session_id="test-001",
            reason="Crash safety test",
        )
        queue.close()

        # Claim the item (simulating start of processing)
        queue = WorkQueue(index_dir)
        queue.open()
        item = queue.claim()
        assert item is not None, "Should successfully claim item"
        claimed_item_id = item.item_id
        queue.close()

        # Verify the claimed item appears in abandoned (FAILS if deleted on claim)
        queue = WorkQueue(index_dir)
        queue.open()
        abandoned = queue.abandoned()
        queue.close()
        assert len(abandoned) == 1, "Claimed item should appear in abandoned"
        assert abandoned[0].item_id == claimed_item_id, "Claimed item ID should match"

        # Verify item is gone only after calling complete()
        # (If delete happened on claim, item would be already gone)
        queue = WorkQueue(index_dir)
        queue.open()
        queue.complete(claimed_item_id)
        queue.close()

        queue = WorkQueue(index_dir)
        queue.open()
        abandoned = queue.abandoned()
        queue.close()
        assert len(abandoned) == 0, "Item should be gone after complete()"
