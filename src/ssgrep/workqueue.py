"""Crash-safe work queue for pending index operations.

The work queue stores hints from SessionEnd hooks that need indexing.
Items are enqueued by short-lived hook processes and drained by the indexer.
The queue survives process death and de-duplicates repeated hints.

Storage: SQLite database in .ssgrep/workqueue.db using WAL mode for durability.
Durability mechanism: WAL (Write-Ahead Logging) ensures all writes are durable
at the filesystem level before fsync returns. Explicit transactions ensure
claim-then-delete semantics where claimed items remain replayable if the
process dies before completion.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkItem:
    """A unit of pending index work enqueued by a hook."""

    item_id: str
    transcript_path: str
    session_id: str
    reason: str
    enqueue_time: float
    claimed_at: float | None = None


class WorkQueue:
    """Crash-safe queue of pending index work."""

    def __init__(self, index_dir: Path):
        """Initialize the work queue at the given index directory.

        Args:
            index_dir: Path to .ssgrep/ directory where workqueue.db lives.
        """
        self.index_dir = index_dir
        self.queue_path = index_dir / "workqueue.db"
        self.conn: sqlite3.Connection | None = None

    def open(self) -> None:
        """Open the work queue database, creating schema if needed."""
        self.index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.conn = sqlite3.connect(str(self.queue_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS work_items (
                item_id TEXT PRIMARY KEY,
                transcript_path TEXT NOT NULL,
                session_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                enqueue_time REAL NOT NULL,
                claimed_at REAL
            );
        """)
        self.conn.commit()

    def close(self) -> None:
        """Close the work queue database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _item_id(self, transcript_path: str, session_id: str) -> str:
        """Derive a stable, content-addressed ID for a work item.

        ID is hash of (transcript_path, session_id) so duplicates collapse.
        """
        key = f"{transcript_path}:{session_id}".encode()
        return hashlib.sha256(key).hexdigest()[:16]

    def enqueue(self, transcript_path: str, session_id: str, reason: str) -> None:
        """Enqueue a work item, idempotent and crash-safe.

        If a work item for the same (transcript_path, session_id) already
        exists, this call is a no-op (de-duplication). Enqueue must be
        fast and never raise, so hooks do not break on failure.

        Args:
            transcript_path: Path to the transcript file to index.
            session_id: Session ID derived by discovery.
            reason: Human-readable reason for indexing (e.g. 'SessionEnd hook').
        """
        try:
            if not self.conn:
                self.open()

            assert self.conn is not None
            item_id = self._item_id(transcript_path, session_id)
            now = time.time()

            # INSERT OR REPLACE: idempotent de-duplication.
            # If an item with this ID exists (whether claimed or not),
            # we only update the reason and reset claimed_at to None.
            # This ensures a new hint for a session re-queues it even if
            # a previous attempt was in progress.
            self.conn.execute(
                """
                INSERT OR REPLACE INTO work_items
                (item_id, transcript_path, session_id, reason, enqueue_time, claimed_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (item_id, transcript_path, session_id, reason, now),
            )
            self.conn.commit()
        except Exception:
            # Never raise into the hook's caller. Log if needed, but exit cleanly.
            # The hook must not break the user's session exit.
            pass

    def claim(self) -> WorkItem | None:
        """Claim the next pending work item for processing.

        Returns the oldest unclaimed work item, marking it as claimed.
        Claimed items remain in the queue until explicitly marked complete
        or deleted. If the process dies after claiming but before completing,
        the item remains replayable on recovery.

        Returns:
            A WorkItem if one is available, None if queue is empty.
        """
        if not self.conn:
            return None

        try:
            # Find oldest unclaimed item (claimed_at IS NULL).
            row = self.conn.execute(
                """
                SELECT item_id, transcript_path, session_id, reason, enqueue_time
                FROM work_items
                WHERE claimed_at IS NULL
                ORDER BY enqueue_time ASC
                LIMIT 1
                """
            ).fetchone()

            if not row:
                return None

            item_id, transcript_path, session_id, reason, enqueue_time = row

            # Mark it as claimed (this transaction ensures atomicity).
            now = time.time()
            self.conn.execute(
                "UPDATE work_items SET claimed_at = ? WHERE item_id = ?",
                (now, item_id),
            )
            self.conn.commit()

            return WorkItem(
                item_id=item_id,
                transcript_path=transcript_path,
                session_id=session_id,
                reason=reason,
                enqueue_time=enqueue_time,
                claimed_at=now,
            )
        except Exception:
            return None

    def complete(self, item_id: str) -> None:
        """Mark a claimed work item as complete and remove it from queue.

        Args:
            item_id: The item_id of the work item to complete.
        """
        if not self.conn:
            return

        try:
            # Delete the item from the queue (it has been successfully processed).
            self.conn.execute(
                "DELETE FROM work_items WHERE item_id = ?",
                (item_id,),
            )
            self.conn.commit()
        except Exception:
            pass

    def pending(self) -> list[WorkItem]:
        """List all unclaimed items currently in the queue.

        Returns:
            List of WorkItem objects for unclaimed items, oldest first.
        """
        if not self.conn:
            return []

        try:
            rows = self.conn.execute(
                """
                SELECT item_id, transcript_path, session_id, reason, enqueue_time
                FROM work_items
                WHERE claimed_at IS NULL
                ORDER BY enqueue_time ASC
                """
            ).fetchall()

            return [
                WorkItem(
                    item_id=row[0],
                    transcript_path=row[1],
                    session_id=row[2],
                    reason=row[3],
                    enqueue_time=row[4],
                    claimed_at=None,
                )
                for row in rows
            ]
        except Exception:
            return []

    def abandoned(self) -> list[WorkItem]:
        """List work items that were claimed but never completed.

        These are items left in a claimed state by a process that crashed
        before marking them complete. They should be re-processed on recovery.

        Returns:
            List of WorkItem objects for claimed-but-incomplete items.
        """
        if not self.conn:
            return []

        try:
            rows = self.conn.execute(
                """
                SELECT item_id, transcript_path, session_id, reason, enqueue_time, claimed_at
                FROM work_items
                WHERE claimed_at IS NOT NULL
                ORDER BY claimed_at ASC
                """
            ).fetchall()

            return [
                WorkItem(
                    item_id=row[0],
                    transcript_path=row[1],
                    session_id=row[2],
                    reason=row[3],
                    enqueue_time=row[4],
                    claimed_at=row[5],
                )
                for row in rows
            ]
        except Exception:
            return []

    def clear(self) -> None:
        """Remove all items from the work queue.

        Used only in tests; normally only complete() removes items.
        """
        if not self.conn:
            return

        try:
            self.conn.execute("DELETE FROM work_items")
            self.conn.commit()
        except Exception:
            pass
