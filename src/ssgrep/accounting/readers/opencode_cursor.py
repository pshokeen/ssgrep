"""Bounded, snapshot-local continuity checks for OpenCode watermarks."""

from __future__ import annotations

import sqlite3
from typing import Any


def watermarks_lost(connection: sqlite3.Connection, marks: dict[str, Any]) -> bool:
    """Whether a persisted cursor no longer has its monotonic row anchor.

    Device/inode can be reused and birth time is not portable. A missing or
    regressed anchor invalidates the old cursor even when other rows have newer
    timestamps. Query the same committed snapshot as the ensuing scan, by primary
    key, without fetching JSON or relying on file metadata changed by live WALs.

    Advancing an anchor or editing its payload is normal, not proof of a new
    incarnation. A replacement preserving all anchors cannot be distinguished
    from ordinary updates by these cursors; neither a content hash nor mtime would
    make that inference trustworthy. Replay only on demonstrated continuity loss.
    """
    for table in ("session", "message", "part"):
        if table not in marks:
            continue  # An empty/unvisited table has no persisted anchor.
        time, record_id = marks[table]
        row = connection.execute(
            f"SELECT time_updated FROM {table} WHERE id = ?",  # noqa: S608
            (record_id,),
        ).fetchone()
        if row is None or row[0] < time:
            return True
    return False
