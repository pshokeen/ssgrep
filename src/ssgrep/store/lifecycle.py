"""Session lifecycle and tombstoning operations."""

from __future__ import annotations

import sqlite3


def tombstone_session_chunks(conn: sqlite3.Connection, session_id: str) -> None:
    """Mark chunks from a session as from an absent (tombstoned) source.

    Instead of deleting chunks, vectors, FTS entries, and episode metadata,
    mark them with source_status='absent'. Chunks remain searchable and the
    file cursor is retained so a reappearing source resumes from the stored
    offset rather than re-indexing from scratch.
    """
    conn.execute("UPDATE chunks SET source_status = 'absent' WHERE session_id = ?", (session_id,))
    conn.execute("UPDATE episodes SET source_status = 'absent' WHERE session_id = ?", (session_id,))
    conn.execute("UPDATE sessions SET source_status = 'absent' WHERE session_id = ?", (session_id,))


def delete_session_chunks(conn: sqlite3.Connection, session_id: str) -> None:
    """Hard-delete chunks, episodes, and sessions (used only by explicit prune).

    This is the only routine that permanently removes data. It must never be
    called during indexing or reconciliation — only by explicit prune.
    """
    # Collect chunk_ids BEFORE deleting from chunks (so the subquery still works)
    chunk_ids = [
        row[0]
        for row in conn.execute(
            "SELECT chunk_id FROM chunks WHERE session_id = ?", (session_id,)
        ).fetchall()
    ]

    # Get session path to clean up the cursor row (Defect 2 fix)
    session_path = conn.execute(
        "SELECT path FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    # Delete from FTS first, using the collected chunk_ids
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
        conn.execute(f"DELETE FROM chunks_fts_tri WHERE chunk_id IN ({placeholders})", chunk_ids)

    # NOW delete from chunks and related tables
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # Delete the session_files cursor row to fix staleness poison (Defect 2)
    if session_path:
        conn.execute("DELETE FROM session_files WHERE path = ?", (session_path[0],))


def mark_source_available(conn: sqlite3.Connection, session_id: str) -> None:
    """Mark chunks and episodes from a session as available (reappeared).

    When a previously tombstoned source reappears, clear its absence marker.
    """
    conn.execute(
        "UPDATE chunks SET source_status = 'available' WHERE session_id = ?", (session_id,)
    )
    conn.execute(
        "UPDATE episodes SET source_status = 'available' WHERE session_id = ?", (session_id,)
    )
    conn.execute(
        "UPDATE sessions SET source_status = 'available' WHERE session_id = ?", (session_id,)
    )


def get_tombstone_stats(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return counts of tombstoned sources and their retained chunks.

    Returns: (tombstoned_source_count, tombstoned_chunk_count)
    """
    tombstoned_sources = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM sessions WHERE source_status = 'absent'"
    ).fetchone()[0]
    tombstoned_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE source_status = 'absent'"
    ).fetchone()[0]
    return (tombstoned_sources, tombstoned_chunks)
