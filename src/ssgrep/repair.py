"""Bounded tail repair on the search path.

When search detects that a session has grown since it was indexed, this
module attempts to reconcile just the appended tail within strict byte and
time caps. Non-blocking: if the write lock is unavailable, returns
immediately. Unbounded appends are refused with an actionable message rather
than silently performing multi-second work. The time budget gates only the
PRE-ENCODE decision: once encoding/writing has started the work is finished
and committed even if it overruns (a committed write must never be refused
retroactively), and an overrun is reported to stderr rather than hidden.

Per D13, staleness detection is cheap (~22ms for all 1,209 project files);
repair of small appends is cheaper still (4.66ms reference for 3,870 bytes).
Large backlogs are refused rather than attempted. The search latency budget
(250ms total, 184ms of which is model load) has ~66ms headroom: this operation
must live in well under 10ms or defer to the next explicit index() call.

This writer never calls GenerationalStore.recover(). It does not need to:
search's index-open step (search._open_ready_index) runs
vectors.validate_alignment and raises IndexNotReadyError before staleness
detection -- and therefore before repair_current_session_tail() is ever
called -- whenever a chunk row references a vec_row the vector file does
not actually have. Residue from an earlier crash is refused upstream by
that detect-and-refuse gate (D13), never recovered here; repair is only
ever reached against a generation search has already confirmed is aligned.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from ssgrep import chunker, embed, indexer, metadata, records, store, vectors
from ssgrep.types import Chunk, FileCursor, SessionFile

# Strict caps per the requirement.
# 256 KB: 4x larger than the 4.66ms reference (3.87 KB) but still "small".
# A transcript line averaging ~50 bytes -> ~5120 lines before refusing.
# Conservative to stay well under the 10ms budget.
MAX_APPEND_BYTES = 256_000

# Millisecond timeout for the operation. Reference is 4.66ms for 3.87 KB;
# we budget 20ms to handle variance and system load before declining repair.
MAX_REPAIR_MS = 20

# Global non-blocking lock to prevent concurrent repair attempts.
# SQLite's built-in locking handles writes, but we need application-level
# coordination to avoid two threads both attempting to repair the same file.
_repair_lock = threading.Lock()


@dataclass(frozen=True)
class RepairResult:
    """Outcome of a tail repair attempt.

    Attributes:
        repair_success: True if repair completed, False if declined or failed.
        message: Human-readable explanation if repair was declined; None if
            successful or if the attempt failed silently (lock unavailable).
    """

    repair_success: bool
    message: str | None = None


def repair_current_session_tail(
    project_dir: Path,
    session_path: Path,
    session_id: str,
    index_dir: Path | None = None,
) -> RepairResult:
    """Attempt to repair a session's tail if it has grown since indexing.

    Non-blocking: returns immediately if the write lock is unavailable.
    Refuses large backlogs with an actionable message rather than blocking.

    Args:
        project_dir: The project directory (used for metadata loading).
        session_path: Path to the session transcript file.
        session_id: The session's unique identifier.
        index_dir: Optional override for the .ssgrep directory; defaults to
            project_dir / ".ssgrep".

    Returns:
        RepairResult indicating success, refusal (with message), or lock
        unavailability (message=None, repair_success=False).
    """
    start_time = time.perf_counter()
    index_dir = index_dir or (project_dir / ".ssgrep")

    # Non-blocking lock acquire. If unavailable, return immediately.
    if not _repair_lock.acquire(blocking=False):
        return RepairResult(repair_success=False)

    try:
        # Check that the file exists and is readable.
        try:
            disk_stat = session_path.stat()
        except (FileNotFoundError, OSError):
            # File vanished or became unreadable; nothing to repair.
            return RepairResult(repair_success=False)

        disk_size = disk_stat.st_size

        # Open the index to read the stored cursor and session metadata.
        gen_store = store.GenerationalStore(index_dir)
        db_path = gen_store.get_index_path()
        if not db_path.exists():
            return RepairResult(repair_success=False)

        # Hold the current generation to prevent concurrent rebuilds from
        # unlinking the live generation's files while this repair is mid-write.
        # See GenerationalStore.hold_generation() docstring.
        guard = ExitStack()
        guard.enter_context(gen_store.hold_generation(gen_store.current_generation))
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA busy_timeout=1000")  # 1 second, non-blocking fail
            try:
                # Read the stored cursor for this file.
                cursor = store.get_session_file(conn, session_path)
                if cursor is None:
                    return RepairResult(repair_success=False)

                # Guard against rewrite/truncation: check first line hash matches.
                current_hash = indexer._first_line_hash(session_path)
                if current_hash != cursor.first_line_hash:
                    # File was rewritten or truncated; refuse repair
                    return RepairResult(repair_success=False)

                # Check if the file has grown.
                if disk_size <= cursor.byte_offset:
                    # File hasn't changed; nothing to repair but not an error
                    return RepairResult(repair_success=True)

                append_size = disk_size - cursor.byte_offset

                # Refuse if the append exceeds the cap.
                if append_size > MAX_APPEND_BYTES:
                    msg = (
                        f"Session transcript has grown by {append_size} bytes "
                        f"({append_size / 1024:.1f} KB) since last index. "
                        f"Run `ssgrep index` to reconcile; search will use cached results "
                        f"until then."
                    )
                    return RepairResult(repair_success=False, message=msg)

                # Time check: if we're already over budget, don't attempt.
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms > MAX_REPAIR_MS * 0.5:
                    return RepairResult(repair_success=False)

                # Parse the tail.
                try:
                    new_records, end_offset, rstats = indexer._read_from_offset(
                        session_path, cursor.byte_offset, records.MAX_LINE_BYTES
                    )
                except (FileNotFoundError, OSError):
                    return RepairResult(repair_success=False)

                if not new_records:
                    # Nothing new to parse; update cursor and succeed.
                    new_hash = indexer._first_line_hash(session_path)
                    store.upsert_session_file(
                        conn,
                        FileCursor(
                            path=session_path,
                            size=disk_size,
                            mtime=disk_stat.st_mtime,
                            byte_offset=end_offset,
                            first_line_hash=new_hash,
                        ),
                    )
                    conn.commit()
                    return RepairResult(repair_success=True)

                # Time check before building episodes.
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms > MAX_REPAIR_MS:
                    return RepairResult(repair_success=False)

                # Load agent metadata if this is a subagent.
                session_row = conn.execute(
                    "SELECT is_main FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                is_main = session_row[0] if session_row else True

                agent_meta = None
                if not is_main:
                    agent_meta = metadata.load_agent_meta(session_path.with_suffix(".meta.json"))

                # Reconstruct the SessionFile object for enrichment.
                session = SessionFile(
                    path=session_path,
                    session_id=session_id,
                    is_main=is_main,
                    size=disk_size,
                    mtime=disk_stat.st_mtime,
                )

                # Build episodes from the new records.
                start_ep_index = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]

                new_episodes = indexer._build_episodes(
                    new_records, session, agent_meta, start_ep_index
                )

                if not new_episodes:
                    # No episodes extracted; just update cursor.
                    new_hash = indexer._first_line_hash(session_path)
                    store.upsert_session_file(
                        conn,
                        FileCursor(
                            path=session_path,
                            size=disk_size,
                            mtime=disk_stat.st_mtime,
                            byte_offset=end_offset,
                            first_line_hash=new_hash,
                        ),
                    )
                    conn.commit()
                    return RepairResult(repair_success=True)

                # Build chunks and prepare for insertion.
                new_chunks: list[Chunk] = []
                for ep in new_episodes:
                    store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)
                    new_chunks.extend(chunker.chunk_episode(ep))

                vectors_appended = False
                new_vec_row_count = None
                if new_chunks:
                    # Open the vector store.
                    vec_path = gen_store.get_vector_path()
                    vec_store = vectors.open_vectors(vec_path, dimension=embed.DIMENSION)
                    try:
                        # Embed the new chunks.
                        vecs = embed.encode([c.text for c in new_chunks])
                        vec_rows = vectors.append(vec_store, vecs)
                        vectors_appended = True
                        # Capture the row count before closing vec_store, to use in checkpoint()
                        # after commit. This matches indexer.py line 420's pattern exactly.
                        new_vec_row_count = vec_store.row_count

                        # Insert chunks with their vector rows.
                        for chunk, vec_row in zip(new_chunks, vec_rows, strict=True):
                            store.insert_chunk(conn, chunk, vec_row)
                    finally:
                        vectors.close(vec_store)

                # Update the file cursor.
                new_hash = indexer._first_line_hash(session_path)
                store.upsert_session_file(
                    conn,
                    FileCursor(
                        path=session_path,
                        size=disk_size,
                        mtime=disk_stat.st_mtime,
                        byte_offset=end_offset,
                        first_line_hash=new_hash,
                    ),
                )

                conn.commit()

                # Checkpoint the vector row count after the commit, so that recover()
                # has an accurate watermark for the appended vectors. This is only
                # strictly needed when vectors_appended is True, but calling it when
                # False is harmless -- checkpoint() is safe to call on cursor-only
                # repairs. See GenerationalStore.checkpoint() docstring.
                if vectors_appended and new_vec_row_count is not None:
                    gen_store.checkpoint(new_vec_row_count)

                # Verify we're still within budget.
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms > MAX_REPAIR_MS:
                    # We overran. The write is committed and stands (never
                    # refuse retroactively) -- but say so instead of hiding
                    # it, so a buyer watching a slow search sees why.
                    print(
                        f"ssgrep: tail repair overran its {MAX_REPAIR_MS:.0f}ms budget "
                        f"({elapsed_ms:.0f}ms); the repair is committed and valid. "
                        f"If this recurs, run `ssgrep index` to reconcile in bulk.",
                        file=sys.stderr,
                    )

                return RepairResult(repair_success=True)

            finally:
                conn.close()
        finally:
            guard.close()

    finally:
        _repair_lock.release()
