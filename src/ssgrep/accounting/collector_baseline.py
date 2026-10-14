"""Baselining: position a source at the CURRENT end of its data without reading content.

A baseline source keeps its history unimported ("skipped") but every record appended after
the baseline is read normally. JSONL sources are positioned just past their last newline
(so an in-flight torn line is still read once completed); OpenCode is positioned by its
per-table ``(time_updated, id)`` high-water marks, taken inside the reader's read-only,
query-only transaction. Nothing here writes to a source.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ssgrep.accounting.collector_store import SourceRecord
from ssgrep.accounting.readers.base import Source, SourceState, checkpoint
from ssgrep.accounting.readers.opencode import OpenCodeReader

KIB = 1024
MIB = 1024 * KIB
FIRST_CHUNK = 64 * KIB
TABLES = ("session", "message", "part")


def last_newline_end(path: Path, size: int) -> int:
    """Byte just past the last newline within ``size`` bytes (0 when there is none)."""
    end = size
    chunk = FIRST_CHUNK
    with path.open("rb") as handle:
        while end > 0:
            start = max(0, end - chunk)
            handle.seek(start)
            index = handle.read(end - start).rfind(b"\n")
            if index >= 0:
                return start + index + 1
            end = start
            chunk = min(chunk * 2, MIB)
    return 0


def _record(source: Source, **fields: Any) -> SourceRecord:
    return SourceRecord(source.runtime, source.source_id, baseline="skipped", **fields)


def _baseline_jsonl(source: Source) -> SourceRecord:
    info = source.path.stat()
    offset = last_newline_end(source.path, info.st_size)
    state = checkpoint(source.path, offset, {})
    return _record(
        source,
        state=state.to_dict(),
        size=info.st_size,
        skipped_bytes=offset,
        pending_tail=info.st_size - offset,
        complete=info.st_size == offset,
        mtime_ns=info.st_mtime_ns,
    )


def _baseline_database(reader: OpenCodeReader, source: Source) -> SourceRecord:
    identity = reader._identity(source.path)
    marks: dict[str, list[Any]] = {}
    with reader.snapshot(source.path) as connection:
        for table in TABLES:  # constants; reads only the three allowed tables
            row = connection.execute(  # noqa: S608
                f"SELECT time_updated, id FROM {table} ORDER BY time_updated DESC, id DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                marks[table] = [row[0], row[1]]
    state = SourceState(
        dev=identity["dev"], inode=identity["inode"], extra={"wm": marks, "identity": identity}
    )
    info = source.path.stat()
    return _record(
        source,
        state=state.to_dict(),
        size=info.st_size,
        complete=True,
        mtime_ns=info.st_mtime_ns,
    )


def baseline_record(reader: Any, source: Source) -> SourceRecord:
    """A record positioned at the current end of ``source``; unreadable => status ``unreadable``."""
    try:
        if isinstance(reader, OpenCodeReader):
            return _baseline_database(reader, source)
        return _baseline_jsonl(source)
    except (OSError, sqlite3.Error):
        return _record(source, status="unreadable")
