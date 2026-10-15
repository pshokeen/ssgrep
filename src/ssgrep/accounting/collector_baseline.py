"""Baselining: position a source at the CURRENT end of its data without reading content.

A baseline source keeps its history unimported ("skipped") but every record appended after
the baseline is read normally. JSONL sources are positioned just past their last newline
(so an in-flight torn line is still read once completed). Native databases cannot
be baselined by file size or a peek at table maxima: they remain explicitly
unvisited until a bounded native read. Admission windows control historical import.
"""

from __future__ import annotations

from typing import Any

from ssgrep.accounting.collector_store import SourceRecord
from ssgrep.accounting.readers.base import Source, SourceState, checkpoint
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.opencode_cursor import progress
from ssgrep.accounting.readers.source_io import SourceFile, open_source

KIB = 1024
MIB = 1024 * KIB
FIRST_CHUNK = 64 * KIB


def last_newline_end(handle: SourceFile, size: int) -> int:
    """Byte just past the last newline within the pinned source's ``size`` bytes."""
    end = size
    chunk = FIRST_CHUNK
    while end > 0:
        start = max(0, end - chunk)
        index = handle.read(start, end - start).rfind(b"\n")
        if index >= 0:
            return start + index + 1
        end = start
        chunk = min(chunk * 2, MIB)
    return 0


def _record(source: Source, **fields: Any) -> SourceRecord:
    return SourceRecord(source.runtime, source.source_id, baseline="skipped", **fields)


def _baseline_jsonl(reader: Any, source: Source) -> SourceRecord:
    with open_source(source.path) as handle:
        info = handle.stat()
        offset = last_newline_end(handle, info.st_size)
        state = checkpoint(handle, offset, reader.baseline_context(source, offset, handle))
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
    extra = {"identity": identity}
    progress(extra)  # Discovery/baseline is not proof that any native row was visited.
    state = SourceState(dev=identity["dev"], inode=identity["inode"], extra=extra)
    info = source.path.stat()
    return _record(
        source,
        state=state.to_dict(),
        size=info.st_size,
        complete=False,
        mtime_ns=info.st_mtime_ns,
    )


def baseline_record(reader: Any, source: Source) -> SourceRecord:
    """Position JSONL at its current end; native DB stays unvisited, never complete."""
    try:
        if isinstance(reader, OpenCodeReader):
            return _baseline_database(reader, source)
        return _baseline_jsonl(reader, source)
    except OSError:
        return _record(source, status="unreadable")
