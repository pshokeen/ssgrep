"""Bounded, newline-complete JSONL scanning.

The scanner never holds more than one in-progress record (at most
``max_record`` bytes) plus one read chunk. A line that exceeds the cap is
discarded *incrementally*: its bytes are counted and dropped as they stream
past, so an 8 MiB+ record is never loaded. Only bytes up to a newline are
reported as complete; an unterminated tail stays pending and is never
accepted, regardless of whether it happens to parse.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.readers.source_io import SourceFile, open_source

CHUNK_BYTES = 1024 * 1024


@dataclass
class Line:
    """One newline-complete record; ``data`` is None when it was oversized."""

    offset: int
    size: int
    data: bytearray | None
    oversized_bytes: int = 0


class JsonlScanner:
    """Iterate complete lines of ``path`` from ``start``; read ``committed``/``pending`` after."""

    def __init__(
        self,
        path: Path,
        start: int = 0,
        *,
        max_record: int = MAX_RAW_RECORD_BYTES,
        budget: int | None = None,
        chunk: int = CHUNK_BYTES,
        source: SourceFile | None = None,
        discarded: int = 0,
        finish_record: bool = True,
    ) -> None:
        self.path = path
        self.start = start
        self.max_record = max_record
        self.budget = budget
        self.chunk = chunk
        self.source = source
        self.discarded = discarded
        self.finish_record = finish_record
        self.bytes_read = 0
        self.committed = start
        self.pending = 0
        self.more = False

    def __iter__(self) -> Generator[Line, None, None]:
        buf = bytearray()
        dropped = self.discarded
        skipping = dropped > 0
        line_start = self.start
        position = self.start + dropped
        authority = open_source(self.path) if self.source is None else nullcontext(self.source)
        with authority as source:
            while True:
                remaining = None if self.budget is None else self.budget - self.bytes_read
                if remaining is not None and remaining <= 0:
                    if skipping or not buf or not self.finish_record:
                        self.discarded = dropped if skipping else 0
                        self.more = True
                        return
                    # A started, under-cap record may finish, but discard cannot
                    # consume arbitrarily many bytes after the quantum expires.
                    width = min(self.chunk, self.max_record - len(buf) + 1)
                else:
                    width = self.chunk if remaining is None else min(self.chunk, remaining)
                block = source.read(position, width)
                self.bytes_read += len(block)
                if not block:
                    break
                position += len(block)
                cursor = 0
                while cursor < len(block):
                    newline = block.find(b"\n", cursor)
                    end = len(block) if newline < 0 else newline
                    width = end - cursor
                    if skipping:
                        dropped += width
                    elif len(buf) + width > self.max_record:
                        skipping = True
                        dropped = len(buf) + width
                        buf = bytearray()
                    else:
                        buf += block[cursor:end]
                    if newline < 0:
                        break
                    size = (dropped if skipping else len(buf)) + 1
                    yield Line(line_start, size, None if skipping else buf, dropped)
                    line_start += size
                    self.committed = line_start
                    buf = bytearray()
                    dropped = 0
                    skipping = False
                    cursor = newline + 1
                    consumed = line_start - self.start - self.discarded
                    if self.budget is not None and consumed >= self.budget:
                        self.discarded = 0
                        self.more = True
                        return
        self.discarded = dropped if skipping else 0
        self.pending = dropped if skipping else len(buf)
