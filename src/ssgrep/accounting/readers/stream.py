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
from dataclasses import dataclass
from pathlib import Path

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES

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
    ) -> None:
        self.path = path
        self.start = start
        self.max_record = max_record
        self.budget = budget
        self.chunk = chunk
        self.committed = start
        self.pending = 0
        self.more = False

    def __iter__(self) -> Generator[Line, None, None]:
        buf = bytearray()
        dropped = 0
        skipping = False
        line_start = self.start
        with self.path.open("rb") as handle:
            handle.seek(self.start)
            while True:
                block = handle.read(self.chunk)
                if not block:
                    break
                cursor = 0
                while True:
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
                    if self.budget is not None and line_start - self.start >= self.budget:
                        self.more = True
                        return
        self.pending = dropped if skipping else len(buf)
