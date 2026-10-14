"""Source discovery and bounded, resumable reading for one collection cycle.

Reads only through the shared readers (never executing or fetching anything found in a
transcript, never writing to a source). A source's checkpoint always advances atomically with
the evidence of the SAME read call, so a cycle that stops anywhere neither loses nor repeats
records when the next cycle resumes from the returned ``SourceRecord``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ssgrep.accounting.collector_baseline import baseline_record
from ssgrep.accounting.collector_store import (
    CONSUMED_KINDS,
    SourceRecord,
    evidence_line,
    source_key,
)
from ssgrep.accounting.contracts import RUNTIMES, RawEvidence
from ssgrep.accounting.readers.base import Source, SourceState
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader

__all__ = [
    "Discovered",
    "ReadOutcome",
    "baseline_record",
    "discover",
    "make_readers",
    "read_sources",
]

Discovered = dict[str, tuple[Any, Source]]


class Dedupe(Protocol):
    """What reading needs of an evidence ledger (a ``ClassifyingLedger`` or a lazily seeded one)."""

    def admit(self, evidence: RawEvidence) -> bool: ...

    def rejection(self, evidence: RawEvidence) -> str: ...


UNAVAILABLE_GAPS = frozenset({"database_unavailable", "unsupported_schema"})


def make_readers(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """One reader per runtime (``env=None`` => the process environment)."""
    readers: dict[str, Any] = {
        "claude": ClaudeReader(env),
        "codex": CodexReader(env),
        "omp": OmpReader(env),
        "opencode": OpenCodeReader(env),
        "pi": PiReader(env),
        "prime-agent": PrimeReader(env),
    }
    assert set(readers) == set(RUNTIMES)
    return readers


def discover(readers: Mapping[str, Any]) -> Discovered:
    """Every native source keyed by ``source_key``; one failing runtime hides no other."""
    found: Discovered = {}
    for reader in readers.values():
        try:
            sources = reader.discover()
        except OSError:
            continue
        for source in sources:
            found.setdefault(source_key(source.source_id), (reader, source))
    return dict(sorted(found.items()))


@dataclass
class ReadOutcome:
    evidence: list[RawEvidence] = field(default_factory=list)
    sources: dict[str, SourceRecord] = field(default_factory=dict)
    bytes_read: int = 0
    stopped_by: str = "complete"  # complete | byte_budget | time_budget | stop_requested
    gaps: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    resets: int = 0
    unreadable: int = 0
    unrecognized: int = 0


def _bump(table: dict[str, int], reason: str, by: int = 1) -> None:
    table[reason] = table.get(reason, 0) + by


def _copy(record: SourceRecord) -> SourceRecord:
    return replace(record, gaps=dict(record.gaps))


def _refresh_unreached(reader: Any, source: Source, record: SourceRecord) -> SourceRecord:
    """Cheap stat only: lag stays current for a source this cycle never reached."""
    try:
        info = os.stat(source.path)
    except OSError:
        return replace(record, status="unreadable", complete=False)
    if isinstance(reader, OpenCodeReader):
        grown = info.st_size != record.size
        complete = record.complete and not record.more and not grown
    else:
        complete = not record.more and info.st_size <= record.offset and record.pending_tail == 0
    return replace(record, size=info.st_size, mtime_ns=info.st_mtime_ns, complete=complete)


class _Run:
    """Mutable state of one ``read_sources`` call (keeps the public function small)."""

    def __init__(self, ledger, admit, byte_budget, time_budget, clock, should_stop) -> None:
        self.ledger = ledger
        self.admit = admit
        self.byte_budget = byte_budget
        self.time_budget = time_budget
        self.clock = clock
        self.should_stop = should_stop
        self.start = clock()
        self.out = ReadOutcome()

    def limit(self) -> str | None:
        if self.should_stop():
            return "stop_requested"
        if self.byte_budget - self.out.bytes_read <= 0:
            return "byte_budget"
        if self.clock() - self.start >= self.time_budget:
            return "time_budget"
        return None

    def take(self, items: list[RawEvidence], record: SourceRecord) -> None:
        out = self.out
        for item in items:
            if item.kind not in CONSUMED_KINDS:
                _bump(out.dropped, "kind_not_consumed")
                continue
            reason = self.admit(item)
            if reason is not None:
                _bump(out.dropped, reason)
                continue
            if not self.ledger.admit(item):
                _bump(out.dropped, self.ledger.rejection(item))
                continue
            out.evidence.append(item)
            if item.ts_utc is not None and (record.last_ts is None or item.ts_utc > record.last_ts):
                record.last_ts = item.ts_utc

    def read(self, reader: Any, source: Source, record: SourceRecord) -> SourceRecord:
        """Read one source until it is finished, unreadable, or a limit stops it."""
        out = self.out
        record = _copy(record)
        opencode = isinstance(reader, OpenCodeReader)
        state = SourceState.from_dict(record.state)
        while True:
            try:
                info = os.stat(source.path)
                budget = None if opencode else self.byte_budget - out.bytes_read
                result = reader.read(source, state, budget)
            except OSError:
                out.unreadable += 1
                return replace(record, status="unreadable", complete=False)
            unavailable = {g.reason for g in result.gaps} & UNAVAILABLE_GAPS
            if "database_unavailable" in unavailable:
                out.unreadable += 1
                return replace(record, status="unreadable", complete=False)
            charged = result.bytes_read
            if opencode:
                charged = max(charged, sum(len(evidence_line(e)) for e in result.evidence))
            out.bytes_read += charged
            self.take(result.evidence, record)
            for gap in result.gaps:
                _bump(out.gaps, gap.reason)
                _bump(record.gaps, gap.reason)
            if result.reset is not None:
                record.resets += 1
                out.resets += 1
                _bump(out.gaps, f"reset_{result.reset}")
                _bump(record.gaps, f"reset_{result.reset}")
            out.unrecognized += sum(result.unrecognized.values())
            state = result.state
            # the scanner flags ``more`` when the budget lands exactly on the last record
            more = result.more and (opencode or state.offset < info.st_size)
            finished = not more and (
                result.pending_records == 0 if opencode else result.pending_bytes == 0
            )
            caught_up = opencode or state.offset >= info.st_size
            record = replace(
                record,
                state=state.to_dict(),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                status="ok",
                pending_tail=result.pending_bytes,
                more=more,
                complete=finished and caught_up and not unavailable,
            )
            if not more:
                return record
            if (reason := self.limit()) is not None:
                out.stopped_by = reason
                return record


def read_sources(
    *,
    discovered: Discovered,
    known: Mapping[str, SourceRecord],
    admit: Callable[[RawEvidence], str | None],
    ledger: Dedupe,
    byte_budget: int,
    time_budget: float,
    clock: Callable[[], float],
    should_stop: Callable[[], bool] = lambda: False,
    baseline_new: bool = False,
    on_source: Callable[[str, SourceRecord], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> ReadOutcome:
    """Read what the budgets allow; unfinished sources resume from their returned record.

    ``on_bytes(total)`` fires as each source settles with the bytes this cycle has consumed so
    far, so a progress reader sees real numbers while the cycle is still running.
    """
    run = _Run(ledger, admit, byte_budget, time_budget, clock, should_stop)
    out = run.out

    def settle(key: str, record: SourceRecord) -> None:
        out.sources[key] = record
        if on_source is not None:
            on_source(key, record)
        if on_bytes is not None:
            on_bytes(out.bytes_read)

    def unfinished(key: str) -> bool:
        old = known.get(key)
        if old is None or old.more or not old.complete:
            return True
        reader, source = discovered[key]
        try:  # a cheap stat spots growth since the last cycle
            size = os.stat(source.path).st_size
        except OSError:
            return False
        return size != old.size or (not isinstance(reader, OpenCodeReader) and size > old.offset)

    order = sorted(discovered, key=lambda key: (not unfinished(key), key))
    for key in sorted(set(known) - set(discovered)):
        settle(key, replace(_copy(known[key]), status="missing", complete=False))
    for key in order:
        reader, source = discovered[key]
        old = known.get(key)
        if old is None and baseline_new:  # cheap, so never deferred: later data must not be skipped
            record = baseline_record(reader, source)
            out.unreadable += int(record.status == "unreadable")
            settle(key, record)
            continue
        record = old if old is not None else SourceRecord(source.runtime, source.source_id)
        if out.stopped_by == "complete" and (reason := run.limit()) is not None:
            out.stopped_by = reason
        if out.stopped_by != "complete":
            settle(key, _refresh_unreached(reader, source, record))
            continue
        settle(key, run.read(reader, source, record))
    return out
