"""Source discovery and bounded, resumable reading for one collection cycle.

Reads only through the shared readers (never executing or fetching anything found in a
transcript, never writing to a source). A source's checkpoint always advances atomically with
the evidence of the SAME read call, so a cycle that stops anywhere neither loses nor repeats
records when the next cycle resumes from the returned ``SourceRecord``.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ssgrep.accounting.capture import fingerprint
from ssgrep.accounting.capture.journal_reader import JournalRead
from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.store import (
    CONSUMED_KINDS,
    Checkpoint,
    CollectError,
    SourceRecord,
    source_key,
)
from ssgrep.accounting.ledger.contracts import RUNTIMES, ExecutionEvent, RawEvidence, digest_of
from ssgrep.accounting.privacy.storage import from_plain, to_plain
from ssgrep.accounting.readers.base import (
    EvidenceLedger,
    Source,
    SourceFile,
    SourceState,
    checkpoint,
    open_source,
)
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader, CodexReader
from ssgrep.accounting.readers.opencode import OpenCodeReader, progress
from ssgrep.accounting.readers.pi_family import OmpReader, PiReader, PrimeReader

if TYPE_CHECKING:
    from ssgrep.accounting.collect.collector import CollectOptions
    from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy


MAX_ATOMS = 600
MAX_REFS = 5

REF_RE = re.compile(r"^(.+):ep:\d+$")


def resolve_ref(ref: str, sessions: Mapping[tuple[str, str], object]) -> tuple[str, str] | None:
    """Resolve a source reference against retained collection sessions."""
    found = REF_RE.match(ref)
    if not found:
        return None
    base = found.group(1)
    for runtime in RUNTIMES:
        if base.startswith(runtime + ":"):
            key = (runtime, base[len(runtime) + 1 :])
            if key in sessions:
                return key
    key = ("claude", base)
    if key in sessions:
        return key
    matches = sorted(k for k in sessions if k[1] == base)
    return matches[0] if len(matches) == 1 else None


Admit = Callable[[RawEvidence], str | None]
NO_TEXT_KINDS = frozenset({"session_header", "lineage"})


class ClassifyingLedger(EvidenceLedger):
    def rejection(self, evidence: RawEvidence) -> str:
        first = self._first.get(self._key(evidence))
        return "conflict" if first is not None and first != evidence.digest else "duplicate"


class LazyLedger:
    def __init__(self, profile: str, checkpoint: Checkpoint | None) -> None:
        self._profile = profile
        self._segments = list(checkpoint.segments) if checkpoint else []
        self._ledger = ClassifyingLedger()
        self._seeded: set[tuple[str, str, str]] = set()
        self._index_cache: dict = {}

    def _ready(self, evidence: RawEvidence) -> ClassifyingLedger:
        from ssgrep.accounting.ledger.contracts import target_key

        key = target_key(evidence) or evidence.native_key
        if key not in self._seeded:
            for item in store.iter_evidence(
                self._profile, self._segments, native_keys={key}, index_cache=self._index_cache
            ):
                item = replace(
                    item, payload={k: v for k, v in item.payload.items() if k != "_evidence_format"}
                )
                self._ledger.admit(item)
            self._seeded.add(key)
        return self._ledger

    def admit(self, evidence: RawEvidence) -> bool:
        return self._ready(evidence).admit(evidence)

    def rejection(self, evidence: RawEvidence) -> str:
        return self._ready(evidence).rejection(evidence)

    def ownership_claim(self, evidence: RawEvidence) -> RawEvidence | None:
        return self._ready(evidence).ownership_claim(evidence)


class Retained:
    def __init__(
        self,
        profile: str,
        segments: Sequence[store.SegmentRef],
        *,
        env: Mapping[str, str] | None = None,
        sessions: set[tuple[str, str]] | None = None,
    ) -> None:
        self._profile = profile
        self._segments = list(segments)
        self._sessions = sessions
        self._items: list[RawEvidence] | None = None
        from ssgrep.accounting.collect.replay import EvidenceReplay

        self._replay = EvidenceReplay(profile, env=env)

    def items(self) -> list[RawEvidence]:
        if self._items is None:
            self._items = self._replay.hydrate_many(
                store.iter_evidence(self._profile, self._segments, sessions=self._sessions)
            )
        return self._items

    @property
    def unavailable(self) -> dict[tuple[str, str], str]:
        self.items()
        return dict(self._replay.unavailable)

    @property
    def replay_available(self) -> bool:
        return not self.unavailable


def _seconds(ts: str) -> str:
    return ts[:19]


@dataclass(frozen=True)
class AdmitRules:
    paused: tuple[tuple[str, str | None], ...] = ()
    expired_sessions: frozenset[tuple[str, str]] = frozenset()
    horizon_day: str | None = None
    capture_start: str | None = None
    window: tuple[str, str] | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __call__(self, item: RawEvidence) -> str | None:
        if (item.runtime, item.session_id) in self.expired_sessions:
            return "expired_conversation"
        ts = item.ts_utc
        if self.window is not None:
            return self._within_window(item, ts)
        if ts is None or item.kind in NO_TEXT_KINDS:
            return None
        if self.horizon_day is not None and ts[:10] < self.horizon_day:
            return "beyond_retention"
        second = _seconds(ts)
        if self.capture_start is not None and second < _seconds(self.capture_start):
            return "before_capture"
        for start, end in self.paused:
            if second >= _seconds(start) and (end is None or second < _seconds(end)):
                return "paused_interval"
        return None

    def _within_window(self, item: RawEvidence, ts: str | None) -> str | None:
        assert self.window is not None
        if item.kind in NO_TEXT_KINDS:
            return None
        if ts is None:
            return "outside_interval"
        since, until = self.window
        return None if since <= _seconds(ts)[:19] + "Z" < until else "outside_interval"


def journal_signature(journal: JournalRead) -> str:
    if journal.signature is not None:
        return journal.signature
    digest = hashlib.sha256()
    for event in journal.events:
        digest.update(f"{event.execution_id}|{event.phase}\n".encode())
    for damage in sorted(journal.damage, key=lambda d: (d.segment, d.offset, d.kind)):
        digest.update(f"!{damage.segment}|{damage.offset}|{damage.kind}|{damage.length}\n".encode())
    digest.update(f"c{len(journal.conflicts)}".encode())
    return digest.hexdigest()[:32]


def journal_stats(journal: JournalRead) -> dict[str, int]:
    return {
        "segments": journal.segments,
        "incomplete_tails": len(journal.incomplete_tails),
        "quarantined": len(journal.quarantined),
        "conflicts": len(journal.conflicts),
    }


def journal_last_ts(journal: JournalRead) -> str | None:
    stamps = [event.ts_utc for event in journal.events]
    return max(stamps) if stamps else None


def load_fingerprint_key(profile: str) -> bytes | None:
    try:
        return fingerprint.load_key(profile, create=False)
    except OSError:
        return None


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
    progress(extra)
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

    def ownership_claim(self, evidence: RawEvidence) -> RawEvidence | None: ...


UNAVAILABLE_GAPS = frozenset(
    {
        "database_unavailable",
        "database_snapshot_unavailable",
        "unsupported_schema",
        "database_changed",
    }
)


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


def _source_signature(reader: Any, source: Source) -> list | None:
    """Content-free descriptor metadata; DB writes can live only in its WAL."""
    paths = [source.path]
    if isinstance(reader, OpenCodeReader):
        paths.append(source.path.with_name(source.path.name + "-wal"))
    signature = []
    for path in paths:
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            signature.append(None)
            continue
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        signature.append(
            [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        )
    return signature if signature[0] is not None else None


def _accepted_database_generation(extra: dict) -> dict | None:
    """Only the reader's accepted v3 clean census can authorize an idle skip."""
    stats = extra.get("opencode", {})
    generation = stats.get("source_generation")
    epoch = extra.get("census_epoch", {})
    census = extra.get("census", {})
    if (
        stats.get("cursor_version") != 3
        or stats.get("status") != "complete"
        or stats.get("snapshot_checked") is not True
        or stats.get("discovery_pending")
        or stats.get("known_pending_lower_bound", 0)
        or not isinstance(generation, dict)
        or epoch.get("dirty") is not False
        or epoch.get("generation") != generation
        or not all(
            census.get(table, {}).get("done") is True for table in ("session", "message", "part")
        )
    ):
        return None
    return generation


def _unchanged(reader: Any, source: Source, record: SourceRecord) -> bool:
    if record.status != "ok" or not record.complete or record.more or record.pending_tail:
        return False
    extra = record.state.get("extra", {})
    if isinstance(reader, OpenCodeReader):
        from ssgrep.accounting.readers.opencode_storage import SnapshotError, source_generation

        accepted = _accepted_database_generation(extra)
        if accepted is None or accepted != extra.get("collector_source_generation"):
            return False
        try:
            return source_generation(source.path) == accepted
        except (OSError, SnapshotError):
            return False
    signature = _source_signature(reader, source)
    return signature is not None and signature == extra.get("collector_source_signature")


def _refresh_unreached(reader: Any, source: Source, record: SourceRecord) -> SourceRecord:
    """Cheap stat only: lag stays current for a source this cycle never reached."""
    try:
        info = os.stat(source.path)
    except OSError:
        return replace(record, status="unreadable", complete=False)
    if isinstance(reader, OpenCodeReader):
        state = SourceState.from_dict(deepcopy(record.state))
        stats = progress(state.extra)
        stats["snapshot_checked"] = False
        return replace(
            record,
            state=state.to_dict(),
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            complete=False,
        )
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
                claim = self.ledger.ownership_claim(item)
                if claim is not None:
                    out.evidence.append(claim)
                continue
            out.evidence.append(item)
            if item.ts_utc is not None and (record.last_ts is None or item.ts_utc > record.last_ts):
                record.last_ts = item.ts_utc

    def read(self, reader: Any, source: Source, record: SourceRecord) -> SourceRecord:
        """One bounded source quantum; persistent scheduling prevents starvation."""
        out = self.out
        record = _copy(record)
        opencode = isinstance(reader, OpenCodeReader)
        state = SourceState.from_dict(record.state)
        remaining = self.byte_budget - out.bytes_read
        signature = None if opencode else _source_signature(reader, source)
        try:
            info = os.stat(source.path)
            result = (
                reader.read(source, state, byte_budget=remaining, maintenance=True)
                if opencode
                else reader.read(source, state, remaining)
            )
        except OSError:
            out.unreadable += 1
            _bump(out.gaps, "source_unreadable")
            _bump(record.gaps, "source_unreadable")
            return replace(record, status="unreadable", complete=False)
        unavailable = {g.reason for g in result.gaps} & UNAVAILABLE_GAPS
        out.bytes_read += result.bytes_read
        # Record failure details BEFORE deciding whether a source is unavailable.
        for gap in result.gaps:
            _bump(out.gaps, gap.reason)
            _bump(record.gaps, gap.reason)
        if result.reset is not None:
            record.resets += 1
            out.resets += 1
            _bump(out.gaps, f"reset_{result.reset}")
            _bump(record.gaps, f"reset_{result.reset}")
        out.unrecognized += sum(result.unrecognized.values())
        self.take(result.evidence, record)
        state = result.state
        if opencode:
            accepted = _accepted_database_generation(state.extra)
            state.extra.pop("collector_source_signature", None)
            if accepted is not None:
                state.extra["collector_source_generation"] = accepted
            else:
                state.extra.pop("collector_source_generation", None)
        else:
            after = _source_signature(reader, source)
            if signature is not None and signature == after:
                state.extra["collector_source_signature"] = signature
            else:
                state.extra.pop("collector_source_signature", None)
        more = result.more and (opencode or state.offset < info.st_size)
        finished = not more and (
            _accepted_database_generation(state.extra) is not None
            if opencode
            else result.pending_bytes == 0 and state.offset >= info.st_size
        )
        status = "unsupported" if "unsupported_schema" in unavailable else "ok"
        if unavailable & {"database_unavailable", "database_snapshot_unavailable"}:
            status = "unreadable"
            out.unreadable += 1
        if "database_changed" in unavailable:
            status = "changed"  # Retryable snapshot loss, never completed input.
        return replace(
            record,
            state=state.to_dict(),
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
            status=status,
            pending_tail=result.pending_bytes,
            more=more,
            complete=finished and not unavailable,
        )


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
        return not _unchanged(reader, source, old)

    def served(key: str) -> int:
        old = known.get(key)
        return int(old.state.get("extra", {}).get("collector_schedule", 0)) if old else 0

    order = sorted(discovered, key=lambda key: (not unfinished(key), served(key), key))
    turn = max((served(key) for key in discovered), default=0)
    for key in sorted(set(known) - set(discovered)):
        settle(key, replace(_copy(known[key]), status="missing", complete=False))
    for key in order:
        reader, source = discovered[key]
        old = known.get(key)
        if old is None and baseline_new and not isinstance(reader, OpenCodeReader):
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
        if _unchanged(reader, source, record):
            settle(key, record)
            continue
        record = run.read(reader, source, record)
        turn += 1
        state = SourceState.from_dict(record.state)
        state.extra["collector_schedule"] = turn
        record = replace(record, state=state.to_dict())
        settle(key, record)
    if out.stopped_by == "complete" and (reason := run.limit()) is not None:
        # A budget crossed while finishing the last source leaves nothing unread; only a
        # source stopped mid-read makes the pass partial.
        if any(r.more or not r.complete for r in out.sources.values()):
            out.stopped_by = reason
    return out


"""Per-segment journal parse reuse under the collector's existing commit point.

Unchanged descriptors are never read. Append-only growth resumes at the last
complete offset (a pending tail is reparsed); replacement/truncation reparses only
that segment. The retained cache is scrubbed by the same committed privacy floor
as physical maintenance, and remembers proof-only pending cleanup after scrubbing.
Missing accepted segments require exact committed maintenance authority and all
retained measurements in surviving inputs; otherwise collection fails closed.
"""


JOURNAL_CACHE_BLOB = "journal-cache"


def reuse_segment(result, proof, cached) -> int | None:
    """Return suffix offset, None for unchanged, or zero for a replacement."""
    from ssgrep.accounting.capture.journal_reader import Damage, SegmentProof

    if cached is None:
        return 0
    try:
        before = from_plain(SegmentProof, cached["proof"])
        events = [from_plain(ExecutionEvent, row) for row in cached["events"]]
        damage = [from_plain(Damage, row) for row in cached["damage"]]
    except (KeyError, ValueError, TypeError) as error:
        raise CollectError("state_corrupt", "Cached journal segment is invalid.") from error
    equal = before == proof
    append = (before.device, before.inode) == (
        proof.device,
        proof.inode,
    ) and proof.size > before.size
    if not equal and not append:
        return 0
    incomplete = [row.offset for row in damage if row.kind == "incomplete_tail"]
    offset = min(incomplete, default=before.size)
    result.events.extend(events)
    result.segment_events[proof.name] = events
    result.damage.extend(
        damage if equal else [row for row in damage if row.kind != "incomplete_tail"]
    )
    if cached.get("retention_pending"):
        result.retention_pending.add(proof.name)
    return None if equal else offset


def _require_cached_segments(journal, previous: dict[str, dict], retention: dict | None) -> None:
    """Distinguish proven retention from loss of an accepted journal input."""
    from ssgrep.accounting.capture.journal_reader import Damage, SegmentProof
    from ssgrep.accounting.capture.journal_retention import retained_event

    missing = previous.keys() - {proof.name for proof in journal.proofs}
    if not missing:
        return
    available: set[str] | None = None
    for name in sorted(missing):
        cached = previous[name]
        try:
            proof = from_plain(SegmentProof, cached["proof"])
            events = [from_plain(ExecutionEvent, row) for row in cached["events"]]
            damage = [from_plain(Damage, row) for row in cached["damage"]]
            authorized = retention is not None and proof in [
                from_plain(SegmentProof, row) for row in retention["proofs"]
            ]
        except (KeyError, ValueError, TypeError) as error:
            raise CollectError("state_corrupt", "Cached journal segment is invalid.") from error
        if retention is None or not authorized or proof.name != name:
            raise CollectError(
                "state_corrupt",
                "A committed journal segment is missing without retention authority; "
                "restore the accepted journal before collecting.",
            )
        modified_day = (
            datetime.fromtimestamp(proof.modified_ns // 1_000_000_000, UTC).date().isoformat()
        )
        if damage and modified_day >= retention["horizon_day"]:
            raise CollectError(
                "state_corrupt", "A committed journal segment with a pending record is missing."
            )
        expected = [
            kept for event in events if (kept := retained_event(event, retention)) is not None
        ]
        if not expected:
            continue
        if available is None:
            available = {
                digest_of(kept)
                for rows in journal.segment_events.values()
                for event in rows
                if (kept := retained_event(event, retention)) is not None
            }
        if any(digest_of(event) not in available for event in expected):
            raise CollectError(
                "state_corrupt",
                "A committed journal segment is missing its retained measurements; "
                "restore the accepted journal or its retained replacement before collecting.",
            )


def read_cached_journal(profile: str, checkpoint: store.Checkpoint | None):
    from ssgrep.accounting.capture.journal_reader import read_journal
    from ssgrep.accounting.capture.journal_retention import expand_journal_retention

    raw = store.read_blob(profile, checkpoint.blobs.get(JOURNAL_CACHE_BLOB)) if checkpoint else None
    if raw is not None and (not isinstance(raw, dict) or raw.get("schema") != 1):
        raise CollectError("state_corrupt", "Journal cache manifest is invalid.")
    names = raw.get("segments", {}) if raw else {}
    previous = {}
    for name, filename in names.items():
        segment = store.read_blob(profile, filename)
        if not isinstance(segment, dict):
            raise CollectError("state_corrupt", "Cached journal segment is invalid.")
        previous[name] = segment
    retention = checkpoint.retention.get("journal") if checkpoint else None
    expanded = expand_journal_retention(profile, retention) if retention else None
    journal = read_journal(profile, previous=previous, retention=expanded)
    _require_cached_segments(journal, previous, expanded)
    journal.cache_blobs = {
        proof.name: names[proof.name] for proof in journal.proofs if proof.name in names
    }
    journal.cache_retention_tag = raw.get("retention_tag") if raw else None
    # Exact descriptor metadata detects same-size rewrites; journal writers only
    # append. Signature creation performs no native-content/retained-event reads.
    journal.signature = digest_of(
        {"proofs": [asdict(proof) for proof in journal.proofs], "retention": retention}
    )[:32]
    return journal


def save_journal_cache(
    profile: str,
    journal,
    blobs: dict[str, str],
    record: dict | None,
    *,
    retention_tag: str | None = None,
) -> None:
    from ssgrep.accounting.capture.journal_retention import retained_event

    files = {}
    damage = {}
    for row in journal.damage:
        damage.setdefault(row.segment, []).append(to_plain(row))
    for proof in journal.proofs:
        name = proof.name
        if (
            retention_tag is not None
            and retention_tag == journal.cache_retention_tag
            and name not in journal.cache_dirty
            and name in journal.cache_blobs
        ):
            files[name] = journal.cache_blobs[name]
            continue
        original = journal.segment_events.get(name, ())
        retained = [
            retained_event(event, record) if record is not None else event for event in original
        ]
        scrubbed = any(old != new for old, new in zip(original, retained, strict=True))
        pending = name in journal.retention_pending or scrubbed
        if not scrubbed and name not in journal.cache_dirty and name in journal.cache_blobs:
            files[name] = journal.cache_blobs[name]
            continue
        files[name] = store.write_blob(
            profile,
            "journal-segment",
            {
                "proof": to_plain(proof),
                "events": [to_plain(row) for row in retained if row is not None],
                "damage": damage.get(name, []),
                "retention_pending": pending,
            },
        )
    blobs[JOURNAL_CACHE_BLOB] = store.write_blob(
        profile,
        JOURNAL_CACHE_BLOB,
        {"schema": 1, "segments": files, "retention_tag": retention_tag},
    )


def journal_references(profile: str, filename: str) -> set[str]:
    raw = store.read_blob(profile, filename)
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != 1
        or not isinstance(raw.get("segments"), dict)
    ):
        raise CollectError("state_corrupt", "Journal cache manifest is invalid.")
    return set(raw["segments"].values())


Phase = Callable[..., None]


def read_stage(
    *,
    profile: str,
    options: CollectOptions,
    now: datetime,
    clock: Callable[[], float],
    should_stop: Callable[[], bool],
    env: Mapping[str, str] | None,
    checkpoint: Checkpoint,
    ledger: LazyLedger,
    expired: frozenset[tuple[str, str]],
    policy: ProfilePolicy,
    horizon: str,
    phase: Phase,
    remaining_seconds: float | None = None,
) -> tuple[ReadOutcome, list[RawEvidence], bool]:
    """(outcome, newly accepted evidence, a backfill job just finished).

    Mutates ``checkpoint`` in memory only (sources / backfill job / cumulative drops); nothing
    is durable until the cycle commits it.
    """
    from ssgrep.accounting.collect import steps
    from ssgrep.accounting.collect.collector import (
        backfill_job_for,
        backfill_read,
        parse_stamp,
        pre_baseline,
        require_interval,
    )
    from ssgrep.accounting.collect.replay_evidence import cache_collected_evidence
    from ssgrep.accounting.privacy.policy import policy_status

    if options.recompute:
        phase("reading", sources_total=0, sources_done=0, **steps.source_progress(()))
        return ReadOutcome(), [], False
    discovered = discover(make_readers(env))
    updated = {
        key: SourceRecord(source.runtime, source.source_id)
        for key, (_, source) in discovered.items()
    }
    visited: set[str] = set()
    phase(
        "reading",
        sources_total=len(discovered),
        sources_done=0,
        **steps.source_progress(updated.values()),
    )

    def on_source(key: str, record: SourceRecord) -> None:
        if key not in discovered:
            return  # missing historical sources are gaps, not newly discovered work
        updated[key] = record
        visited.add(key)
        phase("reading", sources_done=len(visited), **steps.source_progress(updated.values()))

    def on_bytes(total: int) -> None:
        phase("reading", bytes_read=total)

    if options.backfill:
        since, until = require_interval(options)
        job = backfill_job_for(checkpoint.backfill_job, since, until)
        outcome, job = backfill_read(
            discovered=discovered,
            job=job,
            ledger=ledger,
            expired_sessions=expired,
            now=now,
            budget=options.budget
            if remaining_seconds is None
            else replace(options.budget, seconds=max(0, int(remaining_seconds))),
            clock=clock,
            should_stop=should_stop,
            policy_horizon=policy,
            on_source=on_source,
            on_bytes=on_bytes,
        )
        checkpoint.backfill_job = job
        cache_collected_evidence(profile, outcome.evidence)
        return outcome, outcome.evidence, job["status"] == "complete"
    known = dict(checkpoint.sources)
    pre_baseline(discovered, known, parse_stamp(checkpoint.baseline_ts) or now)
    rules = AdmitRules(
        paused=policy_status(profile).gaps,
        expired_sessions=expired,
        horizon_day=horizon,
        capture_start=checkpoint.baseline_ts,
    )
    outcome = read_sources(
        discovered=discovered,
        known=known,
        admit=rules,
        ledger=ledger,
        byte_budget=options.budget.bytes,
        time_budget=float(options.budget.seconds)
        if remaining_seconds is None
        else remaining_seconds,
        clock=clock,
        should_stop=should_stop,
        on_source=on_source,
        on_bytes=on_bytes,
    )
    checkpoint.sources = dict(outcome.sources)
    checkpoint.dropped = steps.merge_dropped(checkpoint.dropped, outcome.dropped)
    cache_collected_evidence(profile, outcome.evidence)
    return outcome, outcome.evidence, False
