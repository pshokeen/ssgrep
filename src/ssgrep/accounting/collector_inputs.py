"""Inputs of one cycle: admission rules, the lazy dedupe ledger, journal signature, pinned views.

Admission is where "resume never silently backfills" is enforced: evidence dated inside a
pause interval, evidence of an expired conversation, and evidence older than the retention
horizon are dropped at READ time (counted, never retained), so only an explicit interval-bearing
backfill can import them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from ssgrep.accounting import collector_store as store, fingerprint
from ssgrep.accounting.collector_store import Checkpoint
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.journal_reader import JournalRead
from ssgrep.accounting.readers.base import EvidenceLedger

Admit = Callable[[RawEvidence], str | None]
# records that carry identity/lineage only, never prompt or tool text: exempt from the
# capture-start and retention-horizon filters so conversation structure stays resolvable
NO_TEXT_KINDS = frozenset({"session_header", "lineage"})


class ClassifyingLedger(EvidenceLedger):
    """An ``EvidenceLedger`` that says WHY it rejected a record, independent of its history.

    ``EvidenceLedger.conflict_count`` only moves the first time an identity conflicts in THIS
    process, and the rejected variants are never retained, so a restart forgets them and the same
    copy is "newly conflicting" again. The reason below depends only on the admitted copy: the
    digest differs (``conflict``) or equals it (``duplicate``), on every schedule and restart.
    """

    def rejection(self, evidence: RawEvidence) -> str:
        first = self._first.get(self._key(evidence))
        return "conflict" if first is not None and first != evidence.digest else "duplicate"


class LazyLedger:
    """Seed only the native identity being admitted from compact indexed metadata.

    No native content is hydrated for dedupe. Each identity is seeded once in this
    process; immutable shard reads are shared through the segment indexes.
    """

    def __init__(self, profile: str, checkpoint: Checkpoint | None) -> None:
        self._profile = profile
        self._segments = list(checkpoint.segments) if checkpoint else []
        self._ledger = ClassifyingLedger()
        self._seeded: set[tuple[str, str, str]] = set()
        self._index_cache: dict = {}

    def _ready(self, evidence: RawEvidence) -> ClassifyingLedger:
        from ssgrep.accounting.identity_claims import target_key

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
    """The retained evidence of a cycle, streamed from disk at most once and then reused."""

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
        from ssgrep.accounting.evidence_replay import EvidenceReplay

        self._replay = EvidenceReplay(profile, env=env)

    def items(self) -> list[RawEvidence]:
        if self._items is None:
            self._items = self._replay.hydrate_many(
                store.iter_evidence(
                    self._profile,
                    self._segments,
                    sessions=self._sessions,
                )
            )
        return self._items

    @property
    def unavailable(self) -> dict[tuple[str, str], str]:
        """Unavailable sessions must keep accepted rows, never be rebuilt from partial text."""
        self.items()
        return dict(self._replay.unavailable)

    @property
    def replay_available(self) -> bool:
        return not self.unavailable


def _seconds(ts: str) -> str:
    return ts[:19]


@dataclass(frozen=True)
class AdmitRules:
    """What this cycle refuses to retain (reason -> predicate)."""

    paused: tuple[tuple[str, str | None], ...] = ()
    expired_sessions: frozenset[tuple[str, str]] = frozenset()
    horizon_day: str | None = None  # evidence dated before this UTC day is not retained
    capture_start: str | None = None  # live: tool evidence before capture began is history
    window: tuple[str, str] | None = None  # explicit backfill interval [since, until)
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
            return None  # structure only: needed to identify tasks, carries no prompt text
        if ts is None:
            return "outside_interval"
        since, until = self.window
        return None if since <= _seconds(ts)[:19] + "Z" < until else "outside_interval"


def journal_signature(journal: JournalRead) -> str:
    """Stable digest of which complete events and damage the journal holds right now."""
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
    """The request-fingerprint key (never minted by a reader)."""
    try:
        return fingerprint.load_key(profile, create=False)
    except OSError:
        return None
