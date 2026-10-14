"""Helper steps of a collection cycle (recovery, schema guard, checkpoint, counters, inputs).

Kept separate from ``collector_cycle`` so the cycle reads as the protocol it is: recover,
read, reduce, value, expire, publish, commit.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta

from ssgrep.accounting import collector_store as store, plainfile, snapshots, storage
from ssgrep.accounting.collector_assemble import utc_stamp
from ssgrep.accounting.collector_modes import parse_stamp
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.collector_store import Checkpoint, SourceRecord
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import SCHEMA_VERSION, RawEvidence, digest_of
from ssgrep.accounting.counters import (
    LiveCounters,
    empty_counters,
    observe,
    prune,
    start_epoch,
)
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.journal_reader import JournalRead
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.reduce_types import Reduction
from ssgrep.accounting.service_records import MAX_RECEIPT_BYTES
from ssgrep.accounting.snapshots import MAX_SNAPSHOT_BYTES

CAPTURE_GRACE_SECONDS = 300  # a tool call is logged before its journal event is written
KEEP_GENERATIONS = 64


def guard_schema(profile: str) -> None:
    """Refuse to overwrite a snapshot written by a NEWER ssgrep (older binaries stay read-only)."""
    try:
        raw = plainfile.read_private(
            storage.data_root(),
            "accounting",
            profile,
            "snapshots",
            "current.json",
            max_bytes=MAX_SNAPSHOT_BYTES,
        )
        document = None if raw is None else store.loads(raw)
    except (plainfile.UnsafePathError, ValueError, OSError):
        return  # unreadable: the snapshot recovery step reports or repairs it
    version = document.get("schema_version") if isinstance(document, dict) else None
    if isinstance(version, int) and not isinstance(version, bool) and version > SCHEMA_VERSION:
        raise CollectError(
            "unsupported_schema",
            f"Accounting state uses schema {version}; this ssgrep reads schema {SCHEMA_VERSION}. "
            "Nothing was changed; upgrade ssgrep to collect.",
        )


def recover(profile: str, checkpoint: Checkpoint | None) -> None:
    """Make disk match the committed checkpoint before doing anything new (lease holder only)."""
    generation = checkpoint.generation if checkpoint else 0
    try:
        snapshots.discard_uncommitted(profile, generation)
        snapshots.sweep_temporaries(profile)
        if generation:
            snapshots.repair_current(profile, generation)
    except (ValueError, FileNotFoundError) as error:
        raise CollectError(
            "state_corrupt",
            "Snapshot files do not match the collector checkpoint; nothing was changed.",
        ) from error
    store.sweep_orphans(profile, checkpoint)


def capture_start(journal: JournalRead, now: datetime) -> datetime:
    """When capture began: the first journal event (minus a grace), else this moment."""
    stamps = [parse_stamp(event.ts_utc) for event in journal.events]
    first = min((s for s in stamps if s is not None), default=None)
    if first is None:
        return now
    return min(first - timedelta(seconds=CAPTURE_GRACE_SECONDS), now)


def new_checkpoint(profile: str, journal: JournalRead, now: datetime) -> Checkpoint:
    start = capture_start(journal, now)
    return Checkpoint(profile, baseline_ts=utc_stamp(start), since_day=utc_stamp(start)[:10])


def input_digest(policy: ProfilePolicy, frequency: FrequencyView, key_present: bool) -> str:
    """Everything besides evidence/journal that changes a reduction's output."""
    return digest_of(
        [policy.policy_version, list(policy.roots), frequency.version_id, bool(key_present)]
    )


def counters_step(
    previous: LiveCounters | None,
    journal: JournalRead,
    *,
    now: datetime,
    horizon_day: str,
    new_epoch: bool,
    observing: bool,
) -> LiveCounters:
    """Live counters for this cycle (epoch per spec: restart preserves, --new-epoch resets)."""
    stamp = utc_stamp(now)
    if new_epoch:
        counters = start_epoch(previous, journal.events, now)
    else:
        counters = previous or empty_counters(1, stamp)
    if observing:
        counters = observe(counters, journal.events, horizon_day=horizon_day)
        counters = prune(counters, horizon_day)
    return counters


# Drops that are expected housekeeping (usage records are never retained; a copy is a copy):
# counting them as "gaps" would bury the real ones.
EXPECTED_DROPS = frozenset({"kind_not_consumed", "duplicate"})


def merge_dropped(total: Mapping[str, int], new: Mapping[str, int]) -> dict[str, int]:
    """Cumulative count of evidence deliberately not retained, by reason (live reads only)."""
    merged = dict(total)
    for reason, count in new.items():
        if reason not in EXPECTED_DROPS and count > 0:
            merged[reason] = merged.get(reason, 0) + count
    return dict(sorted(merged.items()))


def source_gaps(sources: Mapping[str, SourceRecord]) -> dict[str, int]:
    """Cumulative reader gaps (malformed, oversized, reset, ...) over every known source."""
    total: dict[str, int] = {}
    for record in sources.values():
        for reason, count in record.gaps.items():
            total[reason] = total.get(reason, 0) + count
    return dict(sorted(total.items()))


def prune_receipt_files(profile: str, before_day: str) -> int:
    """Expire dated receipts across namespaces; never remove an unexpired archived receipt."""
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return 0
    removed = 0
    try:
        for name in storage.list_names(directory):
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                data = storage.read_file(directory, name, MAX_RECEIPT_BYTES)
                raw = store.loads(data) if data is not None else None
                period = raw.get("period") if isinstance(raw, dict) else None
                if not isinstance(period, str) or date.fromisoformat(period).isoformat() != period:
                    continue
            except (ValueError, OSError):
                continue  # an unreadable receipt has no trustworthy expiry date
            if period < before_day:
                removed += storage.delete_file(directory, name)
    finally:
        os.close(directory)
    return removed


def lag_bytes(coverage: Mapping[str, object]) -> int:
    lag = coverage.get("lag_bytes")
    return lag if isinstance(lag, int) and not isinstance(lag, bool) else 0


def aggregates_expired_before(checkpoint: Checkpoint) -> str | None:
    value = checkpoint.retention.get("aggregates_expired_before")
    return value if isinstance(value, str) else None


def session_seen(items: Iterable[RawEvidence]) -> dict[tuple[str, str], str]:
    """Latest evidence timestamp per ``(runtime, session)`` among newly accepted records."""
    seen: dict[tuple[str, str], str] = {}
    for item in items:
        if item.ts_utc and item.ts_utc > seen.get((item.runtime, item.session_id), ""):
            seen[(item.runtime, item.session_id)] = item.ts_utc
    return seen


def last_tuple(last: Mapping[tuple[str, str], str]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((runtime, session, ts) for (runtime, session), ts in last.items()))


def session_cwds(reduction: Reduction) -> tuple[tuple[str, str, str], ...]:
    """Working directory per in-scope session: the earliest invocation that recorded one."""
    earliest: dict[tuple[str, str], tuple[str, str, str]] = {}
    for record in reduction.invocations:
        if not record.cwd:
            continue
        key = (record.runtime, record.session_id)
        order = (record.ts_utc or "", record.invocation_id, record.cwd)
        if key not in earliest or order < earliest[key]:
            earliest[key] = order
    return tuple(
        sorted((runtime, session, found[2]) for (runtime, session), found in earliest.items())
    )


def invocation_sessions(reduction: Reduction) -> tuple[tuple[str, str, str], ...]:
    """(invocation id, runtime, session) of every recognized invocation, sorted."""
    return tuple(sorted((r.invocation_id, r.runtime, r.session_id) for r in reduction.invocations))


def reduction_doc(reduction: Reduction, last: Mapping[tuple[str, str], str]) -> ReductionDoc:
    return ReductionDoc(
        executions=reduction.executions,
        coverage=dict(reduction.coverage),
        warnings=tuple(reduction.warnings),
        digest=reduction.digest,
        session_roots=reduction.session_roots,
        session_last=last_tuple(last),
        session_parents=reduction.session_parents,
        session_cwd=session_cwds(reduction),
        invocation_sessions=invocation_sessions(reduction),
    )
