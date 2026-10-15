"""Retention as one cycle step: freeze expired conversations, drop their evidence, age aggregates.

Wraps ``retention_hooks`` for the worker. Only owned accounting data is ever touched: retained
evidence segments, ledger content, and derived aggregates. Nothing here accepts a path from
evidence or touches a source transcript or an index.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.collector_store import SegmentRef
from ssgrep.accounting.collector_types import RetentionCoverage
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.reduce_types import ExecutionRecord
from ssgrep.accounting.retention_hooks import (
    AggregateExpiry,
    coverage_after,
    expire_aggregates,
    plan_expiry,
    split_evidence,
)


def sealing_executions(
    previous: tuple[ExecutionRecord, ...],
    current: tuple[ExecutionRecord, ...],
    *,
    evidence_before: str,
    before_day: str,
    aggregate_before: str | None,
    already: Mapping[str, object],
) -> tuple[ExecutionRecord, ...]:
    """Carry accepted compact usage into sealing before expired inputs disappear.

    Only old, unsealed observations inside the still-retained aggregate range
    qualify. Current rows win identities; no raw replay or live counter update
    occurs. The caller's state is already the post-purge, current-scope state.
    """
    lost = [
        row
        for row in previous
        if row.in_scope
        and row.day < evidence_before
        and row.day < before_day
        and row.day not in already
        and (aggregate_before is None or row.day >= aggregate_before)
    ]
    if not lost:
        return current
    present = {row.execution_id for row in current}
    return (*current, *(row for row in lost if row.execution_id not in present))


_SEP = "|"


def encode_sessions(pairs: Iterable[tuple[str, str]]) -> list[str]:
    return sorted(f"{runtime}{_SEP}{session}" for runtime, session in pairs)


def decode_sessions(values: Iterable[str]) -> frozenset[tuple[str, str]]:
    out = set()
    for value in values:
        runtime, _, session = str(value).partition(_SEP)
        out.add((runtime, session))
    return frozenset(out)


def previous_coverage(retention: Mapping[str, object]) -> RetentionCoverage:
    def day(name: str) -> str | None:
        value = retention.get(name)
        return value if isinstance(value, str) else None

    frozen = retention.get("frozen_roots")
    return RetentionCoverage(
        day("evidence_expired_before"),
        day("aggregates_expired_before"),
        frozen if isinstance(frozen, int) and not isinstance(frozen, bool) else 0,
    )


def retention_record(
    coverage: RetentionCoverage,
    sessions: Iterable[tuple[str, str]],
    *,
    journal: Mapping[str, object] | None = None,
) -> dict:
    """Stage pure expiry coverage and the journal plan in the SAME acceptance commit.

    Physical journal maintenance runs only after this record is durable; it cannot
    be folded into ``apply_expiry`` before checkpoint publication.
    """
    record: dict[str, object] = {
        "evidence_expired_before": coverage.evidence_expired_before,
        "aggregates_expired_before": coverage.aggregates_expired_before,
        "frozen_roots": coverage.frozen_roots,
        "expired_sessions": encode_sessions(sessions),
    }
    if journal is not None:
        record["journal"] = dict(journal)
    return record


def merge_session_last(
    previous: Iterable[tuple[str, str, str]], seen: Mapping[tuple[str, str], str]
) -> dict[tuple[str, str], str]:
    """Latest evidence timestamp per ``(runtime, session)`` (only ever moves forward)."""
    merged = {(runtime, session): ts for runtime, session, ts in previous}
    for key, ts in seen.items():
        if ts > merged.get(key, ""):
            merged[key] = ts
    return merged


def root_last_evidence(
    doc: ReductionDoc, session_last: Mapping[tuple[str, str], str]
) -> dict[str, str]:
    """Latest evidence timestamp of ANY record of each root conversation."""
    latest: dict[str, str] = {}
    for runtime, session, root in doc.session_roots:
        ts = session_last.get((runtime, session))
        if ts is not None and ts > latest.get(root, ""):
            latest[root] = ts
    return latest


@dataclass
class ExpiryOutcome:
    frozen: list[str] = field(default_factory=list)
    sessions: frozenset[tuple[str, str]] = frozenset()  # every expired session (cumulative)
    dropped_records: int = 0
    segments: list[SegmentRef] | None = None  # set when retained segments were rewritten
    coverage: RetentionCoverage = RetentionCoverage()
    changed: bool = False


def _frozen_days(ledger: VersionedLedger) -> list[str]:
    return sorted({leaf.day for frozen in ledger.state().frozen for leaf in frozen.leaves})


def apply_expiry(
    profile: str,
    *,
    now: datetime,
    policy: ProfilePolicy,
    ledger: VersionedLedger,
    doc: ReductionDoc,
    session_last: Mapping[tuple[str, str], str],
    already: frozenset[tuple[str, str]],
    previous: RetentionCoverage,
    segments: list[SegmentRef],
    sealed_days: Mapping[str, object],
    compact: bool = True,
) -> tuple[ExpiryOutcome, dict[str, object]]:
    """Freeze conversations whose evidence all aged out; compact retained evidence; age totals.

    Returns the outcome and the sealed days that survive aggregate retention.
    """
    plan = plan_expiry(
        now=now, policy=policy, root_last_evidence=root_last_evidence(doc, session_last)
    )
    aggregate_floor = max(plan.aggregate_horizon_day, previous.aggregates_expired_before or "")
    if previous.aggregates_expired_before is not None and ledger.has_before_day(
        previous.aggregates_expired_before
    ):
        # Recomputed evidence cannot restore aggregates already removed by retention.
        ledger.purge(before_day=previous.aggregates_expired_before)
    out = ExpiryOutcome(sessions=already, coverage=previous)
    if plan.expire_roots:
        out.frozen = ledger.expire_roots(plan.expire_roots)
        wanted = {
            (runtime, session)
            for runtime, session, root in doc.session_roots
            if root in set(plan.expire_roots)
        }
        fresh = frozenset(wanted - already)
        if fresh:
            out.sessions = already | fresh
            if compact:
                kept, dropped = split_evidence(store.iter_evidence(profile, segments), out.sessions)
                out.segments = store.write_segments(profile, kept)
                out.dropped_records = dropped
            out.changed = True
    history = [{"period": day} for day in _frozen_days(ledger) if day < aggregate_floor]
    kept_rows, kept_days, report = expire_aggregates(history, sealed_days, aggregate_floor)
    del kept_rows
    frozen_before_aging = len(ledger.state().frozen)
    aged = _age_aggregates(ledger, report)
    out.coverage = coverage_after(
        previous,
        plan,
        out.frozen,
        report.aggregates_expired_before if aged else None,
        frozen_before_aging,
    )
    if aged:
        # Freezing is an event even when the same cycle expires its aggregate.
        # The published count describes only the frozen roots still retained.
        out.coverage = replace(out.coverage, frozen_roots=len(ledger.state().frozen))
    out.changed = out.changed or aged or bool(out.frozen)
    return out, dict(kept_days)


def _age_aggregates(ledger: VersionedLedger, report: AggregateExpiry) -> bool:
    """Remove frozen totals older than the aggregate horizon; True when anything went."""
    if report.aggregates_expired_before is None:
        return False
    ledger.purge(before_day=report.before_day)
    return True
