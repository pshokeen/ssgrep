"""Retention hooks as consumed by one collection cycle (pure, deterministic, path-free).

The full retention/purge subsystem is separate; this module only answers, for a cycle:
which conversation roots have fully aged out, what must be left out of the reduction
input, and what the snapshot may truthfully say expired. Rules:

* No clocks: ``now`` is always a parameter (timezone-aware; naive is rejected).
* No filesystem: nothing here accepts, derives, opens, or deletes a path. Source
  transcripts, runtime session stores and search indexes are never reachable from here;
  journal files are NOT deleted by ``expire_journal_events``.
* Whole-root freezing: a root conversation expires only when ALL of its evidence is
  older than the horizon day; one recent record keeps the whole root live.
* Expired evidence is never recomputed: ``apply_expiry`` freezes roots inside the
  ledger (totals kept, receipts marked ``evidence_expired``), and removing aggregates is
  *aggregate expiry* (a coverage statement), never a recomputation.
* Horizons only move forward (``coverage_after``); everything is integers/strings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ssgrep.accounting.collector_types import RetentionCoverage
from ssgrep.accounting.contracts import ExecutionEvent, RawEvidence
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.replay import Ledger

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ExpiryPlan:
    """Which roots age out now (sorted, disjoint) and the horizons used to decide."""

    evidence_horizon_day: str
    aggregate_horizon_day: str
    expire_roots: tuple[str, ...]
    keep_roots: tuple[str, ...]


@dataclass(frozen=True)
class AggregateExpiry:
    """Exactly what ``expire_aggregates`` removed (all lists sorted, no duplicates)."""

    before_day: str
    removed_periods: tuple[str, ...]
    removed_history_rows: int
    removed_sealed_days: tuple[str, ...]

    @property
    def aggregates_expired_before(self) -> str | None:
        """The statement the snapshot may make; ``None`` when nothing was removed."""
        if self.removed_history_rows or self.removed_sealed_days:
            return self.before_day
        return None


def _utc_day(now: datetime, days: int) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if days < 0:
        raise ValueError("retention days must not be negative")
    return (now.astimezone(UTC) - timedelta(days=days)).date().isoformat()


def evidence_horizon_day(now: datetime, policy: ProfilePolicy) -> str:
    """UTC day ``evidence_retention_days`` before ``now``; earlier-dated evidence expires."""
    return _utc_day(now, policy.evidence_retention_days)


def aggregate_horizon_day(now: datetime, policy: ProfilePolicy) -> str:
    """UTC day ``receipt_retention_days`` before ``now``; earlier aggregates expire."""
    return _utc_day(now, policy.receipt_retention_days)


def _ts_day(ts: str) -> str:
    match = _TS.match(ts) if isinstance(ts, str) else None
    if match is None:
        raise ValueError("timestamp must be UTC YYYY-MM-DDTHH:MM:SS[.fff]Z")
    return match.group(1)


def _check_day(day: str) -> str:
    if not isinstance(day, str) or _DAY.match(day) is None:
        raise ValueError("day must be YYYY-MM-DD")
    return day


def plan_expiry(
    *, now: datetime, policy: ProfilePolicy, root_last_evidence: Mapping[str, str]
) -> ExpiryPlan:
    """Roots whose latest evidence of ANY record is dated before the evidence horizon."""
    horizon = evidence_horizon_day(now, policy)
    expire: list[str] = []
    keep: list[str] = []
    for root in sorted(root_last_evidence):
        (expire if _ts_day(root_last_evidence[root]) < horizon else keep).append(root)
    return ExpiryPlan(horizon, aggregate_horizon_day(now, policy), tuple(expire), tuple(keep))


def apply_expiry(ledger: Ledger, plan: ExpiryPlan) -> list[str]:
    """Freeze the planned roots the ledger actually knows; returns the roots frozen."""
    return ledger.expire_roots(plan.expire_roots)


def split_evidence(
    evidence: Iterable[RawEvidence], expired_sessions: AbstractSet[tuple[str, str]]
) -> tuple[list[RawEvidence], int]:
    """Drop records of expired conversations so a new tail is never reduced from a prefix."""
    kept: list[RawEvidence] = []
    dropped = 0
    for record in evidence:
        if (record.runtime, record.session_id) in expired_sessions:
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


def expire_journal_events(
    events: Iterable[ExecutionEvent], horizon_day: str
) -> tuple[list[ExecutionEvent], int]:
    """Keep events dated on or after ``horizon_day``; the journal files are untouched."""
    horizon = _check_day(horizon_day)
    kept: list[ExecutionEvent] = []
    dropped = 0
    for event in events:
        if _ts_day(event.ts_utc) < horizon:
            dropped += 1
        else:
            kept.append(event)
    return kept, dropped


def expire_aggregates(
    history_rows: Sequence[Mapping[str, object]],
    sealed_days: Mapping[str, object],
    before_day: str,
) -> tuple[list[dict], dict, AggregateExpiry]:
    """Drop whole history rows / sealed days dated before ``before_day`` (aggregate expiry).

    This states what aggregate retention removed; it recomputes nothing. Rows keep their
    original order; the report names exactly the periods, row count and sealed days gone.
    """
    horizon = _check_day(before_day)
    kept_rows: list[dict] = []
    removed_periods: set[str] = set()
    removed_rows = 0
    for row in history_rows:
        period = row.get("period")
        if not isinstance(period, str):
            raise ValueError("history row without a period")
        if _check_day(period) < horizon:
            removed_rows += 1
            removed_periods.add(period)
        else:
            kept_rows.append(dict(row))
    kept_days = {day: v for day, v in sealed_days.items() if _check_day(day) >= horizon}
    removed_days = sorted(day for day in sealed_days if day not in kept_days)
    report = AggregateExpiry(
        horizon, tuple(sorted(removed_periods)), removed_rows, tuple(removed_days)
    )
    return kept_rows, kept_days, report


def _later(first: str | None, second: str | None) -> str | None:
    present = [d for d in (first, second) if d is not None]
    return max(present) if present else None


def coverage_after(
    previous: RetentionCoverage,
    plan: ExpiryPlan,
    frozen: Sequence[str],
    aggregates_before: str | None,
    total_frozen: int,
) -> RetentionCoverage:
    """New coverage: horizons only move forward; ``frozen_roots`` is the running total.

    ``evidence_expired_before`` advances to the plan's horizon only when roots were
    actually frozen this cycle (a horizon that removed nothing states nothing).
    """
    if total_frozen < len(set(frozen)):
        raise ValueError("total_frozen cannot be less than the roots frozen now")
    evidence = _later(
        previous.evidence_expired_before, plan.evidence_horizon_day if frozen else None
    )
    aggregates = _later(previous.aggregates_expired_before, aggregates_before)
    return RetentionCoverage(evidence, aggregates, total_frozen)
