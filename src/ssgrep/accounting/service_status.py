"""Read-only operational status for ``metrics status`` (no index, no side effects).

Worker, lag, missing evidence and storage come from ``status_worker`` (lease
metadata, ``progress.json``, snapshot coverage, a bounded file-size walk); every
unknown stays ``None`` with a reason. Delivery state is read from the export module's documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.ledger_contracts import ProfilePolicy, SnapshotState
from ssgrep.accounting.policy import policy_status
from ssgrep.accounting.policy_rules import check_profile_id
from ssgrep.accounting.service import STALE_AFTER_SECONDS, read_snapshot, read_summary
from ssgrep.accounting.service_delivery import read_delivery_lines, worst_first
from ssgrep.accounting.service_states import merged_actions
from ssgrep.accounting.service_types import Condition, DeliveryLine, NextAction, command_for
from ssgrep.accounting.status_worker import (
    LagStatus,
    MissingEvidence,
    StoragePressure,
    WorkerStatus,
    lag_status,
    missing_evidence,
    read_progress,
    storage_pressure,
    worker_status,
)

if TYPE_CHECKING:
    from ssgrep.accounting.diagnostic_types import DiagnosticGroup


@dataclass(frozen=True)
class DestinationStatus:
    destination_id: str
    signals: tuple[str, ...]
    enabled_signals: tuple[str, ...]
    delivery: str
    reason: str
    lines: tuple[DeliveryLine, ...] = ()


@dataclass(frozen=True)
class StatusView:
    profile: str
    state: SnapshotState
    policy: ProfilePolicy
    policy_error: str | None
    sending_paused_reason: str | None
    snapshot_id: str | None
    generation: int | None
    collected_at: str | None
    coverage_through: str | None
    age_seconds: int | None
    stale: bool
    stale_after_seconds: int
    gaps: tuple[tuple[str, str | None], ...]
    destinations: tuple[DestinationStatus, ...]
    worker: WorkerStatus
    lag: LagStatus
    missing: MissingEvidence
    storage: StoragePressure
    warnings: tuple[str, ...]
    next_actions: tuple[NextAction, ...]
    conditions: tuple[Condition, ...] = ()
    delivery_lines: tuple[DeliveryLine, ...] = ()
    collection_age_seconds: int | None = None
    delivery_age_seconds: int | None = None

    @property
    def diagnostic_groups(self) -> tuple[DiagnosticGroup, ...]:
        from ssgrep.accounting.diagnostics import grouped_warnings

        return grouped_warnings(self.warnings)

    @property
    def diagnostic_total(self) -> int:
        return sum(group.count for group in self.diagnostic_groups)

    @property
    def diagnostic_alias_duplicates(self) -> int:
        return len(self.warnings) - self.diagnostic_total


def _destination_state(lines: tuple[DeliveryLine, ...]) -> str:
    return worst_first(lines)[0].state if lines else "no_signals"


def read_status(profile: str = PERSONAL_PROFILE, now: datetime | None = None) -> StatusView:
    check_profile_id(profile)
    status = policy_status(profile)
    summary = read_summary(profile, now=now or datetime.now(UTC))
    envelope, _, snapshot_reason = read_snapshot(profile)
    coverage = None if envelope is None else envelope.coverage
    coverage_reason = None if envelope is not None else snapshot_reason or "no_snapshot_yet"
    progress, progress_problem = read_progress(profile)
    worker = worker_status(profile, progress, progress_problem, has_snapshot=envelope is not None)
    policy = status.profile
    current_time = now or datetime.now(UTC)
    lines = read_delivery_lines(profile, policy, current_time)
    destinations = tuple(
        DestinationStatus(
            destination_id=d.destination_id,
            signals=tuple(s.value for s in d.signals),
            enabled_signals=tuple(
                s for dest, s in policy.enabled_exports if dest == d.destination_id
            ),
            delivery=_destination_state(mine),
            reason=" | ".join(line.cause for line in mine),
            lines=mine,
        )
        for d in policy.destinations
        for mine in [tuple(x for x in lines if x.destination_id == d.destination_id)]
    )
    acked = [x.acked_age_seconds for x in lines if x.acked_age_seconds is not None]
    warnings = list(summary.warnings)
    if status.sending_paused_reason:
        warnings.append(f"sending_paused: {status.sending_paused_reason}")
    actions = summary.next_actions
    if status.sending_paused_reason and not policy.paused:
        actions = (
            NextAction(
                command_for(profile, "metrics setup --policy <file>"), "Correct the policy."
            ),
        )
    return StatusView(
        profile=profile,
        state=summary.state,
        policy=policy,
        policy_error=status.error.condition if status.error else None,
        sending_paused_reason=status.sending_paused_reason,
        snapshot_id=summary.snapshot_id,
        generation=summary.generation,
        collected_at=summary.collected_at,
        coverage_through=summary.coverage_through,
        age_seconds=summary.age_seconds,
        stale=summary.stale,
        stale_after_seconds=STALE_AFTER_SECONDS,
        gaps=status.gaps,
        destinations=destinations,
        worker=worker,
        lag=lag_status(coverage, coverage_reason, progress),
        missing=missing_evidence(coverage, coverage_reason),
        storage=storage_pressure(profile),
        warnings=tuple(warnings),
        next_actions=merged_actions(actions, summary.conditions),
        conditions=summary.conditions,
        delivery_lines=lines,
        collection_age_seconds=summary.age_seconds,
        delivery_age_seconds=min(acked) if acked else None,
    )
