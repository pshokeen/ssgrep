"""Framework-free READ facade over the latest snapshot, receipts, and history.

Nothing here collects, indexes, enables, sends, or creates storage: every
file access goes through ``plainfile.read_private`` which only opens existing
owner-safe files. Staleness compares the snapshot's own ``collected_at`` with
an injected ``now`` so repeated reads can never reset freshness.

On-disk inputs (all under ``<data>/accounting/<profile>/``):

* ``snapshots/current.json``: ``SnapshotEnvelope`` JSON (see ``service_types``).
* ``receipts/<receipt-id>.json``: one ``Receipt`` (``Quantity`` objects in costs).
* ``history/periods.json``: list of ``{period, revision, status, supersedes, net}``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.ledger.contracts import (
    DEFAULT_LIMIT,
    PERSONAL_PROFILE,
    SCHEMA_VERSION,
    Quantity,
)
from ssgrep.accounting.ledger.ledger_contracts import (
    DEFAULT_COLLECT_SECONDS,
    DEFAULT_STALE_INTERVALS,
    ProfilePolicy,
    SnapshotEnvelope,
    SnapshotState,
)
from ssgrep.accounting.privacy import storage as plainfile
from ssgrep.accounting.privacy.policy import (
    PolicyError,
    PolicyStatus,
    check_profile_id,
    policy_status,
)
from ssgrep.accounting.privacy.storage import MAX_SNAPSHOT_BYTES, load_json
from ssgrep.accounting.serve.status_worker import (
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
from ssgrep.accounting.views.diagnostics import (
    merged_actions,
    read_delivery_lines,
    summary_conditions,
    worst_first,
)
from ssgrep.accounting.views.filter import evaluate
from ssgrep.accounting.views.history import scenario_row
from ssgrep.accounting.views.ledger import load_ledger
from ssgrep.accounting.views.lookups import read_lookups
from ssgrep.accounting.views.records import RECEIPT_ID_RE, read_history, read_receipt
from ssgrep.accounting.views.types import (
    COVERAGE_COUNTERS,
    ESTIMATE_UNITS,
    MEASURED_ORDER,
    Condition,
    DeliveryLine,
    NextAction,
    ScenarioRow,
    Scope,
    SummaryView,
    command_for,
    parse_quantity,
)

if TYPE_CHECKING:
    from ssgrep.accounting.views.diagnostics import DiagnosticGroup

__all__ = ["read_history", "read_receipt", "read_snapshot", "read_summary", "read_lookups"]

STALE_AFTER_SECONDS = DEFAULT_COLLECT_SECONDS * DEFAULT_STALE_INTERVALS


def parse_instant(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _envelope(raw: Any) -> SnapshotEnvelope:
    return SnapshotEnvelope(
        schema_version=raw["schema_version"],
        snapshot_id=str(raw["snapshot_id"]),
        generation=int(raw["generation"]),
        counter_epoch=int(raw["counter_epoch"]),
        methodology_id=str(raw["methodology_id"]),
        policy_version=int(raw["policy_version"]),
        collected_at=str(raw["collected_at"]),
        coverage_through=str(raw["coverage_through"]),
        source_manifest_digest=str(raw["source_manifest_digest"]),
        windows=dict(raw["windows"]),
        coverage=dict(raw.get("coverage", {})),
        observations=dict(raw.get("observations", {})),
        estimates=dict(raw.get("estimates", {})),
        warnings=tuple(str(w) for w in raw.get("warnings", [])),
    )


def read_snapshot(profile: str) -> tuple[SnapshotEnvelope | None, SnapshotState | None, str | None]:
    """(envelope, None, None) when usable; else (None, failure state, reason)."""
    try:
        raw = load_json(profile, "snapshots", "current.json", limit=MAX_SNAPSHOT_BYTES)
    except plainfile.UnsafePathError:
        return None, SnapshotState.CORRUPT, "snapshot_unsafe_path"
    except (ValueError, OSError):
        return None, SnapshotState.CORRUPT, "snapshot_unreadable"
    if raw is None:
        return None, None, None
    if not isinstance(raw, dict):
        return None, SnapshotState.CORRUPT, "snapshot_not_an_object"
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        return None, SnapshotState.CORRUPT, "schema_version_missing"
    if version != SCHEMA_VERSION:
        return None, SnapshotState.UNSUPPORTED_SCHEMA, f"snapshot_schema_{version}"
    try:
        return _envelope(raw), None, None
    except (KeyError, TypeError, ValueError):
        return None, SnapshotState.CORRUPT, "snapshot_fields_invalid"


def _rows(estimates: Any) -> tuple[ScenarioRow, ...]:
    estimates = estimates if isinstance(estimates, dict) else {}
    keys = sorted(set(ESTIMATE_UNITS) | set(estimates) - {"representative_receipt"})
    return tuple(scenario_row(key, estimates.get(key)) for key in keys)


def _measured(observations: Any) -> tuple[tuple[str, Quantity], ...]:
    observations = observations if isinstance(observations, dict) else {}
    units = {"visible_bytes": "bytes"}
    names = [*MEASURED_ORDER, *sorted(set(observations) - set(MEASURED_ORDER))]
    return tuple(
        (n, parse_quantity(observations.get(n), units.get(n, "count"), "not_recorded"))
        for n in names
    )


def _coverage(raw: Any) -> dict[str, int | None]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        name: raw[name]
        if isinstance(raw.get(name), int) and not isinstance(raw[name], bool)
        else None
        for name in COVERAGE_COUNTERS
    }


def _exclusions(raw: Any) -> tuple[tuple[str, int], ...]:
    raw = raw.get("exclusions") if isinstance(raw, dict) else None
    if not isinstance(raw, dict):
        return ()
    return tuple(
        sorted(
            (str(k), v) for k, v in raw.items() if isinstance(v, int) and not isinstance(v, bool)
        )
    )


def _receipt_id(estimates: dict[str, object]) -> str | None:
    candidate = estimates.get("representative_receipt")
    return candidate if isinstance(candidate, str) and RECEIPT_ID_RE.fullmatch(candidate) else None


def _age(collected_at: str, now: datetime) -> tuple[int | None, bool]:
    try:
        collected = parse_instant(collected_at)
    except ValueError:
        return None, False
    elapsed = now - collected
    return max(int(elapsed.total_seconds()), 0), elapsed > timedelta(seconds=STALE_AFTER_SECONDS)


def _skeleton(profile: str, scope: Scope, status: PolicyStatus) -> SummaryView:
    policy = status.profile
    return SummaryView(
        profile=profile,
        state=SnapshotState.NO_SNAPSHOT,
        paused=policy.paused,
        paused_at=policy.paused_at,
        stale=False,
        scope=scope,
        collected_at=None,
        age_seconds=None,
        collection_enabled=policy.collection_enabled,
        exports=policy.enabled_exports,
    )


def _actions(
    profile: str, view_state: SnapshotState, stale: bool, receipt: str | None
) -> tuple[NextAction, ...]:
    def act(text: str, why: str) -> tuple[NextAction, ...]:
        return (NextAction(command_for(profile, text), why),)

    if view_state is SnapshotState.DISABLED:
        return act("savings setup", "Measurement is off; setup previews what would be kept.")
    if view_state is SnapshotState.PAUSED:
        return act("savings resume", "Capture, collection and sending are paused.")
    if view_state is SnapshotState.NO_SNAPSHOT:
        return act("savings collect", "Capture is enabled but no collection has completed.")
    if stale:
        return act("savings collect", "The last good snapshot is stale.")
    if view_state is SnapshotState.FRESH and receipt:
        return act(f"savings explain {receipt}", "Explain one real receipt behind these numbers.")
    return act("savings status", "Show collection, coverage and delivery state.")


def _finish(view: SummaryView, status: PolicyStatus, now: datetime) -> SummaryView:
    """Attach delivery lines and named conditions; actions gain each condition's command."""
    delivery = read_delivery_lines(view.profile, status.profile, now)
    conditions = summary_conditions(view, delivery, status.sending_paused_reason)
    actions = merged_actions(view.next_actions, conditions)
    return replace(view, delivery=delivery, conditions=conditions, next_actions=actions)


def _retention(coverage: dict[str, Any]) -> tuple[str | None, int | None]:
    block = coverage.get("retention")
    if not isinstance(block, dict):
        return None, None
    before = block.get("evidence_expired_before")
    frozen = block.get("frozen_roots")
    return (
        before if isinstance(before, str) else None,
        frozen if isinstance(frozen, int) and not isinstance(frozen, bool) else None,
    )


def _filtered(
    profile: str,
    scope: Scope,
    envelope: SnapshotEnvelope,
    limit: int,
    cursor: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Values from the committed ledger for filters/groupings, or why they are unavailable."""
    data, why = load_ledger(profile)
    if data is None:
        return {}, [f"filters_unavailable: the ledger cannot be read ({why})"]
    if data.generation != envelope.generation:
        return {}, [
            "filters_unavailable: the ledger is not at the snapshot's generation "
            f"({data.generation} vs {envelope.generation}); run savings collect"
        ]
    outcome = evaluate(
        profile, scope, data, envelope.snapshot_id, envelope.collected_at, limit, cursor
    )
    details: dict[str, Any] = {
        "measured": outcome.measured,
        "scenarios": outcome.scenarios,
        "groups": outcome.groups,
        "group_total": outcome.group_total,
        "next_cursor": outcome.next_cursor,
        "filtered": True,
        "estimates_reason": outcome.estimates_reason,
        "coverage_override": outcome.coverage,
        "receipt_id": outcome.receipt_id,
    }
    return details, list(outcome.notes)


def read_summary(
    profile: str = PERSONAL_PROFILE,
    scope: Scope | None = None,
    now: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
    cursor: str | None = None,
) -> SummaryView:
    """Honest summary of the latest snapshot; never collects or writes.

    `--group-by` selection reads the committed collector ledger (read-only) and binds the page token
    to the snapshot id, generation and normalized selection.
    """
    check_profile_id(profile)
    scope = scope or Scope()
    now = now or datetime.now(UTC)
    status = policy_status(profile)
    skeleton = _skeleton(profile, scope, status)

    def result(state: SnapshotState, stale: bool = False, **extra: Any) -> SummaryView:
        if cursor is not None and "groups" not in extra:
            raise PolicyError(
                "invalid_cursor",
                "This view has no pages for that token (no snapshot, no group-by selection, or the "
                "data changed). Start again without paging.",
                command_for(profile, "savings"),
                2,
            )
        view = replace(skeleton, state=state, stale=stale, **extra)
        if "next_actions" not in extra:
            view = replace(view, next_actions=_actions(profile, state, stale, view.receipt_id))
        return _finish(view, status, now)

    if status.error is not None:
        return result(
            SnapshotState.CORRUPT, reason=status.error.condition, warnings=(str(status.error),)
        )
    envelope, failure, why = read_snapshot(profile)
    if failure is not None:
        return result(failure, reason=why, next_actions=_actions(profile, failure, False, None))
    if envelope is None:
        if status.profile.paused:
            return result(SnapshotState.PAUSED, reason="paused_without_snapshot")
        state = (
            SnapshotState.NO_SNAPSHOT
            if status.profile.collection_enabled
            else SnapshotState.DISABLED
        )
        return result(
            state,
            reason="measurement_off" if state is SnapshotState.DISABLED else "no_snapshot_yet",
        )
    age, stale = _age(envelope.collected_at, now)
    state = (
        SnapshotState.PAUSED
        if status.profile.paused
        else (SnapshotState.STALE if stale else SnapshotState.FRESH)
    )
    window = envelope.windows.get(scope.window)
    warnings = list(envelope.warnings)
    if not status.profile.collection_enabled:
        warnings.append("measurement_off: showing retained history only")
    details: dict[str, Any] = {}
    if not scope.is_default:
        details, notes = _filtered(profile, scope, envelope, limit, cursor)
        warnings.extend(notes)
    elif isinstance(window, dict):
        details = {
            "measured": _measured(window.get("observations", envelope.observations)),
            "scenarios": _rows(window.get("estimates", envelope.estimates)),
        }
    else:
        warnings.append(f"window_missing: {scope.window}")
    window_coverage = window.get("coverage") if isinstance(window, dict) else None
    coverage: dict[str, Any] = {
        **envelope.coverage,
        **(window_coverage if isinstance(window_coverage, dict) else {}),
    }
    for code in ("replay_unavailable_sessions", "frequency_revision_unavailable"):
        count = coverage.get(code)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            warnings.append(
                f"{code}: {count} retained components cannot be replayed under their frozen "
                "evidence/method. Inspect savings status --json; "
                "recompute only from approved sources."
            )
    index = coverage.get("index") if isinstance(coverage.get("index"), str) else "unknown"
    has_values = bool(details)
    receipt = details.pop("receipt_id", None) if details else None
    if receipt is None and scope.is_default:
        receipt = _receipt_id(envelope.estimates)
    snapshot_coverage = _coverage(coverage)
    override = details.pop("coverage_override", None)
    if override:
        snapshot_coverage = {**snapshot_coverage, **override}
    expired_before, frozen = _retention(coverage)
    return result(
        state,
        stale=stale,
        collected_at=envelope.collected_at,
        age_seconds=age,
        coverage=snapshot_coverage,
        exclusions=_exclusions(coverage),
        partial=bool(coverage.get("partial", False)),
        index=index,
        receipt_id=receipt if has_values else None,
        snapshot_id=envelope.snapshot_id,
        generation=envelope.generation,
        methodology_id=envelope.methodology_id,
        policy_version=envelope.policy_version,
        coverage_through=envelope.coverage_through,
        warnings=tuple(warnings),
        evidence_expired_before=expired_before,
        frozen_roots=frozen,
        limit=limit,
        **details,
    )


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
        from ssgrep.accounting.views.diagnostics import grouped_warnings

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
                command_for(profile, "savings setup --policy <policy-file>"), "Correct the policy."
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
