"""Typed views returned by the read facade (``service.py``) and their JSON form.

Snapshot payload contract read by the facade (written by the collector):

* ``windows[<w>]["observations"]``: ``{name: Quantity}`` for ``searches``,
  ``shows``, ``failed``, ``visible_bytes`` (extra names are shown after).
* ``windows[<w>]["estimates"]``: ``{unit_key: {scenario: {credit, debit, net}}}``
  with ``unit_key`` in ``context_tokens`` / ``work_seconds`` and each leaf a
  ``Quantity`` object ``{value, unit, basis, method, reason}``.
* ``coverage``: integer counters (``recorded_lookups``, ``attributed_lookups``,
  ``candidate_receipts``, ``source_costs_available``, ``ambiguous_joins``,
  ``unavailable_sources``), ``index`` (``available``/``unavailable``),
  ``exclusions`` (reason -> count), ``partial`` (bool).
* ``estimates["representative_receipt"]``: receipt id that ``savings explain``
  can resolve, or null.

Anything missing or malformed becomes an unavailable quantity with a reason,
never zero. Floats are rejected so no JSON consumer sees locale/NaN hazards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.contracts import PERSONAL_PROFILE, Basis, Quantity, Scenario
from ssgrep.accounting.ledger_contracts import ScenarioEstimate, SnapshotState

if TYPE_CHECKING:
    from ssgrep.accounting.diagnostic_types import DiagnosticGroup

MEASURED_ORDER = ("searches", "shows", "failed", "visible_bytes")
ESTIMATE_UNITS = ("context_tokens", "work_seconds")
COVERAGE_COUNTERS = (
    "recorded_lookups",
    "attributed_lookups",
    "candidate_receipts",
    "source_costs_available",
    "ambiguous_joins",
    "unavailable_sources",
    "replay_unavailable_sessions",
    "frequency_revision_unavailable",
)


def command_for(profile: str, text: str) -> str:
    """``ssgrep <text>`` with the selected profile preserved (omitted for the default)."""
    suffix = "" if profile == PERSONAL_PROFILE else f" --profile {profile}"
    return f"ssgrep {text}{suffix}"


def parse_quantity(raw: Any, unit: str, missing_reason: str) -> Quantity:
    """A ``Quantity`` from snapshot JSON; anything unusable is unavailable."""
    if raw is None:
        return Quantity.unavailable(unit, missing_reason)
    try:
        value = raw["value"]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError("value must be an integer")
        basis = Basis(raw["basis"])
        return Quantity(
            value=value,
            unit=str(raw.get("unit") or unit),
            basis=basis,
            method=str(raw.get("method", "")),
            reason=str(raw.get("reason", "")),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return Quantity.unavailable(unit, "malformed_quantity")


def quantity_json(quantity: Quantity) -> dict[str, object]:
    return {
        "value": quantity.value,
        "unit": quantity.unit,
        "basis": quantity.basis.value,
        "method": quantity.method,
        "reason": quantity.reason or None,
    }


@dataclass(frozen=True)
class ScenarioRow:
    """One unit (e.g. context tokens) across the three scenarios."""

    unit_key: str
    credit: tuple[tuple[Scenario, Quantity], ...]
    debit: tuple[tuple[Scenario, Quantity], ...]
    net: tuple[tuple[Scenario, Quantity], ...]
    estimates: tuple[ScenarioEstimate, ...] = ()


@dataclass(frozen=True)
class NextAction:
    command: str
    why: str


@dataclass(frozen=True)
class Condition:
    """One named state of a view: its cause and the real command that addresses it."""

    code: str
    cause: str
    action: NextAction | None = None


@dataclass(frozen=True)
class GroupRow:
    """One ``--by`` row: measured counts and per-unit scenario estimates of one key."""

    key: str
    measured: tuple[tuple[str, Quantity], ...]
    scenarios: tuple[ScenarioRow, ...]
    receipts: int = 0


@dataclass(frozen=True)
class DeliveryLine:
    """Delivery state of one destination/signal (collector acceptance, never 'synced')."""

    destination_id: str
    signal: str
    enabled: bool
    state: str  # disabled|waiting|pending|offline|partial_export|credentials_expired|blocked|
    # unknown|acknowledged|unreadable
    cause: str
    pending_batches: int = 0
    rejected_points: int = 0
    unknown_batches: int = 0
    partial_batches: int = 0
    lost_records: int = 0
    collected_generation: int | None = None
    attempted_generation: int | None = None
    acked_generation: int | None = None
    last_acked_at: str | None = None
    last_attempt_at: str | None = None
    next_retry_utc: str | None = None
    last_error_category: str | None = None
    blocked_reason: str | None = None
    acked_age_seconds: int | None = None
    attempt_age_seconds: int | None = None
    action: NextAction | None = None


@dataclass(frozen=True)
class Scope:
    window: str = "7d"
    since: str | None = None
    until: str | None = None
    project: str | None = None
    runtime: str | None = None
    session: str | None = None
    task: str | None = None
    by: str | None = None
    include_children: bool = False

    @property
    def is_default(self) -> bool:
        return not (
            self.since
            or self.until
            or self.project
            or self.runtime
            or self.session
            or self.task
            or self.by
        )


@dataclass(frozen=True)
class SummaryView:
    profile: str
    state: SnapshotState
    paused: bool
    paused_at: str | None
    stale: bool
    scope: Scope
    collected_at: str | None
    age_seconds: int | None
    collection_enabled: bool
    exports: tuple[tuple[str, str], ...]
    measured: tuple[tuple[str, Quantity], ...] = ()
    scenarios: tuple[ScenarioRow, ...] = ()
    coverage: dict[str, int | None] = field(default_factory=dict)
    exclusions: tuple[tuple[str, int], ...] = ()
    partial: bool = False
    index: str = "unknown"
    receipt_id: str | None = None
    snapshot_id: str | None = None
    generation: int | None = None
    methodology_id: str | None = None
    policy_version: int | None = None
    coverage_through: str | None = None
    warnings: tuple[str, ...] = ()
    next_actions: tuple[NextAction, ...] = ()
    reason: str | None = None
    conditions: tuple[Condition, ...] = ()
    delivery: tuple[DeliveryLine, ...] = ()
    groups: tuple[GroupRow, ...] = ()
    group_total: int | None = None
    next_cursor: str | None = None
    filtered: bool = False
    estimates_reason: str | None = None
    frozen_roots: int | None = None
    evidence_expired_before: str | None = None
    limit: int = 20

    @property
    def diagnostic_groups(self) -> tuple[DiagnosticGroup, ...]:
        """Stable, severity-ordered groups; no I/O or hidden truncation."""
        from ssgrep.accounting.diagnostics import grouped_warnings

        return grouped_warnings(self.warnings)

    @property
    def diagnostic_total(self) -> int:
        return sum(group.count for group in self.diagnostic_groups)

    @property
    def diagnostic_alias_duplicates(self) -> int:
        return len(self.warnings) - self.diagnostic_total

    @property
    def replay_unavailable(self) -> bool:
        """Positive snapshot-wide missingness; unknown counts never become zero."""
        return any(
            value is not None and value > 0
            for name in ("replay_unavailable_sessions", "frequency_revision_unavailable")
            for value in (self.coverage.get(name),)
        )

    @property
    def has_values(self) -> bool:
        return bool(self.measured or self.scenarios)

    @property
    def has_negative(self) -> bool:
        return any(
            q.value is not None and q.value < 0 for row in self.scenarios for _, q in row.net
        )

    @property
    def no_reuse(self) -> bool:
        return self.has_values and self.coverage.get("candidate_receipts") == 0


@dataclass(frozen=True)
class ReceiptView:
    receipt_id: str
    revision: int
    root: str
    task_turn: str | None
    candidate_type: str
    atoms: tuple[str, ...]
    allocations: tuple[tuple[str, tuple[str, ...]], ...]
    exclusions: tuple[str, ...]
    costs: tuple[tuple[str, Quantity], ...]
    methodology: str
    evidence_available: bool
    execution_ids: tuple[str, ...]
    content: str | None = None
    note: str | None = None
    schema: int = 1
    unit: str | None = None
    period: str | None = None
    missing_costs: tuple[tuple[str, str], ...] = ()
    credit_raw: tuple[tuple[str, int], ...] = ()
    identities: tuple[tuple[str, str], ...] = ()
    atom_digests: tuple[tuple[str, str], ...] = ()
    retention_boundary: str | None = None
    reuse: tuple[tuple[str, str], ...] | None = None
    eligibility: str = "unavailable"
    eligibility_reasons: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    scenarios: tuple[ScenarioEstimate, ...] = ()
    exposure_ids: tuple[str, ...] = ()
    raw_unit: str = ""
    debit_raw: int | None = None
    debit_raw_min: int | None = None
    debit_raw_max: int | None = None


@dataclass(frozen=True)
class HistoryEntry:
    period: str
    revision: int
    status: str
    current: bool
    supersedes: int | None
    net: Quantity
    recomputable: bool = True
    kind: str = "original"
    scenarios: tuple[ScenarioEstimate, ...] = ()
    namespace: str = ""
    revision_name: str = ""


@dataclass(frozen=True)
class HistoryView:
    profile: str
    entries: tuple[HistoryEntry, ...]
    current_total: Quantity
    next_cursor: str | None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    next_actions: tuple[NextAction, ...] = ()
    since: str | None = None
    until: str | None = None
    limit: int = 20
    scenarios: tuple[ScenarioEstimate, ...] = ()
