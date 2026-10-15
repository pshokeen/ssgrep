"""Content-free local lookup verdicts; evidence dimensions are independent."""

from __future__ import annotations

from dataclasses import dataclass, field

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.ledger_contracts import ScenarioEstimate
from ssgrep.accounting.service_types import Scope


@dataclass(frozen=True)
class LookupRow:
    lookup_id: str
    invocation_id: str | None
    execution_ids: tuple[str, ...]
    ambiguous_execution_ids: tuple[str, ...]
    runtime: str | None
    session_id: str | None
    ts_utc: str | None
    task: str
    actor: str
    actor_basis: str
    scope_basis: str
    fingerprint_key_id: str | None
    join_rejection_reasons: tuple[str, ...]
    in_scope: bool
    operations: tuple[str, ...]
    execution: str
    exposure: str
    attribution: str
    reuse: str
    reuse_evaluated: bool
    eligibility: str
    eligibility_reasons: tuple[str, ...]
    outcome: str
    useful_outcome: str
    purpose: str
    measured_service_ms: Quantity
    visible_bytes: Quantity
    timing_known: bool
    known_costs: tuple[tuple[str, Quantity], ...] = ()
    missing_evidence: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    scenarios: tuple[ScenarioEstimate, ...] = ()
    frequency_view_id: str = ""
    frequency_provenance: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LookupsView:
    profile: str
    scope: Scope
    rows: tuple[LookupRow, ...] = ()
    total: int = 0
    next_cursor: str | None = None
    snapshot_id: str | None = None
    generation: int | None = None
    limit: int = 20
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    coverage: dict[str, int | bool] = field(default_factory=dict)
