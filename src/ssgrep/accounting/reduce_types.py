"""Typed result of the combined reducer (design D4-D6; pure data, no I/O, no floats).

One :class:`Reduction` is a deterministic function of (evidence, journal events,
policy, frequency view): the same inputs in any order give the same
:attr:`Reduction.digest`. Executions, invocations, exposure, reuse and attribution
are separate dimensions; coverage names its denominator for every claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ssgrep.accounting.contracts import (
    Attribution,
    Basis,
    Exposure,
    InvocationClass,
    Outcome,
    Quantity,
    Reuse,
    TaskId,
    digest_of,
)
from ssgrep.accounting.frequency_assignment import FrequencyAssignment
from ssgrep.accounting.valuation_inputs import Candidate, EvidenceBundle

VERIFIER_ACTOR = "verifier"
ASSOCIATION_NOTE = "association_not_causation"

# claim name -> the denominator it is a fraction of (every completeness claim names one)
DENOMINATORS: dict[str, str] = {
    "explicit": "recorded_executions",
    "inferred_unique": "recorded_executions",
    "ambiguous": "recorded_executions",
    "unattributed": "recorded_executions",
    "attributed_lookups": "recorded_lookups",
    "inv_matched": "recognized_invocations",
    "inv_transcript_only": "recognized_invocations",
    "inv_journal_gap": "recognized_invocations",
    "source_costs_available": "candidate_atoms",
    "inv_mention": "classified_tool_calls",
    "inv_unclassified": "classified_tool_calls",
    "tasks_resolved": "recorded_executions",
    "tasks_unknown": "recognized_invocations",
}


@dataclass(frozen=True)
class ExecutionRecord:
    """One recorded journal execution after reconciliation (windowable by ``ts_utc``)."""

    execution_id: str
    ts_utc: str
    day: str
    operation: str
    transport: str
    outcome: Outcome
    outcome_basis: str  # journal | result_override | journal_incomplete
    attribution: Attribution
    source: str  # journal (no transcript join) | both
    runtime: str | None
    session_id: str | None
    invocation_id: str | None
    actor: str
    task: TaskId | None
    rendered_bytes: Quantity
    visible_bytes: Quantity
    exposure: Exposure
    reuse: Reuse
    timing_known: bool
    in_scope: bool
    cwd: str | None = None
    correlation_id: str | None = None
    truncated: bool | None = None
    measured_service_ms: Quantity = field(
        default_factory=lambda: Quantity.unavailable("work_ms", "journal_timing_unavailable")
    )
    scope_basis: str = "unknown_scope"
    join_rejection_reasons: tuple[str, ...] = ()
    actor_basis: str = "unknown"
    fingerprint_key_id: str | None = None
    measured_service_ns: Quantity = field(
        default_factory=lambda: Quantity.unavailable(
            "work_ns", "journal_nanosecond_precision_unavailable"
        )
    )


@dataclass(frozen=True)
class InvocationRecord:
    """One recognized transcript invocation (a shell line may own zero+ executions)."""

    invocation_id: str
    runtime: str
    session_id: str
    call_id: str
    args_hash: str
    ts_utc: str | None
    operations: tuple[str, ...]
    route: str
    cardinality: str
    cls: InvocationClass
    attribution: Attribution
    execution_ids: tuple[str, ...]
    ambiguous_execution_ids: tuple[str, ...]
    outcome: Outcome
    exposure: Exposure
    visible_bytes: Quantity
    task: TaskId | None
    actor: str
    reason: str = ""
    purpose: str = "unknown"
    pre_api_failure: bool = False
    timing_known: bool = True
    in_scope: bool = True
    cwd: str | None = None
    measured_service_ms: Quantity = field(
        default_factory=lambda: Quantity.unavailable("work_ms", "journal_timing_unavailable")
    )
    frequency_view_id: str = ""
    frequency_provenance: tuple[tuple[str, str], ...] = ()
    reuse: Reuse = Reuse.NONE_OBSERVED
    reuse_evaluated: bool = False
    scope_basis: str = "unknown_scope"
    join_rejection_reasons: tuple[str, ...] = ()
    actor_basis: str = "unknown"
    fingerprint_key_id: str | None = None
    measured_service_ns: Quantity = field(
        default_factory=lambda: Quantity.unavailable(
            "work_ns", "journal_nanosecond_precision_unavailable"
        )
    )


@dataclass(frozen=True)
class CandidateRecord:
    """A candidate plus the reducer's own judgments (never summed into valuation)."""

    candidate: Candidate
    invocation_id: str
    candidate_type: str
    novelty_confidence: str  # high | reduced
    novelty_reasons: tuple[str, ...]
    rarity_milli: int | None  # exact milli-bits or a useful lower bound; None stays unknown
    association: str = ASSOCIATION_NOTE
    outcome: str = "none"  # none | inspected | action | validated | contradicted
    frequency_view_id: str = ""
    frequency_provenance: tuple[tuple[str, str], ...] = ()
    rarity_evidence_kind: str = ""

    @property
    def eligibility(self) -> str:
        return self.candidate.eligibility.value


@dataclass(frozen=True)
class Exclusion:
    """Why a candidate or source contribution earns nothing (kept for receipts)."""

    candidate_id: str
    invocation_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class Reduction:
    profile: str
    executions: tuple[ExecutionRecord, ...] = ()
    invocations: tuple[InvocationRecord, ...] = ()
    candidates: tuple[CandidateRecord, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    bundles: tuple[EvidenceBundle, ...] = ()
    bundle_roots: tuple[str, ...] = ()  # parallel to ``bundles`` (one per root conversation)
    session_roots: tuple[tuple[str, str, str], ...] = ()  # (runtime, session_id, root)
    session_parents: tuple[
        tuple[str, str, str | None], ...
    ] = ()  # (runtime, session_id, parent id)
    coverage: dict[str, int] = field(default_factory=dict)
    observations: dict[str, Quantity] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    frequency_version: str = ""
    frequency_assignments: tuple[FrequencyAssignment, ...] = ()

    @property
    def digest(self) -> str:
        """Content digest; identical for any ordering of the same inputs."""
        return digest_of(self)


def _count(name: str, value: int) -> Quantity:
    return Quantity(value, "count", Basis.OBSERVED, name)


def summarize(executions: Iterable[ExecutionRecord]) -> dict[str, Quantity]:
    """Integer observation counters over any subset (for example one event window).

    ``visible_bytes``/``rendered_bytes`` are unavailable unless EVERY counted
    execution has the value; the known subtotal is kept in ``*_known``.
    """
    rows = list(executions)
    out: dict[str, int] = {"searches": 0, "shows": 0, "notes": 0, "failed": 0}
    for row in rows:
        key = {"search": "searches", "show": "shows", "note": "notes"}.get(row.operation)
        if key:
            out[key] += 1
        if row.outcome is Outcome.ERROR:
            out["failed"] += 1
        out[f"outcome.{row.outcome.value}"] = out.get(f"outcome.{row.outcome.value}", 0) + 1
        runtime = f"runtime.{row.runtime or 'unknown'}"
        out[runtime] = out.get(runtime, 0) + 1
    result = {name: _count(name, value) for name, value in sorted(out.items())}
    for label, getter in (
        ("visible_bytes", lambda r: r.visible_bytes),
        ("rendered_bytes", lambda r: r.rendered_bytes),
    ):
        values = [getter(r) for r in rows]
        known = [q.value for q in values if q.value is not None]
        result[f"{label}_known"] = Quantity(sum(known), "bytes", Basis.OBSERVED, label)
        if len(known) == len(values):
            result[label] = Quantity(sum(known), "bytes", Basis.OBSERVED, label)
        else:
            result[label] = Quantity.unavailable("bytes", f"{label}_missing_for_some_executions")
    return result
