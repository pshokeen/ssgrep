"""Read every accepted lookup, including no-receipt calls, without collection or writes.

Native invocations own their matched executions. Orphan journal lookups remain
separate records, never aliases. Pages are bound to one published snapshot and
committed evidence selection. Unavailable historical generations are explicitly
expired; no cursor silently starts reading a newer generation.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ssgrep.accounting.ledger.contracts import (
    PERSONAL_PROFILE,
    Basis,
    ExecutionEvent,
    Exposure,
    InvocationClass,
    Phase,
    Quantity,
    Reuse,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_contracts import ScenarioEstimate
from ssgrep.accounting.ledger.reduce_join import Request
from ssgrep.accounting.ledger.units import (
    CAPPED_GAP_V1,
    GAP_CAP_NS,
    MONOTONIC_V1,
    WORK_NS,
    WORK_SECONDS,
    measured_duration,
    ns_seconds_quantity,
    service_milliseconds,
)
from ssgrep.accounting.ledger.valuation import rollup, value_evidence
from ssgrep.accounting.ledger.valuation_inputs import (
    CURRENT_VALUATION_VERSION,
    Eligibility,
    EvidenceBundle,
)
from ssgrep.accounting.privacy.policy import PolicyError, check_profile_id
from ssgrep.accounting.views.ledger import load_lookup_data
from ssgrep.accounting.views.records import RECEIPT_ID_RE
from ssgrep.accounting.views.selection import lookup_records, task_key
from ssgrep.accounting.views.types import Scope, check_limit, command_for, issue, redeem


def read_lookups(
    profile: str = PERSONAL_PROFILE,
    scope: Scope | None = None,
    *,
    snapshot_id: str | None = None,
    generation: int | None = None,
    limit: int = 20,
    cursor: str | None = None,
) -> LookupsView:
    from ssgrep.accounting.views.service import read_snapshot

    check_profile_id(profile)
    check_limit(limit)
    scope = scope or Scope()
    command = command_for(profile, "savings lookups")
    if scope.by is not None:
        raise PolicyError("invalid_filter", "Lookup pages do not support --group-by.", command, 2)
    token = reference(cursor, command)
    if token is not None:
        if (
            snapshot_id is not None
            and snapshot_id != token["snapshot"]
            or generation is not None
            and generation != token["generation"]
        ):
            raise PolicyError(
                "invalid_cursor", "Cursor conflicts with the requested snapshot.", command, 2
            )
        snapshot_id, generation = token["snapshot"], token["generation"]
    snapshot, failure, why = read_snapshot(profile)
    if snapshot_id is not None or generation is not None:
        if (
            snapshot is None
            or (snapshot_id is not None and snapshot.snapshot_id != snapshot_id)
            or (generation is not None and snapshot.generation != generation)
        ):
            raise PolicyError(
                "lookup_snapshot_expired",
                "That lookup page refers to a snapshot that is no longer available. "
                "Start without --page.",
                command,
                3,
            )
    if snapshot is None:
        return LookupsView(
            profile, scope, limit=limit, reason=why or (failure.value if failure else "no_snapshot")
        )
    doc, committed_generation, ledger, reason = load_lookup_data(profile)
    if doc is None or committed_generation != snapshot.generation:
        if cursor is not None:
            raise PolicyError(
                "lookup_snapshot_expired",
                "Lookup evidence for that page is unavailable. Start without --page.",
                command,
                3,
            )
        return LookupsView(
            profile,
            scope,
            snapshot_id=snapshot.snapshot_id,
            generation=snapshot.generation,
            limit=limit,
            reason=reason or "lookup_generation_pending",
        )
    selected = lookup_records(scope, doc, snapshot.collected_at)
    unique = {}
    for kind, identity, row in selected:
        held = unique.setdefault((kind, identity), row)
        if held != row:
            return LookupsView(
                profile,
                scope,
                snapshot_id=snapshot.snapshot_id,
                generation=snapshot.generation,
                limit=limit,
                reason="conflicting_lookup_identity",
            )
    selected = sorted(
        ((kind, identity, row) for (kind, identity), row in unique.items()),
        key=lambda item: (item[2].ts_utc or "", item[0], item[1]),
    )
    binding = digest_of(
        [
            profile,
            scope,
            limit,
            snapshot.snapshot_id,
            snapshot.generation,
            doc.to_dict(),
            ledger.rows.digest if ledger else reason,
        ]
    )
    start = offset(token, binding, command)
    if token is not None and start >= len(selected):
        raise PolicyError("invalid_cursor", "Cursor has no remaining lookup rows.", command, 2)
    rows = tuple(
        project_lookup(kind, identity, row, doc, ledger)
        for kind, identity, row in selected[start : start + limit]
    )
    # A receipt link is only offered when the actual published explanation exists.
    from dataclasses import replace

    from ssgrep.accounting.views.records import read_receipt

    visible = []
    for row in rows:
        links = []
        missing_links = False
        for receipt_id in row.receipt_ids:
            try:
                if read_receipt(profile, receipt_id) is not None:
                    links.append(receipt_id)
                else:
                    missing_links = True
            except PolicyError:
                missing_links = True
        missing = row.missing_evidence + (
            ("receipt_explanation_unavailable",) if missing_links else ()
        )
        visible.append(replace(row, receipt_ids=tuple(links), missing_evidence=missing))
    more = start + limit < len(selected)
    warnings = () if reason is None else ("valuation_unavailable:" + reason,)
    if doc.lookup_schema != "lookups_v1":
        warnings += ("lookup_metadata_unavailable",)
    untimed = sum(
        r.ts_utc is None
        for r in doc.invocations
        if r.cls
        in {InvocationClass.MATCHED, InvocationClass.TRANSCRIPT_ONLY, InvocationClass.JOURNAL_GAP}
        and set(r.operations) & {"search", "show"}
    )
    if untimed:
        warnings += (
            f"untimed_lookup_rows:{untimed} retained native calls have no timestamp; "
            "default pages retain them without known window membership; "
            "exact time filters exclude them.",
        )
    return LookupsView(
        profile,
        scope,
        tuple(visible),
        len(selected),
        next_cursor(start + limit, binding, snapshot.snapshot_id, snapshot.generation)
        if more
        else None,
        snapshot.snapshot_id,
        snapshot.generation,
        limit,
        warnings=warnings,
        coverage={
            "accepted_lookup_rows_in_selection": len(selected),
            "native_invocation_rows_retained": len(doc.invocations),
            "lookup_metadata_current": doc.lookup_schema == "lookups_v1",
            "untimed_invocation_rows": untimed,
        },
    )


def reference(cursor: str | None, command: str):
    if cursor is None:
        return None
    try:
        if not isinstance(cursor, str) or len(cursor) > 2048 or not cursor.startswith("lookups1."):
            raise ValueError("invalid cursor")
        payload = base64.urlsafe_b64decode(cursor[9:].encode() + b"=" * (-len(cursor[9:]) % 4))
        raw = json.loads(payload)
        if set(raw) != {"snapshot", "generation", "page"}:
            raise ValueError("invalid fields")
        if not isinstance(raw["snapshot"], str) or len(raw["snapshot"]) > 512:
            raise ValueError("invalid snapshot")
        if isinstance(raw["generation"], bool) or not isinstance(raw["generation"], int):
            raise ValueError("invalid generation")
        if not isinstance(raw["page"], str):
            raise ValueError("invalid page")
        return raw
    except (ValueError, TypeError, UnicodeError, KeyError) as error:
        raise PolicyError(
            "invalid_cursor", "That is not a lookup page. Start without --page.", command, 2
        ) from error


def offset(cursor, binding: str, command: str) -> int:
    return redeem("l", cursor["page"] if cursor is not None else None, binding, command)


def next_cursor(offset_: int, binding: str, snapshot: str, generation: int) -> str:
    raw = {"snapshot": snapshot, "generation": generation, "page": issue("l", offset_, binding)}
    return "lookups1." + base64.urlsafe_b64encode(canonical_bytes(raw)).decode().rstrip("=")


def project_lookup(kind, identity, row, doc, data):
    invocation = identity if kind == "invocation" else row.invocation_id
    executions = row.execution_ids if kind == "invocation" else (identity,)
    native_current = doc.lookup_schema == "lookups_v1"
    candidates = (
        tuple(c for c in data.candidates if invocation in c.exposure_ids)
        if data and invocation
        else ()
    )
    debits = (
        tuple(d for d in data.debits if d.exposure_id == invocation) if data and invocation else ()
    )
    reuse = row.reuse
    if candidates:
        strength = {
            Reuse.NONE_OBSERVED: 0,
            Reuse.INSPECTED: 1,
            Reuse.ACTION_REFERENCE: 2,
            Reuse.VALIDATED: 3,
            Reuse.CONTRADICTED: 4,
        }
        reuse = max((reuse, *(c.reuse for c in candidates)), key=strength.__getitem__)
    evaluated = bool(candidates) or (
        kind == "invocation" and native_current and row.reuse_evaluated
    )
    reasons = []
    scenarios = ()
    costs = ()
    method_retired = data is not None and (
        data.methodology_spec.valuation_version != CURRENT_VALUATION_VERSION
    )
    if data and (candidates or debits) and not method_retired:
        valued = value_evidence(EvidenceBundle(candidates, debits), data.methodology_spec)
        by_unit = {
            unit: [leaf for leaf in valued.leaves if leaf.unit == unit]
            for unit in data.methodology_spec.units
        }
        estimates = tuple(
            e for unit, leaves in by_unit.items() if leaves for e in rollup(leaves, unit)
        )
        scenarios = tuple(e.as_scenario_estimate() for e in estimates)
        costs = tuple(
            (f"{unit.unit}:{unit.method}", next(e.debit for e in estimates if e.unit == unit))
            for unit, leaves in by_unit.items()
            if leaves
        )
        reasons.extend(
            reason for e in estimates for _, reason in (*e.credit_missing, *e.debit_missing)
        )
        states = {state for e in estimates for _, state in e.eligibility}
        eligibility = (
            Eligibility.UNAVAILABLE
            if "unavailable" in states
            else Eligibility.SUPPORTED
            if "supported" in states
            else Eligibility.KNOWN_REJECTED
        )
    else:
        eligibility = Eligibility.UNAVAILABLE
    if not debits:
        reasons.append("comparable_lookup_cost_unavailable")
    if row.ts_utc is None:
        reasons.append("lookup_timestamp_unavailable")
    gate_reasons = {r for c in candidates for r in c.eligibility_reasons}
    if not evaluated:
        reasons.append("reuse_not_evaluated")
    if not native_current:
        eligibility = Eligibility.UNAVAILABLE
        gate_reasons.add("lookup_metadata_unavailable")
    elif kind == "execution" and row.invocation_id is None:
        eligibility = Eligibility.UNAVAILABLE
        gate_reasons.add("downstream_exposure_unobserved")
    elif not candidates:
        if row.actor == "verifier" and row.actor_basis == "configured_verifier":
            eligibility = Eligibility.KNOWN_REJECTED
            gate_reasons.add("excluded_verifier_actor")
        elif row.outcome.value in {"empty", "error"}:
            eligibility = Eligibility.KNOWN_REJECTED
            gate_reasons.add("lookup_" + row.outcome.value)
        elif not evaluated:
            eligibility = Eligibility.UNAVAILABLE
            gate_reasons.add("reuse_not_evaluated")
        elif row.exposure is Exposure.UNAVAILABLE:
            eligibility = Eligibility.UNAVAILABLE
            gate_reasons.add("downstream_exposure_unobserved")
        elif reuse in {Reuse.NONE_OBSERVED, Reuse.INSPECTED, Reuse.CONTRADICTED}:
            eligibility = Eligibility.KNOWN_REJECTED
            gate_reasons.add(
                "contradicted" if reuse is Reuse.CONTRADICTED else "no_supported_reuse"
            )
        else:
            eligibility = Eligibility.UNAVAILABLE
            gate_reasons.add("candidate_evidence_expired_or_unavailable")
    if method_retired:
        eligibility = Eligibility.UNAVAILABLE
        gate_reasons.add("methodology_retired")
    receipt_ids = ()
    if data:
        receipt_ids = tuple(
            sorted(
                {
                    r.receipt_id
                    for r in data.rows.receipts
                    if RECEIPT_ID_RE.fullmatch(r.receipt_id)
                    and (
                        invocation in r.receipt.exposure_ids
                        if invocation
                        else bool(set(executions) & set(r.receipt.execution_ids))
                    )
                }
            )
        )
    useful = (
        {
            Reuse.VALIDATED: "validated_association",
            Reuse.ACTION_REFERENCE: "action_reference_association",
            Reuse.INSPECTED: "inspection_only",
            Reuse.CONTRADICTED: "contradicted",
            Reuse.NONE_OBSERVED: "no_reuse_observed",
        }[reuse]
        if evaluated
        else "not_evaluated"
    )
    return LookupRow(
        lookup_id=f"{kind}:{identity}",
        invocation_id=invocation,
        execution_ids=tuple(sorted(set(executions))),
        runtime=row.runtime,
        session_id=row.session_id,
        ambiguous_execution_ids=row.ambiguous_execution_ids if kind == "invocation" else (),
        ts_utc=row.ts_utc,
        task=task_key(row.task),
        actor=row.actor,
        actor_basis=row.actor_basis,
        scope_basis=row.scope_basis,
        fingerprint_key_id=row.fingerprint_key_id,
        join_rejection_reasons=row.join_rejection_reasons,
        in_scope=row.in_scope,
        operations=row.operations if kind == "invocation" else (row.operation,),
        execution=row.cls.value if kind == "invocation" else "journal_only",
        exposure=row.exposure.value,
        attribution=row.attribution.value,
        reuse=reuse.value if evaluated else "unavailable",
        reuse_evaluated=evaluated,
        eligibility=eligibility.value,
        eligibility_reasons=tuple(sorted(gate_reasons)),
        outcome=row.outcome.value,
        useful_outcome=useful,
        purpose=row.purpose if kind == "invocation" else "unknown",
        measured_service_ms=service_milliseconds(row.measured_service_ns),
        known_costs=costs,
        missing_evidence=tuple(sorted(set(reasons) | gate_reasons)),
        receipt_ids=receipt_ids,
        scenarios=scenarios,
        visible_bytes=row.visible_bytes,
        timing_known=row.timing_known,
        frequency_view_id=row.frequency_view_id if kind == "invocation" else "",
        frequency_provenance=row.frequency_provenance if kind == "invocation" else (),
    )


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


LOOKUP_WORK_MAPPING = "journal_active_proxy_ns_v2"
LOOKUP_WORK_ASSUMPTIONS = (
    "Each unique journal operation models one fully occupied active actor during service.",
    "Each modeled operation is capped at 300 seconds, matching capped_gap_v1 source work.",
    "Concurrent unique operations add actor work, not elapsed or human waiting time.",
    "Service occupancy is a proxy, not observed active work or causal time savings.",
    "Raw monotonic nanoseconds are capped per operation and summed before unit conversion.",
)


@dataclass(frozen=True)
class ExecutionTiming:
    """One unique execution: exact latency and its explicitly estimated debit."""

    execution_id: str
    measured_service_ns: Quantity
    work_ns: int | None

    @property
    def measured_service_ms(self) -> Quantity:
        return service_milliseconds(self.measured_service_ns)

    @property
    def work(self) -> Quantity:
        if self.work_ns is None:
            return Quantity.unavailable(WORK_SECONDS, self.measured_service_ns.reason)
        return ns_seconds_quantity(self.work_ns, CAPPED_GAP_V1, Basis.ESTIMATED)


@dataclass(frozen=True)
class LookupTiming:
    """Linked operation rows, plus coverage (partial timing never yields a total)."""

    rows: tuple[ExecutionTiming, ...] = ()
    complete: bool = False
    reason: str = "no_linked_journal_timing"
    mapping_version: str = LOOKUP_WORK_MAPPING
    assumptions: tuple[str, ...] = LOOKUP_WORK_ASSUMPTIONS

    @property
    def measured_service_ns(self) -> Quantity:
        if not self.complete:
            return Quantity.unavailable(WORK_NS, self.reason)
        for row in self.rows:
            if row.measured_service_ns.value is None:
                return Quantity.unavailable(WORK_NS, row.measured_service_ns.reason)
        return Quantity(
            sum(
                row.measured_service_ns.value
                for row in self.rows
                if row.measured_service_ns.value is not None
            ),
            WORK_NS,
            Basis.OBSERVED,
            MONOTONIC_V1,
        )

    @property
    def measured_service_ms(self) -> Quantity:
        return service_milliseconds(self.measured_service_ns)

    @property
    def work(self) -> Quantity:
        if not self.complete:
            return Quantity.unavailable(WORK_SECONDS, self.reason)
        if any(row.work_ns is None for row in self.rows):
            return Quantity.unavailable(WORK_SECONDS, "journal_monotonic_duration_missing")
        return ns_seconds_quantity(
            sum(row.work_ns for row in self.rows if row.work_ns is not None),
            CAPPED_GAP_V1,
            Basis.ESTIMATED,
        )


def _timing(execution_id: str, measured: Quantity) -> ExecutionTiming:
    if measured.value is None:
        return ExecutionTiming(execution_id, measured, None)
    return ExecutionTiming(execution_id, measured, min(measured.value, GAP_CAP_NS))


def journal_timings(events: Iterable[ExecutionEvent]) -> dict[str, ExecutionTiming]:
    """Deduplicate immutable phases; conflicting evidence never picks a duration.

    Capture's COMPLETED.mono_ns is an elapsed duration, not an absolute clock.
    Wall timestamps determine joins elsewhere; a wall discontinuity cannot change
    this independent monotonic measurement. A completed event suffices to retain
    its latency, even if the started/rendered phase is unavailable.
    """
    groups: dict[str, dict[Phase, dict[str, ExecutionEvent]]] = {}
    for event in events:
        phases = groups.setdefault(event.execution_id, {})
        phases.setdefault(event.phase, {})[digest_of(event)] = event
    out: dict[str, ExecutionTiming] = {}
    for execution_id, phases in sorted(groups.items()):
        done = phases.get(Phase.COMPLETED, {})
        reason = ""
        if any(len(variants) != 1 for variants in phases.values()):
            reason = "conflicting_journal_timing"
        elif not done:
            reason = "journal_monotonic_duration_missing"
        else:
            completed = next(iter(done.values()))
            identities = {
                (e.operation, e.transport, e.profile, e.fingerprint)
                for variants in phases.values()
                for e in variants.values()
            }
            if len(identities) != 1:
                reason = "conflicting_journal_timing"
            elif completed.mono_ns is None:
                reason = "journal_monotonic_duration_missing"
        if reason:
            measured = Quantity.unavailable(WORK_NS, reason)
        else:
            assert completed.mono_ns is not None
            measured = measured_duration(0, completed.mono_ns)
        out[execution_id] = _timing(execution_id, measured)
    return out


def link_call_timing(
    requests: Sequence[Request],
    owned: Mapping[tuple[str, int], Sequence[str]],
    ambiguous: Mapping[tuple[str, int], Sequence[str]],
    timings: Mapping[str, ExecutionTiming],
) -> LookupTiming:
    """Attach only uniquely owned journal rows; retain unknown invocation coverage.

    For a shell loop of unknown cardinality, observed rows remain supported but
    cannot prove that every operation was captured. Unknown/ambiguous requests
    therefore add an unavailable debit, rather than treating the known rows as a
    complete estimate. Notes do not contribute to lookup-work quantities.
    """
    execution_ids: set[str] = set()
    reasons: set[str] = set()
    lookups = [r for r in requests if r.operation in ("search", "show")]
    for request in lookups:
        ids = set(owned.get(request.key, ()))
        execution_ids.update(ids)
        if ambiguous.get(request.key):
            reasons.add("ambiguous_journal_timing")
        if request.capacity != 1:
            reasons.add("journal_cardinality_unbounded")
        elif len(ids) != 1:
            reasons.add("no_linked_journal_timing")
    rows = []
    for execution_id in sorted(execution_ids):
        row = timings.get(execution_id) or _timing(
            execution_id, Quantity.unavailable(WORK_NS, "journal_monotonic_duration_missing")
        )
        rows.append(row)
        if row.measured_service_ns.value is None:
            reasons.add(row.measured_service_ns.reason)
    if not lookups:
        reasons.add("no_lookup_request")
    complete = not reasons and bool(rows)
    return LookupTiming(tuple(rows), complete, "" if complete else sorted(reasons)[0])
