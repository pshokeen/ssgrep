"""One lookup's evidence verdict; lookup estimates are explicitly nonadditive."""

from __future__ import annotations

from ssgrep.accounting.contracts import Exposure, Reuse
from ssgrep.accounting.lookup_types import LookupRow
from ssgrep.accounting.service_records import RECEIPT_ID_RE
from ssgrep.accounting.service_select import task_key
from ssgrep.accounting.timeunits import service_milliseconds
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import (
    CURRENT_VALUATION_VERSION,
    Eligibility,
    EvidenceBundle,
)


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
