"""Materialize deterministic execution and invocation records from joined evidence."""

from __future__ import annotations

from collections.abc import Mapping

from ssgrep.accounting.contracts import (
    Attribution,
    Exposure,
    InvocationClass,
    Outcome,
    Quantity,
    Reuse,
)
from ssgrep.accounting.exposure import exposure_of, is_usage_error
from ssgrep.accounting.frequency_assignment import FrequencyAssignments
from ssgrep.accounting.lookup_time import ExecutionTiming
from ssgrep.accounting.reconcile import ExecFact, Link
from ssgrep.accounting.reduce_calls import Call
from ssgrep.accounting.reduce_cands import Outputs
from ssgrep.accounting.reduce_types import ExecutionRecord, InvocationRecord
from ssgrep.accounting.scope_events import Scoper


def _visible(call: Call, shared: bool) -> Quantity:
    interp = call.interp
    if interp is None:
        return Quantity.unavailable("bytes", "downstream_exposure_unobserved")
    if call.shape == "mixed" and not interp.cards:
        return Quantity.unavailable("bytes", "mixed_shell_output_unattributable")
    if shared:
        return Quantity.unavailable("bytes", "output_shared_by_several_executions")
    return Quantity.observed(len(interp.text.encode()), "bytes", "tool_result_text")


def _outcome(fact: ExecFact, call: Call | None, owned_count: int) -> tuple[Outcome, str]:
    """Journal outcome unless a native flag, exit status, or recognized error payload says error."""
    if call is None or call.interp is None or owned_count != 1:
        return fact.outcome, fact.outcome_basis
    interp = call.interp
    if interp.is_error and fact.outcome in (Outcome.OK, Outcome.EMPTY, Outcome.UNKNOWN):
        return Outcome.ERROR, "result_override"
    if fact.outcome is Outcome.UNKNOWN and interp.outcome in (Outcome.OK, Outcome.EMPTY):
        return interp.outcome, "result_override"
    return fact.outcome, fact.outcome_basis


def _invocation_class(
    call: Call, owned: list[str], ambiguous: list[str], span: tuple[int, int] | None
) -> tuple[InvocationClass, str]:
    if owned:
        return InvocationClass.MATCHED, ""
    if ambiguous:
        return InvocationClass.TRANSCRIPT_ONLY, "ambiguous_join"
    if call.interp is not None and is_usage_error(call.interp):
        return InvocationClass.TRANSCRIPT_ONLY, "pre_api_failure"
    if span is not None and call.ts is not None and span[0] <= call.ts <= span[1]:
        return InvocationClass.JOURNAL_GAP, "no_journal_event_in_capture_span"
    return InvocationClass.TRANSCRIPT_ONLY, "outside_capture_span"


def _attribution(owned: list[str], ambiguous: list[str], links: Mapping[str, Link]) -> Attribution:
    if owned:
        kinds = {links[e].attribution for e in owned}
        return (
            Attribution.EXPLICIT if kinds == {Attribution.EXPLICIT} else Attribution.INFERRED_UNIQUE
        )
    return Attribution.AMBIGUOUS if ambiguous else Attribution.UNATTRIBUTED


def _executions(
    facts: list[ExecFact],
    links: dict[str, Link],
    calls_by_key: dict[tuple[str, int], Call],
    joined_owned: dict[tuple[str, int], list[str]],
    scoper: Scoper,
    outputs: Outputs,
    timings: Mapping[str, ExecutionTiming],
) -> tuple[list[ExecutionRecord], int]:
    from ssgrep.accounting.join_scope import execution_scope

    records: list[ExecutionRecord] = []
    skipped = 0
    for fact in facts:
        link = links[fact.execution_id]
        call = calls_by_key[link.requests[0]] if link.requests else None
        cwd, scope_basis, in_scope = execution_scope(fact, link, call, scoper)
        if not in_scope:
            skipped += 1
            continue
        owned_count = len(joined_owned.get(link.requests[0], [])) if link.requests else 0
        outcome, basis = _outcome(fact, call, owned_count)
        exposure = Exposure.UNAVAILABLE
        if call is not None:
            exposure = exposure_of(call.shape, call.interp)
            if owned_count > 1 and exposure is Exposure.OBSERVED:
                exposure = Exposure.PARTIAL
        records.append(
            ExecutionRecord(
                execution_id=fact.execution_id,
                ts_utc=fact.ts,
                day=fact.ts[:10],
                operation=fact.operation,
                transport=fact.transport,
                outcome=outcome,
                outcome_basis=basis,
                attribution=link.attribution,
                source="both" if call is not None else "journal",
                runtime=call.key[0] if call else None,
                session_id=call.key[1] if call else None,
                invocation_id=call.invocation_id if call else None,
                actor=call.actor if call else "unknown",
                task=call.task if call else None,
                rendered_bytes=fact.rendered,
                visible_bytes=_visible(call, owned_count > 1)
                if call
                else Quantity.unavailable("bytes", "no_runtime_exposure_joined"),
                exposure=exposure,
                reuse=outputs.reuse.get(call.invocation_id, Reuse.NONE_OBSERVED)
                if call
                else Reuse.NONE_OBSERVED,
                timing_known=fact.timing_known,
                measured_service_ms=timings[fact.execution_id].measured_service_ms,
                measured_service_ns=timings[fact.execution_id].measured_service_ns,
                in_scope=True,
                cwd=cwd,
                correlation_id=fact.correlation_id,
                truncated=fact.truncated,
                scope_basis=scope_basis,
                join_rejection_reasons=link.reasons,
                actor_basis=(
                    "configured_verifier" if call.actor == "verifier" else "native_lineage"
                )
                if call is not None
                else "unknown",
                fingerprint_key_id=fact.fingerprint_key_id,
            )
        )
    return sorted(records, key=lambda r: (r.ts_utc, r.execution_id)), skipped


def _invocations(
    calls: list[Call],
    joined_owned: dict[tuple[str, int], list[str]],
    joined_ambiguous: dict[tuple[str, int], list[str]],
    links: dict[str, Link],
    span: tuple[int, int] | None,
    outputs: Outputs,
    executions: list[ExecutionRecord],
    frequencies: FrequencyAssignments,
) -> list[InvocationRecord]:
    from ssgrep.accounting.join_scope import invocation_provenance

    out: list[InvocationRecord] = []
    # Preserve canonical execution identity for exposure/session grouping, including
    # trusted native caller metadata. Never resolve historical aliases at read time
    # or substitute an MCP server's cwd for its caller's project.
    captured_cwds: dict[str, set[str]] = {}
    for execution in executions:
        if (
            (execution.transport == "cli" or execution.scope_basis == "native_call_metadata")
            and execution.invocation_id
            and execution.cwd
        ):
            captured_cwds.setdefault(execution.invocation_id, set()).add(execution.cwd)
    for call in calls:
        if not call.in_scope:
            continue
        owned = sorted({e for r in call.requests for e in joined_owned.get(r.key, [])})
        ambiguous = sorted({e for r in call.requests for e in joined_ambiguous.get(r.key, [])})
        cls, reason = _invocation_class(call, owned, ambiguous, span)
        interp = call.interp
        shared = len(owned) > 1
        cwds = captured_cwds.get(call.invocation_id, ())
        out.append(
            InvocationRecord(
                invocation_id=call.invocation_id,
                runtime=call.key[0],
                session_id=call.key[1],
                call_id=str(call.call.ev.payload.get("call_id") or ""),
                args_hash=call.invocation_id[:16],
                ts_utc=call.call.ev.ts_utc,
                operations=call.operations,
                route=call.classification.invocation.route
                if call.classification.invocation
                else "",
                cardinality=call.classification.invocation.cardinality
                if call.classification.invocation
                else "",
                cls=cls,
                attribution=_attribution(owned, ambiguous, links),
                execution_ids=tuple(owned),
                ambiguous_execution_ids=tuple(ambiguous),
                outcome=interp.outcome if interp is not None else Outcome.UNKNOWN,
                exposure=exposure_of(call.shape, interp),
                visible_bytes=_visible(call, shared),
                task=call.task,
                actor=call.actor,
                reason=reason,
                purpose=outputs.purpose.get(call.invocation_id, "unknown"),
                pre_api_failure=reason == "pre_api_failure",
                timing_known=call.requests[0].window is not None if call.requests else False,
                measured_service_ms=call.timing.measured_service_ms,
                measured_service_ns=call.timing.measured_service_ns,
                in_scope=True,
                cwd=next(iter(cwds)) if len(cwds) == 1 else call.cwd,
                reuse=outputs.reuse.get(call.invocation_id, Reuse.NONE_OBSERVED),
                reuse_evaluated=call.invocation_id in outputs.reuse,
                frequency_view_id=(
                    dict(frequencies.provenance(call.invocation_id))["frequency"]
                    if call.lookup
                    else ""
                ),
                frequency_provenance=(
                    frequencies.provenance(call.invocation_id) if call.lookup else ()
                ),
                **invocation_provenance(call, owned, ambiguous, links),
            )
        )
    return sorted(out, key=lambda i: (i.runtime, i.session_id, i.invocation_id))
