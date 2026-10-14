"""The combined deterministic reducer: raw evidence + journal -> :class:`Reduction`.

``reduce_profile`` is a pure function (no clocks, no I/O). Transcripts alone cannot
count loop executions and the journal alone cannot show reuse, so both are joined
(``reconcile``), tasks come from raw lineage (``lineage``), and exposure, reuse,
candidates and debits come from observed results only. Every output is sorted, so
shuffling the inputs yields an identical :attr:`Reduction.digest`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from ssgrep.accounting.contracts import (
    Attribution,
    ExecutionEvent,
    Exposure,
    InvocationClass,
    Outcome,
    Quantity,
    RawEvidence,
    Reuse,
)
from ssgrep.accounting.evidence_tokens import NORMALIZATION
from ssgrep.accounting.exposure import exposure_of, is_usage_error
from ssgrep.accounting.frequency import THRESHOLD_MILLI, FrequencyView
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.lineage import build_lineage
from ssgrep.accounting.policy_rules import PolicyFile
from ssgrep.accounting.reconcile import (
    ExecFact,
    Link,
    fold_events,
    join_index,
    reconcile,
)
from ssgrep.accounting.reduce_calls import Call, collect_calls
from ssgrep.accounting.reduce_cands import CHARS, WORK, Builder, Outputs
from ssgrep.accounting.reduce_types import (
    ExecutionRecord,
    InvocationRecord,
    Reduction,
    summarize,
)
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.sources import Resolver
from ssgrep.accounting.timeline import build_timelines, key_str, ts_ms
from ssgrep.accounting.units import UnitSpec
from ssgrep.accounting.valuation_inputs import (
    UNKNOWN_COST_STATUSES,
    AtomStatus,
    EvidenceBundle,
    Methodology,
)

CONSUMED_KINDS = frozenset({"tool_call", "tool_result", "user_turn", "session_header", "lineage"})
SHELL_WORK_NOTE = "lookup_work_seconds_not_modeled"


def methodology_for(frequency: FrequencyView) -> Methodology:
    """The production methodology namespace that bundles of this reduction belong to."""
    units: tuple[UnitSpec, ...] = (CHARS, WORK)
    return Methodology(
        label=f"production-{NORMALIZATION}",
        units=units,
        thresholds=tuple((s.value, m // 1000) for s, m in THRESHOLD_MILLI.items()),
        frequency_view=frequency.version_id,
        calibrated=False,
    )


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
) -> tuple[list[ExecutionRecord], int]:
    records: list[ExecutionRecord] = []
    skipped = 0
    for fact in facts:
        if not scoper.in_scope(fact.cwd):
            skipped += 1
            continue
        link = links[fact.execution_id]
        call = calls_by_key[link.requests[0]] if link.requests else None
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
                in_scope=True,
                cwd=fact.cwd,
                correlation_id=fact.correlation_id,
                truncated=fact.truncated,
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
) -> list[InvocationRecord]:
    out: list[InvocationRecord] = []
    # A matched CLI process records its canonical cwd at capture time. Reuse that
    # identity for exposure/session grouping; never resolve historical aliases at read
    # time or substitute an MCP server's cwd for its caller's project.
    captured_cwds: dict[str, set[str]] = {}
    for execution in executions:
        if execution.transport == "cli" and execution.invocation_id and execution.cwd:
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
                in_scope=True,
                cwd=next(iter(cwds)) if len(cwds) == 1 else call.cwd,
            )
        )
    return sorted(out, key=lambda i: (i.runtime, i.session_id, i.invocation_id))


def _bundles(
    outputs: Outputs, calls: list[Call]
) -> tuple[tuple[EvidenceBundle, ...], tuple[str, ...]]:
    candidates: dict[str, list] = {}
    debits: dict[str, list] = {}
    for record in outputs.candidates:
        candidates.setdefault(record.candidate.task.root, []).append(record.candidate)
    for debit in outputs.debits:
        debits.setdefault(debit.task.root, []).append(debit)
    roots = sorted(set(candidates) | set(debits))
    bundles = tuple(
        EvidenceBundle(
            tuple(
                sorted(
                    candidates.get(r, ()), key=lambda c: (c.task.turn or "", c.candidate_id, c.day)
                )
            ),
            tuple(sorted(debits.get(r, ()), key=lambda d: d.debit_id)),
        )
        for r in roots
    )
    return bundles, tuple(roots)


def _source_coverage(outputs: Outputs) -> tuple[int, int]:
    known = unavailable = 0
    for record in outputs.candidates:
        atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
        bad = any(a.status in UNKNOWN_COST_STATUSES or a.status is not AtomStatus.OK for a in atoms)
        if atoms and not bad:
            known += 1
        else:
            unavailable += 1
    return known, unavailable


def reduce_profile(
    profile: str,
    evidence: Iterable[RawEvidence],
    events: Iterable[ExecutionEvent],
    *,
    policy: ProfilePolicy,
    frequency: FrequencyView,
    fingerprint_key: bytes | None = None,
    verifier_sessions: frozenset[str] = frozenset(),
    known_atoms: Mapping[str, str] | None = None,
    ownership: PolicyFile | None = None,
) -> Reduction:
    """Reduce retained evidence and journal events to one deterministic :class:`Reduction`.

    ``verifier_sessions`` holds ``"runtime:session"`` ids (or roots) of verifier actors:
    their lookups stay in the execution denominators but never become solver exposure,
    debit or credit. ``known_atoms`` maps origin event identity to the digest recorded
    earlier, so source drift is detected and earns no credit.
    """
    timelines = build_timelines(item for item in evidence if item.kind in CONSUMED_KINDS)
    lineage = build_lineage(timelines)
    scoper = Scoper(policy, ownership)
    callset = collect_calls(timelines, lineage, scoper, fingerprint_key, verifier_sessions)
    mine = [e for e in events if e.profile == profile]
    facts = fold_events(mine)
    requests = [r for call in callset.calls for r in call.requests]
    links = reconcile(facts, requests)
    joined = join_index(links)
    calls_by_key = {r.key: call for call in callset.calls for r in call.requests}
    owned_by_invocation: dict[str, list[str]] = {}
    for call in callset.calls:
        owned_by_invocation[call.invocation_id] = sorted(
            {e for r in call.requests for e in joined.owned.get(r.key, [])}
        )
    lookup_ids: dict = {}
    for call in callset.calls:
        lookup_ids.setdefault(call.key, set()).add(call.call.ev.record_id)
    root_of = dict(lineage.root_of)
    resolver = Resolver(timelines, scoper, known_atoms or {}, root_of)
    builder = Builder(
        timelines, lineage, scoper, frequency, resolver, fingerprint_key,
        {k: frozenset(v) for k, v in lookup_ids.items()},
    )  # fmt: skip
    outputs = builder.run([c for c in callset.calls if c.in_scope], owned_by_invocation)
    executions, out_of_scope = _executions(
        facts, links, calls_by_key, joined.owned, scoper, outputs
    )
    stamps = [t for t in (ts_ms(e.ts_utc) for e in mine) if t is not None]
    span = (min(stamps), max(stamps)) if stamps else None
    invocations = _invocations(
        callset.calls, joined.owned, joined.ambiguous, links, span, outputs, executions
    )
    bundles, bundle_roots = _bundles(outputs, callset.calls)
    coverage = _coverage(
        executions, invocations, callset.counts, outputs, lineage, timelines, out_of_scope, callset
    )
    sessions = tuple(sorted((k[0], k[1], key_str(lineage.root_of[k])) for k in timelines.sessions))
    parents = tuple(
        sorted(
            (k[0], k[1], lineage.parent_of[k][1] if k in lineage.parent_of else None)
            for k in timelines.sessions
        )
    )
    return Reduction(
        profile=profile,
        executions=tuple(executions),
        invocations=tuple(invocations),
        candidates=tuple(sorted(outputs.candidates, key=_candidate_order)),
        exclusions=tuple(
            sorted(
                set(outputs.exclusions),
                key=lambda x: (x.candidate_id, x.invocation_id, x.reason, x.detail),
            )
        ),
        bundles=bundles,
        bundle_roots=bundle_roots,
        session_roots=sessions,
        session_parents=parents,
        coverage=coverage,
        observations=summarize(executions),
        warnings=tuple(sorted(f"lineage:{i.kind}:{i.session}" for i in lineage.issues)),
        frequency_version=frequency.version_id,
    )


def _candidate_order(record) -> tuple:
    c = record.candidate
    return (c.task.root, c.task.turn or "", c.candidate_id, c.day, record.invocation_id)


def _coverage(
    executions, invocations, counts, outputs, lineage, timelines, out_of_scope, callset
) -> dict[str, int]:
    cov: Counter[str] = Counter()
    cov["recorded_executions"] = len(executions)
    for attribution in Attribution:
        cov[attribution.value] = sum(1 for e in executions if e.attribution is attribution)
    lookups = [e for e in executions if e.operation in ("search", "show")]
    cov["recorded_lookups"] = len(lookups)
    cov["attributed_lookups"] = sum(
        1 for e in lookups if e.attribution in (Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE)
    )
    cov["ambiguous_joins"] = cov[Attribution.AMBIGUOUS.value]
    cov["recognized_invocations"] = len(invocations)
    for cls in InvocationClass:
        cov[f"inv_{cls.value}"] = sum(1 for i in invocations if i.cls is cls)
    cov["inv_mention"] = counts["mention"]
    cov["inv_unclassified"] = counts["unclassified"]
    cov["classified_tool_calls"] = (
        counts["mention"] + counts["unclassified"] + counts["non_lookup"] + len(invocations)
    )
    cov["tasks_resolved"] = sum(1 for i in invocations if i.task is not None and i.task.resolved)
    cov["tasks_unknown"] = len(invocations) - cov["tasks_resolved"]
    cov["candidate_receipts"] = len(outputs.candidates)
    known, unavailable = _source_coverage(outputs)
    cov["source_costs_available"] = known
    cov["unavailable_sources"] = unavailable
    cov["executions_out_of_scope"] = out_of_scope
    cov["invocations_out_of_scope"] = callset.out_of_scope
    cov["evidence_duplicates"] = timelines.duplicates
    cov["evidence_conflicts"] = timelines.conflicts
    cov["sessions"] = len(timelines.sessions)
    for issue in lineage.issues:
        cov[f"lineage.{issue.kind}"] += 1
    for name, value in outputs.counts.items():
        cov[name] += value
    return dict(sorted(cov.items()))
