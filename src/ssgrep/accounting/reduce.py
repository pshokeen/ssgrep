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
    InvocationClass,
    RawEvidence,
)
from ssgrep.accounting.evidence_tokens import NORMALIZATION
from ssgrep.accounting.frequency import THRESHOLD_MILLI, FrequencyView
from ssgrep.accounting.frequency_assignment import FrequencyAssignments, model_identity
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.lineage import build_lineage
from ssgrep.accounting.lookup_time import LOOKUP_WORK_MAPPING, journal_timings
from ssgrep.accounting.native_source import NativeSourceAdapter
from ssgrep.accounting.policy_rules import PolicyFile
from ssgrep.accounting.reconcile import (
    fold_events,
    join_index,
    reconcile,
)
from ssgrep.accounting.reduce_calls import attach_timing, collect_calls
from ssgrep.accounting.reduce_cands import CHARS, WORK, Builder, Outputs
from ssgrep.accounting.reduce_records import _executions, _invocations
from ssgrep.accounting.reduce_types import (
    Reduction,
    summarize,
)
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.sources import Resolver
from ssgrep.accounting.timeline import build_timelines, key_str, ts_ms
from ssgrep.accounting.valuation_inputs import (
    UNKNOWN_COST_STATUSES,
    AtomStatus,
    EvidenceBundle,
    Methodology,
)

CONSUMED_KINDS = frozenset(
    {"tool_call", "tool_result", "user_turn", "session_header", "lineage", "ownership_claim"}
)


def methodology_for(frequency: FrequencyView) -> Methodology:
    """The production methodology namespace that bundles of this reduction belong to."""
    return Methodology(
        label=f"production-{NORMALIZATION}-{LOOKUP_WORK_MAPPING}",
        units=(CHARS, WORK),
        thresholds=tuple((s.value, m // 1000) for s, m in THRESHOLD_MILLI.items()),
        frequency_view=model_identity(frequency),
        calibrated=False,
    )


def _bundles(outputs: Outputs) -> tuple[tuple[EvidenceBundle, ...], tuple[str, ...]]:
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
    native_sources: NativeSourceAdapter | None = None,
    frequency_assignments: FrequencyAssignments | None = None,
    capture_span: tuple[int, int] | None = None,
    partitioned: bool = False,
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
    if native_sources is not None:
        native_sources.scoper = scoper
    callset = collect_calls(timelines, lineage, scoper, fingerprint_key, verifier_sessions)
    mine = [e for e in events if e.profile == profile]
    facts = fold_events(mine)
    requests = [r for call in callset.calls for r in call.requests]
    links = reconcile(facts, requests)
    joined = join_index(links)
    timings = journal_timings(mine)
    callset.calls = attach_timing(callset.calls, timings, joined)
    if frequency_assignments is None:
        from ssgrep.accounting.fingerprint import key_id

        frequencies = FrequencyAssignments(frequency, fingerprint_key_id=key_id(fingerprint_key))
    else:
        frequencies = frequency_assignments
    bound_executions: set[str] = set()
    for call in callset.calls:
        if call.in_scope and call.lookup:
            owned = {e for request in call.requests for e in joined.owned.get(request.key, [])}
            frequencies.bind(call.invocation_id, owned)
            bound_executions.update(owned)
    for fact in facts:
        if (
            fact.operation in ("search", "show")
            and fact.execution_id not in bound_executions
            and scoper.in_scope(fact.cwd)
        ):
            frequencies.view_for(f"journal:{fact.execution_id}")
    calls_by_key = {r.key: call for call in callset.calls for r in call.requests}
    owned_by_invocation: dict[str, list[str]] = {}
    for call in callset.calls:
        owned_by_invocation[call.invocation_id] = sorted(
            {e for r in call.requests for e in joined.owned.get(r.key, [])}
        )
    lookup_ids: dict = {}
    for call in callset.calls:
        lookup_ids.setdefault(call.key, set()).add(call.call.ev.record_id)
    resolver = Resolver(
        timelines, scoper, known_atoms or {}, lineage.root_of, native_sources=native_sources
    )
    builder = Builder(
        timelines, lineage, scoper, frequency, resolver, fingerprint_key,
        {k: frozenset(v) for k, v in lookup_ids.items()},
        frequencies=frequencies,
    )  # fmt: skip
    outputs = builder.run([c for c in callset.calls if c.in_scope], owned_by_invocation)
    executions, out_of_scope = _executions(
        facts, links, calls_by_key, joined.owned, scoper, outputs, timings
    )
    stamps = [t for t in (ts_ms(e.ts_utc) for e in mine) if t is not None]
    span = (
        capture_span if capture_span is not None else (min(stamps), max(stamps)) if stamps else None
    )
    invocations = _invocations(
        callset.calls, joined.owned, joined.ambiguous, links, span, outputs, executions, frequencies
    )
    bundles, bundle_roots = _bundles(outputs)
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
        frequency_version=model_identity(frequency),
        frequency_assignments=frequencies.records(
            [
                *(call.invocation_id for call in callset.calls),
                *(f"journal:{fact.execution_id}" for fact in facts),
            ]
            if partitioned
            else None
        ),
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
