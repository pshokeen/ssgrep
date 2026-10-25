"""Deterministic reduction records and invocation/execution materialization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssgrep.accounting.capture.fingerprint import key_id
from ssgrep.accounting.ledger.contracts import (
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
from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment
from ssgrep.accounting.ledger.reduce_join import (
    SINGLE,
    ExecFact,
    Interpretation,
    Link,
    Request,
    correlation_in,
    exposure_of,
    interpret,
    is_usage_error,
    parse_argv,
    request_fingerprint,
    shell_shape,
    window_of,
)
from ssgrep.accounting.ledger.valuation_inputs import Candidate, EvidenceBundle
from ssgrep.accounting.readers.classify import Classification, classify_evidence
from ssgrep.accounting.readers.native_identity import native_in_scope
from ssgrep.accounting.readers.pi_family import Lineage
from ssgrep.accounting.readers.timeline import Record, SessionKey, Timelines, call_text, key_str

if TYPE_CHECKING:
    from ssgrep.accounting.collect.frequency import FrequencyAssignments
    from ssgrep.accounting.ledger.reduce_cands import Outputs
    from ssgrep.accounting.views.filter import Scoper
    from ssgrep.accounting.views.lookups import ExecutionTiming, LookupTiming

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

LOOKUPS = frozenset({"search", "show", "note"})
RESULT_BASES = frozenset({"structured", "error_payload", "native_flag"})


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


# Consolidated from reduce_records.py; public data structures and helpers retained.


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
    from ssgrep.accounting.views.filter import execution_scope

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
    from ssgrep.accounting.views.filter import invocation_provenance

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


def _empty_lookup_timing():
    from ssgrep.accounting.views.lookups import LookupTiming

    return LookupTiming()


@dataclass
class Call:
    """One recognized ssgrep invocation in one session (a shell line may hold many requests)."""

    invocation_id: str
    key: SessionKey
    call: Record
    result: Record | None
    classification: Classification
    interp: Interpretation | None
    shape: str  # pure | filtered | mixed
    requests: tuple[Request, ...]
    cwd: str | None
    ts: int | None
    task: TaskId
    actor: str
    in_scope: bool = True
    timing: LookupTiming = field(default_factory=_empty_lookup_timing)

    @property
    def operations(self) -> tuple[str, ...]:
        invocation = self.classification.invocation
        return invocation.operations if invocation else ()

    @property
    def lookup(self) -> bool:
        """A search/show response the consumer could have read (notes are not lookups)."""
        return any(op in ("search", "show") for op in self.operations)


@dataclass
class CallSet:
    calls: list[Call] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    out_of_scope: int = 0


def invocation_id(record: Record) -> str:
    args = digest_of(record.ev.payload.get("arguments"))[:16]
    return digest_of([record.ev.native_key, args])[:24]


def attach_timing(calls: list[Call], timings: Mapping[str, ExecutionTiming], joined) -> list[Call]:
    """Enrich canonical calls only after the authoritative unique journal join."""
    from ssgrep.accounting.views.lookups import link_call_timing

    return [
        replace(
            call, timing=link_call_timing(call.requests, joined.owned, joined.ambiguous, timings)
        )
        for call in calls
    ]


def _mcp_request(inv_id: str, call: Classification, hmac_key: bytes | None, **base) -> Request:
    invocation = call.invocation
    assert invocation is not None
    operation = invocation.operations[0]
    arguments = invocation.arguments if isinstance(invocation.arguments, dict) else {}
    text = arguments.get("query") if operation == "search" else arguments.get("ref")
    where = arguments.get("where") if operation == "search" else None
    text = text if isinstance(text, str) else None
    where = where if isinstance(where, str) else None
    return Request(
        invocation_id=inv_id,
        index=0,
        operation=operation,
        transport="mcp",
        fingerprint=request_fingerprint(hmac_key, operation, text, where),
        correlation=None,
        **base,
    )


def _shell_requests(
    inv_id: str, call: Classification, hmac_key: bytes | None, **base
) -> list[Request]:
    invocation = call.invocation
    assert invocation is not None
    command = base.pop("command")
    correlation = correlation_in(command)
    out: list[Request] = []
    for index, argv in enumerate(invocation.argv):
        operation, text, where = parse_argv(argv)
        out.append(
            Request(
                invocation_id=inv_id,
                index=index,
                operation=operation,
                transport="cli",
                fingerprint=request_fingerprint(hmac_key, operation, text, where),
                correlation=correlation,
                **base,
            )
        )
    return out


def collect_calls(
    timelines: Timelines,
    lineage: Lineage,
    scoper: Scoper,
    hmac_key: bytes | None,
    verifiers: frozenset[str],
) -> CallSet:
    out = CallSet()
    request_key_id = key_id(hmac_key)
    for key in sorted(timelines.sessions):
        line = timelines.sessions[key]
        solver_root = lineage.root(key)
        actor = "verifier" if key_str(key) in verifiers or solver_root in verifiers else "solver"
        for record in line.records:
            if record.kind != "tool_call":
                continue
            if record.ev.native_identity in timelines.conflicted:
                out.counts["conflicted_lookup_call"] += 1
                continue
            result = classify_evidence(record.ev)
            if result.invocation_class is not InvocationClass.TRANSCRIPT_ONLY:
                out.counts[result.invocation_class.value] += 1
                continue
            invocation = result.invocation
            assert invocation is not None
            if not set(invocation.operations) & LOOKUPS:
                out.counts["non_lookup"] += 1
                continue
            payload_cwd = record.ev.payload.get("cwd")
            cwd = (
                payload_cwd
                if isinstance(payload_cwd, str) and payload_cwd
                else line.cwd_at(record.seq)
            )
            in_scope = native_in_scope(record, timelines, scoper)
            out.out_of_scope += not in_scope
            result_record = line.result_for(record)
            inv_id = invocation_id(record)
            window = window_of(record.ts, result_record.ts if result_record else None)
            interp = interpret(result_record.ev.payload) if result_record else None
            base = {
                "cwd": cwd,
                "cwd_trusted": bool(cwd) and in_scope,
                "fingerprint_key_id": request_key_id,
                "window": window,
                "capacity": 1 if invocation.cardinality in SINGLE else None,
                "result_refs": (
                    frozenset(c.ref for c in interp.cards) if interp and interp.structured else None
                ),
                "result_outcome": (
                    interp.outcome
                    if interp
                    and interp.basis in RESULT_BASES
                    and interp.outcome is not Outcome.UNKNOWN
                    else None
                ),
            }
            if invocation.route == "shell_cli":
                command = call_text(record.ev.payload)
                requests = tuple(_shell_requests(inv_id, result, hmac_key, command=command, **base))
                shape = shell_shape(command).kind
            else:
                requests = (_mcp_request(inv_id, result, hmac_key, **base),)
                shape = "pure"
            out.calls.append(
                Call(
                    inv_id,
                    key,
                    record,
                    result_record,
                    result,
                    interp,
                    shape,
                    requests,
                    cwd,
                    record.ts,
                    lineage.task_of(key, record.seq),
                    actor,
                    in_scope,
                )  # fmt: skip
            )
    out.calls.sort(key=lambda c: (c.key, c.call.seq))
    return out


def shown_refs(call: Call) -> tuple[str, ...]:
    """Refs a ``show`` request in this call opens (empty when dynamic or not a show)."""
    invocation = call.classification.invocation
    assert invocation is not None
    if invocation.route != "shell_cli":
        args = invocation.arguments
        ref = (
            args.get("ref")
            if isinstance(args, dict) and invocation.operations == ("show",)
            else None
        )
        return (ref,) if isinstance(ref, str) else ()
    refs = []
    for argv in invocation.argv:
        if argv[0] == "show":
            _, text, _ = parse_argv(argv)
            if text:
                refs.append(text)
    return tuple(refs)
