"""Recognized transcript lookups: calls, results, requests, and exposures (pure).

Turns timelines into :class:`Call` records keyed by native invocation identity.
Copied wrappers never create another call; genuine native executions retain their
own identities even when arguments/results match. Mentions and unclassified tool
calls are only counted; they never become executions, exposures, or debits.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ssgrep.accounting.contracts import InvocationClass, Outcome, TaskId, digest_of
from ssgrep.accounting.exposure import Interpretation, interpret, shell_shape
from ssgrep.accounting.fingerprint import key_id
from ssgrep.accounting.identity_claims import native_in_scope
from ssgrep.accounting.lineage import Lineage
from ssgrep.accounting.lookup_time import ExecutionTiming, LookupTiming, link_call_timing
from ssgrep.accounting.readers.classify import Classification, classify_evidence
from ssgrep.accounting.reconcile import (
    SINGLE,
    Request,
    correlation_in,
    parse_argv,
    request_fingerprint,
    window_of,
)
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.timeline import Record, SessionKey, Timelines, call_text, key_str
from ssgrep.accounting.valuation_inputs import AtomStatus  # noqa: F401  (re-export for callers)

LOOKUPS = frozenset({"search", "show", "note"})
RESULT_BASES = frozenset({"structured", "error_payload", "native_flag"})


@dataclass(frozen=True)
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
    timing: LookupTiming = field(default_factory=LookupTiming)

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
