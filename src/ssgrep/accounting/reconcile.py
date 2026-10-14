"""Join journal executions to transcript requests without guessing (tasks 5.1, 5.2).

Order of authority: (1) a trusted propagated correlation id with compatible
operation/arguments; otherwise (2) a UNIQUELY compatible request. Compatibility
needs timing evidence AND at least one of {request fingerprint, cwd, result
evidence}; a timestamp alone is never a join, a wall-clock discontinuity makes
timing unknown (no join without explicit correlation), and two compatible
candidates stay ``ambiguous`` -- the nearest timestamp or first candidate is never
chosen. One shell invocation yields ZERO or more executions: loop text never
manufactures a count, and executions with no request stay recorded and
``unattributed``.

Everything is a pure function of its inputs and independent of input order.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from ssgrep.accounting import fingerprint as fp
from ssgrep.accounting.contracts import (
    Attribution,
    ExecutionEvent,
    Operation,
    Outcome,
    Phase,
    Quantity,
    digest_of,
)
from ssgrep.accounting.timeline import ts_ms

SLACK_MS = 5_000
NO_RESULT_WINDOW_MS = 300_000
WALL_MONO_TOLERANCE_MS = 5_000
BUCKET_MS = 60_000
MAX_BUCKETS = 240
CORRELATION_RE = re.compile(r"\bSSGREP_CORRELATION_ID=([A-Za-z0-9._:-]{1,80})")
VALUE_OPTIONS = frozenset({"--limit", "--token-budget", "--where", "--title", "--body"})
SINGLE = frozenset({"single_call", "single_text"})


@lru_cache(maxsize=4096)
def _real(path: str) -> str:
    return os.path.realpath(path)


# ------------------------------------------------------------------ journal side


@dataclass(frozen=True)
class ExecFact:
    """One recorded execution folded from its started/completed/rendered phases."""

    execution_id: str
    operation: str
    transport: str
    ts: str
    ts_ms: int | None
    end_ms: int | None
    cwd: str | None
    fingerprint: str
    correlation_id: str | None
    outcome: Outcome
    outcome_basis: str
    refs: tuple[str, ...]
    refs_complete: bool
    truncated: bool | None
    rendered: Quantity
    timing_known: bool


def _pick(events: list[ExecutionEvent]) -> ExecutionEvent:
    return min(events, key=digest_of)


def fold_events(events: list[ExecutionEvent]) -> list[ExecFact]:
    """One fact per ``execution_id``; identical phases collapse deterministically."""
    phases: dict[str, dict[Phase, list[ExecutionEvent]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        phases[event.execution_id][event.phase].append(event)
    facts: list[ExecFact] = []
    for execution_id in sorted(phases):
        by = {phase: _pick(items) for phase, items in phases[execution_id].items()}
        started, done, shown = (
            by.get(Phase.STARTED),
            by.get(Phase.COMPLETED),
            by.get(Phase.RENDERED),
        )
        base = started or done or shown
        assert base is not None
        start_ms = ts_ms(started.ts_utc) if started else None
        end_ms = ts_ms(done.ts_utc) if done else None
        timing = True
        if started and done:
            wall = (end_ms or 0) - (start_ms or 0)
            mono = done.mono_ns // 1_000_000 if done.mono_ns is not None else None
            timing = (
                start_ms is not None
                and end_ms is not None
                and wall >= 0
                and (mono is None or abs(wall - mono) <= WALL_MONO_TOLERANCE_MS)
            )
        if done is not None and done.outcome is not None:
            outcome, basis = done.outcome, "journal"
        else:
            outcome, basis = Outcome.UNKNOWN, "journal_incomplete"
        rendered = (
            shown.rendered_bytes
            if shown is not None and shown.rendered_bytes is not None
            else Quantity.unavailable("bytes", "no_render_event")
        )
        carrier = done or shown or base
        truncated = carrier.truncated if carrier.truncated is not None else base.truncated
        facts.append(
            ExecFact(
                execution_id=execution_id,
                operation=str(base.operation),
                transport=str(base.transport),
                ts=base.ts_utc,
                ts_ms=start_ms if start_ms is not None else ts_ms(base.ts_utc),
                end_ms=end_ms,
                cwd=base.cwd,
                fingerprint=base.fingerprint,
                correlation_id=base.correlation_id or (done.correlation_id if done else None),
                outcome=outcome,
                outcome_basis=basis,
                refs=carrier.refs,
                refs_complete=not carrier.refs_omitted,
                truncated=truncated,
                rendered=rendered,
                timing_known=timing,
            )
        )
    return facts


# ---------------------------------------------------------------- transcript side


@dataclass(frozen=True)
class Request:
    """One ssgrep request inside a recognized invocation (a shell line has several)."""

    invocation_id: str
    index: int
    operation: str
    transport: str  # "mcp" for structured tool calls, "cli" for shell command lines
    fingerprint: str | None
    cwd: str | None
    correlation: str | None
    window: tuple[int, int] | None
    capacity: int | None  # 1 when text proves one call; None when the count is unknowable
    result_refs: frozenset[str] | None
    result_outcome: Outcome | None

    @property
    def key(self) -> tuple[str, int]:
        return (self.invocation_id, self.index)


def parse_argv(argv: tuple[str, ...]) -> tuple[str, str | None, str | None]:
    """``(operation, primary text or None, where or None)`` of one CLI request.

    The primary text is the search query / show ref / note title; it is None when
    absent or dynamic (shell expansion), in which case no fingerprint exists.
    """
    operation = argv[0]
    positional: str | None = None
    where: str | None = None
    title: str | None = None
    rest = list(argv[1:])
    i = 0
    while i < len(rest):
        word = rest[i]
        if word.startswith("--"):
            name, eq, value = word.partition("=")
            if name in VALUE_OPTIONS:
                if not eq:
                    i += 1
                    value = rest[i] if i < len(rest) else ""
                if name == "--where":
                    where = value
                elif name == "--title":
                    title = value
            i += 1
            continue
        if positional is None and not word.startswith("-"):
            positional = word
        i += 1
    text = title if operation == "note" else positional
    dynamic = any(("$" in part or "`" in part) for part in (text or "", where or ""))
    if dynamic:
        return operation, None, where
    return operation, text, where


def request_fingerprint(
    key: bytes | None, operation: str, text: str | None, where: str | None
) -> str | None:
    if key is None or text is None or operation not in {o.value for o in Operation}:
        return None
    return fp.request_fingerprint(key, Operation(operation), query=text, where=where)


def correlation_in(command: str) -> str | None:
    found = CORRELATION_RE.findall(command)
    return found[0] if len(set(found)) == 1 else None


# ------------------------------------------------------------------- the join


@dataclass(frozen=True)
class Link:
    """The reconciliation outcome of one execution."""

    attribution: Attribution
    requests: tuple[tuple[str, int], ...] = ()
    candidates: tuple[tuple[str, int], ...] = ()  # ambiguous: every compatible request


def _cwd_comparable(execution: ExecFact, request: Request) -> bool:
    """Only a CLI process shares its caller's cwd; an MCP server has its own."""
    return bool(
        execution.transport == "cli"
        and request.transport == "cli"
        and execution.cwd
        and request.cwd
    )


def _contradicts(execution: ExecFact, request: Request) -> bool:
    if execution.operation != request.operation:
        return True
    if execution.transport in ("cli", "mcp") and execution.transport != request.transport:
        return True
    if request.fingerprint is not None and request.fingerprint != execution.fingerprint:
        return True
    return _cwd_comparable(execution, request) and _real(execution.cwd or "") != _real(
        request.cwd or ""
    )


def _supports(execution: ExecFact, request: Request) -> bool:
    """Compatible by timing plus at least one non-time evidence point."""
    if _contradicts(execution, request):
        return False
    # The inference index admits only requests and executions with known timing.
    assert request.window is not None and execution.ts_ms is not None and execution.timing_known
    low, high = request.window
    if not low <= execution.ts_ms <= high:
        return False
    shown = request.result_refs
    refs_ok = shown is not None and bool(execution.refs)
    if refs_ok and shown is not None and not set(execution.refs) <= shown:
        if execution.refs_complete or execution.operation == "show":
            return False
    evidence = request.fingerprint is not None
    evidence = evidence or _cwd_comparable(execution, request)
    evidence = evidence or (refs_ok and execution.refs_complete)
    if request.result_outcome is not None and execution.outcome_basis == "journal":
        evidence = evidence or request.result_outcome is execution.outcome
    return evidence


def _bucket_index(
    requests: list[Request],
) -> tuple[dict[tuple[str, int], list[Request]], list[Request]]:
    buckets: dict[tuple[str, int], list[Request]] = defaultdict(list)
    wide: list[Request] = []
    for request in requests:
        if request.window is None:
            continue
        first, last = request.window[0] // BUCKET_MS, request.window[1] // BUCKET_MS
        if last - first > MAX_BUCKETS:
            wide.append(request)
            continue
        for bucket in range(first, last + 1):
            buckets[(request.operation, bucket)].append(request)
    return buckets, wide


def reconcile(executions: list[ExecFact], requests: list[Request]) -> dict[str, Link]:
    """Attribution of every execution; order of both inputs is irrelevant."""
    by_correlation: dict[str, list[Request]] = defaultdict(list)
    for request in requests:
        if request.correlation:
            by_correlation[request.correlation].append(request)
    buckets, wide = _bucket_index(requests)
    links: dict[str, Link] = {}
    claims: dict[tuple[str, int], list[str]] = defaultdict(list)
    capacity = {request.key: request.capacity for request in requests}
    for execution in sorted(executions, key=lambda e: e.execution_id):
        explicit = [
            r
            for r in by_correlation.get(execution.correlation_id or "", ())
            if not _contradicts(execution, r)
        ]
        if explicit:
            keys = tuple(sorted(r.key for r in explicit))
            if len(keys) == 1:
                links[execution.execution_id] = Link(Attribution.EXPLICIT, keys)
                claims[keys[0]].append(execution.execution_id)
            else:
                links[execution.execution_id] = Link(Attribution.AMBIGUOUS, (), keys)
            continue
        found: dict[tuple[str, int], Request] = {}
        if execution.ts_ms is not None and execution.timing_known:
            bucket = execution.ts_ms // BUCKET_MS
            pool = list(buckets.get((execution.operation, bucket), ()))
            pool += [r for r in wide if r.operation == execution.operation]
            for request in pool:
                if _supports(execution, request):
                    found[request.key] = request
        keys = tuple(sorted(found))
        if not keys:
            links[execution.execution_id] = Link(Attribution.UNATTRIBUTED)
        elif len(keys) == 1:
            links[execution.execution_id] = Link(Attribution.INFERRED_UNIQUE, keys)
            claims[keys[0]].append(execution.execution_id)
        else:
            links[execution.execution_id] = Link(Attribution.AMBIGUOUS, (), keys)
    for key, owners in claims.items():
        if capacity.get(key) == 1 and len(owners) > 1:  # one call cannot own two executions
            for owner in owners:
                links[owner] = Link(Attribution.AMBIGUOUS, (), (key,))
    return links


@dataclass
class Joined:
    """Per request: the executions that attribute to it and those that merely could."""

    owned: dict[tuple[str, int], list[str]] = field(default_factory=dict)
    ambiguous: dict[tuple[str, int], list[str]] = field(default_factory=dict)


def join_index(links: dict[str, Link]) -> Joined:
    joined = Joined()
    for execution_id in sorted(links):
        link = links[execution_id]
        for key in link.requests:
            joined.owned.setdefault(key, []).append(execution_id)
        for key in link.candidates:
            joined.ambiguous.setdefault(key, []).append(execution_id)
    return joined


def window_of(call_ms: int | None, result_ms: int | None) -> tuple[int, int] | None:
    """Time window of one call; None (unknown timing) when clocks are missing or reversed."""
    if call_ms is None:
        return None
    if result_ms is None:
        return (call_ms - SLACK_MS, call_ms + NO_RESULT_WINDOW_MS)
    if result_ms < call_ms:
        return None
    return (call_ms - SLACK_MS, result_ms + SLACK_MS)
