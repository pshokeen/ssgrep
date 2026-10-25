"""Strict execution-to-request reconciliation and result interpretation."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ssgrep.accounting.capture import fingerprint as fp
from ssgrep.accounting.ledger.contracts import (
    Attribution,
    ExecutionEvent,
    Exposure,
    Operation,
    Outcome,
    Phase,
    Quantity,
    digest_of,
)
from ssgrep.accounting.readers.shell_scan import scan_shell
from ssgrep.accounting.readers.timeline import parse_output_json, ts_ms, unwrap_output


def contradictions(execution: ExecFact, request: Request) -> tuple[str, ...]:
    """Hard evidence mismatches cannot be overridden by correlation or timing."""
    reasons = []
    if execution.conflicted:
        reasons.append("journal_conflict")
    if execution.operation != request.operation:
        reasons.append("operation_mismatch")
    if execution.transport in ("cli", "mcp") and execution.transport != request.transport:
        reasons.append("transport_mismatch")
    if (
        execution.correlation_id
        and request.correlation
        and execution.correlation_id != request.correlation
    ):
        reasons.append("correlation_mismatch")
    if (
        execution.fingerprint_key_id is not None
        and request.fingerprint_key_id is not None
        and execution.fingerprint_key_id != request.fingerprint_key_id
    ):
        reasons.append("key_mismatch")
    elif request.fingerprint is not None and request.fingerprint != execution.fingerprint:
        reasons.append(
            "query_mismatch"
            if execution.fingerprint_key_id is not None and request.fingerprint_key_id is not None
            else "fingerprint_mismatch"
        )
    if not request.cwd:
        reasons.append("unknown_scope")
    elif not request.cwd_trusted:
        reasons.append("untrusted_cwd")
    elif not os.path.isabs(request.cwd) or "\0" in request.cwd:
        reasons.append("untrusted_cwd")
    if execution.cwd and request.cwd and request.cwd_trusted and "untrusted_cwd" not in reasons:
        if execution.scope_basis == "unknown_scope":
            reasons.append("untrusted_cwd")
        elif execution.cwd != os.path.realpath(request.cwd):
            reasons.append("cwd_mismatch")
    shown = request.result_refs
    if shown is not None and execution.refs and not set(execution.refs) <= shown:
        if execution.refs_complete or execution.operation == "show":
            reasons.append("result_mismatch")
    return tuple(reasons)


def rejection(execution: ExecFact, request: Request, *, explicit: bool = False) -> tuple[str, ...]:
    """A supported native workspace and compatible query/result are required.

    Unknown journal scope can join trusted native call metadata, but only the
    reducer's same-profile policy ownership check can authorize that workspace.
    Explicit correlation can replace a missing clock, never contradictory facts.
    """
    reasons = contradictions(execution, request)
    if reasons or explicit:
        return reasons
    if (
        request.window is None
        or request.window[1] < request.window[0]
        or execution.ts_ms is None
        or not execution.timing_known
    ):
        return ("clock_uncertainty",)
    if not request.window[0] <= execution.ts_ms <= request.window[1]:
        return ("outside_window",)
    shown = request.result_refs
    refs_ok = shown is not None and bool(execution.refs)
    if request.fingerprint is not None or (refs_ok and execution.refs_complete):
        return ()
    return ("insufficient_evidence",)


SLACK_MS = 5_000
NO_RESULT_WINDOW_MS = 300_000
WALL_MONO_TOLERANCE_MS = 5_000
BUCKET_MS = 60_000
MAX_BUCKETS = 240
CORRELATION_RE = re.compile(r"\bSSGREP_CORRELATION_ID=([A-Za-z0-9._:-]{1,80})")
VALUE_OPTIONS = frozenset({"--limit", "--token-budget", "--where", "--title", "--body"})
SINGLE = frozenset({"single_call", "single_text"})


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
    fingerprint_key_id: str | None = None
    scope_basis: str = "legacy"
    conflicted: bool = False


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
        timing = start_ms is not None
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
                cwd=None
                if str(base.transport) == "mcp" and base.scope_basis == "legacy"
                else base.cwd,
                fingerprint=base.fingerprint,
                correlation_id=(base.correlation_id or (done.correlation_id if done else None))
                if str(base.transport) != "mcp" or base.scope_basis != "legacy"
                else None,
                outcome=outcome,
                outcome_basis=basis,
                refs=carrier.refs,
                refs_complete=not carrier.refs_omitted,
                truncated=truncated,
                rendered=rendered,
                timing_known=timing,
                fingerprint_key_id=base.fingerprint_key_id,
                scope_basis="unknown_scope"
                if str(base.transport) == "mcp" and base.scope_basis == "legacy"
                else base.scope_basis,
                conflicted=any(
                    len({digest_of(item) for item in items}) > 1
                    for items in phases[execution_id].values()
                ),
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
    fingerprint_key_id: str | None = None
    cwd_trusted: bool = False

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
    reasons: tuple[str, ...] = ()


def _bucket_index(
    requests: list[Request],
) -> tuple[dict[tuple[str, int], list[Request]], list[Request]]:
    buckets: dict[tuple[str, int], list[Request]] = defaultdict(list)
    wide: list[Request] = []
    for request in requests:
        if request.window is None or request.window[1] < request.window[0]:
            continue
        first, last = request.window[0] // BUCKET_MS, request.window[1] // BUCKET_MS
        if last - first > MAX_BUCKETS:
            wide.append(request)
            continue
        for bucket in range(first, last + 1):
            buckets[(request.operation, bucket)].append(request)
    return buckets, wide


def reconcile(executions: list[ExecFact], requests: list[Request]) -> dict[str, Link]:
    """Attribution of every execution; duplicate copies never change capacity."""
    variants: dict[tuple[str, int], dict[str, Request]] = defaultdict(dict)
    for request in requests:
        variants[request.key][digest_of(request)] = request
    conflicts = {key for key, items in variants.items() if len(items) > 1}
    unique = [request for items in variants.values() for request in items.values()]
    by_correlation: dict[str, list[Request]] = defaultdict(list)
    unknown_clock: dict[str, list[Request]] = defaultdict(list)
    for request in unique:
        if request.correlation:
            by_correlation[request.correlation].append(request)
        if request.window is None or request.window[1] < request.window[0]:
            unknown_clock[request.operation].append(request)
    buckets, wide = _bucket_index(unique)
    links: dict[str, Link] = {}
    claims: dict[tuple[str, int], list[str]] = defaultdict(list)
    capacity = {request.key: request.capacity for request in unique if request.key not in conflicts}
    for execution in sorted(executions, key=lambda e: e.execution_id):
        rejected: set[str] = set()
        found: dict[tuple[str, int], Request] = {}
        explicit = False
        for request in by_correlation.get(execution.correlation_id or "", ()):
            reasons = rejection(execution, request, explicit=True)
            rejected.update(reasons)
            if not reasons:
                found[request.key] = request
                explicit = True
        if not found and execution.ts_ms is not None and execution.timing_known:
            bucket = execution.ts_ms // BUCKET_MS
            pool = list(buckets.get((execution.operation, bucket), ()))
            pool += [r for r in wide if r.operation == execution.operation]
            pool += unknown_clock.get(execution.operation, [])
            for request in pool:
                reasons = rejection(execution, request)
                rejected.update(reasons)
                if not reasons:
                    found[request.key] = request
        elif not found:
            rejected.add("clock_uncertainty")
        keys = tuple(sorted(found))
        if not keys:
            if execution.scope_basis == "unknown_scope":
                rejected.add("unknown_scope")
            if execution.conflicted:
                rejected.add("journal_conflict")
            links[execution.execution_id] = Link(
                Attribution.UNATTRIBUTED,
                reasons=tuple(sorted(rejected or {"no_compatible_request"})),
            )
        elif set(keys) & conflicts:
            links[execution.execution_id] = Link(
                Attribution.AMBIGUOUS, (), keys, ("request_conflict",)
            )
        elif len(keys) == 1:
            attribution = Attribution.EXPLICIT if explicit else Attribution.INFERRED_UNIQUE
            links[execution.execution_id] = Link(attribution, keys)
            claims[keys[0]].append(execution.execution_id)
        else:
            fingerprints = {request.fingerprint for request in found.values()}
            reason = (
                "equal_query_concurrency"
                if len(fingerprints) == 1 and None not in fingerprints
                else "request_ambiguity"
            )
            if explicit:
                reason = "correlation_ambiguity"
            links[execution.execution_id] = Link(Attribution.AMBIGUOUS, (), keys, (reason,))
    for key, owners in claims.items():
        if capacity.get(key) == 1 and len(owners) > 1:
            for owner in owners:
                links[owner] = Link(Attribution.AMBIGUOUS, (), (key,), ("capacity_ambiguity",))
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


EXIT_RE = re.compile(
    r"(?:\bexit(?:ed)?(?:\s+with)?(?:\s+(?:code|status))?\s*[:=]?\s*|\bexit_code[\"']?\s*[:=]\s*"
    r"|\[exit(?:\s+code)?\s*[:=]?\s*)(-?\d{1,3})\b",
    re.IGNORECASE,
)
TRUNCATION_RE = re.compile(
    r"\[(?:output|content|result|response)?\s*truncated|truncated (?:by|to|at|after)\b"
    r"|\.\.\.\s*\[?\d+\s+(?:more\s+)?(?:lines|chars|characters|bytes)\s+(?:truncated|omitted)"
    r"|\b(?:output|content|response) (?:was )?truncated\b|\[Showing lines \d+-\d+ of \d+",
    re.IGNORECASE,
)
USAGE_ERROR_RE = re.compile(
    r"No such (?:option|command)|Missing (?:argument|option)|Invalid value for|Usage: ssgrep"
    r"|Got unexpected extra argument|unrecognized arguments",
)
NOT_FOUND_RE = re.compile(r"^Episode not found:")
FILTERS = frozenset(
    {"head", "tail", "cat", "less", "more", "jq", "wc", "grep", "sed", "awk", "cut", "sort",
     "uniq", "tee", "tr", "nl", "bat", "column", "fold", "rg"}
)  # fmt: skip
STRUCTURAL = frozenset(
    {"for", "while", "until", "do", "done", "then", "fi", "if", "else", "elif", "in", "case",
     "esac", "{", "}", "!"}
)  # fmt: skip
OPERATORS = ("&&", "||", "|", ";", "&", "\n")


@dataclass(frozen=True)
class Card:
    """One result card or episode: its ref plus the text the model saw for it."""

    ref: str
    text: str


@dataclass(frozen=True)
class Interpretation:
    """What a tool result means, independent of its transport success."""

    outcome: Outcome
    basis: str  # native_flag|error_payload|exit_status|structured|transport_only|none
    explicit_success: bool  # exit status 0 or a recognized successful payload
    native_error: bool | None
    exit_code: int | None
    text: str
    chars: int
    truncated: bool
    cards: tuple[Card, ...] = ()
    structured: bool = False

    @property
    def is_error(self) -> bool:
        return self.outcome is Outcome.ERROR

    @property
    def action_ok(self) -> bool:
        """A result carrying no error evidence in any recognized form (not proof of success)."""
        return self.outcome not in (Outcome.ERROR, Outcome.INTERRUPTED)


def _cards(obj: dict[str, Any]) -> tuple[Card, ...]:
    out: list[Card] = []
    results = obj.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and isinstance(item.get("ref"), str):
                parts = [
                    str(item[key])
                    for key in ("title", "excerpt", "agent_description")
                    if isinstance(item.get(key), str)
                ]
                files = item.get("files_touched")
                if isinstance(files, list):
                    parts.extend(str(f) for f in files if isinstance(f, str))
                out.append(Card(item["ref"], "\n".join(parts)))
    elif isinstance(obj.get("episode_id"), str):
        parts = [str(obj[k]) for k in ("title", "prompt_text", "response_text") if obj.get(k)]
        files = obj.get("files_touched")
        if isinstance(files, list):
            parts.extend(str(f) for f in files if isinstance(f, str))
        out.append(Card(obj["episode_id"], "\n".join(parts)))
    return tuple(out)


def _payload_error(obj: Any) -> str | None:
    """A recognized error payload's category, else None."""
    if not isinstance(obj, dict):
        return None
    if obj.get("ok") is False:
        return "cli_error"
    error = obj.get("error")
    if isinstance(error, str) and error and "results" not in obj and "prompt_text" not in obj:
        return "not_found" if NOT_FOUND_RE.match(error) else "tool_error"
    return None


def interpret(payload: dict[str, Any]) -> Interpretation:
    """Interpret one ``tool_result`` payload (``output``, ``is_error``, ``exit_code``)."""
    raw = payload.get("output")
    text = raw if isinstance(raw, str) else ""
    flag = payload.get("is_error")
    native: bool | None = flag if isinstance(flag, bool) else None
    code = payload.get("exit_code")
    exit_code = code if isinstance(code, int) and not isinstance(code, bool) else None
    visible_chars = len(text)
    text, exit_code = unwrap_output(text, exit_code)
    if exit_code is None:
        found = EXIT_RE.search(text[-600:]) if text else None
        exit_code = int(found.group(1)) if found else None
    obj = parse_output_json(text)
    if isinstance(obj, dict) and obj.get("ok") is True and isinstance(obj.get("data"), dict):
        obj = obj["data"]  # CLI --json envelope
    cards = _cards(obj) if isinstance(obj, dict) else ()
    truncated = bool(TRUNCATION_RE.search(text[-2000:] + text[:400])) if text else False
    if isinstance(obj, dict):
        truncated = truncated or any(
            obj.get(k) is True for k in ("truncated", "prompt_truncated", "response_truncated")
        )
        truncated = truncated or obj.get("excerpts_truncated") is True
        omitted = obj.get("omitted_count")
        truncated = truncated or (isinstance(omitted, int) and omitted > 0)
    problem = _payload_error(obj)
    structured = isinstance(obj, dict) and (
        isinstance(obj.get("results"), list) or "episode_id" in obj or "prompt_text" in obj
    )
    common: dict[str, Any] = {
        "native_error": native,
        "exit_code": exit_code,
        "text": text,
        "chars": visible_chars,
        "truncated": truncated,
        "cards": cards,
        "structured": structured,
    }
    if problem == "not_found":
        return Interpretation(Outcome.EMPTY, "error_payload", False, **common)
    if native is True:
        return Interpretation(Outcome.ERROR, "native_flag", False, **common)
    if problem is not None:
        return Interpretation(Outcome.ERROR, "error_payload", False, **common)
    if exit_code not in (None, 0):
        return Interpretation(Outcome.ERROR, "exit_status", False, **common)
    if structured and isinstance(obj, dict):
        results = obj.get("results")
        empty = isinstance(results, list) and not results
        outcome = Outcome.EMPTY if empty else Outcome.OK
        return Interpretation(outcome, "structured", True, **common)
    if native is False or exit_code == 0:
        return Interpretation(Outcome.UNKNOWN, "transport_only", exit_code == 0, **common)
    return Interpretation(Outcome.UNKNOWN, "none", False, **common)


def is_usage_error(interpretation: Interpretation) -> bool:
    """A CLI argument failure that happened before the service API was entered."""
    return interpretation.is_error and bool(USAGE_ERROR_RE.search(interpretation.text))


@dataclass(frozen=True)
class ShellShape:
    """How much of a shell result is attributable to the ssgrep invocation(s)."""

    kind: str  # pure | filtered | mixed
    segments: int


def _split(command: str) -> list[tuple[str, str]]:
    """Quote-aware split into ``(operator-before, segment)``; no execution, no expansion."""
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    operator = ""
    quote = ""
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                buf.append(command[i + 1])
                i += 2
                continue
            quote = "" if c == quote else quote
            i += 1
            continue
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        matched = next((op for op in OPERATORS if command.startswith(op, i)), None)
        if matched and not (
            matched == "&" and (command[i - 1 : i] in (">", "<") or buf[-1:] == [">"])
        ):
            parts.append((operator, "".join(buf).strip()))
            buf, operator = [], matched
            i += len(matched)
            continue
        buf.append(c)
        i += 1
    parts.append((operator, "".join(buf).strip()))
    return [(op, seg) for op, seg in parts if seg]


def shell_shape(command: str) -> ShellShape:
    """Classify a command line: only-ssgrep, ssgrep through view filters, or mixed."""
    scan = scan_shell(command)
    pieces = _split(command)
    if scan.argvs and not (scan.has_other_commands or scan.unsupported):
        return ShellShape("pure", len(pieces))
    kind = "pure"
    for operator, segment in pieces:
        words = segment.split()
        head = next((w for w in words if "=" not in w.split("/")[-1]), "") if words else ""
        base = head.rsplit("/", 1)[-1]
        segment_scan = scan_shell(segment)
        if segment_scan.argvs:
            if segment_scan.has_other_commands or segment_scan.unsupported:
                kind = "mixed"
            continue
        if base in STRUCTURAL:
            rest = " ".join(words[1:])
            if base in {"do", "then", "else", "elif"} and rest and not scan_shell(rest).argvs:
                kind = "mixed"
            continue
        if operator == "|" and base in FILTERS:
            kind = "mixed" if kind == "mixed" else "filtered"
            continue
        kind = "mixed"
    return ShellShape(kind, len(pieces))


def exposure_of(shape: str | None, interpretation: Interpretation | None) -> Exposure:
    """Exposure state of one invocation's output (see module docstring)."""
    if interpretation is None:
        return Exposure.UNAVAILABLE
    if interpretation.truncated or shape in ("filtered", "mixed"):
        return Exposure.PARTIAL
    return Exposure.OBSERVED
