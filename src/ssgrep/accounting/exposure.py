"""Output exposure and semantic result interpretation (task 5.4).

Transport success is never semantic success: native error flags, exit status and
recognized error payloads override it (for example an incomplete-index error
returned with ``is_error`` false). Exposure is what the *consuming model saw*:
visible vs generated vs truncated. A lookup embedded in a mixed shell command is
only partly attributable to retrieval, and a ``| head`` wrapper hides unseen
facts, so neither is charged or credited beyond what was actually visible.
Nothing here executes or follows transcript content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ssgrep.accounting.contracts import Exposure, Outcome
from ssgrep.accounting.readers.shell_scan import scan_shell

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
MAX_PARSE_CHARS = 8 * 1024 * 1024
NATIVE_TIMING_RE = re.compile(r"\n{2,3}Wall time: (?:0|[1-9][0-9]*)(?:\.[0-9]+)? seconds")
JSON_DECODER = json.JSONDecoder()


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


def _loads(text: str) -> Any:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[" or len(stripped) > MAX_PARSE_CHARS:
        return None
    try:
        obj, end = JSON_DECODER.raw_decode(stripped)
    except (ValueError, RecursionError):
        return None
    if end == len(stripped) or NATIVE_TIMING_RE.fullmatch(stripped, end):
        return obj
    return None


def _unwrap(text: str, exit_code: int | None) -> tuple[str, int | None]:
    """Codex wraps shell output as ``{"output": ..., "metadata": {"exit_code": n}}``."""
    obj = _loads(text)
    if isinstance(obj, dict) and isinstance(obj.get("output"), str):
        meta = obj.get("metadata")
        code = meta.get("exit_code") if isinstance(meta, dict) else None
        if isinstance(code, int) and not isinstance(code, bool):
            exit_code = code
        return obj["output"], exit_code
    return text, exit_code


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
    text, exit_code = _unwrap(text, exit_code)
    if exit_code is None:
        found = EXIT_RE.search(text[-600:]) if text else None
        exit_code = int(found.group(1)) if found else None
    obj = _loads(text)
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
        "chars": len(text),
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
