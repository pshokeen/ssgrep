"""Recognize ssgrep invocations in raw tool calls without trusting names or text.

* Structured MCP calls are recognized by *operation schema plus routing*: a tool
  name (or namespace + name, or omp ``write`` to ``xd://mcp__*``) whose operation
  token sits on a name boundary AND whose arguments match that operation's
  schema. No server prefix is hard-coded, so renamed servers are recognized and
  lookalike names or foreign argument shapes are not.
* Shell CLI calls are recognized by command position only (see ``shell_scan``).
* Documentation reads (``read xd://...``), quoted briefs, source code, heredocs
  and fixtures are ``mention``; anything else stays ``unclassified``.
* Nothing here executes, fetches or opens anything named by a transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ssgrep.accounting.contracts import InvocationClass, RawEvidence
from ssgrep.accounting.readers.shell_scan import scan_shell, scan_words
from ssgrep.accounting.readers.values import clean, loads

UNKNOWN = "unknown"
SINGLE_CALL = "single_call"
SINGLE_TEXT = "single_text"
MCP_OPERATIONS = {
    "search_sessions": "search",
    "show_session": "show",
    "index_status": "index_status",
}
SCHEMAS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "search": (frozenset({"query"}), frozenset({"query", "limit", "where"})),
    "show": (frozenset({"ref"}), frozenset({"ref", "max_chars"})),
    "index_status": (frozenset(), frozenset()),
}
ARG_TYPES = {"query": str, "ref": str, "limit": int, "max_chars": int}
TAIL = re.compile(r"(?:^|[_\-.:/])(search_sessions|show_session|index_status)$")
XD_MCP = re.compile(r"^xd://mcp__(.+)$")
MENTION = re.compile(
    r"\bssgrep\s+(?:-\S+\s+)*(?:search|show|note)\b|search_sessions|show_session|index_status"
)
SHELL_TOOLS = frozenset(
    {"bash", "shell", "exec_command", "local_shell", "sh", "zsh", "terminal", "shell_command"}
)
COMMAND_KEYS = ("command", "cmd", "script")


@dataclass(frozen=True)
class Invocation:
    """One recognized invocation; ``cardinality`` says whether its run count is knowable."""

    operations: tuple[str, ...]
    route: str
    arguments: Any
    cardinality: str
    text_occurrences: int = 1
    argv: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Classification:
    invocation_class: InvocationClass
    invocation: Invocation | None = None
    reason: str = ""


def _unclassified(reason: str = "unknown_tool") -> Classification:
    return Classification(InvocationClass.UNCLASSIFIED, None, reason)


def _mention(reason: str) -> Classification:
    return Classification(InvocationClass.MENTION, None, reason)


def _schema_ok(operation: str, arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    required, allowed = SCHEMAS[operation]
    keys = set(arguments)
    if not required <= keys or not keys <= allowed:
        return False
    for key, value in arguments.items():
        expected = ARG_TYPES.get(key)
        if key == "where" and value is None:
            continue
        if key == "where" and not isinstance(value, str):
            return False
        if expected and (isinstance(value, bool) or not isinstance(value, expected)):
            return False
    return True


def _operation_tail(name: str) -> str | None:
    found = TAIL.search(name)
    return MCP_OPERATIONS[found.group(1)] if found else None


def _mcp(route: str, operation: str, arguments: Any) -> Classification:
    if not _schema_ok(operation, arguments):
        return _unclassified("schema_mismatch")
    invocation = Invocation((operation,), route, clean(arguments), SINGLE_CALL)
    return Classification(InvocationClass.TRANSCRIPT_ONLY, invocation)


def _mentions(arguments: Any) -> bool:
    text = json.dumps(arguments, default=str)
    return bool(MENTION.search(text))


def _xd_write(arguments: Any) -> Classification | None:
    if not isinstance(arguments, dict):
        return None
    found = XD_MCP.match(str(arguments.get("path", "")))
    if not found:
        return None
    operation = _operation_tail(found.group(1))
    if operation is None:
        return _unclassified("unknown_tool")
    content = arguments.get("content", "")
    if isinstance(content, str):
        if not content.strip():
            content = "{}"
        try:
            content = loads(content)
        except ValueError:
            return _unclassified("arguments_not_json")
    return _mcp("omp_xd_write", operation, content)


def _command_text(arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return None
    for key in COMMAND_KEYS:
        if key in arguments:
            return arguments[key]
    return None


def _shell(arguments: Any) -> Classification:
    command = _command_text(arguments)
    if isinstance(command, str):
        scan = scan_shell(command)
    elif isinstance(command, list) and all(isinstance(w, str) for w in command):
        scan = scan_words(command)
        command = " ".join(command)
    else:
        return _unclassified("no_command")
    if scan.error:
        return _unclassified(scan.error)
    if not scan.argvs:
        if scan.unsupported:
            return _unclassified("unsupported_shell_form")
        return _mention("text_mention") if MENTION.search(command) else _unclassified("no_cli_call")
    count = len(scan.argvs)
    cardinality = SINGLE_TEXT if count == 1 and not scan.uncertain else UNKNOWN
    invocation = Invocation(
        scan.operations, "shell_cli", None, cardinality, count, tuple(scan.argvs)
    )
    return Classification(InvocationClass.TRANSCRIPT_ONLY, invocation)


def classify_call(
    runtime: str, name: str | None, arguments: Any, namespace: str | None = None
) -> Classification:
    """Classify one raw tool call. ``runtime`` is part of the evidence, not the routing."""
    if not isinstance(name, str) or not name:
        return _unclassified("no_tool_name")
    lowered = name.lower()
    if lowered == "read" and isinstance(arguments, dict):
        if str(arguments.get("path", "")).startswith("xd://"):
            return _mention("documentation_read")
    if lowered == "write":
        routed = _xd_write(arguments)
        if routed is not None:
            return routed
    operation = _operation_tail(name)
    if operation is not None:
        if namespace and namespace.startswith("mcp__"):
            route = "mcp_namespace"
        else:
            route = "mcp_name" if name.startswith("mcp__") else "tool_name"
        return _mcp(route, operation, arguments)
    if lowered in SHELL_TOOLS:
        return _shell(arguments)
    if _mentions(arguments):
        return _mention("text_mention")
    return _unclassified()


def classify_evidence(evidence: RawEvidence) -> Classification:
    """Classify a ``tool_call`` evidence record from any runtime reader."""
    if evidence.kind != "tool_call":
        return _unclassified("not_a_call")
    payload = evidence.payload
    namespace = payload.get("namespace")
    return classify_call(
        evidence.runtime,
        payload.get("name"),
        payload.get("arguments"),
        namespace if isinstance(namespace, str) else None,
    )
