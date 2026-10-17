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

# Merged census and root inventory.
# Merged reader helpers: shell_scan, shell_options
import os
import re
from collections.abc import Mapping

# Merged helpers: census_model, census_repos
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ssgrep.accounting.ledger.contracts import RUNTIMES, InvocationClass, RawEvidence
from ssgrep.accounting.readers.base import EvidenceLedger, clean, loads
from ssgrep.accounting.readers.shell_scan import (
    scan_shell as _scan_shell,
    scan_words as _scan_words,
)

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
        scan = _scan_shell(command)
    elif isinstance(command, list) and all(isinstance(w, str) for w in command):
        scan = _scan_words(command)
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


__all__ = ["CensusReport", "run_census"]

CYCLE_BYTES = 32 * 1024 * 1024
SESSION_KINDS_WITHOUT_SESSION = frozenset({"lineage"})


def _readers(env: Mapping[str, str] | None) -> dict[str, Any]:
    from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader, CodexReader
    from ssgrep.accounting.readers.opencode import OpenCodeReader
    from ssgrep.accounting.readers.pi_family import OmpReader, PiReader, PrimeReader

    readers = {
        "claude": ClaudeReader(env),
        "codex": CodexReader(env),
        "omp": OmpReader(env),
        "opencode": OpenCodeReader(env),
        "pi": PiReader(env),
        "prime-agent": PrimeReader(env),
    }
    assert set(readers) == set(RUNTIMES)
    return readers


def _cwd_of(evidence) -> str | None:
    payload = evidence.payload
    for key in ("cwd", "directory"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _count(row: RuntimeCensus, evidence) -> None:
    result = classify_evidence(evidence)
    if evidence.kind != "tool_call":
        return
    bump(row.by_class, result.invocation_class.value)
    if result.invocation_class is not InvocationClass.TRANSCRIPT_ONLY:
        return
    invocation = result.invocation
    assert invocation is not None
    row.recognized_calls += 1
    bump(row.by_route, invocation.route)
    for operation in invocation.operations:
        bump(row.by_operation, operation)
    if invocation.cardinality == UNKNOWN:
        row.unknown_cardinality += 1


def _absorb(row: RuntimeCensus, result) -> None:
    for gap in result.gaps:
        bump(row.gaps, gap.reason)
    row.pending_bytes += result.pending_bytes
    row.pending_records += result.pending_records
    row.unrecognized_records += sum(result.unrecognized.values())


def _census_runtime(
    reader, ledger: EvidenceLedger, sessions: dict[tuple[str, str], str | None]
) -> RuntimeCensus:
    row = RuntimeCensus()
    for source in reader.discover():
        row.sources += 1
        state = None
        try:
            while True:
                budget = None if reader.runtime == "opencode" else CYCLE_BYTES
                result = reader.read(source, state, budget)
                _absorb(row, result)
                for evidence in result.evidence:
                    if not ledger.admit(evidence):
                        continue
                    row.evidence_records += 1
                    _count(row, evidence)
                    if evidence.kind not in SESSION_KINDS_WITHOUT_SESSION:
                        key = (evidence.runtime, evidence.session_id)
                        sessions[key] = sessions.get(key) or _cwd_of(evidence)
                state = result.state
                if not result.more:
                    break
        except OSError:
            row.unreadable_sources += 1
    return row


def _associate(sessions: dict[tuple[str, str], str | None], index: RepositoryIndex) -> Association:
    out = Association(sessions=len(sessions))
    for cwd in sessions.values():
        if not cwd:
            out.no_cwd += 1
            continue
        repo = index.match(cwd)
        if repo is None:
            out.unknown_root += 1
            continue
        out.in_repository += 1
        out.nested_repository += repo.nested
        bump(out.by_repository, repo.name)
    return out


def run_census(root: Path | None = None, env: Mapping[str, str] | None = None) -> CensusReport:
    """Census over ``root`` (default: the current directory) and every runtime store."""
    root_is_cwd = root is None
    scan_root = Path(os.getcwd()) if root is None else Path(root)
    repositories, index = inventory(scan_root)
    ledger = EvidenceLedger()
    sessions: dict[tuple[str, str], str | None] = {}
    runtimes: dict[str, RuntimeCensus] = {}
    for name, reader in _readers(env).items():
        runtimes[name] = _census_runtime(reader, ledger, sessions)
    for conflict in ledger.conflicts():
        runtimes[conflict.runtime].conflicts += 1
        bump(runtimes[conflict.runtime].conflicts_by_kind, conflict.kind)
    for final, updates in ledger.final_usage():
        runtimes[final.runtime].streamed_usage_messages += updates > 1
    for name, row in runtimes.items():
        row.duplicates = ledger.duplicates_by_runtime.get(name, 0)
    return CensusReport(root_is_cwd, repositories, runtimes, _associate(sessions, index))


CLASSES = ("transcript_only", "mention", "unclassified")
MAX_REPOSITORIES_LISTED = 500


def bump(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


@dataclass
class RuntimeCensus:
    sources: int = 0
    unreadable_sources: int = 0
    evidence_records: int = 0
    duplicates: int = 0
    conflicts: int = 0
    conflicts_by_kind: dict[str, int] = field(default_factory=dict)
    streamed_usage_messages: int = 0
    pending_records: int = 0
    pending_bytes: int = 0
    unrecognized_records: int = 0
    gaps: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CLASSES, 0))
    by_route: dict[str, int] = field(default_factory=dict)
    by_operation: dict[str, int] = field(default_factory=dict)
    recognized_calls: int = 0
    unknown_cardinality: int = 0

    @property
    def status(self) -> str:
        if self.sources == 0:
            return "no_source"
        return "partial" if self.unreadable_sources else "covered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sources": self.sources,
            "unreadable_sources": self.unreadable_sources,
            "evidence_records": self.evidence_records,
            "duplicates": self.duplicates,
            "conflicts": self.conflicts,
            "conflicts_by_kind": dict(sorted(self.conflicts_by_kind.items())),
            "streamed_usage_messages": self.streamed_usage_messages,
            "pending_records": self.pending_records,
            "pending_bytes": self.pending_bytes,
            "unrecognized_records": self.unrecognized_records,
            "gaps": dict(sorted(self.gaps.items())),
            "invocations": {
                "recognized_calls": self.recognized_calls,
                "by_class": dict(self.by_class),
                "by_route": dict(sorted(self.by_route.items())),
                "by_operation": dict(sorted(self.by_operation.items())),
                "unknown_cardinality": self.unknown_cardinality,
            },
        }


@dataclass
class RepositoryInventory:
    root_status: str = "ok"
    directories_walked: int = 0
    found: int = 0
    nested: int = 0
    worktree_files: int = 0
    symlinks_skipped: int = 0
    jsonl_files: int = 0
    jsonl_in_fixtures: int = 0
    pruned_dirs: int = 0
    unreadable_dirs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_status": self.root_status,
            "directories_walked": self.directories_walked,
            "found": self.found,
            "nested": self.nested,
            "worktree_files": self.worktree_files,
            "symlinks_skipped": self.symlinks_skipped,
            "jsonl_files": self.jsonl_files,
            "jsonl_in_fixtures": self.jsonl_in_fixtures,
            "pruned_dirs": self.pruned_dirs,
            "unreadable_dirs": self.unreadable_dirs,
        }


@dataclass
class Association:
    sessions: int = 0
    in_repository: int = 0
    nested_repository: int = 0
    unknown_root: int = 0
    no_cwd: int = 0
    by_repository: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        ranked = sorted(self.by_repository.items(), key=lambda item: (-item[1], item[0]))
        listed = dict(sorted(ranked[:MAX_REPOSITORIES_LISTED]))
        return {
            "sessions": self.sessions,
            "in_repository": self.in_repository,
            "nested_repository": self.nested_repository,
            "unknown_root": self.unknown_root,
            "no_cwd": self.no_cwd,
            "sessions_by_repository_count": listed,
            "repositories_omitted": len(ranked) - len(listed),
        }


@dataclass
class CensusReport:
    root_is_cwd: bool
    repositories: RepositoryInventory
    runtimes: dict[str, RuntimeCensus]
    association: Association

    def totals(self) -> dict[str, Any]:
        by_class = dict.fromkeys(CLASSES, 0)
        by_operation: dict[str, int] = {}
        for row in self.runtimes.values():
            for name, count in row.by_class.items():
                by_class[name] += count
            for name, count in row.by_operation.items():
                bump(by_operation, name, count)
        return {
            "sources": sum(r.sources for r in self.runtimes.values()),
            "evidence_records": sum(r.evidence_records for r in self.runtimes.values()),
            "recognized_calls": sum(r.recognized_calls for r in self.runtimes.values()),
            "loop_or_unknown_cardinality": sum(
                r.unknown_cardinality for r in self.runtimes.values()
            ),
            "by_class": by_class,
            "by_operation": dict(sorted(by_operation.items())),
            "executions_claimed": None,  # text and tool calls never establish execution counts
        }

    def to_dict(self) -> dict[str, Any]:
        covered = sum(1 for r in self.runtimes.values() if r.status != "no_source")
        return {
            "scope": {"root_is_cwd": self.root_is_cwd, "content_included": False},
            "coverage": {"runtimes_total": len(RUNTIMES), "runtimes_covered": covered},
            "repositories": self.repositories.to_dict(),
            "runtimes": {name: self.runtimes[name].to_dict() for name in sorted(self.runtimes)},
            "association": self.association.to_dict(),
            "totals": self.totals(),
            "notes": {
                "empty_is_not_absence": True,
                "counts_are": "recognized recorded calls, not verified executions",
            },
        }


PRUNE = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
    }
)
FIXTURE_DIRS = frozenset(
    {"fixtures", "fixture", "snapshots", "__snapshots__", "testdata", "golden", "samples"}
)


@dataclass
class Repository:
    """One inventoried repository; ``name`` is relative to the scan root."""

    name: str
    path: str
    nested: bool


class RepositoryIndex:
    """Repository lookup by working directory using longest-path containment."""

    def __init__(self, repos: list[Repository]) -> None:
        self.repos = repos
        self._by_path: dict[str, Repository] = {}
        for repo in repos:
            self._by_path[repo.path] = repo
            self._by_path.setdefault(os.path.realpath(repo.path), repo)

    def match(self, cwd: str) -> Repository | None:
        current = os.path.normpath(cwd)
        while True:
            found = self._by_path.get(current)
            if found is not None:
                return found
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent


def inventory(root: Path) -> tuple[RepositoryInventory, RepositoryIndex]:
    summary = RepositoryInventory()
    base = os.path.abspath(root)
    if not os.path.lexists(base):
        summary.root_status = "missing"
        return summary, RepositoryIndex([])
    if not os.path.isdir(base):
        summary.root_status = "not_a_directory"
        return summary, RepositoryIndex([])
    repos: list[Repository] = []
    found: dict[str, Repository] = {}

    def unreadable(_error: OSError) -> None:
        summary.unreadable_dirs += 1

    for here, dirs, files in os.walk(base, followlinks=False, onerror=unreadable):
        summary.directories_walked += 1
        dirs.sort()
        in_fixtures = _in_fixtures(base, here)
        if ".git" in dirs or ".git" in files:
            summary.found += 1
            if ".git" in files:
                summary.worktree_files += 1
            repo = Repository(os.path.relpath(here, base), here, _has_repo_ancestor(here, found))
            summary.nested += repo.nested
            repos.append(repo)
            found[here] = repo
        dirs[:] = [d for d in dirs if d != ".git"]
        kept = []
        for name in dirs:
            if name in PRUNE:
                summary.pruned_dirs += 1
            elif os.path.islink(os.path.join(here, name)):
                summary.symlinks_skipped += 1
            else:
                kept.append(name)
        dirs[:] = kept
        for name in files:
            if os.path.islink(os.path.join(here, name)):
                summary.symlinks_skipped += 1
            elif name.endswith(".jsonl"):
                summary.jsonl_files += 1
                summary.jsonl_in_fixtures += in_fixtures
    return summary, RepositoryIndex(repos)


def _has_repo_ancestor(path: str, found: dict[str, Repository]) -> bool:
    parent = os.path.dirname(path)
    while parent and parent != os.path.dirname(parent):
        if parent in found:
            return True
        parent = os.path.dirname(parent)
    return False


def _in_fixtures(base: str, here: str) -> bool:
    relative = os.path.relpath(here, base)
    return any(part in FIXTURE_DIRS for part in relative.split(os.sep))


PI_SESSION_VARS = ("SSGREP_PI_SESSIONS_DIR", "PI_SESSION_DIR", "PI_CODING_AGENT_SESSION_DIR")
OMP_SESSION_VARS = ("SSGREP_OMP_SESSIONS_DIR", "OMP_SESSIONS_DIR")
PRIME_SESSION_VARS = (
    "SSGREP_PRIME_AGENT_SESSIONS_DIR",
    "PRIME_AGENT_SESSION_DIR",
    "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
)
CODEX_SESSION_VARS = ("SSGREP_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR")


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _value(env: Mapping[str, str], name: str) -> str:
    return env.get(name, "").strip()


def home_dir(env: Mapping[str, str] | None = None) -> Path:
    raw = _value(_env(env), "HOME")
    return Path(raw) if raw else Path.home()


def _expand(raw: str, env: Mapping[str, str]) -> Path:
    if raw == "~" or raw.startswith("~/"):
        return home_dir(env) / raw[2:]
    return Path(raw)


def _pi_style(
    env: Mapping[str, str] | None, session_vars: tuple[str, ...], agent_var: str, default: str
) -> Path:
    values = _env(env)
    for name in session_vars:
        if _value(values, name):
            return _expand(_value(values, name), values)
    if _value(values, agent_var):
        return _expand(_value(values, agent_var), values) / "sessions"
    return home_dir(values) / default


def claude_projects(env: Mapping[str, str] | None = None) -> Path:
    values = _env(env)
    raw = _value(values, "CLAUDE_CONFIG_DIR")
    base = _expand(raw, values) if raw else home_dir(values) / ".claude"
    return base / "projects"


def codex_sessions(env: Mapping[str, str] | None = None) -> Path:
    values = _env(env)
    for name in CODEX_SESSION_VARS:
        if _value(values, name):
            return _expand(_value(values, name), values)
    raw = _value(values, "CODEX_HOME")
    base = _expand(raw, values) if raw else home_dir(values) / ".codex"
    return base / "sessions"


def omp_sessions(env: Mapping[str, str] | None = None) -> Path:
    return _pi_style(env, OMP_SESSION_VARS, "OMP_AGENT_DIR", ".omp/agent/sessions")


def pi_sessions(env: Mapping[str, str] | None = None) -> Path:
    return _pi_style(env, PI_SESSION_VARS, "PI_CODING_AGENT_DIR", ".pi/agent/sessions")


def prime_sessions(env: Mapping[str, str] | None = None) -> Path:
    return _pi_style(
        env, PRIME_SESSION_VARS, "PRIME_AGENT_CODING_AGENT_DIR", ".prime/agent/sessions"
    )


def prime_artifacts(env: Mapping[str, str] | None = None) -> Path:
    """Prime keeps child session transcripts and lineage beside its sessions root."""
    return prime_sessions(env).parent / "session-artifacts"


def opencode_db(env: Mapping[str, str] | None = None) -> Path:
    values = _env(env)
    override = _value(values, "SSGREP_OPENCODE_DB")
    if override:
        return _expand(override, values)
    data_home = _value(values, "XDG_DATA_HOME")
    base = _expand(data_home, values) if data_home else home_dir(values) / ".local" / "share"
    return base / "opencode" / "opencode.db"
