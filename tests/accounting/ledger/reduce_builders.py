"""Real-shaped fixtures for reducer tests: native transcripts read back through the readers.

Evidence is never hand-built: records are written as native claude/omp JSONL (via
``readers_fixtures``) and decoded by the real readers, so the reducer sees exactly
what production sees. Journal events use the real ``ExecutionEvent`` contract and the
real keyed request fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ssgrep.accounting.capture import fingerprint as fp
from ssgrep.accounting.collect.frequency import FrequencyView, build_view
from ssgrep.accounting.ledger.contracts import (
    ExecutionEvent,
    Operation,
    Outcome,
    Phase,
    Quantity,
    RawEvidence,
    Transport,
)
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.readers.base import EvidenceLedger, JsonlReader
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader
from ssgrep.accounting.readers.pi_family import OmpReader
from tests.accounting.readers.readers_fixtures import (
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    pi_user,
    write_records,
)

KEY = bytes(range(32))
PROFILE = "personal-local"
CWD = "/work/proj"
MCP = "mcp__acme_recall__search_sessions"
SHOW = "mcp__acme_recall__show_session"


def ts(day: int = 12, h: int = 10, m: int = 0, s: int = 0, month: int = 10) -> str:
    return f"2026-{month:02d}-{day:02d}T{h:02d}:{m:02d}:{s:02d}.000Z"


def card(ref: str, excerpt: str, title: str = "earlier session") -> dict[str, Any]:
    return {"ref": ref, "title": title, "excerpt": excerpt, "files_touched": []}


def results_json(*cards: dict[str, Any], **extra: Any) -> str:
    body = {
        "results": list(cards),
        "total_matches": len(cards),
        "omitted_count": 0,
        "excerpts_truncated": False,
        "clamped": False,
        "index_empty": False,
    }
    body.update(extra)
    return json.dumps(body)


def policy(*roots: str, profile: str = PROFILE) -> ProfilePolicy:
    return ProfilePolicy(profile_id=profile, roots=roots, collection_enabled=True)


def fingerprint_of(operation: str, text: str, where: str | None = None) -> str:
    return fp.request_fingerprint(KEY, Operation(operation), query=text, where=where)


def event(
    execution_id: str,
    when: str,
    *,
    operation: str = "search",
    text: str = "deploy failure",
    transport: str = "mcp",
    outcome: Outcome = Outcome.OK,
    refs: tuple[str, ...] = (),
    cwd: str | None = CWD,
    correlation: str | None = None,
    phases: Iterable[Phase] = (Phase.STARTED, Phase.COMPLETED, Phase.RENDERED),
    profile: str = PROFILE,
    rendered: int | None = 900,
    where: str | None = None,
) -> list[ExecutionEvent]:
    """The journal phases of one execution (started/completed/rendered)."""
    out: list[ExecutionEvent] = []
    for phase in phases:
        out.append(
            ExecutionEvent(
                execution_id=execution_id,
                phase=phase,
                operation=Operation(operation),
                transport=Transport(transport),
                profile=profile,
                ts_utc=when,
                cwd=cwd,
                fingerprint=fingerprint_of(operation, text, where),
                version="2.0.1",
                outcome=outcome if phase is Phase.COMPLETED else None,
                mono_ns=1_000_000 if phase is Phase.COMPLETED else None,
                refs=refs if phase is Phase.COMPLETED else (),
                correlation_id=correlation,
                rendered_bytes=(
                    Quantity.observed(rendered, "bytes", "mcp_json")
                    if phase is Phase.RENDERED and rendered is not None
                    else None
                ),
            )
        )
    return out


def read_runtime(reader: JsonlReader) -> list[RawEvidence]:
    """Everything the real reader decodes from its store, copies deduplicated."""
    ledger = EvidenceLedger()
    out: list[RawEvidence] = []
    for source in reader.discover():
        state = None
        while True:
            result = reader.read(source, state)
            out.extend(e for e in result.evidence if ledger.admit(e))
            state = result.state
            if not result.more:
                break
    return out


def claude_evidence(root: Path, files: dict[str, list[dict[str, Any]]]) -> list[RawEvidence]:
    """Write ``files`` (relative path -> native records) as a claude store and read it back."""
    projects = root / ".claude" / "projects"
    for relative, records in files.items():
        write_records(projects / relative, records)
    return read_runtime(ClaudeReader({"CLAUDE_CONFIG_DIR": str(root / ".claude")}))


def omp_evidence(root: Path, files: dict[str, list[dict[str, Any]]]) -> list[RawEvidence]:
    base = root / ".omp" / "agent" / "sessions"
    for relative, records in files.items():
        write_records(base / relative, records)
    return read_runtime(OmpReader({"OMP_SESSIONS_DIR": str(base)}))


def rare_view(*rare: str, corpus: int = 1000) -> FrequencyView:
    """A frozen candidate-frequency view in which each ``rare`` text occurs in 1 document."""
    docs = [f"filler {i}" for i in range(corpus - len(rare))] + list(rare)
    return build_view(docs)


def source_session(
    session: str,
    fact: str,
    *,
    when_day: int = 1,
    cwd: str = CWD,
    prompt: str = "why does the deploy fail",
) -> list[dict[str, Any]]:
    """An earlier session whose tool result first observed ``fact`` (the witness)."""
    stamp = ts(when_day)
    return [
        claude_prompt(session, prompt, uuid="u1", ts=stamp),
        claude_tool_use(
            session, "w1", "Bash", {"command": "cat deploy.log"}, ts=ts(when_day, 10, 0, 5), cwd=cwd
        ),
        claude_tool_result(session, "w1", fact, ts=ts(when_day, 10, 0, 6)),
    ]


def at(record: dict[str, Any], stamp: str) -> dict[str, Any]:
    """The same native record with an explicit timestamp."""
    return {**record, "timestamp": stamp}


def omp_header(session: str, stamp: str, *, parent: Path | None = None, cwd: str = CWD) -> dict:
    extra = {"parentSession": str(parent)} if parent is not None else {}
    return at(pi_header(session, cwd=cwd, **extra), stamp)


def omp_user(entry: str, text: str, stamp: str) -> dict:
    return at(pi_user(entry, text), stamp)


def omp_call(entry: str, call_id: str, name: str, arguments: dict, stamp: str) -> dict:
    return at(pi_tool_call(entry, call_id, name, arguments), stamp)


def omp_result(entry: str, call_id: str, text: str, stamp: str, *, error: bool = False) -> dict:
    return at(pi_tool_result(entry, call_id, text, is_error=error), stamp)


def omp_search(
    entry: str, call_id: str, query: str, body: str, when: tuple[int, int, int, int]
) -> list[dict]:
    """One MCP search call and its result, ``when`` = (day, hour, minute, second)."""
    day, hour, minute, second = when
    return [
        omp_call(entry, call_id, MCP, {"query": query}, ts(day, hour, minute, second)),
        omp_result(entry + "r", call_id, body, ts(day, hour, minute, second + 2)),
    ]


def omp_launch(entry: str, call_id: str, child: str, stamp: str, result_stamp: str) -> list[dict]:
    """A parent ``task`` call that spawns ``child`` and the result naming it."""
    arguments = {"tasks": [{"name": child, "task": "investigate the failing deploy"}]}
    return [
        omp_call(entry, call_id, "task", arguments, stamp),
        omp_result(entry + "r", call_id, f"Spawned 1 background agent.\n- `{child}`", result_stamp),
    ]


__all__ = [
    "CWD",
    "KEY",
    "MCP",
    "at",
    "omp_call",
    "omp_header",
    "omp_launch",
    "omp_result",
    "omp_search",
    "omp_user",
    "PROFILE",
    "SHOW",
    "card",
    "claude_evidence",
    "claude_prompt",
    "claude_tool_result",
    "claude_tool_use",
    "event",
    "fingerprint_of",
    "omp_evidence",
    "pi_header",
    "pi_tool_call",
    "pi_tool_result",
    "pi_user",
    "policy",
    "rare_view",
    "read_runtime",
    "results_json",
    "source_session",
    "ts",
]
