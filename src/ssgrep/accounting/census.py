"""Supported, read-only runtime/repository census (``ssgrep metrics census``).

Reads every runtime's raw store through the shared bounded readers, deduplicates
copied histories through one ledger, classifies recorded tool calls, and reports
aggregate counts only. It never writes (no accounting state, no source changes),
never executes or fetches anything found in a transcript, and never returns
transcript content or paths. Counts are recognized *recorded invocations*, not
verified executions.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ssgrep.accounting.census_model import (
    Association,
    CensusReport,
    RuntimeCensus,
    bump,
    render_census_text,
)
from ssgrep.accounting.census_repos import RepositoryIndex, inventory
from ssgrep.accounting.contracts import RUNTIMES, InvocationClass
from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.classify import UNKNOWN, classify_evidence
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader

__all__ = ["CensusReport", "render_census_text", "run_census"]

CYCLE_BYTES = 32 * 1024 * 1024
SESSION_KINDS_WITHOUT_SESSION = frozenset({"lineage"})


def _readers(env: Mapping[str, str] | None) -> dict[str, Any]:
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
                budget = None if isinstance(reader, OpenCodeReader) else CYCLE_BYTES
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
