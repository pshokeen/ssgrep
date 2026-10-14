"""Census data model: aggregate-only counts, integers and fixed vocabularies.

Nothing here can carry transcript content: fields are counters keyed by
enumerated runtime/class/route/operation names, plus repository names relative
to the scan root. Missing quantities are ``None`` (never zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ssgrep.accounting.contracts import RUNTIMES

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


def render_census_text(report: CensusReport) -> str:
    """Human summary of the same aggregate numbers; no paths or content."""
    data = report.to_dict()
    repos, totals = data["repositories"], data["totals"]
    lines = [
        "ssgrep metrics census (aggregate only)",
        f"repositories: {repos['found']} found ({repos['nested']} nested, "
        f"{repos['worktree_files']} worktree), {repos['symlinks_skipped']} symlinks skipped, "
        f"{repos['jsonl_files']} repo-local jsonl ({repos['jsonl_in_fixtures']} in fixtures)",
        "",
        "runtime        status     sources  records  recognized  loop/unknown  gaps",
    ]
    for name, row in data["runtimes"].items():
        invocations = row["invocations"]
        lines.append(
            f"{name:<14} {row['status']:<10} {row['sources']:>7}  {row['evidence_records']:>7}"
            f"  {invocations['recognized_calls']:>10}  {invocations['unknown_cardinality']:>12}"
            f"  {sum(row['gaps'].values()):>4}"
        )
    association = data["association"]
    lines += [
        "",
        f"recognized calls: {totals['recognized_calls']} "
        f"({totals['loop_or_unknown_cardinality']} loop or unknown-cardinality invocations)",
        f"sessions: {association['sessions']} ({association['in_repository']} in a repository, "
        f"{association['nested_repository']} nested, {association['unknown_root']} unknown root, "
        f"{association['no_cwd']} no cwd)",
        "Recognized calls are recorded invocations, not verified executions; shell loops are one "
        "invocation of unknown cardinality.",
        "An empty or partial result is not evidence that no historical usage occurred.",
    ]
    return "\n".join(lines)
