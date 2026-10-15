"""
Resolving a savings selection: which executions, task-day leaves and receipts it covers.

Intersecting filters (time, project, runtime, session, task) are resolved over the committed
ledger exactly as persisted; a filter that cannot be honored exactly is reported with its
reason, never guessed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ssgrep.accounting.contracts import InvocationClass, Scenario, TaskId
from ssgrep.accounting.receipts import receipt_id_for
from ssgrep.accounting.reduce_types import ExecutionRecord
from ssgrep.accounting.revisions import PeriodRow, current_rows
from ssgrep.accounting.service_ledger import LedgerData
from ssgrep.accounting.service_types import Scope
from ssgrep.accounting.valuation import Leaf, value_evidence
from ssgrep.accounting.valuation_inputs import EvidenceBundle

SCENARIOS = (Scenario.CONSERVATIVE, Scenario.CENTRAL, Scenario.OPTIMISTIC)
MEASURED = ("searches", "shows", "failed", "visible_bytes")
GROUP_MEASURED = (*MEASURED, "notes")
ESTIMATE_UNITS = ("context_tokens", "work_seconds")
UNIT_LABEL = {"context_tokens": "tokens", "work_seconds": "s"}
WINDOW_DAYS = {"24h": 1, "7d": 7, "30d": 30}
_DAY = "%Y-%m-%d"


def task_key(task: TaskId | None) -> str:
    """The user-facing task id: ``<root>#<turn>`` (``task_unknown`` without an origin turn)."""
    if task is None:
        return "task_unknown"
    suffix = task.turn if task.turn else "task_unknown"
    branch = "" if task.branch == "main" else f"@{task.branch}"
    return f"{task.root}#{suffix}{branch}"


def runtime_of_root(root: str) -> str:
    return root.split(":", 1)[0]


def _under(cwd: str | None, project: str) -> bool:
    if cwd is None:
        return False
    base = project.rstrip("/") or "/"
    return cwd == base or cwd.startswith(base.rstrip("/") + "/")


def _moment(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp).astimezone(UTC)
    except ValueError:
        return None


@dataclass(frozen=True)
class Plan:
    """The resolved selection: what the filters mean over this ledger."""

    start: datetime | None
    end: datetime | None
    end_inclusive: bool
    first_day: str | None
    last_day: str | None
    executions: tuple[ExecutionRecord, ...]
    roots: frozenset[str] | None  # None = every root
    estimates_reason: str | None
    notes: tuple[str, ...]
    unknown_project_roots: int


def _bounds(scope: Scope, collected_at: str) -> tuple[datetime | None, datetime | None, bool]:
    now = _moment(collected_at)
    if scope.since or scope.until:
        start = _moment(scope.since) if scope.since else None
        end = _moment(scope.until) if scope.until else None
        return start, end, False
    days = WINDOW_DAYS.get(scope.window)
    if now is None or days is None:
        return None, now, True
    return now - timedelta(days=days), now, True


def _days(
    start: datetime | None, end: datetime | None, inclusive: bool
) -> tuple[str | None, str | None]:
    first = start.strftime(_DAY) if start else None
    if end is None:
        return first, None
    last = end if inclusive else end - timedelta(microseconds=1)
    return first, last.strftime(_DAY)


def root_cwds(data: LedgerData) -> dict[str, str]:
    doc = data.doc
    if doc is None:
        return {}
    by_key = {(r, s): c for r, s, c in doc.session_cwd}
    out: dict[str, str] = {}
    for runtime, session, root in sorted(doc.session_roots):
        cwd = by_key.get((runtime, session))
        if cwd is not None:
            if root == f"{runtime}:{session}" or root not in out:
                out[root] = cwd
    return out


def _session_plan(
    scope: Scope, doc
) -> tuple[set[tuple[str, str]], frozenset[str] | None, str | None, list[str]]:
    """(selected session keys, exact estimate roots or None, reason when not separable, notes)."""
    notes: list[str] = []
    if doc is None:
        return set(), None, "session_lineage_unavailable", notes
    known = {(r, s): root for r, s, root in doc.session_roots}
    seeds = {k for k in known if k[1] == scope.session}
    if not seeds:
        return set(), frozenset(), None, ["session_unknown: no retained session has that id"]
    keys = set(seeds)
    roots = {known[k] for k in seeds}
    children_of: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for runtime, session, parent in doc.session_parents:
        if parent is not None:
            children_of.setdefault((runtime, parent), []).append((runtime, session))
    all_root_keys = all(root == f"{k[0]}:{k[1]}" for k, root in ((k, known[k]) for k in seeds))
    if scope.include_children:
        if all_root_keys:
            keys = {k for k, root in known.items() if root in roots}
        elif not doc.session_parents:
            notes.append("session_parents_unavailable: children of a child session are unknown")
        else:
            stack = list(seeds)
            while stack:
                for child in children_of.get(stack.pop(), []):
                    if child not in keys:
                        keys.add(child)
                        stack.append(child)
    has_children = any(k not in seeds and known[k] in roots for k in known)
    if all_root_keys and (scope.include_children or not has_children):
        return keys, frozenset(roots), None, notes
    reason = "child_session_not_separable" if not all_root_keys else "children_not_separable"
    return keys, None, reason, notes


def plan(scope: Scope, data: LedgerData, collected_at: str) -> Plan:
    doc = data.doc
    start, end, inclusive = _bounds(scope, collected_at)
    first, last = _days(start, end, inclusive)
    rows = doc.executions if doc else ()
    notes: list[str] = []

    def in_time(row: ExecutionRecord) -> bool:
        moment = _moment(row.ts_utc)
        if moment is None:
            return False
        if start is not None and moment < start:
            return False
        if end is not None and (moment > end if inclusive else moment >= end):
            return False
        return True

    chosen = [r for r in rows if in_time(r)]
    roots: frozenset[str] | None = None
    reason: str | None = None
    if scope.runtime:
        chosen = [r for r in chosen if r.runtime == scope.runtime]
    if scope.session:
        keys, roots, reason, extra = _session_plan(scope, data.doc)
        notes.extend(extra)
        chosen = [r for r in chosen if (r.runtime or "", r.session_id or "") in keys]
    if scope.project:
        real = os.path.realpath(scope.project)
        chosen = [r for r in chosen if _under(r.cwd, scope.project) or _under(r.cwd, real)]
    if scope.task:
        wanted = scope.task
        chosen = [
            r
            for r in chosen
            if r.task is not None and (task_key(r.task) == wanted or r.task.root == wanted)
        ]
    return Plan(start, end, inclusive, first, last, tuple(chosen), roots, reason, tuple(notes), 0)


def _day_ok(day: str, first: str | None, last: str | None) -> bool:
    return (first is None or day >= first) and (last is None or day <= last)


def select_leaves(
    scope: Scope, data: LedgerData, selection: Plan, source: list[Leaf] | None = None
) -> tuple[list[Leaf], int, list[PeriodRow]]:
    """Unique leaves of current revisions that satisfy every task-level filter.

    ``source`` replaces the stored leaves with leaves valued over a sub-selection (see
    ``session_leaves``); every other filter still applies.
    """
    current = current_rows(data.rows.periods, data.rows.namespace)
    cwds = root_cwds(data)
    real = os.path.realpath(scope.project) if scope.project else None
    unknown = 0
    out: list[Leaf] = []
    pool = [source] if source is not None else [row.leaves for row in current]
    for leaves in pool:
        for leaf in leaves:
            task = leaf.task
            if not _day_ok(leaf.day, selection.first_day, selection.last_day):
                continue
            if scope.runtime and runtime_of_root(task.root) != scope.runtime:
                continue
            if selection.roots is not None and task.root not in selection.roots:
                continue
            if scope.task and not (task_key(task) == scope.task or task.root == scope.task):
                continue
            if scope.project:
                cwd = cwds.get(task.root)
                if cwd is None:
                    unknown += 1
                    continue
                if not (_under(cwd, scope.project) or _under(cwd, real or scope.project)):
                    continue
            out.append(leaf)
    return out, unknown, current


def session_scope(
    scope: Scope, data: LedgerData, selection: Plan
) -> tuple[list[Leaf], set[str]] | None:
    """Leaves valued over ONLY the selected sessions' invocations, or None when impossible.

    Used when a session (or its children) is a strict part of a task: stored leaves are per
    task-day and cannot be split, so the selection's candidates and debits are valued again
    with the ledger's own methodology. Overlap removal then applies inside the selection
    only, so these figures do not add up to the task totals; the caller says so.
    """
    doc = data.doc
    if doc is None or not doc.invocation_sessions:
        return None
    keys, _, _, _ = _session_plan(scope, data.doc)
    wanted = {
        inv for inv, runtime, session in doc.invocation_sessions if (runtime, session) in keys
    }
    candidates = tuple(c for c in data.candidates if set(c.exposure_ids) & wanted)
    debits = tuple(d for d in data.debits if d.exposure_id in wanted)
    valued = value_evidence(EvidenceBundle(candidates, debits), data.methodology_spec)
    ids = {receipt_id_for(c.task, c.candidate_id, data.methodology_spec) for c in candidates}
    return list(valued.leaves), ids


def lookup_records(scope: Scope, doc, collected_at: str):
    """Intersect the same filters over native invocations and orphan executions.

    Ownership is resolved BEFORE filtering, so a matched execution never becomes
    a second lookup merely because its invocation was outside the selection.
    """
    start, end, inclusive = _bounds(scope, collected_at)
    sessions = _session_plan(scope, doc)[0] if scope.session else None
    real = os.path.realpath(scope.project) if scope.project else None

    def matches(row):
        moment = _moment(row.ts_utc) if row.ts_utc else None
        if moment is None and (scope.since is not None or scope.until is not None):
            return False
        if moment is not None:
            if start is not None and moment < start:
                return False
            if end is not None and (moment > end if inclusive else moment >= end):
                return False
        if scope.runtime and row.runtime != scope.runtime:
            return False
        if sessions is not None and (row.runtime or "", row.session_id or "") not in sessions:
            return False
        if (
            scope.project
            and real is not None
            and not (_under(row.cwd, scope.project) or _under(row.cwd, real))
        ):
            return False
        if scope.task and (
            row.task is None
            or not (task_key(row.task) == scope.task or row.task.root == scope.task)
        ):
            return False
        return True

    accepted = {
        InvocationClass.MATCHED,
        InvocationClass.TRANSCRIPT_ONLY,
        InvocationClass.JOURNAL_GAP,
    }
    invocations = tuple(
        r for r in doc.invocations if r.cls in accepted and set(r.operations) & {"search", "show"}
    )
    owned = {execution for row in doc.invocations for execution in row.execution_ids}
    rows = [("invocation", row.invocation_id, row) for row in invocations if matches(row)]
    rows.extend(
        ("execution", row.execution_id, row)
        for row in doc.executions
        if row.operation in {"search", "show"} and row.execution_id not in owned and matches(row)
    )
    return sorted(rows, key=lambda item: (item[2].ts_utc or "", item[0], item[1]))
