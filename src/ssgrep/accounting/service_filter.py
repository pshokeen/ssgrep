"""
Groupings, estimate rollups and the evaluated outcome of a savings selection (read-only).

Selection itself lives in ``service_select``; cursors are bound to the snapshot, generation,
normalized selection and the full ordered row set (see ``service_cursor``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from ssgrep.accounting.contracts import Attribution, Basis, Quantity, digest_of
from ssgrep.accounting.receipts import ReceiptRecord
from ssgrep.accounting.reduce_types import ExecutionRecord, summarize
from ssgrep.accounting.service_cursor import issue, redeem
from ssgrep.accounting.service_ledger import LedgerData
from ssgrep.accounting.service_records import RECEIPT_ID_RE
from ssgrep.accounting.service_select import (
    ESTIMATE_UNITS,
    GROUP_MEASURED,
    MEASURED,
    SCENARIOS,
    UNIT_LABEL,
    Plan,
    plan,
    root_cwds,
    runtime_of_root,
    select_leaves,
    session_scope,
    task_key,
)
from ssgrep.accounting.service_types import GroupRow, ScenarioRow, Scope
from ssgrep.accounting.valuation import Leaf, rollup
from ssgrep.accounting.windows_estimates import unit_keys


def _unavailable(unit_key: str, reason: str) -> Quantity:
    return Quantity.unavailable(UNIT_LABEL[unit_key], reason)


def scenario_rows(
    leaves: Iterable[Leaf], keys: dict, reason: str | None = None
) -> tuple[ScenarioRow, ...]:
    """Credit/cost/net rows per unit key over ``leaves`` (empty or unsolvable => unavailable)."""
    by_unit: dict[str, list[Leaf]] = {}
    for leaf in leaves:
        by_unit.setdefault(keys.get(leaf.unit, leaf.unit.unit), []).append(leaf)
    rows = []
    for unit_key in ESTIMATE_UNITS:
        group = by_unit.get(unit_key)
        why = reason or "no_estimates_in_selection"
        if reason is not None or not group:
            cells = tuple((s, _unavailable(unit_key, why)) for s in SCENARIOS)
            rows.append(ScenarioRow(unit_key, cells, cells, cells))
            continue
        try:
            estimates = {e.scenario: e for e in rollup(group, group[0].unit)}
        except ValueError:
            cells = tuple((s, _unavailable(unit_key, "rollup_conflict")) for s in SCENARIOS)
            rows.append(ScenarioRow(unit_key, cells, cells, cells))
            continue
        rows.append(
            ScenarioRow(
                unit_key,
                tuple((s, estimates[s].credit) for s in SCENARIOS),
                tuple((s, estimates[s].debit) for s in SCENARIOS),
                tuple((s, estimates[s].net) for s in SCENARIOS),
            )
        )
    return tuple(rows)


def measured_of(
    executions: Iterable[ExecutionRecord], names: tuple[str, ...]
) -> tuple[tuple[str, Quantity], ...]:
    summary = summarize(executions)
    return tuple(
        (n, summary[n]) if n in summary else (n, Quantity(0, "count", Basis.OBSERVED, n))
        for n in names
    )


@dataclass(frozen=True)
class Page:
    groups: tuple[GroupRow, ...]
    total: int
    next_cursor: str | None


def _group_keyers(
    by: str, cwds: dict[str, str]
) -> tuple[Callable[[ExecutionRecord], str], Callable[[Leaf], str]]:
    if by == "task":
        return lambda e: task_key(e.task), lambda leaf: task_key(leaf.task)
    if by == "runtime":
        return (
            lambda e: e.runtime or "unknown",
            lambda leaf: runtime_of_root(leaf.task.root),
        )
    if by == "project":
        return lambda e: e.cwd or "project_unknown", lambda leaf: cwds.get(
            leaf.task.root, "project_unknown"
        )
    return lambda e: e.day, lambda leaf: leaf.day


def groups_of(
    by: str,
    selection: Plan,
    leaves: list[Leaf],
    data: LedgerData,
    keys: dict,
    estimates_reason: str | None,
    binding: str,
    limit: int,
    cursor: str | None,
    command: str,
) -> Page:
    exec_key, leaf_key = _group_keyers(by, root_cwds(data))
    execs: dict[str, list[ExecutionRecord]] = {}
    for row in selection.executions:
        execs.setdefault(exec_key(row), []).append(row)
    members: dict[str, list[Leaf]] = {}
    for leaf in leaves:
        members.setdefault(leaf_key(leaf), []).append(leaf)
    receipts: dict[str, int] = {}
    owner = {(leaf.task.root, leaf.task.turn, leaf.day): leaf_key(leaf) for leaf in leaves}
    for _, _, _, record in receipt_entries(data, leaves):
        key = owner.get((record.receipt.root, record.receipt.task_turn, record.period))
        if key is not None:
            receipts[key] = receipts.get(key, 0) + 1
    names = sorted(set(execs) | set(members))
    rows = tuple(
        GroupRow(
            key,
            measured_of(execs.get(key, ()), GROUP_MEASURED),
            scenario_rows(members.get(key, ()), keys, estimates_reason),
            receipts.get(key, 0),
        )
        for key in names
    )
    shape = digest_of([[g.key, [(n, q.value) for n, q in g.measured], g.receipts] for g in rows])
    bound = f"{binding}|{shape}"
    offset = redeem("g", cursor, bound, command)
    page = rows[offset : offset + limit]
    more = offset + limit < len(rows)
    return Page(page, len(rows), issue("g", offset + limit, bound) if more else None)


def receipt_entries(
    data: LedgerData, leaves: list[Leaf], only: set[str] | None = None
) -> list[tuple[int, str, bool, ReceiptRecord]]:
    """Latest revision of each receipt that lies inside the selected task-days."""
    wanted = {(leaf.task.root, leaf.task.turn, leaf.day) for leaf in leaves}
    latest: dict[str, ReceiptRecord] = {}
    for record in data.rows.receipts:
        receipt = record.receipt
        if (receipt.root, receipt.task_turn, record.period) not in wanted:
            continue
        if not RECEIPT_ID_RE.fullmatch(receipt.receipt_id):
            continue
        if only is not None and receipt.receipt_id not in only:
            continue
        held = latest.get(receipt.receipt_id)
        if held is None or receipt.revision > held.receipt.revision:
            latest[receipt.receipt_id] = record
    return [
        (dict(r.credit_raw).get("central", 0), i, bool(r.receipt.costs), r)
        for i, r in sorted(latest.items())
    ]


def _source_unknown(record: ReceiptRecord) -> bool:
    """True when any source atom of the receipt has an unknown cost (``missing:`` exclusion)."""
    return any(entry.startswith("missing:") for entry in record.receipt.exclusions)


def representative(entries: list[tuple[int, str, bool, ReceiptRecord]]) -> str | None:
    credited = [(-c, i) for c, i, _, _ in entries if c > 0]
    if credited:
        return min(credited)[1]
    costed = [i for _, i, k, _ in entries if k]
    return min(costed) if costed else None


def selection_binding(
    profile: str, snapshot_id: str, scope: Scope, limit: int, ledger: LedgerData
) -> str:
    return "|".join(
        [
            profile,
            snapshot_id,
            f"gen{ledger.generation}",
            ledger.methodology[:16],
            digest_of(
                [
                    scope.window,
                    scope.since,
                    scope.until,
                    scope.project,
                    scope.runtime,
                    scope.session,
                    scope.task,
                    scope.by,
                    scope.include_children,
                ]
            )[:16],
            str(limit),
        ]
    )


@dataclass(frozen=True)
class Outcome:
    measured: tuple[tuple[str, Quantity], ...]
    scenarios: tuple[ScenarioRow, ...]
    groups: tuple[GroupRow, ...]
    group_total: int | None
    next_cursor: str | None
    coverage: dict[str, int | None]
    receipt_id: str | None
    notes: tuple[str, ...]
    estimates_reason: str | None
    unit_keys_: dict = field(default_factory=dict)


def evaluate(
    profile: str,
    scope: Scope,
    data: LedgerData,
    snapshot_id: str,
    collected_at: str,
    limit: int,
    cursor: str | None,
) -> Outcome:
    """The complete filtered selection, one page of groups when ``--by`` is given."""
    selection = plan(scope, data, collected_at)
    source: list[Leaf] | None = None
    only: set[str] | None = None
    notes = list(selection.notes)
    reason = selection.estimates_reason
    if reason in ("child_session_not_separable", "children_not_separable"):
        revalued = session_scope(scope, data, selection)
        if revalued is None:
            reason = "session_estimates_unavailable"
        else:
            source, only = revalued
            reason = None
            notes.append(
                "session_revalued: estimates were valued over this selection's own lookups "
                "only; overlap is removed inside it, so they do not add up to task totals"
            )
    selection = replace(selection, estimates_reason=reason)
    leaves, unknown, current = select_leaves(scope, data, selection, source)
    keys = unit_keys(current)
    if unknown:
        notes.append(
            f"project_unknown: {unknown} estimate period(s) belong to tasks whose project "
            "is not recorded and are excluded from this project filter"
        )
    if selection.estimates_reason:
        notes.append(f"estimates_unavailable: {selection.estimates_reason}")
    rows = scenario_rows(leaves, keys, selection.estimates_reason)
    binding = selection_binding(profile, snapshot_id, scope, limit, data)
    command = f"ssgrep savings{'' if profile == 'personal-local' else f' --profile {profile}'}"
    page = Page((), 0, None)
    if scope.by:
        page = groups_of(
            scope.by,
            selection,
            leaves,
            data,
            keys,
            selection.estimates_reason,
            binding,
            limit,
            cursor,
            command,
        )
    lookups = [e for e in selection.executions if e.operation in ("search", "show")]
    attributed = sum(
        1 for e in lookups if e.attribution in (Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE)
    )
    entries = receipt_entries(data, leaves, only)
    coverage: dict[str, int | None] = {
        "recorded_lookups": len(lookups),
        "attributed_lookups": attributed,
        "candidate_receipts": len(entries),
        "source_costs_available": sum(1 for *_, r in entries if not _source_unknown(r)),
        "ambiguous_joins": sum(1 for e in lookups if e.attribution is Attribution.AMBIGUOUS),
        "unavailable_sources": sum(1 for *_, r in entries if _source_unknown(r)),
    }
    return Outcome(
        measured_of(selection.executions, MEASURED),
        rows,
        page.groups,
        page.total if scope.by else None,
        page.next_cursor,
        coverage,
        representative(entries),
        tuple(notes),
        selection.estimates_reason,
    )
