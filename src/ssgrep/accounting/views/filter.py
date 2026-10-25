"""
Groupings, estimate rollups and the evaluated outcome of a savings selection (read-only).

Selection itself lives in ``service_select``; cursors are bound to the snapshot, generation,
normalized selection and the full ordered row set (see ``service_cursor``).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.ledger.contracts import (
    PERSONAL_PROFILE,
    Attribution,
    Basis,
    ExecutionEvent,
    Quantity,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord, summarize
from ssgrep.accounting.ledger.valuation import Leaf, rollup
from ssgrep.accounting.ledger.valuation_inputs import CURRENT_VALUATION_VERSION, AtomStatus
from ssgrep.accounting.privacy.policy import (
    PolicyError,
    PolicyFile,
    effective_ownership,
    historical_workspace_owner,
)
from ssgrep.accounting.views.history import unit_keys
from ssgrep.accounting.views.ledger import LedgerData
from ssgrep.accounting.views.records import RECEIPT_ID_RE, ReceiptRecord
from ssgrep.accounting.views.selection import (
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
from ssgrep.accounting.views.types import GroupRow, ScenarioRow, Scope, issue, redeem

if TYPE_CHECKING:
    from ssgrep.accounting.ledger.reduce_join import ExecFact, Link
    from ssgrep.accounting.ledger.reduce_records import Call


def _unavailable(unit_key: str, reason: str) -> Quantity:
    return Quantity.unavailable(UNIT_LABEL.get(unit_key, unit_key.split(":", 1)[0]), reason)


def scenario_rows(
    leaves: Iterable[Leaf], keys: dict, reason: str | None = None
) -> tuple[ScenarioRow, ...]:
    """Credit/cost/net rows per unit key over ``leaves`` (empty or unsolvable => unavailable)."""
    by_unit: dict[str, list[Leaf]] = {}
    for leaf in leaves:
        by_unit.setdefault(keys.get(leaf.unit, leaf.unit.unit), []).append(leaf)
    rows = []
    for unit_key in sorted(set(ESTIMATE_UNITS) | set(by_unit)):
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
                tuple(estimates[s].as_scenario_estimate() for s in SCENARIOS),
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
    """The complete filtered selection, one page of groups when `--group-by` is given."""
    selection = plan(scope, data, collected_at)
    source: list[Leaf] | None = None
    only: set[str] | None = None
    notes = list(selection.notes)
    reason = selection.estimates_reason
    if reason in ("child_session_not_separable", "children_not_separable"):
        revalued = session_scope(scope, data, selection)
        if revalued is None:
            reason = (
                "methodology_retired"
                if data.methodology_spec.valuation_version != CURRENT_VALUATION_VERSION
                else "session_estimates_unavailable"
            )
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


SOURCE_OUT_OF_SCOPE = "source_out_of_scope"
SOURCE_SCOPE_UNKNOWN = "source_scope_unknown"


@dataclass
class Scoper:
    """Cached historical canonical-cwd decisions for one profile policy."""

    policy: ProfilePolicy
    ownership: PolicyFile | None = None
    _cache: dict[str | None, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.ownership = effective_ownership(self.policy, self.ownership)
        except PolicyError:
            self.ownership = PolicyFile()

    def in_scope(self, cwd: str | None) -> bool:
        """Whether a recorded canonical cwd belongs to this profile, without rebinding."""
        assert self.ownership is not None
        if cwd in self._cache:
            return self._cache[cwd]
        if cwd is None:
            allowed = (
                self.policy.profile_id == PERSONAL_PROFILE
                and not self.policy.roots
                and PERSONAL_PROFILE in self.ownership.profiles
            )
        else:
            allowed = historical_workspace_owner(self.ownership, cwd) == self.policy.profile_id
        self._cache[cwd] = allowed
        return allowed

    def source_status(self, source_cwd: str | None) -> tuple[AtomStatus, str]:
        """Status of a source atom whose content was produced in ``source_cwd``.

        The consumer's own scope is checked separately (and first); this only
        answers whether the SOURCE cost may be valued under this profile.
        """
        if self.in_scope(source_cwd):
            return AtomStatus.OK, ""
        # Resolving a refused alias only labels missing provenance; it never grants
        # ownership or changes an already accepted canonical decision.
        unknown = source_cwd is None or os.path.realpath(source_cwd) != source_cwd
        reason = SOURCE_SCOPE_UNKNOWN if unknown else SOURCE_OUT_OF_SCOPE
        return AtomStatus.OUT_OF_SCOPE, reason


TRUSTED_BASES = frozenset({"process_cwd", "trusted_invocation", "native_call_metadata"})
JOINED = frozenset({Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE})


def execution_scope(
    fact: ExecFact, link: Link, call: Call | None, scoper: Scoper
) -> tuple[str | None, str, bool]:
    """One scope decision shared by execution rows and derived counter evidence."""
    if fact.cwd and (
        fact.scope_basis in TRUSTED_BASES
        or fact.scope_basis == "legacy"
        and fact.transport == "cli"
    ):
        basis = "process_cwd" if fact.scope_basis == "legacy" else fact.scope_basis
        return fact.cwd, basis, scoper.in_scope(fact.cwd)
    if (
        link.attribution in JOINED
        and len(link.requests) == 1
        and call is not None
        and call.in_scope
    ):
        request = next((item for item in call.requests if item.key == link.requests[0]), None)
        if (
            request is not None
            and request.cwd
            and request.cwd_trusted
            and scoper.in_scope(request.cwd)
        ):
            cwd = request.cwd
            return cwd, "native_call_metadata", scoper.in_scope(cwd)
    return None, "unknown_scope", scoper.in_scope(None)


def invocation_provenance(
    call: Call, owned: list[str], ambiguous: list[str], links: Mapping[str, Link]
) -> dict[str, Any]:
    reasons = {
        reason for execution_id in owned + ambiguous for reason in links[execution_id].reasons
    }
    trusted = (
        bool(call.cwd) and bool(call.requests) and all(item.cwd_trusted for item in call.requests)
    )
    if not trusted:
        reasons.add("unknown_scope" if not call.cwd else "untrusted_cwd")
    key_ids = {item.fingerprint_key_id for item in call.requests}
    return {
        "scope_basis": "native_call_metadata" if trusted else "unknown_scope",
        "join_rejection_reasons": tuple(sorted(reasons)),
        "actor_basis": "configured_verifier" if call.actor == "verifier" else "native_lineage",
        "fingerprint_key_id": next(iter(key_ids)) if len(key_ids) == 1 else None,
    }


def effective_scoped_journal(
    raw: Iterable[ExecutionEvent], executions: Iterable[ExecutionRecord], scoper: Scoper
) -> list[ExecutionEvent]:
    """Counter inputs authorized for this profile, including safe native corrections.

    Derived copies preserve execution/phase identity; nothing writes them back to
    the journal. A trusted unique native consumer may resolve missing capture scope,
    but policy ownership must still agree. Unjoined unknown personal observations
    remain local with cwd=None; organization counters never include those events.
    """
    rows = {row.execution_id: row for row in executions}
    result = []
    for event in raw:
        if event.profile != scoper.policy.profile_id:
            continue
        known = event.scope_basis in TRUSTED_BASES or (
            event.scope_basis == "legacy" and str(event.transport) == "cli"
        )
        if known and event.cwd:
            if scoper.in_scope(event.cwd):
                result.append(event)
            continue
        row = rows.get(event.execution_id)
        if (
            row is not None
            and row.in_scope
            and row.scope_basis == "native_call_metadata"
            and row.attribution in JOINED
            and row.cwd
            and scoper.in_scope(row.cwd)
        ):
            result.append(replace(event, cwd=row.cwd, scope_basis="native_call_metadata"))
        elif scoper.in_scope(None):
            result.append(replace(event, cwd=None, scope_basis="unknown_scope"))
    return result
