"""Selection failures and session revaluation never borrow unrelated evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import service_filter, service_ledger, service_select
from ssgrep.accounting.contracts import Basis, TaskId
from ssgrep.accounting.scope import build_scope
from ssgrep.accounting.service_types import Scope, parse_quantity
from ssgrep.accounting.units import CONTEXT_TOKENS, context_estimate
from ssgrep.accounting.valuation_inputs import Debit
from tests.accounting.test_ux_support import collected_scene

STAMP = "2026-10-12T20:00:00Z"


@pytest.fixture
def data(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and error is None and ledger.doc is not None
    return ledger


def evaluate(data, scope):
    return service_filter.evaluate("personal-local", scope, data, "snapshot", STAMP, 50, None)


def test_child_selection_revalues_only_linked_invocations_and_preserves_negative_net(data):
    task = TaskId("claude:root", "turn")
    unit = next(u for u in data.methodology_spec.units if u.unit == CONTEXT_TOKENS)
    template = data.doc.executions[0]
    doc = replace(
        data.doc,
        executions=(replace(template, runtime="claude", session_id="child", task=task),),
        session_roots=(("claude", "root", task.root), ("claude", "child", task.root)),
        session_parents=(("claude", "root", None), ("claude", "child", "root")),
        invocation_sessions=(("inside", "claude", "child"), ("outside", "claude", "root")),
    )
    debits = tuple(
        Debit(
            name,
            task,
            "2026-10-12",
            "exposure",
            unit,
            context_estimate(chars),
            exposure_id=name,
            chars=chars,
        )
        for name, chars in (("inside", 160), ("outside", 3600))
    )
    selected_data = replace(data, doc=doc, candidates=(), debits=debits)
    outcome = evaluate(selected_data, build_scope(session="child"))
    tokens = next(row for row in outcome.scenarios if row.unit_key == "context_tokens")
    assert [q.value for _, q in tokens.credit] == [0, 0, 0]
    assert [q.value for _, q in tokens.debit] == [40, 40, 40]
    assert [q.value for _, q in tokens.net] == [-40, -40, -40]
    assert outcome.estimates_reason is None
    assert outcome.coverage["candidate_receipts"] == 0
    unavailable = evaluate(
        replace(selected_data, doc=replace(doc, invocation_sessions=())),
        build_scope(session="child"),
    )
    assert unavailable.estimates_reason == "session_estimates_unavailable"
    assert all(q.value is None for row in unavailable.scenarios for _, q in row.net)


def test_session_candidate_receipts_are_restricted_to_selected_exposure(data):
    candidate = data.candidates[0]
    root = candidate.task.root
    runtime, session = root.split(":", 1)
    doc = replace(
        data.doc,
        session_roots=((runtime, session, root), (runtime, "child", root)),
        invocation_sessions=(("wanted", runtime, "child"), ("other", runtime, session)),
    )
    wanted = replace(candidate, exposure_ids=("wanted",))
    unrelated = replace(candidate, candidate_id="unrelated", exposure_ids=("other",))
    selected_data = replace(data, doc=doc, candidates=(wanted, unrelated), debits=())
    scope = build_scope(session="child")
    result = service_select.session_scope(
        scope, selected_data, service_select.plan(scope, selected_data, STAMP)
    )
    assert result is not None
    leaves, ids = result
    assert len(ids) == 1
    assert leaves and {leaf.task for leaf in leaves} == {candidate.task}
    from ssgrep.accounting.receipts import receipt_id_for

    assert ids == {receipt_id_for(candidate.task, candidate.candidate_id, data.methodology_spec)}
    entries = service_filter.receipt_entries(data, list(data.rows.periods[-1].leaves), only=set())
    assert entries == []


def test_absent_lineage_and_bad_execution_dates_never_guess_membership(data):
    missing = replace(data, doc=None)
    selection = service_select.plan(build_scope(session="unknown"), missing, STAMP)
    assert selection.executions == ()
    assert selection.estimates_reason == "session_lineage_unavailable"
    assert service_select.root_cwds(missing) == {}
    assert service_select.session_scope(build_scope(session="unknown"), missing, selection) is None
    bad = replace(data.doc.executions[0], ts_utc="not-a-date", cwd=None)
    selected = service_select.plan(
        build_scope(project="/unknown"),
        replace(data, doc=replace(data.doc, executions=(bad,))),
        STAMP,
    )
    assert selected.executions == ()
    assert service_select.plan(Scope(window="all"), data, "unknown").end is None


def test_project_filter_excludes_unknown_roots_and_missing_execution_cwd(data, tmp_path: Path):
    unknown = replace(
        data,
        doc=replace(
            data.doc, session_cwd=(), executions=(replace(data.doc.executions[0], cwd=None),)
        ),
    )
    outcome = evaluate(unknown, build_scope(project=str(tmp_path)))
    assert dict(outcome.measured)["searches"].value == 0
    assert all(q.value is None for row in outcome.scenarios for _, q in row.net)
    assert outcome.coverage["candidate_receipts"] == 0
    assert any(note.startswith("project_unknown:") for note in outcome.notes)


def test_conflicting_rollup_is_unknown_instead_of_partial_credit(data):
    leaf = data.rows.periods[-1].leaves[0]
    collision = replace(leaf, methodology="incompatible")
    rows = service_filter.scenario_rows([leaf, collision], {leaf.unit: "context_tokens"})
    tokens = rows[0]
    assert all(q.value is None and q.reason == "rollup_conflict" for _, q in tokens.net)
    assert all(q.basis is Basis.UNAVAILABLE for _, q in tokens.credit)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_noninteger_snapshot_quantities_are_unknown_not_coerced(value):
    quantity = parse_quantity({"value": value, "basis": "observed"}, "tokens", "absent")
    assert quantity.value is None and quantity.unit == "tokens"
    assert quantity.reason == "malformed_quantity" and quantity.basis is Basis.UNAVAILABLE


def test_invalid_receipt_id_is_excluded_while_valid_receipt_remains(data):
    record = data.rows.receipts[0]
    invalid = replace(record, receipt=replace(record.receipt, receipt_id="../foreign"))
    rows = replace(data.rows, receipts=(*data.rows.receipts, invalid))
    changed = replace(data, rows=rows)
    leaves = [leaf for period in data.rows.periods for leaf in period.leaves]
    entries = service_filter.receipt_entries(changed, leaves)
    assert entries
    assert record.receipt.receipt_id in {identity for _, identity, _, _ in entries}
    assert "../foreign" not in {identity for _, identity, _, _ in entries}
