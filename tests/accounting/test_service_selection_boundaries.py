"""Selection boundaries retain runtime identity and reject unsplittable lineage totals."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import (
    collector_state,
    collector_store,
    service,
    service_filter,
    service_ledger,
    service_select,
)
from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.receipts import receipt_id_for
from ssgrep.accounting.scope import build_scope
from ssgrep.accounting.service_select import plan
from ssgrep.accounting.service_types import Scope
from ssgrep.accounting.units import CONTEXT_TOKENS, context_estimate
from ssgrep.accounting.valuation_inputs import Debit
from tests.accounting.test_ux_support import NOW, collected_scene, fingerprint

STAMP = "2026-10-12T20:00:00Z"


@pytest.fixture
def data(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and error is None and ledger.doc is not None
    return ledger


def evaluate(data, scope):
    return service_filter.evaluate("personal-local", scope, data, "snapshot", STAMP, 50, None)


@pytest.fixture
def lineage_data(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    data, error = service_ledger.load_ledger("personal-local")
    assert data is not None and error is None and data.doc is not None
    template = data.doc.executions[0]
    identities = (
        ("claude", "root", None),
        ("claude", "child", "root"),
        ("claude", "grandchild", "child"),
        ("claude", "sibling", "root"),
        ("codex", "child", None),
    )
    doc = replace(
        data.doc,
        executions=tuple(
            replace(
                template,
                execution_id=f"{runtime}:{session}",
                runtime=runtime,
                session_id=session,
                ts_utc="2026-10-12T12:00:00Z",
            )
            for runtime, session, _ in identities
        ),
        session_roots=tuple(
            (runtime, session, "claude:root" if runtime == "claude" else "codex:child")
            for runtime, session, _ in identities
        ),
        session_parents=identities,
    )
    return replace(data, doc=doc)


def test_child_descendants_are_transitive_but_never_include_siblings_or_other_runtimes(
    lineage_data,
) -> None:
    selected = plan(
        build_scope(runtime="claude", session="child", include_children=True),
        lineage_data,
        "2026-10-12T20:00:00Z",
    )
    assert {row.execution_id for row in selected.executions} == {
        "claude:child",
        "claude:grandchild",
    }
    assert selected.estimates_reason == "child_session_not_separable"
    assert selected.roots is None
    parent = plan(
        build_scope(runtime="claude", session="root", include_children=True),
        lineage_data,
        "2026-10-12T20:00:00Z",
    )
    assert {row.session_id for row in parent.executions} == {
        "root",
        "child",
        "grandchild",
        "sibling",
    }
    assert parent.roots == frozenset({"claude:root"}) and parent.estimates_reason is None
    alone = plan(build_scope(session="root"), lineage_data, "2026-10-12T20:00:00Z")
    assert [row.session_id for row in alone.executions] == ["root"]
    assert alone.estimates_reason == "children_not_separable"


def test_missing_parent_edges_never_guess_child_descendants(lineage_data) -> None:
    assert lineage_data.doc is not None
    missing = replace(lineage_data, doc=replace(lineage_data.doc, session_parents=()))
    selected = plan(
        build_scope(runtime="claude", session="child", include_children=True),
        missing,
        "2026-10-12T20:00:00Z",
    )
    assert [row.execution_id for row in selected.executions] == ["claude:child"]
    assert selected.notes == (
        "session_parents_unavailable: children of a child session are unknown",
    )
    assert selected.estimates_reason == "child_session_not_separable"


def test_nonobject_ledger_keeps_readable_reduction_available(tmp_path: Path) -> None:
    collected_scene(tmp_path, pairs=(0,))
    scope = build_scope(runtime="claude")
    control = service.read_summary(scope=scope, now=NOW)
    # start_capture writes n-start without a native invocation; only pair(0)'s
    # n-0 can join the Claude transcript and pass a runtime filter.
    initial, reason = service_ledger.load_reduction("personal-local")
    assert initial is not None and reason is None
    assert {row.execution_id: row.runtime for row in initial.executions} == {
        "n-start": None,
        "n-0": "claude",
    }
    assert dict(control.measured)["searches"].value == 1
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    name = collector_state.ledger_blob_name(checkpoint.methodology)
    path = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "state"
    target = path / checkpoint.blobs[name]
    original = target.read_bytes()
    target.write_text('{"state":[]}')
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_ledger("personal-local") == (None, "state_corrupt")
    reduction, reason = service_ledger.load_reduction("personal-local")
    assert reduction is not None and reason is None and reduction.executions
    selected = service.read_summary(scope=scope, now=NOW)
    assert not selected.has_values
    assert "filters_unavailable: the ledger cannot be read (state_corrupt)" in selected.warnings
    assert fingerprint(tmp_path / "data") == before
    target.write_bytes(original)
    recovered = service.read_summary(scope=scope, now=NOW)
    assert recovered.measured == control.measured
    assert recovered.scenarios == control.scenarios


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
