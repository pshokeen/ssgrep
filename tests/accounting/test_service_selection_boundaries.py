"""Selection boundaries retain runtime identity and reject unsplittable lineage totals."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import collector_state, collector_store, service, service_ledger
from ssgrep.accounting.scope import build_scope
from ssgrep.accounting.service_select import plan
from tests.accounting.test_ux_support import NOW, collected_scene, fingerprint


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
