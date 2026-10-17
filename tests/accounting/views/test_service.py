"""Read facade: honest state matrix, staleness boundary, and zero side effects."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from ssgrep.accounting.collect import state as collector_state, store as collector_store
from ssgrep.accounting.export import acks
from ssgrep.accounting.ledger.contracts import Basis, Scenario
from ssgrep.accounting.ledger.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotState,
)
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views import (
    diagnostics as service_delivery,
    filter as service_filter,
    history as service_history,
    ledger as service_ledger,
    records as service_records,
    selection as service_selection,
    service,
    service as service_status,
    snapshots,
)
from ssgrep.accounting.views.types import build_scope, parse_quantity
from tests.accounting.export.http_builders import tree_state, write_delivery
from tests.accounting.views.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
    q,
    scenarios,
    snapshot,
    window_7d,
    write_history,
    write_receipt,
    write_snapshot,
)
from tests.accounting.views.test_ux_support import NOW, collected_scene, fingerprint

T0 = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
DELIVERY_NOW = datetime(2026, 10, 12, 12, 0, 30, tzinfo=UTC)
DEST = DestinationPolicy("otel", "https://collector.example.test", (Signal.METRICS,))


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_unattributed_source_scope_is_allowed_only_for_personal_profile() -> None:
    from ssgrep.accounting.ledger.valuation_inputs import AtomStatus

    personal = service_filter.Scoper(ProfilePolicy())
    organization = service_filter.Scoper(ProfilePolicy(profile_id="acme"))

    assert personal.source_status(None) == (AtomStatus.OK, "")
    assert organization.source_status(None) == (
        AtomStatus.OUT_OF_SCOPE,
        service_filter.SOURCE_SCOPE_UNKNOWN,
    )


def test_paused_without_snapshot_points_to_resume(data: Path) -> None:
    enable_policy(paused=True)
    view = service.read_summary(now=T0)
    assert (view.state, view.paused) == (SnapshotState.PAUSED, True)
    assert view.next_actions[0].command == "ssgrep savings resume"


def test_future_collected_at_does_not_go_negative(data: Path) -> None:
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0 - timedelta(seconds=30))
    assert (view.age_seconds, view.stale) == (0, False)


def test_values_preserve_negatives_and_unknowns_as_none_with_reason(data: Path) -> None:
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0 + timedelta(seconds=10))
    assert [name for name, _ in view.measured] == ["searches", "shows", "failed", "visible_bytes"]
    measured = dict(view.measured)
    assert measured["searches"].value == 100 and measured["visible_bytes"].unit == "bytes"
    tokens = view.scenarios[0]
    nets = dict(tokens.net)
    assert nets[Scenario.CONSERVATIVE].value == -12000
    assert view.has_negative
    assert view.coverage["attributed_lookups"] == 91 and view.partial
    assert dict(view.exclusions) == {"prior_known": 4, "user_supplied": 2}


def test_missing_fields_become_unavailable_not_zero(data: Path) -> None:
    enable_policy()
    broken = window_7d()
    del broken["observations"]["failed"]
    broken["observations"]["shows"] = {"value": 3, "unit": "count", "basis": "bogus"}
    del broken["estimates"]["work_seconds"]
    del broken["coverage"]["ambiguous_joins"]
    write_snapshot(data, windows={"7d": broken})
    view = service.read_summary(now=T0)
    measured = dict(view.measured)
    assert measured["failed"].value is None and measured["failed"].reason == "not_recorded"
    assert measured["shows"].value is None and measured["shows"].reason == "malformed_quantity"
    work = next(row for row in view.scenarios if row.unit_key == "work_seconds")
    assert all(q.value is None and q.reason == "not_estimated" for _, q in work.net)
    assert view.coverage["ambiguous_joins"] is None


def test_float_in_snapshot_marks_it_corrupt_instead_of_coercing(data: Path) -> None:
    enable_policy()
    broken = window_7d()
    broken["observations"]["searches"] = {"value": 1.5, "unit": "count", "basis": "observed"}
    write_snapshot(data, windows={"7d": broken})
    view = service.read_summary(now=T0)
    assert view.state is SnapshotState.CORRUPT and not view.has_values


def test_unknown_window_and_filters_are_reported_not_faked(data: Path) -> None:
    enable_policy()
    write_snapshot(data, windows={})
    view = service.read_summary(now=T0)
    assert not view.has_values and any(w.startswith("window_missing") for w in view.warnings)
    write_snapshot(data)
    filtered = service.read_summary(scope=build_scope(runtime="claude"), now=T0)
    assert not filtered.has_values
    assert any(w.startswith("filters_unavailable") for w in filtered.warnings)


@pytest.mark.parametrize(
    ("content", "state"),
    [
        ("{broken", SnapshotState.CORRUPT),
        ("[]", SnapshotState.CORRUPT),
        (json.dumps({"snapshot_id": "x"}), SnapshotState.CORRUPT),
        (json.dumps({**snapshot(), "schema_version": 99}), SnapshotState.UNSUPPORTED_SCHEMA),
        (json.dumps({**snapshot(), "generation": None}), SnapshotState.CORRUPT),
    ],
)
def test_corrupt_and_foreign_snapshots_are_named_states_with_no_numbers(
    data: Path, content: str, state: SnapshotState
) -> None:
    enable_policy()
    target = write_snapshot(data)
    target.write_text(content)
    view = service.read_summary(now=T0)
    assert view.state is state and not view.has_values and view.reason


def test_unsafe_snapshot_permissions_are_corrupt_not_trusted(data: Path) -> None:
    enable_policy()
    target = write_snapshot(data)
    target.chmod(0o666)
    assert service.read_summary(now=T0).state is SnapshotState.CORRUPT


def test_symlinked_snapshot_is_refused(data: Path, tmp_path: Path) -> None:
    enable_policy()
    real = tmp_path / "real.json"
    real.write_text(json.dumps(snapshot()))
    real.chmod(0o600)
    target = write_snapshot(data)
    target.unlink()
    target.symlink_to(real)
    assert service.read_summary(now=T0).state is SnapshotState.CORRUPT


def test_profile_isolation_and_traversal(data: Path) -> None:
    enable_policy()
    write_snapshot(data)
    other = service.read_summary("other", now=T0)
    assert other.state is SnapshotState.DISABLED and not other.has_values
    with pytest.raises(PolicyError):
        service.read_summary("../personal-local")
    assert service.read_receipt("personal-local", "../../receipts/x") is None
    assert service.read_receipt("personal-local", "a/b") is None


def test_profile_flag_is_preserved_in_next_actions(data: Path) -> None:
    view = service.read_summary("acme", now=T0)
    assert view.next_actions[0].command == "ssgrep savings setup --profile acme"
    policy.enable_collection("acme", {"profile_id": "acme", "roots": [str(data.parent)]})
    write_snapshot(data, "acme")
    assert service.read_summary("acme", now=T0).next_actions[0].command == (
        f"ssgrep savings explain {RECEIPT_ID} --profile acme"
    )


def test_receipt_reads_hide_content_unless_requested_and_flag_expiry(data: Path) -> None:
    write_receipt(data)
    hidden = service.read_receipt("personal-local", RECEIPT_ID)
    assert hidden is not None and hidden.content is None and hidden.revision == 2
    assert dict(hidden.costs)["source_cost"].basis is Basis.UNAVAILABLE
    shown = service.read_receipt("personal-local", RECEIPT_ID, include_content=True)
    assert shown is not None and shown.content == "SECRET-CONTENT-CANARY"
    write_receipt(data, evidence_available=False)
    expired = service.read_receipt("personal-local", RECEIPT_ID, include_content=True)
    assert expired is not None and expired.content is None
    assert "evidence_expired" in (expired.note or "")
    assert service.read_receipt("personal-local", "missing") is None


def test_history_counts_only_current_revision_and_paginates_stably(data: Path) -> None:
    def row(period: str, revision: int, value: int) -> dict:
        return {
            "period": period,
            "revision": revision,
            "status": "sealed" if revision == 1 else "revised",
            "net": {"value": value, "unit": "tokens", "basis": "estimated"},
        }

    write_history(
        data,
        [
            row("2026-10-01", 1, 100),
            row("2026-10-01", 2, 70),
            row("2026-10-02", 1, -20),
            row("2026-10-03", 1, 5),
        ],
    )
    full = service.read_history(limit=100)
    assert [(e.period, e.revision, e.current) for e in full.entries] == [
        ("2026-10-01", 1, False),
        ("2026-10-01", 2, True),
        ("2026-10-02", 1, True),
        ("2026-10-03", 1, True),
    ]
    assert full.current_total.value == 70 - 20 + 5
    first = service.read_history(limit=2)
    assert first.next_cursor is not None
    second = service.read_history(limit=2, cursor=first.next_cursor)
    assert [e.period for e in first.entries + second.entries] == [e.period for e in full.entries]
    assert second.next_cursor is None
    with pytest.raises(PolicyError) as caught:
        service.read_history(limit=3, cursor=first.next_cursor)
    assert caught.value.condition == "invalid_cursor"
    ranged = service.read_history(since="2026-10-02", until="2026-10-03")
    assert [e.period for e in ranged.entries] == ["2026-10-02"]


@pytest.mark.parametrize("corruption", ["complete_flag", "credit_unit"])
def test_history_malformed_current_revision_is_unknown_not_zero(
    data: Path, corruption: str
) -> None:
    from tests.accounting.views.snapshot_builders import scenarios

    estimates = list(scenarios((10, 10, 10), (2, 2, 2), (8, 8, 8), "tokens").values())
    central = estimates[1]
    if corruption == "complete_flag":
        central["complete"] = False
    else:
        central["credit"]["unit"] = "bytes"
    write_history(
        data,
        [{"period": "2026-10-01", "revision": 1, "status": "sealed", "scenarios": estimates}],
    )
    history = service.read_history()
    assert len(history.entries) == 1
    assert history.entries[0].net.value is None
    assert history.entries[0].net.basis is Basis.UNAVAILABLE
    assert history.entries[0].net.reason == "malformed_valuation_uncertainty"
    assert history.current_total.value is None
    assert history.current_total.basis is Basis.UNAVAILABLE


def test_history_half_open_date_range_excludes_until_day_and_reports_unknown_current_total(
    data: Path,
) -> None:
    def row(period: str, revision: int, net: dict[str, object]) -> dict:
        return {
            "period": period,
            "revision": revision,
            "status": "sealed",
            "net": net,
        }

    write_history(
        data,
        [
            row("2026-10-01", 1, {"value": 10, "unit": "tokens", "basis": "estimated"}),
            row("2026-10-02", 1, {"value": 0, "unit": "tokens", "basis": "estimated"}),
            row("2026-10-03", 1, {"value": 4, "unit": "tokens", "basis": "estimated"}),
        ],
    )
    view = service.read_history(since="2026-10-01", until="2026-10-03")
    assert [entry.period for entry in view.entries] == ["2026-10-01", "2026-10-02"]
    assert view.current_total.value == 10
    assert view.entries[1].net.value == 0


def test_history_without_data_is_empty_with_reason(data: Path) -> None:
    view = service.read_history()
    assert view.entries == () and view.reason == "no_closed_periods"
    assert view.current_total.value is None and not data.exists()


def test_scope_validation_rejects_before_any_read() -> None:
    cases: list[dict[str, Any]] = [
        {"since": "yesterday"},
        {"since": "2026-10-05", "until": "2026-10-05"},
        {"until": "2026-13-01"},
        {"runtime": "vim"},
        {"by": "colour"},
        {"include_children": True},
    ]
    for kwargs in cases:
        with pytest.raises(PolicyError) as caught:
            build_scope(**kwargs)
        assert caught.value.exit_code == 2
    scope = build_scope(
        since="2026-10-01", until="2026-10-02T01:00:00+01:00", session="s1", include_children=True
    )
    assert (scope.since, scope.until) == ("2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z")


def test_session_selection_distinguishes_parent_children_and_unsplittable_child(
    ledger_data,
) -> None:
    assert ledger_data.doc is not None
    sessions = list(ledger_data.doc.session_roots)
    assert len(sessions) >= 2
    (runtime, parent, _), (child_runtime, child, _) = sessions[:2]
    assert child_runtime == runtime and child != parent

    root = f"{runtime}:{parent}"
    roots = tuple(
        (rt, session, root if rt == runtime and session in {parent, child} else existing_root)
        for rt, session, existing_root in sessions
    )
    parents = tuple(
        (rt, session, parent if rt == runtime and session == child else None)
        for rt, session, _ in sessions
    )
    lineage = replace(
        ledger_data,
        doc=replace(ledger_data.doc, session_roots=roots, session_parents=parents),
    )
    now = "2026-10-12T20:00:00Z"

    parent_only = service_selection.plan(build_scope(session=parent), lineage, now)
    parent_with_children = service_selection.plan(
        build_scope(session=parent, include_children=True), lineage, now
    )
    child_only = service_selection.plan(build_scope(session=child), lineage, now)

    assert {row.session_id for row in parent_only.executions} == {parent}
    assert parent_only.estimates_reason == "children_not_separable"
    assert {row.session_id for row in parent_with_children.executions} == {parent, child}
    assert parent_with_children.roots == frozenset({root})
    assert parent_with_children.estimates_reason is None
    assert {row.session_id for row in child_only.executions} == {child}
    assert child_only.estimates_reason == "child_session_not_separable"
    assert child_only.roots is None


def test_missing_parent_edges_do_not_expand_child_selection(ledger_data) -> None:
    assert ledger_data.doc is not None
    runtime, child, _ = ledger_data.doc.session_roots[0]
    root = f"{runtime}:parent"
    roots = tuple((rt, session, root) for rt, session, _ in ledger_data.doc.session_roots)
    partial = replace(
        ledger_data,
        doc=replace(ledger_data.doc, session_roots=roots, session_parents=()),
    )

    selected = service_selection.plan(
        build_scope(runtime=runtime, session=child, include_children=True),
        partial,
        "2026-10-12T20:00:00Z",
    )
    assert {row.session_id for row in selected.executions} == {child}
    assert selected.notes == (
        "session_parents_unavailable: children of a child session are unknown",
    )
    assert selected.estimates_reason == "child_session_not_separable"


def test_unknown_session_does_not_select_executions_or_estimate_roots(ledger_data) -> None:
    assert ledger_data.doc is not None
    selected = service_selection.plan(
        build_scope(session="not-retained-session"),
        ledger_data,
        "2026-10-12T20:00:00Z",
    )
    assert selected.executions == ()
    assert selected.roots == frozenset()
    assert selected.estimates_reason is None
    assert selected.notes == ("session_unknown: no retained session has that id",)


def test_project_selection_drops_unknown_cwds_instead_of_treating_them_as_matches(
    ledger_data,
) -> None:
    assert ledger_data.doc is not None and ledger_data.doc.session_cwd
    runtime, session, cwd = ledger_data.doc.session_cwd[0]
    scope = build_scope(runtime=runtime, project=cwd)
    selected = service_selection.plan(scope, ledger_data, "2026-10-12T20:00:00Z")
    known, unknown, _ = service_selection.select_leaves(scope, ledger_data, selected)
    assert known, "the retained session cwd provides a positive project-match control"
    assert unknown == 0

    missing_cwds = replace(ledger_data, doc=replace(ledger_data.doc, session_cwd=()))
    unavailable = service_selection.plan(scope, missing_cwds, "2026-10-12T20:00:00Z")
    absent, unknown, _ = service_selection.select_leaves(scope, missing_cwds, unavailable)
    assert absent == []
    assert unknown > 0, f"project-filtered leaves without cwd should be disclosed: {session}"


def test_selection_window_boundaries_are_half_open_and_invalid_rows_are_excluded(ledger_data):
    assert ledger_data.doc is not None and ledger_data.doc.executions
    row = replace(ledger_data.doc.executions[0], ts_utc="2026-10-12T19:30:00Z")
    before = replace(row, ts_utc="not-a-timestamp")
    boundary = replace(row, ts_utc="2026-10-12T20:00:00Z")
    data = replace(ledger_data, doc=replace(ledger_data.doc, executions=(before, row, boundary)))
    scope = build_scope(since="2026-10-12T19:00:00Z", until="2026-10-12T20:00:00Z")
    selected = service_selection.plan(scope, data, "2026-10-12T20:00:00Z")
    assert selected.executions == (row,)
    assert selected.end_inclusive is False
    assert selected.first_day == "2026-10-12" and selected.last_day == "2026-10-12"


def test_session_selection_without_retained_lineage_is_explicitly_unavailable(ledger_data):
    unavailable = replace(ledger_data, doc=None)
    selected = service_selection.plan(
        build_scope(session="unknown-session"), unavailable, "bad-collected-time"
    )
    assert selected.executions == ()
    assert selected.roots is None
    assert selected.estimates_reason == "session_lineage_unavailable"
    assert selected.start is None and selected.end is None
    assert selected.notes == ()


@pytest.fixture
def ledger_data(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0, 1))
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and error is None and ledger.doc is not None
    return ledger


def test_snapshot_ahead_of_ledger_never_exposes_unfiltered_totals(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    scope = build_scope(runtime="claude")
    before = service.read_summary(scope=scope, now=NOW)
    assert dict(before.measured)["searches"].value == 1
    assert before.generation is not None
    envelope = snapshots.read_generation("personal-local", before.generation)
    assert envelope is not None
    newer = snapshots.build_envelope(
        generation=envelope.generation + 1,
        counter_epoch=envelope.counter_epoch,
        methodology_id=envelope.methodology_id,
        policy_version=envelope.policy_version,
        collected_at=envelope.collected_at,
        coverage_through=envelope.coverage_through,
        source_manifest_digest=envelope.source_manifest_digest,
        windows=envelope.windows,
        coverage=envelope.coverage,
        observations=envelope.observations,
        estimates=envelope.estimates,
        warnings=envelope.warnings,
    )
    snapshots.publish("personal-local", newer)
    state = fingerprint(tmp_path / "data")
    selected = service.read_summary(scope=scope, now=NOW)
    assert not selected.has_values
    assert selected.generation == envelope.generation + 1
    assert any(warning.startswith("filters_unavailable:") for warning in selected.warnings)
    assert service.read_summary(now=NOW).has_values
    assert fingerprint(tmp_path / "data") == state


def test_absent_checkpoint_and_absent_reduction_are_distinct_without_writes(tmp_path: Path):
    assert service_ledger.load_reduction("personal-local") == (None, "no_checkpoint")
    assert not (tmp_path / "data" / "accounting").exists()
    collected_scene(tmp_path, pairs=(0,))
    checkpoint_path = (
        tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    )
    original = checkpoint_path.read_bytes()
    document = json.loads(original)
    document["blobs"].pop(collector_state.REDUCTION)
    checkpoint_path.write_text(json.dumps(document))
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_reduction("personal-local") == (None, "no_reduction")
    assert fingerprint(tmp_path / "data") == before
    checkpoint_path.write_bytes(original)
    doc, error = service_ledger.load_reduction("personal-local")
    assert doc is not None and doc.executions and error is None


def test_missing_ledger_does_not_erase_readable_execution_state(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    checkpoint_path = (
        tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    )
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    original = checkpoint_path.read_bytes()
    document = json.loads(original)
    document["blobs"].pop(collector_state.ledger_blob_name(checkpoint.methodology))
    checkpoint_path.write_text(json.dumps(document))
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_ledger("personal-local") == (None, "no_ledger")
    reduction, error = service_ledger.load_reduction("personal-local")
    assert reduction is not None and reduction.executions and error is None
    assert fingerprint(tmp_path / "data") == before
    checkpoint_path.write_bytes(original)
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and ledger.rows.periods and error is None


def test_checkpoint_cannot_relabel_ledger_as_another_methodology(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    path = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    original = path.read_bytes()
    raw = json.loads(original)
    wrong = "f" * len(checkpoint.methodology)
    original_name = collector_state.ledger_blob_name(checkpoint.methodology)
    raw["blobs"][collector_state.ledger_blob_name(wrong)] = raw["blobs"].pop(original_name)
    raw["methodology"] = wrong
    path.write_text(json.dumps(raw))
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_ledger("personal-local") == (None, "state_corrupt")
    assert fingerprint(tmp_path / "data") == before
    path.write_bytes(original)
    ledger, error = service_ledger.load_ledger("personal-local")
    assert ledger is not None and error is None
    assert ledger.methodology == checkpoint.methodology


def test_unsafe_committed_blob_directory_is_not_misclassified_as_corruption(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    directory = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "state"
    directory.chmod(0o777)
    before = fingerprint(tmp_path / "data")
    assert service_ledger.load_reduction("personal-local") == (None, "storage_unsafe")
    assert service_ledger.load_ledger("personal-local") == (None, "storage_unsafe")
    assert fingerprint(tmp_path / "data") == before
    assert directory.stat().st_mode & 0o777 == 0o777
    directory.chmod(0o700)
    doc, error = service_ledger.load_reduction("personal-local")
    assert error is None and doc is not None and doc.executions
    ledger, error = service_ledger.load_ledger("personal-local")
    assert error is None and ledger is not None and ledger.rows.periods


@pytest.mark.parametrize(
    ("seconds", "quantity"),
    [
        (180, None),
        (181, "3 min"),
        (7199, "119 min"),
        (7200, "2 h"),
        (172799, "47 h"),
        (172800, "2 d"),
    ],
)
def test_stale_age_boundary_preserves_elapsed_quantity_and_refresh_action(
    tmp_path: Path, seconds, quantity
):
    collected_scene(tmp_path, pairs=(0,))
    initial = service.read_summary(now=NOW)
    assert initial.collected_at is not None
    collected = datetime.fromisoformat(initial.collected_at)
    before = fingerprint(tmp_path / "data")
    view = service.read_summary(now=collected + timedelta(seconds=seconds))
    assert view.age_seconds == seconds
    assert view.stale is (seconds > 180)
    stale = [condition for condition in view.conditions if condition.code == "stale"]
    if quantity is None:
        assert stale == []
    else:
        assert len(stale) == 1
        assert quantity in stale[0].cause
        assert stale[0].action is not None
        assert stale[0].action.command == "ssgrep savings"
    assert view.measured == initial.measured and view.scenarios == initial.scenarios
    assert fingerprint(tmp_path / "data") == before


def test_unparseable_snapshot_age_stays_unknown_without_fabricating_staleness(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    path = tmp_path / "data" / "accounting" / "personal-local" / "snapshots" / "current.json"
    original = path.read_bytes()
    raw = json.loads(original)
    raw["collected_at"] = "2026-02-30T12:00:00Z"
    path.write_text(json.dumps(raw))
    before = fingerprint(tmp_path / "data")
    view = service.read_summary(now=NOW)
    assert view.age_seconds is None and not view.stale
    assert all(condition.code != "stale" for condition in view.conditions)
    assert view.has_values
    assert fingerprint(tmp_path / "data") == before
    path.write_bytes(original)
    recovered = service.read_summary(now=NOW)
    assert recovered.age_seconds is not None
    assert recovered.measured == view.measured and recovered.scenarios == view.scenarios


@pytest.mark.parametrize("field", ["session", "task", "project"])
@pytest.mark.parametrize("value", ["", "x\0y", "x" * 5000])
def test_invalid_scope_filter_is_rejected_before_any_read(field: str, value: str, tmp_path: Path):
    before = tree_state(tmp_path)
    with pytest.raises(PolicyError) as caught:
        build_scope(
            session=value if field == "session" else None,
            task=value if field == "task" else None,
            project=value if field == "project" else None,
        )
    assert caught.value.condition == "invalid_filter" and caught.value.exit_code == 2
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize(
    ("fields", "expected", "suffix"),
    [
        (
            {"blocked_reason": "policy_blocked", "pending_batches": 2},
            "blocked",
            "savings export preview otel --profile acme",
        ),
        (
            {"unknown_batches": 2},
            "unknown",
            "savings export check otel --profile acme",
        ),
        ({"pending_batches": 2}, "pending", "savings status --profile acme"),
        ({}, "waiting", "savings export preview otel --profile acme"),
    ],
)
def test_delivery_states_keep_profile_and_real_attempt_ages(fields, expected, suffix):
    state = acks.Delivery("otel", Signal.METRICS, last_attempt_at="2026-10-12T12:00:00Z", **fields)
    line = service_delivery._line("acme", "otel", "metrics", False, state, DELIVERY_NOW)
    assert line.state == expected and not line.enabled
    assert line.attempt_age_seconds == 30 and line.acked_age_seconds is None
    assert line.action is not None and line.action.command.endswith(suffix)
    assert line.pending_batches == fields.get("pending_batches", 0)
    assert line.unknown_batches == fields.get("unknown_batches", 0)
    assert line.acked_generation is None


def test_absent_ack_age_is_unknown_and_future_attempt_is_clamped():
    state = acks.Delivery(
        "otel", Signal.METRICS, last_acked_generation=2, last_attempt_at="2026-10-13T00:00:00Z"
    )
    line = service_delivery._line("acme", "otel", "metrics", True, state, DELIVERY_NOW)
    assert line.state == "acknowledged" and line.action is None
    assert line.acked_age_seconds is None and line.attempt_age_seconds == 0
    assert line.acked_generation == 2


def test_unreadable_delivery_is_not_waiting_and_is_never_repaired(tmp_path: Path):
    profile = ProfilePolicy(
        profile_id="acme", destinations=(DEST,), enabled_exports=(("otel", "metrics"),)
    )
    path = write_delivery("acme", "otel", "metrics", pending_batches=4)
    path.write_text("{broken")
    before = tree_state(tmp_path / "data")
    (line,) = service_delivery.read_delivery_lines("acme", profile, DELIVERY_NOW)
    assert line.state == "unreadable" and line.enabled
    assert line.action is not None and line.action.command == "ssgrep savings status --profile acme"
    assert tree_state(tmp_path / "data") == before
    path.unlink()
    acks.update_delivery("acme", "otel", Signal.METRICS, lambda d: replace(d, pending_batches=4))
    (recovered,) = service_delivery.read_delivery_lines("acme", profile, DELIVERY_NOW)
    assert recovered.state == "pending" and recovered.pending_batches == 4


def test_paused_profile_replaces_connectivity_check_with_resume_but_keeps_other_actions(
    tmp_path: Path,
):
    def lines(paused: bool, **fields):
        profile = ProfilePolicy(
            profile_id="acme",
            destinations=(DEST,),
            enabled_exports=(("otel", "metrics"),),
            paused=paused,
        )
        acks.update_delivery("acme", "otel", Signal.METRICS, lambda d: replace(d, **fields))
        (line,) = service_delivery.read_delivery_lines("acme", profile, DELIVERY_NOW)
        return line

    running = lines(False, unknown_batches=2)
    assert running.action is not None
    assert running.action.command == "ssgrep savings export check otel --profile acme"

    paused = lines(True, unknown_batches=2)
    assert "Sending is paused, so nothing is retried until resume." in paused.cause
    assert paused.action is not None
    assert paused.action.command == "ssgrep savings resume --profile acme"

    pending = lines(True, unknown_batches=0, pending_batches=3)
    assert pending.action is not None
    assert pending.action.command == "ssgrep savings status --profile acme"
    assert "Sending is paused" not in pending.cause


def test_sending_policy_failure_preserves_local_status_but_requires_policy_repair(tmp_path: Path):
    policy.enable_collection("personal-local")
    path = policy.policy_path()
    original = path.read_bytes()
    document = json.loads(original)
    document["sending_paused"] = {"personal-local": "scope_expansion"}
    path.write_text(json.dumps(document))
    before = tree_state(tmp_path / "data")
    status = service_status.read_status(now=DELIVERY_NOW)
    assert status.sending_paused_reason == "scope_expansion"
    assert status.policy.collection_enabled and not status.policy.paused
    # Repair needs an operator-supplied path: keep its template on the condition,
    # not among directly runnable actions. Local collection remains actionable.
    (repair,) = (condition for condition in status.conditions if condition.code == "sending_paused")
    assert repair.action is not None
    assert repair.action.command == "ssgrep savings setup <policy-file>"
    assert status.next_actions[0].command == "ssgrep savings"
    assert all("<" not in action.command for action in status.next_actions)
    summary = service.read_summary(now=DELIVERY_NOW)
    assert any(condition.code == "sending_paused" for condition in summary.conditions)
    assert tree_state(tmp_path / "data") == before
    path.write_text("{broken")
    broken = service.read_summary(now=DELIVERY_NOW)
    assert broken.reason == "policy_corrupt" and not broken.has_values
    path.write_bytes(original)
    assert service_status.read_status(now=DELIVERY_NOW).sending_paused_reason is None


def test_conflicting_rollup_is_unknown_instead_of_partial_credit(ledger_data):
    leaf = ledger_data.rows.periods[-1].leaves[0]
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


def test_summary_diagnostic_alias_count_excludes_grouped_warning_aliases(data: Path) -> None:
    enable_policy()
    write_snapshot(data, warnings=["late_data", "late_data"])
    view = service.read_summary(now=T0)
    assert view.diagnostic_total == 1
    assert view.diagnostic_alias_duplicates == 1


def test_empty_profile_status_uses_no_signals_without_creating_state(tmp_path: Path):
    before = fingerprint(tmp_path / "data")
    status = service_status.read_status("personal-local", now=DELIVERY_NOW)
    assert status.destinations == ()
    assert status.worker.state == "never_run"
    assert service_status._destination_state(()) == "no_signals"
    assert fingerprint(tmp_path / "data") == before


def test_load_lookup_data_keeps_generation_when_ledger_blob_is_missing(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
    path = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    original = path.read_bytes()
    raw = json.loads(original)
    raw["blobs"].pop(collector_state.ledger_blob_name(checkpoint.methodology))
    path.write_text(json.dumps(raw))
    before = fingerprint(tmp_path / "data")
    doc, generation, ledger, reason = service_ledger.load_lookup_data("personal-local")
    assert doc is not None and generation == checkpoint.generation
    assert ledger is None and reason == "no_ledger"
    assert fingerprint(tmp_path / "data") == before
    path.write_bytes(original)


def test_empty_summary_does_not_create_policy_or_snapshot_state(tmp_path: Path):
    before = fingerprint(tmp_path / "data")
    view = service.read_summary(now=T0)
    assert view.state is SnapshotState.DISABLED
    assert not view.has_values
    assert fingerprint(tmp_path / "data") == before


def test_ledger_reason_normalizes_retired_methodology_and_legacy_migration():
    legacy = ValueError("legacy_valuation_migration_required: old ledger")
    assert service_ledger._reason(legacy) == "legacy_valuation_migration_required"
    retired = ValueError("methodology_read_only")
    assert service_ledger._reason(retired) == "methodology_retired"


def _child_lineage(ledger_data):
    """cons1 is a child of cons0; src0 is a grandchild; unrelated src1 stays its own root."""
    doc = ledger_data.doc
    roots = (
        ("claude", "cons0", "claude:cons0"),
        ("claude", "cons1", "claude:cons0"),
        ("claude", "src0", "claude:cons0"),
        ("claude", "src1", "claude:src1"),
    )
    parents = (
        ("claude", "cons0", None),
        ("claude", "cons1", "cons0"),
        ("claude", "src0", "cons1"),
        ("claude", "src1", None),
    )
    return replace(ledger_data, doc=replace(doc, session_roots=roots, session_parents=parents))


def test_children_of_a_child_session_follow_parent_edges_transitively(ledger_data) -> None:
    lineage = _child_lineage(ledger_data)
    child = build_scope(session="cons1", include_children=True)
    keys, roots, reason, notes = service_selection._session_plan(child, lineage.doc)
    assert keys == {("claude", "cons1"), ("claude", "src0")}
    assert roots is None and reason == "child_session_not_separable" and notes == []
    # Positive control: without --children the walk never leaves the seed.
    alone = service_selection._session_plan(build_scope(session="cons1"), lineage.doc)[0]
    assert alone == {("claude", "cons1")}


def test_root_cwds_is_empty_without_a_reduction_document_and_maps_roots_otherwise(
    ledger_data,
) -> None:
    assert service_selection.root_cwds(ledger_data) == {
        "claude:cons0": "/work/demo",
        "claude:cons1": "/work/demo",
    }
    assert service_selection.root_cwds(replace(ledger_data, doc=None)) == {}


def test_child_session_estimates_are_revalued_over_its_own_lookups_only(ledger_data) -> None:
    lineage = _child_lineage(ledger_data)
    scope = build_scope(session="cons1")
    revalued = service_selection.session_scope(
        scope, lineage, service_selection.plan(scope, lineage, "2026-10-12T20:00:00Z")
    )
    assert revalued is not None
    leaves, receipt_ids = revalued
    assert leaves and {leaf.task.root for leaf in leaves} == {"claude:cons1"}
    wanted = {inv for inv, _, session in lineage.doc.invocation_sessions if session == "cons1"}
    owned = {c for c in lineage.candidates if set(c.exposure_ids) & wanted}
    assert len(receipt_ids) == len(owned) == 1

    outcome = service_filter.evaluate(
        "personal-local", scope, lineage, "snap", "2026-10-12T20:00:00Z", 20, None
    )
    assert outcome.estimates_reason is None
    assert any(note.startswith("session_revalued:") for note in outcome.notes)
    assert outcome.coverage["candidate_receipts"] == 1
    assert outcome.scenarios

    no_sessions = replace(lineage, doc=replace(lineage.doc, invocation_sessions=()))
    assert (
        service_selection.session_scope(
            scope, no_sessions, service_selection.plan(scope, no_sessions, "2026-10-12T20:00:00Z")
        )
        is None
    )
    unavailable = service_filter.evaluate(
        "personal-local", scope, no_sessions, "snap", "2026-10-12T20:00:00Z", 20, None
    )
    assert unavailable.estimates_reason == "session_estimates_unavailable"
    assert not any(note.startswith("session_revalued:") for note in unavailable.notes)


def test_receipt_entries_honour_the_revalued_receipt_set_and_reject_unsafe_ids(
    ledger_data,
) -> None:
    scope = build_scope()
    selected = service_selection.plan(scope, ledger_data, "2026-10-12T20:00:00Z")
    leaves, _, _ = service_selection.select_leaves(scope, ledger_data, selected)
    entries = service_filter.receipt_entries(ledger_data, leaves)
    assert entries, "positive control: the scene stores receipts for the selected task-days"
    kept = {receipt_id for _, receipt_id, _, _ in entries}
    assert service_filter.receipt_entries(ledger_data, leaves, only=set()) == []
    one = {next(iter(kept))}
    assert {i for _, i, _, _ in service_filter.receipt_entries(ledger_data, leaves, one)} == one

    first = ledger_data.rows.receipts[0]
    unsafe = replace(first, receipt=replace(first.receipt, receipt_id="../not-a-receipt-id"))
    tampered = replace(
        ledger_data,
        rows=replace(ledger_data.rows, receipts=(unsafe, *ledger_data.rows.receipts[1:])),
    )
    after = {i for _, i, _, _ in service_filter.receipt_entries(tampered, leaves)}
    assert "../not-a-receipt-id" not in after
    assert after == kept - {first.receipt.receipt_id}


def test_lookup_records_apply_window_session_project_and_task_filters(ledger_data) -> None:
    doc = ledger_data.doc
    now = "2026-10-12T20:00:00Z"

    def ids(scope, document=doc, at=now):
        return [
            (kind, ident)
            for kind, ident, _ in service_selection.lookup_records(scope, document, at)
        ]

    everything = ids(build_scope())
    assert everything == [
        ("execution", "n-start"),
        ("invocation", doc.invocations[0].invocation_id),
        ("invocation", doc.invocations[1].invocation_id),
    ]
    # inclusive 7-day window ends at collected_at: the later invocation is after it.
    assert ids(build_scope(), at="2026-10-12T09:30:00Z") == everything[:2]
    # explicit --until is exclusive.
    assert ids(build_scope(until="2026-10-12T09:00:10Z")) == [("execution", "n-start")]
    assert ids(build_scope(until="2026-10-12T09:00:11Z"))[-1] == everything[1]

    assert ids(build_scope(session="cons1")) == [everything[2]]
    assert ids(build_scope(task="claude:cons0")) == [everything[1]]
    assert ids(build_scope(task="claude:cons0#turn:u0")) == [everything[1]]
    assert ids(build_scope(project="/work/demo")) == everything[1:]
    assert ids(build_scope(project="/work/proj")) == [everything[0]]

    unknown_cwd = replace(doc.invocations[0], cwd=None)
    unlocated = replace(doc, invocations=(unknown_cwd, doc.invocations[1]))
    assert ids(build_scope(project="/work/demo"), unlocated) == [everything[2]]


def test_invocation_provenance_names_why_a_call_scope_is_not_trusted() -> None:
    from types import SimpleNamespace

    from ssgrep.accounting.ledger.contracts import Attribution
    from ssgrep.accounting.ledger.reduce_join import Link, Request
    from ssgrep.accounting.ledger.reduce_records import Call

    def request(trusted):
        return Request(
            "inv",
            0,
            "search",
            "cli",
            None,
            "/work/demo",
            None,
            None,
            1,
            None,
            None,
            fingerprint_key_id="key-1",
            cwd_trusted=trusted,
        )

    links = {"x1": Link(Attribution.UNATTRIBUTED, reasons=("insufficient_evidence",))}
    trusted = SimpleNamespace(cwd="/work/demo", requests=(request(True),), actor="verifier")
    got = service_filter.invocation_provenance(cast(Call, trusted), ["x1"], [], links)
    assert got["scope_basis"] == "native_call_metadata"
    assert got["join_rejection_reasons"] == ("insufficient_evidence",)
    assert got["actor_basis"] == "configured_verifier" and got["fingerprint_key_id"] == "key-1"

    untrusted = SimpleNamespace(cwd="/work/demo", requests=(request(False),), actor="main")
    got = service_filter.invocation_provenance(cast(Call, untrusted), [], [], {})
    assert got["scope_basis"] == "unknown_scope"
    assert got["join_rejection_reasons"] == ("untrusted_cwd",)
    assert got["actor_basis"] == "native_lineage"

    no_cwd = SimpleNamespace(cwd=None, requests=(), actor="main")
    got = service_filter.invocation_provenance(cast(Call, no_cwd), [], [], {})
    assert got["join_rejection_reasons"] == ("unknown_scope",)
    assert got["fingerprint_key_id"] is None


def test_scoped_journal_drops_events_recorded_for_another_profile(tmp_path: Path) -> None:
    from ssgrep.accounting.privacy.policy import ProfilePolicy
    from tests.accounting.ledger.reduce_builders import event, ts

    policy = ProfilePolicy(profile_id="personal-local", roots=(), collection_enabled=True)
    scoper = service_filter.Scoper(policy)
    mine = [replace(e, cwd=None, scope_basis="unknown_scope") for e in event("a", ts(12, 10))]
    other = [
        replace(e, cwd=None, scope_basis="unknown_scope")
        for e in event("b", ts(12, 11), profile="org-a")
    ]
    kept = service_filter.effective_scoped_journal([*mine, *other], (), scoper)
    assert kept and {e.execution_id for e in kept} == {"a"}
    assert {e.profile for e in kept} == {"personal-local"}


def test_original_publication_must_be_an_inactive_namespaced_row() -> None:
    from ssgrep.accounting.privacy.published import FORMAT

    base = {"period": "2026-10-01", "revision": 1, "published_format": FORMAT}
    bound = {
        **base,
        "current": False,
        "namespace": "ns-1",
        "net": {"value": 5, "basis": "observed"},
    }
    [entry] = service_records._history_entries([bound])
    assert entry.kind == "original_published" and not entry.current
    for bad in ({**bound, "current": True}, {**bound, "namespace": ""}, {**bound, "current": None}):
        with pytest.raises(ValueError, match="not bound to an inactive namespace"):
            service_records._history_entries([bad])


def test_lookup_data_reports_missing_checkpoint_reduction_and_corrupt_reduction(
    tmp_path: Path,
) -> None:
    assert service_ledger.load_lookup_data("personal-local") == (
        None,
        None,
        None,
        "no_checkpoint",
    )

    collected_scene(tmp_path, pairs=(0,))
    path = tmp_path / "data" / "accounting" / "personal-local" / "collector" / "checkpoint.json"
    original = path.read_bytes()
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    raw = json.loads(original)
    ledger_blob = raw["blobs"][collector_state.ledger_blob_name(checkpoint.methodology)]

    raw["blobs"].pop(collector_state.REDUCTION)
    path.write_text(json.dumps(raw))
    assert service_ledger.load_lookup_data("personal-local") == (
        None,
        checkpoint.generation,
        None,
        "lookup_metadata_unavailable",
    )

    raw["blobs"][collector_state.REDUCTION] = ledger_blob
    path.write_text(json.dumps(raw))
    assert service_ledger.load_lookup_data("personal-local") == (
        None,
        None,
        None,
        "unsupported_schema",
    )

    path.write_bytes(original)
    doc, generation, _, _ = service_ledger.load_lookup_data("personal-local")
    assert doc is not None and generation == checkpoint.generation


def test_status_diagnostics_count_only_distinct_warnings(data: Path) -> None:
    enable_policy()
    write_snapshot(data, warnings=["late_data", "late_data"])
    view = service_status.read_status(now=DELIVERY_NOW)
    assert view.warnings.count("late_data") == 2
    assert [group.count for group in view.diagnostic_groups] == [1]
    assert view.diagnostic_total == 1
    assert view.diagnostic_alias_duplicates == len(view.warnings) - 1


def test_project_filter_discloses_estimate_periods_whose_project_is_unrecorded(
    ledger_data,
) -> None:
    scope = build_scope(project="/work/demo")
    now = "2026-10-12T20:00:00Z"
    known = service_filter.evaluate("personal-local", scope, ledger_data, "snap", now, 20, None)
    assert not any(note.startswith("project_unknown:") for note in known.notes)
    candidates = known.coverage["candidate_receipts"]
    assert candidates is not None and candidates > 0

    blind = replace(ledger_data, doc=replace(ledger_data.doc, session_cwd=()))
    hidden = service_filter.evaluate("personal-local", scope, blind, "snap", now, 20, None)
    unknown_notes = [note for note in hidden.notes if note.startswith("project_unknown:")]
    assert len(unknown_notes) == 1 and "excluded from this project filter" in unknown_notes[0]
    assert hidden.coverage["candidate_receipts"] == 0


def _valid_scenarios() -> dict[str, Any]:
    return scenarios((10,) * 3, (3,) * 3, (7,) * 3, "context_tokens")


def test_scenario_estimates_roundtrip_and_sum_across_periods() -> None:
    parsed = service_history.parse_scenarios(list(_valid_scenarios().values()))
    assert [(e.scenario, e.net.value, e.complete) for e in parsed] == [
        (Scenario.CONSERVATIVE, 7, True),
        (Scenario.CENTRAL, 7, True),
        (Scenario.OPTIMISTIC, 7, True),
    ]
    total = service_history.sum_scenarios([parsed, parsed])
    assert [(e.credit.value, e.debit.value, e.net.value) for e in total] == [(20, 6, 14)] * 3
    assert service_history.parse_scenarios("not a list") == ()


def test_estimate_naming_the_wrong_scenario_is_malformed_not_reinterpreted() -> None:
    central = _valid_scenarios()["central"]
    mismatch = service_history.parse_estimate(central, Scenario.CONSERVATIVE, "context_tokens")
    assert mismatch.net.value is None
    assert mismatch.net.reason == "malformed_valuation_uncertainty"
    assert not mismatch.complete
    control = service_history.parse_estimate(central, Scenario.CENTRAL, "context_tokens")
    assert control.net.value == 7 and control.complete


def test_estimate_claiming_completeness_without_a_net_is_malformed() -> None:
    entry = {**_valid_scenarios()["central"], "net": q(None, "context_tokens", reason="gone")}
    parsed = service_history.parse_estimate(entry, Scenario.CENTRAL, "context_tokens")
    assert parsed.net.value is None
    assert parsed.net.reason == "malformed_valuation_uncertainty"
    assert not parsed.complete


def test_non_current_scenarios_drop_uncertainty_and_keep_only_raw_amounts() -> None:
    legacy = service_history.parse_scenarios(list(_valid_scenarios().values()), current=False)
    assert [e.credit.value for e in legacy] == [10, 10, 10]
    assert all(e.net.value is None for e in legacy)
    assert all(e.net.reason == service_history.LEGACY_REASON for e in legacy)


def test_scenario_lists_with_non_objects_or_duplicates_are_rejected() -> None:
    entries = list(_valid_scenarios().values())
    with pytest.raises(ValueError, match="scenario must be an object"):
        service_history.parse_scenarios([entries[0], "central"])
    with pytest.raises(ValueError, match="duplicate scenario"):
        service_history.parse_scenarios([entries[0], entries[0]])


def test_summing_periods_without_scenarios_is_unavailable_not_zero() -> None:
    parsed = service_history.parse_scenarios(list(_valid_scenarios().values()))
    total = service_history.sum_scenarios([(), parsed])
    assert len(total) == 3
    assert all(e.net.value is None for e in total)
    assert all(e.net.reason == "incompatible_or_legacy_methods" for e in total)
    assert all(e.net.unit == "mixed" for e in total)
