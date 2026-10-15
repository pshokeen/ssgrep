"""Read facade: honest state matrix, staleness boundary, and zero side effects."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import (
    acks,
    collector_state,
    collector_store,
    policy,
    service,
    service_delivery,
    service_filter,
    service_ledger,
    service_status,
    snapshots,
)
from ssgrep.accounting.contracts import Basis, Scenario
from ssgrep.accounting.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotState,
)
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import build_scope
from ssgrep.accounting.service_types import parse_quantity
from tests.accounting.http_builders import tree_state, write_delivery
from tests.accounting.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
    snapshot,
    window_7d,
    write_history,
    write_receipt,
    write_snapshot,
)
from tests.accounting.test_ux_support import NOW, collected_scene, fingerprint

T0 = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
DELIVERY_NOW = datetime(2026, 10, 12, 12, 0, 30, tzinfo=UTC)
DEST = DestinationPolicy("otel", "https://collector.example.test", (Signal.METRICS,))


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_paused_without_snapshot_points_to_resume(data: Path) -> None:
    enable_policy(paused=True)
    view = service.read_summary(now=T0)
    assert (view.state, view.paused) == (SnapshotState.PAUSED, True)
    assert view.next_actions[0].command == "ssgrep metrics resume"


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
    assert view.next_actions[0].command == "ssgrep metrics setup --profile acme"
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


@pytest.fixture
def ledger_data(tmp_path: Path):
    collected_scene(tmp_path, pairs=(0,))
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
        assert stale[0].action.command == "ssgrep metrics collect"
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
            "--preview --profile acme",
        ),
        ({"unknown_batches": 2}, "unknown", "--check --profile acme"),
        ({"pending_batches": 2}, "pending", "metrics status --profile acme"),
        ({}, "waiting", "--preview --profile acme"),
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
    assert line.action is not None and line.action.command == "ssgrep metrics status --profile acme"
    assert tree_state(tmp_path / "data") == before
    path.unlink()
    acks.update_delivery("acme", "otel", Signal.METRICS, lambda d: replace(d, pending_batches=4))
    (recovered,) = service_delivery.read_delivery_lines("acme", profile, DELIVERY_NOW)
    assert recovered.state == "pending" and recovered.pending_batches == 4


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
    assert repair.action.command == "ssgrep metrics setup --policy <file>"
    assert status.next_actions[0].command == "ssgrep metrics collect"
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
