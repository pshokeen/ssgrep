"""Read facade: honest state matrix, staleness boundary, and zero side effects."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import policy, service
from ssgrep.accounting.contracts import Basis, Scenario
from ssgrep.accounting.ledger_contracts import SnapshotState
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import build_scope
from tests.accounting.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
    snapshot,
    window_7d,
    write_history,
    write_receipt,
    write_snapshot,
)

T0 = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data"


def tree(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*")
    }


def test_disabled_without_any_storage_creates_nothing(data: Path) -> None:
    view = service.read_summary(now=T0)
    assert view.state is SnapshotState.DISABLED
    assert not view.has_values and view.measured == () and view.scenarios == ()
    assert [a.command for a in view.next_actions] == ["ssgrep metrics setup"]
    assert not data.exists()


def test_enabled_without_snapshot_shows_no_numbers_and_collect(data: Path) -> None:
    enable_policy()
    view = service.read_summary(now=T0)
    assert view.state is SnapshotState.NO_SNAPSHOT
    assert view.measured == () and view.scenarios == ()
    assert view.next_actions[0].command == "ssgrep metrics collect"


def test_paused_without_snapshot_points_to_resume(data: Path) -> None:
    enable_policy(paused=True)
    view = service.read_summary(now=T0)
    assert (view.state, view.paused) == (SnapshotState.PAUSED, True)
    assert view.next_actions[0].command == "ssgrep metrics resume"


@pytest.mark.parametrize(
    ("age", "state", "stale"),
    [
        (0, SnapshotState.FRESH, False),
        (38, SnapshotState.FRESH, False),
        (180, SnapshotState.FRESH, False),
        (181, SnapshotState.STALE, True),
        (86400, SnapshotState.STALE, True),
    ],
)
def test_staleness_boundary_is_three_intervals_exclusive(
    data: Path, age: int, state: SnapshotState, stale: bool
) -> None:
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0 + timedelta(seconds=age))
    assert (view.state, view.stale, view.age_seconds) == (state, stale, age)
    assert view.has_values, "last good values stay visible, never replaced by zeros"
    expected = "ssgrep metrics collect" if stale else f"ssgrep savings explain {RECEIPT_ID}"
    assert view.next_actions[0].command == expected


def test_future_collected_at_does_not_go_negative(data: Path) -> None:
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0 - timedelta(seconds=30))
    assert (view.age_seconds, view.stale) == (0, False)


def test_paused_with_snapshot_keeps_age_and_values_and_is_not_stale_state(data: Path) -> None:
    enable_policy(paused=True)
    write_snapshot(data)
    view = service.read_summary(now=T0 + timedelta(seconds=600))
    assert view.state is SnapshotState.PAUSED
    assert view.stale is True and view.age_seconds == 600 and view.has_values
    assert view.next_actions[0].command == "ssgrep metrics resume"


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


def test_reads_never_change_the_data_directory(data: Path) -> None:
    enable_policy()
    write_snapshot(data)
    write_receipt(data)
    write_history(
        data,
        [
            {
                "period": "2026-10-01",
                "revision": 1,
                "net": {"value": 5, "unit": "tokens", "basis": "estimated"},
            }
        ],
    )
    before = tree(data)
    for _ in range(3):
        service.read_summary(now=T0 + timedelta(seconds=500))
        service.read_receipt("personal-local", RECEIPT_ID, True)
        service.read_history()
        policy.policy_status()
    assert tree(data) == before


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
