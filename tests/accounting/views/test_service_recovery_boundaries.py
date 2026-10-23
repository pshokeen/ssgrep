"""Read-only recovery keeps unknown quantities and committed-state failures visible."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.collect import state as collector_state, store as collector_store
from ssgrep.accounting.ledger.contracts import Basis
from ssgrep.accounting.ledger.units import UnitSpec
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views import ledger as service_ledger, service as service
from ssgrep.accounting.views.types import build_scope
from tests.accounting.export.http_builders import synthetic_current_scenarios
from tests.accounting.views.snapshot_builders import RECEIPT_ID, write_history, write_receipt
from tests.accounting.views.test_ux_support import (
    NOW,
    collected_scene,
    current_rarity_index,
    fingerprint,
)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("unsupported", "unsupported_schema"),
        ("checkpoint_json", "state_corrupt"),
        ("reduction_shape", "state_corrupt"),
        ("missing_blob", "state_corrupt"),
        ("unsafe", "storage_unsafe"),
    ],
)
def test_committed_state_damage_disables_filters_without_repair_or_unfiltered_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, reason: str
) -> None:
    current_rarity_index(tmp_path, monkeypatch)
    collected_scene(tmp_path, pairs=(0,))
    profile = "personal-local"
    scope = build_scope(runtime="claude", task="claude:cons0#turn:u0")
    control = service.read_summary(scope=scope, now=NOW)
    assert dict(control.measured)["searches"].value == 1
    assert control.scenarios[0].net[0][1].value is not None
    checkpoint_path = tmp_path / "data" / "accounting" / profile / "collector" / "checkpoint.json"
    checkpoint = collector_store.load_checkpoint(profile)
    assert checkpoint is not None
    saved_checkpoint = checkpoint_path.read_bytes()
    target = checkpoint_path
    restore = saved_checkpoint
    if damage == "unsupported":
        document = json.loads(saved_checkpoint)
        document["schema"] = 999
        checkpoint_path.write_text(json.dumps(document))
    elif damage == "checkpoint_json":
        checkpoint_path.write_text("{damaged")
    elif damage in ("reduction_shape", "missing_blob"):
        target = checkpoint_path.parent / "state" / checkpoint.blobs[collector_state.REDUCTION]
        restore = target.read_bytes()
        if damage == "missing_blob":
            target.unlink()
        else:
            target.write_text("[]")
    else:
        checkpoint_path.parent.chmod(0o777)
    before = fingerprint(tmp_path / "data")
    original_mode = checkpoint_path.parent.stat().st_mode
    with monkeypatch.context() as readonly:

        def forbidden_write(*args, **kwargs):
            pytest.fail("read view tried to write accounting state")

        readonly.setattr(storage, "atomic_write", forbidden_write)
        readonly.setattr(storage, "write_once", forbidden_write)
        assert service_ledger.load_ledger(profile) == (None, reason)
        assert service_ledger.load_reduction(profile) == (None, reason)
        view = service.read_summary(scope=scope, now=NOW)
        assert not view.has_values
        assert f"filters_unavailable: the ledger cannot be read ({reason})" in view.warnings
        assert service.read_summary(now=NOW).has_values
        assert fingerprint(tmp_path / "data") == before
        assert checkpoint_path.parent.stat().st_mode == original_mode
    checkpoint_path.parent.chmod(0o700)
    target.write_bytes(restore)
    target.chmod(0o600)
    recovered = service.read_summary(scope=scope, now=NOW)
    assert recovered.measured == control.measured
    assert recovered.scenarios == control.scenarios


@pytest.mark.parametrize(
    ("overrides", "condition", "code"),
    [
        ({"schema": True}, "receipt_unreadable", 1),
        ({"schema": 999}, "receipt_unsupported_schema", 3),
        ({"revision": "broken"}, "receipt_unreadable", 1),
        ({"allocations": [["central"]]}, "receipt_unreadable", 1),
    ],
)
def test_unreadable_receipt_is_not_confused_with_absent_and_never_rewritten(
    tmp_path: Path, overrides: dict[str, Any], condition: str, code: int
) -> None:
    root = tmp_path / "data"
    write_receipt(root, "acme", **overrides)
    before = fingerprint(root)
    with pytest.raises(PolicyError) as caught:
        service.read_receipt("acme", RECEIPT_ID, include_content=True)
    assert (caught.value.condition, caught.value.exit_code) == (condition, code)
    assert caught.value.command == "ssgrep savings status --profile acme"
    assert service.read_receipt("personal-local", RECEIPT_ID) is None
    assert fingerprint(root) == before
    write_receipt(root, "acme")
    recovered = service.read_receipt("acme", RECEIPT_ID)
    assert recovered is not None and recovered.revision == 2
    assert recovered.content is None
    assert dict(recovered.costs)["source_cost"].value is None


@pytest.mark.parametrize("unknown", [False, True])
def test_history_total_is_unknown_for_mixed_units_or_incomplete_current_revision(
    tmp_path: Path, unknown: bool
) -> None:
    root = tmp_path / "data"

    def estimates(value: int, *, seconds: bool = False, incomplete: bool = False) -> list:
        unit = (
            UnitSpec("work_seconds", "capped_gap_v1")
            if seconds
            else UnitSpec("context_tokens", "chars_div4_v1")
        )
        credit = max(value, 0)
        return list(
            synthetic_current_scenarios(
                (credit, credit, credit),
                max(-value, 0),
                unit,
                incomplete_credit=("conservative", "central", "optimistic") if incomplete else (),
            ).values()
        )

    write_history(
        root,
        [
            {"period": "2026-10-10", "revision": 1, "scenarios": estimates(99)},
            {"period": "2026-10-10", "revision": 2, "scenarios": estimates(-4)},
            {
                "period": "2026-10-11",
                "revision": 1,
                "scenarios": estimates(5, seconds=not unknown, incomplete=unknown),
            },
        ],
    )
    before = fingerprint(root)
    history = service.read_history(limit=1)
    assert history.next_cursor is not None
    assert history.current_total.value is None
    assert history.current_total.basis is Basis.UNAVAILABLE
    assert history.current_total.reason == (
        "incomplete_current_revisions" if unknown else "incompatible_or_legacy_methods"
    )
    assert history.current_total.unit == ("context_tokens" if unknown else "mixed")
    if unknown:
        for estimate in history.scenarios:
            assert not estimate.complete and estimate.known_net == 1
            assert estimate.bounds is not None
            assert estimate.bounds.net_lower.value == 1
            assert estimate.bounds.net_upper.value is None
    assert history.entries[0].current is False
    selected = service.read_history(since="2026-10-10", until="2026-10-11", limit=1)
    assert selected.current_total.value == -4 and selected.current_total.unit == "context_tokens"
    assert fingerprint(root) == before


def test_history_malformed_valuation_uncertainty_keeps_known_net_but_withholds_total(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    scenarios = synthetic_current_scenarios(
        (10, 10, 10), 1, UnitSpec("context_tokens", "chars_div4_v1")
    )
    scenarios["central"]["complete"] = False
    write_history(
        root,
        [{"period": "2026-10-10", "revision": 1, "scenarios": list(scenarios.values())}],
    )
    before = fingerprint(root)

    history = service.read_history()

    central = next(
        estimate
        for estimate in history.entries[0].scenarios
        if estimate.scenario.value == "central"
    )
    assert not central.complete and central.known_net == 9
    assert central.net.value is None
    assert central.net.reason == "malformed_valuation_uncertainty"
    assert history.current_total.value is None
    assert history.current_total.basis is Basis.UNAVAILABLE
    assert fingerprint(root) == before


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({"period": "2026-10-10", "revision": 1, "schema": 999}, "history_unsupported_schema"),
        ({"period": "2026-10-10", "revision": 1, "schema": True}, "history_unreadable"),
        ({"period": "2026-10-10", "revision": "broken"}, "history_unreadable"),
    ],
)
def test_history_schema_failures_preserve_evidence_and_recovery_action(
    tmp_path: Path, row: dict[str, Any], reason: str
) -> None:
    root = tmp_path / "data"
    write_history(root, [row], "acme")
    before = fingerprint(root)
    history = service.read_history("acme")
    assert history.reason == reason and history.entries == ()
    assert history.current_total.value is None and history.current_total.reason == reason
    assert history.next_actions[0].command == "ssgrep savings status --profile acme"
    assert fingerprint(root) == before
