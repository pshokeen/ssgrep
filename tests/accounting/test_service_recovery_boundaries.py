"""Read-only recovery keeps unknown quantities and committed-state failures visible."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_state, collector_store, service, service_ledger, storage
from ssgrep.accounting.contracts import Basis
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import build_scope
from tests.accounting.snapshot_builders import RECEIPT_ID, q, write_history, write_receipt
from tests.accounting.test_ux_support import NOW, collected_scene, fingerprint


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
    assert caught.value.command == "ssgrep metrics status --profile acme"
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
    write_history(
        root,
        [
            {"period": "2026-10-10", "revision": 1, "net": q(99, "tokens")},
            {"period": "2026-10-10", "revision": 2, "net": q(-4, "tokens")},
            {
                "period": "2026-10-11",
                "revision": 1,
                "net": q(None, "tokens", reason="missing_cost") if unknown else q(5, "s"),
            },
        ],
    )
    before = fingerprint(root)
    history = service.read_history(limit=1)
    assert history.next_cursor is not None
    assert history.current_total.value is None
    assert history.current_total.basis is Basis.UNAVAILABLE
    assert history.current_total.reason == "incomplete_or_mixed_units"
    assert history.current_total.unit == ("tokens" if unknown else "mixed")
    assert history.entries[0].current is False
    selected = service.read_history(since="2026-10-10", until="2026-10-11", limit=1)
    assert selected.current_total.value == -4 and selected.current_total.unit == "tokens"
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
    assert history.next_actions[0].command == "ssgrep metrics status --profile acme"
    assert fingerprint(root) == before
