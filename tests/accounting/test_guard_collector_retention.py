"""Aggregate expiry removes receipt files even if publication was interrupted after commit."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import collector_steps
from ssgrep.accounting.collector_types import CollectOptions
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_guard_collector_recovery import NOW, seeded


@pytest.mark.parametrize("interrupt", [False, True])
def test_expired_receipts_are_pruned_on_publish_or_committed_restart(
    tmp_path: Path, interrupt: bool
) -> None:
    seeded(tmp_path)
    receipts = vs.accounting_dir() / "receipts"
    original = {p.name: p.read_bytes() for p in receipts.glob("*.json")}
    assert original  # the no-resurrection assertion below has real credit to remove
    later = NOW + timedelta(days=401)

    def fail(step: str) -> None:
        if step == "checkpoint_committed":
            raise KeyboardInterrupt

    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            scene.collect(later, on_step=fail)
        assert {p.name: p.read_bytes() for p in receipts.glob("*.json")} == original
        assert scene.current().generation == 1
        assert scene.committed().retention["aggregates_expired_before"] is not None
    scene.collect(later)
    assert list(receipts.glob("*.json")) == []
    retention = scene.coverage(scene.current())["retention"]
    assert retention["aggregates_expired_before"] is not None
    assert retention["evidence_expired_before"] is not None
    assert retention["frozen_roots"] == 0  # the same cycle froze and then aged out the root
    assert scene.counter_total(scene.current()) == 2  # expiry is not a counter epoch reset
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0
    scene.collect(later)
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0
    scene.collect(later, CollectOptions(recompute=True))
    assert list(receipts.glob("*.json")) == []
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0


def test_receipt_pruning_uses_period_not_current_methodology_membership(tmp_path: Path) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir() / "receipts"
    original = {p.name: p.read_bytes() for p in directory.glob("*.json")}
    sample = json.loads(next(iter(original.values())))
    old = {**sample, "period": "2026-10-10", "methodology": "archived-methodology"}
    old_path = directory / "old-namespace.json"
    old_path.write_text(json.dumps(old))
    old_path.chmod(0o600)
    hidden = directory / ".unfinished.json"
    hidden.write_bytes(b"not a published receipt")
    hidden.chmod(0o600)
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-12") == 1
    assert not old_path.exists()
    assert {
        p.name: p.read_bytes() for p in directory.glob("*.json") if not p.name.startswith(".")
    } == original
    assert hidden.read_bytes() == b"not a published receipt"
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-13") == len(original)
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-13") == 0
