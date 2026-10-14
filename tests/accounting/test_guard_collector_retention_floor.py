"""Widening retention cannot revive aggregates already removed at a committed cutoff."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from ssgrep.accounting import collector_state, policy
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_types import CollectOptions
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE

NOW = scene.at(hour=20)


def ledger_periods() -> set[str]:
    checkpoint = scene.committed()
    state = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.ledger is not None
    rows = VersionedLedger.restore_versioned(state.ledger, state.superseded).rows()
    return {row.period for row in rows.periods} | {row.period for row in rows.receipts}


def assert_floor_preserved(floor: str, expected_searches: int) -> None:
    checkpoint = scene.committed()
    assert checkpoint.retention["aggregates_expired_before"] == floor
    state = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.reduction is not None
    assert all(day >= floor for day in state.sealed)
    assert all(row.day >= floor for row in state.reduction.executions)
    assert all(day >= floor for day in ledger_periods())
    directory = vs.accounting_dir()
    history = json.loads((directory / "history" / "periods.json").read_bytes())
    assert all(row["period"] >= floor for row in history)
    receipts = [json.loads(path.read_bytes()) for path in (directory / "receipts").glob("*.json")]
    assert all(row["period"] >= floor for row in receipts)
    assert scene.quantity(scene.current(), "lifetime", "searches") == expected_searches


def test_widening_retention_and_recompute_never_resurrect_expired_observations_or_ledger(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    configured = policy.load_profile_policy(PROFILE)
    policy.save_profile(replace(configured, evidence_retention_days=30, receipt_retention_days=1))
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    scene.collect(NOW)
    assert scene.quantity(scene.current(), "lifetime", "searches") == 2
    assert "2026-10-12" in ledger_periods()
    assert list((vs.accounting_dir() / "receipts").glob("*.json"))
    retained_evidence = vs.evidence_identities()
    assert retained_evidence
    counter_total = scene.counter_total(scene.current())

    later = NOW + timedelta(days=10)
    scene.collect(later)
    floor = (later - timedelta(days=1)).date().isoformat()
    assert_floor_preserved(floor, 0)
    assert ledger_periods() == set()
    assert vs.evidence_identities() == retained_evidence  # still inside the 30-day raw horizon

    policy.save_profile(replace(policy.load_profile_policy(PROFILE), receipt_retention_days=400))
    scene.collect(later, CollectOptions(recompute=True))
    assert_floor_preserved(floor, 0)
    assert scene.counter_total(scene.current()) == counter_total
    scene.collect(later)
    assert_floor_preserved(floor, 0)

    # The cutoff is exclusive, not a ban on future data: an execution at its midnight
    # remains eligible while the old retained inputs are still presented to recompute.
    scene.journal_events(scene.execution("at-retention-floor", f"{floor}T00:00:00Z"))
    scene.collect(later, CollectOptions(recompute=True))
    assert_floor_preserved(floor, 1)
    assert scene.counter_total(scene.current()) == counter_total + 1
    scene.collect(later)
    assert_floor_preserved(floor, 1)
