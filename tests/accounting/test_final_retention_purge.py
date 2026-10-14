"""Retention and privacy cleanup preserve committed delivery truth across read races."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import (
    acks,
    export_spool,
    plainfile,
    policy,
    purge,
    purge_spool,
    retention,
    retention_hooks,
    snapshots,
    storage,
    tombstones,
)
from ssgrep.accounting.ledger_contracts import ProfilePolicy, Signal
from tests.accounting.snapshots_driver import gen_name, make_envelope
from tests.accounting.test_purge import (
    build_scene,
    configure_policy,
    export_all,
    old_record_ids,
    select,
    tree,
)
from tests.accounting.test_retention import (
    DEST,
    NOW,
    NOW_Z,
    PROFILE,
    acked_through,
    delivery,
    eff,
    put,
    seed_delivery,
)
from tests.accounting.test_tombstones import DEST as TOMBSTONE_DEST, make_policy


def test_naive_acknowledged_snapshot_time_cannot_silently_expire_original_data(
    tmp_path: Path,
) -> None:
    before = acked_through(1)
    directory = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    target = directory / gen_name(1)
    raw = json.loads(target.read_bytes())
    raw["collected_at"] = "2026-10-01T00:00:00"
    target.write_text(json.dumps(raw))
    original = tree(tmp_path / "data")
    with pytest.raises(ValueError, match="timezone-aware"):
        retention.expire_acknowledged_snapshots(
            PROFILE, DEST, Signal.METRICS, now=NOW, effective=eff()
        )
    assert tree(tmp_path / "data") == original
    assert delivery() == before
    raw["collected_at"] = "2026-10-01T00:00:00Z"
    target.write_text(json.dumps(raw))
    assert not retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=NOW, effective=eff()
    )
    assert delivery() == before


@pytest.mark.parametrize("field", ["evidence_retention_days", "receipt_retention_days"])
def test_negative_retention_horizon_never_produces_a_future_expiry_cutoff(field: str) -> None:
    policy_value = replace(ProfilePolicy(), **{field: -1})
    with pytest.raises(ValueError, match="must not be negative"):
        retention_hooks.plan_expiry(
            now=NOW, policy=policy_value, root_last_evidence={"root": NOW_Z}
        )
    valid = retention_hooks.plan_expiry(
        now=NOW, policy=ProfilePolicy(), root_last_evidence={"root": NOW_Z}
    )
    assert valid.keep_roots == ("root",) and valid.expire_roots == ()


@pytest.mark.parametrize("when", ["before-mark", "after-mark"])
def test_spool_disappearing_after_plan_does_not_invent_loss_or_touch_other_payload(
    monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    doomed, kept = put(1), put(2)
    body = export_spool.read_body(PROFILE, DEST, Signal.METRICS, kept.batch_id)
    plan = purge_spool.SpoolPlan((doomed,))
    update = export_spool.update_meta

    def remove_planned(profile, destination, signal, batch_id, change):
        marked = update(profile, destination, signal, batch_id, change)
        export_spool.delete_batch(profile, destination, signal, batch_id)
        return marked

    with monkeypatch.context() as race:
        if when == "before-mark":
            export_spool.delete_batch(PROFILE, DEST, Signal.METRICS, doomed.batch_id)
        else:
            race.setattr(export_spool, "update_meta", remove_planned)
        assert purge_spool.apply_spool(PROFILE, plan, NOW_Z) == (0, 0)
    state = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert state is None or (state.lost_records == 0 and state.loss_summary == ())
    assert export_spool.read_body(PROFILE, DEST, Signal.METRICS, kept.batch_id) == body
    assert purge_spool.apply_spool(PROFILE, purge_spool.SpoolPlan((kept,)), NOW_Z) == (1, 0)
    assert delivery().lost_records == kept.items
    assert len(delivery().loss_summary) == 1


def test_ack_committed_before_meta_settlement_is_not_counted_as_delivery_loss() -> None:
    stale = put(3)
    original = seed_delivery(
        lambda d: acks.outcome(
            d,
            generation=stale.generation,
            snapshot_id=stale.snapshot_id,
            index=0,
            total=1,
            kind="acked",
            now=NOW_Z,
        )
    )
    queued = export_spool.read_meta(PROFILE, DEST, Signal.METRICS, stale.batch_id)
    assert queued is not None and queued.state == "queued"
    assert purge_spool.apply_spool(PROFILE, purge_spool.SpoolPlan((stale,)), NOW_Z) == (1, 0)
    after = delivery()
    assert (after.last_acked_generation, after.last_acked_snapshot_id, after.last_acked_at) == (
        original.last_acked_generation,
        original.last_acked_snapshot_id,
        original.last_acked_at,
    )
    assert after.lost_records == 0 and after.loss_summary == ()
    assert export_spool.list_batches(PROFILE, DEST, Signal.METRICS) == []


def test_pin_release_cannot_clear_a_new_ack_landing_after_delivery_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acked_through(1)
    newer = make_envelope(2)
    snapshots.publish(PROFILE, newer)
    update = acks.update_delivery

    def acknowledge_first(profile, destination, signal, change):
        update(
            profile,
            destination,
            signal,
            lambda d: acks.outcome(
                d,
                generation=2,
                snapshot_id=newer.snapshot_id,
                index=0,
                total=1,
                kind="acked",
                now=NOW_Z,
            ),
        )
        return update(profile, destination, signal, change)

    with monkeypatch.context() as race:
        race.setattr(acks, "update_delivery", acknowledge_first)
        assert purge_spool.release_pins(PROFILE, frozenset({newer.snapshot_id})) == 0
    resolved = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert resolved.state == "acknowledged" and resolved.envelope == newer
    assert delivery().last_acked_generation == 2
    assert purge_spool.release_pins(PROFILE, frozenset()) == 1
    assert delivery().last_acked_snapshot_id is None
    assert delivery().last_acked_generation == 2


@pytest.mark.parametrize("damage", [b"{torn", b"[]", b'{"schema_version":999}'])
def test_corrupt_metrics_delivery_cannot_hide_nonreversible_remote_data(
    tmp_path: Path, damage: bytes
) -> None:
    saved = make_policy()
    assert tombstones.plan_tombstones(PROFILE, saved, before_day=None, root_keys=None) == []
    plainfile.write_private(
        storage.data_root(),
        "accounting",
        PROFILE,
        "delivery",
        TOMBSTONE_DEST,
        "metrics.json",
        data=damage,
    )
    before = tree(tmp_path / "data")
    plans = tombstones.plan_tombstones(PROFILE, saved, before_day=None, root_keys=None)
    assert len(plans) == 1
    assert plans[0].destination_id == TOMBSTONE_DEST
    assert plans[0].non_reversible_metrics and plans[0].record_count == 0
    assert tree(tmp_path / "data") == before


def test_policy_revocation_after_plan_preserves_explicit_remote_deletion_obligations(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    configure_policy(logs_permitted=True)
    export_all(TOMBSTONE_DEST, ack=True)
    expected = tuple(old_record_ids(TOMBSTONE_DEST))
    planned = purge.plan(select())
    assert planned.tomb_plans and all(p.permitted for p in planned.tomb_plans)
    policy.save_profile(replace(planned.policy, enabled_exports=()))
    original = tree(tmp_path / "data")
    results = tombstones.enqueue_tombstones(PROFILE, planned.policy, planned.tomb_plans, now=NOW)
    report = purge.report(planned, performed=True, spool_removed=0, results=results)
    assert len(report.tombstones) == 1 and report.tombstones[0].status == "skipped"
    assert len(report.obligations) == 1
    owed = report.obligations[0]
    assert (owed.destination, owed.records, owed.record_ids) == (
        TOMBSTONE_DEST,
        len(expected),
        expected,
    )
    assert owed.reason == results[0].skipped_reason
    assert tree(tmp_path / "data") == original
