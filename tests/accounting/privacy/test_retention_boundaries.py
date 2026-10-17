"""Retention recovery preserves acknowledgments and records each discarded batch once."""

from __future__ import annotations

import errno
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

import ssgrep.accounting.views.snapshots as snapshots
from ssgrep.accounting.export import acks, spool as export_spool
from ssgrep.accounting.ledger.ledger_contracts import Signal
from ssgrep.accounting.privacy import retention, storage
from tests.accounting.export import delivery_fixtures as fx
from tests.accounting.export.lab_receiver import LabReceiver
from tests.accounting.export.snapshots_driver import gen_name, make_envelope
from tests.accounting.privacy.test_retention import (
    DEST,
    NOW,
    NOW_Z,
    PROFILE,
    acked_through,
    delivery,
    eff,
    put,
    seed_delivery,
    spool_dir,
)


@pytest.mark.parametrize("damage", [b"{torn", b"[]", b'{"schema_version":999}'])
def test_corrupt_acknowledged_generation_expires_without_substituting_current(
    tmp_path: Path, damage: bytes
) -> None:
    before = acked_through(7)
    snapshots.publish(PROFILE, make_envelope(8))
    directory = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    (directory / gen_name(7)).write_bytes(damage)
    assert retention.expire_acknowledged_snapshots(
        PROFILE, DEST, Signal.METRICS, now=NOW, effective=eff()
    )
    resolved = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert resolved.state == "acknowledged_snapshot_expired" and resolved.envelope is None
    after = delivery()
    assert (after.last_acked_generation, after.last_acked_at) == (
        before.last_acked_generation,
        before.last_acked_at,
    )
    assert after.expired_snapshot_id == make_envelope(7).snapshot_id
    assert snapshots.read_generation(PROFILE, 8) == make_envelope(8)
    assert snapshots.current_generation(PROFILE) == 8


def test_expiry_cannot_clear_a_newer_ack_that_lands_after_the_old_pin_was_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acked_through(7)
    original_read = snapshots.read_by_snapshot_id

    def acknowledge_between_reads(profile: str, snapshot_id: str):
        old = original_read(profile, snapshot_id)
        acked_through(8)
        return old

    with monkeypatch.context() as race:
        race.setattr(snapshots, "read_by_snapshot_id", acknowledge_between_reads)
        released = retention.expire_acknowledged_snapshots(
            PROFILE, DEST, Signal.METRICS, now=NOW + timedelta(days=401), effective=eff()
        )
    assert released is False
    resolved = acks.resolve_acknowledged(PROFILE, DEST, Signal.METRICS)
    assert resolved.state == "acknowledged" and resolved.envelope == make_envelope(8)
    assert delivery().expired_snapshot_id is None
    assert snapshots.read_generation(PROFILE, 7) == make_envelope(7)


@pytest.mark.parametrize("kind", ["data", "tombstone"])
@pytest.mark.parametrize("boundary", ["callback", "meta-unlink", "body-unlink"])
def test_interrupted_retention_replays_one_loss_then_erases_only_the_selected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, boundary: str
) -> None:
    acked = acked_through(1, Signal.LOGS)
    doomed = put(2, kind=kind, signal=Signal.LOGS, age=timedelta(days=9), items=4)
    kept = put(3, signal=Signal.LOGS, age=timedelta(days=1), items=2)
    other = put(4, signal=Signal.METRICS, age=timedelta(days=9), items=7)
    before_body = export_spool.read_body(PROFILE, DEST, Signal.LOGS, kept.batch_id)
    other_body = export_spool.read_body(PROFILE, DEST, Signal.METRICS, other.batch_id)
    doomed_body = export_spool.read_body(PROFILE, DEST, Signal.LOGS, doomed.batch_id)
    original_delete = storage.delete_file

    def fail_unlink(fd: int, name: str) -> bool:
        suffix = ".meta.json" if boundary == "meta-unlink" else ".json"
        if name == doomed.batch_id + suffix:
            raise OSError(errno.EIO, "interrupted spool deletion")
        return original_delete(fd, name)

    def discard(meta) -> None:
        if boundary == "callback":
            raise OSError(errno.EIO, "interrupted registry cleanup")

    with monkeypatch.context() as fault:
        if boundary != "callback":
            fault.setattr(storage, "delete_file", fail_unlink)
        with pytest.raises((OSError, export_spool.SpoolCorrupt)):
            retention.enforce_spool(
                PROFILE, DEST, Signal.LOGS, now=NOW, effective=eff(), on_discard=discard
            )
    after_fault = delivery(Signal.LOGS)
    assert after_fault.lost_records == 4
    assert after_fault.retention_pending_batch_id == doomed.batch_id
    assert (
        spool_dir(tmp_path, Signal.LOGS) / f"{doomed.batch_id}.json"
    ).read_bytes() == doomed_body
    result = retention.enforce_spool(
        PROFILE, DEST, Signal.LOGS, now=NOW, effective=eff(spool_days=30)
    )
    final = delivery(Signal.LOGS)
    assert final.lost_records == 4 and sum(loss.batches for loss in final.loss_summary) == 1
    assert final.retention_pending_batch_id is None
    assert (
        final.last_acked_generation,
        final.last_acked_snapshot_id,
        final.last_acked_at,
    ) == (acked.last_acked_generation, acked.last_acked_snapshot_id, acked.last_acked_at)
    assert result.records_lost == 0
    assert export_spool.read_meta(PROFILE, DEST, Signal.LOGS, doomed.batch_id) is None
    assert all(
        doomed.batch_id not in path.name for path in spool_dir(tmp_path, Signal.LOGS).iterdir()
    )
    assert export_spool.read_body(PROFILE, DEST, Signal.LOGS, kept.batch_id) == before_body
    assert export_spool.read_body(PROFILE, DEST, Signal.METRICS, other.batch_id) == other_body


def test_pending_loss_is_recovered_before_a_new_older_batch_can_replace_its_marker() -> None:
    first = put(2, kind="tombstone", signal=Signal.LOGS, age=timedelta(days=9), items=4)

    def interrupted(meta) -> None:
        raise OSError("registry unavailable")

    with pytest.raises(OSError):
        retention.enforce_spool(
            PROFILE, DEST, Signal.LOGS, now=NOW, effective=eff(), on_discard=interrupted
        )
    earlier = put(1, kind="tombstone", signal=Signal.LOGS, age=timedelta(days=10), items=2)
    seen: list[str] = []
    result = retention.enforce_spool(
        PROFILE,
        DEST,
        Signal.LOGS,
        now=NOW,
        effective=eff(),
        on_discard=lambda meta: seen.append(meta.batch_id),
    )
    assert seen == [first.batch_id, earlier.batch_id]
    assert result.records_lost == 2
    assert delivery(Signal.LOGS).lost_records == 6
    assert sum(loss.batches for loss in delivery(Signal.LOGS).loss_summary) == 2
    assert export_spool.list_batches(PROFILE, DEST, Signal.LOGS) == []


def test_conflicting_generation_progress_is_preserved_when_retention_records_loss() -> None:
    doomed = put(2, age=timedelta(days=9), state="in_flight", items=4)
    seed_delivery(
        lambda d: replace(
            acks.outcome(
                d, generation=2, snapshot_id="a" * 32, index=0, total=2, kind="acked", now=NOW_Z
            ),
            in_flight_nonrecallable=doomed.batch_id,
        )
    )
    before = delivery().progress
    result = retention.enforce_spool(PROFILE, DEST, Signal.METRICS, now=NOW, effective=eff())
    after = delivery()
    assert (result.records_lost, result.unknown_expired) == (4, 1)
    assert after.progress == before
    assert after.last_acked_generation is None and after.in_flight_nonrecallable is None
    assert after.loss_summary == (acks.Loss("delivery_unknown_expired", 1, 4, NOW_Z),)
    assert export_spool.read_meta(PROFILE, DEST, Signal.METRICS, doomed.batch_id) is None


def test_deleted_batch_recovers_a_failed_marker_clear_without_a_second_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doomed = put(2, kind="tombstone", signal=Signal.LOGS, age=timedelta(days=9), items=4)
    original_write = storage.atomic_write

    def fail_clear(fd: int, name: str, data: bytes) -> None:
        if name == "logs.json":
            document = json.loads(data)
            if document["lost_records"] and document["retention_pending_batch_id"] is None:
                raise OSError(errno.EIO, "marker clear interrupted")
        original_write(fd, name, data)

    with monkeypatch.context() as fault:
        fault.setattr(storage, "atomic_write", fail_clear)
        with pytest.raises(OSError, match="marker clear interrupted"):
            retention.enforce_spool(PROFILE, DEST, Signal.LOGS, now=NOW, effective=eff())
    assert delivery(Signal.LOGS).retention_pending_batch_id == doomed.batch_id
    assert export_spool.list_batches(PROFILE, DEST, Signal.LOGS) == []
    assert all(
        doomed.batch_id not in path.name for path in spool_dir(tmp_path, Signal.LOGS).iterdir()
    )
    retention.enforce_spool(PROFILE, DEST, Signal.LOGS, now=NOW, effective=eff())
    final = delivery(Signal.LOGS)
    assert final.retention_pending_batch_id is None and final.lost_records == 4
    assert final.loss_summary == (acks.Loss("tombstone_expired", 1, 4, NOW_Z),)


@pytest.mark.parametrize("bad_marker", [True, 7, [], "../outside", "x" * 129])
def test_pending_loss_marker_keeps_legacy_default_and_strict_file_validation(
    tmp_path: Path, bad_marker: object
) -> None:
    before = acked_through(1)
    path = tmp_path / "data" / "accounting" / PROFILE / "delivery" / DEST / "metrics.json"
    legacy = json.loads(path.read_bytes())
    del legacy["retention_pending_batch_id"]
    path.write_text(json.dumps(legacy))
    assert delivery() == before
    path.write_text(json.dumps({**legacy, "retention_pending_batch_id": bad_marker}))
    with pytest.raises(acks.DeliveryCorrupt):
        acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_real_send_lock_serializes_loss_and_restart_never_resends_committed_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.amend(evidence_retention_days=1)
        fx.publish(1)
        initial = fx.cycle(deps=fx.deps(post=fx.offline_post))
        assert initial.enqueued_batches == 1
        (doomed,) = fx.metas()
        old_body = fx.body_of(doomed)
        original_delete = storage.delete_file
        competitors: list[dict] = []
        later = 2 * 24 * 60 * 60

        def interrupt_after_competing_cycle(fd: int, name: str) -> bool:
            if name == doomed.batch_id + ".meta.json":
                pending = fx.state()
                assert pending.retention_pending_batch_id == doomed.batch_id
                rival = fx.run_driver("run", "metrics", str(later))
                assert rival.returncode == 0, rival.stderr
                competitors.append(json.loads(rival.stdout))
                assert fx.state() == pending  # the rival did not clobber the one-slot marker
                raise OSError(errno.EIO, "retention interrupted")
            return original_delete(fd, name)

        with monkeypatch.context() as fault:
            fault.setattr(storage, "delete_file", interrupt_after_competing_cycle)
            interrupted = fx.cycle(seconds=later)
        assert interrupted.state == "failed"
        assert competitors == [{"state": "busy", "reason": "send_in_progress", "sent": 0}]
        assert lab.requests == []
        assert fx.body_of(doomed) == old_body
        fx.amend(evidence_retention_days=30)  # the two-day-old batch is no longer age-expired
        restarted = fx.run_driver("run", "metrics", str(later + 1))
        assert restarted.returncode == 0, restarted.stderr
        assert json.loads(restarted.stdout)["sent"] == 0 and lab.requests == []
        assert fx.metas() == []
        final = fx.state()
        assert final.retention_pending_batch_id is None and final.lost_records == doomed.items
        assert final.last_acked_generation is None
        fx.publish(2, collected_at=fx.stamp(fx.at(later + 2)))
        sent = fx.cycle(seconds=later + 3)
        assert sent.acked == 1 and len(lab.requests) == 1
        assert lab.requests[0].body != old_body
        assert fx.state().last_acked_generation == 2
