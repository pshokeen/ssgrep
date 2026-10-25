"""Fail-closed export state and consent boundaries, with real persistence and recovery."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from ssgrep.accounting.export import (
    acks,
    consent as export_consent,
    consent as export_policy,
    spool as export_spool,
)
from ssgrep.accounting.privacy import policy, storage
from tests.accounting.export import delivery_fixtures as fx
from tests.accounting.export.delivery_fixtures import DEST, PROFILE, L, M
from tests.accounting.export.lab_receiver import LabReceiver


def test_refused_lock_closes_its_directory_and_preserves_the_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = fx.profile_root() / "export" / DEST
    directory.mkdir(parents=True)
    target = tmp_path / "not-a-lock"
    target.write_bytes(b"private canary, not a lock")
    lock_path = directory / ".send-metrics.lock"
    lock_path.symlink_to(target)
    opened = []
    real_open = storage.open_profile_dir

    def track_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(storage, "open_profile_dir", track_open)
    with pytest.raises(OSError):
        with export_spool.locked(PROFILE, ("export", DEST), "send-metrics"):
            pytest.fail("a symlinked lock must not be acquired")
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF
    assert lock_path.is_symlink() and target.read_bytes() == b"private canary, not a lock"

    lock_path.unlink()
    with export_spool.locked(PROFILE, ("export", DEST), "send-metrics") as fd:
        assert fd is not None
        export_spool.write_json(fd, "control.json", {"sent": False, "generation": 1})
    assert json.loads((directory / "control.json").read_bytes()) == {
        "sent": False,
        "generation": 1,
    }


@pytest.mark.parametrize(
    "damage",
    [
        {"schema": True},
        {"digest": "a" * 63},
        {"digest": "G" * 64},
        {"changed_at": "2026-02-30T12:00:00Z"},
        {"changed_at": 17},
        {"collection_scope": {}},
    ],
    ids=["boolean-schema", "short-digest", "nonhex-digest", "bad-date", "nontext-date", "scope"],
)
def test_invalid_scope_state_installs_a_strict_recollection_barrier(damage: dict) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        assert fx.cycle(enqueue=False).sent == 0
        scope_path = fx.profile_root() / "export" / DEST / "scope-metrics.json"
        document = json.loads(scope_path.read_bytes())
        document.update(damage)
        scope_path.write_text(json.dumps(document))
        fx.publish(1, collected_at=fx.stamp(fx.at()))

        refused = fx.cycle()
        assert (refused.state, refused.reason, refused.sent) == ("idle", "ineligible_scope", 0)
        assert lab.requests == [] and fx.metas() == []
        recovered_scope = export_policy.read_scope(PROFILE, DEST, M)
        assert recovered_scope is not None and recovered_scope.changed_at == fx.stamp(fx.at())
        # An equal timestamp is not evidence that collection followed recovery. A later cycle
        # alone cannot make the same snapshot eligible; the persisted barrier must survive it.
        assert fx.cycle(seconds=1).sent == 0
        assert export_policy.read_scope(PROFILE, DEST, M) == recovered_scope
        snapshot_id = fx.publish(2, collected_at=fx.stamp(fx.at(2)))
        assert fx.cycle(seconds=3).acked == 1
        assert len(lab.requests) == 1
        assert lab.requests[0].body == fx.body_of(fx.metas()[0])
        assert fx.state().last_acked_snapshot_id == snapshot_id


def test_disable_survives_unsafe_spool_and_corrupt_ack_without_erasing_either(
    tmp_path: Path,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        assert fx.cycle(deps=fx.deps(max_batches=0)).enqueued_batches == 1
        (meta,) = fx.metas()
        expected_body = fx.body_of(meta)
        delivery_path = fx.profile_root() / "delivery" / DEST / "metrics.json"
        saved_delivery = delivery_path.read_bytes()
        delivery_path.write_bytes(b"{broken ACK-STATE-CANARY")
        spool_path = fx.profile_root() / "spool" / DEST / "metrics"
        saved_spool = tmp_path / "saved-spool"
        spool_path.rename(saved_spool)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "sentinel").write_bytes(b"do not inspect or erase")
        spool_path.symlink_to(outside, target_is_directory=True)
        before = fx.tree(saved_spool)

        report = export_consent.apply_disable(PROFILE, DEST, M)
        assert report.was_enabled and report.pending_batches is None
        assert report.in_flight is None and report.in_flight_nonrecallable is None
        assert policy.load_profile_policy(PROFILE).enabled_exports == ()
        assert delivery_path.read_bytes() == b"{broken ACK-STATE-CANARY"
        assert spool_path.is_symlink() and fx.tree(saved_spool) == before
        assert fx.tree(outside) == {"sentinel": b"do not inspect or erase"}
        assert lab.requests == []

        spool_path.unlink()
        saved_spool.rename(spool_path)
        delivery_path.write_bytes(saved_delivery)
        assert fx.cycle(seconds=1).state == "not_enabled"
        assert lab.requests == []
        assert export_consent.apply_enable(PROFILE, DEST, M)
        assert fx.cycle(seconds=2).acked == 1
        assert [request.body for request in lab.requests] == [expected_body]


@pytest.mark.parametrize("bad_queue", [None, "queued", "duplicate", "descending"])
def test_malformed_registry_queue_never_sends_and_restored_state_delivers_original_bytes(
    bad_queue: str | None,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(2)
        fx.publish(1)
        assert fx.cycle(L, deps=fx.deps(max_batches=0)).enqueued_batches == 1
        (meta,) = fx.metas(L)
        expected_body = fx.body_of(meta)
        path = fx.profile_root() / "export" / DEST / "records.json"
        original = path.read_bytes()
        document = json.loads(original)
        entry = next(iter(document["records"].values()))
        queued = entry["queued"][0]
        if bad_queue == "duplicate":
            entry["queued"] = [queued, queued]
        elif bad_queue == "descending":
            entry["queued"] = [{**queued, "revision": 2}, queued]
        else:
            entry["queued"] = bad_queue
        damaged = json.dumps(document).encode()
        path.write_bytes(damaged)

        failed = fx.cycle(L, seconds=1)
        assert (failed.state, failed.reason, failed.sent) == ("failed", "registry_invalid", 0)
        assert path.read_bytes() == damaged and lab.requests == []
        assert fx.metas(L) == [meta] and fx.body_of(meta) == expected_body
        assert fx.state(L).last_acked_generation is None
        path.write_bytes(original)
        assert fx.cycle(L, seconds=2).acked == 1
        assert [request.body for request in lab.requests] == [expected_body]
        assert all(entry.acked and not entry.queued for entry in fx.registry().entries.values())


@pytest.mark.parametrize("bad_indexes", [None, [0, 0], [1, 0]])
def test_corrupt_ack_progress_cannot_expose_an_acknowledged_snapshot(bad_indexes) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        snapshot_id = fx.publish(1)
        assert fx.cycle().acked == 1
        path = fx.profile_root() / "delivery" / DEST / "metrics.json"
        original = path.read_bytes()
        document = json.loads(original)
        document["progress"] = [
            {"generation": 2, "snapshot_id": snapshot_id, "total": 2, "acked": bad_indexes}
        ]
        damaged = json.dumps(document).encode()
        path.write_bytes(damaged)
        unavailable = acks.resolve_acknowledged(PROFILE, DEST, M)
        assert unavailable.state == "not_acknowledged"
        assert unavailable.delivery is None and unavailable.envelope is None
        with pytest.raises(acks.DeliveryCorrupt):
            acks.update_delivery(PROFILE, DEST, M, lambda d: acks.collected(d, 2))
        assert path.read_bytes() == damaged
        path.write_bytes(original)
        restored = acks.resolve_acknowledged(PROFILE, DEST, M)
        assert restored.state == "acknowledged" and restored.envelope is not None
        assert restored.envelope.snapshot_id == snapshot_id
        assert fx.cycle(seconds=1).sent == 0 and len(lab.requests) == 1


def test_spool_metadata_encoding_enforces_document_limit_after_validation() -> None:
    def meta(record_ids: int):
        return export_spool.Meta(
            batch_id="g0000000001-0001000100010001-0000",
            destination_id=DEST,
            signal=M,
            snapshot_id="00010001000100010001000100010001",
            generation=1,
            index=0,
            total=1,
            kind="data",
            created_utc="2026-10-13T10:00:00Z",
            bytes=1,
            sha256="0" * 64,
            items=record_ids,
            record_ids=("r" * 128,) * record_ids,
            scope={},
        )

    assert export_spool.encode_meta(meta(5_000))
    with pytest.raises(ValueError, match="meta exceeds"):
        export_spool.encode_meta(meta(10_000))
