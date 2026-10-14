"""Recovery between durable export writes, not duplicates of the post/ack subprocess test."""

from __future__ import annotations

import json

import pytest

from ssgrep.accounting import export_spool, storage
from ssgrep.accounting.ledger_contracts import Signal
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, L, M
from tests.accounting.lab_receiver import LabReceiver, Respond


def configure(lab: LabReceiver, signal: Signal, *, records: int = 2) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(signal,))
    if signal is L:
        fx.commit_ledger(records)
    fx.publish(1)


@pytest.mark.parametrize("signal", [M, L])
def test_partial_build_before_registry_commit_is_recreated_before_any_dispatch(
    signal: Signal, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.split_small(monkeypatch)
    with LabReceiver.start() as lab:
        configure(lab, signal, records=8)
        real_write = storage.atomic_write

        def fail_second_meta(dir_fd, name, data):
            if name.endswith(".meta.json") and json.loads(data)["index"] == 1:
                raise OSError("second batch metadata write interrupted")
            return real_write(dir_fd, name, data)

        with monkeypatch.context() as fault:
            fault.setattr(storage, "atomic_write", fail_second_meta)
            interrupted = fx.cycle(signal)
        assert (interrupted.state, interrupted.reason, interrupted.sent) == (
            "failed",
            "storage_error",
            0,
        )
        (first,) = fx.metas(signal)
        assert first.total > 1 and first.attempts == 0 and first.state == "queued"
        first_body = fx.body_of(first)
        assert lab.requests == [] and fx.delivery(signal) is None
        if signal is L:
            assert fx.registry().entries == {}

        recovered = fx.cycle(signal, seconds=1)
        assert recovered.enqueued_batches == recovered.acked == first.total
        assert len(lab.requests) == first.total
        assert lab.requests[0].body == first_body
        assert [request.body for request in lab.requests] == [
            fx.body_of(meta) for meta in fx.metas(signal)
        ]
        assert all(meta.state == "acked" and meta.attempts == 1 for meta in fx.metas(signal))
        assert fx.state(signal).last_acked_generation == 1
        assert fx.state(signal).loss_summary == ()
        if signal is L:
            entries = list(fx.registry().entries.values())
            assert len(entries) == 8
            assert all(entry.acked and entry.acked.revision == 1 for entry in entries)
            assert all(not entry.queued and entry.unknown is None for entry in entries)
        assert fx.cycle(signal, seconds=2).sent == 0
        assert len(lab.requests) == first.total


@pytest.mark.parametrize("gap", ["registry", "registry-entry"])
def test_spooled_logs_with_incomplete_registration_are_rebuilt_identically(
    gap: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.split_small(monkeypatch)
    with LabReceiver.start() as lab:
        configure(lab, L, records=8)
        seeded = fx.cycle(L, deps=fx.deps(max_batches=0))
        original = fx.metas(L)
        assert len(original) > 1 and seeded.enqueued_batches == len(original)
        expected = [fx.body_of(meta) for meta in original]
        path = fx.profile_root() / "export" / DEST / "records.json"
        if gap == "registry":
            path.unlink()
        else:
            document = json.loads(path.read_bytes())
            document["records"].pop(next(iter(document["records"])))
            path.write_text(json.dumps(document))

        recovered = fx.cycle(L, seconds=1)
        assert recovered.enqueued_batches == recovered.acked == len(original)
        assert [request.body for request in lab.requests] == expected
        assert [meta.batch_id for meta in fx.metas(L)] == [m.batch_id for m in original]
        assert fx.state(L).last_acked_generation == 1 and fx.state(L).loss_summary == ()
        entries = list(fx.registry().entries.values())
        assert len(entries) == 8
        assert all(entry.acked and entry.acked.revision == 1 for entry in entries)
        assert all(not entry.queued and entry.unknown is None for entry in entries)


def test_attempted_incomplete_generation_is_not_rebuilt_as_a_new_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx.split_small(monkeypatch)
    with LabReceiver.start() as lab:
        configure(lab, M)
        lab.queue(Respond(503))
        assert fx.cycle().sent == 1
        original = fx.metas()
        assert len(original) > 1 and original[0].attempts == 1
        missing = original[-1]
        assert export_spool.delete_batch(PROFILE, DEST, M, missing.batch_id)
        surviving = [fx.body_of(meta) for meta in original[:-1]]

        recovered = fx.cycle(seconds=400)
        assert recovered.enqueued_batches == 0
        assert recovered.acked == len(surviving)
        assert [request.body for request in lab.requests[1:]] == surviving
        assert fx.state().last_acked_generation is None
        (progress,) = fx.state().progress
        assert progress.total == len(original) and missing.index not in progress.acked
        assert fx.cycle(seconds=800).sent == 0
        assert len(lab.requests) == 1 + len(surviving)


@pytest.mark.parametrize("signal", [M, L])
@pytest.mark.parametrize("boundary", ["before-attempt", "after-ack"])
def test_durable_write_failure_recovers_without_losing_or_changing_request_bytes(
    signal: Signal, boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LabReceiver.start() as lab:
        configure(lab, signal)
        real_write = storage.atomic_write
        failures = []

        def interrupted_write(dir_fd, name, data):
            if name == f"{signal.value}.json":
                fail_here = boundary == "before-attempt" and json.loads(data).get("in_flight")
            elif name.endswith(".meta.json"):
                fail_here = boundary == "after-ack" and json.loads(data).get("state") == "acked"
            else:
                fail_here = False
            if fail_here and not failures:
                failures.append(name)
                raise OSError("WRITE-FAILURE-CANARY")
            return real_write(dir_fd, name, data)

        with monkeypatch.context() as fault:
            fault.setattr(storage, "atomic_write", interrupted_write)
            interrupted = fx.cycle(signal)
        assert (interrupted.state, interrupted.reason) == ("failed", "storage_error")
        assert len(failures) == 1
        (pending,) = fx.metas(signal)
        assert pending.state == "in_flight" and pending.attempts == 1
        expected_body = fx.body_of(pending)
        state = fx.state(signal)
        if boundary == "before-attempt":
            assert lab.requests == []
            assert state.last_attempted_generation is None and state.last_acked_generation is None
        else:
            assert [request.body for request in lab.requests] == [expected_body]
            assert state.last_acked_generation == 1
        assert "WRITE-FAILURE-CANARY" not in repr(interrupted) + fx.everything_on_disk()

        restarted = fx.cycle(signal, seconds=1)
        assert (restarted.sent, restarted.acked) == (1, 1)
        count = 1 if boundary == "before-attempt" else 2
        assert [request.body for request in lab.requests] == [expected_body] * count
        (settled,) = fx.metas(signal)
        assert settled.state == "acked" and settled.attempts == 2
        final = fx.state(signal)
        assert final.last_acked_snapshot_id == pending.snapshot_id
        assert (final.pending_batches, final.unknown_batches, final.in_flight) == (0, 0, None)
        if signal is L:
            assert all(
                entry.acked and entry.unknown is None for entry in fx.registry().entries.values()
            )
        assert fx.cycle(signal, seconds=2).sent == 0 and len(lab.requests) == count


def test_corrupt_queued_body_is_quarantined_without_poisoning_a_later_snapshot() -> None:
    with LabReceiver.start() as lab:
        configure(lab, M)
        assert fx.cycle(deps=fx.deps(max_batches=0)).enqueued_batches == 1
        (original,) = fx.metas()
        body_path = fx.profile_root() / "spool" / DEST / M.value / f"{original.batch_id}.json"
        body_path.write_bytes(b"TAMPERED-BODY-CANARY")
        rejected = fx.cycle(seconds=1)
        assert rejected.sent == 0 and rejected.quarantined == 1
        assert lab.requests == []
        (quarantined,) = fx.metas()
        assert quarantined.state == "quarantined" and quarantined.attempts == 0
        assert quarantined.last_category == "spool_corrupt"
        assert fx.state().last_acked_generation is None
        assert fx.state().progress[0].quarantined == (0,)

        fresh_id = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        fresh = next(meta for meta in fx.metas() if meta.generation == 2)
        assert [request.body for request in lab.requests] == [fx.body_of(fresh)]
        assert b"TAMPERED-BODY-CANARY" not in lab.requests[0].body
        assert fx.state().last_acked_snapshot_id == fresh_id
        assert fx.state().quarantined_batches == 1
        assert fx.cycle(seconds=3).sent == 0 and len(lab.requests) == 1


def test_batch_removed_between_listing_and_body_read_is_not_dispatched_or_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        configure(lab, M)
        assert fx.cycle(deps=fx.deps(max_batches=0)).enqueued_batches == 1
        (original,) = fx.metas()
        expected_body = fx.body_of(original)
        real_read = export_spool.read_body
        removed = []

        def concurrent_removal(profile, destination, signal, batch_id):
            if not removed:
                removed.append(batch_id)
                export_spool.delete_batch(profile, destination, signal, batch_id)
            return real_read(profile, destination, signal, batch_id)

        with monkeypatch.context() as race:
            race.setattr(export_spool, "read_body", concurrent_removal)
            vanished = fx.cycle(seconds=1)
        assert removed == [original.batch_id]
        assert vanished.sent == 0 and vanished.acked == 0 and lab.requests == []
        assert fx.metas() == [] and fx.state().last_acked_generation is None
        assert fx.cycle(seconds=2).acked == 1
        assert [request.body for request in lab.requests] == [expected_body]


@pytest.mark.parametrize("missing", ["body", "meta", "both"])
@pytest.mark.parametrize("new_snapshot", [False, True])
def test_missing_registered_batch_blocks_rebuild_and_false_ack_until_original_files_restored(
    missing: str, new_snapshot: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.split_small(monkeypatch)
    with LabReceiver.start() as lab:
        configure(lab, L, records=8)
        assert fx.cycle(L, deps=fx.deps(max_batches=0)).enqueued_batches > 1
        original = fx.metas(L)
        expected_bodies = [fx.body_of(meta) for meta in original]
        lost = original[-1]
        folder = fx.profile_root() / "spool" / DEST / L.value
        body_path = folder / f"{lost.batch_id}.json"
        meta_path = folder / f"{lost.batch_id}.meta.json"
        removed = {}
        for path in (
            (body_path,)
            if missing == "body"
            else (meta_path,)
            if missing == "meta"
            else (body_path, meta_path)
        ):
            removed[path] = path.read_bytes()
            path.unlink()
        registry_path = fx.profile_root() / "export" / DEST / "records.json"
        registry_before = registry_path.read_bytes()
        if new_snapshot:
            fx.publish(2)

        for second in (1, 2):
            refused = fx.cycle(L, seconds=second, enqueue=second == 1)
            assert (refused.state, refused.reason, refused.sent, refused.acked) == (
                "failed",
                "spool_corrupt",
                0,
                0,
            )
            assert refused.enqueued_batches == 0
            assert registry_path.read_bytes() == registry_before
            assert fx.state(L).last_acked_generation is None
            assert fx.state(L).last_collected_generation == 1
            assert all(meta.attempts == 0 for meta in fx.metas(L))
            assert lab.requests == []
        # Do not fabricate the lost record's attempt history or reuse its revision. Restoring
        # the original immutable pair is a positive recovery control for the entire generation.
        for path, content in removed.items():
            path.write_bytes(content)
            path.chmod(0o600)
        recovered = fx.cycle(L, seconds=3)
        assert recovered.acked == len(original)
        assert [request.body for request in lab.requests] == expected_bodies
        assert fx.state(L).last_acked_generation == (2 if new_snapshot else 1)
        assert len(fx.registry().entries) == 8
        assert all(
            entry.acked and not entry.queued and entry.unknown is None
            for entry in fx.registry().entries.values()
        )
        assert fx.cycle(L, seconds=4).sent == 0
        assert len(lab.requests) == len(original)
