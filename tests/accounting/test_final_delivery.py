"""Real queued generations survive final orchestration and dispatch fault boundaries."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import (
    delivery,
    delivery_candidate,
    export_consent,
    export_consent_preview,
    export_files,
    export_http,
    export_ids,
    export_policy,
    export_spool,
    otlp,
    policy,
    snapshots,
    storage,
)
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, L, M
from tests.accounting.lab_receiver import LabReceiver, Respond


def test_no_snapshot_cycle_and_unreadable_preview_recover_after_collection(
    tmp_path: Path,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, credential_ref=None)
        assert export_consent.plan_enable(PROFILE, DEST, M).credential_ref is None
        empty = fx.cycle()
        assert (empty.state, empty.sent) == ("no_snapshot", 0)
        verdict = export_consent.classify_cycle(empty, PROFILE)
        assert not verdict.ok and verdict.condition == "no_snapshot"
        assert fx.delivery() is None and lab.requests == []
        directory = fx.profile_root() / "snapshots"
        saved = tmp_path / "snapshots"
        if directory.exists():
            directory.rename(saved)
        directory.symlink_to(tmp_path, target_is_directory=True)
        assert delivery_candidate.current_envelope(PROFILE) is None
        assert fx.cycle().state == "no_snapshot"
        directory.unlink()
        if saved.exists():
            saved.rename(directory)
        sid = fx.publish(1)
        assert fx.cycle(seconds=1).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert lab.requests[0].body == fx.body_of(fx.metas()[0])


def test_disabled_target_under_send_lock_is_busy_without_losing_queued_bytes() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        (meta,) = fx.metas()
        body = fx.body_of(meta)
        fx.set_enabled(M, False)
        with export_files.locked(PROFILE, ("export", DEST), "send-metrics"):
            busy = fx.cycle(seconds=1)
        assert (busy.state, busy.reason, busy.sent) == ("busy", "send_in_progress", 0)
        assert export_consent.classify_cycle(busy, PROFILE).condition == "export_busy"
        assert fx.body_of(meta) == body and lab.requests == []
        fx.set_enabled(M, True)
        assert fx.cycle(seconds=2).acked == 1
        assert [r.body for r in lab.requests] == [body]


@pytest.mark.parametrize("argument", ["profile", "destination", "clock"])
def test_programmer_argument_errors_create_no_export_state(argument: str) -> None:
    options: dict[str, Any] = {"profile": PROFILE, "destination": DEST, "signal": M, "now": fx.at()}
    if argument == "clock":
        options["now"] = datetime(2026, 10, 13)
    else:
        options[argument] = "../escape"
    before = fx.tree()
    with pytest.raises(ValueError):
        delivery.run_destination(**options)
    assert fx.tree() == before
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        assert fx.cycle().acked == 1 and len(lab.requests) == 1


@pytest.mark.parametrize("damage", ["short-key", "unsafe-key", "too-large", "unexpected-write"])
def test_failed_enqueue_exposes_only_stable_code_then_delivers_after_repair(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "short-key": "storage_unsafe",
        "unsafe-key": "storage_unsafe",
        "too-large": "export_too_large",
        "unexpected-write": "internal_error",
    }[damage]
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        sid = fx.publish(1)
        export_ids.load_key(PROFILE, create=True)
        key_path = fx.profile_root() / "export" / "pseudonym.key"
        key = key_path.read_bytes()
        with monkeypatch.context() as fault:
            if damage == "short-key":
                key_path.write_bytes(b"DAMAGED-KEY")
            elif damage == "unsafe-key":
                target = tmp_path / "secret"
                key_path.rename(target)
                key_path.symlink_to(target)
            elif damage == "too-large":
                real_split = otlp.split_metrics
                fault.setattr(otlp, "split_metrics", lambda r, p: real_split(r, p, max_bytes=1))
            else:
                real_write = storage.atomic_write

                def failed_write(fd, name, data):
                    if name.endswith(".meta.json"):
                        raise RuntimeError("PRIVATE-WRITE-CANARY")
                    return real_write(fd, name, data)

                fault.setattr(storage, "atomic_write", failed_write)
            failed = fx.cycle()
        assert (failed.state, failed.reason, failed.sent, failed.acked) == (
            "failed",
            expected,
            0,
            0,
        )
        verdict = export_consent.classify_cycle(failed, PROFILE)
        assert not verdict.ok and verdict.condition == "send_failed"
        assert lab.requests == [] and fx.delivery() is None
        assert "PRIVATE-WRITE-CANARY" not in repr(failed) + fx.everything_on_disk()
        if key_path.is_symlink():
            key_path.unlink()
        key_path.write_bytes(key)
        key_path.chmod(0o600)
        assert fx.cycle(seconds=1).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert len(lab.requests) == 1


def test_unexpected_credential_mapping_failure_blocks_until_credentials_recover() -> None:
    class Unavailable(Mapping):
        def __getitem__(self, key):
            raise RuntimeError("CREDENTIAL-BACKEND-CANARY")

        def __iter__(self):
            return iter(fx.ENV)

        def __len__(self):
            return len(fx.ENV)

    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        blocked = fx.cycle(deps=fx.deps(environ=Unavailable()))
        assert (blocked.state, blocked.blocked_reason, blocked.sent) == (
            "blocked",
            "credential_invalid",
            0,
        )
        assert lab.requests == [] and fx.metas()[0].attempts == 0
        assert "CREDENTIAL-BACKEND-CANARY" not in fx.everything_on_disk() + repr(blocked)
        assert fx.cycle(seconds=1).acked == 1
        assert lab.requests[0].headers["authorization"] == f"Bearer {fx.SECRET}"


def test_transport_size_refusal_blocks_without_post_and_changed_credential_unblocks() -> None:
    def bounded_post(target, body, **kwargs):
        return export_http.post(target, body, max_request_bytes=1, **kwargs)

    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        blocked = fx.cycle(deps=fx.deps(post=bounded_post))
        assert (blocked.state, blocked.blocked_reason, blocked.acked) == (
            "blocked",
            "policy_blocked",
            0,
        )
        assert lab.requests == [] and fx.state().last_acked_generation is None
        original = fx.body_of(fx.metas()[0])
        assert fx.cycle(seconds=1).state == "blocked"
        changed = {fx.ENV_NAME: "ROTATED-INGEST-CONTROL"}
        assert fx.cycle(seconds=2, deps=fx.deps(environ=changed)).acked == 1
        assert [r.body for r in lab.requests] == [original]


def test_permanent_log_rejection_preserves_unknown_revision_and_does_not_retry() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(1)
        fx.publish(1)
        lab.queue(Respond(400))
        result = fx.cycle(L)
        assert result.quarantined == 1 and result.acked == 0
        (entry,) = fx.registry().entries.values()
        assert entry.acked is None and not entry.queued
        assert entry.unknown is not None and entry.unknown.revision == 1
        assert fx.cycle(L, seconds=1).sent == 0
        fx.commit_ledger(1, extra_for=(1,), generation=2)
        fx.publish(2)
        assert fx.cycle(L, seconds=2).acked == 1
        (entry,) = fx.registry().entries.values()
        assert entry.acked is not None and entry.acked.revision == 2
        assert len(lab.requests) == 2


@pytest.mark.parametrize("window", ["before-begin", "before-read"])
def test_privacy_marker_racing_dispatch_prevents_request_and_later_generation_succeeds(
    window: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        (meta,) = fx.metas()
        body = fx.body_of(meta)
        real_read = export_spool.read_body
        marked = []

        def privacy_race(profile, destination, signal, batch_id):
            if not marked:
                marked.append(batch_id)
                export_spool.update_meta(
                    profile,
                    destination,
                    signal,
                    batch_id,
                    lambda m: replace(m, purge_requested=True),
                )
                if window == "before-read":
                    path = fx.profile_root() / "spool" / DEST / M.value / f"{batch_id}.json"
                    path.write_bytes(b"privacy-erased")
            return real_read(profile, destination, signal, batch_id)

        with monkeypatch.context() as race:
            race.setattr(export_spool, "read_body", privacy_race)
            refused = fx.cycle(seconds=1)
        assert marked == [meta.batch_id]
        assert (refused.sent, refused.acked, refused.quarantined) == (0, 0, 0)
        assert lab.requests == [] and fx.state().last_acked_generation is None
        current = fx.metas()[0]
        assert current.purge_requested and current.attempts == 0
        if window == "before-begin":
            assert fx.body_of(current) == body
        sid = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert len(lab.requests) == 1 and lab.requests[0].body != body


def test_log_ack_survives_body_becoming_unreadable_after_receiver_accepts() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(L,), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(1)
        sid = fx.publish(1)
        captured = []

        def post_then_damage(target, body, **kwargs):
            result = export_http.post(target, body, **kwargs)
            captured.append(body)
            (meta,) = fx.metas(L)
            path = fx.profile_root() / "spool" / DEST / L.value / f"{meta.batch_id}.json"
            path.write_bytes(b"corrupt-after-accept")
            return result

        result = fx.cycle(L, deps=fx.deps(post=post_then_damage))
        assert result.acked == 1 and fx.state(L).last_acked_snapshot_id == sid
        assert [r.body for r in lab.requests] == captured
        (entry,) = fx.registry().entries.values()
        assert entry.acked and entry.acked.revision == 1 and not entry.queued
        assert fx.cycle(L, seconds=1).sent == 0
        preview = delivery_candidate.preview_candidate(PROFILE, DEST, L, now=fx.at(2))
        assert preview.state == "no_changes" and preview.candidate is not None
        assert preview.candidate.batches == ()


def test_policy_becomes_unreadable_after_scope_refresh_holds_queue_until_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        body = fx.body_of(fx.metas()[0])
        path = policy.policy_path()
        original = path.read_bytes()
        real_observe = export_policy.observe_scope

        def revoke_after_scope(*args, **kwargs):
            result = real_observe(*args, **kwargs)
            path.write_bytes(b"{bad policy")
            return result

        with monkeypatch.context() as race:
            race.setattr(export_policy, "observe_scope", revoke_after_scope)
            held = fx.cycle(seconds=1, enqueue=False)
        assert held.state == "paused" and held.sent == 0
        assert fx.body_of(fx.metas()[0]) == body and lab.requests == []
        path.write_bytes(original)
        assert fx.cycle(seconds=2).acked == 1
        assert [r.body for r in lab.requests] == [body]


def test_preview_rechecks_policy_after_target_validation_and_exposes_no_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        configured = fx.configure(lab.url)
        fx.publish(1)
        real_require = export_consent_preview.require_target

        def revoke_after_validation(*args, **kwargs):
            result = real_require(*args, **kwargs)
            fx.amend(destinations=(), enabled_exports=())
            return result

        with monkeypatch.context() as race:
            race.setattr(export_consent_preview, "require_target", revoke_after_validation)
            preview = export_consent_preview.build_preview(PROFILE, DEST, M)
        assert preview.state == "policy_blocked" and preview.reason is not None
        assert preview.batches == () and preview.total_bytes == 0 and lab.requests == []
        policy.save_profile(configured)
        control = export_consent_preview.build_preview(PROFILE, DEST, M)
        assert control.state == "ok" and control.total_bytes > 0
        assert fx.cycle().acked == 1


def test_missing_generation_during_revalidation_discards_without_false_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        real_read = snapshots.read_generation

        def unavailable(*args, **kwargs):
            raise OSError("snapshot device temporarily unavailable")

        with monkeypatch.context() as fault:
            fault.setattr(snapshots, "read_generation", unavailable)
            result = fx.cycle(seconds=1, enqueue=False)
        assert result.invalidated_batches == 1 and result.sent == 0
        assert fx.state().last_acked_generation is None and fx.metas() == []
        assert fx.state().loss_summary[0].reason == "policy_contraction"
        assert lab.requests == []
        assert real_read(PROFILE, 1) is not None
        sid = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        assert fx.state().last_acked_snapshot_id == sid


def test_conflicting_orphan_created_during_enqueue_preserves_bytes_until_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        real_write = export_spool.write_batch
        orphan = []

        def conflicting_writer(profile, destination, signal, batch, **kwargs):
            fd = storage.open_profile_dir(profile, "spool", destination, signal.value)
            try:
                assert storage.write_once(fd, f"{batch.batch_id}.json", b"old orphan bytes")
            finally:
                os.close(fd)
            orphan.append(batch.batch_id)
            return real_write(profile, destination, signal, batch, **kwargs)

        with monkeypatch.context() as race:
            race.setattr(export_spool, "write_batch", conflicting_writer)
            failed = fx.cycle()
        assert (failed.state, failed.reason, failed.sent) == ("failed", "batch_conflict", 0)
        assert len(orphan) == 1 and fx.metas() == [] and lab.requests == []
        path = fx.profile_root() / "spool" / DEST / M.value / f"{orphan[0]}.json"
        assert path.read_bytes() == b"old orphan bytes"
        assert export_spool.sweep_orphans(PROFILE, DEST, M) == 1
        assert fx.cycle(seconds=1).acked == 1
        assert lab.requests[0].body == fx.body_of(fx.metas()[0])
        assert b"old orphan bytes" not in lab.requests[0].body


def test_disabled_settled_spool_cannot_recreate_a_deleted_ack_document() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        assert fx.cycle().acked == 1
        (settled,) = fx.metas()
        body = fx.body_of(settled)
        path = fx.profile_root() / "delivery" / DEST / M.value
        path.with_suffix(".json").unlink()
        fx.set_enabled(M, False)
        held = fx.cycle(seconds=1)
        assert held.state == "not_enabled" and held.sent == 0
        assert fx.delivery() is None
        assert fx.metas() == [settled] and fx.body_of(settled) == body
        assert len(lab.requests) == 1
        fx.set_enabled(M, True)
        sid = fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1
        assert fx.state().last_acked_snapshot_id == sid
        assert len(lab.requests) == 2 and lab.requests[1].body != body
