"""Best-effort callback reporting and consent races remain bounded and recoverable."""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from ssgrep.accounting import (
    delivery,
    delivery_candidate,
    export_consent,
    export_files,
    export_ids,
    export_policy,
    policy,
    storage,
)
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, M
from tests.accounting.lab_receiver import LabReceiver


def test_callback_absent_export_directory_does_not_create_state() -> None:
    before = fx.tree()
    delivery.export_callback(PROFILE)()
    assert fx.tree() == before
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        delivery.export_callback(PROFILE, deps=fx.deps())()
        assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1


@pytest.mark.parametrize("failure", ["report-write", "report-delete"])
def test_callback_reporting_failures_never_stop_recovery(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        export_ids.load_key(PROFILE, create=True)
        key_path = fx.profile_root() / "export" / "pseudonym.key"
        key = key_path.read_bytes()
        key_path.write_bytes(b"broken")
        callback = delivery.export_callback(PROFILE, deps=fx.deps())
        error_path = fx.profile_root() / "export" / "last_error.json"
        callback()
        report = error_path.read_bytes()
        assert json.loads(report)["error"] == "storage_unsafe"
        assert lab.requests == [] and fx.delivery() is None
        if failure == "report-delete":
            key_path.write_bytes(key)
        real_write, real_delete = storage.atomic_write, storage.delete_file

        def fail_write(fd, name, data):
            if name == "last_error.json":
                raise OSError(errno.ENOSPC, "REPORT-WRITE-CANARY")
            return real_write(fd, name, data)

        def fail_delete(fd, name):
            if name == "last_error.json":
                raise OSError(errno.EACCES, "REPORT-DELETE-CANARY")
            return real_delete(fd, name)

        with monkeypatch.context() as fault:
            fault.setattr(
                storage, "atomic_write", fail_write if failure == "report-write" else real_write
            )
            fault.setattr(
                storage, "delete_file", fail_delete if failure == "report-delete" else real_delete
            )
            callback()
        assert error_path.read_bytes() == report
        assert "REPORT-WRITE-CANARY" not in fx.everything_on_disk()
        assert "REPORT-DELETE-CANARY" not in fx.everything_on_disk()
        key_path.write_bytes(key)
        callback()
        assert not error_path.exists()
        assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1


def test_target_discovery_failure_does_not_erase_queued_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        before = fx.tree()
        body = fx.body_of(fx.metas()[0])

        def unavailable(profile):
            raise OSError(errno.EIO, "spool mount disappeared")

        with monkeypatch.context() as fault:
            fault.setattr(delivery.export_spool, "iter_spools", unavailable)
            assert delivery.run_export(PROFILE, now=fx.at(), deps=fx.deps()) == []
        assert fx.tree() == before and lab.requests == []
        results = delivery.run_export(PROFILE, now=fx.at(1), deps=fx.deps())
        assert len(results) == 1 and results[0].acked == 1
        assert [r.body for r in lab.requests] == [body]


def test_malformed_enabled_signal_is_ignored_without_suppressing_valid_destination() -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.plant_policy(lambda p: p["enabled_exports"].append([DEST, "bogus-signal"]))
        results = delivery.run_export(PROFILE, now=fx.at(), deps=fx.deps())
        assert [(r.destination, r.signal, r.acked) for r in results] == [(DEST, M, 1)]
        assert len(lab.requests) == 1 and fx.state().last_acked_generation == 1


@pytest.mark.parametrize("damage", ["json", "directory"])
def test_unreadable_scope_is_not_first_send_authority_and_restoration_recovers(
    damage: str,
    tmp_path: Path,
) -> None:
    with LabReceiver.start() as lab:
        fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        path = fx.profile_root() / "export" / DEST / "scope-metrics.json"
        original = path.read_bytes()
        directory = path.parent
        saved = tmp_path / "saved-scope"
        if damage == "directory":
            directory.rename(saved)
            directory.symlink_to(saved, target_is_directory=True)
        else:
            path.write_bytes(b"{broken scope")
        scope = export_policy.read_scope(PROFILE, DEST, M)
        assert scope is not None and scope.digest == "" and scope.changed_at == ""
        preview = delivery_candidate.preview_candidate(PROFILE, DEST, M, now=fx.at())
        assert preview.state == "ineligible_scope" and preview.candidate is None
        assert lab.requests == []
        if damage == "directory":
            with pytest.raises(export_files.StateCorrupt):
                with export_files.locked(PROFILE, ("export", DEST), "control"):
                    pytest.fail("unsafe state directory was accepted")
            directory.unlink()
            saved.rename(directory)
        else:
            assert path.read_bytes() == b"{broken scope"
            path.write_bytes(original)
        assert fx.cycle(seconds=1).acked == 1
        assert len(lab.requests) == 1


def test_revoked_cycle_verdict_retains_actionable_failure_and_reenable_control() -> None:
    with LabReceiver.start() as lab:
        configured = fx.configure(lab.url)
        fx.publish(1)
        fx.cycle(deps=fx.deps(max_batches=0))
        fx.amend(destinations=(), enabled_exports=())
        result = fx.cycle(seconds=1)
        verdict = export_consent.classify_cycle(result, PROFILE)
        assert result.state == "revoked" and result.invalidated_batches == 1
        assert not verdict.ok and verdict.code != 0
        assert verdict.condition == result.reason and verdict.next_command is not None
        assert fx.state().last_acked_generation is None and lab.requests == []
        policy.save_profile(configured)
        fx.publish(2)
        assert fx.cycle(seconds=2).acked == 1 and len(lab.requests) == 1
