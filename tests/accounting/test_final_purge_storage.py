"""Purge selection and read failures preserve unselected and original private data."""

from __future__ import annotations

import errno
import os
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import (
    collector_store as store,
    lease,
    plainfile,
    purge,
    purge_files,
    purge_select,
    purge_snapshot,
    purge_state,
    snapshots,
    storage,
)
from ssgrep.accounting.collector_state import REDUCTION, State, load_state
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.purge_model import PurgeError
from tests.accounting.collector_scene import accounting_dir
from tests.accounting.conftest import build_evidence
from tests.accounting.snapshots_driver import make_envelope
from tests.accounting.test_purge import ALPHA, PROFILE, PURGE_AT, build_scene, select, tree


def test_absent_file_helpers_create_nothing_and_present_helpers_delete_only_selected_files(
    tmp_path: Path,
) -> None:
    assert purge_files.receipt_ids(PROFILE) == set()
    assert purge_files.delete_receipts(PROFILE, {"old"}) == 0
    assert purge_files.generation_files(PROFILE) == []
    assert purge_files.has_current(PROFILE) is False
    assert purge_files.delete_generations(PROFILE, []) == 0
    assert purge_files.delete_current(PROFILE) is False
    purge_files.clear_marker(PROFILE)
    assert not (tmp_path / "data").exists()
    first, second = make_envelope(1), make_envelope(2)
    snapshots.publish(PROFILE, first)
    snapshots.publish(PROFILE, second)
    for name, data in (("old.json", b"old receipt"), ("keep.json", b"keep receipt")):
        plainfile.write_private(
            storage.data_root(), "accounting", PROFILE, "receipts", name, data=data
        )
    assert purge_files.receipt_ids(PROFILE) == {"old", "keep"}
    assert purge_files.delete_receipts(PROFILE, {"old", "absent"}) == 1
    assert purge_files.has_current(PROFILE)
    assert purge_files.delete_current(PROFILE)
    assert not purge_files.delete_current(PROFILE)
    generations = purge_files.generation_files(PROFILE)
    assert [g.number for g in generations] == [1, 2]
    assert purge_files.delete_generations(PROFILE, generations[:1]) == 1
    assert snapshots.read_generation(PROFILE, 1) is None
    assert snapshots.read_generation(PROFILE, 2) == second
    assert (accounting_dir() / "receipts" / "keep.json").read_bytes() == b"keep receipt"


@pytest.mark.parametrize("component", ["root", "snapshots"])
def test_readonly_purge_refuses_exposed_storage_without_repairing_permissions(
    tmp_path: Path, component: str
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    target = accounting_dir() if component == "root" else accounting_dir() / "snapshots"
    target.chmod(0o750)
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as caught:
        if component == "root":
            purge.dry_run(select())
        else:
            purge_files.generation_files(PROFILE)
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    target.chmod(0o700)
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


@pytest.mark.parametrize("component,mode", [("directory", 0o750), ("file", 0o644), ("file", 0o700)])
def test_preflight_refuses_exposed_unknown_entries_without_changing_anything(
    tmp_path: Path, component: str, mode: int
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    plainfile.write_private(
        storage.data_root(), "accounting", PROFILE, "extra", "original", data=b"secret"
    )
    target = accounting_dir() / "extra"
    if component == "file":
        target /= "original"
    target.chmod(mode)
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as caught:
        purge.dry_run(select())
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    target.chmod(0o700 if component == "directory" else 0o600)
    assert purge_files.preflight(PROFILE)
    assert (accounting_dir() / "extra" / "original").read_bytes() == b"secret"


def test_preflight_listing_io_failure_preserves_data_and_successful_retry_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    before = tree(tmp_path / "data")
    listdir = os.listdir

    def interrupted(path):
        if isinstance(path, int):
            raise OSError(errno.EIO, "directory read interrupted")
        return listdir(path)

    with monkeypatch.context() as fault:
        fault.setattr(purge_files.os, "listdir", interrupted)
        with pytest.raises(PurgeError) as caught:
            purge.dry_run(select())
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    assert purge_files.preflight(PROFILE)
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_lease_open_failure_stops_purge_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    with lease.Lease.acquire(PROFILE, "collect"):
        pass
    before = tree(tmp_path / "data")
    open_file = os.open

    def fail_lock(name, flags, mode=0o777, *, dir_fd=None):
        if name == lease.LEASE_FILE:
            raise storage.UnsafeStorageError(errno.EPERM, "foreign lock")
        return open_file(name, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as fault:
        fault.setattr(lease.os, "open", fail_lock)
        with pytest.raises(PurgeError) as caught:
            purge.execute(select(), now=PURGE_AT)
    assert caught.value.condition == "storage_unsafe"
    assert tree(tmp_path / "data") == before
    assert purge.execute(select(), now=PURGE_AT).local_deleted.total() == 0
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_dry_run_reports_unknown_worker_when_owner_record_read_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with lease.Lease.acquire(PROFILE, "serve"):
        before = tree(tmp_path / "data")

        def unreadable(fd, length, offset):
            raise OSError(errno.EIO, "owner record unreadable")

        with monkeypatch.context() as fault:
            fault.setattr(lease.os, "pread", unreadable)
            report = purge.dry_run(select())
        assert report.worker is None
        assert tree(tmp_path / "data") == before
        readable = purge.dry_run(select())
        assert readable.worker is not None and readable.worker.running


def test_scoper_rejects_ambiguous_paths_and_tracks_mixed_conversations() -> None:
    scoper = purge_select.Scoper(ALPHA)
    for invalid in (None, "", "relative", "\0bad"):
        assert not scoper.inside(invalid)
    assert scoper.inside(ALPHA + "/child")
    assert not scoper.inside(ALPHA + "-other")
    records = [
        build_evidence(
            session_id=session,
            record_id=str(index),
            ts_utc="2026-10-01T00:00:00Z",
            payload={"cwd": cwd},
        )
        for index, (session, cwd) in enumerate(
            [
                ("selected", ALPHA),
                ("mixed", ALPHA),
                ("mixed", "/work/beta"),
                ("unknown", None),
                ("other", "/work/beta"),
            ]
        )
    ]
    checkpoint = store.Checkpoint(PROFILE, segments=store.write_segments(PROFILE, records))
    scopes = purge_select.classify(PROFILE, checkpoint, None, (), scoper)
    assert scopes.selected == frozenset({"claude:selected"})
    assert (scopes.unknown, scopes.mixed) == (1, 1)
    assert scopes.root(None, "selected") is None
    assert scopes.root("claude", None) is None
    assert list(store.iter_evidence(PROFILE, checkpoint.segments)) == records


def test_workspace_scope_skips_invalid_approved_root_without_ignoring_the_valid_one() -> None:
    policy = ProfilePolicy(profile_id="team", roots=("relative", ALPHA))
    chosen = purge_select.parse_selection(
        profile="team", before="2026-10-10", workspace=ALPHA + "/child", policy=policy
    )
    assert chosen.workspace == ALPHA + "/child"
    with pytest.raises(PurgeError) as caught:
        purge_select.parse_selection(
            profile="team", before="2026-10-10", workspace="/work/beta", policy=policy
        )
    assert caught.value.condition == "workspace_not_in_scope"


def test_undated_structure_follows_session_cutoff_and_unselected_segments_keep_identity() -> None:
    def record(session, name, stamp):
        return build_evidence(
            session_id=session, record_id=name, ts_utc=stamp, payload={"cwd": ALPHA}
        )

    gone = [record("old", "structure-old", None), record("old", "old", "2026-10-01T00:00:00Z")]
    live = [record("live", "structure-live", None), record("live", "new", "2026-10-10T00:00:00Z")]
    old_refs = store.write_segments(PROFILE, gone)
    live_refs = store.write_segments(PROFILE, live)
    checkpoint = store.Checkpoint(PROFILE, segments=[*old_refs, *live_refs])
    store.commit_checkpoint(PROFILE, checkpoint)
    planned = purge_state.plan_state(PROFILE, select(), checkpoint)
    assert planned is not None and planned.evidence_records == 2
    assert planned.gone_sessions == frozenset({("claude", "old")})
    rewritten = purge_state.write_purged_state(PROFILE, planned, select())
    assert rewritten.segments == live_refs
    assert list(store.iter_evidence(PROFILE, rewritten.segments)) == live
    assert list(store.iter_evidence(PROFILE, checkpoint.segments)) == gone + live
    assert store.load_checkpoint(PROFILE) == checkpoint


def test_missing_execution_cwd_uses_proven_session_scope_and_retains_unknowns(
    tmp_path: Path,
) -> None:
    build_scene(tmp_path)
    checkpoint = store.load_checkpoint(PROFILE)
    assert checkpoint is not None
    state = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.reduction is not None and state.reduction.executions
    prototype = state.reduction.executions[0]
    executions = tuple(
        replace(
            prototype,
            execution_id=name,
            day="2026-10-05",
            ts_utc="2026-10-05T00:00:00Z",
            cwd=None,
            runtime="claude",
            session_id=session,
        )
        for name, session in (("proven", "a1"), ("outside", "b1"), ("unknown", None))
    )
    doc = replace(state.reduction, executions=executions)
    checkpoint = replace(
        checkpoint,
        blobs={**checkpoint.blobs, REDUCTION: store.write_blob(PROFILE, REDUCTION, doc.to_dict())},
    )
    planned = purge_state.plan_state(PROFILE, select(workspace=ALPHA), checkpoint)
    assert planned is not None and planned.reduction_after is not None
    assert planned.executions_removed == 1 and planned.unknown_scope_executions == 2
    assert planned.reduction_after.executions == executions[1:]
    assert store.read_blob(PROFILE, checkpoint.blobs[REDUCTION]) == doc.to_dict()


def test_snapshot_build_requires_current_methodology_instead_of_publishing_empty_totals() -> None:
    checkpoint = store.Checkpoint(PROFILE)
    planned = purge_state.StatePlan(checkpoint, State(), purge_select.Scopes(None))
    with pytest.raises(ValueError, match="no ledger"):
        purge_snapshot.build(PROFILE, planned, replace(checkpoint, generation=1), PURGE_AT)
    assert snapshots.generations(PROFILE) == []


def test_missing_referenced_legacy_ledger_stops_purge_without_dropping_other_data(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    name = store.write_blob(PROFILE, "ledger-legacy", {})
    checkpoint = store.Checkpoint(PROFILE, blobs={"ledger-legacy": name})
    store.commit_checkpoint(PROFILE, checkpoint)
    (accounting_dir() / "collector" / store.STATE_DIR / name).unlink()
    before = tree(tmp_path / "data")
    with pytest.raises(PurgeError) as caught:
        purge.dry_run(select())
    assert caught.value.condition == "state_corrupt"
    assert tree(tmp_path / "data") == before
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_preflight_reads_owner_readonly_entries_without_broadening_permissions(
    tmp_path: Path,
) -> None:
    plainfile.write_private(
        storage.data_root(), "accounting", PROFILE, "extra", "original", data=b"readonly secret"
    )
    directory = accounting_dir() / "extra"
    target = directory / "original"
    directory.chmod(0o500)
    target.chmod(0o400)
    try:
        before = tree(tmp_path / "data")
        assert purge_files.preflight(PROFILE)
        assert tree(tmp_path / "data") == before
        assert (
            plainfile.read_private(
                storage.data_root(), "accounting", PROFILE, "extra", "original", max_bytes=15
            )
            == b"readonly secret"
        )
    finally:
        directory.chmod(0o700)
        target.chmod(0o600)
