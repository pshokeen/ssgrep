"""Real private files at read bounds, failed publication, and lease recovery boundaries."""

from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import lease, plainfile, snapshots, storage
from tests.accounting.snapshots_driver import gen_name, make_envelope
from tests.accounting.test_collect_lease import _fake_stat

PROFILE = "personal-local"


def test_private_read_rejects_growth_after_the_size_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"{}")
    plainfile.write_private(tmp_path, "private", "other.json", data=b"unrelated")
    target = tmp_path / "private" / "state.json"
    original_fstat = os.fstat

    def append_after_stat(fd: int):
        info = original_fstat(fd)
        with target.open("ab") as handle:
            handle.write(b" ")  # still valid JSON, but no longer within the reader's bound
        return info

    with monkeypatch.context() as fault:
        fault.setattr(plainfile.os, "fstat", append_after_stat)
        with pytest.raises(ValueError, match="exceeds 2 bytes"):
            plainfile.read_private(tmp_path, "private", "state.json", max_bytes=2)
    assert plainfile.read_private(tmp_path, "private", "state.json", max_bytes=3) == b"{} "
    assert plainfile.read_private(tmp_path, "private", "other.json", max_bytes=9) == b"unrelated"


@pytest.mark.parametrize("unsafe_component", ["directory", "file"])
def test_private_read_refuses_other_writable_components_without_changing_data(
    tmp_path: Path, unsafe_component: str
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"secret")
    target = tmp_path / "private" / "state.json"
    component = target.parent if unsafe_component == "directory" else target
    component.chmod(0o770 if unsafe_component == "directory" else 0o660)
    with pytest.raises(plainfile.UnsafePathError, match="writable"):
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=6)
    assert target.read_bytes() == b"secret"
    component.chmod(0o700 if unsafe_component == "directory" else 0o600)
    assert plainfile.read_private(tmp_path, "private", "state.json", max_bytes=6) == b"secret"


def test_private_read_refuses_a_directory_in_place_of_a_file(tmp_path: Path) -> None:
    (tmp_path / "not-a-file").mkdir(mode=0o700)
    with pytest.raises(plainfile.UnsafePathError, match="regular file"):
        plainfile.read_private(tmp_path, "not-a-file", max_bytes=100)
    assert list((tmp_path / "not-a-file").iterdir()) == []
    plainfile.write_private(tmp_path, "valid.json", data=b"{}")
    assert plainfile.read_private(tmp_path, "valid.json", max_bytes=2) == b"{}"


@pytest.mark.parametrize("boundary", ["flush", "replace"])
def test_private_write_failure_preserves_readable_old_value_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"old-complete")
    plainfile.write_private(tmp_path, "private", "other.json", data=b"unrelated")
    original_fsync = os.fsync

    def fail_flush(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.ENOSPC, "no space for a durable write")
        original_fsync(fd)

    def fail_replace(*args, **kwargs) -> None:
        raise OSError(errno.EIO, "rename failed")

    with monkeypatch.context() as fault:
        if boundary == "flush":
            fault.setattr(plainfile.os, "fsync", fail_flush)
        else:
            fault.setattr(plainfile.os, "replace", fail_replace)
        with pytest.raises(OSError):
            plainfile.write_private(tmp_path, "private", "state.json", data=b"new-complete")
    assert {p.name: p.read_bytes() for p in (tmp_path / "private").iterdir()} == {
        "state.json": b"old-complete",
        "other.json": b"unrelated",
    }
    assert (
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=12) == b"old-complete"
    )
    plainfile.write_private(tmp_path, "private", "state.json", data=b"new-complete")
    assert (
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=12) == b"new-complete"
    )


@pytest.mark.parametrize("boundary", ["partial-write", "current-rename"])
def test_failed_snapshot_publication_preserves_current_and_retry_uses_complete_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    first, second = make_envelope(1), make_envelope(2)
    snapshots.publish(PROFILE, first)
    directory = tmp_path / "data" / "accounting" / PROFILE / "snapshots"

    def partial_write(fd: int, data: bytes) -> None:
        os.write(fd, data[:17])
        raise OSError(errno.ENOSPC, "partial snapshot write")

    def fail_rename(*args, **kwargs) -> None:
        raise OSError(errno.EIO, "current rename failed")

    with monkeypatch.context() as fault:
        if boundary == "partial-write":
            fault.setattr(snapshots.storage, "write_all", partial_write)
        else:
            fault.setattr(snapshots.os, "rename", fail_rename)
        with pytest.raises(OSError):
            snapshots.publish(PROFILE, second)
    assert snapshots.decode((directory / "current.json").read_bytes()) == first
    assert snapshots.read_generation(PROFILE, 1) == first
    assert snapshots.read_generation(PROFILE, 2) == (
        second if boundary == "current-rename" else None
    )
    assert not any(p.name.endswith(".tmp") for p in directory.iterdir())
    snapshots.publish(PROFILE, second)
    assert snapshots.decode((directory / "current.json").read_bytes()) == second
    assert snapshots.read_generation(PROFILE, 2) == second


@pytest.mark.parametrize("competitor", ["identical", "same-id-other-time", "different-id"])
def test_concurrent_snapshot_staging_never_overwrites_the_winner(
    tmp_path: Path, competitor: str
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    attempted = make_envelope(2)
    winner = attempted
    if competitor == "same-id-other-time":
        winner = replace(attempted, collected_at="2026-10-13T01:00:00Z")
    elif competitor == "different-id":
        winner = make_envelope(2, observations={"seq": 999})

    def interleave(step: str) -> None:
        if step == "gen_tmp_written":
            snapshots.stage_generation(PROFILE, winner)

    if competitor == "identical":
        snapshots.stage_generation(PROFILE, attempted, interleave)
    else:
        with pytest.raises(snapshots.GenerationExists):
            snapshots.stage_generation(PROFILE, attempted, interleave)
    assert snapshots.read_generation(PROFILE, 2) == winner
    assert snapshots.current_generation(PROFILE) == 1
    snapshots.make_current(PROFILE, 2)
    directory = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    assert snapshots.decode((directory / "current.json").read_bytes()) == winner
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        ["current.json", gen_name(1), f"gen-0000000002-{winner.snapshot_id}.json"]
    )


def test_exhausted_generation_space_does_not_wrap_or_rewrite_the_last_snapshot(
    tmp_path: Path,
) -> None:
    last = make_envelope(snapshots.MAX_GENERATION)
    snapshots.publish(PROFILE, last)
    with pytest.raises(ValueError, match="generation space exhausted"):
        snapshots.next_generation(PROFILE)
    assert snapshots.read_generation(PROFILE, snapshots.MAX_GENERATION) == last
    directory = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    assert snapshots.decode((directory / "current.json").read_bytes()) == last


def test_failed_lease_owner_write_releases_the_real_flock_for_the_next_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def partial_owner(fd: int, data: bytes) -> None:
        os.write(fd, data[:7])
        raise OSError(errno.ENOSPC, "owner record interrupted")

    with monkeypatch.context() as fault:
        fault.setattr(lease.storage, "write_all", partial_owner)
        with pytest.raises(OSError, match="owner record interrupted"):
            lease.Lease.acquire(PROFILE, "collect", retry_seconds=0)
    with lease.Lease.acquire(PROFILE, "serve", retry_seconds=0) as recovered:
        assert recovered.recovered_from == lease.LeaseInfo(None, None, None, None, False)
        info = lease.inspect_lease(PROFILE)
        assert info is not None and (info.pid, info.kind, info.alive) == (
            os.getpid(),
            "serve",
            True,
        )
        with pytest.raises(lease.AlreadyRunning):
            lease.Lease.acquire(PROFILE, "collect", retry_seconds=0)


def test_failed_lease_record_cleanup_does_not_keep_the_kernel_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = lease.Lease.acquire(PROFILE, "collect")
    before = lease.inspect_lease(PROFILE)

    def fail_truncate(fd: int, length: int) -> None:
        raise OSError(errno.EIO, "metadata cleanup failed")

    with monkeypatch.context() as fault:
        fault.setattr(lease.os, "ftruncate", fail_truncate)
        held.release()
    assert lease.inspect_lease(PROFILE) == before
    with lease.Lease.acquire(PROFILE, "serve", retry_seconds=0) as recovered:
        assert recovered.recovered_from == before
        info = lease.inspect_lease(PROFILE)
        assert info is not None and info.kind == "serve" and info.alive
    assert lease.inspect_lease(PROFILE) == lease.LeaseInfo(None, None, None, None, False)


def test_missing_process_tools_fall_back_to_live_pid_existence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease, "_PROC_ROOT", tmp_path / "absent-proc")
    monkeypatch.setattr(lease, "_PS", str(tmp_path / "absent-ps"))
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    assert lease.process_start_token(child.pid) is None
    with lease.Lease.acquire(PROFILE, "collect"):
        info = lease.inspect_lease(PROFILE)
        assert info is not None
        assert (info.process_start, info.alive) == (f"pid:{os.getpid()}", True)
        with pytest.raises(lease.AlreadyRunning):
            lease.Lease.acquire(PROFILE, "serve", retry_seconds=0)


def test_missing_boot_id_keeps_proc_start_time_and_detects_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    (proc / "sys").mkdir(parents=True)
    _fake_stat(proc, 4242, "worker", "S", 123)
    monkeypatch.setattr(lease, "_PROC_ROOT", proc)
    assert lease.process_start_token(4242) == "123"
    _fake_stat(proc, 4242, "new worker", "S", 456)
    assert lease.process_start_token(4242) == "456"


@pytest.mark.parametrize("failed_directory", ["delivery", "bad-destination"])
def test_unreadable_pin_listing_is_read_only_and_does_not_hide_other_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_directory: str
) -> None:
    first, second = make_envelope(1), make_envelope(2)
    snapshots.publish(PROFILE, first)
    snapshots.publish(PROFILE, second)
    for destination, snapshot_id in (
        ("good-destination", first.snapshot_id),
        ("bad-destination", second.snapshot_id),
    ):
        plainfile.write_private(
            storage.data_root(),
            "accounting",
            PROFILE,
            "delivery",
            destination,
            "metrics.json",
            data=f'{{"last_acked_snapshot_id":"{snapshot_id}"}}'.encode(),
        )
    root = tmp_path / "data" / "accounting" / PROFILE / "delivery"
    denied = root if failed_directory == "delivery" else root / failed_directory
    identity = denied.stat().st_ino
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*.json")}
    original_list = storage.list_names

    def deny_listing(fd: int):
        if os.fstat(fd).st_ino == identity:
            raise PermissionError(errno.EACCES, "pin directory denied")
        return original_list(fd)

    with monkeypatch.context() as fault:
        fault.setattr(storage, "list_names", deny_listing)
        pins = snapshots.pinned_snapshot_ids(PROFILE)
    expected = frozenset() if failed_directory == "delivery" else frozenset({first.snapshot_id})
    assert pins == expected
    assert {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*.json")} == before
    assert snapshots.read_generation(PROFILE, 1) == first
    assert snapshots.read_generation(PROFILE, 2) == second


@pytest.mark.parametrize("component", ["accounting", "profile", "snapshots"])
@pytest.mark.parametrize("mode", [0o710, 0o740, 0o777])
def test_read_only_directory_open_refuses_public_permissions_without_repair(
    tmp_path: Path, component: str, mode: int
) -> None:
    envelope = make_envelope(1)
    snapshots.publish(PROFILE, envelope)
    accounting = tmp_path / "data" / "accounting"
    directory = accounting / PROFILE / "snapshots"
    target = {"accounting": accounting, "profile": accounting / PROFILE, "snapshots": directory}[
        component
    ]
    target.chmod(mode)
    paths = [accounting, accounting / PROFILE, directory, directory / "current.json"]
    before = {
        p: (p.stat().st_mode, p.stat().st_ctime_ns, p.stat().st_mtime_ns, p.stat().st_ino)
        for p in paths
    }
    current = (directory / "current.json").read_bytes()
    with pytest.raises(storage.UnsafeStorageError):
        snapshots.read_generation(PROFILE, 1)
    assert {
        p: (p.stat().st_mode, p.stat().st_ctime_ns, p.stat().st_mtime_ns, p.stat().st_ino)
        for p in paths
    } == before
    assert (directory / "current.json").read_bytes() == current
    target.chmod(0o700)  # recovery is an explicit owner action, never a read side effect
    assert snapshots.read_generation(PROFILE, 1) == envelope
    assert snapshots.decode((directory / "current.json").read_bytes()) == envelope


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660, 0o700])
def test_regular_file_reads_refuse_unsafe_modes_without_changing_inode_or_bytes(
    tmp_path: Path, mode: int
) -> None:
    directory = storage.open_profile_dir(PROFILE, "snapshots")
    target = tmp_path / "data" / "accounting" / PROFILE / "snapshots" / "state.json"
    try:
        storage.atomic_write(directory, "state.json", b'{"value":7}')
        storage.atomic_write(directory, "other.json", b'{"value":9}')
        target.chmod(mode)
        before = target.stat()
        with pytest.raises(storage.UnsafeStorageError):
            storage.read_file(directory, "state.json", 32)
        after = target.stat()
        assert (after.st_mode, after.st_ctime_ns, after.st_mtime_ns, after.st_ino) == (
            before.st_mode,
            before.st_ctime_ns,
            before.st_mtime_ns,
            before.st_ino,
        )
        assert target.read_bytes() == b'{"value":7}'
        assert storage.read_file(directory, "other.json", 32) == b'{"value":9}'
        target.chmod(0o600)
        assert storage.read_file(directory, "state.json", 32) == b'{"value":7}'
    finally:
        os.close(directory)


def test_owner_readonly_directory_and_file_remain_readable_without_permission_broadening(
    tmp_path: Path,
) -> None:
    envelope = make_envelope(1)
    snapshots.publish(PROFILE, envelope)
    directory = tmp_path / "data" / "accounting" / PROFILE / "snapshots"
    generation = directory / gen_name(1)
    generation.chmod(0o400)
    directory.chmod(0o500)
    before_dir, before_file = directory.stat(), generation.stat()
    try:
        assert snapshots.read_generation(PROFILE, 1) == envelope
        assert snapshots.current_generation(PROFILE) == 1
        assert (directory.stat().st_mode, directory.stat().st_ctime_ns) == (
            before_dir.st_mode,
            before_dir.st_ctime_ns,
        )
        assert (generation.stat().st_mode, generation.stat().st_ctime_ns) == (
            before_file.st_mode,
            before_file.st_ctime_ns,
        )
    finally:
        directory.chmod(0o700)


def test_writer_open_can_tighten_a_directory_and_atomically_replace_an_unsafe_file(
    tmp_path: Path,
) -> None:
    directory_fd = storage.open_profile_dir(PROFILE, "collector")
    storage.atomic_write(directory_fd, "state.json", b"old")
    storage.atomic_write(directory_fd, "other.json", b"keep")
    os.close(directory_fd)
    directory = tmp_path / "data" / "accounting" / PROFILE / "collector"
    target = directory / "state.json"
    sibling = directory / "other.json"
    sibling.chmod(0o400)
    sibling_before = sibling.stat()
    directory.chmod(0o750)
    target.chmod(0o644)
    directory_fd = storage.open_profile_dir(PROFILE, "collector", create=True)
    try:
        storage.atomic_write(directory_fd, "state.json", b"new-complete")
        assert storage.read_file(directory_fd, "state.json", 32) == b"new-complete"
        assert storage.read_file(directory_fd, "other.json", 32) == b"keep"
    finally:
        os.close(directory_fd)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert (sibling.stat().st_mode, sibling.stat().st_ctime_ns, sibling.stat().st_ino) == (
        sibling_before.st_mode,
        sibling_before.st_ctime_ns,
        sibling_before.st_ino,
    )


def test_standalone_delivery_builder_creates_a_readable_private_pin_tree(tmp_path: Path) -> None:
    from tests.accounting.http_builders import write_delivery

    snapshot_id = "a" * 32
    target = write_delivery(PROFILE, "standalone", "metrics", last_acked_snapshot_id=snapshot_id)
    root = tmp_path / "data"
    directories = [
        root / "accounting",
        root / "accounting" / PROFILE,
        root / "accounting" / PROFILE / "delivery",
        target.parent,
    ]
    assert [stat.S_IMODE(path.stat().st_mode) for path in directories] == [0o700] * 4
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    before = target.read_bytes(), target.stat().st_ctime_ns
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset({snapshot_id})
    assert (target.read_bytes(), target.stat().st_ctime_ns) == before
