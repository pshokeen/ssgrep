"""Private read failures and lease acquisition faults preserve bytes and kernel locks."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from ssgrep.accounting import lease, plainfile, storage

PROFILE = "personal-local"


def test_private_directory_owned_by_another_user_is_not_read_or_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"original")
    directory = tmp_path / "private"
    before = directory.stat()
    uid = os.geteuid()
    with monkeypatch.context() as foreign:
        foreign.setattr(plainfile.os, "geteuid", lambda: uid + 1)
        with pytest.raises(plainfile.UnsafePathError, match="not owned"):
            plainfile.read_private(tmp_path, "private", "state.json", max_bytes=8)
    assert directory.stat() == before
    assert plainfile.read_private(tmp_path, "private", "state.json", max_bytes=8) == b"original"


@pytest.mark.parametrize("fault_kind", ["io", "creation-race"])
def test_lease_open_failure_keeps_prior_owner_bytes_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_kind: str
) -> None:
    with lease.Lease.acquire(PROFILE, "collect"):
        pass
    path = tmp_path / "data" / "accounting" / PROFILE / lease.LEASE_DIR / lease.LEASE_FILE
    original = path.read_bytes()
    open_file = os.open

    def failed_open(name, flags, mode=0o777, *, dir_fd=None):
        if name == lease.LEASE_FILE:
            if fault_kind == "io":
                raise OSError(errno.EIO, "lock storage unavailable")
            if flags & os.O_CREAT:
                raise FileExistsError(errno.EEXIST, "competing creator")
            raise FileNotFoundError(errno.ENOENT, "name disappeared")
        return open_file(name, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as fault:
        fault.setattr(lease.os, "open", failed_open)
        with pytest.raises(OSError) as caught:
            lease.Lease.acquire(PROFILE, "serve", retry_seconds=0)
    assert caught.value.errno == (errno.EIO if fault_kind == "io" else errno.EAGAIN)
    assert path.read_bytes() == original
    with lease.Lease.acquire(PROFILE, "serve", retry_seconds=0):
        info = lease.inspect_lease(PROFILE)
        assert info is not None and info.kind == "serve" and info.alive
        with pytest.raises(lease.AlreadyRunning):
            lease.Lease.acquire(PROFILE, "collect", retry_seconds=0)


def test_foreign_lock_file_is_refused_without_truncating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fd = storage.open_profile_dir(PROFILE, lease.LEASE_DIR)
    storage.atomic_write(fd, lease.LEASE_FILE, b"foreign owner record")
    uid = os.geteuid()
    try:
        with monkeypatch.context() as foreign:
            foreign.setattr(lease.os, "geteuid", lambda: uid + 1)
            with pytest.raises(storage.UnsafeStorageError, match="not owned"):
                lease._open_lock_file(fd)
        assert storage.read_file(fd, lease.LEASE_FILE, 100) == b"foreign owner record"
        descriptor = lease._open_lock_file(fd)
        os.close(descriptor)
    finally:
        os.close(fd)
    with lease.Lease.acquire(PROFILE, "collect", retry_seconds=0):
        info = lease.inspect_lease(PROFILE)
        assert info is not None and info.alive


def test_permission_denied_process_probe_does_not_misclassify_live_owner_as_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inaccessible(pid: int, signal: int) -> None:
        raise PermissionError(errno.EPERM, "different uid")

    with monkeypatch.context() as fault:
        fault.setattr(lease.os, "kill", inaccessible)
        assert lease._existence_token(1234) == "pid:1234"
    assert lease._existence_token(os.getpid()) == f"pid:{os.getpid()}"


def test_finalizer_suppresses_close_error_after_releasing_its_kernel_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = lease.Lease.acquire(PROFILE, "collect")
    close = os.close

    def interrupted_close(fd: int) -> None:
        close(fd)
        raise OSError(errno.EIO, "close interrupted after descriptor release")

    with monkeypatch.context() as fault:
        fault.setattr(lease.os, "close", interrupted_close)
        held.__del__()
    held.release()
    with lease.Lease.acquire(PROFILE, "serve", retry_seconds=0):
        info = lease.inspect_lease(PROFILE)
        assert info is not None and info.kind == "serve" and info.alive


@pytest.mark.parametrize("component,mode", [("directory", 0o750), ("file", 0o640), ("file", 0o700)])
def test_private_read_rejects_exposed_modes_without_mutating_original(
    tmp_path: Path, component: str, mode: int
) -> None:
    plainfile.write_private(tmp_path, "private", "state.json", data=b"private-value")
    target = tmp_path / "private" / "state.json"
    changed = target.parent if component == "directory" else target
    changed.chmod(mode)
    before = changed.stat()
    with pytest.raises(plainfile.UnsafePathError):
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=13)
    after = changed.stat()
    assert (after.st_mode, after.st_mtime_ns) == (before.st_mode, before.st_mtime_ns)
    assert target.read_bytes() == b"private-value"
    changed.chmod(0o700 if component == "directory" else 0o600)
    assert (
        plainfile.read_private(tmp_path, "private", "state.json", max_bytes=13) == b"private-value"
    )


def test_private_read_accepts_owner_readonly_modes_and_does_not_check_public_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "configured-data-root"
    base.mkdir(mode=0o755)
    plainfile.write_private(base, "private", "state.json", data=b"readonly")
    directory = base / "private"
    target = directory / "state.json"
    directory.chmod(0o500)
    target.chmod(0o400)
    before = (base.stat().st_mode, directory.stat().st_mode, target.stat().st_mode)
    try:
        assert plainfile.read_private(base, "private", "state.json", max_bytes=8) == b"readonly"
        assert (base.stat().st_mode, directory.stat().st_mode, target.stat().st_mode) == before
    finally:
        directory.chmod(0o700)
        target.chmod(0o600)
