"""Exclusive per-profile worker lease: ``flock`` is the truth, the JSON is only a report.

``metrics collect`` and ``metrics serve`` share ``<profile>/worker/lease.lock``. Holding
the lease means holding ``flock(LOCK_EX)`` on an ``O_CLOEXEC`` descriptor of that file:
the kernel drops it when the owner dies for any reason (``SIGKILL`` included) and it can
never be taken from a live holder, whatever the file says. Once the lock is ours we
write ``{pid, process_start, kind, acquired_at}`` into the same file so a refused
acquirer, ``inspect_lease`` and the status surface can name the owner. A pid is only
believed alive when its *start token* still matches the recorded one (pid reuse), and
that judgement is never used to decide who owns the lock.

A forked (never exec'd) child inherits the open file description and therefore keeps the
flock after an unreleased parent dies; that is reported honestly as ``AlreadyRunning``
with ``alive=False`` for the recorded owner. Owner release explicitly unlocks the shared
description, even if a duplicate or inherited descriptor remains open. A forked copy
may close its descriptor but must never unlock or erase the owner's record. Exec closes
the lease descriptor (``O_CLOEXEC``).
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from ssgrep.accounting import storage
from ssgrep.accounting.contracts import canonical_bytes

LEASE_DIR = "worker"
LEASE_FILE = "lease.lock"

_MAX_META = 4096  # the record is ~150 bytes; anything larger is not ours
_CREATE_ATTEMPTS = 100  # bounded: each pass loses a creation race to somebody else's success
_MAX_KIND = 128
_MAX_TOKEN = 256
_MAX_PID = 2**31 - 1
_POLL_SECONDS = 0.01
_PS_TIMEOUT_SECONDS = 5
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_PROC_ROOT = Path("/proc")  # tests point this at a fake tree
_PS = "ps"


@dataclass(frozen=True)
class LeaseInfo:
    """The owner as recorded in the lock file (every field None when unreadable)."""

    pid: int | None
    process_start: str | None
    kind: str | None
    acquired_at: str | None  # UTC YYYY-MM-DDTHH:MM:SSZ
    alive: bool  # pid exists AND its start token still equals ``process_start``


class AlreadyRunning(Exception):
    """Another worker holds the lease; ``info`` is the owner as recorded.

    ``info.alive`` can be False while the lease is still held: a forked child of the
    recorded owner may keep the flock after the owner itself has died.
    """

    def __init__(self, info: LeaseInfo) -> None:
        super().__init__(f"worker lease is held (pid={info.pid}, kind={info.kind})")
        self.info = info


def process_start_token(pid: int) -> str | None:
    """Identity of a running process that differs from any later process reusing ``pid``.

    Linux: ``/proc/<pid>/stat`` start time plus the boot id. Elsewhere: ``ps -o lstart``
    (``LC_ALL=C``, ``TZ=UTC``, second resolution). None when no such process runs;
    zombies and processes being reaped count as not running.
    """
    if not 1 <= pid <= _MAX_PID:
        return None
    return _proc_token(pid) if (_PROC_ROOT / "sys").is_dir() else _ps_token(pid)


def _proc_token(pid: int) -> str | None:
    try:
        raw = (_PROC_ROOT / str(pid) / "stat").read_bytes()
    except OSError:  # absent, exited meanwhile, or not ours to see: not our worker
        return None
    fields = raw.rpartition(b")")[2].split()  # comm may hold spaces and parentheses
    if len(fields) < 20 or fields[0] in (b"Z", b"X", b"x") or not fields[19].isdigit():
        return None
    try:
        boot = (_PROC_ROOT / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = ""
    start = fields[19].decode()
    return f"{boot}:{start}" if boot else start


def _ps_token(pid: int) -> str | None:
    environment = {**os.environ, "LC_ALL": "C", "TZ": "UTC"}
    command = [_PS, "-o", "lstart=", "-o", "stat=", "-p", str(pid)]
    try:
        done = subprocess.run(
            command, capture_output=True, env=environment, timeout=_PS_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return _existence_token(pid)  # no usable ps: existence is all we can attest
    words = done.stdout.decode(errors="replace").split()
    if done.returncode != 0 or len(words) < 2 or words[-1].startswith("Z"):
        return None
    return " ".join(words[:-1])


def _existence_token(pid: int) -> str | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # exists, owned by someone else
    return f"pid:{pid}"


def _load(data: bytes) -> dict[str, Any]:
    if not data or len(data) > _MAX_META:
        return {}
    try:
        raw = json.loads(data)
    except (ValueError, RecursionError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _text(value: object, limit: int) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= limit else None


def _parse(data: bytes) -> LeaseInfo:
    """Metadata from lock-file bytes; each invalid field is None, never an exception."""
    raw = _load(data)
    pid = raw.get("pid")
    pid = pid if type(pid) is int and 0 < pid <= _MAX_PID else None
    start = _text(raw.get("process_start"), _MAX_TOKEN)
    stamp = _text(raw.get("acquired_at"), len("2026-01-01T00:00:00Z"))
    stamp = stamp if stamp and _STAMP.fullmatch(stamp) else None
    alive = pid is not None and start is not None and process_start_token(pid) == start
    return LeaseInfo(pid, start, _text(raw.get("kind"), _MAX_KIND), stamp, alive)


def _read_info(descriptor: int) -> LeaseInfo:
    return _parse(os.pread(descriptor, _MAX_META + 1, 0))


def _open_or_create(dir_fd: int) -> int:
    """Open ``lease.lock`` or create it, whoever else is doing the same right now.

    A plain ``O_CREAT`` open relative to a directory descriptor can fail with ENOENT when
    several processes create the same name at once (observed on APFS), so open first and
    create exclusively, looping until one of the two sticks.
    """
    base = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK  # NONBLOCK: no FIFO hang
    for _ in range(_CREATE_ATTEMPTS):
        for extra in (0, os.O_CREAT | os.O_EXCL):
            try:
                return os.open(LEASE_FILE, base | extra, 0o600, dir_fd=dir_fd)
            except (FileNotFoundError, FileExistsError):
                continue
            except OSError as error:
                if error.errno == errno.ELOOP:
                    raise storage.UnsafeStorageError(
                        error.errno, f"{LEASE_FILE} is a symlink"
                    ) from error
                raise
    raise OSError(errno.EAGAIN, f"could not open or create {LEASE_FILE}")


def _open_lock_file(dir_fd: int) -> int:
    """Create-or-open ``lease.lock`` for locking: owned regular file, mode 0600, CLOEXEC."""
    descriptor = _open_or_create(dir_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise storage.UnsafeStorageError(errno.EINVAL, f"{LEASE_FILE} is not a regular file")
        if info.st_uid != os.geteuid():
            raise storage.UnsafeStorageError(errno.EPERM, f"{LEASE_FILE} is not owned by us")
        if stat.S_IMODE(info.st_mode) != 0o600:  # drop group/other access, restore owner rw
            os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _lock(descriptor: int, retry_seconds: float) -> None:
    """Take the exclusive flock, retrying for ``retry_seconds``; AlreadyRunning otherwise."""
    deadline = time.monotonic() + max(retry_seconds, 0.0)
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AlreadyRunning(_read_info(descriptor)) from None
            time.sleep(min(_POLL_SECONDS, remaining))


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Lease:
    """A held worker lease; use ``Lease.acquire`` (the constructor is internal)."""

    def __init__(
        self, profile: str, kind: str, descriptor: int, recovered_from: LeaseInfo | None
    ) -> None:
        self.profile = profile
        self.kind = kind
        self.recovered_from = recovered_from  # abandoned previous owner's record, if any
        self._descriptor: int | None = descriptor
        self._pid = os.getpid()
        self._guard = threading.Lock()

    @classmethod
    def acquire(
        cls,
        profile: str,
        kind: str,
        *,
        now: datetime | None = None,
        retry_seconds: float = 0.1,
    ) -> Self:
        """Take the profile's exclusive lease or raise ``AlreadyRunning``.

        ``recovered_from`` carries the previous owner's record when one was left behind
        by a worker that died without releasing. ``storage.UnsafeStorageError`` for an
        invalid profile id or an unsafe path.
        """
        if not 0 < len(kind) <= _MAX_KIND:
            raise ValueError(f"lease kind must be 1..{_MAX_KIND} characters")
        dir_fd = storage.open_profile_dir(profile, LEASE_DIR)
        try:
            descriptor = _open_lock_file(dir_fd)
        finally:
            os.close(dir_fd)
        try:
            _lock(descriptor, retry_seconds)
            leftover = os.pread(descriptor, _MAX_META + 1, 0)
            previous = _parse(leftover) if leftover else None
            _record_owner(descriptor, kind, now or datetime.now(UTC))
        except BaseException:
            os.close(descriptor)  # also drops the flock if it was taken
            raise
        return cls(profile, kind, descriptor, previous)

    def release(self) -> None:
        """Unlock and close the owner's lease; forked copies only close their descriptor."""
        with self._guard:
            descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        try:
            if os.getpid() == self._pid:  # a forked copy must not erase the real owner's record
                try:
                    os.ftruncate(descriptor, 0)  # orderly release leaves nothing to recover
                except OSError:
                    pass  # the flock, not the record, is the lease
                # close alone keeps the flock alive through duplicated/inherited descriptors.
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:  # noqa: BLE001 - finalizer at interpreter shutdown: nothing to report
            pass


def _record_owner(descriptor: int, kind: str, moment: datetime) -> None:
    pid = os.getpid()
    record = {
        "pid": pid,
        "process_start": process_start_token(pid),
        "kind": kind,
        "acquired_at": _stamp(moment),
    }
    os.ftruncate(descriptor, 0)
    storage.write_all(descriptor, canonical_bytes(record))  # offset is still 0: nothing seeks
    os.fsync(descriptor)


def inspect_lease(profile: str) -> LeaseInfo | None:
    """The recorded owner, or None when no lease file exists; never creates, never locks.

    A file with no readable record (released, torn, forged) is a ``LeaseInfo`` of Nones.
    Raises ``storage.UnsafeStorageError`` for an invalid profile id or an unsafe path.
    """
    try:
        dir_fd = storage.open_profile_dir(profile, LEASE_DIR, create=False)
    except FileNotFoundError:
        return None
    try:
        descriptor = storage.open_regular(dir_fd, LEASE_FILE)
    except FileNotFoundError:
        return None
    finally:
        os.close(dir_fd)
    try:
        return _read_info(descriptor)
    finally:
        os.close(descriptor)
