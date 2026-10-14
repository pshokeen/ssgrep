"""Stdlib-only owner-protected file primitives for the policy and read surface.

Independent of the capture-side ``storage`` module on purpose: policy must be
loadable and writable even when journal storage is unavailable, and a read
must never create anything. Every directory below the data root is checked
for symlinks, foreign owners, and group/other write bits before use.
"""

from __future__ import annotations

import fcntl
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class UnsafePathError(OSError):
    """A path component is linked, foreign-owned, or has unsafe permissions."""


def _check_dir(path: Path, *, private: bool = False) -> bool:
    """True when ``path`` is a safe directory; False when absent."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise UnsafePathError(f"{path} is not a plain directory")
    if info.st_uid != os.geteuid():
        raise UnsafePathError(f"{path} is not owned by the current user")
    if info.st_mode & 0o022:
        raise UnsafePathError(f"{path} is writable by other users")
    if private and info.st_mode & 0o077:
        raise UnsafePathError(f"{path} is accessible to other users")
    return True


def read_private(base: Path, *parts: str, max_bytes: int) -> bytes | None:
    """File bytes under ``base``/``parts``; None when any component is absent.

    Raises ``UnsafePathError`` for unsafe components and ``ValueError`` when
    the file exceeds ``max_bytes``. Never creates or chmods anything. Components below
    ``base`` must be owner-only; owner read-only permissions remain valid.
    """
    current = base
    for part in parts[:-1]:
        current = current / part
        if not _check_dir(current, private=True):
            return None
    target = current / parts[-1]
    try:
        descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise UnsafePathError(f"{target} cannot be opened safely: {error.strerror}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError(f"{target} is not a regular file")
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise UnsafePathError(f"{target} is foreign-owned or writable by other users")
        if info.st_mode & 0o177:
            raise UnsafePathError(f"{target} has unsafe file permissions")
        if info.st_size > max_bytes:
            raise ValueError(f"{target} exceeds {max_bytes} bytes")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:  # a writer may have appended since fstat
            raise ValueError(f"{target} exceeds {max_bytes} bytes")
        return data
    finally:
        os.close(descriptor)


def _ensure_dirs(base: Path, parts: tuple[str, ...]) -> Path:
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = base
    for part in parts:
        current = current / part
        if not _check_dir(current):
            os.mkdir(current, 0o700)
    return current


def write_private(base: Path, *parts: str, data: bytes) -> None:
    """Atomically replace ``base``/``parts`` with ``data`` (dirs 0700, file 0600)."""
    directory = _ensure_dirs(base, parts[:-1])
    target = directory / parts[-1]
    temporary = directory / f".{parts[-1]}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def exclusive_lock(base: Path, *dir_parts: str) -> Iterator[None]:
    """Serialize read-modify-write cycles across processes (flock on ``.lock``)."""
    directory = _ensure_dirs(base, dir_parts)
    descriptor = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)
