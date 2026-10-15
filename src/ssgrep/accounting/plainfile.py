"""Stdlib-only owner-protected file primitives for the policy and read surface.

Independent of the capture-side ``storage`` module on purpose: policy must be
loadable and writable even when journal storage is unavailable, and a read
must never create anything. Every directory below the configured data root is
opened relative to its pinned parent without following symlinks and checked
for foreign owners or group/other access before private reads.
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


def _validate_dir(info: os.stat_result, path: Path | str, *, private: bool) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafePathError(f"{path} is not a plain directory")
    if info.st_uid != os.geteuid():
        raise UnsafePathError(f"{path} is not owned by the current user")
    if info.st_mode & 0o022:
        raise UnsafePathError(f"{path} is writable by other users")
    if private and info.st_mode & 0o077:
        raise UnsafePathError(f"{path} is accessible to other users")


def _check_dir(path: Path, *, private: bool = False) -> bool:
    """True when ``path`` is a safe directory; False when absent."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    _validate_dir(info, path, private=private)
    return True


def _open_private_directory(parent: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise UnsafePathError(f"{name} cannot be opened safely: {error.strerror}") from error
    try:
        _validate_dir(os.fstat(descriptor), name, private=True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def open_private(base: Path | int, *parts: str) -> Iterator[int | None]:
    """Pin an owned private regular file; yield None when any component is absent.

    A root pathname preserves operator-selected symlink locations; a supplied
    root descriptor is duplicated, preserving its pinned identity. Every
    component below it is a single no-follow relative open.
    Namespace changes cannot redirect a read after its parent has been pinned.
    Never creates, chmods or waits for a FIFO writer.
    """
    if not parts or any(
        not part or part in {".", ".."} or "/" in part or "\0" in part for part in parts
    ):
        raise UnsafePathError("unsafe private path component")
    try:
        directory = (
            os.dup(base)
            if isinstance(base, int)
            else os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        )
    except FileNotFoundError:
        yield None
        return
    try:
        for part in parts[:-1]:
            try:
                following = _open_private_directory(directory, part)
            except FileNotFoundError:
                yield None
                return
            os.close(directory)
            directory = following
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=directory,
            )
        except FileNotFoundError:
            yield None
            return
        except OSError as error:
            raise UnsafePathError(
                f"{parts[-1]} cannot be opened safely: {error.strerror}"
            ) from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise UnsafePathError(f"{parts[-1]} is not a regular file")
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise UnsafePathError(f"{parts[-1]} is foreign-owned or writable by other users")
            if info.st_mode & 0o177:
                raise UnsafePathError(f"{parts[-1]} has unsafe file permissions")
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def read_descriptor(descriptor: int, *, max_bytes: int) -> bytes:
    """Bounded bytes from an already validated private regular-file descriptor."""
    if os.fstat(descriptor).st_size > max_bytes:
        raise ValueError(f"private file exceeds {max_bytes} bytes")
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:  # a writer may have appended since fstat
        raise ValueError(f"private file exceeds {max_bytes} bytes")
    return data


def read_private(base: Path, *parts: str, max_bytes: int) -> bytes | None:
    """Bounded private bytes under ``base``; None when any component is absent.

    Unsafe components are rejected without repairing permissions. Owner read-only
    directories and files remain valid, and the read never creates anything.
    """
    with open_private(base, *parts) as descriptor:
        return None if descriptor is None else read_descriptor(descriptor, max_bytes=max_bytes)


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
