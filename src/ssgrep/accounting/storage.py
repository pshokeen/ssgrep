"""Private accounting storage: owner/symlink checks, 0700/0600, atomic writes.

Everything lives under ``<data-root>/accounting/`` (never ``notes/``, the repo
``.ssgrep/``, or LanceDB). Directory components below the data root are opened
one at a time with ``O_NOFOLLOW`` relative to the parent descriptor, so a
component replaced by a symlink (or owned by another user) is refused rather
than followed: callers get :class:`UnsafeStorageError` and must disable the
affected capture instead of writing through an untrusted destination.

Files are addressed by (directory descriptor, name) only, which also stops a
name from escaping its directory. Atomic publication is write to a unique
temporary name, fsync, rename, then fsync of the directory. Nothing here runs
on import beyond stdlib imports; the data root is resolved lazily.
"""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import stat
from pathlib import Path

from ssgrep.accounting import datadir
from ssgrep.accounting.ledger_contracts import valid_profile_id

ACCOUNTING_DIR = "accounting"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_DIR_MODE = 0o700
_FILE_MODE = 0o600


class UnsafeStorageError(OSError):
    """A component is linked, foreign-owned, non-plain, or exposed to other users."""


def data_root() -> Path:
    """The application data root (resolved without importing the heavy store package)."""
    return datadir.data_root()


def _own(info: os.stat_result, what: str) -> None:
    if info.st_uid != os.geteuid():
        raise UnsafeStorageError(errno.EPERM, f"{what} is not owned by the current user")


def _open_component(parent_fd: int, name: str, *, create: bool) -> int:
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise UnsafeStorageError(errno.EINVAL, f"unsafe path component {name!r}")
    try:
        try:
            descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            with contextlib.suppress(FileExistsError):  # lost a creation race: open the winner's
                os.mkdir(name, _DIR_MODE, dir_fd=parent_fd)
            descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafeStorageError(
                error.errno, f"{name} is a symlink or not a directory"
            ) from error
        raise
    try:
        info = os.fstat(descriptor)
        _own(info, name)
        if info.st_mode & 0o077:
            if not create:
                raise UnsafeStorageError(errno.EPERM, f"{name} is accessible to other users")
            os.fchmod(descriptor, stat.S_IMODE(info.st_mode) & 0o700)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_dir(
    *parts: str,
    create: bool = True,
    root: datadir.SafeRootSnapshot | None = None,
) -> int:
    """Descriptor for ``<data-root>/accounting/<parts...>`` (owned, private directories).

    The data root itself is the operator's configured location and is opened
    as given (created 0700 when absent and ``create``); every component below
    it must be a plain, owned, non-symlink directory. Read-only opens refuse
    group/other access without changing permissions; creating opens may tighten
    them, never broadening owner access. ``FileNotFoundError`` when a component
    is absent and ``create`` is false.

    An explicit root binds the operation to its authorized path and device/inode;
    it is never created or reopened from mutable configuration. Child mutations
    remain relative to that pinned descriptor even if configuration changes.
    """
    if root is None:
        path = data_root()
        if create:
            path.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        current = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    else:
        current = datadir.open_root(root)
    try:
        for part in (ACCOUNTING_DIR, *parts):
            following = _open_component(current, part, create=create)
            os.close(current)
            current = following
    except BaseException:
        os.close(current)
        raise
    return current


def open_profile_dir(
    profile: str,
    *sub: str,
    create: bool = True,
    root: datadir.SafeRootSnapshot | None = None,
) -> int:
    """Per-profile directory; a profile id is an identifier, never a path."""
    if not valid_profile_id(profile):
        raise UnsafeStorageError(errno.EINVAL, "profile ids are bounded identifiers, not paths")
    return open_dir(profile, *sub, create=create, root=root)


def create_exclusive(dir_fd: int, name: str) -> int:
    """Create a brand-new 0600 append-only file; never reuses or follows an existing name."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
    return os.open(name, flags, _FILE_MODE, dir_fd=dir_fd)


def open_regular(dir_fd: int, name: str) -> int:
    """Read an owned private regular file; refuse unsafe modes without changing permissions."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK  # a FIFO must not hang
    try:
        descriptor = os.open(name, flags, dir_fd=dir_fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise UnsafeStorageError(error.errno, f"{name} is a symlink") from error
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise UnsafeStorageError(errno.EINVAL, f"{name} is not a regular file")
        _own(info, name)
        if info.st_mode & 0o177:  # group/other access or any execute bit
            raise UnsafeStorageError(errno.EPERM, f"{name} has unsafe file permissions")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def read_file(dir_fd: int, name: str, max_bytes: int) -> bytes | None:
    """Whole file, or None when absent; ``ValueError`` if larger than ``max_bytes``."""
    try:
        descriptor = open_regular(dir_fd, name)
    except FileNotFoundError:
        return None
    try:
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(descriptor, max_bytes + 1 - len(data))
            if not chunk:
                return data
            data += chunk
        raise ValueError(f"{name} exceeds {max_bytes} bytes")
    finally:
        os.close(descriptor)


def list_names(dir_fd: int) -> list[str]:
    return sorted(os.listdir(dir_fd))


def fsync_dir(dir_fd: int) -> None:
    os.fsync(dir_fd)


def write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "write made no progress")
        view = view[written:]


def _temporary_file(dir_fd: int, name: str, data: bytes) -> str:
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(temporary, flags, _FILE_MODE, dir_fd=dir_fd)
    try:
        write_all(descriptor, data)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=dir_fd)
        raise
    os.close(descriptor)
    return temporary


def atomic_write(dir_fd: int, name: str, data: bytes) -> None:
    """Replace ``name``: write temporary, fsync, rename, fsync the directory."""
    temporary = _temporary_file(dir_fd, name, data)
    try:
        os.rename(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        os.unlink(temporary, dir_fd=dir_fd)
        raise
    fsync_dir(dir_fd)


def write_once(dir_fd: int, name: str, data: bytes) -> bool:
    """Atomically create ``name`` iff absent (hard link); False when it already exists."""
    temporary = _temporary_file(dir_fd, name, data)
    try:
        os.link(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    except FileExistsError:
        return False
    finally:
        os.unlink(temporary, dir_fd=dir_fd)
    fsync_dir(dir_fd)
    return True


def delete_file(dir_fd: int, name: str) -> bool:
    """Unlink an owned regular file inside ``dir_fd``; refuse symlinks, dirs, foreign files."""
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise UnsafeStorageError(errno.EINVAL, f"{name} is not a regular file")
    _own(info, name)
    os.unlink(name, dir_fd=dir_fd)
    return True
