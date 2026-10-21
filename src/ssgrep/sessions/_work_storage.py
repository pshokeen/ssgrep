"""Pinned owner-only filesystem operations for saved investigation metadata."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager

from ssgrep.accounting.privacy.storage import (
    UnsafeStorageError,
    atomic_write,
    data_root,
    read_file,
    write_once,
)

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _check(descriptor: int, *, directory: bool, private: bool = True) -> None:
    info = os.fstat(descriptor)
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    mask = (0o077 if private else 0o022) if directory else 0o177
    if not expected(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & mask:
        raise UnsafeStorageError("unsafe work metadata storage")


@contextmanager
def directory(parent: int, name: str, *, create: bool) -> Iterator[int]:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
    descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent)
    try:
        _check(descriptor, directory=True)
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def store(*, create: bool = False) -> Iterator[int]:
    """Pin the configured root and private namespace; only writers take a lock."""
    root = data_root()
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(root, _DIR_FLAGS)
    try:
        _check(descriptor, directory=True, private=False)
        with directory(descriptor, "work-metadata", create=create) as metadata:
            if not create:
                yield metadata
                return
            flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            lock = os.open(".lock", flags, 0o600, dir_fd=metadata)
            try:
                _check(lock, directory=False)
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield metadata
            finally:
                os.close(lock)
    finally:
        os.close(descriptor)


def replace(directory_fd: int, name: str, data: bytes, *, limit: int) -> None:
    """Validate an existing target before replacing it atomically."""
    read_file(directory_fd, name, limit)
    atomic_write(directory_fd, name, data)


def immutable(directory_fd: int, name: str, data: bytes) -> None:
    """Publish a new revision; the caller has already recovered any old orphans."""
    if not write_once(directory_fd, name, data):
        raise ValueError("work revision changed during publication")
