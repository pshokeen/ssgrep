"""Small flock + canonical-JSON writer for export-owned state files.

Export state lives under ``<data>/accounting/<profile>/<dir...>/<name>``. Writers use
``atomic_write`` (write + fsync + rename + directory fsync) and serialize across processes
with an ``flock`` on a dot-file lock next to the document. Directory descriptors are closed
even when an unsafe lock file prevents acquisition. Floats are refused on write.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.ledger_contracts import valid_profile_id

MAX_STATE_BYTES = 1024 * 1024
LOCK_OPEN_ATTEMPTS = 16


class StateCorrupt(Exception):
    """A state file is unreadable, unsafe, or not strict float-free JSON (never rewritten)."""


def _check(profile: str, parts: tuple[str, ...]) -> None:
    if not valid_profile_id(profile) or not all(
        valid_profile_id(p) or p == "export" for p in parts
    ):
        raise ValueError("profile and path components must be bounded identifiers")


def _open_lock(dir_fd: int, name: str) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(LOCK_OPEN_ATTEMPTS - 1):
        with contextlib.suppress(FileNotFoundError):  # first-creation race reports ENOENT
            return os.open(name, flags, 0o600, dir_fd=dir_fd)
    return os.open(name, flags, 0o600, dir_fd=dir_fd)


@contextlib.contextmanager
def locked(
    profile: str, dirs: tuple[str, ...], name: str, *, wait: bool = True
) -> Iterator[int | None]:
    """Hold the exclusive lock for ``name`` (yields the directory fd; ``None`` if ``wait=False``
    and another process holds it)."""
    _check(profile, dirs)
    try:
        dir_fd = storage.open_profile_dir(profile, *dirs)
    except OSError as error:
        raise StateCorrupt(f"unsafe state path: {error}") from error
    try:
        lock = _open_lock(dir_fd, f".{name}.lock")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield None
                return
            yield dir_fd
        finally:
            os.close(lock)
    finally:
        os.close(dir_fd)


def write_json(dir_fd: int, name: str, document: Any) -> None:
    """Atomically replace ``name`` with the canonical bytes of ``document`` (no floats)."""
    storage.atomic_write(dir_fd, name, canonical_bytes(document) + b"\n")
