"""Data-root resolution without importing the heavy ``ssgrep.store`` package.

``ssgrep.store.paths.data_dir`` lives inside a package whose ``__init__``
imports LanceDB (hundreds of milliseconds). Capture resolves the same location
without that dependency: ``SSGREP_DATA_DIR`` when set, otherwise the platform
user-data directory for ``ssgrep``. A test pins this function to ``data_dir()``
for both branches so the two cannot drift. Root snapshots are noncreating by
default and bind private capture operations to one path and device/inode.
"""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path


def data_root() -> Path:
    override = os.environ.get("SSGREP_DATA_DIR")
    if override:
        return Path(override).expanduser().absolute()
    from platformdirs import user_data_path

    return Path(user_data_path("ssgrep", appauthor=False))


@dataclass(frozen=True, slots=True)
class SafeRootSnapshot:
    """Private configured namespace identity; never serialized into evidence."""

    path: Path
    identity: tuple[int, int]


class RootChanged(OSError):
    """Transient configuration/namespace change, not an unsafe-profile latch."""


def snapshot(path: Path | None = None, *, create: bool = False) -> SafeRootSnapshot | None:
    """Observe the configured root inode without creating anything by default.

    Operator-selected root symlinks remain valid. Capture authorization uses the
    noncreating default; explicit standalone storage callers may create a root.
    """
    path = data_root() if path is None else path
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        return SafeRootSnapshot(path, (info.st_dev, info.st_ino))
    finally:
        os.close(descriptor)


def open_root(root: SafeRootSnapshot) -> int:
    """Pin exactly the authorized inode, never a newly resolved destination.

    Configuration is only a rejection guard: the open always uses ``root.path``,
    and all child operations must be relative to the returned descriptor.
    """
    if data_root() != root.path:
        raise RootChanged(errno.ESTALE, "accounting data root configuration changed")
    try:
        descriptor = os.open(root.path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as error:
        raise RootChanged(errno.ESTALE, "accounting data root is no longer available") from error
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != root.identity:
            raise RootChanged(errno.ESTALE, "accounting data root identity changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def require_current(root: SafeRootSnapshot) -> None:
    """Reject changed configuration or root identity without creating anything."""
    os.close(open_root(root))
