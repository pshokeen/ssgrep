"""Read-only, descriptor-pinned native files with no symlink traversal."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceFile:
    descriptor: int

    def stat(self) -> os.stat_result:
        return os.fstat(self.descriptor)

    def read(self, offset: int, size: int) -> bytes:
        return os.pread(self.descriptor, size, offset)


@contextmanager
def open_source(path: Path) -> Iterator[SourceFile]:
    """Pin every directory before opening a nonblocking, regular leaf file.

    As with native-source reconstruction, consent never follows a replaced
    symlink component. A renamed directory already pinned by this walk remains
    the authority for the rest of the read.
    """
    path = path.absolute()
    if ".." in path.parts:
        raise OSError(errno.EINVAL, "native source contains a parent traversal", str(path))
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open(path.anchor, flags | os.O_DIRECTORY)
    descriptor = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, flags | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "native source is not a regular file", str(path))
        yield SourceFile(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
