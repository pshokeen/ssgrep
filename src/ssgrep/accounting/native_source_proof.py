"""Read-range integrity proofs: tolerate appends, never infer unchanged old bytes."""

from __future__ import annotations

import hashlib
import os
import stat

from ssgrep.accounting.native_source import NativeSourceStatus as Status, SourceReadError


class ReadProof:
    """Hashes only bytes actually read for native identity, context and payload.

    Changed descriptor metadata requires re-reading those ranges under the same
    budget. Size growth alone is NOT evidence that the prefix remained intact.
    Nothing here retains plaintext or reads unselected source tails.
    """

    def __init__(self):
        self.ranges = []

    def add(self, offset, data):
        if not data:
            return
        if self.ranges:
            start, end, digest = self.ranges[-1]
            if start <= offset and offset + len(data) <= end:
                return
            if offset == end:
                digest.update(data)
                self.ranges[-1] = (start, offset + len(data), digest)
                return
        self.ranges.append((offset, offset + len(data), hashlib.sha256(data)))

    def verify(self, descriptor, before, charge):
        after = os.fstat(descriptor)
        self._identity(before, after)

        def fields(info):
            return info.st_size, info.st_mtime_ns, info.st_ctime_ns

        if fields(before) == fields(after):
            return
        for start, end, expected in self.ranges:
            actual = hashlib.sha256()
            position = start
            while position < end:
                data = os.pread(descriptor, min(64 * 1024, end - position), position)
                charge(len(data))
                if not data:
                    raise SourceReadError(Status.CHANGED)
                actual.update(data)
                position += len(data)
            if actual.digest() != expected.digest():
                raise SourceReadError(Status.CHANGED)
        self._identity(before, os.fstat(descriptor))

    @staticmethod
    def _identity(before, after):
        if (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ) or after.st_size < before.st_size:
            raise SourceReadError(Status.CHANGED)
        if not stat.S_ISREG(after.st_mode) or after.st_uid != os.geteuid() or after.st_mode & 0o022:
            raise SourceReadError(Status.UNAPPROVED)
