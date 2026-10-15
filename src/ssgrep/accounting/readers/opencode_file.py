"""Minimal public APSW file protocol for an already-approved readonly descriptor.

No path is reopened. SQLite retains all SQL/WAL/checksum/recovery behavior; this
object supplies only readonly POSIX bytes and the explicit no-lock contract used
by the heap WAL index. APSW zero-fills returned short reads and reports
SQLITE_IOERR_SHORT_READ; returning padded success would be incorrect.

https://rogerbinns.github.io/apsw/vfs.html#apsw.VFSFile.xRead
"""

from __future__ import annotations

import fcntl
import os

READ_CHUNK_BYTES = 64 * 1024


class DescriptorFile:
    def __init__(self, approved_fd: int):
        import apsw

        self._apsw = apsw
        self._pid = os.getpid()
        self._fd = None
        try:
            if fcntl.fcntl(approved_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
                raise apsw.ReadOnlyError("native snapshot requires a readonly descriptor")
            self._fd = os.dup(approved_fd)
        except OSError as error:
            raise apsw.CantOpenError("approved native descriptor cannot be duplicated") from error

    def _descriptor(self):
        if self._pid != os.getpid():
            raise self._apsw.IOError("native snapshot file cannot be reused across fork")
        if self._fd is None:
            raise self._apsw.IOError("native snapshot file is closed")
        return self._fd

    def _read(self, amount, offset):
        while True:
            try:
                return os.pread(self._descriptor(), min(amount, READ_CHUNK_BYTES), offset)
            except InterruptedError:
                continue
            except OSError as error:
                raise self._apsw.IOError("native descriptor read failed") from error

    def xRead(self, amount: int, offset: int) -> bytes:
        """Profiler hook for actual logical bytes served, not requested/padded bytes."""
        self._descriptor()
        if amount < 0 or offset < 0:
            raise self._apsw.IOError("invalid native read range")
        if amount == 0:
            return b""
        first = self._read(amount, offset)
        if len(first) == amount or not first:
            return first
        pieces, read = [first], len(first)
        while read < amount:
            piece = self._read(amount - read, offset + read)
            if not piece:
                break
            pieces.append(piece)
            read += len(piece)
        return b"".join(pieces)

    def xFileSize(self) -> int:
        try:
            return os.fstat(self._descriptor()).st_size
        except OSError as error:
            raise self._apsw.IOError("native descriptor size unavailable") from error

    def xClose(self):
        # Child cleanup may close its inherited duplicate, never use it for IO.
        descriptor, self._fd = self._fd, None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                raise self._apsw.IOError("native descriptor close failed") from error

    def xWrite(self, data, offset):
        self._descriptor()
        raise self._apsw.ReadOnlyError("native snapshot writes are forbidden")

    def xTruncate(self, newsize):
        self._descriptor()
        raise self._apsw.ReadOnlyError("native snapshot truncation is forbidden")

    def xLock(self, level):
        self._descriptor()  # Explicit no-lock readonly strategy; guards detect races.

    def xUnlock(self, level):
        self._descriptor()

    def xCheckReservedLock(self):
        self._descriptor()
        return False

    def xSync(self, flags):
        self._descriptor()  # No write is accepted, hence no dirty data to sync.

    def xFileControl(self, op, pointer):
        self._descriptor()
        return False  # Unsupported; never dereference a caller/native pointer.

    def xDeviceCharacteristics(self):
        self._descriptor()
        return 0

    def xSectorSize(self):
        self._descriptor()
        return 512
