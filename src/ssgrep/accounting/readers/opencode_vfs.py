"""Lazy APSW-native readonly policy with snapshot-scoped descriptor authority.

SQLite sees virtual names. Every native main/WAL open and existence decision is
bound to the same securely pinned SourceGuard, never its mutable original path.
Readonly descriptor I/O uses the public APSW file protocol directly. Native SQLite
still owns SQL/WAL/checksum/recovery; no source pathname or FD proxy is reopened.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any

from ssgrep.accounting.readers.opencode_file import DescriptorFile

_lock = threading.Lock()
_instances: dict[int, Any] = {}
_bindings: dict[str, Any] = {}


def _before_fork():
    _lock.acquire()


def _after_fork():
    _lock.release()


os.register_at_fork(before=_before_fork, after_in_parent=_after_fork, after_in_child=_after_fork)


def readonly_vfs():
    """Process-lifetime callbacks; individual FD bindings last only their snapshot."""
    import apsw

    pid = os.getpid()
    with _lock:
        if pid in _instances:
            return _instances[pid]
        if "unix-none" not in apsw.vfs_names():
            raise apsw.CantOpenError("native read-only WAL VFS is unavailable")

        class ReadOnlyVFS(apsw.VFS):
            def __init__(self):
                self.name = f"ssgrep-native-readonly-{pid}"
                super().__init__(self.name, "unix-none")

            @contextmanager
            def bind(self, guard):
                if guard.pid != os.getpid():
                    raise apsw.CantOpenError("native descriptor belongs to another process")
                main = f"/__ssgrep_native__/{pid}/{id(guard)}/main.db"
                with _lock:
                    _bindings[main] = guard
                try:
                    yield main
                finally:
                    with _lock:
                        _bindings.pop(main, None)

            @staticmethod
            def binding(name):
                filename = name.filename() if isinstance(name, apsw.URIFilename) else name
                if not isinstance(filename, str):
                    raise apsw.CantOpenError("native snapshot requires a bound filename")
                main = filename[:-4] if filename.endswith("-wal") else filename
                with _lock:
                    guard = _bindings.get(main)
                if guard is None or guard.pid != os.getpid():
                    raise apsw.CantOpenError("native snapshot filename is not bound")
                return guard, filename != main

            def xFullPathname(self, name):
                self.binding(name)
                return name

            def xAccess(self, pathname: str, flags: int) -> bool:
                if flags == apsw.SQLITE_ACCESS_READWRITE:
                    return False
                try:
                    guard, wal = self.binding(pathname)
                except apsw.CantOpenError:
                    return False
                return guard.descriptor(wal) is not None

            def xOpen(self, name, flags):
                native = apsw.SQLITE_OPEN_MAIN_DB | apsw.SQLITE_OPEN_WAL
                if name is None or not flags[0] & native:
                    raise apsw.ReadOnlyError("native snapshots deny disk temp/journal opens")
                guard, wal = self.binding(name)
                if bool(flags[0] & apsw.SQLITE_OPEN_WAL) != wal:
                    raise apsw.CantOpenError("native file role does not match binding")
                descriptor = guard.descriptor(wal)
                if descriptor is None:
                    raise apsw.CantOpenError("native WAL was absent at descriptor pinning")
                forbidden = (
                    apsw.SQLITE_OPEN_READWRITE
                    | apsw.SQLITE_OPEN_CREATE
                    | apsw.SQLITE_OPEN_DELETEONCLOSE
                    | apsw.SQLITE_OPEN_EXCLUSIVE
                )
                flags[0] = (flags[0] & ~forbidden) | apsw.SQLITE_OPEN_READONLY
                flags[1] = flags[0]
                return DescriptorFile(descriptor)

            def xDelete(self, filename, syncdir):
                raise apsw.ReadOnlyError("native source deletion is forbidden")

        instance = ReadOnlyVFS()
        _instances[pid] = instance
        return instance
