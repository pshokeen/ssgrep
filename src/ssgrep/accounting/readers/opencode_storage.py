"""Read-only OpenCode cursor, WAL-backed snapshot, and record storage machinery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.readers.base import ReadResult, loads

IGNORED_PARTS = frozenset(
    {
        "text",
        "reasoning",
        "step-start",
        "step-finish",
        "snapshot",
        "patch",
        "file",
        "agent",
        "subtask",
        "compaction",
        "retry",
    }
)
# The owning session's directory, joined in SQL so it never depends on paging order.
DIRECTORY = "(SELECT s.directory FROM session AS s WHERE s.id = {table}.session_id) AS cwd, "
PENDING_STATES = frozenset({"pending", "running"})
DEFAULT_BUDGET = 1000
REQUIRED = {
    "session": {"id", "parent_id", "directory", "project_id", "time_updated"},
    "message": {"id", "session_id", "time_updated", "data"},
    "part": {"id", "message_id", "session_id", "time_updated", "data"},
}
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


class SnapshotError(RuntimeError):
    """Native snapshot rejected; no rows/cursor may be committed."""


class SnapshotChanged(SnapshotError):
    """An approved source changed during the optimistic native transaction."""


class SnapshotUnavailable(SnapshotError):
    """A nonmutating native backend cannot be established."""


@dataclass
class SnapshotTracker:
    before: dict | None = None
    after: dict | None = None


def metadata(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def pin_directory(path):
    """No-follow traversal from filesystem root, retaining the actual parent FD."""
    absolute = path.absolute()
    # Only fixed OS-owned Darwin aliases are normalized, never user ancestors.
    if sys.platform == "darwin" and len(absolute.parts) > 1:
        prefix = absolute.parts[1]
        if prefix in {"var", "tmp", "etc"}:
            try:
                target = os.readlink("/" + prefix)
            except OSError:
                target = None
            if target in {"private/" + prefix, "/private/" + prefix}:
                absolute = Path("/private", *absolute.parts[1:])
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            if part == "..":
                raise SnapshotChanged("native source parent traversal is ambiguous")
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def published_commit(directory, shared_name, descriptor, wal_header, size):
    """Optional bounded same-size append proof; SHM read marks/mtime are ignored."""
    shared = None
    try:
        shared = os.open(
            shared_name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory,
        )
        if not stat.S_ISREG(os.fstat(shared).st_mode):
            return None
        header = os.pread(shared, 96, 0)
        if len(header) != 96 or header[:48] != header[48:]:
            return None
        if int.from_bytes(header[:4], sys.byteorder) != 3007000 or header[12] != 1:
            return None
        frame = int.from_bytes(header[16:20], sys.byteorder)
        if frame == 0:
            return 0, 0, b""
        if header[32:40] != wal_header[16:24]:
            return None
        page_size = int.from_bytes(wal_header[8:12], "big")
        if page_size < 512 or page_size > 65536 or page_size & (page_size - 1):
            return None
        offset = 32 + (frame - 1) * (24 + page_size)
        if offset + 24 + page_size > size:
            return None
        anchor = os.pread(descriptor, 24, offset)
        if (
            len(anchor) != 24
            or anchor[8:16] != wal_header[16:24]
            or not int.from_bytes(anchor[4:8], "big")
        ):
            return None
        return frame, offset, anchor
    except OSError:
        return None
    finally:
        if shared is not None:
            os.close(shared)


class SourceGuard:
    """Main/WAL descriptors from one securely traversed directory; bounded headers."""

    def __init__(self, database: Path, descriptor: int | None = None):
        self.pid = os.getpid()
        self.files = []
        self.directory = None
        self.shared_name = database.name + "-shm"
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            self.directory = pin_directory(database.parent)
            for suffix, header_size in (("", 100), ("-wal", 32)):
                name = database.name + suffix
                try:
                    held = (
                        os.dup(descriptor)
                        if not suffix and descriptor is not None
                        else os.open(name, flags, dir_fd=self.directory)
                    )
                except FileNotFoundError:
                    if not suffix:
                        raise
                    self.files.append((name, None, header_size, None))
                    continue
                self.files.append((name, held, header_size, None))
                if not stat.S_ISREG(os.fstat(held).st_mode):
                    raise SnapshotChanged("native SQLite source is not a regular file")
                stamp = self.stamp(name, held, header_size)
                self.files[-1] = name, held, header_size, stamp
            self.commit = self.commit_at(self.files[1][3]) if self.has_wal else None
        except BaseException:
            self.close()
            raise

    def stamp(self, name, descriptor, header_size):
        if self.pid != os.getpid():
            raise SnapshotChanged("native descriptor belongs to another process")
        before = metadata(os.fstat(descriptor))
        header = os.pread(descriptor, header_size, 0)
        after = metadata(os.fstat(descriptor))
        current = metadata(os.stat(name, dir_fd=self.directory, follow_symlinks=False))
        if before != after or after != current:
            raise SnapshotChanged("native SQLite descriptor/directory entry changed")
        return after, header

    def descriptor(self, wal=False):
        if self.pid != os.getpid():
            raise SnapshotChanged("native descriptor belongs to another process")
        return self.files[int(wal)][1]

    @property
    def has_wal(self):
        return self.descriptor(True) is not None

    def commit_at(self, stamp):
        return published_commit(
            self.directory, self.shared_name, self.descriptor(True), stamp[1], stamp[0][2]
        )

    def generation(self):
        result = {}
        for key, (name, descriptor, header_size, _) in zip(
            ("main", "wal"), self.files, strict=True
        ):
            if descriptor is None:
                result[key] = None
            else:
                info, header = self.stamp(name, descriptor, header_size)
                result[key] = [*info, header.hex()]
        return result

    def verify(self):
        try:
            for name, descriptor, header_size, expected in self.files:
                if descriptor is None:
                    try:
                        os.stat(name, dir_fd=self.directory, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    raise SnapshotChanged("native SQLite sidecar appeared")
                assert expected is not None  # Present descriptors are always pinned with a stamp.
                actual = self.stamp(name, descriptor, header_size)
                if actual == expected:
                    continue
                append = (
                    self.has_wal
                    and name == self.files[1][0]
                    and actual[0][:2] == expected[0][:2]
                    and actual[0][2] >= expected[0][2]
                    and actual[1] == expected[1]
                )
                if append and self.commit is not None and self.commit[0] > 0:
                    append = os.pread(descriptor, 24, self.commit[1]) == self.commit[2]
                if append and actual[0][2] == expected[0][2]:
                    current = self.commit_at(actual)
                    append = (
                        self.commit is not None
                        and current is not None
                        and current[0] > self.commit[0]
                    )
                if not append:
                    raise SnapshotChanged("native SQLite source changed during snapshot")
        except OSError as error:
            raise SnapshotChanged("native SQLite source became unavailable") from error

    def close(self):
        for _, descriptor, _, _ in self.files:
            if descriptor is not None:
                os.close(descriptor)
        self.files.clear()
        if self.directory is not None:
            os.close(self.directory)
            self.directory = None


def source_generation(database: Path):
    """Current secure metadata/header identity for idle checks; no SQL/APSW needed."""
    guard = SourceGuard(database)
    try:
        guard.verify()
        return guard.generation()
    finally:
        guard.close()


@contextmanager
def snapshot(database: Path, authorize, *, descriptor=None, tracker=None):
    """Yield a real APSW connection using only this snapshot's descriptor binding."""
    try:
        import apsw
    except ImportError as error:
        raise SnapshotUnavailable("APSW native SQLite backend is unavailable") from error
    try:
        guard = SourceGuard(database, descriptor)
    except OSError as error:
        raise SnapshotChanged("native SQLite source changed before snapshot") from error
    connection = None
    try:
        if tracker is not None:
            tracker.before = guard.generation()
        vfs = readonly_vfs()
        with vfs.bind(guard) as main:
            try:
                uri = "file:" + main + ("" if guard.has_wal else "?immutable=1")
                connection = apsw.Connection(
                    uri, flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI, vfs=vfs.name
                )
                if guard.has_wal:
                    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA temp_store=MEMORY")
                connection.execute("BEGIN")
                connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
                guard.verify()
                connection.set_row_trace(named_row)
                connection.set_authorizer(authorize)
                yield connection
            finally:
                if connection is not None:
                    try:
                        connection.set_progress_handler(None)
                        if not connection.get_autocommit():
                            connection.execute("ROLLBACK")
                    finally:
                        connection.close()
                guard.verify()
                if tracker is not None:
                    tracker.after = guard.generation()
    finally:
        guard.close()


def named_row(cursor, values):
    return {
        column[0]: value for column, value in zip(cursor.get_description(), values, strict=True)
    }


def authorize(tables_read, action, table_name, column_name):
    """Native APSW authorizer: source/schema reads, never writes or credentials."""
    import apsw

    if action == apsw.SQLITE_READ:
        table = (table_name or "").lower()
        if table in {"session", "message", "part"}:
            tables_read.add(table)
            return apsw.SQLITE_OK
        schemas = {"sqlite_master", "sqlite_schema", "sqlite_temp_master"}
        return apsw.SQLITE_OK if table in schemas else apsw.SQLITE_DENY
    if action in {
        apsw.SQLITE_SELECT,
        apsw.SQLITE_FUNCTION,
        apsw.SQLITE_TRANSACTION,
        apsw.SQLITE_RECURSIVE,
    }:
        return apsw.SQLITE_OK
    if action == apsw.SQLITE_PRAGMA and (table_name or "").lower() in {"query_only", "table_info"}:
        return apsw.SQLITE_OK
    return apsw.SQLITE_DENY


USER_ROLE_SQL = (
    "(SELECT CASE WHEN length(CAST(m.data AS BLOB)) > 8388608 THEN 'source_read_bound' "
    "ELSE json_extract(m.data, '$.role') END FROM message AS m "
    "WHERE m.id = part.message_id AND m.session_id = part.session_id) AS message_role"
)


def session_row(connection, session_id, budget):
    """One native owner row, bounded before strings enter Python."""
    from ssgrep.accounting.readers.native_source import NativeSourceStatus, SourceReadError

    fields = ", ".join(
        f"CASE WHEN length(CAST({name} AS BLOB)) > 16384 THEN NULL ELSE {name} END AS {name}"
        for name in ("parent_id", "directory", "project_id")
    )
    size = " + ".join(
        f"coalesce(length(CAST({name} AS BLOB)), 0)"
        for name in ("id", "parent_id", "directory", "project_id")
    )
    row = connection.execute(
        f"SELECT id, {fields}, time_updated, NULL AS data, ({size}) AS n "
        "FROM session WHERE id = ? LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        raise SourceReadError(NativeSourceStatus.MISSING)
    if row["n"] > min(16384, budget):
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    return row


def user_part(reader, source, row, result):
    """Native user text is prior knowledge, not a discovered tool witness."""
    from ssgrep.accounting.readers.native_source import NativeSourceStatus, SourceReadError

    if row["message_role"] == "source_read_bound":
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    if row["message_role"] != "user":
        return False
    try:
        data = loads(row["data"])
    except (ValueError, TypeError, RecursionError):
        raise SourceReadError(NativeSourceStatus.INCOMPLETE) from None
    if not isinstance(data, dict) or data.get("type") != "text":
        return False
    text = data.get("text")
    if isinstance(text, str) and text:
        result.evidence.append(
            reader._evidence(
                source,
                row["session_id"],
                f"turn:{row['id']}",
                "user_turn",
                f"part:{row['id']}",
                row["time_updated"],
                {"text": text, "cwd": row["cwd"]},
            )
        )
    return True


def source_rows(connection, table, reference, cwd, limit, budget):
    """Whole selected session once, or only a previously verified native row set."""
    from ssgrep.accounting.readers.native_source import (
        MAX_SOURCE_RECORDS,
        NativeSourceStatus,
        SourceReadError,
    )

    if len(reference.row_locators) > MAX_SOURCE_RECORDS:
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    ids = sorted(
        {
            locator.split(":", 1)[1]
            for locator in reference.row_locators
            if locator.startswith(table + ":")
        }
    )
    if any(len(row_id) > 512 for row_id in ids):
        raise SourceReadError(NativeSourceStatus.BOUNDED)
    if reference.row_locators and not ids:
        return
    batches = (
        [ids[start : start + 500] for start in range(0, len(ids), 500)]
        if reference.row_locators
        else [()]
    )
    message_id = (
        "CASE WHEN length(message_id) > 512 THEN NULL ELSE message_id END AS message_id"
        if table == "part"
        else "NULL AS message_id"
    )
    role = USER_ROLE_SQL if table == "part" else "NULL AS message_role"
    for batch in batches:
        clause = " AND id IN (" + ",".join("?" for _ in batch) + ")" if batch else ""
        query = (
            "SELECT CASE WHEN length(id) > 512 THEN NULL ELSE id END AS id, "
            f"session_id, {message_id}, {role}, time_updated, ? AS cwd, "
            "CASE WHEN length(CAST(data AS BLOB)) > ? THEN NULL ELSE data END AS data, "
            f"length(CAST(data AS BLOB)) AS n FROM {table} WHERE session_id = ?{clause} "
            "ORDER BY time_updated, id LIMIT ?"
        )
        yield from connection.execute(
            query, (cwd, min(MAX_RAW_RECORD_BYTES, budget), reference.session_id, *batch, limit)
        )


def read_session(reader, reference, descriptor, budget, max_records):
    """Use native IDs and the existing decoder; never scan unrelated sessions.

    SQL bounds data before materialization; a progress handler bounds query work.
    The database digest covers selected logical rows (including timing), not the
    main file alone, so committed WAL changes remain observable.
    """
    from ssgrep.accounting.readers.native_source import (
        MAX_SOURCE_RECORDS,
        SOURCE_KINDS,
        NativeSourceStatus,
        SourceReadError,
    )

    try:
        import apsw
    except ImportError:
        raise SourceReadError(NativeSourceStatus.UNREADABLE) from None
    source = reader.source_for(reference.path)
    info = os.fstat(descriptor)
    result = ReadResult()
    count = rows_read = 0
    row_limit = min(MAX_SOURCE_RECORDS, max_records)
    steps = 0

    def progress():
        nonlocal steps
        steps += 1
        return int(steps > 1000)

    try:
        with reader.snapshot(reference.path, descriptor=descriptor) as connection:
            connection.set_progress_handler(progress, 1000)
            connection.set_row_trace(named_row)
            for table, columns in REQUIRED.items():
                found = {row["name"] for row in connection.execute(f'PRAGMA table_info("{table}")')}
                if not columns <= found:
                    raise SourceReadError(NativeSourceStatus.UNSUPPORTED)
            session = session_row(connection, reference.session_id, budget)
            count += session["n"]
            reader._row(source, "session", session, [], result)
            for table in ("message", "part"):
                cursor = source_rows(
                    connection,
                    table,
                    reference,
                    session["directory"],
                    row_limit + 1,
                    max(0, budget - count),
                )
                held = []
                for row in cursor:
                    rows_read += 1
                    if rows_read > row_limit or row["data"] is None or row["id"] is None:
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                    size = int(row["n"] or 0)
                    if count + size > budget:
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                    count += size
                    if table != "part" or not user_part(reader, source, row, result):
                        reader._row(source, table, row, held, result)
                    if len(result.evidence) > row_limit:
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                if held or result.gaps:
                    raise SourceReadError(NativeSourceStatus.INCOMPLETE)
            connection.set_progress_handler(None, 0)
    except SourceReadError as error:
        error.bytes_read = count
        raise
    except SnapshotChanged:
        raise SourceReadError(NativeSourceStatus.CHANGED, count) from None
    except SnapshotUnavailable:
        raise SourceReadError(NativeSourceStatus.UNREADABLE, count) from None
    except apsw.Error as error:
        status = (
            NativeSourceStatus.BOUNDED
            if isinstance(error, apsw.InterruptError)
            else NativeSourceStatus.UNREADABLE
        )
        raise SourceReadError(status, count) from None
    # SQLite has no descriptor-open API: reject replacement between the securely
    # checked descriptor and the path used for the native WAL-aware connection.
    try:
        after = os.stat(reference.path, follow_symlinks=False)
    except FileNotFoundError:
        raise SourceReadError(NativeSourceStatus.MISSING, count) from None
    except OSError:
        raise SourceReadError(NativeSourceStatus.UNREADABLE, count) from None
    if (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
        raise SourceReadError(NativeSourceStatus.UNAPPROVED, count)
    evidence = [item for item in result.evidence if item.kind in SOURCE_KINDS]
    # Pin native measurements, not session.time_updated or uncredited usage rows:
    # those can legitimately change when new messages append to the same session.
    manifest = sorted((item.native_identity, item.digest, item.ts_utc) for item in evidence)
    digest = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    return evidence, digest, count
