"""Descriptor-bound APSW snapshots: readonly native I/O and accepted generations.

Native opens/existence checks use pinned descriptors through a snapshot-scoped
VFS binding. Namespace checks alone are not authority: ancestor replacement and
restoration must never substitute another file. Header/metadata guards separately
reject main/checkpoint/reset changes and retain prior commits during native append.
SQLite heap-WAL startup work remains cooperative, not a measured physical bound.
"""

from __future__ import annotations

import os
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ssgrep.accounting.readers.opencode_vfs import readonly_vfs


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
