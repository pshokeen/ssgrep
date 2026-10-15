"""Readonly callbacks belong to the actual maintained driver and outlive readers."""

from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import apsw
import pytest

from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.opencode_snapshot import SourceGuard
from ssgrep.accounting.readers.opencode_vfs import readonly_vfs
from tests.accounting.readers.test_opencode_nonmutating import inventory, seed


def test_native_file_open_and_delete_are_readonly_even_when_requested_writable(tmp_path):
    database = tmp_path / "native.db"
    seed(database).close()
    vfs = readonly_vfs()
    before = inventory(database)
    guard = SourceGuard(database)
    try:
        with vfs.bind(guard) as main:
            flags = [
                apsw.SQLITE_OPEN_MAIN_DB
                | apsw.SQLITE_OPEN_READWRITE
                | apsw.SQLITE_OPEN_CREATE
                | apsw.SQLITE_OPEN_DELETEONCLOSE
                | apsw.SQLITE_OPEN_NOFOLLOW,
                0,
            ]
            handle = vfs.xOpen(main, flags)
            try:
                assert flags[1] & apsw.SQLITE_OPEN_READONLY
                assert not flags[1] & apsw.SQLITE_OPEN_READWRITE
                with pytest.raises(apsw.Error):
                    handle.xWrite(b"unsafe", 0)
                with pytest.raises(apsw.Error):
                    handle.xTruncate(0)
            finally:
                handle.xClose()
            with pytest.raises(apsw.ReadOnlyError):
                vfs.xDelete(main, False)
            assert not vfs.xAccess(main + "-wal", apsw.SQLITE_ACCESS_EXISTS)
            with pytest.raises(apsw.CantOpenError):
                vfs.xOpen(
                    main + "-wal",
                    [
                        apsw.SQLITE_OPEN_WAL | apsw.SQLITE_OPEN_CREATE | apsw.SQLITE_OPEN_READWRITE,
                        0,
                    ],
                )
            for role in (apsw.SQLITE_OPEN_TEMP_DB, apsw.SQLITE_OPEN_MAIN_JOURNAL):
                with pytest.raises(apsw.ReadOnlyError):
                    vfs.xOpen(main + "-journal", [role | apsw.SQLITE_OPEN_READWRITE, 0])
        assert inventory(database) == before
        assert not vfs.xAccess(main, apsw.SQLITE_ACCESS_EXISTS)
    finally:
        guard.close()


def test_callbacks_survive_gc_and_concurrent_readers_without_source_writes(tmp_path):
    database = tmp_path / "shared.db"
    seed(database).close()
    before = inventory(database)

    def native_read(_):
        reader = OpenCodeReader()
        with reader.snapshot(database) as connection:
            gc.collect()
            assert isinstance(connection, apsw.Connection)
            return connection.execute("SELECT id FROM part").fetchall()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(native_read, range(8)))
    assert results == [[{"id": "part"}]] * 8
    assert inventory(database) == before


def test_child_process_registers_its_own_vfs_without_inheriting_connections(tmp_path):
    database = tmp_path / "fork.db"
    seed(database).close()
    before = inventory(database)
    parent_name = readonly_vfs().name
    receiving, sending = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(receiving)
        try:
            name = readonly_vfs().name
            result = OpenCodeReader().read(OpenCodeReader().source_for(database))
            report = {"name": name, "result": "result:initial" in result.evidence_ids()}
            os.write(sending, json.dumps(report).encode())
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(sending)
    try:
        payload = os.read(receiving, 4096)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        report = json.loads(payload)
        assert report["name"] != parent_name and report["result"]
        assert inventory(database) == before
    finally:
        os.close(receiving)


def test_active_wal_fd_cannot_be_written_or_truncated(tmp_path):
    database = tmp_path / "active-wal.db"
    writer = seed(database)
    guard = SourceGuard(database)
    vfs = readonly_vfs()
    before = inventory(database)
    wal = tmp_path / "active-wal.db-wal"
    before_meta = (wal.stat().st_size, wal.stat().st_mtime_ns, wal.stat().st_ctime_ns)
    try:
        with vfs.bind(guard) as main:
            flags = [
                apsw.SQLITE_OPEN_WAL
                | apsw.SQLITE_OPEN_READWRITE
                | apsw.SQLITE_OPEN_CREATE
                | apsw.SQLITE_OPEN_DELETEONCLOSE,
                0,
            ]
            handle = vfs.xOpen(main + "-wal", flags)
            try:
                with pytest.raises(apsw.Error):
                    handle.xWrite(b"unsafe", 0)
                with pytest.raises(apsw.Error):
                    handle.xTruncate(0)
            finally:
                handle.xClose()
        assert inventory(database) == before
        assert (wal.stat().st_size, wal.stat().st_mtime_ns, wal.stat().st_ctime_ns) == before_meta
    finally:
        guard.close()
        writer.close()
