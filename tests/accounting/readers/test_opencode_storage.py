from __future__ import annotations

import gc
import hashlib
import json
import os
import select
import signal
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import apsw
import pytest

from ssgrep.accounting.collect.collector import Progress
from ssgrep.accounting.collect.read import ClassifyingLedger, read_sources
from ssgrep.accounting.collect.steps import source_progress
from ssgrep.accounting.collect.store import SourceRecord
from ssgrep.accounting.readers import opencode_storage as native_vfs
from ssgrep.accounting.readers.base import EvidenceLedger, SourceState
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.opencode_storage import (
    DescriptorFile,
    SnapshotChanged,
    SourceGuard,
    readonly_vfs,
)
from tests.accounting.readers.readers_fixtures import (
    T0,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)

"""OpenCode storage and VFS regression coverage."""
"""Logical source coverage is separate from settlement and publication."""


def test_source_progress_does_not_turn_settlement_into_completion():
    records = [
        SourceRecord("opencode", "opencode:unvisited"),
        SourceRecord(
            "opencode",
            "opencode:pending",
            state={
                "extra": {
                    "opencode": {
                        "status": "pending",
                        "rows_visited": {"session": 1, "part": 3},
                        "pending_records": None,
                        "known_pending_lower_bound": 3,
                        "pending_count_complete": False,
                        "abandoned_records": 1,
                    }
                }
            },
        ),
        SourceRecord(
            "opencode",
            "opencode:unsupported",
            status="unsupported",
            state={
                "extra": {"opencode": {"status": "unsupported"}},
            },
        ),
        SourceRecord("claude", "claude:done", complete=True),
    ]
    counts = source_progress(records)
    assert counts == {
        "sources_complete": 1,
        "sources_unvisited": 1,
        "sources_pending": 1,
        "rows_visited": 4,
        "pending_records": None,
        "abandoned_records": 1,
        "unsupported_sources": 1,
        "known_pending_lower_bound": 3,
        "pending_count_complete": False,
    }
    document = Progress("isolated", "collect", sources_done=4, **counts).to_dict()
    assert document["sources_done"] == 4
    assert document["sources_complete"] == 1


"""A native snapshot never creates or modifies SQLite main/WAL/SHM bytes."""


def inventory(database):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm"))
        if p.exists()
    }


def seed(database):
    writer = opencode_connect(database)
    opencode_add_session(writer, "owner")
    opencode_add_message(writer, "message", "owner", opencode_assistant_data())
    opencode_add_part(writer, "part", "message", "owner", opencode_tool_part("initial", "bash", {}))
    return writer


def test_closed_wal_mode_database_keeps_exact_sidecar_inventory(tmp_path):
    database = tmp_path / "closed.db"
    seed(database).close()
    before = inventory(database)
    assert set(before) == {"closed.db"}
    reader = OpenCodeReader()
    result = reader.read(reader.source_for(database))
    assert "result:initial" in result.evidence_ids()
    assert inventory(database) == before


def test_active_committed_wal_and_writer_checkpoint_are_read_without_sidecar_writes(tmp_path):
    database = tmp_path / "active.db"
    writer = seed(database)
    reader = OpenCodeReader()
    try:
        assert Path(str(database) + "-wal").stat().st_size > 0
        before = inventory(database)
        first = reader.read(reader.source_for(database))
        assert "result:initial" in first.evidence_ids()
        assert inventory(database) == before  # Includes actual SHM bytes, not only main/WAL.
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpointed = inventory(database)
        with reader.snapshot(database) as connection:
            assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
        assert inventory(database) == checkpointed
        opencode_add_part(
            writer, "later", "message", "owner", opencode_tool_part("later", "bash", {}), at=T0 + 10
        )
        committed = inventory(database)
        later = reader.read(reader.source_for(database), first.state)
        assert "result:later" in later.evidence_ids()
        assert inventory(database) == committed
    finally:
        writer.close()


def test_concurrent_checkpoint_rejects_snapshot_without_advancing_cursor(tmp_path, monkeypatch):
    database = tmp_path / "changed.db"
    writer = seed(database)
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        first = reader.read(source)
        snapshot = reader.snapshot

        @contextmanager
        def checkpoint_after_pin(path, *, descriptor=None, tracker=None):
            with snapshot(path, descriptor=descriptor, tracker=tracker) as connection:
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                yield connection

        with monkeypatch.context() as patch:
            patch.setattr(reader, "snapshot", checkpoint_after_pin)
            changed = reader.read(source, first.state)
        assert changed.evidence == []
        assert changed.state.extra["wm"] == first.state.extra["wm"]
        assert any(g.reason == "database_changed" for g in changed.gaps)
        assert changed.state.extra["opencode"]["status"] == "changed"
        retry = reader.read(source, changed.state)
        assert not any(g.reason == "database_changed" for g in retry.gaps)
    finally:
        writer.close()


def test_new_wal_after_inactive_guard_is_typed_changed_not_silent_main_only_read(tmp_path):
    database = tmp_path / "activation.db"
    seed(database).close()
    reader = OpenCodeReader()
    writer = None
    try:
        with pytest.raises(SnapshotChanged):
            with reader.snapshot(database) as connection:
                writer = opencode_connect(database)
                opencode_add_part(
                    writer,
                    "new-wal",
                    "message",
                    "owner",
                    opencode_tool_part("new-wal", "bash", {}),
                    at=T0 + 10,
                )
                assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
        current = reader.read(reader.source_for(database))
        assert "result:new-wal" in current.evidence_ids()
    finally:
        if writer is not None:
            writer.close()


def test_legitimate_wal_append_keeps_pinned_prior_commit_and_does_not_block_writer(tmp_path):
    database = tmp_path / "append.db"
    writer = seed(database)
    reader = OpenCodeReader()
    try:
        with reader.snapshot(database) as connection:
            assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
            opencode_add_part(
                writer,
                "later",
                "message",
                "owner",
                opencode_tool_part("later", "bash", {}),
                at=T0 + 10,
            )
            assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
        # Append changes WAL/SHM, but leaves already committed frames and main intact.
        current = reader.read(reader.source_for(database))
        assert "result:later" in current.evidence_ids()
    finally:
        writer.close()


def test_append_inside_reused_wal_capacity_is_not_mistaken_for_checkpoint(tmp_path):
    database = tmp_path / "reuse.db"
    writer = seed(database)
    reader = OpenCodeReader()
    try:
        # Build spare physical WAL capacity, then start a new salt generation.
        writer.execute(
            "UPDATE part SET data = ?", ('{"type":"text","text":"' + "x" * 65536 + '"}',)
        )
        writer.execute("PRAGMA wal_checkpoint(RESTART)")
        writer.execute("UPDATE part SET data = ?", ('{"type":"text","text":"current generation"}',))
        capacity = Path(str(database) + "-wal").stat().st_size
        with reader.snapshot(database) as connection:
            assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
            opencode_add_part(
                writer,
                "later",
                "message",
                "owner",
                opencode_tool_part("later", "bash", {}),
                at=T0 + 10,
            )
            assert Path(str(database) + "-wal").stat().st_size == capacity
            assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
        current = reader.read(reader.source_for(database))
        assert "result:later" in current.evidence_ids()
    finally:
        writer.close()


def test_spilled_uncommitted_wal_preserves_all_native_files_and_reads_committed_main(tmp_path):
    database = tmp_path / "spilled.db"
    writer = seed(database)
    reader = OpenCodeReader()
    try:
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("PRAGMA cache_size=2")
        writer.execute("BEGIN")
        for index in range(40):
            opencode_add_part(
                writer,
                f"pending-{index}",
                "message",
                "owner",
                opencode_tool_part(f"pending-{index}", "bash", {}, output="x" * 4096),
                at=T0 + index + 1,
            )
        assert Path(str(database) + "-wal").stat().st_size > 32
        before = inventory(database)
        result = reader.read(reader.source_for(database))
        assert "result:initial" in result.evidence_ids()
        assert not any(e.record_id.startswith("result:pending-") for e in result.evidence)
        assert inventory(database) == before
        writer.execute("COMMIT")
        committed = reader.read(reader.source_for(database), result.state)
        assert {f"result:pending-{i}" for i in range(40)} <= set(committed.evidence_ids())
    finally:
        if writer.in_transaction:
            writer.execute("ROLLBACK")
        writer.close()


def test_committed_wal_without_shared_index_is_read_without_recreating_sidecars(tmp_path):
    active = tmp_path / "active-source.db"
    writer = seed(active)
    database = tmp_path / "no-shared-index.db"
    try:
        database.write_bytes(active.read_bytes())
        Path(str(database) + "-wal").write_bytes(Path(str(active) + "-wal").read_bytes())
        before = inventory(database)
        reader = OpenCodeReader()
        result = reader.read(reader.source_for(database))
        assert "result:initial" in result.evidence_ids()
        assert result.gaps == []
        assert inventory(database) == before
    finally:
        writer.close()


def test_checkpoint_before_sqlite_pin_never_deletes_empty_native_wal(tmp_path, monkeypatch):
    import apsw

    database = tmp_path / "before-pin.db"
    writer = seed(database)
    reader = OpenCodeReader()
    connect = apsw.Connection

    def checkpoint_before_open(*args, **kwargs):
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return connect(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(apsw, "Connection", checkpoint_before_open)
            changed = reader.read(reader.source_for(database))
        assert changed.evidence == []
        assert any(g.reason == "database_changed" for g in changed.gaps)
        assert Path(str(database) + "-wal").exists()
        assert Path(str(database) + "-wal").stat().st_size == 0
        before = inventory(database)
        assert "result:initial" in reader.read(reader.source_for(database)).evidence_ids()
        assert inventory(database) == before
    finally:
        writer.close()


def test_snapshot_exposes_native_apsw_rows_and_progress_errors(tmp_path):
    import apsw

    database = tmp_path / "native-api.db"
    seed(database).close()
    reader = OpenCodeReader()
    with reader.snapshot(database) as connection:
        assert isinstance(connection, apsw.Connection)
        assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
        assert not connection.get_autocommit()
        connection.set_progress_handler(lambda: True, 1)
        with pytest.raises(apsw.InterruptError):
            connection.execute("SELECT id FROM part").fetchall()


def test_writer_source_close_rejects_changed_view_without_recreating_sidecars(tmp_path):
    database = tmp_path / "writer-close.db"
    writer = seed(database)
    reader = OpenCodeReader()
    after_writer = None
    try:
        with pytest.raises(SnapshotChanged):
            with reader.snapshot(database) as connection:
                assert connection.execute("SELECT id FROM part").fetchall() == [{"id": "part"}]
                writer.close()
                after_writer = inventory(database)
        assert set(after_writer) == {"writer-close.db"}
        assert inventory(database) == after_writer
    finally:
        writer.close()


def test_corrupt_wal_cannot_be_cleaned_up_by_readonly_recovery(tmp_path):
    active = tmp_path / "valid.db"
    writer = seed(active)
    database = tmp_path / "corrupt.db"
    try:
        database.write_bytes(active.read_bytes())
        raw = bytearray(Path(str(active) + "-wal").read_bytes())
        raw[32 + 24 + 64] ^= 1  # Corrupt a frame payload, not the WAL header/salts.
        Path(str(database) + "-wal").write_bytes(raw)
        before = inventory(database)
        reader = OpenCodeReader()
        result = reader.read(reader.source_for(database))
        assert not result.evidence
        assert result.gaps
        assert inventory(database) == before
    finally:
        writer.close()


"""Public APSW descriptor I/O: real bytes, partial/EOF semantics and ownership."""


def open_file(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return DescriptorFile(descriptor)
    finally:
        os.close(descriptor)


def test_owned_dup_reads_after_approval_fd_closes_and_never_pads_eof(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"abcdef")
    file = open_file(source)
    assert file.xRead(3, 2) == b"cde"
    assert file.xRead(5, 4) == b"ef"
    assert file.xRead(5, 9) == b""
    assert file.xFileSize() == 6
    file.xClose()
    file.xClose()
    with pytest.raises(apsw.IOError):
        file.xRead(1, 0)


def test_partial_and_interrupted_pread_complete_real_bytes_without_fake_padding(
    tmp_path, monkeypatch
):
    source = tmp_path / "partial"
    source.write_bytes(b"abcdefgh")
    file = open_file(source)
    pread = os.pread
    interrupted = False

    def short_read(fd, amount, offset):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise InterruptedError()
        return pread(fd, min(amount, 2), offset)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "pread", short_read)
            assert file.xRead(8, 0) == b"abcdefgh"
            assert file.xRead(8, 5) == b"fgh"
        assert interrupted
    finally:
        file.xClose()


def test_descriptor_reads_validate_ranges_and_duplicate_only_readonly_fds(tmp_path):
    source = tmp_path / "range-check"
    source.write_bytes(b"native")
    descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    file = DescriptorFile(descriptor)
    os.close(descriptor)
    try:
        assert file.xRead(0, 0) == b""
        assert file.xRead(6, 0) == b"native"
        for amount, offset in ((-1, 0), (1, -1)):
            with pytest.raises(apsw.IOError):
                file.xRead(amount, offset)
    finally:
        file.xClose()

    closed = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    os.close(closed)
    with pytest.raises(apsw.Error):
        DescriptorFile(closed)


def test_descriptor_io_failures_are_reported_as_native_io_errors(tmp_path, monkeypatch):
    source = tmp_path / "io-errors"
    source.write_bytes(b"stable source")
    file = open_file(source)
    try:
        assert file.xRead(6, 0) == b"stable"
        with monkeypatch.context() as patch:
            patch.setattr(os, "pread", lambda *_: (_ for _ in ()).throw(PermissionError()))
            with pytest.raises(apsw.IOError):
                file.xRead(1, 0)
        with monkeypatch.context() as patch:
            patch.setattr(os, "fstat", lambda *_: (_ for _ in ()).throw(PermissionError()))
            with pytest.raises(apsw.IOError):
                file.xFileSize()
    finally:
        file.xClose()


def test_descriptor_close_os_failure_is_a_typed_error_and_releases_handle_state(
    tmp_path, monkeypatch
):
    source = tmp_path / "close-error"
    source.write_bytes(b"kept bytes")
    file = open_file(source)
    duplicate = file._fd
    real_close = os.close
    assert file.xRead(4, 0) == b"kept"
    try:

        def fail_duplicate_close(fd):
            if fd == duplicate:
                raise OSError("simulated close failure")
            real_close(fd)

        with monkeypatch.context() as patch:
            patch.setattr(os, "close", fail_duplicate_close)
            with pytest.raises(apsw.IOError):
                file.xClose()
        assert file._fd is None
    finally:
        real_close(duplicate)


def test_readonly_methods_never_modify_bytes_or_metadata_and_controls_are_conservative(tmp_path):
    source = tmp_path / "readonly"
    source.write_bytes(b"preserved native bytes")
    file = open_file(source)
    before = source.stat()
    try:
        with pytest.raises(apsw.ReadOnlyError):
            file.xWrite(b"bad", 0)
        with pytest.raises(apsw.ReadOnlyError):
            file.xTruncate(0)
        file.xLock(apsw.SQLITE_LOCK_SHARED)
        file.xUnlock(apsw.SQLITE_LOCK_NONE)
        file.xSync(apsw.SQLITE_SYNC_NORMAL)
        assert file.xCheckReservedLock() is False
        assert file.xFileControl(1234, 0xDEADBEEF) is False
        assert file.xDeviceCharacteristics() == 0
        assert file.xSectorSize() == 512
        assert source.read_bytes() == b"preserved native bytes"
        after = source.stat()
        assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
    finally:
        file.xClose()
    writable = os.open(source, os.O_RDWR)
    try:
        with pytest.raises(apsw.ReadOnlyError):
            DescriptorFile(writable)
    finally:
        os.close(writable)


def test_inherited_file_rejects_reads_but_child_can_close_its_duplicate(tmp_path):
    source = tmp_path / "fork"
    source.write_bytes(b"parent data")
    file = open_file(source)
    receiving, sending = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(receiving)
        try:
            try:
                file.xRead(1, 0)
            except apsw.IOError:
                file.xClose()
                os.write(sending, b"rejected")
                os._exit(0)
            os._exit(1)
        except BaseException:
            os._exit(2)
    os.close(sending)
    reaped = False
    try:
        assert select.select([receiving], [], [], 2)[0], "descriptor child exceeded watchdog"
        assert os.read(receiving, 64) == b"rejected"
        _, status = os.waitpid(child, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
        assert file.xRead(11, 0) == b"parent data"
    finally:
        os.close(receiving)
        if not reaped:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(child, 0)
        file.xClose()


"""Readonly callbacks belong to the actual maintained driver and outlive readers."""


def test_native_file_open_and_delete_are_readonly_even_when_requested_writable(tmp_path):
    database = tmp_path / "native.db"
    seed(database).close()
    vfs = readonly_vfs()
    before = inventory(database)
    guard = SourceGuard(database)
    try:
        with vfs.bind(guard) as main:
            assert vfs.xFullPathname(main) == main
            assert vfs.xAccess(main, apsw.SQLITE_ACCESS_EXISTS)
            assert not vfs.xAccess(main, apsw.SQLITE_ACCESS_READWRITE)
            assert not vfs.xAccess("/unbound/main.db", apsw.SQLITE_ACCESS_EXISTS)
            with pytest.raises(apsw.CantOpenError):
                vfs.xFullPathname("/unbound/main.db")
            with pytest.raises(apsw.CantOpenError):
                vfs.xOpen(main, [apsw.SQLITE_OPEN_WAL | apsw.SQLITE_OPEN_READONLY, 0])
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
            with pytest.raises(apsw.CantOpenError):
                vfs.xOpen(main + "-wal", [apsw.SQLITE_OPEN_MAIN_DB | apsw.SQLITE_OPEN_READONLY, 0])
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


"""Independent row-count acceptance for bounded, fair SQLite/WAL collection."""


def store(path: Path, parts: int = 1, pending: int = 0):
    writer = opencode_connect(path)
    writer.execute("BEGIN")
    opencode_add_session(writer, "owner")
    opencode_add_message(writer, "message", "owner", opencode_assistant_data())
    for index in range(parts):
        # Deterministic realistic skew: mostly small, occasional large output.
        output = "x" * (1048576 if index % 37 == 0 else 512)
        data = opencode_tool_part(
            f"call-{index}",
            "bash",
            {"command": "ssgrep search alpha"},
            status="running" if index < pending else "completed",
            output=output,
        )
        opencode_add_part(writer, f"part-{index:05}", "message", "owner", data, at=T0 + index)
    writer.execute("COMMIT")
    return writer


def collect(discovered, known, ledger, budget=1, baseline=False):
    return read_sources(
        discovered=discovered,
        known=known,
        ledger=ledger,
        admit=lambda _: None,
        byte_budget=budget,
        time_budget=100,
        clock=lambda: 0,
        baseline_new=baseline,
    )


def test_large_wal_store_eventually_visits_independently_counted_rows(tmp_path):
    path = tmp_path / "large.db"
    # Expected counts are construction inputs, not inferred from reader output.
    parts, pending = 1203, 503
    writer = store(path, parts, pending)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    state, calls, results, pending_ids = None, set(), set(), set()
    try:
        assert Path(f"{path}-wal").stat().st_size > 0
        for _ in range(parts * 8 + 20):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = SourceState.from_dict(json.loads(json.dumps(step.state.to_dict())))
            calls.update(e.record_id for e in step.evidence if e.kind == "tool_call")
            results.update(e.record_id for e in step.evidence if e.kind == "tool_result")
            pending_ids.update(e.payload["id"] for e in step.evidence if e.kind == "pending_tool")
            if not step.more:
                break
        else:
            raise AssertionError("fresh rows starved behind unfinished rows")
        progress = state.extra["opencode"]
        assert all(
            progress["rows_visited"][table] >= n
            for table, n in {"session": 1, "message": 1, "part": parts}.items()
        )
        assert calls == {f"call:call-{i}" for i in range(pending, parts)}
        assert len(results) == parts - pending
        assert pending_ids == {f"part-{i:05}" for i in range(pending)}
        assert 1 <= step.pending_records <= 32
        assert len(state.extra["pending"]["part"]) <= 32
        assert progress["pending_records"] is None
        assert not progress["pending_count_complete"]
        assert len(json.dumps(state.to_dict())) < 65536
        assert progress["status"] == "pending"
        assert progress["bytes_read"] > 0
        # Complete the LAST formerly capped pending row without advancing its time.
        done = opencode_tool_part("call-502", "bash", {}, output="late completion")
        writer.execute("UPDATE part SET data=? WHERE id='part-00502'", (json.dumps(done),))
        found = False
        for _ in range(parts * 8 + 20):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            if "result:call-502" in step.evidence_ids():
                found = True
                break
        assert found, "bounded rechecks must reach every retained pending id"
        assert step.pending_records <= 32
    finally:
        writer.close()


def test_source_fairness_and_zero_budget_never_completes_unvisited_database(tmp_path):
    paths = [tmp_path / "first.db", tmp_path / "second.db"]
    writers = [store(p, parts=20) for p in paths]
    reader = OpenCodeReader()
    discovered = {str(i): (reader, reader.source_for(p)) for i, p in enumerate(paths)}
    ledger = ClassifyingLedger()
    try:
        zero = collect(discovered, {}, ledger, budget=0, baseline=True)
        assert all(not r.complete for r in zero.sources.values())
        assert all(
            r.state["extra"]["opencode"]["status"] == "unvisited" for r in zero.sources.values()
        )
        first = collect(discovered, zero.sources, ledger)
        second = collect(discovered, first.sources, ledger)
        assert all(r.state["extra"]["opencode"]["rows_visited"] for r in second.sources.values())
        assert all(not r.complete for r in second.sources.values())
    finally:
        for writer in writers:
            writer.close()


def test_failure_details_survive_a_successful_source(tmp_path):
    good, bad = tmp_path / "good.db", tmp_path / "bad.db"
    writer = store(good)
    bad.write_bytes(b"not SQLite")
    reader = OpenCodeReader()
    discovered = {
        "bad": (reader, reader.source_for(bad)),
        "good": (reader, reader.source_for(good)),
    }
    try:
        outcome = collect(discovered, {}, ClassifyingLedger(), budget=16 * 1024 * 1024)
        assert outcome.sources["good"].complete
        assert not outcome.sources["bad"].complete
        assert outcome.sources["bad"].gaps["database_unavailable"] == 1
        assert outcome.gaps["database_unavailable"] == 1
        assert outcome.sources["bad"].state["extra"]["opencode"]["status"] == "unavailable"
    finally:
        writer.close()


def test_deleted_pending_and_unknown_terminal_status_keep_provenance_without_success(tmp_path):
    path = tmp_path / "pending.db"
    writer = store(path, parts=3, pending=1)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    try:
        future = opencode_tool_part("future", "bash", {}, status="cancelled")
        opencode_add_part(writer, "future", "message", "owner", future, at=T0 + 10)
        initial = reader.read(source)
        assert "result:future" not in initial.evidence_ids()
        assert initial.state.extra["unsupported"]["part:future"] == "unsupported_tool_status"
        writer.execute("DELETE FROM part WHERE id='part-00000'")
        after = reader.read(source, initial.state)
        abandoned = after.state.extra["abandoned"]["part:part-00000"]
        assert abandoned["call_id"] == "call-0"
        assert abandoned["session_id"] == "owner"
        assert abandoned["reason"] == "pending_row_missing"
        assert after.state.extra["opencode"]["abandoned_records"] == 1
        assert not any(e.record_id == "result:call-0" for e in after.evidence)
        assert after.state.extra["opencode"]["status"] != "complete"
    finally:
        writer.close()


def test_concurrent_append_is_deferred_until_next_committed_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "concurrent.db"
    writer = store(path)
    reader = OpenCodeReader()
    snapshot = reader.snapshot
    source = reader.source_for(path)

    @contextmanager
    def append_after_pin(database, *, descriptor=None, tracker=None):
        with snapshot(database, descriptor=descriptor, tracker=tracker) as connection:
            opencode_add_part(
                writer,
                "later",
                "message",
                "owner",
                opencode_tool_part("later", "bash", {}),
                at=T0 + 20,
            )
            yield connection

    try:
        with monkeypatch.context() as patch:
            patch.setattr(reader, "snapshot", append_after_pin)
            before = reader.read(source)
        assert "call:later" not in before.evidence_ids()
        assert before.state.extra["opencode"]["rows_visited"]["part"] == 1
        after = reader.read(source, before.state)
        assert "call:later" in after.evidence_ids()
        assert after.state.extra["opencode"]["rows_visited"]["part"] == 2
    finally:
        writer.close()


def test_backdated_committed_insert_behind_timestamp_keyset_is_eventually_visited(tmp_path):
    path = tmp_path / "backdated.db"
    writer = store(path)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    try:
        initial = reader.read(source)
        opencode_add_part(
            writer,
            "earlier-id",
            "message",
            "owner",
            opencode_tool_part("backdated", "bash", {}),
            at=T0 - 100,
        )
        state = initial.state
        seen = set()
        for _ in range(40):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            seen.update(step.evidence_ids())
            if "result:backdated" in seen:
                break
        assert "call:backdated" in seen and "result:backdated" in seen
        assert state.extra["opencode"]["rows_visited"]["part"] >= 2
    finally:
        writer.close()


def test_terminal_observations_clear_pending_without_native_timestamp_changes(tmp_path):
    path = tmp_path / "lifecycle.db"
    writer = store(path, parts=35, pending=35)
    reader = OpenCodeReader()
    source = reader.source_for(path)
    ledger = EvidenceLedger()
    try:
        initial = reader.read(source, maintenance=True)
        for evidence in initial.evidence:
            ledger.admit(evidence)
        for index in range(35):
            writer.execute(
                "UPDATE part SET data=? WHERE id=?",
                (
                    json.dumps(opencode_tool_part(f"call-{index}", "bash", {}, output="done")),
                    f"part-{index:05}",
                ),
            )
        state, results = initial.state, set()
        for _ in range(500):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            for evidence in step.evidence:
                if ledger.admit(evidence) and evidence.kind == "tool_result":
                    results.add(evidence.record_id)
            if state.extra["opencode"]["status"] == "complete":
                break
        assert results == {f"result:call-{i}" for i in range(35)}
        assert state.extra["opencode"]["pending_records"] == 0
        assert ledger.conflicts() == []
    finally:
        writer.close()


def test_native_bytes_include_ignored_payload_not_database_size_or_emitted_evidence(tmp_path):
    path = tmp_path / "bytes.db"
    writer = store(path, parts=0)
    ignored = {"type": "text", "text": "private" * 65536}
    expected_payload_bytes = len(json.dumps(ignored).encode())
    opencode_add_part(writer, "ignored", "message", "owner", ignored, at=T0 + 1)
    reader = OpenCodeReader()
    try:
        step = reader.read(reader.source_for(path))
        assert not any(e.kind in {"tool_call", "tool_result"} for e in step.evidence)
        assert step.bytes_read >= expected_payload_bytes
        assert step.state.extra["opencode"]["bytes_read"] == step.bytes_read
        assert step.state.extra["opencode"]["rows_visited"]["part"] == 1
        assert step.state.offset == 0  # No fabricated SQLite file-byte cursor.
    finally:
        writer.close()


def test_database_baseline_is_unvisited_not_a_skipped_complete_high_watermark(tmp_path):
    from ssgrep.accounting.collect.read import baseline_record

    database = tmp_path / "baseline.db"
    writer = store(database)
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        baseline = baseline_record(reader, source)
        assert not baseline.complete
        assert baseline.state["extra"]["opencode"]["status"] == "unvisited"
        assert baseline.state["extra"]["opencode"]["rows_visited"] == {}
        assert "wm" not in baseline.state["extra"]
        assert baseline.offset == baseline.skipped_bytes == 0
        outcome = collect(
            {"native": (reader, source)},
            {"native": baseline},
            ClassifyingLedger(),
            budget=16 * 1024 * 1024,
        )
        assert outcome.sources["native"].complete
        (result,) = [item for item in outcome.evidence if item.kind == "tool_result"]
        assert result.payload["call_id"] == "call-0"
        assert result.payload["output"] == "x" * 1048576
    finally:
        writer.close()


"""Portable continuity checks for OpenCode's persisted SQLite cursors."""


def reader_for(path: Path, monkeypatch: pytest.MonkeyPatch) -> OpenCodeReader:
    reader = OpenCodeReader()
    monkeypatch.setattr(reader, "_identity", lambda _: {"dev": 1, "inode": 2, "birth": None})
    return reader


def add_row(connection: sqlite3.Connection, table: str, record_id: str, at: int) -> None:
    if table == "session":
        opencode_add_session(connection, record_id)
        connection.execute("UPDATE session SET time_updated=? WHERE id=?", (at, record_id))
    elif table == "message":
        opencode_add_message(connection, record_id, "owner", opencode_assistant_data(), at=at)
    else:
        opencode_add_part(
            connection,
            record_id,
            "message",
            "owner",
            opencode_tool_part(record_id, "bash", {}),
            at=at,
        )


def continuity_seed(path: Path) -> sqlite3.Connection:
    connection = opencode_connect(path)
    add_row(connection, "session", "owner", T0 - 3)
    for table in ("session", "message", "part"):
        add_row(connection, table, table, T0)
    return connection


@pytest.mark.parametrize("table", ("session", "message", "part"))
@pytest.mark.parametrize("change", ("missing", "regressed"))
def test_lost_watermark_anchor_replays_even_with_a_newer_table_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: str, change: str
) -> None:
    path = tmp_path / "opencode.db"
    writer = continuity_seed(path)
    try:
        reader = reader_for(path, monkeypatch)
        source = reader.source_for(path)
        first = reader.read(source)
        # Exercise the persisted JSON representation, not an in-memory-only marker.
        state = SourceState.from_dict(json.loads(json.dumps(first.state.to_dict())))
        if change == "missing":
            writer.execute(f"DELETE FROM {table} WHERE id=?", (table,))
        else:
            writer.execute(f"UPDATE {table} SET time_updated=? WHERE id=?", (T0 - 1, table))
        add_row(writer, table, "replacement", T0 - 2)
        add_row(writer, table, "newer", T0 + 1)

        replay = reader.read(source, state, budget=1)
        assert replay.reset == "rotated"
        seen = list(replay.evidence)
        while replay.more:
            replay = reader.read(source, replay.state, budget=1)
            assert replay.reset is None
            seen.extend(replay.evidence)
        record_ids = [item.record_id for item in seen]
        prefix = {"session": "session", "message": "usage", "part": "call"}[table]
        assert f"{prefix}:replacement" in record_ids
        assert f"{prefix}:newer" in record_ids
        if change == "regressed":
            assert f"{prefix}:{table}" in record_ids
        assert reader.read(source, replay.state).evidence == []

        ledger = EvidenceLedger()
        for item in first.evidence:
            assert ledger.admit(item)
        admitted = [item.record_id for item in seen if ledger.admit(item)]
        expected = [f"{prefix}:replacement", f"{prefix}:newer"]
        if table == "part":
            expected += ["result:replacement", "result:newer"]
        assert sorted(admitted) == sorted(expected)
        assert ledger.duplicates > 0 and ledger.conflicts() == []
    finally:
        writer.close()


def test_committed_anchor_updates_and_wal_additions_do_not_rotate_or_replay_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "opencode.db"
    writer = continuity_seed(path)
    try:
        reader = reader_for(path, monkeypatch)
        source = reader.source_for(path)
        first = reader.read(source)
        checkpoint = json.dumps(first.state.to_dict(), sort_keys=True)
        updated = opencode_tool_part("part", "bash", {}, output="changed committed output")
        writer.execute(
            "UPDATE part SET time_updated=?, data=? WHERE id='part'",
            (T0 + 1, json.dumps(updated)),
        )
        add_row(writer, "part", "newer", T0 + 2)
        after = reader.read(source, first.state)
        assert after.reset is None
        assert after.evidence_ids() == ["call:part", "result:part", "call:newer", "result:newer"]
        assert after.evidence[1].payload["output"] == "changed committed output"
        assert json.dumps(first.state.to_dict(), sort_keys=True) == checkpoint
        # A checkpoint changes the file/WAL metadata, but not logical continuity.
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        unchanged = reader.read(source, after.state)
        assert unchanged.reset is None and unchanged.evidence == []
    finally:
        writer.close()


def test_anchor_check_and_rows_share_the_same_committed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "opencode.db"
    writer = continuity_seed(path)
    try:
        reader = reader_for(path, monkeypatch)
        source = reader.source_for(path)
        first = reader.read(source)
        snapshot = reader.snapshot

        @contextmanager
        def replace_after_snapshot_is_pinned(database: Path, *, descriptor=None, tracker=None):
            with snapshot(database, descriptor=descriptor, tracker=tracker) as connection:
                writer.execute("DELETE FROM part")
                add_row(writer, "part", "replacement", T0 - 1)
                yield connection

        with monkeypatch.context() as patch:
            patch.setattr(reader, "snapshot", replace_after_snapshot_is_pinned)
            pinned = reader.read(source, first.state)
        assert pinned.reset is None and pinned.evidence == []
        replacement = reader.read(source, pinned.state)
        assert replacement.reset == "rotated"
        assert "call:replacement" in replacement.evidence_ids()
        assert "call:part" not in replacement.evidence_ids()
    finally:
        writer.close()


def test_empty_tables_and_zero_or_negative_watermark_boundaries_are_not_sentinels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "opencode.db"
    writer = opencode_connect(path)
    try:
        reader = reader_for(path, monkeypatch)
        source = reader.source_for(path)
        empty = reader.read(source)
        assert empty.evidence == []
        add_row(writer, "session", "owner", -1)
        add_row(writer, "message", "", 0)
        add_row(writer, "part", "", 0)
        first = reader.read(source, empty.state, budget=1)
        assert first.reset is None
        assert first.evidence_ids() == ["session:owner", "usage:", "call:", "result:"]
        assert first.state.extra["wm"]["session"] == [-1, "owner"]
        assert first.state.extra["wm"]["part"] == [0, ""]
        assert reader.read(source, first.state).evidence == []
        writer.execute("DELETE FROM part WHERE id=''")
        add_row(writer, "part", "replacement", -1)
        replaced = reader.read(source, first.state)
        assert replaced.reset == "rotated"
        assert "call:replacement" in replaced.evidence_ids()
    finally:
        writer.close()


def test_changed_filesystem_identity_replays_replacement_with_preserved_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "opencode.db"
    continuity_seed(path).close()
    reader = reader_for(path, monkeypatch)
    source = reader.source_for(path)
    first = reader.read(source)
    path.unlink()
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    replacement = continuity_seed(path)
    try:
        replacement.execute(
            "UPDATE part SET data=? WHERE id=?",
            (
                json.dumps(opencode_tool_part("part", "bash", {}, output="replacement payload")),
                "part",
            ),
        )
    finally:
        replacement.close()
    monkeypatch.setattr(reader, "_identity", lambda _: {"dev": 1, "inode": 3, "birth": None})
    replay = reader.read(source, first.state)
    assert replay.reset == "rotated"
    output = next(item for item in replay.evidence if item.record_id == "result:part")
    assert output.payload["output"] == "replacement payload"
    assert reader.read(source, replay.state).evidence == []


"""Independent regressions for descriptor authority, dirty sweeps and JSON depth."""


def review_seed(path, call):
    writer = opencode_connect(path)
    opencode_add_session(writer, "owner")
    opencode_add_message(writer, "message", "owner", opencode_assistant_data())
    opencode_add_part(writer, "m", "message", "owner", opencode_tool_part(call, "bash", {}))
    return writer


@pytest.mark.parametrize("kind", [apsw.SQLITE_OPEN_MAIN_DB, apsw.SQLITE_OPEN_WAL])
def test_ancestor_aba_during_native_open_cannot_substitute_an_unapproved_database(
    tmp_path, monkeypatch, kind
):
    approved, outside, parked = tmp_path / "approved", tmp_path / "outside", tmp_path / "parked"
    approved.mkdir()
    outside.mkdir()
    database = approved / "native.db"
    a = review_seed(database, "approved")
    b = review_seed(outside / "native.db", "unapproved")
    original = native_vfs.DescriptorFile
    target = (
        database if kind == apsw.SQLITE_OPEN_MAIN_DB else database.with_name(database.name + "-wal")
    )
    expected = target.stat()
    substitutions = []

    def swap_namespace(descriptor):
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            approved.rename(parked)
            outside.rename(approved)
            try:
                substitutions.append(kind)
                return original(descriptor)
            finally:
                approved.rename(outside)
                parked.rename(approved)
        return original(descriptor)

    reader = OpenCodeReader()
    try:
        before_a, before_b = inventory(database), inventory(outside / "native.db")
        with monkeypatch.context() as patch:
            patch.setattr(native_vfs, "DescriptorFile", swap_namespace)
            result = reader.read(reader.source_for(database))
        assert substitutions
        assert "result:unapproved" not in result.evidence_ids()
        assert "result:approved" in result.evidence_ids()
        assert inventory(database) == before_a
        assert inventory(outside / "native.db") == before_b
    finally:
        a.close()
        b.close()


def test_backdated_insert_behind_inflight_census_is_read_before_complete(tmp_path):
    database = tmp_path / "dirty-sweep.db"
    writer = review_seed(database, "original")
    opencode_add_part(
        writer, "z", "message", "owner", opencode_tool_part("last", "bash", {}), at=T0 + 10
    )
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        initial = reader.read(source)
        state = initial.state
        for _ in range(30):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            part = state.extra.get("census", {}).get("part", {})
            if part.get("id") == "m" and not part.get("done"):
                break
        else:
            raise AssertionError("fixture did not reach the independent census boundary")
        opencode_add_part(
            writer, "a", "message", "owner", opencode_tool_part("backdated", "bash", {}), at=T0 - 10
        )
        found, saw_dirty, completed = False, False, False
        for _ in range(80):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            found |= "result:backdated" in step.evidence_ids()
            saw_dirty |= state.extra.get("census_epoch", {}).get("dirty", False)
            if state.extra["opencode"]["status"] == "complete":
                assert found, "a dirty prefix must not become the collector's idle-skip proof"
                completed = True
                break
        assert found and saw_dirty and completed
    finally:
        writer.close()


def test_busy_commit_changes_do_not_restart_an_advanced_census_at_first_page(tmp_path):
    database = tmp_path / "busy.db"
    writer = review_seed(database, "original")
    for index in range(12):
        opencode_add_part(
            writer,
            f"part-{index:02}",
            "message",
            "owner",
            opencode_tool_part(f"call-{index}", "bash", {}),
            at=T0 + index + 1,
        )
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        state = reader.read(source).state
        boundaries = set()
        for index in range(90):
            writer.execute(
                "UPDATE session SET time_updated=? WHERE id='owner'", (T0 + 100 + index,)
            )
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            boundary = state.extra.get("census", {}).get("part", {}).get("id")
            if boundary:
                boundaries.add(boundary)
        assert "part-11" in boundaries
        assert state.extra["opencode"]["status"] != "complete"
    finally:
        writer.close()


@pytest.mark.parametrize("table", ["message", "part"])
def test_deep_subcap_json_is_a_row_local_gap_and_later_native_rows_are_read(tmp_path, table):
    database = tmp_path / "nested.db"
    writer = review_seed(database, "original")
    nested = "[" * 5000 + "0" + "]" * 5000
    try:
        if table == "message":
            writer.execute(
                "INSERT INTO message VALUES(?,?,?,?,?)", ("deep", "owner", T0 + 1, T0 + 1, nested)
            )
        else:
            opencode_add_part(writer, "deep", "message", "owner", nested, at=T0 + 1)
        opencode_add_part(
            writer, "later", "message", "owner", opencode_tool_part("later", "bash", {}), at=T0 + 2
        )
        reader = OpenCodeReader()
        result = reader.read(reader.source_for(database))
        assert "result:later" in result.evidence_ids()
        (gap,) = [g for g in result.gaps if g.locator == f"{table}:deep"]
        assert gap.size == len(nested.encode())
        assert not any(e.locator == gap.locator for e in result.evidence)
        assert result.bytes_read >= len(nested.encode())
    finally:
        writer.close()


def test_json_decodes_but_completed_tool_normalization_depth_is_contained(tmp_path):
    import json
    import sys

    def reference_walk(value):
        if isinstance(value, dict):
            return {key: reference_walk(item) for key, item in value.items()}
        return value

    raw = None
    # Select from the live recursion limit and independently exercise both
    # decodability and recursive container traversal; no fixed portable depth.
    for numerator in range(12, 40):
        depth = max(1, sys.getrecursionlimit() * numerator // 20)
        nested = '{"node":' * depth + "0" + "}" * depth
        candidate = (
            '{"type":"tool","tool":"bash","callID":"deep-normalization",'
            '"state":{"status":"completed","input":'
            + nested
            + ',"output":"done","time":{"start":1,"end":2}}}'
        )
        try:
            decoded = json.loads(candidate)
        except RecursionError:
            continue
        try:
            reference_walk(decoded["state"]["input"])
        except RecursionError:
            raw = candidate
            break
    assert raw is not None, "fixture needs decodable JSON with recursive traversal overflow"
    assert json.loads(raw)["state"]["status"] == "completed"

    database = tmp_path / "normalization.db"
    writer = review_seed(database, "original")
    try:
        opencode_add_part(writer, "deep-normalization", "message", "owner", raw, at=T0 + 1)
        opencode_add_part(
            writer,
            "later-normalization",
            "message",
            "owner",
            opencode_tool_part("later-normalization", "bash", {}),
            at=T0 + 2,
        )
        reader = OpenCodeReader()
        result = reader.read(reader.source_for(database))
        assert "result:later-normalization" in result.evidence_ids()
        assert not any(e.locator == "part:deep-normalization" for e in result.evidence)
        assert any(
            g.reason == "normalization_recursion" and g.locator == "part:deep-normalization"
            for g in result.gaps
        )
        assert result.bytes_read >= len(raw.encode())
        assert result.state.extra["wm"]["part"] == [T0 + 2, "later-normalization"]
    finally:
        writer.close()


def native_adapter_for_opencode(database: Path, approved_root: Path):
    from ssgrep.accounting.readers.native_source import (
        NativeSourceAdapter,
        NativeSourceReference,
    )
    from ssgrep.accounting.views.filter import Scoper
    from tests.accounting.ledger.reduce_builders import policy

    reference = NativeSourceReference(
        "opencode:owner:ep:0",
        "opencode",
        "owner",
        database,
        cwd="/work",
    )
    reader = OpenCodeReader()
    adapter = NativeSourceAdapter(
        {reference.ref: reference},
        {"opencode": reader},
        (approved_root,),
        Scoper(policy()),
        1_800_000_000_000,
    )
    return reader, reference, adapter


def test_native_opencode_record_lookup_is_exact_and_keeps_sqlite_source_unchanged(
    tmp_path: Path,
) -> None:
    database = tmp_path / "direct-source.db"
    writer = store(database, parts=1)
    reader, reference, adapter = native_adapter_for_opencode(database, tmp_path)
    before = inventory(database)
    try:
        source = reader.source_for(database)
        witness = next(item for item in reader.read(source).evidence if item.kind == "tool_result")
        control = adapter.read_record(
            reference,
            witness.native_identity,
            witness.digest,
            locator=witness.locator,
        )
        assert control.status.value == "ok"
        assert control.evidence is not None
        assert control.evidence.native_identity == witness.native_identity
        assert control.evidence.digest == witness.digest
        assert control.evidence.payload == witness.payload

        missing = adapter.read_record(
            reference,
            witness.native_identity,
            witness.digest,
            locator="part:absent",
        )
        assert missing.status.value == "source_missing" and missing.evidence is None
        mismatch = adapter.read_record(
            reference,
            witness.native_identity,
            witness.digest,
            locator="session:other-owner",
        )
        assert mismatch.status.value == "source_identity_mismatch"
        assert mismatch.evidence is None and mismatch.bytes_read > 0
        assert inventory(database) == before
        from dataclasses import replace

        from ssgrep.accounting.readers.native_source import NativeSourceStatus
        from ssgrep.accounting.views.filter import Scoper
        from tests.accounting.ledger.reduce_builders import policy

        writer.execute(
            "UPDATE session SET directory = ? WHERE id = ?", ("/private/foreign", "owner")
        )
        writer.commit()
        after_owner_change = inventory(database)
        adapter.scoper = Scoper(policy("/work", profile="company"))
        out_of_scope = adapter.read_record(
            replace(reference, cwd=None),
            witness.native_identity,
            witness.digest,
            locator=witness.locator,
        )
        assert out_of_scope.status is NativeSourceStatus.OUT_OF_SCOPE
        assert out_of_scope.evidence is None and out_of_scope.bytes_read > 0
        assert inventory(database) == after_owner_change
    finally:
        writer.close()


def test_native_opencode_snapshot_bounds_rows_and_rejects_missing_required_table(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.readers.native_source import NativeSourceStatus

    database = tmp_path / "bounded-source.db"
    writer = store(database, parts=1)
    _, reference, adapter = native_adapter_for_opencode(database, tmp_path)
    try:
        before = inventory(database)
        positive = adapter.resolve(reference.ref)
        assert positive.status is NativeSourceStatus.OK and positive.line is not None
        assert inventory(database) == before

        _, bounded_reference, bounded_adapter = native_adapter_for_opencode(database, tmp_path)
        bounded_adapter.max_records = 1
        bounded = bounded_adapter.resolve(bounded_reference.ref)
        assert bounded.status is NativeSourceStatus.BOUNDED and bounded.line is None
        assert bounded.bytes_read > 0
        assert inventory(database) == before

        writer.execute("DROP TABLE part")
        writer.commit()
        after_schema_change = inventory(database)
        _, schema_reference, schema_adapter = native_adapter_for_opencode(database, tmp_path)
        unsupported = schema_adapter.resolve(schema_reference.ref)
        assert unsupported.status is NativeSourceStatus.UNSUPPORTED
        assert unsupported.line is None
        assert inventory(database) == after_schema_change
    finally:
        writer.close()


def test_darwin_var_alias_pins_the_same_native_sqlite_source(tmp_path: Path) -> None:
    import sys

    if sys.platform != "darwin":
        pytest.skip("Darwin-only fixed /var alias")
    try:
        relative = tmp_path.relative_to("/private/var")
    except ValueError:
        pytest.skip("temporary directory is not below /private/var")

    database = tmp_path / "alias.db"
    seed(database).close()
    alias = Path("/var") / relative / database.name
    before = inventory(database)
    assert native_vfs.source_generation(alias) == native_vfs.source_generation(database)
    assert inventory(database) == before


def test_native_opencode_row_limit_counts_ignored_real_sqlite_rows(tmp_path: Path) -> None:
    from ssgrep.accounting.readers.native_source import NativeSourceStatus

    database = tmp_path / "row-limit.db"
    writer = opencode_connect(database)
    writer.execute("BEGIN")
    opencode_add_session(writer, "owner")
    opencode_add_message(writer, "message", "owner", opencode_assistant_data())
    for index in range(3):
        opencode_add_part(
            writer,
            f"ignored-{index}",
            "message",
            "owner",
            {"type": "text", "text": "not a tool witness"},
            at=T0 + index,
        )
    writer.execute("COMMIT")
    reader, reference, adapter = native_adapter_for_opencode(database, tmp_path)
    try:
        before = inventory(database)
        positive = adapter.resolve(reference.ref)
        assert positive.status is NativeSourceStatus.OK
        assert inventory(database) == before

        _, bounded_reference, bounded_adapter = native_adapter_for_opencode(database, tmp_path)
        bounded_adapter.max_records = 2
        bounded = bounded_adapter.resolve(bounded_reference.ref)
        assert bounded.status is NativeSourceStatus.BOUNDED and bounded.line is None
        assert bounded.bytes_read > 0
        assert inventory(database) == before
    finally:
        writer.close()


def test_native_source_parent_symlink_is_rejected_without_reading_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    database = real / "source.db"
    seed(database).close()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)

    guard = SourceGuard(database)
    guard.close()
    before = inventory(database)
    with pytest.raises(OSError):
        SourceGuard(link / database.name)
    assert inventory(database) == before


def test_pin_directory_rejects_parent_traversal_and_unreadable_aliases(tmp_path, monkeypatch):
    import sys

    (tmp_path / "child").mkdir()
    pinned = native_vfs.pin_directory(tmp_path / "child")
    os.close(pinned)
    with pytest.raises(SnapshotChanged, match="parent traversal"):
        native_vfs.pin_directory(tmp_path / "child" / "..")
    with pytest.raises(SnapshotChanged, match="parent traversal"):
        SourceGuard(tmp_path / "child" / ".." / "native.db")

    if sys.platform != "darwin":
        pytest.skip("Darwin-only fixed /var alias")
    try:
        relative = tmp_path.relative_to("/private/var")
    except ValueError:
        pytest.skip("temporary directory is not below /private/var")
    alias = Path("/var") / relative
    os.close(native_vfs.pin_directory(alias))  # the alias normalizes while readlink works

    def unreadable(path, *args, **kwargs):
        raise PermissionError(path)

    monkeypatch.setattr(os, "readlink", unreadable)
    with pytest.raises(OSError):  # unnormalized /var is a symlink, so no-follow traversal refuses
        native_vfs.pin_directory(alias)


def test_source_guard_requires_existing_main_file_and_rejects_foreign_process_use(tmp_path):
    with pytest.raises(FileNotFoundError):
        SourceGuard(tmp_path / "absent.db")
    database = tmp_path / "guard.db"
    seed(database).close()
    guard = SourceGuard(database)
    try:
        assert guard.verify() is None
        assert guard.descriptor() is not None
        guard.pid = os.getpid() + 1
        with pytest.raises(SnapshotChanged, match="another process"):
            guard.descriptor()
        with pytest.raises(SnapshotChanged, match="another process"):
            guard.stamp(database.name, guard.files[0][1], 100)
        guard.pid = os.getpid()
        assert guard.descriptor() is not None
        database.unlink()
        with pytest.raises(SnapshotChanged, match="became unavailable"):
            guard.verify()
    finally:
        guard.close()


def test_readonly_vfs_refuses_foreign_process_guard_and_non_string_names(tmp_path, monkeypatch):
    database = tmp_path / "vfs.db"
    seed(database).close()
    vfs = readonly_vfs()
    guard = SourceGuard(database)
    try:
        guard.pid = os.getpid() + 1
        with pytest.raises(apsw.CantOpenError, match="another process"):
            with vfs.bind(guard):
                pytest.fail("a foreign descriptor must never be bound")
        guard.pid = os.getpid()
        with vfs.bind(guard) as main:
            assert vfs.binding(main)[0] is guard
            guard.pid = os.getpid() + 1
            with pytest.raises(apsw.CantOpenError, match="not bound"):
                vfs.binding(main)
            guard.pid = os.getpid()
        with pytest.raises(apsw.CantOpenError, match="bound filename"):
            vfs.binding(None)
    finally:
        guard.close()

    monkeypatch.setattr(native_vfs, "_instances", {})
    monkeypatch.setattr(apsw, "vfs_names", lambda: ["unix"])
    with pytest.raises(apsw.CantOpenError, match="unavailable"):
        readonly_vfs()


def test_published_commit_accepts_only_a_coherent_wal_index_and_wal_anchor(tmp_path):
    database = tmp_path / "commit.db"
    writer = seed(database)
    wal = Path(str(database) + "-wal")
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = os.open(wal, os.O_RDONLY)
    try:
        header = os.pread(descriptor, 32, 0)
        size = os.fstat(descriptor).st_size
        shared = database.name + "-shm"
        good = native_vfs.published_commit(directory, shared, descriptor, header, size)
        assert good is not None
        frame, offset, anchor = good
        page = int.from_bytes(header[8:12], "big")
        assert frame > 0 and offset == 32 + (frame - 1) * (24 + page)
        assert anchor == os.pread(descriptor, 24, offset)

        def reject(**changes):
            args = {
                "shared": shared,
                "wal_header": header,
                "size": size,
                "descriptor": descriptor,
            }
            args.update(changes)
            return native_vfs.published_commit(
                directory, args["shared"], args["descriptor"], args["wal_header"], args["size"]
            )

        assert reject(shared="missing-shm") is None
        (tmp_path / "dir-shm").mkdir()
        assert reject(shared="dir-shm") is None  # not a regular file
        (tmp_path / "short-shm").write_bytes(b"x" * 10)
        assert reject(shared="short-shm") is None
        raw = (tmp_path / "commit.db-shm").read_bytes()[:96]
        (tmp_path / "torn-shm").write_bytes(raw[:48] + bytes([raw[48] ^ 1]) + raw[49:96])
        assert reject(shared="torn-shm") is None
        versioned = bytearray(raw[:48])
        versioned[0:4] = (3006000).to_bytes(4, "little")
        (tmp_path / "old-shm").write_bytes(bytes(versioned) * 2)
        assert reject(shared="old-shm") is None
        uninitialised = bytearray(raw[:48])
        uninitialised[12] = 0
        (tmp_path / "uninit-shm").write_bytes(bytes(uninitialised) * 2)
        assert reject(shared="uninit-shm") is None
        empty = bytearray(raw[:48])
        empty[16:20] = (0).to_bytes(4, "little")
        (tmp_path / "empty-shm").write_bytes(bytes(empty) * 2)
        assert reject(shared="empty-shm") == (0, 0, b"")
        assert reject(wal_header=header[:16] + b"\0" * 8 + header[24:]) is None  # salt differs
        bad_page = header[:8] + (100).to_bytes(4, "big") + header[12:]
        assert reject(wal_header=bad_page) is None
        assert reject(size=offset + 24) is None  # frame does not fit in the observed WAL
        other = tmp_path / "other"
        other.write_bytes(b"\0" * size)
        other_fd = os.open(other, os.O_RDONLY)
        try:
            assert reject(descriptor=other_fd) is None  # anchor lacks this WAL's salt
        finally:
            os.close(other_fd)
        short = tmp_path / "shorter"
        short.write_bytes(b"\0" * 8)
        short_fd = os.open(short, os.O_RDONLY)
        try:
            assert reject(descriptor=short_fd) is None  # truncated anchor
        finally:
            os.close(short_fd)
    finally:
        os.close(descriptor)
        os.close(directory)
        writer.close()


def native_resolve(database, tmp_path, *, budget=None, wrap=None):
    _, reference, adapter = native_adapter_for_opencode(database, tmp_path)
    if budget is not None:
        adapter.byte_budget = budget
    if wrap is not None:
        reader = adapter.readers["opencode"]
        original = reader.snapshot
        reader.snapshot = wrap(original)
    return adapter.resolve(reference.ref)


def test_native_session_read_budget_and_runtime_failures_are_typed(tmp_path, monkeypatch):
    import sys

    from ssgrep.accounting.readers.native_source import NativeSourceStatus

    database = tmp_path / "typed.db"
    writer = store(database, parts=1)
    try:
        assert native_resolve(database, tmp_path).status is NativeSourceStatus.OK
        tight = native_resolve(database, tmp_path, budget=120)
        assert tight.status is NativeSourceStatus.BOUNDED and tight.line is None

        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "apsw", None)
            unavailable = native_resolve(database, tmp_path)
        assert unavailable.status is NativeSourceStatus.UNREADABLE

        def vanish_for_snapshot(original):
            @contextmanager
            def snapshot(path, *, descriptor=None, tracker=None):
                real = sys._getframe  # keep the real import machinery for everything else
                import builtins

                saved = builtins.__import__

                def refuse(name, *args, **kwargs):
                    if name == "apsw" and real(1).f_code.co_name == "snapshot":
                        raise ImportError("apsw disappeared")
                    return saved(name, *args, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(builtins, "__import__", refuse)
                    with original(path, descriptor=descriptor, tracker=tracker) as connection:
                        yield connection

            return snapshot

        # reader.snapshot delegates to the storage snapshot(); its own import is what is refused.
        refused = native_resolve(database, tmp_path, wrap=vanish_for_snapshot)
        assert refused.status is NativeSourceStatus.UNREADABLE

        def checkpoint(original):
            @contextmanager
            def snapshot(path, *, descriptor=None, tracker=None):
                with original(path, descriptor=descriptor, tracker=tracker) as connection:
                    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    yield connection

            return snapshot

        changed = native_resolve(database, tmp_path, wrap=checkpoint)
        assert changed.status is NativeSourceStatus.CHANGED and changed.line is None
    finally:
        writer.close()


@pytest.mark.parametrize(
    "mutation,status",
    [
        ("missing", "source_missing"),
        ("replaced", "source_unapproved"),
        ("notdir", "source_unreadable"),
    ],
)
def test_native_session_path_is_rechecked_after_snapshot(tmp_path, mutation, status):
    root = tmp_path / "root"
    root.mkdir()
    database = root / "recheck.db"
    store(database, parts=1).close()
    assert native_resolve(database, tmp_path).status.value == "ok"

    def mutate(original):
        @contextmanager
        def snapshot(path, *, descriptor=None, tracker=None):
            with original(path, descriptor=descriptor, tracker=tracker) as connection:
                yield connection
            if mutation == "missing":
                for suffix in ("", "-wal", "-shm"):
                    Path(str(database) + suffix).unlink(missing_ok=True)
            elif mutation == "replaced":
                copy = root / "copy.db"
                copy.write_bytes(database.read_bytes())
                os.replace(copy, database)
            else:
                database.rename(tmp_path / "moved")
                root.rename(tmp_path / "root-away")
                root.write_bytes(b"not a directory")

        return snapshot

    outcome = native_resolve(database, tmp_path, wrap=mutate)
    assert outcome.status.value == status and outcome.line is None
    assert outcome.bytes_read > 0


def test_native_session_query_work_is_interrupted_by_the_progress_bound(tmp_path):
    from ssgrep.accounting.readers.native_source import NativeSourceStatus

    database = tmp_path / "heavy.db"
    writer = store(database, parts=1)
    try:
        writer.execute("BEGIN")
        writer.executemany(
            "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) "
            "VALUES (?, 'other', 'other', 1, 1, '{}')",
            [(f"filler-{index}",) for index in range(600_000)],
        )
        writer.execute("COMMIT")
        before = inventory(database)
        outcome = native_resolve(database, tmp_path)
        assert outcome.status is NativeSourceStatus.BOUNDED and outcome.line is None
        assert inventory(database) == before
    finally:
        writer.close()


def test_native_session_cumulative_bytes_are_bounded_below_the_exact_total(tmp_path):
    from ssgrep.accounting.readers.native_source import NativeSourceStatus

    database = tmp_path / "cumulative.db"
    writer = opencode_connect(database)
    try:
        opencode_add_session(writer, "owner")
        opencode_add_message(writer, "message", "owner", opencode_assistant_data())
        for index in range(3):
            opencode_add_part(
                writer,
                f"part-{index}",
                "message",
                "owner",
                opencode_tool_part(f"call-{index}", "bash", {}, output="y" * 200),
                at=T0 + index,
            )
        whole = native_resolve(database, tmp_path)
        assert whole.status is NativeSourceStatus.OK and whole.bytes_read > 600
        exact = native_resolve(database, tmp_path, budget=whole.bytes_read)
        assert exact.status is NativeSourceStatus.OK and exact.bytes_read == whole.bytes_read
        short = native_resolve(database, tmp_path, budget=whole.bytes_read - 1)
        assert short.status is NativeSourceStatus.BOUNDED and short.line is None
    finally:
        writer.close()
