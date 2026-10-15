"""A native snapshot never creates or modifies SQLite main/WAL/SHM bytes."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import pytest

from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.opencode_snapshot import SnapshotChanged
from tests.accounting.readers_fixtures import (
    T0,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)


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
