"""Portable continuity checks for OpenCode's persisted SQLite cursors."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from ssgrep.accounting.readers.base import EvidenceLedger, SourceState
from ssgrep.accounting.readers.opencode import OpenCodeReader
from tests.accounting.readers_fixtures import (
    T0,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)


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


def seed(path: Path) -> sqlite3.Connection:
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
    writer = seed(path)
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
    writer = seed(path)
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
    writer = seed(path)
    try:
        reader = reader_for(path, monkeypatch)
        source = reader.source_for(path)
        first = reader.read(source)
        snapshot = reader.snapshot

        @contextmanager
        def replace_after_snapshot_is_pinned(database: Path):
            with snapshot(database) as connection:
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
    seed(path).close()
    reader = reader_for(path, monkeypatch)
    source = reader.source_for(path)
    first = reader.read(source)
    path.unlink()
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    replacement = seed(path)
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
