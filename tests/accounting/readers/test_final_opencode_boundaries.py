"""Real SQLite malformed shapes, schema failures, and bounded pending recovery."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.readers.opencode import PENDING_CAP, OpenCodeReader
from tests.accounting.readers_fixtures import (
    T0,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)


def test_unknown_and_malformed_parts_are_counted_without_hiding_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    writer = opencode_connect(path)
    try:
        opencode_add_session(writer, "session")
        opencode_add_message(writer, "user", "session", {"role": "user"})
        opencode_add_message(writer, "assistant", "session", opencode_assistant_data())
        parts = [
            "null",
            "[]",
            {"type": "text", "text": "ssgrep search is only a mention"},
            {"type": "future-" + "x" * 100},
            {"type": []},
            {"type": "tool", "callID": 7, "state": {"status": "completed"}},
            {"type": "tool", "callID": "bad-state", "state": {"status": []}},
            opencode_tool_part("good", "search_sessions", {"query": "recovery"}),
        ]
        for index, part in enumerate(parts):
            opencode_add_part(writer, f"part-{index}", "assistant", "session", part)
        reader = OpenCodeReader({"SSGREP_OPENCODE_DB": str(path)})
        result = reader.read(reader.source_for(path))
        assert result.unrecognized == {
            "<non-object>": 2,
            ("future-" + "x" * 100)[:64]: 1,
            "[]": 1,
            "tool": 2,
        }
        assert result.evidence_ids() == [
            "session:session",
            "usage:assistant",
            "call:good",
            "result:good",
        ]
        assert result.evidence[-1].payload["output"] == "ok"
        assert result.pending_records == 0
    finally:
        writer.close()


def test_unsupported_schema_is_explicit_and_recovers_after_real_migration(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    writer = opencode_connect(path)
    try:
        writer.execute("ALTER TABLE part RENAME COLUMN data TO legacy_data")
        opencode_add_session(writer, "session")
        reader = OpenCodeReader({"SSGREP_OPENCODE_DB": str(path)})
        source = reader.source_for(path)
        unsupported = reader.read(source)
        assert unsupported.evidence == []
        assert [(gap.reason, gap.locator) for gap in unsupported.gaps] == [
            ("unsupported_schema", "part")
        ]
        assert "wm" not in unsupported.state.extra
        writer.execute("ALTER TABLE part RENAME COLUMN legacy_data TO data")
        opencode_add_part(
            writer,
            "part",
            "message",
            "session",
            opencode_tool_part("good", "bash", {"command": "ssgrep show ref"}),
        )
        recovered = reader.read(source, unsupported.state)
        assert recovered.gaps == []
        assert recovered.evidence_ids() == ["session:session", "call:good", "result:good"]
    finally:
        writer.close()


def test_snapshot_authorizer_denies_sensitive_tables_but_allows_schema_and_native_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    writer = opencode_connect(path)
    try:
        opencode_add_session(writer, "session")
        writer.execute("CREATE TABLE credentials(secret TEXT)")
        writer.execute("INSERT INTO credentials VALUES ('isolated-secret-canary')")
        writer.execute("INSERT INTO event VALUES ('e','s',1,'copy','private event')")
        reader = OpenCodeReader({"SSGREP_OPENCODE_DB": str(path)})
        with reader.snapshot(path) as snapshot:
            assert snapshot.execute("SELECT id FROM session").fetchall() == [("session",)]
            assert snapshot.execute(
                "SELECT name FROM sqlite_master WHERE name = 'session'"
            ).fetchall() == [("session",)]
            for table in ("credentials", "event"):
                with pytest.raises(sqlite3.DatabaseError):
                    snapshot.execute(f"SELECT * FROM {table}").fetchall()
            assert snapshot.execute("SELECT id FROM session").fetchall() == [("session",)]
        assert reader.tables_read == {"session"}
    finally:
        writer.close()


def test_pending_overflow_is_reported_and_retained_ids_recover_without_timestamp_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    writer = opencode_connect(path)
    try:
        opencode_add_session(writer, "session")
        for index in range(PENDING_CAP + 1):
            opencode_add_part(
                writer,
                f"part-{index:04}",
                "message",
                "session",
                opencode_tool_part(f"call-{index}", "bash", {}, status="running"),
            )
        reader = OpenCodeReader({"SSGREP_OPENCODE_DB": str(path)})
        source = reader.source_for(path)
        pending = reader.read(source)
        assert pending.evidence_ids() == ["session:session"]
        assert pending.pending_records == PENDING_CAP + 1
        assert len(pending.state.extra["pending"]["part"]) == PENDING_CAP
        assert [(gap.reason, gap.size, gap.locator) for gap in pending.gaps] == [
            ("pending_overflow", PENDING_CAP + 1, "part")
        ]
        done = opencode_tool_part("call-0", "bash", {}, output="completed retained id")
        writer.execute("UPDATE part SET data=? WHERE id='part-0000'", (json.dumps(done),))
        resumed = reader.read(source, pending.state)
        assert resumed.evidence_ids() == ["call:call-0", "result:call-0"]
        assert resumed.evidence[-1].payload["output"] == "completed retained id"
        assert resumed.pending_records == PENDING_CAP - 1
        assert resumed.gaps == []
        # An overflowed id can still be recovered when its actual update advances the watermark.
        last = PENDING_CAP
        writer.execute(
            "UPDATE part SET data=?, time_updated=? WHERE id=?",
            (
                json.dumps(opencode_tool_part(f"call-{last}", "bash", {}, output="late update")),
                T0 + 1,
                f"part-{last:04}",
            ),
        )
        recovered = reader.read(source, resumed.state)
        assert recovered.evidence_ids() == [f"call:call-{last}", f"result:call-{last}"]
        assert recovered.evidence[-1].payload["output"] == "late update"
    finally:
        writer.close()


@pytest.mark.parametrize("native,expected", [(0.125, "0.125"), (2.0, 2)])
def test_sqlite_numeric_metadata_is_canonical_without_changing_native_session_identity(
    tmp_path: Path, native: float, expected: str | int
) -> None:
    path = tmp_path / "opencode.db"
    writer = opencode_connect(path)
    try:
        opencode_add_session(writer, "native-session")
        # SQLite schema support checks column names, not affinity. Unlike JSON data,
        # session metadata can therefore arrive as a Python float from the DB driver.
        writer.execute("ALTER TABLE session RENAME COLUMN project_id TO old_project_id")
        writer.execute("ALTER TABLE session ADD COLUMN project_id REAL")
        writer.execute("UPDATE session SET project_id=?", (native,))
        reader = OpenCodeReader({"SSGREP_OPENCODE_DB": str(path)})
        result = reader.read(reader.source_for(path))
        assert result.gaps == []
        assert result.evidence_ids() == ["session:native-session"]
        header = result.evidence[0]
        assert header.session_id == "native-session"
        encoded = json.loads(canonical_bytes(header))
        assert encoded["payload"]["project_id"] == expected
        assert not isinstance(header.payload["project_id"], float)
    finally:
        writer.close()
