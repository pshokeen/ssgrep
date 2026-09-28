"""Exhaustive tests for the local, read-only OpenCode SQLite adapter."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ssgrep.sessions import records
from ssgrep.sessions.adapters import opencode
from ssgrep.sessions.adapters.base import SourceFingerprint, TranscriptSource
from ssgrep.utilities.types import SessionFile


def _database(path: Path, *, minimal: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    if minimal:
        connection.executescript(
            """
            CREATE TABLE session (id TEXT PRIMARY KEY);
            CREATE TABLE message (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY, message_id TEXT NOT NULL,
                session_id TEXT NOT NULL, data TEXT NOT NULL
            );
            """
        )
    else:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE session (
                id TEXT PRIMARY KEY, project_id TEXT, parent_id TEXT, directory TEXT,
                title TEXT, version TEXT, time_created INTEGER, time_updated INTEGER,
                model TEXT, agent TEXT
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                time_created INTEGER, time_updated INTEGER, data
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT NOT NULL,
                time_created INTEGER, time_updated INTEGER, data
            );
            CREATE TABLE credential (id TEXT, value TEXT);
            """
        )
    connection.commit()
    return connection


def _insert_session(
    connection: sqlite3.Connection,
    identifier: object,
    *,
    parent: object = None,
    directory: object = "/work/project",
    title: object = "OpenCode title",
    model: object = '{"id":"session-model","providerID":"local"}',
    updated: object = 1_700_000_001_000,
) -> None:
    connection.execute(
        "INSERT INTO session VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            identifier,
            "project-hash",
            parent,
            directory,
            title,
            "1.2.3",
            1_700_000_000_000,
            updated,
            model,
            "build-agent",
        ),
    )


def _insert_message(
    connection: sqlite3.Connection,
    identifier: object,
    session_id: str,
    data: object,
    *,
    created: object = 1_700_000_002_000,
    updated: object = 1_700_000_003_000,
) -> None:
    encoded = json.dumps(data) if isinstance(data, (dict, list)) else data
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        (identifier, session_id, created, updated, encoded),
    )


def _insert_part(
    connection: sqlite3.Connection,
    identifier: str,
    message_id: object,
    session_id: str,
    data: object,
    *,
    created: int,
) -> None:
    encoded = json.dumps(data) if isinstance(data, (dict, list)) else data
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
        (identifier, message_id, session_id, created, created + 1, encoded),
    )


def _source(path: Path, session_id: str = "opencode:session-1") -> TranscriptSource:
    return TranscriptSource(
        adapter="opencode",
        key=session_id,
        session=SessionFile(
            path=path,
            session_id=session_id,
            is_main=True,
            runtime="opencode",
        ),
        fingerprint=SourceFingerprint(0, 0.0, "digest"),
    )


def test_database_path_obeys_override_xdg_and_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    override = tmp_path / "override.db"
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(override))
    assert opencode._database_path() == override

    monkeypatch.delenv("SSGREP_OPENCODE_DB")
    monkeypatch.setenv("XDG_DATA_HOME", "~/xdg-data")
    assert opencode._database_path() == Path.home() / "xdg-data/opencode/opencode.db"

    monkeypatch.delenv("XDG_DATA_HOME")
    assert opencode._database_path() == Path.home() / ".local/share/opencode/opencode.db"


def test_snapshot_is_read_only_consistent_and_always_closes(tmp_path: Path):
    path = tmp_path / "open code #1.db"
    writer = _database(path)
    _insert_session(writer, "one")
    writer.commit()

    with opencode._snapshot(path) as reader:
        assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
        assert reader.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 1
        _insert_session(writer, "two")
        writer.commit()
        assert reader.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("DELETE FROM session")
    with pytest.raises(sqlite3.ProgrammingError):
        reader.execute("SELECT 1")

    with pytest.raises(RuntimeError):
        with opencode._snapshot(path):
            raise RuntimeError("caller failed")
    writer.close()


def test_schema_and_query_helpers_support_full_and_minimal_schema(tmp_path: Path):
    full = _database(tmp_path / "full.db")
    schema = opencode._schema(full)
    assert schema is not None
    assert opencode._selected(schema["session"], "title") == 's."title"'
    assert opencode._selected(set(), "title") == "NULL"
    assert opencode._time_value(schema["session"], table="s") == (
        'COALESCE(s."time_updated", s."time_created", 0)'
    )
    assert opencode._time_value(set(), table="s") == "0"
    assert 'ORDER BY mm."time_created" DESC, mm."id" DESC' in opencode._latest_message_data(
        schema["message"]
    )
    query = opencode._discovery_query(schema)
    assert 'FROM "session" AS s' in query
    full.close()

    minimal = _database(tmp_path / "minimal.db", minimal=True)
    minimal_schema = opencode._schema(minimal)
    assert minimal_schema is not None
    assert 'ORDER BY mm."id" DESC' in opencode._latest_message_data(minimal_schema["message"])
    assert "COALESCE" not in opencode._discovery_query(minimal_schema)
    minimal.close()

    missing = sqlite3.connect(tmp_path / "missing.db")
    missing.execute("CREATE TABLE session (id TEXT)")
    assert opencode._schema(missing) is None
    missing.close()


def test_discover_namespaces_filters_and_fingerprints_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(
        connection,
        "main",
        title="<system-reminder>hide</system-reminder>Main",
        updated=1_700_000_001_000,
    )
    _insert_session(connection, "child", parent="main", directory="/work/project/sub", model=None)
    _insert_session(connection, "elsewhere", directory="/elsewhere", updated="not-a-time")
    _insert_message(
        connection,
        "message-main",
        "main",
        {"role": "user", "model": {"modelID": "message-model"}},
    )
    _insert_part(
        connection,
        "part-main",
        "message-main",
        "main",
        {"type": "text", "text": "hello"},
        created=1_700_000_004_000,
    )
    _insert_message(
        connection,
        "message-child",
        "child",
        {"role": "user", "model": {"modelID": "fallback-model"}},
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(path))

    adapter = opencode.OpenCodeAdapter()
    assert adapter.name == "opencode"
    discovered = adapter.discover()
    assert [source.key for source in discovered] == [
        "opencode:child",
        "opencode:elsewhere",
        "opencode:main",
    ]
    child, elsewhere, main = discovered
    assert child.session.parent_session_id == "opencode:main"
    assert not child.session.is_main
    assert child.session.agent_model == "fallback-model"
    assert child.session.agent_name == "build-agent"
    assert child.session.project_paths == ("/work/project/sub",)
    assert child.session.source_project == "project-hash"
    assert child.session.runtime == "opencode"
    assert child.session.path == path
    assert main.session.agent_model == "session-model"
    assert main.fingerprint.size == 2
    assert main.fingerprint.mtime == 1_700_000_004.001
    assert len(main.fingerprint.digest) == 64
    assert elsewhere.fingerprint.mtime == 0.0
    assert len({source.session.session_id for source in discovered}) == 3

    assert [source.key for source in adapter.discover(no_subagents=True)] == [
        "opencode:elsewhere",
        "opencode:main",
    ]
    assert [source.key for source in adapter.discover(scope="/work/project")] == [
        "opencode:child",
        "opencode:main",
    ]
    assert adapter.discover(scope="/work/pro") == []


def test_discover_fingerprints_stable_then_change_with_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(connection, "main", updated=1_700_000_001_000)
    _insert_session(connection, "other", updated=1_700_000_011_000)
    _insert_message(connection, "m1", "main", {"role": "user"})
    connection.commit()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(path))

    adapter = opencode.OpenCodeAdapter()

    def fingerprints() -> dict[str, SourceFingerprint]:
        return {source.key: source.fingerprint for source in adapter.discover()}

    first = fingerprints()
    assert fingerprints() == first

    # A new message in one session changes only that session's fingerprint, so
    # unchanged OpenCode sessions memo-hit during a later ``ssgrep index``.
    _insert_message(
        connection,
        "m2",
        "main",
        {"role": "assistant"},
        created=1_700_000_005_000,
        updated=1_700_000_006_000,
    )
    connection.commit()
    after = fingerprints()
    assert after["opencode:other"] == first["opencode:other"]
    assert after["opencode:main"] != first["opencode:main"]
    connection.close()


def test_discover_tolerates_absent_invalid_and_duplicate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = opencode.OpenCodeAdapter()
    missing = tmp_path / "missing.db"
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(missing))
    assert adapter.discover() == []

    invalid = tmp_path / "invalid.db"
    invalid.write_text("not sqlite")
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(invalid))
    assert adapter.discover() == []

    incomplete = tmp_path / "incomplete.db"
    connection = sqlite3.connect(incomplete)
    connection.execute("CREATE TABLE session (id TEXT)")
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(incomplete))
    assert adapter.discover() == []

    fake_path = tmp_path / "fake.db"
    fake_path.touch()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(fake_path))
    row = MagicMock()
    row.__getitem__.return_value = None
    duplicate = _source(fake_path, "opencode:duplicate")

    @contextmanager
    def fake_snapshot(_path: Path):
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [row, row, row]
        yield connection

    monkeypatch.setattr(opencode, "_snapshot", fake_snapshot)
    monkeypatch.setattr(
        opencode,
        "_schema",
        lambda _connection: {"session": set(), "message": {"id"}, "part": set()},
    )
    produced = iter([None, duplicate, duplicate])
    monkeypatch.setattr(opencode, "_source_from_row", lambda _path, _row: next(produced))
    assert [source.key for source in adapter.discover()] == ["opencode:duplicate"]

    @contextmanager
    def broken_snapshot(_path: Path):
        raise OSError("gone")
        yield  # pragma: no cover

    monkeypatch.setattr(opencode, "_snapshot", broken_snapshot)
    assert adapter.discover() == []


def test_read_normalizes_text_tools_files_and_safe_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(
        connection,
        "session-1",
        directory="/work/project",
        title="<tag>hidden</tag>  Useful\n title\x00",
    )
    _insert_message(
        connection,
        "message-user",
        "session-1",
        {
            "role": "user",
            "time": {"created": 1_700_000_002_000},
            "model": {"modelID": "user-model"},
        },
        created=1_700_000_002_000,
    )
    _insert_message(
        connection,
        "message-assistant",
        "session-1",
        {
            "role": "assistant",
            "parentID": "message-user",
            "time": {"created": 1_700_000_005_000},
            "path": {"cwd": "/work/project/nested"},
            "modelID": "assistant-model",
        },
        created=1_700_000_005_000,
    )
    part_number = 0

    def part(message: str, value: object) -> None:
        nonlocal part_number
        part_number += 1
        _insert_part(
            connection,
            f"part-{part_number:02}",
            message,
            "session-1",
            value,
            created=1_700_000_010_000 + part_number,
        )

    part("message-user", {"type": "text", "text": "user prompt"})
    part(
        "message-user",
        {"type": "file", "source": {"path": "src/input.py"}, "url": "ignored"},
    )
    part("message-user", {"type": "reasoning", "text": "SECRET REASONING"})
    part("message-user", {"type": "step-start", "snapshot": "SECRET SNAPSHOT"})
    part("message-assistant", {"type": "text", "text": "assistant response"})
    part(
        "message-assistant",
        {
            "type": "tool",
            "tool": "edit",
            "state": {
                "input": {"filePath": "src/app.py", "newString": "SECRET INPUT"},
                "output": "SECRET TOOL OUTPUT",
            },
        },
    )
    part(
        "message-assistant",
        {
            "type": "tool",
            "tool": "bash",
            "state": {"input": {"command": "echo SECRET COMMAND"}, "output": "SECRET"},
        },
    )
    part("message-assistant", {"type": "patch", "files": ["src/a.py", "src/b.py"]})
    part("message-assistant", {"type": "text", "text": "ignored", "ignored": True})
    connection.execute("INSERT INTO credential VALUES ('token', 'CREDENTIAL SECRET')")
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(path))

    source = opencode.OpenCodeAdapter().discover()[0]
    result = opencode.OpenCodeAdapter().read(source)
    assert result.malformed_records == 0
    assert result.skipped_records == 0
    assert [record["type"] for record in result.records] == [
        "custom-title",
        "user",
        "assistant",
    ]
    title, user, assistant = result.records
    assert title["custom-title"] == "Useful title"
    assert title["cwd"] == "/work/project"
    assert title["timestamp"] == "2023-11-14T22:13:20+00:00"
    assert user["uuid"] == "opencode:message-user"
    assert user["sessionId"] == "opencode:session-1"
    assert user["message"]["model"] == "user-model"
    assert user["message"]["content"] == [
        {"type": "text", "text": "user prompt"},
        {"type": "tool_use", "name": "Read", "input": {"file_path": "src/input.py"}},
    ]
    assert assistant["parentUuid"] == "opencode:message-user"
    assert assistant["cwd"] == "/work/project/nested"
    assert assistant["timestamp"] == "2023-11-14T22:13:25+00:00"
    assert assistant["message"]["model"] == "assistant-model"
    assert assistant["message"]["content"] == [
        {"type": "text", "text": "assistant response"},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/app.py"}},
        {"type": "tool_use", "name": "Bash", "input": {}},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/a.py"}},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/b.py"}},
    ]
    serialized = json.dumps(result.records)
    assert "SECRET" not in serialized
    assert "CREDENTIAL" not in serialized
    assert "reasoning" not in serialized


def test_read_drops_synthetic_and_system_reminder_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(connection, "session-1")
    _insert_message(
        connection,
        "message-user",
        "session-1",
        {"role": "user", "time": {"created": 1_700_000_002_000}},
        created=1_700_000_002_000,
    )
    _insert_message(
        connection,
        "message-assistant",
        "session-1",
        {"role": "assistant", "parentID": "message-user"},
        created=1_700_000_005_000,
    )
    _insert_part(
        connection,
        "part-01",
        "message-user",
        "session-1",
        {
            "type": "text",
            "text": "Called the Read tool with the following input: {}",
            "synthetic": True,
        },
        created=1_700_000_010_000,
    )
    _insert_part(
        connection,
        "part-02",
        "message-user",
        "session-1",
        {
            "type": "text",
            "text": (
                "<system-reminder>\n[BACKGROUND TASK COMPLETED]\n"
                "</system-reminder>\n<!-- OMO_INTERNAL_INITIATOR -->"
            ),
        },
        created=1_700_000_010_001,
    )
    _insert_part(
        connection,
        "part-03",
        "message-user",
        "session-1",
        {"type": "file", "synthetic": True, "source": {"path": "context.txt"}},
        created=1_700_000_010_002,
    )
    _insert_part(
        connection,
        "part-04",
        "message-user",
        "session-1",
        {"type": "text", "text": "real user prompt"},
        created=1_700_000_010_003,
    )
    _insert_part(
        connection,
        "part-05",
        "message-user",
        "session-1",
        {"type": "text", "text": "real body\n<system-reminder>todo</system-reminder>"},
        created=1_700_000_010_004,
    )
    _insert_part(
        connection,
        "part-06",
        "message-assistant",
        "session-1",
        {"type": "text", "text": "assistant response"},
        created=1_700_000_010_005,
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(path))

    source = opencode.OpenCodeAdapter().discover()[0]
    result = opencode.OpenCodeAdapter().read(source)
    assert result.malformed_records == 0
    assert result.skipped_records == 0
    assert [record["type"] for record in result.records] == [
        "custom-title",
        "user",
        "assistant",
    ]
    user, assistant = result.records[1], result.records[2]
    assert user["message"]["content"] == [
        {"type": "text", "text": "real user prompt"},
        {"type": "text", "text": "real body\n<system-reminder>todo</system-reminder>"},
    ]
    assert assistant["message"]["content"] == [{"type": "text", "text": "assistant response"}]
    serialized = json.dumps(result.records)
    assert "Called the Read tool" not in serialized
    assert "[BACKGROUND TASK COMPLETED]" not in serialized
    assert "context.txt" not in serialized


def test_read_tolerates_malformed_rows_and_uses_legacy_message_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(connection, "session-1", directory=None, title=None, model="plain-model")
    _insert_message(
        connection,
        "fallback-string",
        "session-1",
        {"role": "user", "content": "fallback prompt", "time": "wrong-shape"},
        created=1_700_000_002_000,
    )
    _insert_message(
        connection,
        "fallback-list",
        "session-1",
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "fallback response"},
                {"type": "reasoning", "text": "ignored"},
                {"type": "text", "text": 3},
                "bad",
            ],
            "parent_id": "fallback-string",
            "model_id": "message-model",
        },
        created=1_700_000_003_000,
    )
    _insert_message(connection, "bad-json", "session-1", "{")
    _insert_message(connection, "scalar", "session-1", "[]")
    _insert_message(connection, "system", "session-1", {"role": "system"})
    _insert_message(connection, None, "session-1", {"role": "user", "content": "bad id"})
    _insert_part(
        connection,
        "bad-part-json",
        "fallback-string",
        "session-1",
        "{",
        created=1_700_000_010_000,
    )
    _insert_part(
        connection,
        "scalar-part",
        "fallback-string",
        "session-1",
        "[]",
        created=1_700_000_010_001,
    )
    _insert_part(
        connection,
        "orphan",
        "missing-message",
        "session-1",
        {"type": "text", "text": "orphan"},
        created=1_700_000_010_002,
    )
    _insert_part(
        connection,
        "unknown",
        "fallback-string",
        "session-1",
        {"type": "future"},
        created=1_700_000_010_003,
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(path))

    source = opencode.OpenCodeAdapter().discover()[0]
    result = opencode.OpenCodeAdapter().read(source)
    assert result.malformed_records == 2
    assert result.skipped_records == 6
    assert [record["type"] for record in result.records] == ["user", "assistant"]
    assert result.records[0]["message"]["content"] == [{"type": "text", "text": "fallback prompt"}]
    assert result.records[0]["message"]["model"] == "plain-model"
    assert result.records[0]["timestamp"] == "2023-11-14T22:13:22+00:00"
    assert result.records[1]["message"]["content"] == [
        {"type": "text", "text": "fallback response"}
    ]
    assert result.records[1]["parentUuid"] == "opencode:fallback-string"
    assert "cwd" not in result.records[1]


def test_read_rejects_wrong_sources_and_database_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = opencode.OpenCodeAdapter()
    missing = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError) as missing_db:
        adapter.read(_source(missing))
    assert missing_db.value.filename == str(missing)

    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(connection, "session-1")
    connection.commit()
    connection.close()
    assert adapter.read(_source(path, "native:session-1")).records == ()
    assert adapter.read(_source(path, "opencode:")).records == ()
    # A row deleted from an otherwise-intact database (issue #8) must raise
    # the same missing-source condition as a missing database, not return an
    # empty result: `process_source` distinguishes "nothing to declare" from
    # "redeclare this session's rows from the archive" by that exception, and
    # a silent empty read would reconcile this session's history away.
    with pytest.raises(FileNotFoundError) as missing_row:
        adapter.read(_source(path, "opencode:missing"))
    assert missing_row.value.filename == str(path)

    incomplete = tmp_path / "incomplete.db"
    connection = sqlite3.connect(incomplete)
    connection.execute("CREATE TABLE session (id TEXT)")
    connection.commit()
    connection.close()
    assert adapter.read(_source(incomplete)).records == ()

    # A file that exists but is not a readable database is a genuine read
    # error, not "no records": returning empty would reconcile the session's
    # indexed rows away (issue #13), so it raises like a locked database.
    invalid = tmp_path / "invalid.db"
    invalid.write_text("not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        adapter.read(_source(invalid))

    @contextmanager
    def broken_snapshot(_path: Path):
        raise OSError("gone")
        yield  # pragma: no cover

    monkeypatch.setattr(opencode, "_snapshot", broken_snapshot)
    with pytest.raises(OSError, match="gone"):
        adapter.read(_source(path))


def test_read_raises_on_transient_sqlite_error_instead_of_returning_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Issue #13: a locked database is retryable, never an empty transcript.

    The same source reads fine before the fault and again after it clears, so
    the raise is attributable to the injected error alone (not to a bad
    fixture), and it must be the sqlite error itself -- not the
    missing-source ``FileNotFoundError`` that triggers archive recovery.
    """
    adapter = opencode.OpenCodeAdapter()
    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(connection, "session-1")
    _insert_message(
        connection,
        "m1",
        "session-1",
        {"role": "user", "content": "hello"},
        created=1_700_000_002_000,
    )
    _insert_part(
        connection,
        "p1",
        "m1",
        "session-1",
        {"type": "text", "text": "hi"},
        created=1_700_000_003_000,
    )
    connection.commit()
    connection.close()
    source = _source(path, "opencode:session-1")
    healthy = adapter.read(source)
    assert healthy.records

    real_snapshot = opencode._snapshot

    @contextmanager
    def locked_snapshot(_path: Path):
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover

    monkeypatch.setattr(opencode, "_snapshot", locked_snapshot)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        adapter.read(source)
    # present() must keep reporting True here (no archive snapshot needed
    # because read() raises rather than returning an empty result).
    assert adapter.present(source) is True

    monkeypatch.setattr(opencode, "_snapshot", real_snapshot)
    assert adapter.read(source).records == healthy.records


def test_present_mirrors_reads_raise_no_raise_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``present()`` is False exactly where ``read()`` raises the
    missing-source condition, and True everywhere ``read()`` instead
    tolerates the case by returning an empty result -- never treating "I
    couldn't tell" the same as "it's gone" (see ``read()``'s own tests for
    the mirrored empty-result cases).
    """
    adapter = opencode.OpenCodeAdapter()
    missing_db = tmp_path / "missing.db"
    assert adapter.present(_source(missing_db)) is False

    path = tmp_path / "opencode.db"
    connection = _database(path)
    _insert_session(connection, "session-1")
    connection.commit()
    connection.close()
    assert adapter.present(_source(path)) is True
    assert adapter.present(_source(path, "opencode:missing")) is False
    # Malformed identifiers (wrong namespace, empty id): read() returns an
    # empty result rather than raising, so present() reports True too.
    assert adapter.present(_source(path, "native:session-1")) is True
    assert adapter.present(_source(path, "opencode:")) is True

    incomplete = tmp_path / "incomplete.db"
    connection = sqlite3.connect(incomplete)
    connection.execute("CREATE TABLE session (id TEXT)")
    connection.commit()
    connection.close()
    assert adapter.present(_source(incomplete)) is True

    invalid = tmp_path / "invalid.db"
    invalid.write_text("not sqlite")
    assert adapter.present(_source(invalid)) is True

    @contextmanager
    def broken_snapshot(_path: Path):
        raise OSError("gone")
        yield  # pragma: no cover

    monkeypatch.setattr(opencode, "_snapshot", broken_snapshot)
    assert adapter.present(_source(path)) is True


def test_minimal_schema_is_confidently_detected_and_read(tmp_path: Path):
    path = tmp_path / "minimal.db"
    connection = _database(path, minimal=True)
    connection.execute("INSERT INTO session VALUES ('session-1')")
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?)",
        ("message-1", "session-1", json.dumps({"role": "user", "content": "hello"})),
    )
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?, ?)",
        ("part-1", "message-1", "session-1", json.dumps({"type": "text", "text": "hi"})),
    )
    connection.commit()
    connection.close()

    source = opencode._source_from_row  # ensure the current SQLite path, not legacy JSON, is used
    assert source is not None
    adapter = opencode.OpenCodeAdapter()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SSGREP_OPENCODE_DB", str(path))
        discovered = adapter.discover()
    assert len(discovered) == 1
    assert discovered[0].fingerprint.size == 2
    assert discovered[0].fingerprint.mtime == 0.0
    assert adapter.read(discovered[0]).records[0]["message"]["content"] == [
        {"type": "text", "text": "hi"}
    ]


def test_json_model_time_and_path_helpers_cover_defensive_shapes(tmp_path: Path):
    assert opencode._json_object(b'{"x": 1}') == ({"x": 1}, None)
    assert opencode._json_object(b"\xff") == (None, "malformed")
    assert opencode._json_object(3) == (None, "skipped")
    assert opencode._json_object("{") == (None, "malformed")
    assert opencode._json_object("[]") == (None, "skipped")
    oversized = "x" * (records.MAX_LINE_BYTES + 1)
    assert opencode._json_object(oversized) == (None, "skipped")

    assert opencode._model_from_object({"modelID": " model-a "}) == "model-a"
    assert opencode._model_from_object({"modelID": 1, "id": "model-b"}) == "model-b"
    assert opencode._model_from_object({"model": "x" * 300}) == "x" * 200
    assert opencode._model_from_object("safe/model") == "safe/model"
    assert opencode._model_from_object("unsafe model") is None
    assert opencode._model_from_object(3) is None
    assert opencode._model_from_json({"id": "dict-model"}) == "dict-model"
    assert opencode._model_from_json('"json-model"') == "json-model"
    assert opencode._model_from_json("plain-model") == "plain-model"
    assert opencode._model_from_json(b"\xff") is None
    assert opencode._message_model({"model": {"id": "one"}}) == "one"
    assert opencode._message_model({"modelID": "two"}) == "two"
    assert opencode._message_model({"model_id": "three"}) == "three"
    assert opencode._message_model({}) is None

    assert opencode._milliseconds(True) == 0
    assert opencode._milliseconds("bad") == 0
    assert opencode._milliseconds("inf") == 0
    assert opencode._milliseconds("12.5") == 12.5
    assert opencode._iso_timestamp("2024-01-01T01:00:00+01:00") == ("2024-01-01T00:00:00+00:00")
    assert opencode._iso_timestamp("2024-01-01T00:00:00") == "2024-01-01T00:00:00+00:00"
    assert opencode._iso_timestamp("bad") is None
    assert opencode._iso_timestamp(0) is None
    assert opencode._iso_timestamp(1) == "1970-01-01T00:00:01+00:00"
    assert opencode._iso_timestamp(1_700_000_000_000) == "2023-11-14T22:13:20+00:00"
    assert opencode._iso_timestamp(10**1000) is None
    first = opencode._message_timestamp({"time": {"created": 1}}, 2)
    second = opencode._message_timestamp({"time": "bad"}, 2)
    assert first is not None and first.endswith("01+00:00")
    assert second is not None and second.endswith("02+00:00")

    assert opencode._message_cwd({"path": {"cwd": "/message"}}, "/fallback") == "/message"
    assert opencode._message_cwd({"path": {"cwd": ""}}, "/fallback") == "/fallback"
    assert opencode._message_cwd({}, None) is None
    assert opencode._clean_string(" value ") == " value "
    assert opencode._clean_string(1) is None
    assert opencode._namespaced("id") == "opencode:id"
    assert opencode._raw_session_id(_source(tmp_path / "x", "opencode:id")) == "id"
    assert opencode._raw_session_id(_source(tmp_path / "x", "native:id")) is None
    assert opencode._iso_timestamp(1e20) is None
    empty_row = MagicMock()
    empty_row.__getitem__.return_value = None
    assert opencode._source_from_row(tmp_path / "x", empty_row) is None

    fake = MagicMock()
    fake.is_file.side_effect = OSError("denied")
    assert not opencode._is_file(fake)


def test_tool_file_part_and_record_helpers_cover_every_supported_shape():
    assert opencode._canonical_tool(None) is None
    assert opencode._canonical_tool("  ") is None
    assert opencode._canonical_tool("read") == "Read"
    assert opencode._canonical_tool("custom") == "custom"
    assert opencode._file_path(1) is None
    assert opencode._input_paths(None) == ()
    assert opencode._input_paths(
        {"file_path": "a", "filePath": "a", "paths": ["b", 3], "filename": "c"}
    ) == ("a", "b", "c")
    assert opencode._tool_blocks({"type": "tool"}) == []
    assert opencode._tool_blocks({"name": "bash", "input": {"command": "secret"}}) == [
        {"type": "tool_use", "name": "Bash", "input": {}}
    ]
    assert opencode._tool_blocks(
        {"tool": "write", "state": {"input": {"path": "a", "paths": ["b"]}}}
    ) == [
        {"type": "tool_use", "name": "Write", "input": {"file_path": "a"}},
        {"type": "tool_use", "name": "Write", "input": {"file_path": "b"}},
    ]

    assert opencode._file_part_path({"source": {"path": "source.py"}}) == "source.py"
    assert opencode._file_part_path({"url": "file:///tmp/a%20b.py"}) == "/tmp/a b.py"
    assert opencode._file_part_path({"url": "https://example.test/x", "filename": "x"}) == "x"
    assert opencode._file_part_path({}) is None

    assert opencode._part_blocks({"type": "text", "ignored": True}) == ([], False)
    assert opencode._part_blocks({"type": "text", "text": 3}) == ([], True)
    assert opencode._part_blocks({"type": "text", "text": ""}) == ([], False)
    assert opencode._part_blocks({"type": "text", "text": "x"}) == (
        [{"type": "text", "text": "x"}],
        False,
    )
    assert opencode._part_blocks({"type": "tool", "tool": "grep"})[1] is False
    assert opencode._part_blocks({"type": "file"}) == ([], True)
    assert opencode._part_blocks({"type": "patch", "files": "bad"}) == ([], True)
    assert opencode._part_blocks({"type": "patch", "files": ["a", 2]}) == (
        [{"type": "tool_use", "name": "Edit", "input": {"file_path": "a"}}],
        True,
    )
    for ignored in opencode._IGNORED_PART_TYPES:
        assert opencode._part_blocks({"type": ignored}) == ([], False)
    assert opencode._part_blocks({"type": "future"}) == ([], True)

    assert opencode._fallback_content({"content": "text"}) == [{"type": "text", "text": "text"}]
    assert opencode._fallback_content({"content": ""}) == []
    assert opencode._fallback_content({"content": 3}) == []
    assert opencode._fallback_content(
        {"content": [{"type": "text", "text": "x"}, {"type": "text", "text": ""}]}
    ) == [{"type": "text", "text": "x"}]

    assert opencode._title_record("t", "s", None, None) == {
        "type": "custom-title",
        "custom-title": "t",
        "sessionId": "s",
    }
    minimal = opencode._message_record(
        role="user",
        message_id="m",
        session_id="s",
        data={},
        content=[],
        cwd=None,
        timestamp=None,
        model=None,
    )
    assert "parentUuid" not in minimal and "cwd" not in minimal and "timestamp" not in minimal
    assert "model" not in minimal["message"]
