"""Read OpenCode sessions from its local SQLite transcript database."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ssgrep.sessions.adapters.base import ReadResult, SourceFingerprint, TranscriptSource
from ssgrep.sessions.adapters.opencode_records import (  # noqa: F401  (re-exports)
    _IGNORED_PART_TYPES,
    _NAMESPACE,
    _SYSTEM_REMINDER_TAG,
    _TOOL_NAMES,
    _canonical_tool,
    _clean_string,
    _fallback_content,
    _file_part_path,
    _file_path,
    _input_paths,
    _iso_timestamp,
    _json_object,
    _message_cwd,
    _message_model,
    _message_record,
    _message_timestamp,
    _milliseconds,
    _model_from_json,
    _model_from_object,
    _namespaced,
    _normalize,
    _part_blocks,
    _title_record,
    _tool_blocks,
)
from ssgrep.utilities import paths
from ssgrep.utilities.types import SessionFile


def _database_path() -> Path:
    override = os.environ.get("SSGREP_OPENCODE_DB")
    if override:
        return Path(os.path.expanduser(override))
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(os.path.expanduser(data_home)) if data_home else Path.home() / ".local" / "share"
    return base / "opencode" / "opencode.db"


@contextmanager
def _snapshot(database: Path) -> Iterator[sqlite3.Connection]:
    """Open a read-only connection and hold one SQLite snapshot until close."""
    uri = f"{database.absolute().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=1.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    # ``table`` is always one of the three hard-coded transcript table names.
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _schema(connection: sqlite3.Connection) -> dict[str, set[str]] | None:
    schema = {name: _columns(connection, name) for name in ("session", "message", "part")}
    required = {
        "session": {"id"},
        "message": {"id", "session_id", "data"},
        "part": {"id", "message_id", "session_id", "data"},
    }
    if any(not required[name].issubset(schema[name]) for name in required):
        return None
    return schema


def _selected(columns: set[str], name: str, *, table: str = "s") -> str:
    return f'{table}."{name}"' if name in columns else "NULL"


def _time_value(columns: set[str], *, table: str) -> str:
    available = [
        f'{table}."{name}"' for name in ("time_updated", "time_created") if name in columns
    ]
    if not available:
        return "0"
    return f"COALESCE({', '.join(available)}, 0)"


def _latest_message_data(message_columns: set[str]) -> str:
    order = [f'mm."{name}" DESC' for name in ("time_created", "id") if name in message_columns]
    return (
        '(SELECT mm."data" FROM "message" AS mm '
        'WHERE mm."session_id" = s."id" '
        f"ORDER BY {', '.join(order)} LIMIT 1)"
    )


def _discovery_query(schema: dict[str, set[str]]) -> str:
    session_columns = schema["session"]
    message_columns = schema["message"]
    part_columns = schema["part"]
    fields = {
        name: _selected(session_columns, name)
        for name in (
            "parent_id",
            "project_id",
            "directory",
            "title",
            "version",
            "time_created",
            "model",
            "agent",
        )
    }
    return f"""SELECT
        s."id" AS id,
        {fields["parent_id"]} AS parent_id,
        {fields["project_id"]} AS project_id,
        {fields["directory"]} AS directory,
        {fields["title"]} AS title,
        {fields["version"]} AS version,
        {fields["time_created"]} AS time_created,
        {fields["model"]} AS model,
        {fields["agent"]} AS agent,
        {_time_value(session_columns, table="s")} AS session_updated,
        (SELECT COUNT(*) FROM "message" AS m WHERE m."session_id" = s."id") AS message_count,
        (SELECT MAX({_time_value(message_columns, table="m")})
           FROM "message" AS m WHERE m."session_id" = s."id") AS message_updated,
        (SELECT COUNT(*) FROM "part" AS p WHERE p."session_id" = s."id") AS part_count,
        (SELECT MAX({_time_value(part_columns, table="p")})
           FROM "part" AS p WHERE p."session_id" = s."id") AS part_updated,
        {_latest_message_data(message_columns)} AS latest_message_data
    FROM "session" AS s
    ORDER BY s."id"
    """


def _fingerprint(row: sqlite3.Row) -> SourceFingerprint:
    counts = (int(row["message_count"] or 0), int(row["part_count"] or 0))
    updates = tuple(
        _milliseconds(row[name]) for name in ("session_updated", "message_updated", "part_updated")
    )
    payload = json.dumps(
        {"id": str(row["id"]), "counts": counts, "updates": updates},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    newest = max(updates, default=0.0)
    return SourceFingerprint(
        size=sum(counts),
        mtime=newest / 1000.0,
        digest=hashlib.sha256(payload).hexdigest(),
    )


def _raw_session_id(source: TranscriptSource) -> str | None:
    identifier = source.session.session_id
    if not identifier.startswith(_NAMESPACE):
        return None
    raw = identifier[len(_NAMESPACE) :]
    return raw or None


def _row_model(row: sqlite3.Row) -> str | None:
    model = _model_from_json(row["model"])
    if model:
        return model
    latest, _ = _json_object(row["latest_message_data"])
    return _message_model(latest) if latest else None


def _source_from_row(database: Path, row: sqlite3.Row) -> TranscriptSource | None:
    raw_id = _clean_string(row["id"])
    if raw_id is None:
        return None
    parent = _clean_string(row["parent_id"])
    cwd = _clean_string(row["directory"])
    project_id = _clean_string(row["project_id"])
    agent = _clean_string(row["agent"])
    session_id = _namespaced(raw_id)
    session = SessionFile(
        path=database,
        session_id=session_id,
        is_main=parent is None,
        parent_session_id=_namespaced(parent) if parent else None,
        agent_name=agent,
        agent_model=_row_model(row),
        project_paths=(cwd,) if cwd else (),
        source_project=project_id,
        runtime="opencode",
    )
    return TranscriptSource(
        adapter="opencode",
        key=session_id,
        session=session,
        fingerprint=_fingerprint(row),
    )


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


class OpenCodeAdapter:
    """Discover and normalize the current OpenCode SQLite storage format."""

    name = "opencode"

    def discover(
        self, *, scope: str | None = None, no_subagents: bool = False
    ) -> list[TranscriptSource]:
        database = _database_path()
        if not _is_file(database):
            return []
        try:
            with _snapshot(database) as connection:
                schema = _schema(connection)
                if schema is None:
                    return []
                rows = connection.execute(_discovery_query(schema)).fetchall()
        except (OSError, sqlite3.Error):
            return []

        sources: list[TranscriptSource] = []
        seen: set[str] = set()
        for row in rows:
            source = _source_from_row(database, row)
            if source is None or source.key in seen:
                continue
            if no_subagents and not source.session.is_main:
                continue
            if scope and not any(
                paths.is_at_or_beneath(cwd, scope) for cwd in source.session.project_paths
            ):
                continue
            seen.add(source.key)
            sources.append(source)
        return sources

    def read(self, source: TranscriptSource) -> ReadResult:
        raw_session_id = _raw_session_id(source)
        database = source.session.path
        if not _is_file(database):
            # Matches the other adapters' bare ``path.open()`` failure exactly
            # (same exception type, same ``.filename``), so a deleted OpenCode
            # database reaches the same archive-recovery path in
            # ``process_source`` instead of silently reconciling rows to empty.
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(database))
        if raw_session_id is None:
            return ReadResult(())
        try:
            with _snapshot(database) as connection:
                schema = _schema(connection)
                if schema is None:
                    return ReadResult(())
                session_row = _read_session(connection, schema["session"], raw_session_id)
                if session_row is None:
                    return ReadResult(())
                messages = _read_messages(connection, schema["message"], raw_session_id)
                parts = _read_parts(connection, schema["part"], raw_session_id)
        except (OSError, sqlite3.Error):
            return ReadResult(())
        return _normalize(session_row, messages, parts, source.session.session_id)


def _read_session(
    connection: sqlite3.Connection, columns: set[str], session_id: str
) -> sqlite3.Row | None:
    selected = [
        f"{_selected(columns, name)} AS {name}"
        for name in ("id", "directory", "title", "time_created", "model")
    ]
    query = f'SELECT {", ".join(selected)} FROM "session" AS s WHERE s."id" = ?'
    return connection.execute(query, (session_id,)).fetchone()


def _read_messages(
    connection: sqlite3.Connection, columns: set[str], session_id: str
) -> list[sqlite3.Row]:
    selected = [
        f"{_selected(columns, name, table='m')} AS {name}"
        for name in ("id", "time_created", "data")
    ]
    order = [f'm."{name}"' for name in ("time_created", "id") if name in columns]
    query = (
        f'SELECT {", ".join(selected)} FROM "message" AS m WHERE m."session_id" = ? '
        f"ORDER BY {', '.join(order)}"
    )
    return connection.execute(query, (session_id,)).fetchall()


def _read_parts(
    connection: sqlite3.Connection, columns: set[str], session_id: str
) -> list[sqlite3.Row]:
    selected = [
        f"{_selected(columns, name, table='p')} AS {name}"
        for name in ("id", "message_id", "time_created", "data")
    ]
    order = [f'p."{name}"' for name in ("time_created", "id") if name in columns]
    query = (
        f'SELECT {", ".join(selected)} FROM "part" AS p WHERE p."session_id" = ? '
        f"ORDER BY {', '.join(order)}"
    )
    return connection.execute(query, (session_id,)).fetchall()
