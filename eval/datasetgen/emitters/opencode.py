"""OpenCode SQLite emitter: ``sessions.parquet`` rows -> ``opencode.db``.

Expected ``sessions.parquet`` schema (T6 output; columns are read
defensively, missing optional columns are skipped):

- ``runtime`` (str) — rows with ``runtime == "opencode"`` are emitted.
- ``episodes`` (list) — one entry per episode, either a ``(prompt, response)``
  pair or a ``{"prompt": ..., "response": ...}`` dict. Each pair becomes one
  user/assistant message pair, i.e. one episode after segmentation.
- ``title`` (str, optional) — session title; falls back to the first prompt
  truncated to 100 chars.
- ``project`` (str, optional) — fictional project name; becomes the session
  ``directory`` (``/fictional/<project>``) and the per-message cwd.
- ``files_touched`` (list[str], optional) — one ``tool`` part per path on the
  assistant message, so ``files_touched``/``tool_names`` metadata survives
  ingestion (``metadata.py:103-133``).
- ``tool_names`` (list[str], optional) — tool name for those parts; defaults
  to ``read``.
- ``session_id`` (str, optional) — stable id; generated when absent.

Emitted database (matches the OpenCodeAdapter's read-only snapshot queries,
``src/ssgrep/sessions/adapters/opencode.py:81-90``):

- ``session(id TEXT PRIMARY KEY, directory TEXT, title TEXT, time_created INTEGER, model TEXT)``
- ``message(id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER)``
- ``part(id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT, data TEXT, time_created INTEGER)``

``message.data``/``part.data`` hold JSON strings; roles are only ``user`` and
``assistant`` (``opencode.py:591-594``); part types are only ``text`` and
``tool`` (``opencode.py:475-501``). Timestamps are deterministic integer
milliseconds derived from the row index (never the wall clock), strictly
increasing within a session so message/part ordering matches intent
(``opencode.py:357-384``) and the fingerprint stays stable across incremental
re-ingestion (``opencode.py:213-228``).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

#: Fixed epoch (2023-11-14T13:33:20Z) so timestamps never depend on the wall
#: clock; per-session offsets keep every value deterministic and increasing.
_BASE_MS = 1_700_000_000_000
_SESSION_STEP_MS = 3_600_000
_MESSAGE_STEP_MS = 60_000
_PART_STEP_MS = 1_000

_DEFAULT_MODEL = "synthetic-oss-120b"

_SCHEMA_SQL = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    directory TEXT,
    title TEXT,
    time_created INTEGER,
    model TEXT
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    data TEXT,
    time_created INTEGER
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    data TEXT,
    time_created INTEGER
);
"""


def _clean(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _string_list(value: object) -> list[str]:
    """Normalize a files/tools column: list of strings, or empty when absent.

    Parquet missing values surface as ``None`` or ``nan``; both are treated as
    an empty list rather than an error.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def _normalize_episodes(episodes: object) -> list[tuple[str, str]]:
    if not isinstance(episodes, list):
        raise ValueError(f"episodes must be a list, got {type(episodes).__name__}")
    normalized: list[tuple[str, str]] = []
    for item in episodes:
        if isinstance(item, Mapping):
            prompt = item.get("prompt")
            response = item.get("response")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            prompt, response = item
        else:
            raise ValueError(
                "each episode must be a (prompt, response) pair or a "
                f'{{"prompt", "response"}} dict, got {item!r}'
            )
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("episode prompt and response must be strings")
        normalized.append((prompt, response))
    return normalized


def _session_id(session: Mapping[str, Any], index: int) -> str:
    return _clean(session.get("session_id")) or f"sess-{index:04d}"


def _session_title(session: Mapping[str, Any], index: int) -> str:
    title = _clean(session.get("title"))
    if title:
        return title
    episodes = _normalize_episodes(session.get("episodes"))
    if episodes:
        return episodes[0][0][:100]
    return f"Session {index}"


def _session_cwd(session: Mapping[str, Any], index: int) -> str:
    project = _clean(session.get("project"))
    if project:
        return f"/fictional/{project}"
    return f"/fictional/project-{index:02d}"


def _message_data(role: str, cwd: str, created_ms: int, model: str | None) -> str:
    data: dict[str, Any] = {"role": role, "time": {"created": created_ms}, "path": {"cwd": cwd}}
    if model:
        data["model"] = model
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _part_data(kind: str, payload: Any) -> str:
    if kind == "text":
        return json.dumps({"type": "text", "text": payload}, sort_keys=True, separators=(",", ":"))
    tool, path = payload
    return json.dumps(
        {"type": "tool", "tool": tool, "state": {"input": {"file_path": path}}},
        sort_keys=True,
        separators=(",", ":"),
    )


def _insert_session(connection: sqlite3.Connection, session: Mapping[str, Any], index: int) -> None:
    session_id = _session_id(session, index)
    cwd = _session_cwd(session, index)
    title = _session_title(session, index)
    model = _clean(session.get("model")) or _DEFAULT_MODEL
    episodes = _normalize_episodes(session.get("episodes"))
    if not episodes:
        raise ValueError(f"session {session_id!r} has no episodes")
    files = _string_list(session.get("files_touched"))
    tools = _string_list(session.get("tool_names"))
    tool = tools[0] if tools else "read"

    base_ms = _BASE_MS + index * _SESSION_STEP_MS
    connection.execute(
        "INSERT INTO session (id, directory, title, time_created, model) VALUES (?, ?, ?, ?, ?)",
        (session_id, cwd, title, base_ms, model),
    )

    message_seq = 0
    for prompt, response in episodes:
        for role, text in (("user", prompt), ("assistant", response)):
            message_ms = base_ms + message_seq * _MESSAGE_STEP_MS
            message_id = f"{session_id}-m{message_seq:03d}"
            connection.execute(
                "INSERT INTO message (id, session_id, data, time_created) VALUES (?, ?, ?, ?)",
                (message_id, session_id, _message_data(role, cwd, message_ms, model), message_ms),
            )
            parts: list[tuple[str, Any]] = [("text", text)]
            if role == "assistant":
                parts.extend(("tool", (tool, path)) for path in files)
            for part_index, part in enumerate(parts):
                part_ms = message_ms + (part_index + 1) * _PART_STEP_MS
                part_id = f"{message_id}-p{part_index:02d}"
                connection.execute(
                    "INSERT INTO part (id, message_id, session_id, data, time_created) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (part_id, message_id, session_id, _part_data(*part), part_ms),
                )
            message_seq += 1


def emit_sessions(sessions: list[dict[str, Any]], output: Path) -> None:
    """Write ``sessions`` into a fresh ``opencode.db`` at ``output``.

    Deterministic: row insertion order follows list order, timestamps derive
    from the row index, and every JSON payload is serialized with sorted keys,
    so identical input produces a byte-identical database.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    try:
        connection.executescript(_SCHEMA_SQL)
        for index, session in enumerate(sessions):
            _insert_session(connection, session, index)
        connection.commit()
    finally:
        connection.close()


def emit_parquet(parquet_path: Path, output: Path) -> int:
    """Emit the ``runtime == "opencode"`` rows of ``sessions.parquet``.

    Returns the number of sessions emitted.
    """
    rows = pq.read_table(parquet_path).to_pylist()
    sessions = [row for row in rows if row.get("runtime") == "opencode"]
    emit_sessions(sessions, output)
    return len(sessions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit opencode sessions from sessions.parquet into opencode.db"
    )
    parser.add_argument("--parquet", required=True, type=Path, help="sessions.parquet input")
    parser.add_argument("--out", required=True, type=Path, help="opencode.db output path")
    args = parser.parse_args(argv)
    count = emit_parquet(args.parquet, args.out)
    print(f"emitted {count} opencode sessions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
