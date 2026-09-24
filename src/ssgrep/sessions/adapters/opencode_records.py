"""Normalize OpenCode message and part rows into Claude-shaped records.

Pure functions over decoded SQLite rows: no database access, no discovery.
`opencode.py` owns discovery and row reading and calls `_normalize` here.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote, urlsplit

from ssgrep.sessions import metadata, records
from ssgrep.sessions.adapters.base import ReadResult

_NAMESPACE = "opencode:"
_IGNORED_PART_TYPES = {
    "agent",
    "compaction",
    "reasoning",
    "retry",
    "snapshot",
    "step-finish",
    "step-start",
    "subtask",
}
_SYSTEM_REMINDER_TAG = "<system-reminder>"
_TOOL_NAMES = {
    "apply_patch": "Edit",
    "bash": "Bash",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "patch": "Edit",
    "read": "Read",
    "task": "Task",
    "todowrite": "TodoWrite",
    "webfetch": "WebFetch",
    "websearch": "WebSearch",
    "write": "Write",
}
_MODEL_TEXT = re.compile(r"[A-Za-z0-9_.:/@+-]{1,200}")


def _json_object(raw: object) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "malformed"
    if not isinstance(raw, str):
        return None, "skipped"
    if len(raw.encode("utf-8")) > records.MAX_LINE_BYTES:
        return None, "skipped"
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, "malformed"
    if not isinstance(value, dict):
        return None, "skipped"
    return value, None


def _model_from_object(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("modelID", "id", "model"):
            model = value.get(key)
            if isinstance(model, str) and model.strip():
                return model.strip()[:200]
    if isinstance(value, str) and _MODEL_TEXT.fullmatch(value.strip()):
        return value.strip()
    return None


def _model_from_json(raw: object) -> str | None:
    if not isinstance(raw, (str, bytes)):
        return _model_from_object(raw)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        value = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
    return _model_from_object(value)


def _message_model(data: Mapping[str, Any]) -> str | None:
    return (
        _model_from_object(data.get("model"))
        or _model_from_object(data.get("modelID"))
        or _model_from_object(data.get("model_id"))
    )


def _milliseconds(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _namespaced(identifier: str) -> str:
    return f"{_NAMESPACE}{identifier}"


def _clean_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _iso_timestamp(value: object) -> str | None:
    if isinstance(value, str) and not value.strip().replace(".", "", 1).isdigit():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat()
    milliseconds = _milliseconds(value)
    if not milliseconds:
        return None
    seconds = milliseconds / 1000.0 if abs(milliseconds) >= 10_000_000_000 else milliseconds
    try:
        return datetime.fromtimestamp(seconds, UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _message_timestamp(data: Mapping[str, Any], fallback: object) -> str | None:
    timing = data.get("time")
    created = timing.get("created") if isinstance(timing, Mapping) else None
    return _iso_timestamp(created) or _iso_timestamp(fallback)


def _message_cwd(data: Mapping[str, Any], fallback: str | None) -> str | None:
    message_path = data.get("path")
    if isinstance(message_path, Mapping):
        cwd = _clean_string(message_path.get("cwd"))
        if cwd:
            return cwd
    return fallback


def _canonical_tool(name: object) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    clean = name.strip()
    return _TOOL_NAMES.get(clean.lower(), clean[:200])


def _file_path(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _input_paths(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    found: list[str] = []
    for key in ("file_path", "filePath", "path", "paths", "filename"):
        candidate = value.get(key)
        candidates = candidate if isinstance(candidate, list) else [candidate]
        for item in candidates:
            path = _file_path(item)
            if path and path not in found:
                found.append(path)
    return tuple(found)


def _tool_blocks(part: Mapping[str, Any]) -> list[dict[str, Any]]:
    name = _canonical_tool(part.get("tool") or part.get("name"))
    if name is None:
        return []
    state = part.get("state")
    tool_input = state.get("input") if isinstance(state, Mapping) else part.get("input")
    file_paths = _input_paths(tool_input)
    if not file_paths:
        return [{"type": "tool_use", "name": name, "input": {}}]
    return [
        {"type": "tool_use", "name": name, "input": {"file_path": file_path}}
        for file_path in file_paths
    ]


def _file_part_path(part: Mapping[str, Any]) -> str | None:
    source = part.get("source")
    if isinstance(source, Mapping):
        path = _file_path(source.get("path"))
        if path:
            return path
    url = part.get("url")
    if isinstance(url, str):
        parsed = urlsplit(url)
        if parsed.scheme == "file" and parsed.path:
            return unquote(parsed.path)
    return _file_path(part.get("filename"))


def _part_blocks(part: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    # opencode marks harness-injected parts (tool-call echoes, attached
    # context) "synthetic" on any part type: never transcript content.
    if part.get("synthetic"):
        return [], False
    part_type = part.get("type")
    if part_type == "text":
        if part.get("ignored"):
            return [], False
        text = part.get("text")
        if not isinstance(text, str):
            return [], True
        # Reminder blocks appended after real user text stay in place;
        # only reminder-first parts are notifications, not prompts.
        if text.startswith(_SYSTEM_REMINDER_TAG):
            return [], False
        return ([{"type": "text", "text": text}] if text else []), False
    if part_type == "tool":
        return _tool_blocks(part), False
    if part_type == "file":
        path = _file_part_path(part)
        block = {"type": "tool_use", "name": "Read", "input": {"file_path": path}}
        return ([block] if path else []), path is None
    if part_type == "patch":
        files = part.get("files")
        if not isinstance(files, list):
            return [], True
        blocks = [
            {"type": "tool_use", "name": "Edit", "input": {"file_path": path}}
            for item in files
            if (path := _file_path(item))
        ]
        return blocks, len(blocks) != len(files)
    if part_type in _IGNORED_PART_TYPES:
        return [], False
    return [], True


def _fallback_content(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = data.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    return [
        {"type": "text", "text": block["text"]}
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"]
    ]


def _title_record(
    title: str, session_id: str, cwd: str | None, timestamp: str | None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "custom-title",
        "custom-title": title,
        "sessionId": session_id,
    }
    if cwd:
        record["cwd"] = cwd
    if timestamp:
        record["timestamp"] = timestamp
    return record


def _message_record(
    *,
    role: str,
    message_id: str,
    session_id: str,
    data: Mapping[str, Any],
    content: list[dict[str, Any]],
    cwd: str | None,
    timestamp: str | None,
    model: str | None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "id": _namespaced(message_id),
        "role": role,
        "content": content,
    }
    if model:
        message["model"] = model
    record: dict[str, Any] = {
        "type": role,
        "uuid": _namespaced(message_id),
        "sessionId": session_id,
        "message": message,
    }
    parent = _clean_string(data.get("parentID") or data.get("parent_id"))
    if parent:
        record["parentUuid"] = _namespaced(parent)
    if cwd:
        record["cwd"] = cwd
    if timestamp:
        record["timestamp"] = timestamp
    return record


def _normalize(
    session_row: sqlite3.Row,
    message_rows: list[sqlite3.Row],
    part_rows: list[sqlite3.Row],
    session_id: str,
) -> ReadResult:
    malformed = 0
    skipped = 0
    decoded_messages: list[tuple[sqlite3.Row, dict[str, Any]]] = []
    message_ids: set[str] = set()
    for row in message_rows:
        message_id = _clean_string(row["id"])
        data, problem = _json_object(row["data"])
        if problem == "malformed":
            malformed += 1
        elif problem:
            skipped += 1
        if message_id is None or data is None:
            if message_id is None and data is not None:
                skipped += 1
            continue
        role = data.get("role")
        if role not in ("user", "assistant"):
            skipped += 1
            continue
        decoded_messages.append((row, data))
        message_ids.add(message_id)

    blocks_by_message: dict[str, list[dict[str, Any]]] = {item: [] for item in message_ids}
    for row in part_rows:
        message_id = _clean_string(row["message_id"])
        data, problem = _json_object(row["data"])
        if problem == "malformed":
            malformed += 1
        elif problem:
            skipped += 1
        if message_id not in blocks_by_message or data is None:
            if data is not None:
                skipped += 1
            continue
        blocks, invalid = _part_blocks(data)
        skipped += int(invalid)
        blocks_by_message[message_id].extend(blocks)

    cwd = _clean_string(session_row["directory"])
    title_raw = session_row["title"]
    title = metadata.normalize_title(title_raw) if isinstance(title_raw, str) else ""
    session_timestamp = _iso_timestamp(session_row["time_created"])
    session_model = _model_from_json(session_row["model"])
    normalized: list[dict[str, Any]] = []
    if title:
        normalized.append(_title_record(title, session_id, cwd, session_timestamp))
    for row, data in decoded_messages:
        message_id = str(row["id"])
        content = blocks_by_message[message_id] or _fallback_content(data)
        normalized.append(
            _message_record(
                role=str(data["role"]),
                message_id=message_id,
                session_id=session_id,
                data=data,
                content=content,
                cwd=_message_cwd(data, cwd),
                timestamp=_message_timestamp(data, row["time_created"]),
                model=_message_model(data) or session_model,
            )
        )
    return ReadResult(tuple(normalized), malformed_records=malformed, skipped_records=skipped)
