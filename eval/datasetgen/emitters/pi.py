"""Pi session emitter: ``sessions.parquet`` rows -> Pi JSONL transcripts.

Converts ``eval/datasetgen/sessions.parquet`` (Task 6 output) rows where
``runtime == "pi"`` into the append-only JSONL format the real ``PiAdapter``
parses (``src/ssgrep/sessions/adapters/pi.py``), one file per session, written
to a directory that the indexer consumes via ``SSGREP_PI_SESSIONS_DIR``
(``pi.py:364``). Every record shape here is pinned to the T4 contract in
``eval/datasetgen/adapter_formats.md`` section 6, never guessed.

Expected parquet schema (documented contract for T6; both tasks ran in
parallel, so the emitter reads only the documented columns and silently
ignores any others):

    runtime        str          -- must equal "pi" for emission
    scenario_class str          -- D5 class label; carried for provenance only
    episodes       list[dict]   # [{"prompt": str, "response": str}, ...]
    title          str          # becomes the custom-title record
    project        str          # fictional project root, becomes cwd
    files_touched  list[str]    # optional; routed to assistant toolCall blocks
    tool_names     list[str]    # optional; routed to assistant toolCall blocks
    difficulty / episode_length_bucket: str  # optional; not emitted

Ruled out by construction (so indexed content matches the parser's intent):
no ``thinking`` blocks, no ``image`` blocks, no ``toolResult``-role messages,
and no entry types beyond ``session`` / ``custom-title`` / ``message``
(pi.py:38-51, pi.py:158-188, pi.py:238-239). The parser drops those silently,
so emitting them would only waste corpus bytes -- it also means emitting them
is safe, but this emitter does not.

The writer is fully deterministic: stable field order, timestamps derived from
the session index over a fixed seed base (never wall-clock), and session ids
``pi-<n:04d>`` derived from the sorted ``runtime == "pi"`` rows, so two runs
over the same parquet produce byte-identical trees.

Session ids in the index take the ``pi:<raw-id>`` shape (pi.py:99-100, 128-134),
and episodes are ``<session_id>:ep:<n>`` (episodes.py:8-9) -- the reference
scheme the T12 query generator must use.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

#: Synthetic (fictional) persona model stamped on the session record; the T4
#: contract's minimal example uses this exact value.
MODEL_ID = "claude-sonnet-4-5"

#: Fixed epoch so every emitted timestamp is content-derived, never wall-clock.
_BASE_TIME = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Content-derived sort key so ordering and ids survive parquet reorders."""
    return (
        str(row.get("project", "")),
        str(row.get("title", "")),
        json.dumps(row.get("episodes", []), sort_keys=True),
    )


def _timestamp(session_index: int, offset_minutes: int) -> str:
    return (_BASE_TIME + timedelta(hours=session_index, minutes=offset_minutes)).isoformat()


def _tool_blocks(
    episode_index: int, tool_names: list[str], files_touched: list[str], cwd: str
) -> list[dict[str, Any]]:
    """Assistant ``toolCall`` blocks (adapter maps them to ``tool_use``).

    pi.py:163-184 only accepts ``toolCall`` blocks on assistant messages, so
    tool metadata must ride these blocks. Session-level lists are distributed
    round-robin over the episodes to stay deterministic.
    """
    if not tool_names:
        return []
    name = tool_names[episode_index % len(tool_names)]
    arguments: dict[str, str] = {}
    if files_touched:
        path = files_touched[episode_index % len(files_touched)]
        arguments["file_path"] = path if path.startswith("/") else f"{cwd}/{path}"
    return [
        {
            "type": "toolCall",
            "name": name,
            "arguments": arguments,
            "id": f"tool-ep{episode_index}",
        }
    ]


def _assistant_content(
    response: str, episode_index: int, tool_names: list[str], files_touched: list[str], cwd: str
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": response}]
    blocks.extend(_tool_blocks(episode_index, tool_names, files_touched, cwd))
    return blocks


def _session_records(row: dict[str, Any], raw_id: str, session_index: int) -> list[dict[str, Any]]:
    episodes = row.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError(f"session {raw_id}: 'episodes' must be a list")
    tool_names = [str(item) for item in (row.get("tool_names") or [])]
    files_touched = [str(item) for item in (row.get("files_touched") or [])]
    cwd = str(row.get("project") or "")

    records: list[dict[str, Any]] = [
        {
            "type": "session",
            "id": raw_id,
            "cwd": cwd,
            "modelId": MODEL_ID,
            "timestamp": _timestamp(session_index, 0),
        }
    ]
    title = row.get("title")
    if title is not None and str(title):
        records.append(
            {"type": "custom-title", "name": str(title), "timestamp": _timestamp(session_index, 0)}
        )

    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict) or not isinstance(episode.get("prompt"), str):
            raise ValueError(f"session {raw_id}: episode {index} needs a 'prompt' string")
        prompt = episode["prompt"]
        response = episode.get("response", "")
        if not isinstance(response, str):
            raise ValueError(f"session {raw_id}: episode {index} 'response' must be a string")
        user_stamp = _timestamp(session_index, index * 2 + 1)
        assistant_stamp = _timestamp(session_index, index * 2 + 2)
        records.append(
            {
                "type": "message",
                "id": f"m-ep{index}-user",
                "timestamp": user_stamp,
                "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
            }
        )
        records.append(
            {
                "type": "message",
                "id": f"m-ep{index}-assistant",
                "timestamp": assistant_stamp,
                "message": {
                    "role": "assistant",
                    "content": _assistant_content(response, index, tool_names, files_touched, cwd),
                },
            }
        )
    return records


def emit_sessions(parquet_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Write one Pi JSONL file per ``runtime == 'pi'`` parquet row.

    Files are written to ``out_dir`` (created if absent), each named
    ``<raw_session_id>.jsonl``. Returns the written paths sorted by absolute
    path. Raises ``ValueError`` for a parquet missing the ``runtime`` column or
    for a repeated session id.
    """
    source = Path(parquet_path)
    table = pq.read_table(source)
    if "runtime" not in table.column_names:
        raise ValueError(f"{source}: missing required 'runtime' column")

    rows = [row for row in table.to_pylist() if row.get("runtime") == "pi"]
    rows.sort(key=_row_key)

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        raw_id = f"pi-{index:04d}"
        if raw_id in seen:
            raise ValueError(f"duplicate derived session id {raw_id!r} in {source}")
        seen.add(raw_id)
        records = _session_records(row, raw_id, index)
        path = destination / f"{raw_id}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        written.append(path)
    written.sort(key=lambda item: str(item.absolute()))
    return written
