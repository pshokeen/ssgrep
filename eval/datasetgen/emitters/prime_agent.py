"""Prime-agent session emitter for the synthetic benchmark.

Converts ``sessions.parquet`` rows whose ``runtime == "prime-agent"`` into the
native Prime Agent on-disk layout the real adapter ingests unchanged
(``src/ssgrep/sessions/adapters/prime_agent.py``, ``PrimeAgentAdapter``):

- ``<out>/sessions/<session_id>.jsonl`` — main sessions
- ``<out>/session-artifacts/<session_id>.jsonl`` — child artifacts

``session-artifacts/`` is always the *sibling* of the sessions root
(``root.parent / "session-artifacts"``), so pointing
``SSGREP_PRIME_AGENT_SESSIONS_DIR`` at ``<out>/sessions`` makes the adapter
discover both roots. Every file under ``session-artifacts/`` is forced
``is_main=False`` by location regardless of ``rlmDepth``;
the episodes index with ``is_subagent=true`` (``pipeline/episodes.py:128``).

Expected input schema (``sessions.parquet``, plan T6 output). Rows are a
pandas DataFrame or a sequence of mappings. Mandatory columns:

- ``runtime`` — ``str``; only rows equal to ``"prime-agent"`` are emitted
  (rows without the column are treated as prime-agent, since this emitter is
  runtime-specific by construction).
- ``session_id`` — ``str``; the raw id. The index id becomes
  ``prime-agent:<session_id>`` (``pi.py:99-100``).
- ``episodes`` — ``list`` of ``{"prompt": str, "response": str}`` dicts or
  ``(prompt, response)`` pairs. Each text-bearing prompt becomes one episode
  (``episodes.py:81-87``).
- ``title`` — ``str``; emitted as a ``custom-title``/``name`` record before
  the first user message so the title is set before every episode.
- ``project`` (or ``cwd``) — ``str``; absolute fictional project path used as
  the session ``cwd`` (``cwd`` wins when both are present).

Optional columns:

- ``model`` — ``str``; session ``modelId`` and assistant message ``model``.
- ``git_branch`` — ``str``; emitted as a ``git`` record.
- ``is_artifact`` (or ``is_subagent``) — ``bool``; when true the session is
  written under ``session-artifacts/`` and indexes with ``is_main=False``.
- ``parent_session_id`` — ``str``; raw parent id on the child ``session``
  header (namespaced to ``prime-agent:<parent>`` by the adapter).
- ``rlmDepth`` — ``int``; carried on the ``session`` header.
- ``files_touched`` / ``tool_names`` — ``list[str]`` or newline-joined
  ``str``; emitted as ``toolCall`` blocks on assistant messages (one per
  episode, cycled) so ``files_touched``/``tool_names`` metadata survives.
- ``scenario_class``, ``difficulty``, ``episode_length_bucket`` — ignored
  metadata columns; accepted for schema compatibility with T6.

The writer is deterministic: timestamps derive from a fixed epoch plus a
per-record offset (never wall-clock), JSON keys are emitted sorted, and the
same input rows produce byte-identical files across runs. Every file ends
with a newline (unterminated lines are treated as concurrent appends and
skipped, ``pi.py:63-68``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

#: Subdirectory holding main sessions; the adapter's sessions root.
MAIN_DIR = "sessions"
#: Sibling subdirectory holding child artifacts (forced ``is_main=False``).
ARTIFACTS_DIR = "session-artifacts"

#: Fixed epoch so emitted timestamps are deterministic across runs.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _clean(value: Any) -> Any:
    """Map pandas NaN to None so downstream ``isinstance`` checks stay honest."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_rows(rows: Any) -> list[dict]:
    """Accept a pandas DataFrame or a sequence of mappings."""
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict(orient="records")
    return [{key: _clean(value) for key, value in dict(row).items()} for row in rows]


def _optional_str(row: Mapping[str, Any], key: str) -> str | None:
    value = _clean(row.get(key))
    return value if isinstance(value, str) and value else None


def _as_list(value: Any) -> list[str]:
    """Normalize a list, newline-joined string, or None into a string list."""
    value = _clean(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split("\n") if part]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def _episode_pairs(episodes: Any) -> list[tuple[str, str]]:
    """Normalize the ``episodes`` column to ``(prompt, response)`` pairs."""
    episodes = _clean(episodes)
    pairs: list[tuple[str, str]] = []
    for item in episodes or []:
        if isinstance(item, Mapping):
            prompt = item.get("prompt", "")
            response = item.get("response", "")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            prompt, response = item[0], item[1]
        else:
            raise ValueError(f"unsupported episode shape: {item!r}")
        pairs.append((str(prompt), str(response)))
    return pairs


def _timestamp(offset_seconds: int) -> str:
    return (_EPOCH + timedelta(seconds=offset_seconds)).isoformat()


def _cwd(row: Mapping[str, Any]) -> str:
    cwd = _optional_str(row, "cwd")
    if cwd:
        return cwd
    project = _optional_str(row, "project")
    if project:
        return project
    return "/fictional/work"


def _session_header(row: Mapping[str, Any]) -> dict:
    header: dict[str, Any] = {
        "type": "session",
        "id": str(row["session_id"]),
        "cwd": _cwd(row),
        "timestamp": _timestamp(0),
    }
    model = _optional_str(row, "model")
    if model:
        header["modelId"] = model
    depth = _clean(row.get("rlmDepth"))
    if isinstance(depth, int) and not isinstance(depth, bool):
        header["rlmDepth"] = depth
    parent = _optional_str(row, "parent_session_id")
    if parent:
        header["parentSessionId"] = parent
    return header


def _tool_blocks(row: Mapping[str, Any], episode_index: int) -> list[dict]:
    """One ``toolCall`` block per episode, cycling through the row's tools."""
    tools = _as_list(row.get("tool_names"))
    if not tools:
        return []
    files = _as_list(row.get("files_touched"))
    arguments: dict[str, Any] = {}
    if files:
        arguments["file_path"] = files[episode_index % len(files)]
    return [
        {
            "type": "toolCall",
            "name": tools[episode_index % len(tools)],
            "arguments": arguments,
            "id": f"t-{episode_index}",
        }
    ]


def _message_record(
    row: Mapping[str, Any],
    role: str,
    text: str,
    offset: int,
    *,
    tool_blocks: list[dict] | None = None,
) -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    if tool_blocks:
        content.extend(tool_blocks)
    return {
        "type": "message",
        "id": f"{row['session_id']}-{role}-{offset}",
        "timestamp": _timestamp(offset),
        "message": {"role": role, "content": content},
    }


def _session_records(row: Mapping[str, Any]) -> list[dict]:
    """Build the deterministic record stream for one session file."""
    records: list[dict] = [_session_header(row)]
    branch = _optional_str(row, "git_branch")
    if branch:
        records.append({"type": "git", "git": {"branch": branch}})
    title = _optional_str(row, "title")
    if title:
        records.append(
            {
                "type": "custom-title",
                "id": f"{row['session_id']}-title",
                "name": title,
                "timestamp": _timestamp(2),
            }
        )
    offset = 10
    for index, (prompt, response) in enumerate(_episode_pairs(row.get("episodes"))):
        records.append(_message_record(row, "user", prompt, offset))
        offset += 1
        records.append(
            _message_record(
                row,
                "assistant",
                response,
                offset,
                tool_blocks=_tool_blocks(row, index),
            )
        )
        offset += 1
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n")


def emit(rows: Any, out_dir: Path) -> dict[str, int]:
    """Write prime-agent sessions and artifacts beneath ``out_dir``.

    Layout: ``<out_dir>/sessions/`` (main sessions) and
    ``<out_dir>/session-artifacts/`` (children). Point
    ``SSGREP_PRIME_AGENT_SESSIONS_DIR`` at ``<out_dir>/sessions`` to ingest
    both roots through the real adapter.

    Returns a report dict ``{"main_sessions": int, "artifact_sessions": int}``.
    """
    out_dir = Path(out_dir)
    main_dir = out_dir / MAIN_DIR
    artifacts_dir = out_dir / ARTIFACTS_DIR
    main_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    main_written = 0
    artifact_written = 0
    for row in _normalize_rows(rows):
        runtime = _clean(row.get("runtime"))
        if runtime is not None and runtime != "prime-agent":
            continue
        is_artifact = bool(_clean(row.get("is_artifact")) or _clean(row.get("is_subagent")))
        target = artifacts_dir if is_artifact else main_dir
        _write_jsonl(target / f"{row['session_id']}.jsonl", _session_records(row))
        if is_artifact:
            artifact_written += 1
        else:
            main_written += 1
    return {"main_sessions": main_written, "artifact_sessions": artifact_written}


__all__ = ["ARTIFACTS_DIR", "MAIN_DIR", "emit"]
