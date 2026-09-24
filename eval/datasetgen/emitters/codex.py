"""Codex rollout-session emitter (``runtime == "codex"``).

Task 8 of the retrieval-eval overhaul. Reads ``sessions.parquet`` (the T6 NeMo
workflow output) and writes one Codex rollout JSONL file per codex row into a
directory that the real ``codex`` adapter ingests unchanged when the harness
points ``SSGREP_CODEX_SESSIONS_DIR`` at it. Every claim here is pinned to the
adapter contract in ``eval/datasetgen/adapter_formats.md`` section 5 / 10.2,
which was derived from ``src/ssgrep/sessions/adapters/codex.py``.

Expected parquet schema (documented contract for T6; T6 and T8 ran in
parallel, so the emitter reads only the documented columns and ignores any
others):

    runtime        str          -- must equal "codex" for emission
    scenario_class str          -- D5 class label; carried for provenance only
    episodes       list[dict]   # [{"prompt": str, "response": str}, ...]
    title          str          # unused: Codex emits no title record, so
                                # indexed titles fall back to first user text
    project        str          # fictional project root, becomes cwd
    files_touched  list[str]    # optional; routed to "read" function_call
    tool_names     list[str]    # optional; routed to function_call records
    difficulty / episode_length_bucket: str  # optional; not emitted

Emission rules (verbatim from the T4 contract):

- One file per session, any layout under the root (discovery is recursive
  ``*.jsonl``, codex.py:296-299); the file starts with a ``session_meta``
  envelope carrying a non-empty ``payload.id`` (codex.py:97-106).
- Every line is an envelope ``{"timestamp", "type", "payload"}`` with strictly
  increasing timestamps so the adapter's ``codex:<seq>:<ts>`` uuids stay
  stable (codex.py:212). Timestamps derive from a fixed epoch and the session
  index, never the wall clock, so output is byte-stable across runs.
- ``turn_context`` repeats at every turn boundary so cwd/model stay tracked
  (codex.py:203-207).
- Message roles are only ``user``/``assistant`` with ``input_text`` /
  ``output_text`` content blocks (codex.py:131-147); developer-role messages
  are intentionally omitted (codex.py:238-239).
- Tool events are ``function_call`` records placed immediately after the
  assistant message they belong to so they append to its blocks
  (codex.py:240-256); files_touched map to a ``read`` call with ``file_path``
  so they surface as ``tool_use``/``files_touched`` metadata.
- ``reasoning``, ``*_output`` echoes, bookkeeping, and ``agent_message``
  records are never emitted (codex.py:257-258). When ``include_excluded`` is
  set (test-only), a bounded deterministic canary set of reasoning /
  ``function_call_output`` / developer-role lines is appended so the round-trip
  test can prove the real adapter drops them from the index; the default
  ``False`` output is benchmark-clean.
- Every file ends with a newline: readers treat unterminated lines as a
  concurrent append and skip them (codex.py:50-58).

Session ids in the index take the ``codex:<raw-id>`` shape (codex.py:82-83)
and episodes are ``<session_id>:ep:<n>`` (episodes.py:8-9); raw ids are
``codex-<nnnn>`` derived from the sorted ``runtime == "codex"`` rows.

Episode folding follows the shared rule: a ``user`` record with empty text
does not split (episodes.py:81-87), so a non-empty prompt maps one turn to
exactly one episode.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

RUNTIME = "codex"

#: Synthetic (fictional) model/provider stamped on every session record.
_MODEL = "gpt-5-codex-fictional"
_PROVIDER = "openai"

#: Fixed epoch so every emitted timestamp is content-derived, never wall-clock.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

_FALLBACK_CWD = "/fictional/checkout"


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Content-derived sort key so ordering and ids survive parquet reorders."""
    return (
        str(row.get("project", "")),
        str(row.get("title", "")),
        json.dumps(row.get("episodes", []), sort_keys=True),
    )


def _episode_pairs(episodes: object) -> list[tuple[str, str]]:
    """Normalize the ``episodes`` column to ``(prompt, response)`` string pairs."""
    pairs: list[tuple[str, str]] = []
    if not isinstance(episodes, list):
        return pairs
    for item in episodes:
        if isinstance(item, dict):
            prompt = item.get("prompt") or item.get("query")
            response = item.get("response") or item.get("answer")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prompt, response = item[0], item[1]
        else:
            continue
        pairs.append((str(prompt or ""), str(response or "")))
    return pairs


def _cwd(row: dict[str, Any]) -> str:
    project = row.get("project")
    return str(project) if isinstance(project, str) and project else _FALLBACK_CWD


def _unique_strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values if value})


class _Clock:
    """Strictly increasing, deterministic timestamps derived from ``_EPOCH``."""

    def __init__(self) -> None:
        self._stamp = _EPOCH

    def stamp(self) -> str:
        self._stamp += timedelta(seconds=1)
        return self._stamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_lines(row: dict[str, Any], raw_id: str, *, include_excluded: bool) -> list[str]:
    """Render one deterministic rollout JSONL file (one line per envelope)."""
    clock = _Clock()
    cwd = _cwd(row)
    lines: list[str] = []

    def envelope(record_type: str, payload: dict[str, Any]) -> str:
        line = {"timestamp": clock.stamp(), "type": record_type, "payload": payload}
        return json.dumps(line, ensure_ascii=False)

    lines.append(
        envelope(
            "session_meta",
            {"id": raw_id, "cwd": cwd, "model_provider": _PROVIDER},
        )
    )
    tool_seq = 0
    for prompt, response in _episode_pairs(row.get("episodes")):
        lines.append(envelope("turn_context", {"cwd": cwd, "model": _MODEL}))
        lines.append(
            envelope(
                "response_item",
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            )
        )
        lines.append(
            envelope(
                "response_item",
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response}],
                },
            )
        )
        for file_path in _unique_strings(row.get("files_touched")):
            lines.append(
                envelope(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": "read",
                        "arguments": {"file_path": file_path},
                        "call_id": f"call-{raw_id}-{tool_seq}",
                    },
                )
            )
            tool_seq += 1
        for tool in _unique_strings(row.get("tool_names")):
            if str(tool).lower() == "read":
                continue
            lines.append(
                envelope(
                    "response_item",
                    {
                        "type": "function_call",
                        "name": str(tool),
                        "arguments": {},
                        "call_id": f"call-{raw_id}-{tool_seq}",
                    },
                )
            )
            tool_seq += 1
    if include_excluded:
        # Test-only canaries: the adapter silently drops these record shapes
        # (codex.py:238-258); their marker text must never reach the chunks.
        canaries: list[tuple[str, dict[str, object]]] = [
            (f"CANARY_DEVELOPER_{raw_id}", {"type": "message", "role": "developer"}),
            (f"CANARY_REASONING_{raw_id}", {"type": "reasoning"}),
            (f"CANARY_TOOL_OUTPUT_{raw_id}", {"type": "function_call_output"}),
        ]
        for marker, payload in canaries:
            payload["content"] = [{"type": "text", "text": marker}]
            lines.append(envelope("response_item", payload))
    return lines


def emit_sessions(
    parquet_path: str | Path,
    out_dir: str | Path,
    *,
    include_excluded: bool = False,
) -> list[Path]:
    """Write one Codex rollout JSONL per ``runtime == 'codex'`` parquet row.

    Files are written to ``out_dir`` (created if absent), each named
    ``<raw_session_id>.jsonl``. Returns the written paths sorted by absolute
    path. ``include_excluded`` (test-only) appends deterministic reasoning /
    tool-output / developer canaries that the adapter must silently drop.
    Raises ``ValueError`` for a parquet without the ``runtime`` column.
    """
    source = Path(parquet_path)
    table = pq.read_table(source)
    if "runtime" not in table.column_names:
        raise ValueError(f"{source}: missing required 'runtime' column")

    rows = [row for row in table.to_pylist() if row.get("runtime") == RUNTIME]
    rows.sort(key=_row_key)

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, row in enumerate(rows):
        raw_id = f"codex-{index:04d}"
        path = destination / f"{raw_id}.jsonl"
        path.write_text(
            "\n".join(_session_lines(row, raw_id, include_excluded=include_excluded)) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    written.sort(key=lambda item: str(item))
    return written
