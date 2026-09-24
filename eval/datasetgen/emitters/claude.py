"""Claude native transcript emitter (``runtime == "claude"``).

Task 7 of the retrieval-eval overhaul. Reads ``sessions.parquet`` (the T6 NeMo
workflow output) and writes one native-format JSONL record-pair file per claude
row into a directory that the real ``native`` adapter ingests unchanged when
the harness points ``SSGREP_TRANSCRIPT_DIRS`` at it. Every claim here is pinned
to the adapter contract in ``eval/datasetgen/adapter_formats.md`` sections 3
and 10.1, which was derived from ``src/ssgrep/sessions/adapters/native.py``.

Expected parquet schema (documented contract for T6; T6 and T7 ran in
parallel, so the emitter reads only the documented columns and ignores any
others):

    runtime        str        -- must equal "claude" for emission
    scenario_class str        -- D5 class label; carried for provenance only
    episodes       list[dict] # [{"prompt": str, "response": str}, ...]
    title          str        # becomes the session custom-title record
    project        str        # fictional project root, becomes every record's cwd
    files_touched  list[str]  # optional; surfaced via tool_use "Read" blocks
    tool_names     list[str]  # optional; surfaced via tool_use blocks
    version        str        # optional; stored as claude_version, defaults "1.0.0"
    git_branch     str        # optional; recorded on every record
    sessionId      str        # optional; used verbatim when present
    difficulty / episode_length_bucket: str  # optional; not emitted

Emission rules (verbatim from the T4 contract):

- Flat directory, one ``.jsonl`` file per session; ingestion goes through
  ``SSGREP_TRANSCRIPT_DIRS`` with the ``native=<dir>`` tag
  (``discovery_roots.py:38``), which requires a ``type`` key on an early line
  (``discovery_roots.py:58-92``) and a newline-terminated final line
  (``base.py:96-98``).
- Each session leads with a ``custom-title`` record so the title is the
  highest-precedence title source (``metadata.py:203-209``), followed by one
  ``user``/``assistant`` pair per episode.
- ``user`` records carry only ``text`` blocks; ``assistant`` records carry
  ``text`` plus ``tool_use`` blocks. A file-tool block carries a ``file_path``
  input so ``harvest_metadata`` surfaces ``files_touched``
  (``metadata.py:103-133``). No ``thinking``/``tool_result``/``image`` blocks
  (``signal.py:30-42``).
- Text-bearing ``user`` text opens an episode (``episodes.py:81-87``); the
  emitter drops text-less prompts so episode counts match the expected
  prompt/response pairs exactly.
- Timestamps are fixed-epoch, content-derived (never wall clock); raw ids are
  ``claude-<nnnn>`` derived from the sorted ``runtime == "claude"`` rows, so a
  second run over the same parquet is byte-identical.

Runtime-label note (deliberate deviation, documented in T4 section 9): the
native adapter stamps ``runtime="native"`` on every source discovered through
``SSGREP_TRANSCRIPT_DIRS`` external roots (``native.py:26-27``). Only a
synthetic ``CLAUDE_CONFIG_DIR/projects/<dir>`` tree yields ``runtime="claude"``
(``native.py:20-22``). The T7 round-trip test therefore ingests through
``SSGREP_TRANSCRIPT_DIRS`` and asserts ``runtime == "native"`` — the same
ingestion path the T13 benchmark harness will use. The ``claude`` census
bucket is satisfied by the file format, not by the stored runtime label.
Session ids in the index take the ``<stem>~<8-hex>`` external-root shape
(``discovery_roots.py:113-117``) and episodes are ``<session_id>:ep:<n>``
(``episodes.py:8-9``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

RUNTIME = "claude"

#: Fixed Claude version stamped on records when the parquet has no ``version``
#: column; the adapter reads it as ``claude_version``.
DEFAULT_VERSION = "1.0.0"

#: Fixed epoch so every emitted timestamp is content-derived, never wall-clock.
_BASE_TIME = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

_FALLBACK_CWD = "/fictional/checkout/claude"

#: Assistant tool names that surface ``files_touched`` via their file_path
#: input (metadata.py:106). Other tool names still count toward ``tool_names``.
_FILE_TOOLS = {"read", "edit", "write", "patch", "apply_patch", "multiedit"}


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Content-derived sort key so ordering and ids survive parquet reorders."""
    return (
        str(row.get("project", "")),
        str(row.get("title", "")),
        json.dumps(row.get("episodes", []), sort_keys=True),
    )


def _episode_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize the ``episodes`` column to ``(prompt, response)`` string pairs.

    Text-less prompts are skipped so the emitted text-bearing user records
    equal the expected episode count (episodes.py:81-87): a text-less prompt
    would fold into the following episode and misalign the count.
    """
    pairs: list[tuple[str, str]] = []
    episodes = row.get("episodes")
    if not isinstance(episodes, list):
        return pairs
    for item in episodes:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            response = item.get("response")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prompt, response = item[0], item[1]
        else:
            continue
        if prompt is None or str(prompt) == "":
            continue
        pairs.append((str(prompt), str(response or "")))
    return pairs


def _cwd(row: dict[str, Any]) -> str:
    project = row.get("project")
    return str(project) if isinstance(project, str) and project else _FALLBACK_CWD


def _unique_strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values if value})


def _timestamp(session_index: int, offset_minutes: int) -> str:
    return (_BASE_TIME + timedelta(hours=session_index, minutes=offset_minutes)).isoformat()


def _tool_use_block(name: str, *, file_path: str | None = None) -> dict[str, Any]:
    """One ``tool_use`` content block with deterministic key order."""
    block: dict[str, Any] = {"type": "tool_use", "name": name}
    block["input"] = {"file_path": file_path} if file_path is not None else {}
    return block


def _assistant_content(
    response: str,
    tool_names: list[str],
    files_touched: list[str],
) -> list[dict[str, Any]]:
    """assistant message.content: a text block plus allowed tool_use blocks.

    Each file-tool name carries a ``file_path`` so ``files_touched`` surfaces
    (metadata.py:111-133); every other tool name emits a bare ``tool_use``
    block so ``tool_names`` still records it (metadata.py:190-199).
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": response}]
    for index, tool in enumerate(tool_names):
        if tool.lower() in _FILE_TOOLS:
            path = files_touched[index % len(files_touched)] if files_touched else None
            content.append(_tool_use_block(tool, file_path=path))
        else:
            content.append(_tool_use_block(tool))
    for index, path in enumerate(files_touched):
        paired = [tool for tool in tool_names if tool.lower() in _FILE_TOOLS]
        if index < len(paired):
            continue
        content.append(_tool_use_block("Read", file_path=path))
    return content


def _session_records(
    row: dict[str, Any],
    raw_id: str,
    session_index: int,
) -> list[dict[str, Any]]:
    """Ordered native-format records for one session (deterministic)."""
    cwd = _cwd(row)
    pairs = _episode_pairs(row)
    tool_names = _unique_strings(row.get("tool_names"))
    files_touched = _unique_strings(row.get("files_touched"))
    version = str(row.get("version") or DEFAULT_VERSION)
    git_branch = row.get("git_branch")
    title = str(row.get("title") or f"Session {session_index}")

    shared: dict[str, Any] = {"cwd": cwd, "sessionId": raw_id}
    if git_branch is not None and str(git_branch):
        shared["gitBranch"] = str(git_branch)

    records: list[dict[str, Any]] = [
        {
            "type": "custom-title",
            "custom-title": title,
            "sessionId": raw_id,
            "cwd": cwd,
            "timestamp": _timestamp(session_index, 0),
        }
    ]
    for index, (prompt, response) in enumerate(pairs):
        records.append(
            {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
                **shared,
                "timestamp": _timestamp(session_index, index * 2 + 1),
                "uuid": f"{raw_id}-u-{index:04d}",
                "version": version,
            }
        )
        records.append(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": _assistant_content(response, tool_names, files_touched),
                },
                **shared,
                "timestamp": _timestamp(session_index, index * 2 + 2),
                "uuid": f"{raw_id}-a-{index:04d}",
                "version": version,
            }
        )
    return records


def emit_sessions(parquet_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Write one native-format JSONL file per ``runtime == 'claude'`` parquet row.

    Files are written to ``out_dir`` (created if absent), each named
    ``<raw_session_id>.jsonl``. Returns the written paths sorted by absolute
    path. Raises ``ValueError`` for a parquet missing the ``runtime`` column.
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
        raw_id = f"{RUNTIME}-{index:04d}"
        records = _session_records(row, raw_id, index)
        path = destination / f"{raw_id}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        written.append(path)
    written.sort(key=lambda item: str(item.absolute()))
    return written


__all__ = ["DEFAULT_VERSION", "RUNTIME", "emit_sessions"]
