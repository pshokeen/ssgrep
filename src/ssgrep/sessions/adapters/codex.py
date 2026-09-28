"""Codex CLI rollout JSONL transcript discovery and normalization.

Codex persists each interactive/exec session as an append-only ``rollout``
JSONL file under ``~/.codex/sessions`` (one file per session, organized by
date).  Every line is a top-level envelope ``{"timestamp", "type", "payload"}``
whose ``payload`` varies by ``type``:

- ``session_meta`` (first record) carries the session ``id`` and starting ``cwd``.
- ``turn_context`` records carry per-turn ``cwd`` and the active ``model``.
- ``response_item`` records carry the actual conversation: ``message`` items
  (role ``user``/``assistant``/``developer``), ``function_call`` and
  ``custom_tool_call`` items (assistant tool use), and ``reasoning`` items
  (thinking).  ``function_call_output`` / ``custom_tool_call_output`` echo tool
  results and ``token_count`` / ``item_completed`` / ``agent_message`` etc. are
  bookkeeping noise.

Like the other adapters, this one treats the files as untrusted local data: it
reads only newline-terminated records, applies the shared JSONL size limit, and
never invokes the runtime or any network service while discovering sessions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ssgrep.sessions.adapters.base import (
    ReadResult,
    TranscriptSource,
    file_fingerprint,
)
from ssgrep.utilities import paths
from ssgrep.utilities.types import SessionFile


@dataclass(frozen=True)
class _SourceInfo:
    session_id: str
    cwd: str | None
    model: str | None
    provider: str | None


def _read_complete_records(path: Path) -> ReadResult:
    """Parse complete, bounded JSONL records and report degradation counts."""
    parsed: list[dict] = []
    malformed = 0
    skipped = 0
    with path.open("rb") as stream:
        for line_bytes in stream:
            if not line_bytes.endswith(b"\n"):
                # A writer may currently be appending this record.  It must not
                # enter the index until the terminating newline is durable.
                if line_bytes.strip():
                    skipped += 1
                break
            payload = line_bytes[:-1]
            if len(payload) > 500_000:
                skipped += 1
                continue
            if not payload.strip():
                continue
            try:
                line = payload.decode("utf-8")
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
                continue
            if not isinstance(value, dict):
                skipped += 1
                continue
            parsed.append(value)
    return ReadResult(tuple(parsed), malformed_records=malformed, skipped_records=skipped)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _namespaced(runtime: str, value: str) -> str:
    return f"{runtime}:{value}"


def _inspect(records_: tuple[dict, ...], runtime: str) -> _SourceInfo | None:
    """Collect discovery metadata without retaining or exposing content."""
    session_id: str | None = None
    cwd: str | None = None
    model: str | None = None
    provider: str | None = None
    for record in records_:
        record_type = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record_type == "session_meta":
            session_id = _string(payload.get("id")) or session_id
            cwd = _string(payload.get("cwd")) or cwd
            provider = _string(payload.get("model_provider")) or provider
        elif record_type == "turn_context":
            cwd = _string(payload.get("cwd")) or cwd
            model = _string(payload.get("model")) or model
    if session_id is None:
        return None
    return _SourceInfo(
        session_id=_namespaced(runtime, session_id),
        cwd=cwd,
        model=model,
        provider=provider,
    )


def _metadata_fields(
    source: TranscriptSource,
    *,
    cwd: str | None,
    timestamp: str | None,
    uuid: str | None,
) -> dict:
    normalized: dict = {"sessionId": source.session.session_id}
    if uuid:
        normalized["uuid"] = uuid
    if timestamp:
        normalized["timestamp"] = timestamp
    if cwd:
        normalized["cwd"] = cwd
    return normalized


def _text_blocks(content: object) -> list[dict] | None:
    """Return Claude-shaped text blocks, or None for invalid content."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return None
    blocks: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type not in ("text", "input_text", "output_text"):
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            blocks.append({"type": "text", "text": text})
    return blocks


def _tool_block(payload: dict) -> dict | None:
    """Map a Codex tool-call payload to a Claude ``tool_use`` block."""
    payload_type = payload.get("type")
    if payload_type == "function_call":
        name = _string(payload.get("name"))
        if name is None:
            return None
        raw_arguments = payload.get("arguments")
        tool_input: dict = {}
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed = None
            tool_input = parsed if isinstance(parsed, dict) else {"arguments": raw_arguments}
        elif isinstance(raw_arguments, dict):
            tool_input = raw_arguments
    elif payload_type == "custom_tool_call":
        name = _string(payload.get("name"))
        if name is None:
            return None
        raw_input = payload.get("input")
        tool_input = raw_input if isinstance(raw_input, dict) else {"input": raw_input}
    else:
        return None
    block: dict = {"type": "tool_use", "name": name, "input": tool_input}
    call_id = _string(payload.get("call_id"))
    if call_id:
        block["id"] = call_id
    return block


def _normalize(records_: tuple[dict, ...], source: TranscriptSource) -> tuple[dict, ...]:
    normalized: list[dict] = []
    cwd: str | None = source.session.project_paths[0] if source.session.project_paths else None
    model: str | None = source.session.agent_model
    provider: str | None = None
    pending: dict | None = None
    seq = 0

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            normalized.append(pending)
            pending = None

    for record in records_:
        record_type = record.get("type")
        payload = record.get("payload")
        if record_type == "session_meta" and isinstance(payload, dict):
            cwd = _string(payload.get("cwd")) or cwd
            provider = _string(payload.get("model_provider")) or provider
            continue
        if record_type == "turn_context" and isinstance(payload, dict):
            cwd = _string(payload.get("cwd")) or cwd
            model = _string(payload.get("model")) or model
            continue
        if record_type != "response_item" or not isinstance(payload, dict):
            continue

        payload_type = payload.get("type")
        timestamp = _string(record.get("timestamp"))
        uuid = _namespaced(source.session.runtime, f"{seq}:{timestamp or '0'}")
        seq += 1

        if payload_type == "message":
            role = payload.get("role")
            if role == "user":
                flush()
                blocks = _text_blocks(payload.get("content"))
                if blocks is None:
                    continue
                item = _metadata_fields(source, cwd=cwd, timestamp=timestamp, uuid=uuid)
                item.update({"type": "user", "message": {"role": "user", "content": blocks}})
                normalized.append(item)
            elif role == "assistant":
                flush()
                blocks = _text_blocks(payload.get("content"))
                if blocks is None:
                    continue
                message: dict = {"role": "assistant", "content": blocks}
                if model:
                    message["model"] = model
                if provider:
                    message["provider"] = provider
                item = _metadata_fields(source, cwd=cwd, timestamp=timestamp, uuid=uuid)
                item.update({"type": "assistant", "message": message})
                pending = item
            # developer-role messages carry system instructions, not transcript
            # signal, and are intentionally omitted.
        elif payload_type in ("function_call", "custom_tool_call"):
            block = _tool_block(payload)
            if block is None:
                continue
            if pending is None:
                message = {"role": "assistant", "content": [block]}
                if model:
                    message["model"] = model
                if provider:
                    message["provider"] = provider
                item = _metadata_fields(source, cwd=cwd, timestamp=timestamp, uuid=uuid)
                item.update({"type": "assistant", "message": message})
                pending = item
            else:
                content = pending["message"].get("content")
                if isinstance(content, list):
                    content.append(block)
        # reasoning (thinking), *_output echoes, token counts, item bookkeeping,
        # and world-state records carry no indexable signal and are omitted.

    flush()
    return tuple(normalized)


class CodexAdapter:
    """Discover and normalize Codex CLI rollout sessions."""

    name = "codex"
    runtime = "codex"
    session_env_vars = (
        "SSGREP_CODEX_SESSIONS_DIR",
        "CODEX_SESSIONS_DIR",
    )

    def __init__(self, root: str | Path | None = None) -> None:
        self._root_override = Path(root).expanduser() if root is not None else None

    @property
    def root(self) -> Path:
        """Effective session root (exposed to make discovery straightforward to test)."""
        if self._root_override is not None:
            return self._root_override
        for name in self.session_env_vars:
            value = os.environ.get(name, "").strip()
            if value:
                return Path(value).expanduser()
        return paths.resolve_codex_dir() / "sessions"

    def discover(
        self, *, scope: str | None = None, no_subagents: bool = False
    ) -> list[TranscriptSource]:
        del no_subagents  # Codex rollouts are single-threaded; no subagents.
        root = self.root
        if not root.is_dir():
            return []
        try:
            candidates = sorted(
                {path for path in root.rglob("*.jsonl")},
                key=lambda item: str(item.absolute()),
            )
        except OSError:
            return []

        sources: list[TranscriptSource] = []
        for path in candidates:
            try:
                raw = _read_complete_records(path)
            except OSError:
                continue
            info = _inspect(raw.records, self.runtime)
            if info is None:
                continue
            if scope and (not info.cwd or not paths.is_at_or_beneath(info.cwd, scope)):
                continue
            # Codex rollouts are single-threaded; there is no subagent concept
            # to filter on.
            fingerprint = file_fingerprint(path)
            if fingerprint is None:
                continue
            session = SessionFile(
                path=path,
                session_id=info.session_id,
                is_main=True,
                parent_session_id=None,
                agent_model=info.model,
                project_paths=(info.cwd,) if info.cwd else (),
                source_project=Path(info.cwd).name if info.cwd else path.parent.name,
                runtime=self.runtime,
            )
            sources.append(
                TranscriptSource(
                    adapter=self.name,
                    key=f"{self.runtime}:{path.absolute()}",
                    session=session,
                    fingerprint=fingerprint,
                )
            )
        return sources

    def read(self, source: TranscriptSource) -> ReadResult:
        raw = _read_complete_records(source.session.path)
        normalized = _normalize(raw.records, source)
        return ReadResult(
            normalized,
            malformed_records=raw.malformed_records,
            skipped_records=raw.skipped_records,
        )

    def present(self, source: TranscriptSource) -> bool:
        return source.session.path.exists()
