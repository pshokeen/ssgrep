"""Pi JSONL transcript discovery and normalization.

Pi persists an append-only session format.  This adapter deliberately treats
those files as untrusted local data: it reads only newline-terminated records,
applies the shared JSONL size limit, and never invokes the runtime (or any
network service) while discovering sessions.

Derived runtimes (omp, prime-agent) subclass ``_PiRuntimeAdapter`` in their own
modules and reuse this parsing unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ssgrep.sessions import records
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
    is_main: bool
    parent_session_id: str | None


# Entry types which are valid Pi state/context records but intentionally do not
# contain searchable human/assistant transcript signal.
_IGNORED_ENTRY_TYPES = frozenset(
    {
        "agent_status",
        "branch_summary",
        "child_usage_attributed",
        "compaction",
        "custom",
        "custom_message",
        "label",
        "service_tier_change",
        "session_state",
        "thinking_level_change",
    }
)
_TITLE_ENTRY_TYPES = frozenset(
    {"custom-title", "session_info", "session_title", "title", "title_change"}
)
_GIT_ENTRY_TYPES = frozenset({"git", "git_state"})


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
            if len(payload) > records.MAX_LINE_BYTES:
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


def _branch(record: dict) -> str | None:
    git = record.get("git")
    if isinstance(git, dict):
        return _string(git.get("branch"))
    return _string(record.get("branch"))


def _namespaced(runtime: str, value: str) -> str:
    return f"{runtime}:{value}"


def _inspect(records_: tuple[dict, ...], runtime: str) -> _SourceInfo | None:
    """Collect discovery metadata without retaining or exposing content."""
    if not records_:
        return None
    header = records_[0]
    raw_session_id = _string(header.get("id"))
    if header.get("type") != "session" or raw_session_id is None:
        return None

    cwd = _string(header.get("cwd"))
    model = _string(header.get("modelId")) or _string(header.get("model"))
    raw_depth = header.get("rlmDepth")
    is_child = isinstance(raw_depth, int) and not isinstance(raw_depth, bool) and raw_depth > 0
    raw_parent_id = _string(header.get("parentSessionId"))
    parent_session_id = _namespaced(runtime, raw_parent_id) if raw_parent_id else None

    for record in records_[1:]:
        record_type = record.get("type")
        if record_type == "model_change":
            model = _string(record.get("modelId")) or _string(record.get("model")) or model
        elif record_type == "message":
            message = record.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                model = _string(message.get("model")) or model

    return _SourceInfo(
        session_id=_namespaced(runtime, raw_session_id),
        cwd=cwd,
        model=model,
        is_main=not is_child,
        parent_session_id=parent_session_id,
    )


def _metadata_fields(
    record: dict,
    source: TranscriptSource,
    *,
    cwd: str | None,
    git_branch: str | None,
) -> dict:
    normalized: dict = {"sessionId": source.session.session_id}
    entry_id = _string(record.get("id"))
    timestamp = _string(record.get("timestamp"))
    if entry_id:
        normalized["uuid"] = _namespaced(source.session.runtime, entry_id)
    if timestamp:
        normalized["timestamp"] = timestamp
    if cwd:
        normalized["cwd"] = cwd
    if git_branch:
        normalized["gitBranch"] = git_branch
    return normalized


def _content_blocks(content: object, *, assistant: bool) -> list[dict] | None:
    """Return Claude-shaped signal/tool blocks, or None for invalid content."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return None

    normalized: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            normalized.append({"type": "text", "text": block["text"]})
        elif block_type == "toolCall" and assistant:
            tool_name = _string(block.get("name"))
            if tool_name is None:
                continue
            tool: dict = {
                "type": "tool_use",
                "name": tool_name,
                "input": block.get("arguments", {}),
            }
            tool_id = _string(block.get("id"))
            if tool_id:
                tool["id"] = tool_id
            normalized.append(tool)
        # Thinking, images, tool results, and extension-specific blocks carry
        # no indexable transcript signal and are intentionally omitted.
    return normalized


def _normalize(
    records_: tuple[dict, ...], source: TranscriptSource
) -> tuple[tuple[dict, ...], int]:
    normalized: list[dict] = []
    skipped = 0
    cwd: str | None = source.session.project_paths[0] if source.session.project_paths else None
    git_branch: str | None = None
    model: str | None = source.session.agent_model
    provider: str | None = None

    for record in records_:
        record_type = record.get("type")
        if record_type == "session":
            cwd = _string(record.get("cwd")) or cwd
            git_branch = _branch(record) or git_branch
            model = _string(record.get("modelId")) or _string(record.get("model")) or model
            continue
        if record_type == "model_change":
            model = _string(record.get("modelId")) or _string(record.get("model")) or model
            provider = _string(record.get("provider")) or provider
            continue
        if record_type in _GIT_ENTRY_TYPES:
            git_branch = _branch(record) or git_branch
            continue
        if record_type in _TITLE_ENTRY_TYPES:
            title = (
                _string(record.get("name"))
                or _string(record.get("title"))
                or _string(record.get("custom-title"))
            )
            if title is None:
                skipped += 1
                continue
            item = _metadata_fields(record, source, cwd=cwd, git_branch=git_branch)
            item.update({"type": "custom-title", "custom-title": title})
            normalized.append(item)
            continue
        if record_type in _IGNORED_ENTRY_TYPES:
            continue
        if record_type != "message":
            skipped += 1
            continue

        message = record.get("message")
        if not isinstance(message, dict):
            skipped += 1
            continue
        role = message.get("role")
        if role == "toolResult":
            continue
        if role not in ("user", "assistant"):
            skipped += 1
            continue
        is_assistant = role == "assistant"
        blocks = _content_blocks(message.get("content"), assistant=is_assistant)
        if blocks is None:
            skipped += 1
            continue

        if is_assistant:
            model = _string(message.get("model")) or model
            provider = _string(message.get("provider")) or provider
        target_message: dict = {"role": role, "content": blocks}
        if is_assistant and model:
            target_message["model"] = model
        if is_assistant and provider:
            target_message["provider"] = provider
        item = _metadata_fields(record, source, cwd=cwd, git_branch=git_branch)
        item.update({"type": role, "message": target_message})
        normalized.append(item)

    return tuple(normalized), skipped


class _PiRuntimeAdapter:
    """Shared implementation for runtimes derived from Pi's session format."""

    name: ClassVar[str]
    runtime: ClassVar[str]
    default_relative_root: ClassVar[tuple[str, ...]]
    session_env_vars: ClassVar[tuple[str, ...]]
    agent_env_vars: ClassVar[tuple[str, ...]]

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
        for name in self.agent_env_vars:
            value = os.environ.get(name, "").strip()
            if value:
                return Path(value).expanduser() / "sessions"
        return Path.home().joinpath(*self.default_relative_root)

    def _discovery_roots(self) -> tuple[Path, ...]:
        return (self.root,)

    def _is_main(self, path: Path, info: _SourceInfo) -> bool:
        return info.is_main

    def _inspect_source(self, records_: tuple[dict, ...], *, runtime: str) -> _SourceInfo | None:
        """Discovery-metadata hook so derived runtimes can adjust header handling."""
        return _inspect(records_, runtime)

    def discover(
        self, *, scope: str | None = None, no_subagents: bool = False
    ) -> list[TranscriptSource]:
        roots = tuple(root for root in self._discovery_roots() if root.is_dir())
        if not roots:
            return []
        try:
            candidates = sorted(
                {path for root in roots for path in root.rglob("*.jsonl")},
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
            info = self._inspect_source(raw.records, runtime=self.runtime)
            if info is None:
                continue
            is_main = self._is_main(path, info)
            if scope and (not info.cwd or not paths.is_at_or_beneath(info.cwd, scope)):
                continue
            if no_subagents and not is_main:
                continue
            fingerprint = file_fingerprint(path)
            if fingerprint is None:
                continue
            session = SessionFile(
                path=path,
                session_id=info.session_id,
                is_main=is_main,
                parent_session_id=info.parent_session_id,
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
        normalized, normalization_skips = _normalize(raw.records, source)
        return ReadResult(
            normalized,
            malformed_records=raw.malformed_records,
            skipped_records=raw.skipped_records + normalization_skips,
        )

    def present(self, source: TranscriptSource) -> bool:
        return source.session.path.exists()


class PiAdapter(_PiRuntimeAdapter):
    """Discover standard Pi sessions."""

    name = "pi"
    runtime = "pi"
    default_relative_root = (".pi", "agent", "sessions")
    session_env_vars = ("SSGREP_PI_SESSIONS_DIR", "PI_SESSION_DIR", "PI_CODING_AGENT_SESSION_DIR")
    agent_env_vars = ("PI_CODING_AGENT_DIR",)
