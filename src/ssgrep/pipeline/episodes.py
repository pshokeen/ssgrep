"""Episode building for the pipeline: enrichment, segmentation, text extraction.

These functions are the exact ingestion semantics of the pre-CocoIndex
indexer, relocated so the traced components can call them. They are pure over
their inputs (``raw`` records plus one ``SessionFile``) and never touch the
engine, which is what lets CocoIndex memoize whole sources safely.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ssgrep.pipeline.rows import project_path
from ssgrep.sessions import episodes as episodes_mod, metadata, signal as signal_mod
from ssgrep.utilities.types import ContentType, Episode, SessionFile


def claude_metadata(raw: list[dict]) -> dict[str, str | None]:
    """First-seen Claude runtime metadata from raw record pairs."""
    aliases = {
        "claude_version": ("version",),
        "entrypoint": ("entrypoint",),
        "permission_mode": ("permissionMode", "permission_mode"),
        "user_type": ("userType", "user_type"),
    }
    values: dict[str, str | None] = {name: None for name in aliases}
    values["agent_model"] = None
    for record in raw:
        message = record.get("message")
        if values["agent_model"] is None and isinstance(message, dict):
            model = message.get("model")
            if model is not None:
                values["agent_model"] = str(model)
        for name, keys in aliases.items():
            if values[name] is not None:
                continue
            for key in keys:
                value = record.get(key)
                if value is not None:
                    values[name] = str(value)
                    break
    return values


def agent_metadata(session: SessionFile) -> metadata.AgentMeta | None:
    """Subagent identity from the sidecar file Claude writes."""
    if session.is_main or session.runtime != "claude":
        return None
    return metadata.load_agent_meta(session.path.with_suffix(".meta.json"))


def enrich_session(session: SessionFile, raw: list[dict]) -> SessionFile:
    """Fold agent sidecar and Claude runtime metadata into the session."""
    agent = agent_metadata(session)
    claude = claude_metadata(raw) if session.runtime == "claude" else {}
    return dataclasses.replace(
        session,
        agent_type=(agent.agent_type if agent else None) or session.agent_type,
        agent_name=(agent.name if agent else None) or session.agent_name,
        agent_description=(agent.description if agent else None) or session.agent_description,
        agent_model=(agent.model if agent else None)
        or session.agent_model
        or claude.get("agent_model"),
        claude_version=claude.get("claude_version") or session.claude_version,
        entrypoint=claude.get("entrypoint") or session.entrypoint,
        permission_mode=claude.get("permission_mode") or session.permission_mode,
        user_type=claude.get("user_type") or session.user_type,
    )


def extract_episode_text(group: list[dict]) -> tuple[str, str]:
    """Prompt and response signal text for one record group."""
    prompt_parts: list[str] = []
    response_parts: list[str] = []
    for record in group:
        record_type = record.get("type", "")
        if record_type in ("user", "assistant"):
            content = record.get("message", {}).get("content", [])
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                classified = signal_mod.classify_signal(record, block)
                if not classified.is_signal or not classified.text:
                    continue
                if classified.content_type == ContentType.PROMPT:
                    prompt_parts.append(classified.text)
                elif classified.content_type == ContentType.RESPONSE:
                    response_parts.append(classified.text)
        elif record_type == "system":
            classified = signal_mod.classify_signal(record)
            if (
                classified.is_signal
                and classified.text
                and classified.content_type == ContentType.RESPONSE
            ):
                response_parts.append(classified.text)
    return "\n".join(prompt_parts), "\n".join(response_parts)


def build_episodes(raw: list[dict], session: SessionFile, start_index: int = 0) -> list[Episode]:
    """Segment raw records and enrich every episode for retrieval."""
    grouped = episodes_mod.segment_episode_groups(raw, session.session_id)
    built: list[Episode] = []
    for index, (episode, group) in enumerate(grouped, start=start_index):
        harvested = metadata.harvest_metadata(group, episode_index=index)
        prompt, response = extract_episode_text(group)
        project = (
            str(Path(harvested.cwd).expanduser().absolute())
            if harvested.cwd
            else project_path(episode, session)
        )
        built.append(
            dataclasses.replace(
                episode,
                episode_id=f"{session.session_id}:ep:{index}",
                prompt_text=prompt or episode.prompt_text,
                response_text=response or episode.response_text,
                title=harvested.title,
                timestamp=harvested.timestamp,
                git_branch=harvested.git_branch,
                cwd=harvested.cwd,
                files_touched=harvested.files_touched,
                tool_names=harvested.tool_names,
                is_subagent=not session.is_main,
                agent_type=session.agent_type,
                agent_name=session.agent_name,
                agent_description=session.agent_description,
                parent_session_id=session.parent_session_id,
                agent_model=session.agent_model,
                project=project,
                source_path=str(session.path.absolute()),
                source_project=session.source_project,
                claude_version=session.claude_version,
                entrypoint=session.entrypoint,
                permission_mode=session.permission_mode,
                user_type=session.user_type,
                runtime=session.runtime,
            )
        )
    return built


__all__ = [
    "agent_metadata",
    "build_episodes",
    "claude_metadata",
    "enrich_session",
    "extract_episode_text",
]
