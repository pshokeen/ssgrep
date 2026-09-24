"""Episode drill-down from the global LanceDB store."""

from __future__ import annotations

import re
from datetime import datetime

from ssgrep.store import EPISODES_TABLE, LanceStore, quote
from ssgrep.utilities.types import EpisodeDetail, IndexNotFoundError, IndexNotReadyError

MAX_PROMPT_CHARS = 50_000
MAX_RESPONSE_CHARS = 150_000
TRUNCATION_MARKER = "\n[... truncated, showing first {max_chars} characters ...]"


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _split(value: object) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if item)
    return tuple(str(value).split("\n")) if value else ()


def _validate_episode_id(ref: str) -> str | None:
    match = re.match(r"^(.+):ep:(\d+)$", ref)
    return match.group(1) if match else None


def _apply_bound(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_MARKER.format(max_chars=max_chars)


def show(ref: str) -> EpisodeDetail | None:
    """Return one globally addressed episode."""
    if _validate_episode_id(ref) is None:
        return None
    repository = LanceStore()
    if not repository.exists():
        raise IndexNotFoundError("No global index found. Run `ssgrep index` first.")
    if repository.get_meta("index_state") != "ready":
        raise IndexNotReadyError(
            "The global index is incomplete; run `ssgrep index` to resume it.",
            condition="index_incomplete",
            command="ssgrep index",
        )
    if repository.get_meta("schema_version") != str(repository.schema_version):
        raise IndexNotReadyError("The global index schema is incompatible; rebuild it.")
    rows = repository.rows(
        EPISODES_TABLE,
        where=f"episode_id = {quote(ref)}",
        limit=1,
    )
    if not rows:
        return None
    row = rows[0]
    prompt = str(row.get("prompt_text") or "")
    response = str(row.get("response_text") or "")
    project = row.get("project")
    return EpisodeDetail(
        episode_id=str(row["episode_id"]),
        session_id=str(row["session_id"]),
        title=str(row.get("title") or ref),
        timestamp=_parse_timestamp(row.get("timestamp")),
        git_branch=row.get("git_branch"),
        cwd=row.get("cwd"),
        prompt_text=_apply_bound(prompt, MAX_PROMPT_CHARS),
        response_text=_apply_bound(response, MAX_RESPONSE_CHARS),
        files_touched=_split(row.get("files_touched")),
        tool_names=_split(row.get("tool_names")),
        is_subagent=bool(row.get("is_subagent", False)),
        agent_type=row.get("agent_type"),
        agent_name=row.get("agent_name"),
        agent_description=row.get("agent_description"),
        parent_session_id=row.get("parent_session_id"),
        project=str(project) if project else None,
        source_path=row.get("source_path"),
        source_project=row.get("source_project"),
        agent_model=row.get("agent_model"),
        claude_version=row.get("claude_version"),
        entrypoint=row.get("entrypoint"),
        permission_mode=row.get("permission_mode"),
        user_type=row.get("user_type"),
        prompt_truncated=len(prompt) > MAX_PROMPT_CHARS,
        response_truncated=len(response) > MAX_RESPONSE_CHARS,
        runtime=str(row.get("runtime") or "claude"),
    )
