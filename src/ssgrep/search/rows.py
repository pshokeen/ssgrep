"""Data classes for chunk and episode rows from the index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class _ChunkHit:
    """Minimal chunk projection needed to fuse, filter, and roll up."""

    chunk_id: str
    episode_id: str
    content_type: str
    text: str
    source_status: str


@dataclass(frozen=True)
class _EpisodeRow:
    """Minimal episode projection needed to filter, boost, and render."""

    episode_id: str
    session_id: str
    title: str
    timestamp: datetime | None
    git_branch: str | None
    files_touched: tuple[str, ...]
    is_subagent: bool
    agent_name: str | None
    agent_description: str | None
    parent_session_id: str | None
    source_status: str
