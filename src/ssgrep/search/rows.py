"""Data classes for chunk and episode rows from the index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class _ChunkHit:
    """Minimal chunk projection needed to fuse, filter, and roll up."""

    content_type: str
    text: str
    source_status: str
    chunk_id: str = ""


@dataclass(frozen=True)
class _EpisodeRow:
    """Minimal episode projection needed to filter, boost, and render."""

    title: str
    timestamp: datetime | None
    git_branch: str | None
    files_touched: tuple[str, ...]
    is_subagent: bool
    agent_name: str | None
    agent_description: str | None
    parent_session_id: str | None
    project: str | None = None
    source_path: str | None = None
    source_project: str | None = None
    agent_model: str | None = None
    runtime: str = "claude"
