"""Tests for the internal search row dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from ssgrep.search.rows import _ChunkHit, _EpisodeRow


def test_chunk_hit_is_a_frozen_value_object() -> None:
    hit = _ChunkHit(content_type="response", text="answer", source_status="available")

    assert hit == _ChunkHit("response", "answer", "available")
    with pytest.raises(FrozenInstanceError):
        hit.text = "changed"  # type: ignore


def test_episode_row_defaults_and_frozen_fields() -> None:
    timestamp = datetime(2025, 1, 2, 3, 4)
    row = _EpisodeRow(
        title="A title",
        timestamp=timestamp,
        git_branch="main",
        files_touched=("one.py",),
        is_subagent=True,
        agent_name="helper",
        agent_description="do work",
        parent_session_id="parent",
    )

    assert row.timestamp is timestamp
    assert row.project is None
    assert row.source_path is None
    assert row.source_project is None
    assert row.agent_model is None
    with pytest.raises(FrozenInstanceError):
        row.title = "changed"  # type: ignore
