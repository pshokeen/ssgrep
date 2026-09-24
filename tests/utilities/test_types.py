"""Tests for framework-independent frozen contract types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.utilities import types


def test_search_exception_base_preserves_message_condition_and_command() -> None:
    error = types.SearchException("failed", condition="machine_code", command="repair")
    assert str(error) == "failed"
    assert error.args == ("failed",)
    assert error.condition == "machine_code"
    assert error.command == "repair"


def test_degenerate_search_exceptions_have_safe_defaults_and_allow_overrides() -> None:
    empty = types.EmptyQueryError("empty")
    assert (empty.condition, empty.command) == ("empty_query", None)
    assert isinstance(empty, types.SearchException)
    overridden_empty = types.EmptyQueryError("empty", "other", "try-again")
    assert (overridden_empty.condition, overridden_empty.command) == ("other", "try-again")

    invalid = types.InvalidPredicateError("bad predicate")
    assert (invalid.condition, invalid.command) == ("invalid_predicate", None)
    assert types.InvalidPredicateError("bad", "custom").condition == "custom"

    missing = types.IndexNotFoundError("missing")
    assert (missing.condition, missing.command) == ("missing_index", "ssgrep index")
    custom_missing = types.IndexNotFoundError("missing", "other", "initialize")
    assert (custom_missing.condition, custom_missing.command) == ("other", "initialize")

    not_ready = types.IndexNotReadyError("broken")
    assert (not_ready.condition, not_ready.command) == (
        "corrupt_index",
        "ssgrep index --rebuild",
    )
    custom_not_ready = types.IndexNotReadyError("broken", "busy", None)
    assert (custom_not_ready.condition, custom_not_ready.command) == ("busy", None)


def test_rebuild_would_shrink_error_records_both_censuses() -> None:
    error = types.RebuildWouldShrinkError("refused", (9, 8, 7), (3, 2, 1))
    assert str(error) == "refused"
    assert error.condition == "rebuild_would_shrink"
    assert error.command is None
    assert error.old_counts == (9, 8, 7)
    assert error.new_counts == (3, 2, 1)

    custom = types.RebuildWouldShrinkError(
        "refused", (1, 2, 3), (0, 0, 0), "custom", "safe recovery"
    )
    assert (custom.condition, custom.command) == ("custom", "safe recovery")


def test_content_type_values_are_stable() -> None:
    assert list(types.ContentType) == [types.ContentType.PROMPT, types.ContentType.RESPONSE]
    assert types.ContentType.PROMPT.value == "prompt"
    assert types.ContentType("response") is types.ContentType.RESPONSE


def test_session_file_defaults_full_metadata_and_frozen_contract(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    minimal = types.SessionFile(path=path, session_id="session", is_main=True)
    assert minimal.path == path
    assert minimal.parent_session_id is None
    assert minimal.project_paths == ()
    assert minimal.source_project is None
    assert minimal.agent_type is None
    assert minimal.agent_name is None
    assert minimal.agent_description is None
    assert minimal.agent_model is None
    assert minimal.claude_version is None
    assert minimal.entrypoint is None
    assert minimal.permission_mode is None
    assert minimal.user_type is None

    full = types.SessionFile(
        path=path,
        session_id="parent:agent",
        is_main=False,
        parent_session_id="parent",
        agent_type="Explore",
        agent_name="researcher",
        agent_description="find facts",
        agent_model="local-model",
        project_paths=("/one", "/two"),
        source_project="project",
        claude_version="1.2.3",
        entrypoint="cli",
        permission_mode="safe",
        user_type="developer",
    )
    assert full.project_paths == ("/one", "/two")
    assert full.parent_session_id == "parent"
    assert full.agent_description == "find facts"
    assert full.user_type == "developer"
    with pytest.raises(FrozenInstanceError):
        full.session_id = "changed"  # type: ignore


def test_episode_defaults_and_full_metadata() -> None:
    stamp = datetime(2025, 1, 2, tzinfo=UTC)
    minimal = types.Episode("e", "s", "prompt", "response", "title")
    assert minimal.timestamp is None
    assert minimal.files_touched == ()
    assert minimal.tool_names == ()
    assert minimal.is_subagent is False
    assert minimal.project is None
    assert minimal.source_path is None
    assert minimal.user_type is None

    full = types.Episode(
        episode_id="e",
        session_id="s",
        prompt_text="prompt",
        response_text="response",
        title="title",
        timestamp=stamp,
        git_branch="main",
        cwd="/work",
        files_touched=("a.py",),
        tool_names=("Read",),
        is_subagent=True,
        agent_type="Explore",
        agent_name="agent",
        agent_description="description",
        parent_session_id="parent",
        agent_model="model",
        project="/work",
        source_path="/sessions/s.jsonl",
        source_project="project",
        claude_version="1",
        entrypoint="cli",
        permission_mode="plan",
        user_type="human",
    )
    assert full.timestamp == stamp
    assert full.files_touched == ("a.py",)
    assert full.tool_names == ("Read",)
    assert full.is_subagent is True
    assert full.agent_model == "model"
    assert full.permission_mode == "plan"


def test_chunk_and_search_filters_defaults_and_composition() -> None:
    chunk = types.Chunk("chunk", "body", types.ContentType.PROMPT)
    assert (chunk.chunk_id, chunk.text, chunk.content_type) == (
        "chunk",
        "body",
        types.ContentType.PROMPT,
    )
    assert types.SearchFilters() == types.SearchFilters()

    stamp = datetime(2025, 1, 2, tzinfo=UTC)
    filters = types.SearchFilters(
        date_from=stamp,
        date_to=stamp,
        file_path="a.py",
        content_type=types.ContentType.RESPONSE,
        branch="main",
        project="/work",
        source_path="/session",
        session_id="s",
        is_subagent=True,
        agent_type="Explore",
        agent_model="model",
        tool_name="Read",
    )
    assert filters.date_from == filters.date_to == stamp
    assert filters.content_type is types.ContentType.RESPONSE
    assert filters.source_path == "/session"
    assert filters.is_subagent is True
    assert filters.tool_name == "Read"


def test_result_card_defaults_and_full_metadata() -> None:
    minimal = types.ResultCard(
        ref="e",
        title="title",
        timestamp=None,
        score=0.75,
        excerpt="excerpt",
        files_touched=(),
        is_subagent=False,
    )
    assert minimal.agent_name is None
    assert minimal.content_type is None
    assert minimal.source_absent is False

    full = types.ResultCard(
        ref="e",
        title="title",
        timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        score=0.75,
        excerpt="excerpt",
        files_touched=("a.py",),
        is_subagent=True,
        agent_name="agent",
        agent_description="description",
        parent_session_id="parent",
        content_type=types.ContentType.RESPONSE,
        project="/work",
        source_path="/session",
        source_project="project",
        git_branch="main",
        agent_model="model",
        source_absent=True,
    )
    assert full.agent_name == "agent"
    assert full.content_type is types.ContentType.RESPONSE
    assert full.source_project == "project"
    assert full.source_absent is True


def test_search_response_defaults_and_outcome_metadata() -> None:
    response = types.SearchResponse(results=[])
    assert response.omitted_count == 0
    assert response.index_empty is False
    assert response.total_matches == 0
    assert response.excerpts_truncated is False
    assert response.clamped is False

    full = types.SearchResponse(
        results=[],
        omitted_count=4,
        index_empty=True,
        total_matches=4,
        excerpts_truncated=True,
        clamped=True,
    )
    assert (full.omitted_count, full.total_matches) == (4, 4)
    assert full.index_empty and full.excerpts_truncated and full.clamped


def test_episode_detail_defaults_and_full_metadata() -> None:
    minimal = types.EpisodeDetail(
        episode_id="e",
        session_id="s",
        title="title",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text="prompt",
        response_text="response",
        files_touched=(),
        tool_names=(),
        is_subagent=False,
    )
    assert minimal.agent_type is None
    assert minimal.project is None
    assert minimal.prompt_truncated is False
    assert minimal.response_truncated is False

    full = types.EpisodeDetail(
        episode_id="e",
        session_id="s",
        title="title",
        timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        git_branch="main",
        cwd="/work",
        prompt_text="prompt",
        response_text="response",
        files_touched=("a.py",),
        tool_names=("Read",),
        is_subagent=True,
        agent_type="Explore",
        agent_name="agent",
        agent_description="description",
        parent_session_id="parent",
        project="/work",
        source_path="/session",
        source_project="project",
        agent_model="model",
        claude_version="1",
        entrypoint="cli",
        permission_mode="plan",
        user_type="human",
        prompt_truncated=True,
        response_truncated=True,
    )
    assert full.agent_description == "description"
    assert full.agent_model == "model"
    assert full.claude_version == "1"
    assert full.user_type == "human"
    assert full.prompt_truncated and full.response_truncated


def test_index_stats_defaults_and_tombstone_observability() -> None:
    minimal = types.IndexStats(
        session_count=1,
        episode_count=2,
        chunk_count=3,
        index_size_bytes=4,
        last_index_time=None,
        model_id="local",
        vector_dimension=256,
        skipped_records=0,
        malformed_records=0,
        schema_version=2,
    )
    assert minimal.tombstoned_source_count == 0
    assert minimal.tombstoned_chunk_count == 0
    assert minimal.index_exists is True
    assert minimal.data_dir is None

    full = types.IndexStats(
        session_count=1,
        episode_count=2,
        chunk_count=3,
        index_size_bytes=4,
        last_index_time=datetime(2025, 1, 2, tzinfo=UTC),
        model_id="local",
        vector_dimension=256,
        skipped_records=5,
        malformed_records=6,
        schema_version=2,
        tombstoned_source_count=7,
        tombstoned_chunk_count=8,
        index_exists=False,
        data_dir="/private/data",
    )
    assert full.last_index_time is not None
    assert full.tombstoned_source_count == 7
    assert full.tombstoned_chunk_count == 8
    assert full.index_exists is False
    assert full.data_dir == "/private/data"
