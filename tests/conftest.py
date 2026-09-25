"""Shared fixtures for the mirrored ssgrep test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.utilities.types import (
    ContentType,
    Episode,
    EpisodeDetail,
    IndexStats,
    ResultCard,
    SearchResponse,
    SessionFile,
)


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every database, transcript root, and module cache test-local."""
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "opencode"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi-agent"))
    monkeypatch.setenv("PRIME_AGENT_CODING_AGENT_DIR", str(tmp_path / "prime-agent"))
    monkeypatch.setenv("PI_SESSION_DIR", str(tmp_path / "pi-sessions"))
    monkeypatch.setenv("PRIME_AGENT_SESSION_DIR", str(tmp_path / "prime-agent-sessions"))
    monkeypatch.setenv("OMP_SESSIONS_DIR", str(tmp_path / "omp-sessions"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.delenv("SSGREP_PI_SESSIONS_DIR", raising=False)
    monkeypatch.delenv("SSGREP_PRIME_AGENT_SESSIONS_DIR", raising=False)
    monkeypatch.delenv("SSGREP_OMP_SESSIONS_DIR", raising=False)
    monkeypatch.delenv("SSGREP_CODEX_SESSIONS_DIR", raising=False)
    monkeypatch.delenv("SSGREP_OPENCODE_DB", raising=False)
    monkeypatch.delenv("SSGREP_TRANSCRIPT_DIRS", raising=False)

    from ssgrep.services import mcp_server
    from ssgrep.sessions import discovery
    from ssgrep.utilities import paths

    mcp_server._mcp = None
    discovery._cwd_index_cache.clear()
    paths._probe_result = None


@pytest.fixture
def sample_session(tmp_path: Path) -> SessionFile:
    return SessionFile(
        path=tmp_path / "session.jsonl",
        session_id="session-1",
        is_main=True,
        project_paths=(str(tmp_path),),
        source_project="project-1",
    )


@pytest.fixture
def sample_episode(tmp_path: Path) -> Episode:
    return Episode(
        episode_id="session-1:ep:0",
        session_id="session-1",
        prompt_text="How do I test this?",
        response_text="Use pytest.",
        title="Testing",
        timestamp=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        git_branch="main",
        cwd=str(tmp_path),
        files_touched=("tests/test_example.py",),
        tool_names=("Read",),
        project=str(tmp_path),
        source_path=str(tmp_path / "session.jsonl"),
        source_project="project-1",
    )


@pytest.fixture
def sample_card() -> ResultCard:
    return ResultCard(
        ref="session-1:ep:0",
        title="Testing",
        timestamp=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        score=0.75,
        excerpt="Use pytest.",
        files_touched=("tests/test_example.py",),
        is_subagent=False,
        content_type=ContentType.RESPONSE,
        project="/tmp/project",
        source_path="/tmp/session.jsonl",
    )


@pytest.fixture
def sample_response(sample_card: ResultCard) -> SearchResponse:
    return SearchResponse(results=[sample_card], total_matches=1)


@pytest.fixture
def sample_detail() -> EpisodeDetail:
    return EpisodeDetail(
        episode_id="session-1:ep:0",
        session_id="session-1",
        title="Testing",
        timestamp=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        git_branch="main",
        cwd="/tmp/project",
        prompt_text="How do I test this?",
        response_text="Use pytest.",
        files_touched=("tests/test_example.py",),
        tool_names=("Read",),
        is_subagent=False,
    )


@pytest.fixture
def sample_stats(tmp_path: Path) -> IndexStats:
    return IndexStats(
        session_count=1,
        episode_count=1,
        chunk_count=2,
        index_size_bytes=1024,
        last_index_time=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
        model_id="lightonai/answerai-colbert-small-v1",
        vector_dimension=96,
        skipped_records=0,
        malformed_records=0,
        schema_version=5,
        data_dir=str(tmp_path),
    )
