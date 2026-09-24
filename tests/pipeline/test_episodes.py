"""Tests for pipeline episode building (ported ingestion semantics)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ssgrep.pipeline import episodes
from ssgrep.utilities.types import SessionFile


def _claude_records() -> list[dict]:
    return [
        {
            "type": "user",
            "cwd": "/work/app",
            "gitBranch": "main",
            "timestamp": "2025-01-02T03:04:05Z",
            "sessionId": "session-1",
            "message": {"content": [{"type": "text", "text": "How do I fix the auth bug?"}]},
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-02T03:04:06Z",
            "sessionId": "session-1",
            "message": {
                "content": [
                    {"type": "text", "text": "Use a retry policy with exponential backoff."}
                ]
            },
        },
    ]


def test_claude_metadata_first_seen_and_aliases() -> None:
    raw = [
        {"message": {"model": "claude-3-5-sonnet"}},
        {
            "version": "2025-01-01",
            "entrypoint": "cli",
            "permissionMode": "default",
            "userType": "user",
        },
        {"version": "ignored-second"},
    ]
    metadata = episodes.claude_metadata(raw)
    assert metadata["agent_model"] == "claude-3-5-sonnet"
    assert metadata["claude_version"] == "2025-01-01"
    assert metadata["entrypoint"] == "cli"
    assert metadata["permission_mode"] == "default"
    assert metadata["user_type"] == "user"


def test_claude_metadata_missing_values_are_none() -> None:
    metadata = episodes.claude_metadata([{"type": "user", "message": {"content": "x"}}])
    assert metadata == {
        "agent_model": None,
        "claude_version": None,
        "entrypoint": None,
        "permission_mode": None,
        "user_type": None,
    }


def test_extract_episode_text_splits_prompt_and_response() -> None:
    prompt, response = episodes.extract_episode_text(_claude_records())
    assert prompt == "How do I fix the auth bug?"
    assert response == "Use a retry policy with exponential backoff."


def test_extract_episode_text_ignores_tool_and_thinking_blocks() -> None:
    raw = [
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "nope"}]}},
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "..."}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "real answer"}]}},
    ]
    prompt, response = episodes.extract_episode_text(raw)
    assert prompt == ""
    assert response == "real answer"


def test_extract_episode_text_skips_non_list_content() -> None:
    raw = [{"type": "user", "message": {"content": 42}}]
    assert episodes.extract_episode_text(raw) == ("", "")


def test_extract_episode_text_skips_non_dict_blocks() -> None:
    raw = [{"type": "assistant", "message": {"content": ["plain string"]}}]
    assert episodes.extract_episode_text(raw) == ("", "")


def test_extract_episode_text_system_away_summary() -> None:
    raw = [{"type": "system", "subtype": "away_summary", "content": "context restored"}]
    prompt, response = episodes.extract_episode_text(raw)
    assert prompt == ""
    assert response == "context restored"


def test_enrich_session_reads_subagent_sidecar(tmp_path) -> None:
    path = tmp_path / "sub.jsonl"
    (tmp_path / "sub.meta.json").write_text(
        json.dumps(
            {
                "agentType": "coder",
                "name": "worker-1",
                "model": "sonnet",
                "description": "  A subagent that reviews diffs.  ",
            }
        )
    )
    session = SessionFile(
        path=path,
        session_id="parent:sub",
        is_main=False,
        parent_session_id="parent",
    )
    enriched = episodes.enrich_session(session, [])
    assert enriched.agent_type == "coder"
    assert enriched.agent_name == "worker-1"
    assert enriched.agent_model == "sonnet"
    assert enriched.agent_description == "A subagent that reviews diffs."


def test_enrich_session_skips_sidecar_for_main_and_non_claude(tmp_path) -> None:
    main = SessionFile(path=tmp_path / "main.jsonl", session_id="main", is_main=True)
    assert episodes.enrich_session(main, []).agent_type is None
    pi = SessionFile(path=tmp_path / "pi.jsonl", session_id="pi", is_main=False, runtime="pi")
    assert episodes.enrich_session(pi, []).agent_type is None  # no sidecar load


def test_build_episodes_segments_and_enriches(sample_session) -> None:
    built = episodes.build_episodes(_claude_records(), sample_session)
    assert len(built) == 1
    episode = built[0]
    assert episode.episode_id == "session-1:ep:0"
    assert episode.prompt_text == "How do I fix the auth bug?"
    assert episode.response_text == "Use a retry policy with exponential backoff."
    assert episode.title.startswith("How do I fix the auth bug?")
    assert episode.timestamp == datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert episode.git_branch == "main"
    assert episode.cwd == "/work/app"
    assert episode.project == str(Path("/work/app").absolute())
    assert episode.source_path == str(Path(sample_session.path).absolute())
    assert episode.runtime == "claude"
