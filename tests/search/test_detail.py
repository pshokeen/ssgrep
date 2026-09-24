"""Tests for episode drill-down retrieval."""

from __future__ import annotations

from datetime import datetime

import pytest

from ssgrep.search import detail
from ssgrep.utilities.types import IndexNotFoundError, IndexNotReadyError


class FakeLanceStore:
    schema_version = 7

    def __init__(
        self,
        *,
        exists: bool = True,
        state: str | None = "ready",
        schema: str | None = "7",
        rows: list[dict] | None = None,
    ) -> None:
        self.exists_result = exists
        self.meta = {"index_state": state, "schema_version": schema}
        self.row_results = [] if rows is None else rows
        self.row_calls: list[tuple[str, str | None, int | None]] = []

    def exists(self) -> bool:
        return self.exists_result

    def get_meta(self, key: str) -> str | None:
        return self.meta.get(key)

    def rows(self, table: str, *, where: str | None = None, limit: int | None = None) -> list[dict]:
        self.row_calls.append((table, where, limit))
        return self.row_results


def use_store(monkeypatch: pytest.MonkeyPatch, store: FakeLanceStore) -> None:
    monkeypatch.setattr(detail, "LanceStore", lambda: store)


def test_parse_timestamp_accepts_datetime_and_iso_and_rejects_other_values() -> None:
    value = datetime(2025, 1, 2, 3, 4)

    assert detail._parse_timestamp(value) is value
    assert detail._parse_timestamp("2025-01-02T03:04:05") == datetime(2025, 1, 2, 3, 4, 5)
    assert detail._parse_timestamp("not a timestamp") is None
    assert detail._parse_timestamp(123) is None
    assert detail._parse_timestamp("") is None


def test_split_normalizes_sequences_strings_and_empty_values() -> None:
    assert detail._split(["one", "", 2]) == ("one", "2")
    assert detail._split(("one", None, "two")) == ("one", "two")
    assert detail._split("one\ntwo") == ("one", "two")
    assert detail._split(None) == ()


def test_episode_reference_validation() -> None:
    assert detail._validate_episode_id("session:ep:12") == "session"
    assert detail._validate_episode_id("parent:worker:ep:0") == "parent:worker"
    assert detail._validate_episode_id("session") is None
    assert detail._validate_episode_id("session:ep:nope") is None


def test_apply_bound_leaves_short_text_and_marks_long_text() -> None:
    assert detail._apply_bound("abc", 3) == "abc"
    assert detail._apply_bound("abcdef", 3) == (
        "abc\n[... truncated, showing first 3 characters ...]"
    )


def test_show_rejects_invalid_reference_without_opening_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_store() -> FakeLanceStore:
        raise AssertionError("invalid refs must not access the index")

    monkeypatch.setattr(detail, "LanceStore", unexpected_store)

    assert detail.show("not-an-episode") is None


def test_show_raises_when_index_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    use_store(monkeypatch, FakeLanceStore(exists=False))

    with pytest.raises(IndexNotFoundError, match="No global index found") as caught:
        detail.show("session:ep:0")

    assert caught.value.condition == "missing_index"
    assert caught.value.command == "ssgrep index"


def test_show_raises_when_index_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    use_store(monkeypatch, FakeLanceStore(state="building"))

    with pytest.raises(IndexNotReadyError, match="incomplete") as caught:
        detail.show("session:ep:0")

    assert caught.value.condition == "index_incomplete"
    assert caught.value.command == "ssgrep index"


def test_show_raises_when_schema_is_incompatible(monkeypatch: pytest.MonkeyPatch) -> None:
    use_store(monkeypatch, FakeLanceStore(schema="6"))

    with pytest.raises(IndexNotReadyError, match="schema is incompatible"):
        detail.show("session:ep:0")


def test_show_returns_none_when_episode_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeLanceStore(rows=[])
    use_store(monkeypatch, store)

    assert detail.show("session:ep:99") is None
    assert store.row_calls == [
        (detail.EPISODES_TABLE, "episode_id = 'session:ep:99'", 1),
    ]


def test_show_builds_complete_bounded_episode_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = "p" * (detail.MAX_PROMPT_CHARS + 1)
    response = "r" * (detail.MAX_RESPONSE_CHARS + 1)
    row = {
        "episode_id": "sess'o:ep:12",
        "session_id": 123,
        "title": "Complete episode",
        "timestamp": "2025-01-02T03:04:05",
        "git_branch": "main",
        "cwd": "/project",
        "prompt_text": prompt,
        "response_text": response,
        "files_touched": ["a.py", "", "b.py"],
        "tool_names": "Read\nEdit",
        "is_subagent": True,
        "agent_type": "general",
        "agent_name": "worker",
        "agent_description": "perform task",
        "parent_session_id": "parent",
        "project": 456,
        "source_path": "/transcript.jsonl",
        "source_project": "source-project",
        "agent_model": "model",
        "claude_version": "1.2.3",
        "entrypoint": "cli",
        "permission_mode": "default",
        "user_type": "external",
    }
    store = FakeLanceStore(rows=[row])
    use_store(monkeypatch, store)

    result = detail.show("sess'o:ep:12")

    assert result is not None
    assert result.episode_id == "sess'o:ep:12"
    assert result.session_id == "123"
    assert result.timestamp == datetime(2025, 1, 2, 3, 4, 5)
    assert result.prompt_text == prompt[
        : detail.MAX_PROMPT_CHARS
    ] + detail.TRUNCATION_MARKER.format(max_chars=detail.MAX_PROMPT_CHARS)
    assert result.response_text == response[
        : detail.MAX_RESPONSE_CHARS
    ] + detail.TRUNCATION_MARKER.format(max_chars=detail.MAX_RESPONSE_CHARS)
    assert result.files_touched == ("a.py", "b.py")
    assert result.tool_names == ("Read", "Edit")
    assert result.is_subagent is True
    assert result.agent_type == "general"
    assert result.agent_name == "worker"
    assert result.agent_description == "perform task"
    assert result.parent_session_id == "parent"
    assert result.project == "456"
    assert result.source_path == "/transcript.jsonl"
    assert result.source_project == "source-project"
    assert result.agent_model == "model"
    assert result.claude_version == "1.2.3"
    assert result.entrypoint == "cli"
    assert result.permission_mode == "default"
    assert result.user_type == "external"
    assert result.prompt_truncated is True
    assert result.response_truncated is True
    assert store.row_calls == [(detail.EPISODES_TABLE, "episode_id = 'sess''o:ep:12'", 1)]


def test_show_applies_defaults_to_minimal_row(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "episode_id": "session:ep:0",
        "session_id": "session",
        "timestamp": "invalid",
        "project": "",
    }
    use_store(monkeypatch, FakeLanceStore(rows=[row]))

    result = detail.show("session:ep:0")

    assert result is not None
    assert result.title == "session:ep:0"
    assert result.timestamp is None
    assert result.prompt_text == ""
    assert result.response_text == ""
    assert result.files_touched == ()
    assert result.tool_names == ()
    assert result.is_subagent is False
    assert result.project is None
    assert result.prompt_truncated is False
    assert result.response_truncated is False
