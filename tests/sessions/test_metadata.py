"""Unit tests for metadata normalization and harvesting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.sessions import metadata


@pytest.mark.parametrize("value", [None, 12, [], {}])
def test_normalize_title_rejects_non_strings(value):
    assert metadata.normalize_title(value) == ""


def test_normalize_title_strips_wrappers_ansi_controls_and_whitespace():
    raw = (
        "  <command-name arg='x'>discard me</command-name> "
        "<system-reminder>\nUseful\t title  \x1b[1mBOLD\x1b[0m\x00  "
    )
    assert metadata.normalize_title(raw) == "Useful title BOLD"
    assert metadata.normalize_title("<tag>only wrapper</tag>") == ""
    assert metadata.normalize_title("   \n\t ") == ""


def test_normalize_title_caps_adversarial_input():
    normalized = metadata.normalize_title("x" * 3000)
    assert normalized == "x" * 2000


def test_load_agent_meta_success_empty_description_missing_and_malformed(tmp_path: Path):
    good = tmp_path / "agent.meta.json"
    good.write_text(
        json.dumps(
            {
                "agentType": "reviewer",
                "name": "Ada",
                "model": "opus",
                "description": "<reminder>\nReview  carefully",
            }
        )
    )
    loaded = metadata.load_agent_meta(good)
    assert loaded == metadata.AgentMeta(
        agent_type="reviewer",
        name="Ada",
        model="opus",
        description="Review carefully",
    )

    empty = tmp_path / "empty-description.meta.json"
    empty.write_text("{}")
    assert metadata.load_agent_meta(empty) == metadata.AgentMeta()
    assert metadata.load_agent_meta(tmp_path / "missing.json") is None

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert metadata.load_agent_meta(malformed) is None


def test_derive_files_touched_filters_shapes_tools_and_duplicates():
    records = [
        {
            "isMeta": True,
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "poison.py"},
                    }
                ]
            },
        },
        {"message": {"content": "not block content"}},
        {
            "message": {
                "content": [
                    "not a dictionary",
                    {"type": "text", "text": "ignore"},
                    {"type": "tool_use", "name": "Bash", "input": {"file_path": "no"}},
                    {"type": "tool_use", "name": "Read", "input": []},
                    {"type": "tool_use", "name": "Edit", "input": {}},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "b.py"}},
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "a.py"}},
                    {"type": "tool_use", "name": "Edit", "input": {"file_path": "b.py"}},
                ]
            }
        },
    ]
    assert metadata.derive_files_touched(records) == ("a.py", "b.py")


def test_derive_files_touched_reads_direct_adapter_files_and_lowercase_names():
    records = [
        {"_files_touched": "direct.txt"},
        {"_files_touched": ["z.py", "a.py", "z.py"]},
        {
            "message": {
                "content": [
                    {"type": "tool_use", "name": "read", "input": {"path": "named.py"}},
                    {"type": "tool_use", "name": "bogus", "input": {"file_path": "ignored.py"}},
                    {"type": "tool_use", "name": "edit", "input": {"filename": "fn.py"}},
                    {"type": "tool_use", "name": "write", "input": {"file_path": "w.py"}},
                    {"type": "tool_use", "name": "apply_patch", "input": {"file_path": "p.py"}},
                ]
            }
        },
    ]
    assert metadata.derive_files_touched(records) == (
        "a.py",
        "direct.txt",
        "fn.py",
        "named.py",
        "p.py",
        "w.py",
        "z.py",
    )


def test_harvest_metadata_collects_values_tools_files_and_title_precedence():
    records = [
        {
            "isMeta": True,
            "type": "custom-title",
            "custom-title": "poison",
            "timestamp": "2020-01-01T00:00:00Z",
            "gitBranch": "bad",
            "cwd": "/bad",
        },
        {"type": "last-prompt", "last-prompt": "Last\n Prompt", "timestamp": "invalid"},
        {"type": "ai-title", "ai-title": "AI Title"},
        {"type": "custom-title", "custom-title": " Custom\tTitle "},
        {
            "type": "user",
            "message": {"content": "User prompt"},
            "timestamp": "2025-01-02T03:04:05Z",
            "gitBranch": "main",
            "cwd": "/work",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    "not a block",
                    {"type": "text", "text": "response"},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "b.py"}},
                    {"type": "tool_use", "name": "", "input": {}},
                    {"type": "tool_use", "input": {}},
                ]
            },
        },
        {"type": "assistant", "message": {"content": "plain response"}},
    ]
    result = metadata.harvest_metadata(records, episode_index=7)
    assert result == metadata.EpisodeMetadata(
        title="Custom Title",
        timestamp=datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC),
        git_branch="main",
        cwd="/work",
        files_touched=("b.py",),
        tool_names=("Read",),
    )


def test_harvest_user_title_from_list_skips_invalid_and_empty_blocks():
    long_text = "word " * 30
    result = metadata.harvest_metadata(
        [
            {
                "type": "user",
                "message": {
                    "content": [
                        "bad",
                        {"type": "thinking", "text": "hidden"},
                        {"type": "text", "text": ""},
                        {"type": "text", "text": long_text},
                    ]
                },
            },
            {"type": "user", "message": {"content": "later title"}},
        ]
    )
    assert result.title == metadata.normalize_title(long_text)[:100].strip()
    assert len(result.title) <= 100


@pytest.mark.parametrize(
    ("records", "episode_index", "expected"),
    [
        ([{"type": "ai-title", "ai-title": "AI"}], None, "AI"),
        ([{"type": "last-prompt", "last-prompt": "Last"}], None, "Last"),
        ([{"type": "user", "message": {"content": "String title"}}], None, "String title"),
        ([{"type": "user", "message": {"content": ""}}], None, "Untitled Episode"),
        ([], 3, "Episode 3"),
    ],
)
def test_harvest_title_fallbacks(records, episode_index, expected):
    assert metadata.harvest_metadata(records, episode_index).title == expected


def test_harvest_retries_missing_branch_and_cwd_until_values_appear():
    result = metadata.harvest_metadata(
        [
            {"type": "queue-operation", "gitBranch": None, "cwd": None},
            {"type": "queue-operation", "gitBranch": "dev", "cwd": "/repo"},
        ]
    )
    assert result.git_branch == "dev"
    assert result.cwd == "/repo"
