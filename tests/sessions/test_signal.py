"""Unit tests for signal/noise classification and markup stripping."""

from __future__ import annotations

import pytest

from ssgrep.sessions import signal
from ssgrep.utilities.types import ContentType


def test_meta_records_are_always_noise():
    result = signal.classify_signal(
        {"isMeta": True, "type": "user"}, {"type": "text", "text": "hidden"}
    )
    assert result == signal.SignalResult(False)
    assert result.content_type is None


def test_user_text_is_prompt_signal_with_markup_removed():
    result = signal.classify_signal(
        {"type": "user"},
        {
            "type": "text",
            "text": " <command-name>/model</command-name> actual question ",
        },
    )
    assert result == signal.SignalResult(True, "actual question", ContentType.PROMPT)


@pytest.mark.parametrize("block_type", ["tool_result", "image"])
def test_user_nontext_blocks_are_noise(block_type: str):
    assert signal.classify_signal({"type": "user"}, {"type": block_type}) == signal.SignalResult(
        False
    )


def test_assistant_text_is_response_signal():
    result = signal.classify_signal(
        {"type": "assistant"},
        {"type": "text", "text": "<command-message>x</command-message> answer"},
    )
    assert result.is_signal
    assert result.text == "answer"
    assert result.content_type is ContentType.RESPONSE


@pytest.mark.parametrize("block_type", ["thinking", "tool_use"])
def test_assistant_internal_blocks_are_noise(block_type: str):
    assert not signal.classify_signal({"type": "assistant"}, {"type": block_type}).is_signal


def test_away_summary_is_response_signal_for_string_content():
    result = signal.classify_signal(
        {
            "type": "system",
            "subtype": "away_summary",
            "content": " <command-name>/compact</command-name> recap ",
        }
    )
    assert result == signal.SignalResult(True, "recap", ContentType.RESPONSE)


def test_away_summary_with_nonstring_content_has_empty_signal_text():
    result = signal.classify_signal(
        {"type": "system", "subtype": "away_summary", "content": ["unexpected"]}
    )
    assert result == signal.SignalResult(True, "", ContentType.RESPONSE)


def test_local_command_and_unrecognized_shapes_are_noise():
    cases = [
        ({"type": "system", "subtype": "local_command"}, None),
        ({"type": "system", "subtype": "other"}, None),
        ({"type": "user"}, None),
        ({"type": "user"}, {"type": "unknown"}),
        ({"type": "assistant"}, None),
        ({"type": "assistant"}, {"type": "unknown"}),
        ({"type": "future"}, {"type": "text", "text": "ignored"}),
        ({}, None),
    ]
    for record, block in cases:
        assert signal.classify_signal(record, block) == signal.SignalResult(False)


def test_strip_markup_removes_both_known_wrappers_and_trims():
    text = (
        "  before <command-name>/x\n</command-name> middle "
        "<command-message>payload</command-message> after  "
    )
    assert signal.strip_markup(text) == "before  middle  after"
