"""Tests for signal/noise classification."""

import json
from pathlib import Path

from ssgrep.signal import classify_signal, strip_markup

FIXTURES = Path(__file__).parent / "fixtures"


def test_user_text_is_signal():
    record = {"type": "user"}
    block = {"type": "text", "text": "Hello"}
    result = classify_signal(record, block)
    assert result.is_signal
    assert result.text == "Hello"


def test_assistant_thinking_is_noise():
    record = {"type": "assistant"}
    block = {"type": "thinking", "thinking": ""}
    result = classify_signal(record, block)
    assert not result.is_signal


def test_tool_result_is_noise():
    record = {"type": "user"}
    block = {"type": "tool_result", "content": "file contents"}
    result = classify_signal(record, block)
    assert not result.is_signal


def test_image_is_noise():
    record = {"type": "user"}
    block = {"type": "image"}
    result = classify_signal(record, block)
    assert not result.is_signal


def test_away_summary_is_signal():
    # away_summary records carry content as a top-level string field
    record = {
        "type": "system",
        "subtype": "away_summary",
        "content": "Recap: the user asked how to configure the widget cache TTL",
    }
    result = classify_signal(record)
    assert result.is_signal
    assert "Recap" in result.text


def test_strip_markup():
    text = "Before <command-name>/model</command-name> after"
    assert strip_markup(text) == "Before  after"


def test_ismeta_record_is_not_signal():
    """isMeta records must never surface as signal, even carrying a
    user/text block that would otherwise clearly qualify.

    Every other fixture in the repo has isMeta: false on every record (see
    fixtures/manifest.md), so nothing previously distinguished "the isMeta
    guard filtered this" from "nothing here would have been signal anyway".
    ismeta-record.jsonl mirrors the real corpus's only observed isMeta:true
    shape (type=user); this test proves the guard itself, not just the
    absence of qualifying content, by also checking the identical block
    IS signal once isMeta is turned off.
    """
    records = []
    with open(FIXTURES / "ismeta-record.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Line 2 has list-form content: pull its one text block, which is
    # exactly the shape test_user_text_is_signal proves IS signal.
    list_record = next(r for r in records if isinstance(r["message"]["content"], list))
    block = list_record["message"]["content"][0]
    assert block["type"] == "text"
    assert list_record["isMeta"] is True

    result = classify_signal(list_record, block)
    assert not result.is_signal
    assert result.source_type == "isMeta"
    assert result.text == ""

    # Selectivity: the identical record/block pair, with only isMeta
    # flipped off, IS signal -- proving the guard (not the content) is
    # what suppressed it above.
    non_meta_record = dict(list_record, isMeta=False)
    control = classify_signal(non_meta_record, block)
    assert control.is_signal
    assert control.text == block["text"]


def test_thinking_blocks_empty():
    records = []
    with open(FIXTURES / "thinking-signatures.jsonl") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    for rec in records:
        if rec.get("type") == "assistant":
            for block in rec.get("message", {}).get("content", []):
                if block.get("type") == "thinking":
                    assert block.get("thinking", "") == ""
