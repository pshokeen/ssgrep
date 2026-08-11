"""Tests for streaming JSONL reader."""

from pathlib import Path

from ssgrep.records import RecordType, classify_record, read_records

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_main_session():
    records = list(read_records(FIXTURES / "main-session.jsonl"))
    assert len(records) > 0


def test_classify_record_types():
    assert classify_record({"type": "user"}) == RecordType.USER
    assert classify_record({"type": "assistant"}) == RecordType.ASSISTANT
    assert classify_record({"type": "unknown-type"}) == RecordType.UNKNOWN


def test_malformed_line():
    records = list(read_records(FIXTURES / "malformed-line.jsonl"))
    assert len(records) >= 4


def test_oversized_line():
    records = list(read_records(FIXTURES / "oversized-block.jsonl"))
    assert len(records) >= 2


def test_empty_thinking_blocks():
    records = list(read_records(FIXTURES / "thinking-signatures.jsonl"))
    for rec in records:
        if rec.get("type") == "assistant":
            for block in rec.get("message", {}).get("content", []):
                if block.get("type") == "thinking":
                    assert block.get("thinking", "") == ""
