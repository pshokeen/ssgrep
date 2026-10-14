"""Non-JSON numeric constants are explicit gaps, never quantities or lost later calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    claude_tool_use,
    codex_call,
    opencode_add_part,
    opencode_add_session,
    opencode_connect,
    opencode_tool_part,
    pi_tool_call,
)


def call_record(reader_type, call_id: str, query: str) -> dict:
    if reader_type is ClaudeReader:
        return claude_tool_use("session", call_id, "search_sessions", {"query": query})
    if reader_type is CodexReader:
        record = codex_call(call_id, "search_sessions", {})
        record["payload"]["arguments"] = {"query": query}
        return record
    if reader_type is OpenCodeReader:
        return opencode_tool_part(call_id, "search_sessions", {"query": query})
    return pi_tool_call(call_id, call_id, "search_sessions", {"query": query})


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    "reader_type", (ClaudeReader, CodexReader, OmpReader, OpenCodeReader, PiReader, PrimeReader)
)
def test_nonfinite_json_is_a_gap_and_following_native_call_survives(
    tmp_path: Path, reader_type, literal: str
) -> None:
    invalid = json.dumps(call_record(reader_type, "invalid", "MALFORMED_NUMBER"))
    invalid = invalid.replace('"MALFORMED_NUMBER"', literal)
    valid = call_record(reader_type, "valid", "positive control")
    reader = reader_type({"HOME": str(tmp_path)})
    if reader_type is OpenCodeReader:
        path = tmp_path / "opencode.db"
        writer = opencode_connect(path)
        try:
            opencode_add_session(writer, "session")
            opencode_add_part(writer, "a-invalid", "message", "session", invalid)
            opencode_add_part(writer, "b-valid", "message", "session", valid)
        finally:
            writer.close()
    else:
        path = tmp_path / "session.jsonl"
        path.write_text(invalid + "\n" + json.dumps(valid) + "\n", encoding="utf-8")
    result = reader.read(reader.source_for(path))
    assert [gap.reason for gap in result.gaps] == ["malformed_json"]
    calls = [row for row in result.evidence if row.kind == "tool_call"]
    assert [row.record_id for row in calls] == ["call:valid"]
    assert calls[0].payload["arguments"] == {"query": "positive control"}
    assert result.pending_bytes == 0
    assert result.pending_records == 0
