"""Malformed native fields stay unknown without hiding later valid evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    claude_tool_result,
    claude_tool_use,
    codex_call,
    codex_meta,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)

JSONL_READERS = (ClaudeReader, CodexReader, PiReader, OmpReader, PrimeReader)
PI_READERS = (PiReader, OmpReader, PrimeReader)


def native_call(reader_type, call_id: str = "good") -> dict:
    arguments = {"query": "isolated positive control"}
    if reader_type is ClaudeReader:
        return claude_tool_use("session", call_id, "search_sessions", arguments)
    if reader_type is CodexReader:
        return codex_call(call_id, "search_sessions", arguments)
    return pi_tool_call("entry", call_id, "search_sessions", arguments)


@pytest.mark.parametrize("reader_type", JSONL_READERS)
def test_nonobjects_and_untyped_records_preserve_following_native_call(
    tmp_path: Path, reader_type
) -> None:
    path = tmp_path / "fallback-session.jsonl"
    records = [None, [], 7, {}, {"type": []}, {"type": {}}, native_call(reader_type)]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    reader = reader_type({"HOME": str(tmp_path)})
    result = reader.read(reader.source_for(path))
    assert result.unrecognized == {"<non-object>": 3, "<untyped>": 3}
    assert result.evidence_ids() == ["call:good"]
    assert result.evidence[0].payload["arguments"] == {"query": "isolated positive control"}
    assert result.state.offset == path.stat().st_size
    assert result.pending_bytes == 0


@pytest.mark.parametrize("content", [4, True, {"type": "tool_use", "id": "not-a-list"}])
@pytest.mark.parametrize("reader_type", (ClaudeReader, *PI_READERS))
def test_malformed_assistant_content_does_not_hide_independent_usage_or_next_call(
    tmp_path: Path, reader_type, content: object
) -> None:
    malformed = native_call(reader_type, "malformed")
    malformed["message"]["content"] = content
    malformed["message"]["usage"] = (
        {"input_tokens": 11} if reader_type is ClaudeReader else {"input": 11}
    )
    path = write_records(tmp_path / "session.jsonl", [malformed, native_call(reader_type)])
    reader = reader_type({"HOME": str(tmp_path)})
    result = reader.read(reader.source_for(path))
    usage = [row for row in result.evidence if row.kind == "usage"]
    calls = [row for row in result.evidence if row.kind == "tool_call"]
    assert len(usage) == 1
    assert usage[0].payload["tokens"] == {"input_tokens": 11}
    assert [row.record_id for row in calls] == ["call:good"]
    assert result.state.offset == path.stat().st_size


def test_claude_mixed_blocks_keep_only_identifiable_calls_and_results(tmp_path: Path) -> None:
    call = native_call(ClaudeReader)
    call["timestamp"] = "2026-10-01T10:00:00.125"
    call["message"]["content"][:0] = [
        None,
        "quoted ssgrep search is not a call",
        {"type": "text", "text": "thinking"},
        {"type": "tool_use", "id": 9, "name": "bash"},
        {"type": "tool_use", "id": "bad-name", "name": []},
    ]
    reply = claude_tool_result("session", "good", ["récovered", {"text": "next"}, 9])
    reply["message"]["content"].insert(0, {"type": "tool_result", "tool_use_id": 9})
    path = write_records(tmp_path / "session.jsonl", [{"type": "system"}, call, reply])
    reader = ClaudeReader({"HOME": str(tmp_path)})
    result = reader.read(reader.source_for(path))
    assert result.unrecognized == {}
    assert result.evidence_ids() == ["call:good", "result:good"]
    accepted, returned = result.evidence
    assert accepted.ts_utc == "2026-10-01T10:00:00.125Z"
    assert returned.payload["output"] == "récovered\nnext"
    assert returned.payload["output_bytes"] == len("récovered\nnext".encode())


@pytest.mark.parametrize("reader_type", PI_READERS)
def test_pi_family_mixed_roles_and_blocks_do_not_create_phantom_calls(
    tmp_path: Path, reader_type
) -> None:
    call = native_call(reader_type)
    call["message"]["content"][:0] = [
        None,
        "quoted text",
        {"type": "text", "text": "not a tool"},
        {"type": "toolCall", "id": 9, "name": "bash"},
    ]
    path = write_records(
        tmp_path / "session.jsonl",
        [
            pi_header("session"),
            {"type": "message", "id": "system", "message": {"role": "system"}},
            {"type": "message", "id": "empty", "message": {"role": "user", "content": 9}},
            call,
            pi_tool_result("reply", "good", "positive result"),
        ],
    )
    reader = reader_type({"HOME": str(tmp_path)})
    result = reader.read(reader.source_for(path))
    assert result.evidence_ids() == ["session:session", "call:good", "result:good"]
    assert result.evidence[-1].payload["output"] == "positive result"


@pytest.mark.parametrize("subagent", [[], 7, "not an object"])
def test_codex_invalid_parent_shape_never_reassigns_source_session(
    tmp_path: Path, subagent: object
) -> None:
    meta = codex_meta("native-session")
    meta["payload"]["source"] = {"subagent": subagent}
    path = write_records(
        tmp_path / "rollout.jsonl",
        [
            meta,
            {"type": "response_item", "payload": {"type": []}},
            {"type": "response_item", "payload": {"type": "reasoning"}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": 9},
            },
            native_call(CodexReader),
        ],
    )
    reader = CodexReader({"HOME": str(tmp_path)})
    result = reader.read(reader.source_for(path))
    assert result.unrecognized == {"response_item": 1}
    assert result.evidence_ids() == ["call:good"]
    assert result.evidence[0].session_id == "native-session"
    assert "parent_session" not in result.evidence[0].payload


def test_ledger_drain_distinguishes_copies_from_same_id_in_another_session(
    tmp_path: Path,
) -> None:
    original = claude_tool_use("first", "same-call", "search_sessions", {"query": "one"})
    other_session = claude_tool_use("second", "same-call", "search_sessions", {"query": "one"})
    reader = ClaudeReader({"HOME": str(tmp_path)})
    batches = []
    for name, record in (("original", original), ("copy", original), ("other", other_session)):
        path = write_records(tmp_path / f"{name}.jsonl", [record])
        batches.extend(reader.read(reader.source_for(path)).evidence)
    ledger = EvidenceLedger()
    admitted = list(ledger.drain(batches))
    assert [row.session_id for row in admitted] == ["first", "second"]
    assert admitted[0].source_id != admitted[1].source_id
    assert ledger.duplicates == 1
    assert ledger.conflict_count == 0
