"""Malformed native records cannot steal identities or invent provider quantities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting.contracts import RawEvidence, canonical_bytes
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.opencode import OpenCodeReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    claude_tool_result,
    claude_tool_use,
    codex_call,
    codex_meta,
    codex_output,
    opencode_add_part,
    opencode_add_session,
    opencode_connect,
    opencode_tool_part,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)


def only_call(rows, call_id: str, arguments: dict):
    """The one surviving native tool call: its native ID and arguments, not an ID spelling."""
    (call,) = [row for row in rows if row.kind == "tool_call"]
    assert call.payload["call_id"] == call_id
    assert call.payload["arguments"] == arguments
    return call


def linked_result(rows: list[RawEvidence], call: RawEvidence) -> RawEvidence:
    """Pi family: the one tool result, bound both ways to ``call`` by native call ID."""
    (result,) = [row for row in rows if row.kind == "tool_result"]
    assert result.payload["call_id"] == call.payload["call_id"]
    assert result.payload["call_record_id"] == call.record_id
    assert result.payload["call_link_status"] == "unique_pending"
    assert call.origin_id == call.record_id and result.origin_id == result.record_id
    assert call.origin_domain == result.origin_domain == call.session_id
    assert call.origin_status == result.origin_status == "session_scoped"
    return result


@pytest.mark.parametrize(
    "reader_type, variable",
    [
        (PiReader, "PI_SESSION_DIR"),
        (OmpReader, "OMP_SESSIONS_DIR"),
        (PrimeReader, "PRIME_AGENT_SESSION_DIR"),
    ],
)
def test_invalid_message_shapes_do_not_swallow_following_tool_result(
    tmp_path: Path, reader_type, variable: str
) -> None:
    result = pi_tool_result("r", "call-1", "récovered")
    result["message"]["isError"] = "false"  # not a native boolean
    path = write_records(
        tmp_path / "session.jsonl",
        [
            pi_header("consumer"),
            {"type": "message", "id": "bad-message", "message": []},
            {"type": "message", "id": "bad-result", "message": {"role": "toolResult"}},
            pi_tool_call("c", "call-1", "bash", {"command": "ssgrep status"}),
            result,
        ],
    )
    reader = reader_type({variable: str(tmp_path)})
    batch = reader.read(reader.source_for(path))
    assert batch.unrecognized == {"message": 2}
    assert [row.kind for row in batch.evidence] == ["session_header", "tool_call", "tool_result"]
    assert batch.evidence[0].record_id == "session:consumer"
    call = only_call(batch.evidence, "call-1", {"command": "ssgrep status"})
    assert call.payload["name"] == "bash" and call.session_id == "consumer"
    returned = linked_result(batch.evidence, call)
    assert returned.session_id == "consumer"
    assert returned.payload["output"] == "récovered"
    assert returned.payload["output_bytes"] == len("récovered".encode())
    assert returned.payload["is_error"] is None
    assert batch.state.offset == path.stat().st_size


@pytest.mark.parametrize("unsafe", [True, "0.000001", None, {"amount": 1}])
def test_non_numeric_native_cost_never_becomes_zero_or_boolean_money(
    tmp_path: Path, unsafe: object
) -> None:
    path = write_records(
        tmp_path / "session.jsonl",
        [
            pi_header("consumer"),
            pi_tool_call(
                "c",
                "call-1",
                "bash",
                {},
                usage={
                    "input": True,
                    "output": 2.0,
                    "cacheRead": 1.25,
                    "cost": {"total": unsafe},
                },
            ),
        ],
    )
    reader = OmpReader({"OMP_SESSIONS_DIR": str(tmp_path)})
    batch = reader.read(reader.source_for(path))
    usage = next(row for row in batch.evidence if row.kind == "usage")
    assert usage.payload["tokens"] == {"output_tokens": 2}
    assert usage.payload["cost_nano_usd"] is None
    assert usage.session_id == "consumer"
    assert b'"output_tokens":2' in canonical_bytes(usage)


def test_native_sub_nanodollar_costs_round_half_even_without_float_drift(tmp_path: Path) -> None:
    path = write_records(
        tmp_path / "session.jsonl",
        [
            pi_header("consumer"),
            pi_tool_call(
                "c",
                "call-1",
                "bash",
                {},
                usage={
                    "cost": {
                        "input": 0.0000000005,
                        "output": 0.0000000015,
                        "total": 0.0000000025,
                    }
                },
            ),
        ],
    )
    reader = OmpReader({"OMP_SESSIONS_DIR": str(tmp_path)})
    batch = reader.read(reader.source_for(path))
    usage = next(row for row in batch.evidence if row.kind == "usage")
    assert usage.payload["cost_nano_usd"] == {"input": 0, "output": 2, "total": 2}


def test_codex_malformed_arguments_and_overflow_timestamp_preserve_native_identity(
    tmp_path: Path,
) -> None:
    malformed = codex_call("call-1", "mcp__recall__search_sessions", '{"query":')
    malformed["timestamp"] = 10**100
    output = codex_output("call-1", ["first", {"type": "text", "text": "second"}])
    path = write_records(tmp_path / "rollout.jsonl", [codex_meta("consumer"), malformed, output])
    reader = CodexReader({"CODEX_HOME": str(tmp_path)})
    batch = reader.read(reader.source_for(path))
    call, result = batch.evidence
    assert (call.session_id, call.record_id, call.ts_utc) == ("consumer", "call:call-1", None)
    assert call.payload["arguments"] == '{"query":'
    assert result.payload["output"] == "first\nsecond"
    assert result.payload["call_id"] == "call-1"
    assert result.payload["output_bytes"] == 12


def test_codex_task_started_changes_turn_without_reassigning_session(tmp_path: Path) -> None:
    path = write_records(
        tmp_path / "rollout.jsonl",
        [
            codex_meta("consumer"),
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-2"}},
            codex_meta("embedded-parent"),
            codex_call("call-1", "shell", {"command": "ssgrep status"}),
        ],
    )
    reader = CodexReader({"CODEX_HOME": str(tmp_path)})
    (call,) = reader.read(reader.source_for(path)).evidence
    assert call.session_id == "consumer"
    assert call.payload["turn_id"] == "turn-2"


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
    assert [row.kind for row in result.evidence] == ["tool_call"]
    only_call(result.evidence, "good", {"query": "isolated positive control"})
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
    only_call(calls, "good", {"query": "isolated positive control"})
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
    assert [row.kind for row in result.evidence] == ["session_header", "tool_call", "tool_result"]
    assert result.evidence[0].record_id == "session:session"
    call = only_call(result.evidence, "good", {"query": "isolated positive control"})
    assert call.payload["name"] == "search_sessions"
    assert linked_result(result.evidence, call).payload["output"] == "positive result"


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
    only_call(result.evidence, "valid", {"query": "positive control"})
    assert result.pending_bytes == 0
    assert result.pending_records == 0
