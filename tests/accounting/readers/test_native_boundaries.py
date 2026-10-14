"""Malformed native records cannot steal identities or invent provider quantities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    codex_call,
    codex_meta,
    codex_output,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)


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
    assert {row.record_id for row in batch.evidence} == {
        "session:consumer",
        "call:call-1",
        "result:call-1",
    }
    returned = next(row for row in batch.evidence if row.kind == "tool_result")
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
