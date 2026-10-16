"""4.3 Codex reader: session_meta/turn_context, calls/outputs, cumulative usage."""

from __future__ import annotations

from pathlib import Path

from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.jsonl_runtimes import CodexReader, total_usage
from tests.accounting.readers.readers_fixtures import (
    append_bytes,
    codex_call,
    codex_custom_call,
    codex_meta,
    codex_output,
    codex_token_count,
    codex_turn,
    jsonl,
    write_records,
)


def reader_for(tmp_path: Path) -> CodexReader:
    return CodexReader({"CODEX_HOME": str(tmp_path / "codex")})


def rollout(tmp_path: Path, name: str = "rollout-1") -> Path:
    return tmp_path / "codex" / "sessions" / "2026" / "10" / f"{name}.jsonl"


def read_all(tmp_path: Path, records: list[dict]):
    write_records(rollout(tmp_path), records)
    reader = reader_for(tmp_path)
    [source] = reader.discover()
    return reader, source, reader.read(source)


def usage_rows(result):
    return [e for e in result.evidence if e.kind == "usage"]


def test_calls_outputs_and_context_are_linked_by_call_id(tmp_path: Path) -> None:
    _, _, result = read_all(
        tmp_path,
        [
            codex_meta("sess-1", "/work/a"),
            codex_turn("turn-1", "/work/a"),
            codex_call("call_1", "exec_command", {"cmd": "ssgrep search 'x'"}),
            codex_output("call_1", [{"type": "input_text", "text": "Wall time: 1.0 seconds"}]),
            codex_custom_call("call_2", "exec", "text('hi')"),
            codex_output("call_2", "done", custom=True),
        ],
    )
    by_id = {e.record_id: e for e in result.evidence}
    call = by_id["call:call_1"]
    assert call.session_id == "sess-1" and call.branch_id == "main"
    assert call.payload["name"] == "exec_command"
    assert call.payload["arguments"] == {"cmd": "ssgrep search 'x'"}  # decoded from JSON string
    assert call.payload["turn_id"] == "turn-1" and call.payload["cwd"] == "/work/a"
    assert by_id["result:call_1"].payload["output"] == "Wall time: 1.0 seconds"
    assert by_id["result:call_1"].payload["call_id"] == "call_1"
    custom = by_id["call:call_2"]
    assert custom.payload["arguments"] == "text('hi')"
    assert by_id["result:call_2"].payload["output"] == "done"
    assert by_id["result:call_2"].kind == "tool_result"


def test_cumulative_100_150_150_normalizes_to_150_not_400(tmp_path: Path) -> None:
    _, _, result = read_all(
        tmp_path,
        [
            codex_meta("sess-1"),
            codex_turn("turn-1"),
            codex_token_count(100, output=10),
            codex_token_count(150, output=20),
            codex_token_count(150, output=20),
        ],
    )
    rows = usage_rows(result)
    assert [r.payload["cumulative"]["total_tokens"] for r in rows] == [100, 150, 150]
    assert [r.payload["delta"]["total_tokens"] for r in rows] == [100, 50, 0]
    assert total_usage(result.evidence) == {
        "total_tokens": 150,
        "input_tokens": 130,
        "output_tokens": 20,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    # the repeated 150 is a distinct (zero-delta) update, so summing admitted rows is still 150
    ledger = EvidenceLedger()
    assert total_usage([r for r in rows if ledger.admit(r)])["total_tokens"] == 150


def test_usage_state_carries_across_incremental_reads(tmp_path: Path) -> None:
    path = write_records(
        rollout(tmp_path), [codex_meta("sess-1"), codex_turn("t1"), codex_token_count(100)]
    )
    reader = reader_for(tmp_path)
    source = reader.source_for(path)
    first = reader.read(source)
    append_bytes(path, jsonl([codex_token_count(150), codex_token_count(150)]))
    second = reader.read(source, first.state)
    assert [r.payload["delta"]["total_tokens"] for r in usage_rows(second)] == [50, 0]
    assert total_usage(first.evidence + second.evidence)["total_tokens"] == 150


def test_counter_reset_is_flagged_and_never_negative(tmp_path: Path) -> None:
    _, _, result = read_all(
        tmp_path,
        [
            codex_meta("sess-1"),
            codex_turn("t1"),
            codex_token_count(200),
            codex_token_count(30),  # new window / resumed process: counter restarted
            codex_token_count(70),
        ],
    )
    rows = usage_rows(result)
    assert [r.payload["reset"] for r in rows] == [False, True, False]
    for row in rows:
        assert all(v >= 0 for v in row.payload["delta"].values())
    assert [r.payload["delta"]["total_tokens"] for r in rows] == [200, 30, 40]
    assert total_usage(result.evidence)["total_tokens"] == 270


def test_missing_counters_stay_explicit_not_zero(tmp_path: Path) -> None:
    empty = {
        "timestamp": "t",
        "type": "event_msg",
        "payload": {"type": "token_count", "info": None},
    }
    _, _, result = read_all(tmp_path, [codex_meta("sess-1"), codex_turn("t1"), empty])
    [row] = usage_rows(result)
    assert row.payload["cumulative"] is None and row.payload["delta"] is None
    assert row.payload["unavailable"] == "missing_counters"
    assert total_usage(result.evidence) == {}


def test_namespace_is_preserved_and_unknown_items_are_unclassified(tmp_path: Path) -> None:
    _, _, result = read_all(
        tmp_path,
        [
            codex_meta("sess-1"),
            codex_call("call_1", "js", {"code": "1"}, namespace="mcp__acme_repl"),
            {"timestamp": "t", "type": "brand_new_record", "payload": {}},
            {"timestamp": "t", "type": "world_state", "payload": {"full": True}},
        ],
    )
    [call] = [e for e in result.evidence if e.kind == "tool_call"]
    assert call.payload["namespace"] == "mcp__acme_repl"
    assert result.unrecognized == {"brand_new_record": 1}


def test_sessions_with_the_same_call_id_do_not_merge(tmp_path: Path) -> None:
    write_records(
        rollout(tmp_path, "a"),
        [codex_meta("A"), codex_call("call_0", "exec_command", {"cmd": "x"})],
    )
    write_records(
        rollout(tmp_path, "b"),
        [codex_meta("B"), codex_call("call_0", "exec_command", {"cmd": "x"})],
    )
    reader = reader_for(tmp_path)
    ledger = EvidenceLedger()
    admitted = [e for s in reader.discover() for e in reader.read(s).evidence if ledger.admit(e)]
    assert len(admitted) == 2 and {e.session_id for e in admitted} == {"A", "B"}


def test_subagent_thread_records_parent_thread(tmp_path: Path) -> None:
    meta = codex_meta("child-1")
    meta["payload"]["source"] = {"subagent": {"thread_spawn": {"parent_thread_id": "parent-1"}}}
    _, _, result = read_all(tmp_path, [meta, codex_call("call_1", "exec_command", {"cmd": "x"})])
    [call] = [e for e in result.evidence if e.kind == "tool_call"]
    assert call.payload["parent_session"] == "parent-1"


def test_codex_session_override_wins_over_home_and_expands_isolated_tilde(tmp_path: Path) -> None:
    selected = write_records(
        tmp_path / "chosen" / "rollout.jsonl",
        [codex_meta("selected"), codex_call("selected", "bash", {})],
    )
    write_records(
        tmp_path / "codex-home" / "sessions" / "rollout.jsonl",
        [codex_meta("ignored"), codex_call("ignored", "bash", {})],
    )
    reader = CodexReader(
        {
            "HOME": str(tmp_path),
            "CODEX_HOME": str(tmp_path / "codex-home"),
            "CODEX_SESSIONS_DIR": "~/chosen",
        }
    )
    assert [source.path for source in reader.discover()] == [selected]
    assert reader.read(reader.source_for(selected)).evidence_ids() == ["call:selected"]


def response_message(role: str, content, **extra) -> dict:
    payload = {"type": "message", "role": role, "content": content, **extra}
    return {"timestamp": "2026-10-01T10:00:00.000Z", "type": "response_item", "payload": payload}


def test_codex_decode_rejects_non_objects_and_non_string_record_types(tmp_path: Path) -> None:
    reader = reader_for(tmp_path)
    source = reader.source_for(rollout(tmp_path))
    assert reader.decode(source, ["response_item"], {}, "offset:0") is None
    assert reader.decode(source, {"type": 9, "payload": {}}, {}, "offset:0") is None
    meta = reader.decode(source, codex_meta("s"), {}, "offset:0")
    assert meta == []


def test_codex_only_non_empty_user_messages_become_turns(tmp_path: Path) -> None:
    reader = reader_for(tmp_path)
    source = reader.source_for(rollout(tmp_path))
    blocks = [{"type": "input_text", "text": "please look"}]
    assistant = response_message("assistant", blocks, id="a1")
    assert reader.decode(source, assistant, {}, "offset:0") == []
    assert reader.decode(source, response_message("user", [], id="u0"), {}, "offset:1") == []
    decoded = reader.decode(source, response_message("user", blocks, id="u1"), {}, "offset:2")
    assert decoded is not None
    [turn] = decoded
    assert turn.kind == "user_turn" and turn.record_id == "turn:u1"
    assert turn.payload["text"] == "please look"
    ctx: dict = {}
    decoded = reader.decode(source, response_message("user", blocks), ctx, "offset:3")
    assert decoded is not None
    [first] = decoded
    assert first.record_id == "turn:n1" and ctx["turns"] == 1
