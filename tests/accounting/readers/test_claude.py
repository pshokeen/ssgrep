"""4.2 Claude reader: raw tool_use/tool_result, usage, errors, child identity."""

from __future__ import annotations

import shutil
from pathlib import Path

from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader
from tests.accounting.readers.readers_fixtures import (
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    write_records,
)


def reader_for(tmp_path: Path) -> ClaudeReader:
    return ClaudeReader({"CLAUDE_CONFIG_DIR": str(tmp_path / "claude")})


def projects(tmp_path: Path) -> Path:
    return tmp_path / "claude" / "projects" / "-work-demo"


def test_tool_call_result_usage_and_error_are_separate_linked_kinds(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 9,
        "output_tokens": 496,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 50335,
        "output_tokens_details": {"thinking_tokens": 335},
    }
    write_records(
        projects(tmp_path) / "s1.jsonl",
        [
            claude_prompt("s1", "find notes"),
            claude_tool_use("s1", "toolu_1", "Bash", {"command": "ls"}, usage=usage),
            claude_tool_result("s1", "toolu_1", [{"type": "text", "text": "boom"}], is_error=True),
            claude_tool_use("s1", "toolu_2", "Read", {"path": "a"}, message_id="m2"),
            claude_tool_result("s1", "toolu_2", "fine", is_error=False),
            claude_tool_result("s1", "toolu_3", "unflagged"),
        ],
    )
    reader = reader_for(tmp_path)
    [source] = reader.discover()
    evidence = {e.record_id: e for e in reader.read(source).evidence}

    call = evidence["call:toolu_1"]
    assert call.kind == "tool_call" and call.payload["name"] == "Bash"
    assert call.payload["arguments"] == {"command": "ls"}
    assert call.payload["cwd"] == "/work/demo"
    result = evidence["result:toolu_1"]
    assert result.kind == "tool_result"
    assert result.payload["call_id"] == call.payload["call_id"] == "toolu_1"
    assert result.payload["is_error"] is True and result.payload["output"] == "boom"
    assert result.payload["output_bytes"] == 4
    assert evidence["result:toolu_2"].payload["is_error"] is False
    assert evidence["result:toolu_3"].payload["is_error"] is None  # unknown stays unknown
    assert evidence["usage:msg_1"].payload["tokens"] == {
        "input_tokens": 9,
        "output_tokens": 496,
        "cache_read_tokens": 3,
        "cache_write_tokens": 50335,
        "reasoning_tokens": 335,
    }
    assert evidence["turn:u-1"].kind == "user_turn"
    assert all(e.runtime == "claude" and e.session_id == "s1" for e in evidence.values())
    assert call.ts_utc == "2026-10-01T10:00:00.000Z"
    assert call.locator.startswith("offset:")


def test_children_sharing_session_id_stay_distinct(tmp_path: Path) -> None:
    base = projects(tmp_path)
    shared = "parent-session"
    main = write_records(
        base / f"{shared}.jsonl",
        [claude_tool_use(shared, "toolu_1", "Bash", {"command": "echo main"})],
    )
    child_a = write_records(
        base / shared / "subagents" / "agent-aaa.jsonl",
        [claude_tool_use(shared, "toolu_1", "Bash", {"command": "echo a"}, child="aaa")],
    )
    child_b = write_records(
        base / shared / "subagents" / "agent-bbb.jsonl",
        [claude_tool_use(shared, "toolu_1", "Bash", {"command": "echo b"}, child="bbb")],
    )
    reader = reader_for(tmp_path)
    ledger = EvidenceLedger()
    admitted = []
    for path in (main, child_a, child_b):
        for item in reader.read(reader.source_for(path)).evidence:
            if ledger.admit(item):
                admitted.append(item)
    assert len(admitted) == 3  # same call id, three different executions
    assert ledger.conflicts() == []
    assert len({e.source_id for e in admitted}) == 3
    assert {e.branch_id for e in admitted} == {"main", "child:aaa", "child:bbb"}
    child = next(e for e in admitted if e.branch_id == "child:aaa")
    assert child.session_id == f"{shared}/aaa"
    assert child.payload["parent_session"] == shared


def test_only_identical_native_copies_of_a_child_dedupe(tmp_path: Path) -> None:
    base = projects(tmp_path)
    original = write_records(
        base / "p" / "subagents" / "agent-aaa.jsonl",
        [claude_tool_use("p", "toolu_1", "Bash", {"command": "echo a"}, child="aaa")],
    )
    copy = base / "p-copy" / "subagents" / "agent-aaa.jsonl"
    copy.parent.mkdir(parents=True)
    shutil.copy(original, copy)
    reader = reader_for(tmp_path)
    ledger = EvidenceLedger()
    admitted = [
        e
        for p in (original, copy)
        for e in reader.read(reader.source_for(p)).evidence
        if ledger.admit(e)
    ]
    assert len(admitted) == 1 and ledger.duplicates == 1


def test_discovery_skips_tool_results_and_memory_but_keeps_subagents(tmp_path: Path) -> None:
    base = projects(tmp_path)
    keep = [
        write_records(base / "s.jsonl", [claude_prompt("s", "hi")]),
        write_records(base / "s" / "subagents" / "agent-1.jsonl", [claude_prompt("s", "x")]),
        write_records(
            base / "s" / "subagents" / "workflows" / "agent-2.jsonl", [claude_prompt("s", "y")]
        ),
    ]
    write_records(base / "memory" / "m.jsonl", [claude_prompt("m", "no")])
    write_records(base / "s" / "tool-results" / "t.jsonl", [claude_prompt("t", "no")])
    write_records(base / "s" / "workflows" / "w.jsonl", [claude_prompt("w", "no")])
    found = {s.path for s in reader_for(tmp_path).discover()}
    assert found == set(keep)
    roles = {s.path: s.role for s in reader_for(tmp_path).discover()}
    assert roles[keep[0]] == "main" and roles[keep[1]] == "child"


def test_symlinked_transcripts_are_not_followed(tmp_path: Path) -> None:
    base = projects(tmp_path)
    real = write_records(tmp_path / "elsewhere" / "x.jsonl", [claude_prompt("x", "hi")])
    base.mkdir(parents=True)
    (base / "link.jsonl").symlink_to(real)
    assert reader_for(tmp_path).discover() == []


def test_unknown_record_types_are_counted_not_guessed(tmp_path: Path) -> None:
    write_records(
        projects(tmp_path) / "s.jsonl",
        [
            {"type": "mystery-format", "payload": 1},
            {"type": "mystery-format", "payload": 2},
            claude_tool_use("s", "toolu_1", "Bash", {"command": "ls"}),
        ],
    )
    reader = reader_for(tmp_path)
    [source] = reader.discover()
    result = reader.read(source)
    assert result.unrecognized == {"mystery-format": 2}
    assert [e.record_id for e in result.evidence] == ["call:toolu_1"]


def test_malformed_complete_line_is_a_gap_and_reading_continues(tmp_path: Path) -> None:
    path = projects(tmp_path) / "s.jsonl"
    path.parent.mkdir(parents=True)
    good = claude_tool_use("s", "toolu_1", "Bash", {"command": "ls"})
    import json

    path.write_bytes(b"{not json\n" + json.dumps(good).encode() + b"\n")
    reader = reader_for(tmp_path)
    result = reader.read(reader.source_for(path))
    assert [g.reason for g in result.gaps] == ["malformed_json"]
    assert [e.record_id for e in result.evidence] == ["call:toolu_1"]


def test_float_values_never_enter_evidence(tmp_path: Path) -> None:
    path = projects(tmp_path) / "s.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"assistant","sessionId":"s","timestamp":"2026-10-01T10:00:00Z",'
        '"message":{"id":"m","content":[{"type":"tool_use","id":"t","name":"X",'
        '"input":{"a":1.5,"b":2.0}}]}}\n'
    )
    reader = reader_for(tmp_path)
    [call] = reader.read(reader.source_for(path)).evidence
    assert call.payload["arguments"] == {"a": "1.5", "b": 2}
    from ssgrep.accounting.ledger.contracts import digest_of

    digest_of(call)  # canonical serialization rejects floats; must not raise


def test_claude_decode_rejects_records_without_a_string_type(tmp_path: Path) -> None:
    reader = reader_for(tmp_path)
    source = reader.source_for(projects(tmp_path) / "s1.jsonl")
    assert reader.decode(source, {"type": 3, "uuid": "u"}, {}, "offset:0") is None
    assert reader.decode(source, {"uuid": "u"}, {}, "offset:0") is None
    decoded = reader.decode(source, claude_prompt("s1", "hello"), {}, "offset:0")
    assert decoded is not None and [e.kind for e in decoded] == ["user_turn"]
