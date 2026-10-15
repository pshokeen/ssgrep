"""Same-identity/different-digest: streamed update vs real conflict (shapes from real data).

Real findings that drive these tests: Claude writes one record per streamed content block with
the same ``message.id`` and a growing ``usage`` (``output_tokens`` 2,2,344); Codex forks embed a
second ``session_meta``; Prime reuses ``call_0`` many times per session and rewrites ``lineage``
status (running, completed, deleted).
"""

from __future__ import annotations

from pathlib import Path

from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.codex import CodexReader, total_usage
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    append_bytes,
    claude_tool_use,
    codex_call,
    codex_meta,
    codex_token_count,
    codex_turn,
    jsonl,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)


def stream(session: str, message: str, outputs: list[int | None], tool: str = "t1") -> list[dict]:
    """One streamed assistant message: every record repeats id, usage grows, same tool block."""
    records = []
    for output in outputs:
        usage = None if output is None else {"input_tokens": 9, "output_tokens": output}
        records.append(
            claude_tool_use(
                session, tool, "Bash", {"command": "ls"}, message_id=message, usage=usage
            )
        )
    return records


def claude(tmp_path: Path) -> ClaudeReader:
    return ClaudeReader({"CLAUDE_CONFIG_DIR": str(tmp_path / "claude")})


def path_for(tmp_path: Path, name: str = "s1") -> Path:
    return tmp_path / "claude" / "projects" / "-p" / f"{name}.jsonl"


def admit_all(reader, paths: list[Path], ledger: EvidenceLedger | None = None):
    ledger = ledger or EvidenceLedger()
    admitted = []
    for path in paths:
        for item in reader.read(reader.source_for(path)).evidence:
            if ledger.admit(item):
                admitted.append(item)
    return ledger, admitted


def test_streaming_usage_2_2_344_resolves_to_final_counted_once(tmp_path: Path) -> None:
    path = write_records(path_for(tmp_path), stream("s1", "msg_1", [2, 2, 344]))
    reader = claude(tmp_path)
    result = reader.read(reader.source_for(path))
    usage = [e for e in result.evidence if e.kind == "usage"]
    assert [e.record_id for e in usage] == ["usage:msg_1"]
    assert usage[0].payload["tokens"] == {"input_tokens": 9, "output_tokens": 344}
    assert usage[0].payload["streaming_updates"] == 3
    # the tool block repeated on every streamed record is still one call
    ledger, admitted = admit_all(reader, [path])
    assert sorted(e.record_id for e in admitted) == ["call:t1", "usage:msg_1"]
    assert ledger.conflict_count == 0


def test_stream_split_across_read_cycles_supersedes_instead_of_conflicting(
    tmp_path: Path,
) -> None:
    records = stream("s1", "msg_1", [2, 7, 204])
    path = write_records(path_for(tmp_path), records[:2])
    reader = claude(tmp_path)
    source = reader.source_for(path)
    ledger = EvidenceLedger()
    first = reader.read(source)
    assert [ledger.admit(e) for e in first.evidence if e.kind == "usage"] == [True]
    append_bytes(path, jsonl(records[2:]))
    second = reader.read(source, first.state)
    assert [ledger.admit(e) for e in second.evidence if e.kind == "usage"] == [False]
    assert ledger.conflict_count == 0 and ledger.superseded == 1
    [(final, updates)] = list(ledger.final_usage())
    assert final.payload["tokens"]["output_tokens"] == 204
    assert updates == 3


def test_older_prefix_copy_read_later_is_not_a_conflict(tmp_path: Path) -> None:
    full = write_records(path_for(tmp_path, "full"), stream("s1", "msg_1", [2, 344]))
    prefix = write_records(path_for(tmp_path, "prefix"), stream("s1", "msg_1", [2]))
    ledger, _ = admit_all(claude(tmp_path), [full, prefix])
    assert ledger.conflict_count == 0
    [(final, _)] = list(ledger.final_usage())
    assert final.payload["tokens"]["output_tokens"] == 344


def test_genuine_decrease_5_then_3_is_the_only_conflict(tmp_path: Path) -> None:
    records = stream("s1", "msg_1", [5, 3]) + stream("s1", "msg_2", [2, 2, 344], tool="t2")
    path = write_records(path_for(tmp_path), records)
    reader = claude(tmp_path)
    ledger, admitted = admit_all(reader, [path])
    [conflict] = ledger.conflicts()
    assert conflict.record_id == "usage:msg_1" and len(conflict.digests) == 2
    assert ledger.conflict_count == 1
    ids = sorted(e.record_id for e in admitted)
    assert ids == ["call:t1", "call:t2", "usage:msg_1", "usage:msg_2"]  # counted once each
    finals = {e.record_id: e.payload["tokens"]["output_tokens"] for e, _ in ledger.final_usage()}
    assert finals == {"usage:msg_1": 5, "usage:msg_2": 344}  # a conflict never lowers the total


def test_mixed_field_movement_is_a_conflict(tmp_path: Path) -> None:
    one = claude_tool_use("s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 5})
    two = claude_tool_use(
        "s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 9, "input_tokens": 1}
    )
    other = write_records(path_for(tmp_path, "b"), [two])
    first = write_records(path_for(tmp_path, "a"), [one, {**one, "uuid": "x"}])
    # field set differs (input_tokens appears) but nothing decreases: still a legal stream
    ledger, _ = admit_all(claude(tmp_path), [first, other])
    assert ledger.conflict_count == 0
    down = claude_tool_use(
        "s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 8, "input_tokens": 2}
    )
    mixed = write_records(path_for(tmp_path, "c"), [down])
    ledger, _ = admit_all(claude(tmp_path), [other, mixed])
    assert ledger.conflict_count == 1  # output went down while input_tokens moved the other way


def test_model_change_for_same_message_id_is_a_conflict(tmp_path: Path) -> None:
    one = claude_tool_use("s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 5})
    two = claude_tool_use("s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 9})
    two["message"]["model"] = "another-model"
    path = write_records(path_for(tmp_path), [one, two])
    ledger, _ = admit_all(claude(tmp_path), [path])
    assert [c.record_id for c in ledger.conflicts()] == ["usage:m"]


def test_identical_tool_call_and_result_duplicates_still_dedupe(tmp_path: Path) -> None:
    records = stream("s1", "msg_1", [2, 344])
    a = write_records(path_for(tmp_path, "a"), records)
    b = write_records(path_for(tmp_path, "b"), records)
    ledger, admitted = admit_all(claude(tmp_path), [a, b])
    assert sorted(e.record_id for e in admitted) == ["call:t1", "usage:msg_1"]
    assert ledger.conflict_count == 0 and ledger.duplicates >= 2
    assert {e.kind for e in admitted} == {"tool_call", "usage"}


def test_changed_tool_call_arguments_remain_a_true_conflict(tmp_path: Path) -> None:
    a = write_records(path_for(tmp_path, "a"), [claude_tool_use("s1", "t1", "Bash", {"c": "ls"})])
    b = write_records(path_for(tmp_path, "b"), [claude_tool_use("s1", "t1", "Bash", {"c": "pwd"})])
    ledger, _ = admit_all(claude(tmp_path), [a, b])
    assert [c.record_id for c in ledger.conflicts()] == ["call:t1"]


# ----------------------------------------------------------------------- codex


def codex_reader(tmp_path: Path) -> CodexReader:
    return CodexReader({"CODEX_HOME": str(tmp_path / "codex")})


def rollout(tmp_path: Path, name: str) -> Path:
    return tmp_path / "codex" / "sessions" / f"{name}.jsonl"


def test_codex_fork_with_embedded_parent_meta_keeps_its_own_identity(tmp_path: Path) -> None:
    parent = write_records(
        rollout(tmp_path, "parent"),
        [
            codex_meta("parent-id"),
            codex_turn("t1"),
            codex_call("call_0", "exec_command", {"cmd": "ls"}),
            codex_token_count(100),
            codex_token_count(150),
        ],
    )
    fork_meta = codex_meta("fork-id")
    fork_meta["payload"]["forked_from_id"] = "parent-id"
    fork = write_records(
        rollout(tmp_path, "fork"),
        [
            fork_meta,
            codex_meta("parent-id"),  # embedded parent metadata, not a new owner of what follows
            codex_turn("t9"),
            codex_call("call_0", "exec_command", {"cmd": "ls"}),
            codex_token_count(19),  # the fork's own counter starts over
            codex_token_count(41),
        ],
    )
    reader = codex_reader(tmp_path)
    ledger, admitted = admit_all(reader, [parent, fork])
    assert ledger.conflict_count == 0
    assert {e.session_id for e in admitted} == {"parent-id", "fork-id"}
    fork_rows = [e for e in admitted if e.session_id == "fork-id"]
    assert total_usage(fork_rows)["total_tokens"] == 41  # not mixed into the parent's 150
    assert total_usage([e for e in admitted if e.session_id == "parent-id"])["total_tokens"] == 150


def test_codex_user_turns_with_equal_length_text_do_not_collide(tmp_path: Path) -> None:
    def turn(text: str) -> dict:
        payload = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        return {"timestamp": "t", "type": "response_item", "payload": payload}

    path = write_records(
        rollout(tmp_path, "r"), [codex_meta("s"), turn("fix the bug"), turn("add the test")]
    )
    ledger, admitted = admit_all(codex_reader(tmp_path), [path])
    assert ledger.conflict_count == 0
    assert len([e for e in admitted if e.kind == "user_turn"]) == 2


# ----------------------------------------------------------------------- prime


def prime(tmp_path: Path) -> PrimeReader:
    return PrimeReader({"PRIME_AGENT_SESSION_DIR": str(tmp_path / "prime" / "sessions")})


def test_prime_repeated_call_ids_in_one_session_are_distinct_calls(tmp_path: Path) -> None:
    records = [pi_header("m1")]
    for index in range(3):  # `call_0` reused for three different invocations
        records.append(pi_tool_call(f"e{index}a", "call_0", "ipython", {"code": f"x={index}"}))
        records.append(pi_tool_result(f"e{index}b", "call_0", f"out {index}"))
    one = write_records(tmp_path / "prime" / "sessions" / "m1.jsonl", records)
    copy = write_records(tmp_path / "prime" / "sessions" / "copy.jsonl", records)
    reader = prime(tmp_path)
    ledger, admitted = admit_all(reader, [one, copy])
    calls = [e for e in admitted if e.kind == "tool_call"]
    results = [e for e in admitted if e.kind == "tool_result"]
    assert ledger.conflict_count == 0
    assert len(calls) == 3 and len(results) == 3  # not collapsed, copies still dedupe
    assert [c.payload["arguments"]["code"] for c in calls] == ["x=0", "x=1", "x=2"]
    assert [r.payload["output"] for r in results] == ["out 0", "out 1", "out 2"]
    assert len({c.record_id for c in calls}) == 3 and len({r.record_id for r in results}) == 3
    assert len({c.origin_id for c in calls}) == 3 and len({r.origin_id for r in results}) == 3
    assert {c.origin_domain for c in calls} | {r.origin_domain for r in results} == {"m1"}
    assert all(c.payload["call_id"] == "call_0" for c in calls)
    # each result is bound to exactly the call that preceded it, in order
    assert [r.payload["call_record_id"] for r in results] == [c.record_id for c in calls]
    assert {r.payload["call_link_status"] for r in results} == {"unique_pending"}


def test_prime_repeated_call_ids_survive_incremental_reads(tmp_path: Path) -> None:
    first_half = [pi_header("m1"), pi_tool_call("e1", "call_0", "ipython", {"code": "a"})]
    path = write_records(tmp_path / "prime" / "sessions" / "m1.jsonl", first_half)
    reader = prime(tmp_path)
    source = reader.source_for(path)
    first = reader.read(source)
    append_bytes(path, jsonl([pi_tool_call("e2", "call_0", "ipython", {"code": "b"})]))
    second = reader.read(source, first.state)
    ids = [e.record_id for e in first.evidence + second.evidence if e.kind == "tool_call"]
    assert len(set(ids)) == 2


def test_prime_lineage_status_transitions_are_distinct_events(tmp_path: Path) -> None:
    def row(status: str) -> dict:
        return {"type": "rlm_subagent", "childId": "sub-1", "status": status, "rlmDepth": 1}

    path = write_records(
        tmp_path / "prime" / "session-artifacts" / "m1" / "rlm-subagents.jsonl",
        [row("running"), row("completed"), row("deleted")],
    )
    reader = prime(tmp_path)
    ledger, admitted = admit_all(reader, [path, path])
    assert ledger.conflict_count == 0
    assert [e.payload["status"] for e in admitted] == ["running", "completed", "deleted"]
