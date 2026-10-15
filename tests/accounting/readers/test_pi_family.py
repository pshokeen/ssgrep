"""4.4 Pi-family readers: Pi, omp (parentSession), Prime (session-artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.pi_family import HEADER_LINES, header_session_id
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
    jsonl,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    pi_user,
    write_records,
)

USAGE = {
    "input": 4,
    "output": 267,
    "cacheRead": 1,
    "cacheWrite": 17745,
    "totalTokens": 18017,
    "reasoningTokens": 14,
    "cost": {"input": 8e-06, "output": 0.00267, "total": 0.0470405},
}


def omp_reader(tmp_path: Path) -> OmpReader:
    return OmpReader({"OMP_SESSIONS_DIR": str(tmp_path / "omp")})


def prime_reader(tmp_path: Path) -> PrimeReader:
    return PrimeReader({"PRIME_AGENT_SESSION_DIR": str(tmp_path / "prime" / "sessions")})


def read(reader, path: Path):
    return reader.read(reader.source_for(path))


def test_tool_call_result_error_and_usage_are_preserved(tmp_path: Path) -> None:
    path = write_records(
        tmp_path / "pi" / "-w" / "s1.jsonl",
        [
            pi_header("s1"),
            pi_user("e0", "do it"),
            pi_tool_call("e1", "call_1", "bash", {"command": "ls"}, usage=USAGE),
            pi_tool_result("e2", "call_1", "denied", is_error=True),
            pi_tool_call("e3", "call_2", "read", {"path": "a"}),
            pi_tool_result("e4", "call_2", "contents"),
        ],
    )
    reader = PiReader({"PI_SESSION_DIR": str(tmp_path / "pi")})
    assert [s.path for s in reader.discover()] == [path]
    rows = read(reader, path).evidence
    assert [row.kind for row in rows] == [
        "session_header",
        "user_turn",
        "usage",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
    ]
    assert len({row.record_id for row in rows}) == len(rows)
    assert len({row.origin_id for row in rows if row.kind != "session_header"}) == len(rows) - 1

    def one(kind: str, call_id: str):
        (row,) = [r for r in rows if r.kind == kind and r.payload["call_id"] == call_id]
        return row

    header, turn, usage_row = rows[0], rows[1], rows[2]
    assert header.kind == "session_header" and header.record_id == "session:s1"
    call = one("tool_call", "call_1")
    assert call.payload["name"] == "bash" and call.payload["arguments"] == {"command": "ls"}
    assert call.payload["cwd"] == "/work/demo" and call.session_id == "s1"
    assert call.runtime == "pi" and call.payload["entry_id"] == "e1"
    error = one("tool_result", "call_1")
    assert error.payload["is_error"] is True and error.payload["output"] == "denied"
    assert error.payload["call_record_id"] == call.record_id
    assert error.payload["call_link_status"] == "unique_pending"
    second = one("tool_result", "call_2")
    assert second.payload["is_error"] is False and second.payload["output"] == "contents"
    assert second.payload["call_record_id"] == one("tool_call", "call_2").record_id
    assert usage_row.payload["entry_id"] == "e1" and usage_row.payload["model"] == "pi-test"
    usage = usage_row.payload
    assert usage["tokens"] == {
        "input_tokens": 4,
        "output_tokens": 267,
        "cache_read_tokens": 1,
        "cache_write_tokens": 17745,
        "total_tokens": 18017,
        "reasoning_tokens": 14,
    }
    assert usage["cost_nano_usd"]["total"] == 47_040_500  # exact integer, never a float
    assert turn.kind == "user_turn" and turn.payload["text"] == "do it"
    assert turn.payload["entry_id"] == "e0"


def test_pi_without_cost_leaves_cost_unavailable(tmp_path: Path) -> None:
    usage = {"input": 1, "output": 2, "totalTokens": 3}
    path = write_records(
        tmp_path / "pi" / "s.jsonl",
        [pi_header("s"), pi_tool_call("e1", "c", "bash", {"command": "ls"}, usage=usage)],
    )
    reader = PiReader({"PI_SESSION_DIR": str(tmp_path / "pi")})
    row = next(e for e in read(reader, path).evidence if e.kind == "usage")
    assert row.payload["cost_nano_usd"] is None


def test_omp_resolves_parent_session_path_without_rlm_depth(tmp_path: Path) -> None:
    root = tmp_path / "omp" / "-w"
    parent = write_records(root / "2026_parent.jsonl", [pi_header("parent-id")])
    child = write_records(
        root / "2026_child.jsonl",
        [pi_header("child-id", parentSession=str(parent)), pi_tool_call("e1", "c", "bash", {})],
    )
    reader = omp_reader(tmp_path)
    result = read(reader, child)
    [header] = [e for e in result.evidence if e.kind == "session_header"]
    assert "rlmDepth" not in header.payload and header.payload["rlm_depth"] is None
    assert header.payload["parent_session_path"] == str(parent)  # raw evidence kept as-is
    link = reader.parent_link(header)
    assert (link.state, link.parent_session, link.parent_path) == ("resolved", "parent-id", parent)
    assert link.child_session == "child-id"


def test_omp_parent_missing_or_outside_roots_is_explicit_and_never_read(tmp_path: Path) -> None:
    secret = tmp_path / "outside" / "secret.jsonl"
    secret.parent.mkdir(parents=True)
    secret.write_text(json.dumps(pi_header("leaked-id")) + "\n")
    root = tmp_path / "omp" / "-w"
    missing = write_records(
        root / "a.jsonl", [pi_header("a", parentSession=str(root / "gone.jsonl"))]
    )
    outside = write_records(root / "b.jsonl", [pi_header("b", parentSession=str(secret))])
    traversal = write_records(
        root / "c.jsonl",
        [pi_header("c", parentSession=str(root / ".." / ".." / "outside" / "secret.jsonl"))],
    )
    none = write_records(root / "d.jsonl", [pi_header("d")])
    reader = omp_reader(tmp_path)

    def link_for(path: Path):
        [header] = [e for e in read(reader, path).evidence if e.kind == "session_header"]
        return reader.parent_link(header)

    assert link_for(missing).state == "missing"
    assert link_for(outside).state == "outside_roots"
    assert link_for(outside).parent_session is None  # content of the outside file never used
    assert link_for(traversal).state == "outside_roots"
    assert link_for(none).state == "none"


def test_omp_title_record_before_header_and_model_usage(tmp_path: Path) -> None:
    path = write_records(
        tmp_path / "omp" / "s.jsonl",
        [
            {"type": "title", "title": ""},
            pi_header("o1"),
            {"type": "session_init", "id": "i1", "task": "x"},
            {
                "type": "model_usage",
                "id": "mu1",
                "timestamp": "2026-10-01T10:00:00.000Z",
                "purpose": "auto-thinking",
                "usage": {"input": 498, "output": 46, "totalTokens": 544},
            },
            {"type": "brand_new_thing"},
        ],
    )
    reader = omp_reader(tmp_path)
    result = read(reader, path)
    assert {e.session_id for e in result.evidence} == {"o1"}
    row = next(e for e in result.evidence if e.record_id == "usage:mu1")
    assert row.payload["purpose"] == "auto-thinking"
    assert row.payload["tokens"]["total_tokens"] == 544
    assert result.unrecognized == {"brand_new_thing": 1}


def test_prime_reads_session_artifacts_and_lineage(tmp_path: Path) -> None:
    sessions = tmp_path / "prime" / "sessions"
    artifacts = tmp_path / "prime" / "session-artifacts"
    main = write_records(
        sessions / "-w" / "m1.jsonl", [pi_header("m1"), pi_tool_call("e1", "c", "bash", {})]
    )
    child = write_records(
        artifacts / "m1" / "sub-1" / "m1-child.jsonl",
        [
            pi_header("m1-child", parentSession=str(main), rlmDepth=1),
            pi_tool_call("e1", "c", "bash", {}),
            pi_tool_result("e2", "c", "ok"),
        ],
    )
    lineage = write_records(
        artifacts / "m1" / "rlm-subagents.jsonl",
        [
            {
                "type": "rlm_subagent",
                "childId": "sub-1",
                "sessionFile": str(child),
                "parentSessionId": "m1",
                "parentSessionFile": str(main),
                "rlmDepth": 1,
                "prompt": "do the thing",
                "spawnCode": "import os; os.system('rm -rf /')",
                "status": "done",
            }
        ],
    )
    write_records(artifacts / "m1" / "semantic-edges.jsonl", [{"type": "session_registered"}])
    reader = prime_reader(tmp_path)
    discovered = {s.path: s.role for s in reader.discover()}
    assert discovered == {main: "main", child: "artifact", lineage: "lineage"}
    ledger = EvidenceLedger()
    admitted = [
        e for s in reader.discover() for e in read(reader, s.path).evidence if ledger.admit(e)
    ]
    kinds = {(e.session_id, e.kind) for e in admitted}
    assert ("m1-child", "tool_call") in kinds and ("m1-child", "tool_result") in kinds
    link_row = next(e for e in admitted if e.kind == "lineage")
    assert link_row.payload["child_id"] == "sub-1"
    assert link_row.payload["parent_session_id"] == "m1"
    assert link_row.payload["rlm_depth"] == 1
    assert "spawnCode" not in link_row.payload  # prompts/code are not retained as evidence
    header = next(e for e in admitted if e.session_id == "m1-child" and e.kind == "session_header")
    assert reader.parent_link(header).parent_session == "m1"


def test_prime_parent_by_id_resolves_through_the_session_index(tmp_path: Path) -> None:
    sessions = tmp_path / "prime" / "sessions"
    write_records(sessions / "m.jsonl", [pi_header("m-id")])
    write_records(
        tmp_path / "prime" / "session-artifacts" / "m" / "c.jsonl",
        [pi_header("c-id", parentSessionId="m-id")],
    )
    reader = prime_reader(tmp_path)
    child = next(s.path for s in reader.discover() if s.path.name == "c.jsonl")
    header = next(e for e in read(reader, child).evidence if e.kind == "session_header")
    link = reader.parent_link(header)
    assert (link.state, link.parent_session) == ("resolved", "m-id")


def test_runtime_label_and_identity_differ_per_runtime(tmp_path: Path) -> None:
    records = [pi_header("same"), pi_tool_call("e1", "call_0", "bash", {"command": "ls"})]
    pi_path = write_records(tmp_path / "pi" / "same.jsonl", records)
    omp_path = write_records(tmp_path / "omp" / "same.jsonl", records)
    pi = PiReader({"PI_SESSION_DIR": str(tmp_path / "pi")})
    omp = omp_reader(tmp_path)
    ledger = EvidenceLedger()
    admitted = [
        e
        for reader, path in ((pi, pi_path), (omp, omp_path))
        for e in read(reader, path).evidence
        if ledger.admit(e)
    ]
    assert {e.runtime for e in admitted if e.kind == "tool_call"} == {"pi", "omp"}
    assert sum(1 for e in admitted if e.kind == "tool_call") == 2


def test_header_probe_skips_malformed_and_oversized_lines_before_valid_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "parent.jsonl"
    path.write_bytes(
        b"not json\n" + b"x" * (MAX_RAW_RECORD_BYTES + 1) + b"\n" + jsonl([pi_header("parent")])
    )
    assert header_session_id(path) == "parent"
    assert path.stat().st_size > MAX_RAW_RECORD_BYTES


def test_header_probe_accepts_last_bounded_line_but_not_later_header(tmp_path: Path) -> None:
    ignored = {"type": "title", "title": "not a session identity"}
    inside = write_records(
        tmp_path / "inside.jsonl", [ignored] * (HEADER_LINES - 1) + [pi_header("inside")]
    )
    outside = write_records(
        tmp_path / "outside.jsonl", [ignored] * HEADER_LINES + [pi_header("outside")]
    )
    assert header_session_id(inside) == "inside"
    assert header_session_id(outside) is None
    empty = write_records(tmp_path / "empty.jsonl", [])
    assert header_session_id(empty) is None
    assert header_session_id(tmp_path / "missing.jsonl") is None
    assert header_session_id(tmp_path) is None


def test_parent_resolution_distinguishes_missing_unknown_and_explicit_native_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    parent = write_records(root / "parent.jsonl", [pi_header("native-parent")])
    unreadable = write_records(root / "no-header.jsonl", [{"type": "title"}])
    reader = OmpReader({"OMP_SESSIONS_DIR": str(root)})

    def link_for(name: str, **parent_fields):
        child = write_records(root / f"{name}.jsonl", [pi_header(name, **parent_fields)])
        header = reader.read(reader.source_for(child)).evidence[0]
        return reader.parent_link(header)

    unknown = link_for("unknown", parentSession=str(unreadable))
    assert (unknown.state, unknown.parent_session, unknown.parent_path) == (
        "unreadable",
        None,
        unreadable,
    )
    resolved = link_for("resolved", parentSessionId="native-parent")
    assert (resolved.state, resolved.parent_session, resolved.parent_path) == (
        "resolved",
        "native-parent",
        parent,
    )
    external = write_records(tmp_path / "external.jsonl", [pi_header("external-parent")])
    alias = root / "alias.jsonl"
    alias.symlink_to(external)
    outside = link_for("outside", parentSession=str(alias))
    assert outside.state == "outside_roots"
    assert outside.parent_session is None
    assert alias not in [source.path for source in reader.discover()]
    assert parent in [source.path for source in reader.discover()]


@pytest.mark.parametrize(
    "reader_type,agent_var,session_var",
    [
        (PiReader, "PI_CODING_AGENT_DIR", "PI_SESSION_DIR"),
        (OmpReader, "OMP_AGENT_DIR", "OMP_SESSIONS_DIR"),
        (PrimeReader, "PRIME_AGENT_CODING_AGENT_DIR", "PRIME_AGENT_SESSION_DIR"),
    ],
)
def test_agent_root_expansion_and_explicit_session_precedence_use_only_selected_store(
    tmp_path: Path, reader_type, agent_var: str, session_var: str
) -> None:
    agent = write_records(
        tmp_path / "agent" / "sessions" / "agent.jsonl",
        [pi_header("agent"), pi_tool_call("e", "agent-call", "bash", {})],
    )
    explicit = write_records(
        tmp_path / "explicit" / "selected.jsonl",
        [pi_header("selected"), pi_tool_call("e", "selected-call", "bash", {})],
    )
    env = {"HOME": str(tmp_path), agent_var: " ~/agent "}
    reader = reader_type(env)
    assert [source.path for source in reader.discover()] == [agent]
    agent_call = reader.read(reader.source_for(agent)).evidence[-1]
    assert (agent_call.kind, agent_call.payload["call_id"], agent_call.session_id) == (
        "tool_call",
        "agent-call",
        "agent",
    )
    selected = reader_type({**env, session_var: " ~/explicit "})
    assert [source.path for source in selected.discover()] == [explicit]
    assert selected.read(selected.source_for(explicit)).evidence[-1].session_id == "selected"


def test_bare_tilde_root_does_not_resolve_against_real_home(tmp_path: Path) -> None:
    path = write_records(tmp_path / "session.jsonl", [pi_header("isolated")])
    reader = OmpReader({"HOME": str(tmp_path), "OMP_SESSIONS_DIR": "~"})
    assert [source.path for source in reader.discover()] == [path]
    assert reader.read(reader.source_for(path)).evidence[0].session_id == "isolated"
