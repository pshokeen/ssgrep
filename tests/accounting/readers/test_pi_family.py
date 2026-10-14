"""4.4 Pi-family readers: Pi, omp (parentSession), Prime (session-artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from tests.accounting.readers_fixtures import (
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
    evidence = {e.record_id: e for e in read(reader, path).evidence}
    assert evidence["session:s1"].kind == "session_header"
    call = evidence["call:call_1"]
    assert call.payload["name"] == "bash" and call.payload["arguments"] == {"command": "ls"}
    assert call.payload["cwd"] == "/work/demo" and call.session_id == "s1"
    assert call.runtime == "pi"
    error = evidence["result:call_1"]
    assert error.payload["is_error"] is True and error.payload["output"] == "denied"
    assert error.payload["call_id"] == "call_1"
    assert evidence["result:call_2"].payload["is_error"] is False
    usage = evidence["usage:e1"].payload
    assert usage["tokens"] == {
        "input_tokens": 4,
        "output_tokens": 267,
        "cache_read_tokens": 1,
        "cache_write_tokens": 17745,
        "total_tokens": 18017,
        "reasoning_tokens": 14,
    }
    assert usage["cost_nano_usd"]["total"] == 47_040_500  # exact integer, never a float
    assert evidence["turn:e0"].kind == "user_turn"


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


def test_retrieval_adapters_are_not_imported_or_modified(tmp_path: Path) -> None:
    import ssgrep.accounting.readers.pi_family as module

    assert "ssgrep.sessions" not in "".join(
        line for line in Path(module.__file__).read_text().splitlines() if "import" in line
    )
