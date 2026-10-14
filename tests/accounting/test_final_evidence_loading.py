"""Bounded real-store loading and finite child-relay evidence remain conservative."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_io import load_evidence
from tests.accounting.readers_fixtures import build_home, claude_prompt, write_records
from tests.accounting.reduce_builders import (
    KEY,
    PROFILE,
    card,
    omp_call,
    omp_evidence,
    omp_header,
    omp_launch,
    omp_result,
    omp_search,
    omp_user,
    policy,
    rare_view,
    results_json,
    ts,
)


def test_loader_restricts_canonical_subtree_and_reports_bad_record_coverage(tmp_path: Path) -> None:
    home = build_home(tmp_path)
    target = home.paths["claude"] / "final" / "selected.jsonl"
    write_records(target, [claude_prompt("selected", "visible_fact", ts=ts())])
    with target.open("ab") as stream:
        stream.write(b"not-json\n")
    alias = tmp_path / "selected-alias"
    alias.symlink_to(target)
    loaded = load_evidence(home.env, only_under=[alias])
    assert {item.session_id for item in loaded.evidence} == {"selected"}
    assert loaded.sources_read["claude"] == 1
    assert sum(loaded.sources_read.values()) == 1
    assert sum(loaded.gaps.values()) == 1
    rejected = load_evidence(home.env, only_under=[target], include=lambda source: False)
    assert rejected.evidence == [] and sum(rejected.sources_read.values()) == 0


def test_loader_counts_source_read_error_without_discarding_other_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = build_home(tmp_path)
    folder = home.paths["claude"] / "read-fault"
    missing = folder / "missing.jsonl"
    write_records(missing, [claude_prompt("missing", "first fact", ts=ts())])
    write_records(folder / "retained.jsonl", [claude_prompt("retained", "second fact", ts=ts())])
    before = load_evidence(home.env, only_under=[folder])
    assert {item.session_id for item in before.evidence} == {"missing", "retained"}
    assert before.unreadable_sources == 0
    original = ClaudeReader.read

    def lose_first_source(self, source, *args, **kwargs):
        if source.path == missing:
            raise OSError("source disappeared after discovery")
        return original(self, source, *args, **kwargs)

    monkeypatch.setattr(ClaudeReader, "read", lose_first_source)
    after = load_evidence(home.env, only_under=[folder])
    assert after.unreadable_sources == 1
    assert {item.session_id for item in after.evidence} == {"retained"}
    assert after.sources_read == before.sources_read


def test_partial_native_lineage_status_cannot_invent_a_parent_or_consumer_root(
    tmp_path: Path,
) -> None:
    home = build_home(tmp_path)
    loaded = load_evidence(home.env)
    before = reduce_profile(PROFILE, loaded.evidence, [], policy=policy(), frequency=rare_view())
    assert ("prime-agent", "m1-child", "m1") in before.session_parents
    path = home.paths["artifacts"] / "m1" / "rlm-subagents.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"type":"rlm_subagent","childId":"pending","status":"running"}\n')
    loaded = load_evidence(home.env)
    assert any(
        row.kind == "lineage" and row.payload.get("child_id") == "pending"
        for row in loaded.evidence
    )
    after = reduce_profile(PROFILE, loaded.evidence, [], policy=policy(), frequency=rare_view())
    assert after.session_parents == before.session_parents
    assert after.session_roots == before.session_roots


@pytest.mark.parametrize(
    "depth, act, credited", [(4, True, True), (5, True, False), (1, False, False)]
)
def test_child_relay_requires_action_within_supported_ancestor_depth(
    tmp_path: Path, depth: int, act: bool, credited: bool
) -> None:
    fact = "services/quota/repair.py"
    base = tmp_path / ".omp" / "agent" / "sessions" / "project"
    files = {}
    for index in range(depth + 1):
        name = f"Node{index}"
        parent = base / f"Node{index - 1}.jsonl" if index else None
        records = [omp_header(name, ts(12, 9, index), parent=parent)]
        if index < depth:
            records += [
                omp_user(f"u{index}", "investigate deployment", ts(12, 9, index, 1)),
                *omp_launch(
                    f"launch{index}",
                    f"launch{index}",
                    f"Node{index + 1}",
                    ts(12, 9, index, 2),
                    ts(12, 9, index, 3),
                ),
            ]
        else:
            records += omp_search(
                "lookup",
                "lookup",
                "deployment",
                results_json(card("claude:source:ep:0", fact)),
                (12, 10, 0, 0),
            )
        if index == 0:
            records += [omp_result("report", "launch0", f"Node1 reports {fact}", ts(12, 11))]
            if act:
                records += [
                    omp_call("act", "act", "bash", {"command": f"cat {fact}"}, ts(12, 11, 1)),
                    omp_result("act-result", "act", "ok", ts(12, 11, 2)),
                ]
        files[f"project/{name}.jsonl"] = records
    evidence = omp_evidence(tmp_path, files)
    reduction = reduce_profile(
        PROFILE, evidence, [], policy=policy(), frequency=rare_view(fact), fingerprint_key=KEY
    )
    assert bool(reduction.candidates) is credited
    assert reduction.bundles[0].debits
    if credited:
        assert reduction.candidates[0].candidate.explicit_relay
    else:
        assert reduction.coverage["candidates_unreferenced"] == 1


def test_native_lineage_to_missing_child_is_reported_without_inventing_an_edge(
    tmp_path: Path,
) -> None:
    home = build_home(tmp_path)
    original = load_evidence(home.env)
    before = reduce_profile(PROFILE, original.evidence, [], policy=policy(), frequency=rare_view())
    assert ("prime-agent", "m1-child", "m1") in before.session_parents
    path = home.paths["artifacts"] / "m1" / "rlm-subagents.jsonl"
    missing = {
        "type": "rlm_subagent",
        "childId": "absent-child",
        "sessionFile": str(home.paths["artifacts"] / "never-created.jsonl"),
        "parentSessionId": "m1",
        "status": "running",
    }
    with path.open("ab") as stream:
        stream.write(json.dumps(missing).encode() + b"\n")
    loaded = load_evidence(home.env)
    assert any(
        row.kind == "lineage" and row.payload.get("session_file") == missing["sessionFile"]
        for row in loaded.evidence
    )
    after = reduce_profile(PROFILE, loaded.evidence, [], policy=policy(), frequency=rare_view())
    assert after.session_parents == before.session_parents
    assert after.session_roots == before.session_roots
    warning = "lineage:unresolved_lineage_row:prime-agent:m1"
    assert warning not in before.warnings and warning in after.warnings
