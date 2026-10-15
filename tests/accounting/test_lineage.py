"""5.3: root / originating-turn / branch task identity over real omp-shaped trees."""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import pytest

from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import load_state
from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_io import load_evidence
from ssgrep.accounting.reduce_types import Reduction
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.readers_fixtures import build_home, write_records
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

EMPTY = results_json()
SESSIONS = Path(".omp") / "agent" / "sessions" / "-work"


def reduce_tree(root: Path, files: dict[str, list[dict]]) -> Reduction:
    evidence = omp_evidence(root, {f"-work/{k}": v for k, v in files.items()})
    return reduce_profile(PROFILE, evidence, [], policy=policy(), frequency=FrequencyView.unknown())


def path_of(root: Path, name: str) -> Path:
    return root / SESSIONS / name


def tasks_by_query(reduction: Reduction) -> dict[str, TaskId | None]:
    return {i.session_id: i.task for i in reduction.invocations}


def test_child_inherits_launching_turn_even_after_parent_gets_a_newer_prompt(
    tmp_path: Path,
) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    files = {
        "root.jsonl": [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "investigate the failing deploy", ts(12, 9)),
            *omp_launch("e-l1", "c-launch", "ChildAlpha", ts(12, 9, 10), ts(12, 9, 11)),
            omp_user("e-b", "now something else entirely", ts(12, 11)),
        ],
        "root/ChildAlpha.jsonl": [
            omp_header("child-a", ts(12, 9, 12), parent=parent),
            *omp_search("c1", "call-1", "deploy", EMPTY, (12, 11, 30, 0)),  # after turn B
        ],
    }
    reduction = reduce_tree(tmp_path, files)
    (invocation,) = reduction.invocations
    assert invocation.task == TaskId("omp:root-1", "turn:e-a", "main")
    assert invocation.task.resolved
    assert reduction.coverage["tasks_resolved"] == 1
    assert reduction.warnings == ()


def test_long_root_many_children_several_turns_stay_distinct_tasks_under_one_root(
    tmp_path: Path,
) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    root_records = [omp_header("root-1", ts(1, 8))]
    files: dict[str, list[dict]] = {}
    expected: dict[str, str] = {}
    for day, turn in ((1, "t1"), (9, "t2"), (17, "t3")):
        root_records.append(omp_user(f"e-{turn}", f"task {turn}", ts(day, 9)))
        for n in range(2):
            child = f"Child{turn.upper()}{n}"
            root_records += omp_launch(
                f"l-{turn}{n}", f"c-{turn}{n}", child, ts(day, 9, 10 + n), ts(day, 9, 11 + n)
            )
            files[f"root/{child}.jsonl"] = [
                omp_header(f"id-{turn}{n}", ts(day, 9, 20), parent=parent),
                *omp_search(f"s-{turn}{n}", f"k-{turn}{n}", "deploy", EMPTY, (day + 1, 10, n, 0)),
            ]
            expected[f"id-{turn}{n}"] = f"turn:e-{turn}"
    files["root.jsonl"] = root_records
    reduction = reduce_tree(tmp_path, files)
    got = {i.session_id: i.task for i in reduction.invocations if i.task is not None}
    assert {k: t.turn for k, t in got.items()} == expected
    assert {t.root for t in got.values()} == {"omp:root-1"}
    assert len({t.turn for t in got.values()}) == 3
    roots = {r for _, _, r in reduction.session_roots}
    assert roots == {"omp:root-1"}
    assert len(reduction.session_roots) == 7  # root + six children, every session listed
    assert reduction.coverage["tasks_unknown"] == 0


def test_missing_launch_link_keeps_the_root_with_task_unknown_not_a_guessed_task(
    tmp_path: Path,
) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    files = {
        "root.jsonl": [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "first thing", ts(12, 9)),
            omp_user("e-b", "second thing", ts(12, 10)),
        ],
        "root/Orphan.jsonl": [
            omp_header("orphan", ts(12, 9, 30), parent=parent),
            *omp_search("c1", "call-1", "deploy", EMPTY, (12, 10, 30, 0)),
        ],
    }
    reduction = reduce_tree(tmp_path, files)
    (invocation,) = reduction.invocations
    assert invocation.task == TaskId("omp:root-1", None, "main")
    assert invocation.task is not None and not invocation.task.resolved
    assert reduction.coverage["tasks_unknown"] == 1
    assert "lineage:unlinked_launch:omp:orphan" in reduction.warnings


def test_missing_parent_is_surfaced_and_creates_no_task(tmp_path: Path) -> None:
    gone = path_of(tmp_path, "vanished.jsonl")
    files = {
        "lonely.jsonl": [
            omp_header("lonely", ts(12, 9), parent=gone),
            omp_user("e-a", "do a thing", ts(12, 9, 1)),
            *omp_search("c1", "call-1", "deploy", EMPTY, (12, 10, 0, 0)),
        ]
    }
    reduction = reduce_tree(tmp_path, files)
    (invocation,) = reduction.invocations
    assert invocation.task == TaskId("omp:lonely", None, "main")
    assert "lineage:missing_parent:omp:lonely" in reduction.warnings


def test_cyclic_parents_are_surfaced_and_never_resolve_tasks_or_duplicate_credit(
    tmp_path: Path,
) -> None:
    a, b = path_of(tmp_path, "a.jsonl"), path_of(tmp_path, "b.jsonl")
    files = {
        "a.jsonl": [
            omp_header("sess-a", ts(12, 9), parent=b),
            omp_user("e-a", "from a", ts(12, 9, 1)),
            *omp_search("c1", "call-a", "deploy", EMPTY, (12, 10, 0, 0)),
        ],
        "b.jsonl": [
            omp_header("sess-b", ts(12, 9), parent=a),
            omp_user("e-b", "from b", ts(12, 9, 1)),
            *omp_search("c2", "call-b", "deploy", EMPTY, (12, 10, 1, 0)),
        ],
    }
    reduction = reduce_tree(tmp_path, files)
    tasks = [i.task for i in reduction.invocations if i.task is not None]
    assert len(tasks) == 2
    assert {t.root for t in tasks} == {"omp:sess-a"}  # one shared root
    assert all(not t.resolved for t in tasks)
    assert "lineage:cyclic_parent:omp:sess-a" in reduction.warnings
    assert "lineage:cyclic_parent:omp:sess-b" in reduction.warnings
    assert {r for _, _, r in reduction.session_roots} == {"omp:sess-a"}


def test_parentsession_resolves_without_any_rlm_depth(tmp_path: Path) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    files = {
        "root.jsonl": [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "go", ts(12, 9)),
            *omp_launch("l1", "c-l", "Worker", ts(12, 9, 5), ts(12, 9, 6)),
        ],
        "root/Worker.jsonl": [omp_header("w", ts(12, 9, 7), parent=parent)],
    }
    reduction = reduce_tree(tmp_path, files)
    roots = dict(((rt, sid), root) for rt, sid, root in reduction.session_roots)
    assert roots[("omp", "w")] == "omp:root-1"
    assert reduction.warnings == ()


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


# ---- a late parent through the real collector --------------------------------------------------
# A child transcript read before its parent is its own root; once the parent appears, the whole
# conversation is one root and the incremental ledger must not keep the child's earlier root.

NOW = scene.at(hour=20)
STAMP = scene.at(hour=10).timestamp()


def omp_sessions(root: Path) -> Path:
    return root / "omp-sessions" / "-work"


def write_child(root: Path) -> None:
    parent = omp_sessions(root) / "root.jsonl"
    child = omp_sessions(root) / "root" / "ChildAlpha.jsonl"
    write_records(
        child,
        [
            omp_header("child-a", ts(12, 9, 12), parent=parent),
            *omp_search("c1", "call-1", "deploy", results_json(), (12, 10, 0, 0)),
            *omp_search("c2", "call-2", "billing", results_json(), (12, 10, 1, 0)),
        ],
    )
    os.utime(child, (STAMP, STAMP))


def write_parent(root: Path) -> None:
    parent = omp_sessions(root) / "root.jsonl"
    write_records(
        parent,
        [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "investigate the failing deploy", ts(12, 9)),
            *omp_launch("e-l1", "c-launch", "ChildAlpha", ts(12, 9, 10), ts(12, 9, 11)),
        ],
    )
    os.utime(parent, (STAMP, STAMP))


def ledger() -> VersionedLedger:
    committed = vs.checkpoint()
    assert committed is not None
    state = load_state(PROFILE, committed, committed.methodology)
    assert state.ledger is not None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded)


def live_debit_roots() -> dict[str, set[str]]:
    by_id: dict[str, set[str]] = collections.defaultdict(set)
    for debit in ledger()._debits.values():
        by_id[debit.debit_id].add(debit.task.root)
    return dict(by_id)


def run_late_parent(root: Path, *, parent_later: bool) -> vs.Facts:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    write_child(root)
    if parent_later:
        first = scene.collect(NOW)
        assert first.state == "published"
        assert {r for roots in live_debit_roots().values() for r in roots} == {"omp:child-a"}
    write_parent(root)
    assert scene.collect(NOW).state == "published"
    # the scene really resolves the child under the parent root
    assert {r for roots in live_debit_roots().values() for r in roots} == {"omp:root-1"}
    return vs.facts()


def test_a_late_parent_leaves_no_debit_under_the_childs_earlier_root(tmp_path: Path) -> None:
    run_late_parent(tmp_path, parent_later=True)
    owners = live_debit_roots()
    assert owners, "the scene must produce lookup debits"
    assert all(roots == {"omp:root-1"} for roots in owners.values()), owners  # one root each
    assert len(ledger()._debits) == len(owners)  # and no id is held twice
