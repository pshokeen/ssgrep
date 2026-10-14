"""5.3: root / originating-turn / branch task identity over real omp-shaped trees."""

from __future__ import annotations

from pathlib import Path

from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_types import Reduction
from tests.accounting.reduce_builders import (
    PROFILE,
    omp_evidence,
    omp_header,
    omp_launch,
    omp_search,
    omp_user,
    policy,
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
