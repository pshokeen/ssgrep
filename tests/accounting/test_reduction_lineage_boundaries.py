"""Launch-result identity and chronological contradictions govern child task ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.contracts import Reuse, TaskId
from ssgrep.accounting.reduce import methodology_for, reduce_profile
from ssgrep.accounting.reduce_cands import CHARS
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import Eligibility
from tests.accounting.reduce_builders import (
    KEY,
    PROFILE,
    card,
    claude_evidence,
    omp_call,
    omp_evidence,
    omp_header,
    omp_result,
    omp_search,
    omp_user,
    policy,
    results_json,
    source_session,
    ts,
)
from tests.accounting.test_lineage import path_of, reduce_tree
from tests.accounting.test_reduce_evidence import FACT, PATH, VIEW


@pytest.mark.parametrize("child_hour, resolved", [(10, True), (8, False)])
def test_child_named_only_by_launch_result_inherits_call_turn_unless_it_predates_launch(
    tmp_path: Path, child_hour: int, resolved: bool
) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    reduction = reduce_tree(
        tmp_path,
        {
            "root.jsonl": [
                omp_header("root-1", ts(12, 7)),
                omp_user("original", "investigate deploy", ts(12, 9)),
                omp_call("launch", "launch-1", "task", {"task": "investigate"}, ts(12, 9, 30)),
                omp_user("new", "a different task", ts(12, 9, 31)),
                omp_result("launched", "launch-1", "Spawned `WorkerAlpha`", ts(12, 10)),
            ],
            "root/2026_WorkerAlpha.jsonl": [
                omp_header("child-id", ts(12, child_hour), parent=parent),
                *omp_search("search", "search-1", "deploy", results_json(), (12, 11, 0, 0)),
            ],
        },
    )
    (invocation,) = reduction.invocations
    expected_turn = "turn:original" if resolved else None
    assert invocation.task == TaskId("omp:root-1", expected_turn, "main")
    if resolved:
        assert reduction.warnings == ()
    else:
        assert "lineage:unlinked_launch:omp:child-id" in reduction.warnings


def test_similar_child_alias_does_not_take_another_workers_launch(tmp_path: Path) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    reduction = reduce_tree(
        tmp_path,
        {
            "root.jsonl": [
                omp_header("root-1", ts(12, 8)),
                omp_user("original", "investigate deploy", ts(12, 9)),
                omp_call(
                    "launch", "launch-1", "task", {"name": "WorkerAlphaBackup"}, ts(12, 9, 30)
                ),
                omp_result("launched", "launch-1", "WorkerAlphaBackup started", ts(12, 9, 31)),
            ],
            "root/WorkerAlpha.jsonl": [
                omp_header("child-id", ts(12, 10), parent=parent),
                *omp_search("search", "search-1", "deploy", results_json(), (12, 11, 0, 0)),
            ],
        },
    )
    (invocation,) = reduction.invocations
    assert invocation.task == TaskId("omp:root-1", None, "main")
    assert "lineage:unlinked_launch:omp:child-id" in reduction.warnings


@pytest.mark.parametrize(
    "parent_present, limitation",
    [(False, "missing_or_cyclic_ancestor_context"), (True, "launch_context_unknown")],
)
def test_child_without_launch_context_keeps_fact_but_marks_novelty_uncertain(
    tmp_path: Path, parent_present: bool, limitation: str
) -> None:
    files = {
        "-work/ChildAlpha.jsonl": [
            omp_header("child-id", ts(12, 9), parent=path_of(tmp_path, "root.jsonl")),
            *omp_search(
                "search",
                "search-1",
                "investigate deploy",
                results_json(card("claude:src1:ep:0", FACT)),
                (12, 10, 0, 0),
            ),
            omp_call("action", "action-1", "bash", {"command": f"cat {PATH}"}, ts(12, 10, 1)),
            omp_result("acted", "action-1", "ok", ts(12, 10, 1, 2)),
        ],
    }
    if parent_present:
        files["-work/root.jsonl"] = [
            omp_header("root-1", ts(12, 8)),
            omp_user("original", "investigate deploy", ts(12, 8, 30)),
        ]
    evidence = omp_evidence(tmp_path, files)
    evidence += claude_evidence(tmp_path, {"-work/src1.jsonl": source_session("src1", FACT)})
    reduction = reduce_profile(
        PROFILE, evidence, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    (record,) = reduction.candidates
    assert record.candidate.task.turn is None
    assert record.novelty_confidence == "reduced"
    assert limitation in record.novelty_reasons
    assert record.candidate.candidate_type == "path"
    assert record.candidate.reuse is Reuse.ACTION_REFERENCE
    assert record.candidate.eligibility is Eligibility.UNAVAILABLE
    assert record.candidate.exposure_ids == (record.invocation_id,)
    (action,) = [
        item
        for item in evidence
        if item.kind == "tool_call" and item.payload.get("call_id") == "action-1"
    ]
    assert record.candidate.action_locators == (f"{action.source_id}@{action.locator}",)
    valued = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    estimates = rollup((leaf for leaf in valued.leaves if leaf.unit == CHARS), CHARS)
    assert estimates and all(not row.complete and row.net.value is None for row in estimates)
    assert all(row.debit.value is not None and row.debit.value > 0 for row in estimates)


def test_launch_result_without_proven_call_does_not_take_the_result_turn(tmp_path: Path) -> None:
    parent = path_of(tmp_path, "root.jsonl")
    reduction = reduce_tree(
        tmp_path,
        {
            "root.jsonl": [
                omp_header("root-1", ts(12, 7)),
                omp_user("original", "investigate deploy", ts(12, 9)),
                omp_user("new", "a different task", ts(12, 9, 31)),
                omp_result("orphan-result", "missing-call", "Spawned `WorkerAlpha`", ts(12, 10)),
            ],
            "root/2026_WorkerAlpha.jsonl": [
                omp_header("child-id", ts(12, 10), parent=parent),
                *omp_search("search", "search-1", "deploy", results_json(), (12, 11, 0, 0)),
            ],
        },
    )
    (invocation,) = reduction.invocations
    assert invocation.task == TaskId("omp:root-1", None, "main")
    assert "lineage:unlinked_launch:omp:child-id" in reduction.warnings
