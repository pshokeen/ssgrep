"""Attribution boundaries exercised through native consumers and journal events."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.contracts import Attribution, Outcome, Phase
from tests.accounting.reduce_builders import (
    CWD,
    MCP,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    event,
    results_json,
    ts,
)
from tests.accounting.test_reduce_reconcile import run


def shell_records(command: str, *, end: str = ts(12, 10, 0, 3)) -> list[dict]:
    return [
        claude_prompt("s1", "investigate", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", "Bash", {"command": command}, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "t1", results_json(), ts=end),
    ]


def test_one_literal_call_cannot_claim_two_recorded_executions(tmp_path: Path) -> None:
    records = shell_records("ssgrep search 'deploy failure'")
    events = [
        *event("one", ts(12, 10, 0, 1), transport="cli"),
        *event("two", ts(12, 10, 0, 2), transport="cli"),
    ]
    reduction = run(tmp_path, records, events)
    assert len(reduction.executions) == 2
    assert {row.attribution for row in reduction.executions} == {Attribution.AMBIGUOUS}
    assert all(row.invocation_id is None for row in reduction.executions)
    (invocation,) = reduction.invocations
    assert invocation.execution_ids == ()
    assert invocation.ambiguous_execution_ids == ("one", "two")


def test_reused_correlation_does_not_pick_the_first_consumer(tmp_path: Path) -> None:
    command = "SSGREP_CORRELATION_ID=shared ssgrep search 'deploy failure'"
    records = shell_records(command)
    records += [
        claude_tool_use("s1", "t2", "Bash", {"command": command}, ts=ts(12, 10, 0, 1), cwd=CWD),
        claude_tool_result("s1", "t2", results_json(), ts=ts(12, 10, 0, 4)),
    ]
    reduction = run(
        tmp_path, records, event("one", ts(12, 10, 0, 2), transport="cli", correlation="shared")
    )
    (execution,) = reduction.executions
    assert execution.attribution is Attribution.AMBIGUOUS
    assert execution.invocation_id is None
    assert len(reduction.invocations) == 2
    assert all(row.ambiguous_execution_ids == ("one",) for row in reduction.invocations)


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation": "show"},
        {"transport": "mcp"},
        {"text": "different request"},
        {"cwd": "/work/other-project"},
    ],
)
def test_correlation_cannot_override_contradictory_request_facts(
    tmp_path: Path, overrides: dict[str, str]
) -> None:
    records = shell_records("SSGREP_CORRELATION_ID=shared ssgrep search 'deploy failure'")
    events = event(
        "one",
        ts(12, 10, 0, 1),
        operation=overrides.get("operation", "search"),
        transport=overrides.get("transport", "cli"),
        text=overrides.get("text", "deploy failure"),
        cwd=overrides.get("cwd", CWD),
        correlation="shared",
    )
    reduction = run(tmp_path, records, events)
    (execution,) = reduction.executions
    assert execution.attribution is Attribution.UNATTRIBUTED
    assert execution.invocation_id is None
    assert reduction.invocations[0].execution_ids == ()


def test_reversed_native_result_clock_blocks_inference(tmp_path: Path) -> None:
    reduction = run(
        tmp_path,
        shell_records("ssgrep search 'deploy failure'", end=ts(12, 9, 59, 59)),
        event("one", ts(12, 10, 0, 1), transport="cli"),
    )
    assert reduction.executions[0].attribution is Attribution.UNATTRIBUTED
    assert reduction.invocations[0].execution_ids == ()


def test_long_running_native_call_still_matches_without_nearest_time_guess(tmp_path: Path) -> None:
    records = shell_records("ssgrep search 'deploy failure'", end=ts(12, 16))
    reduction = run(tmp_path, records, event("one", ts(12, 15), transport="cli"))
    assert reduction.executions[0].attribution is Attribution.INFERRED_UNIQUE
    assert reduction.invocations[0].execution_ids == ("one",)


def test_started_only_execution_can_use_a_recognized_native_result(tmp_path: Path) -> None:
    reduction = run(
        tmp_path,
        shell_records("ssgrep search 'deploy failure'"),
        event("one", ts(12, 10, 0, 1), transport="cli", phases=(Phase.STARTED,)),
    )
    (execution,) = reduction.executions
    assert execution.attribution is Attribution.INFERRED_UNIQUE
    assert execution.rendered_bytes.value is None
    assert execution.visible_bytes.value == len(results_json().encode())
    assert execution.outcome is Outcome.EMPTY
    assert execution.outcome_basis == "result_override"


@pytest.mark.parametrize("omitted", [False, True])
def test_complete_result_ref_contradiction_blocks_inference_but_omitted_refs_do_not(
    tmp_path: Path, omitted: bool
) -> None:
    records = [
        claude_prompt("s1", "investigate", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", MCP, {"query": "deploy failure"}, ts=ts(12, 10)),
        claude_tool_result("s1", "t1", results_json(), ts=ts(12, 10, 0, 3)),
    ]
    events = event("one", ts(12, 10, 0, 1), refs=("claude:source:ep:0",))
    if omitted:
        events = [replace(row, refs_omitted=True) for row in events]
    reduction = run(tmp_path, records, events)
    expected = Attribution.INFERRED_UNIQUE if omitted else Attribution.UNATTRIBUTED
    assert reduction.executions[0].attribution is expected
