"""5.1/5.2: correlation, ambiguity, shell cardinality, classes, denominators."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.collect.frequency import FrequencyView
from ssgrep.accounting.ledger.contracts import (
    Attribution,
    InvocationClass,
    Operation,
    Outcome,
    Phase,
)
from ssgrep.accounting.ledger.reduce import reduce_profile
from ssgrep.accounting.ledger.reduce_join import parse_argv
from ssgrep.accounting.ledger.reduce_records import DENOMINATORS, Reduction
from tests.accounting.ledger.reduce_builders import (
    CWD,
    KEY,
    MCP,
    PROFILE,
    SHOW,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    event,
    policy,
    results_json,
    ts,
)
from tests.accounting.ledger.test_reduce_evidence import VIEW, consumer

EMPTY = results_json()


def run(root: Path, records: list[dict], events: list, *, name: str = "s1") -> Reduction:
    evidence = claude_evidence(root, {f"-work/{name}.jsonl": records})
    return reduce_profile(
        PROFILE,
        evidence,
        events,
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )


def search_call(call_id: str, query: str, at: tuple[int, int, int], **kw) -> list[dict]:
    hour, minute, second = at
    return [
        claude_tool_use(
            "s1", call_id, MCP, {"query": query}, ts=ts(12, hour, minute, second), **kw
        ),
        claude_tool_result("s1", call_id, EMPTY, ts=ts(12, hour, minute, second + 3)),
    ]


def test_explicit_correlation_beats_everything_else(tmp_path: Path) -> None:
    command = 'SSGREP_CORRELATION_ID=run-7 ssgrep search "deploy failure"'
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", "Bash", {"command": command}, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "t1", EMPTY, ts=ts(12, 10, 0, 3)),
    ]
    events = event("n-1", ts(12, 10, 0, 1), transport="cli", correlation="run-7")
    reduction = run(tmp_path, records, events)
    (execution,) = reduction.executions
    assert execution.attribution is Attribution.EXPLICIT
    assert execution.invocation_id is not None
    assert execution.source == "both"
    assert reduction.coverage["explicit"] == 1


def test_two_identical_concurrent_searches_are_ambiguous_never_nearest(tmp_path: Path) -> None:
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        *search_call("t1", "deploy failure", (10, 0, 0)),
        *search_call("t2", "deploy failure", (10, 0, 1)),
    ]
    events = [
        *event("n-1", ts(12, 10, 0, 2), text="deploy failure"),
        *event("n-2", ts(12, 10, 0, 3), text="deploy failure"),
    ]
    reduction = run(tmp_path, records, events)
    assert {e.attribution for e in reduction.executions} == {Attribution.AMBIGUOUS}
    assert all(e.invocation_id is None for e in reduction.executions)
    assert all(i.attribution is Attribution.AMBIGUOUS for i in reduction.invocations)
    assert all(len(i.ambiguous_execution_ids) == 2 for i in reduction.invocations)
    assert reduction.coverage["ambiguous_joins"] == 2


def test_distinct_queries_in_the_same_window_join_uniquely(tmp_path: Path) -> None:
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        *search_call("t1", "deploy failure", (10, 0, 0)),
        *search_call("t2", "billing quota", (10, 0, 1)),
    ]
    events = [
        *event("n-1", ts(12, 10, 0, 2), text="deploy failure"),
        *event("n-2", ts(12, 10, 0, 3), text="billing quota"),
    ]
    reduction = run(tmp_path, records, events)
    by_id = {e.execution_id: e for e in reduction.executions}
    assert {e.attribution for e in reduction.executions} == {Attribution.INFERRED_UNIQUE}
    assert by_id["n-1"].invocation_id != by_id["n-2"].invocation_id


def test_timestamp_alone_never_joins(tmp_path: Path) -> None:
    """Right window, right operation, but a different query: no compatible evidence."""
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        *search_call("t1", "deploy failure", (10, 0, 0)),
    ]
    events = event("n-1", ts(12, 10, 0, 2), text="something else entirely")
    reduction = run(tmp_path, records, events)
    (execution,) = reduction.executions
    assert execution.attribution is Attribution.UNATTRIBUTED
    assert execution.invocation_id is None


@pytest.mark.parametrize("correlation", [None, "loop-run"])
def test_loop_with_three_journal_executions_is_one_invocation_three_executions(
    tmp_path: Path,
    correlation: str | None,
) -> None:
    prefix = f"SSGREP_CORRELATION_ID={correlation} " if correlation else ""
    loop = f'for q in alpha beta gamma; do {prefix}ssgrep search "$q"; done'
    aggregate = "\n".join([EMPTY] * 3)
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", "Bash", {"command": loop}, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "t1", aggregate, ts=ts(12, 10, 0, 9)),
    ]
    events = [
        *event("n-1", ts(12, 10, 0, 1), transport="cli", text="alpha", correlation=correlation),
        *event("n-2", ts(12, 10, 0, 3), transport="cli", text="beta", correlation=correlation),
        *event("n-3", ts(12, 10, 0, 5), transport="cli", text="gamma", correlation=correlation),
    ]
    reduction = run(tmp_path, records, events)
    (invocation,) = reduction.invocations
    assert invocation.cardinality == "unknown"
    if correlation:
        assert invocation.execution_ids == ("n-1", "n-2", "n-3")
        assert invocation.cls is InvocationClass.MATCHED
        assert {e.invocation_id for e in reduction.executions} == {invocation.invocation_id}
        assert {e.attribution for e in reduction.executions} == {Attribution.EXPLICIT}
    else:
        # Dynamic $q has no request fingerprint, and empty output has no refs:
        # time/cwd and a generic success outcome cannot identify these calls.
        assert invocation.execution_ids == ()
        assert invocation.cls is InvocationClass.TRANSCRIPT_ONLY
        assert all(e.invocation_id is None for e in reduction.executions)
        assert {e.attribution for e in reduction.executions} == {Attribution.UNATTRIBUTED}
        assert all(
            "insufficient_evidence" in e.join_rejection_reasons for e in reduction.executions
        )
    assert {e.execution_id for e in reduction.executions} == {"n-1", "n-2", "n-3"}
    assert reduction.coverage["recorded_executions"] == 3
    # The observed native aggregate is charged once in both supported and
    # unsupported join cases; unknown loop cardinality never invents charges.
    context_debits = [
        debit for bundle in reduction.bundles for debit in bundle.debits if debit.chars is not None
    ]
    assert len(context_debits) == 1
    assert context_debits[0].exposure_id == invocation.invocation_id
    assert context_debits[0].chars == len(aggregate)
    assert context_debits[0].amount.value is not None
    # the shared shell output cannot be charged three times: it is unavailable per execution
    assert all(e.visible_bytes.value is None for e in reduction.executions)


def test_loop_text_alone_never_manufactures_executions(tmp_path: Path) -> None:
    loop = 'for q in alpha beta gamma; do ssgrep search "$q"; done'
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", "Bash", {"command": loop}, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "t1", EMPTY, ts=ts(12, 10, 0, 9)),
    ]
    reduction = run(tmp_path, records, [])
    assert reduction.executions == ()
    assert len(reduction.invocations) == 1
    assert reduction.invocations[0].cls is InvocationClass.TRANSCRIPT_ONLY
    assert reduction.coverage["recorded_executions"] == 0


def test_unjoined_journal_execution_stays_recorded_with_unavailable_exposure(
    tmp_path: Path,
) -> None:
    records = [claude_prompt("s1", "just chatting", ts=ts(12, 9))]
    events = event("n-9", ts(12, 11), text="orphan", outcome=Outcome.EMPTY)
    reduction = run(tmp_path, records, events)
    (execution,) = reduction.executions
    assert execution.outcome is Outcome.EMPTY
    assert execution.attribution is Attribution.UNATTRIBUTED
    assert execution.source == "journal"
    assert execution.exposure.value == "unavailable"
    assert execution.visible_bytes.value is None
    assert execution.rendered_bytes.value == 900
    assert reduction.coverage["recorded_executions"] == 1


def test_attribution_partitions_recorded_executions_exactly_and_names_denominators(
    tmp_path: Path,
) -> None:
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        *search_call("t1", "alpha one", (10, 0, 0)),
        *search_call("t2", "dup query", (10, 5, 0)),
        *search_call("t3", "dup query", (10, 5, 1)),
    ]
    events = [
        *event("n-1", ts(12, 10, 0, 1), text="alpha one"),
        *event("n-2", ts(12, 10, 5, 2), text="dup query"),
        *event("n-3", ts(12, 10, 5, 3), text="dup query"),
        *event("n-4", ts(12, 12), text="no transcript"),
    ]
    cov = run(tmp_path, records, events).coverage
    parts = {k: cov[k] for k in ("explicit", "inferred_unique", "ambiguous", "unattributed")}
    assert parts == {"explicit": 0, "inferred_unique": 1, "ambiguous": 2, "unattributed": 1}
    assert sum(parts.values()) == cov["recorded_executions"] == 4
    assert all(DENOMINATORS[name] == "recorded_executions" for name in parts)
    assert cov["attributed_lookups"] == 1
    assert cov["recorded_lookups"] == 4


def test_wall_clock_discontinuity_makes_timing_unknown_and_blocks_inferred_join(
    tmp_path: Path,
) -> None:
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        *search_call("t1", "deploy failure", (10, 0, 0)),
    ]
    started, completed, rendered = event("n-1", ts(12, 10, 0, 1))
    # wall clock jumped one hour between started and completed; monotonic says 1 ms
    completed = replace(completed, ts_utc=ts(12, 11, 0, 1))
    reduction = run(tmp_path, records, [started, completed, rendered])
    (execution,) = reduction.executions
    assert execution.timing_known is False
    assert execution.attribution is Attribution.UNATTRIBUTED
    # explicit correlation still joins despite unknown timing
    started, completed, rendered = event(
        "n-2", ts(12, 10, 0, 1), transport="cli", correlation="c-1"
    )
    completed = replace(completed, ts_utc=ts(12, 11, 0, 1))
    command = "SSGREP_CORRELATION_ID=c-1 ssgrep search 'deploy failure'"
    explicit = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", "Bash", {"command": command}, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "t1", EMPTY, ts=ts(12, 10, 0, 3)),
    ]
    joined = run(tmp_path / "b", explicit, [started, completed, rendered])
    assert joined.executions[0].attribution is Attribution.EXPLICIT
    assert joined.executions[0].timing_known is False


def test_invocation_classes_journal_gap_transcript_only_mention_unclassified(
    tmp_path: Path,
) -> None:
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        # before the first journal event: transcript_only (outside the capture span)
        *search_call("t0", "early one", (9, 30, 0)),
        # inside the capture span with no event: journal_gap
        *search_call("t1", "uncaptured", (10, 30, 0)),
        # joined
        *search_call("t2", "captured", (10, 59, 0)),
        # documentation read and an unknown tool
        claude_tool_use("s1", "t3", "Read", {"file_path": "/x/README.md"}, ts=ts(12, 10, 40)),
        claude_tool_use("s1", "t4", "mystery_tool", {"x": 1}, ts=ts(12, 10, 41)),
        claude_tool_use(
            "s1", "t5", "Bash", {"command": 'echo "ssgrep search not-run"'}, ts=ts(12, 10, 42)
        ),
    ]
    events = [
        *event("n-1", ts(12, 10, 0, 0), text="seed one"),
        *event("n-2", ts(12, 10, 59, 1), text="captured"),
        *event("n-3", ts(12, 11, 30, 0), text="late"),
    ]
    reduction = run(tmp_path, records, events)
    classes = {i.invocation_id: i for i in reduction.invocations}
    by_query = {i.reason or "matched": i.cls for i in reduction.invocations}
    assert sorted(i.cls.value for i in classes.values()) == [
        "journal_gap",
        "matched",
        "transcript_only",
    ]
    assert by_query["no_journal_event_in_capture_span"] is InvocationClass.JOURNAL_GAP
    assert by_query["outside_capture_span"] is InvocationClass.TRANSCRIPT_ONLY
    cov = reduction.coverage
    assert cov["inv_mention"] == 1
    assert cov["inv_unclassified"] == 2  # unknown tool + the plain Read
    assert cov["recognized_invocations"] == 3
    assert cov["classified_tool_calls"] == 6
    assert DENOMINATORS["inv_journal_gap"] == "recognized_invocations"


def test_pre_api_failure_is_a_transcript_only_invocation_not_an_execution(
    tmp_path: Path,
) -> None:
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use(
            "s1", "t1", "Bash", {"command": "ssgrep search --bogus x"}, ts=ts(12, 10), cwd=CWD
        ),
        claude_tool_result(
            "s1", "t1", "Error: No such option: --bogus", is_error=True, ts=ts(12, 10, 0, 1)
        ),
    ]
    events = event("n-1", ts(12, 10, 0, 30), text="far away", transport="cli")
    reduction = run(tmp_path, records, events)
    (invocation,) = reduction.invocations
    assert invocation.pre_api_failure is True
    assert invocation.cls is InvocationClass.TRANSCRIPT_ONLY
    assert invocation.outcome is Outcome.ERROR
    assert invocation.execution_ids == ()


def test_show_requests_join_on_ref_fingerprint(tmp_path: Path) -> None:
    ref = "claude:src:ep:0"
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", SHOW, {"ref": ref}, ts=ts(12, 10)),
        claude_tool_result(
            "s1",
            "t1",
            json.dumps({"episode_id": ref, "prompt_text": "p", "response_text": "r"}),
            ts=ts(12, 10, 0, 2),
        ),
    ]
    events = event("n-1", ts(12, 10, 0, 1), operation="show", text=ref, refs=(ref,))
    reduction = run(tmp_path, records, events)
    assert reduction.executions[0].attribution is Attribution.INFERRED_UNIQUE
    assert reduction.executions[0].operation == Operation.SHOW.value
    assert Phase.STARTED.value == "started"


def test_cli_value_options_preserve_fingerprint_and_canonical_cwd(tmp_path: Path) -> None:
    actual = tmp_path / "workspace"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    command = "ssgrep search --limit 7 --where=runtime=claude repair"
    records = [
        claude_prompt("cons", "fix deployment", ts=ts()),
        claude_tool_use("cons", "search", "Bash", {"command": command}, cwd=str(alias), ts=ts(s=1)),
        claude_tool_result("cons", "search", results_json(), ts=ts(s=3)),
    ]
    evidence = claude_evidence(tmp_path / "store", {"project/cons.jsonl": records})
    journals = event(
        "search", ts(s=2), text="repair", where="runtime=claude", transport="cli", cwd=str(actual)
    )
    reduced = reduce_profile(
        PROFILE, evidence, journals, policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    assert reduced.executions[0].attribution is Attribution.INFERRED_UNIQUE
    assert reduced.executions[0].session_id == "cons"


@pytest.mark.parametrize(
    "argv, expected",
    [
        (("note", "--title", "repair notes", "--body", "body"), ("note", "repair notes", None)),
        (("search", "--where", "runtime=claude", "repair"), ("search", "repair", "runtime=claude")),
        (("search", "repair", "--where"), ("search", "repair", "")),
    ],
)
def test_cli_option_values_never_become_query_text(argv: tuple[str, ...], expected: tuple) -> None:
    assert parse_argv(argv) == expected


def test_missing_transcript_clock_prevents_time_inferred_join(tmp_path: Path) -> None:
    records = consumer(action=None)
    for record in records:
        record.pop("timestamp", None)
    evidence = claude_evidence(tmp_path, {"project/consumer.jsonl": records})
    reduced = reduce_profile(
        PROFILE,
        evidence,
        event("lookup", ts(), text="why did the deploy fail"),
        policy=policy(),
        frequency=VIEW,
        fingerprint_key=KEY,
    )
    assert reduced.executions[0].attribution is Attribution.UNATTRIBUTED
    assert reduced.executions[0].session_id is None
    assert reduced.invocations[0].execution_ids == ()


"""Attribution boundaries exercised through native consumers and journal events."""


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
