"""5.4/5.5: result interpretation, exposure, shell mixing, per-event scope."""

from __future__ import annotations

import json
from pathlib import Path

from ssgrep.accounting.contracts import Attribution, Exposure, Outcome
from ssgrep.accounting.exposure import interpret, shell_shape
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_types import Reduction
from tests.accounting.reduce_builders import (
    CWD,
    KEY,
    MCP,
    PROFILE,
    card,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    event,
    policy,
    results_json,
    ts,
)

INDEX_ERROR = json.dumps({"error": "The global index is incomplete; run `ssgrep index`."})


def reduce_one(root: Path, records: list[dict], events: list, **kw) -> Reduction:
    evidence = claude_evidence(root, {"-work/s1.jsonl": records})
    return reduce_profile(
        PROFILE,
        evidence,
        events,
        policy=kw.pop("policy", policy()),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )


def mcp_search(body: str, *, is_error: bool | None = False) -> list[dict]:
    return [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", MCP, {"query": "deploy failure"}, ts=ts(12, 10)),
        claude_tool_result("s1", "t1", body, is_error=is_error, ts=ts(12, 10, 0, 2)),
    ]


def test_error_payload_with_transport_success_is_an_error_whose_bytes_are_still_exposure(
    tmp_path: Path,
) -> None:
    events = event("n-1", ts(12, 10, 0, 1), outcome=Outcome.UNKNOWN, phases=())
    started = event("n-1", ts(12, 10, 0, 1), outcome=Outcome.OK)
    reduction = reduce_one(tmp_path, mcp_search(INDEX_ERROR, is_error=False), started)
    (execution,) = reduction.executions
    assert events == []
    assert execution.outcome is Outcome.ERROR  # journal said ok; the payload overrides it
    assert execution.outcome_basis == "result_override"
    assert execution.exposure is Exposure.OBSERVED
    assert execution.visible_bytes.value == len(INDEX_ERROR.encode())
    (invocation,) = reduction.invocations
    assert invocation.outcome is Outcome.ERROR
    assert reduction.candidates == ()
    assert reduction.observations["failed"].value == 1
    # the error response is still a debit: it cost the consumer context
    (bundle,) = reduction.bundles
    chars = [d for d in bundle.debits if d.unit.unit == "context_tokens"]
    assert [d.chars for d in chars] == [len(INDEX_ERROR)]


def test_native_error_flag_and_exit_status_override_transport_success() -> None:
    flagged = interpret({"output": "anything", "is_error": True})
    assert flagged.outcome is Outcome.ERROR and flagged.basis == "native_flag"
    exited = interpret({"output": "boom\nexit code: 2", "is_error": False})
    assert exited.outcome is Outcome.ERROR and exited.basis == "exit_status"
    wrapped = interpret(
        {"output": json.dumps({"output": "boom", "metadata": {"exit_code": 1}}), "is_error": None}
    )
    assert wrapped.outcome is Outcome.ERROR and wrapped.exit_code == 1
    good = interpret({"output": results_json(card("claude:a:ep:0", "x")), "is_error": False})
    assert good.outcome is Outcome.OK and good.explicit_success
    empty = interpret({"output": results_json(), "is_error": False})
    assert empty.outcome is Outcome.EMPTY
    opaque = interpret({"output": "some free text", "is_error": None})
    assert opaque.outcome is Outcome.UNKNOWN  # an uninterpretable format stays unknown


def test_truncated_response_is_partial_exposure_and_hidden_candidates_earn_nothing(
    tmp_path: Path,
) -> None:
    visible = card("claude:src:ep:0", "intro text only")
    body = results_json(visible, excerpts_truncated=True, omitted_count=3)
    call = mcp_search(body)
    call.append(
        claude_tool_use(
            "s1", "t2", "Bash", {"command": "vim services/billing/limits.py"}, ts=ts(12, 10, 1)
        )
    )
    reduction = reduce_one(tmp_path, call, event("n-1", ts(12, 10, 0, 1)))
    (execution,) = reduction.executions
    assert execution.exposure is Exposure.PARTIAL
    assert execution.reuse.value == "none_observed"  # the path was never visible
    assert reduction.candidates == ()


def test_piped_or_mixed_shell_output_is_partial_and_not_charged_to_retrieval(
    tmp_path: Path,
) -> None:
    command = 'ssgrep search "deploy failure" && cat build.log && make test'
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", "Bash", {"command": command}, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "t1", "x" * 5000, ts=ts(12, 10, 0, 5)),
    ]
    reduction = reduce_one(tmp_path, records, event("n-1", ts(12, 10, 0, 1), transport="cli"))
    (execution,) = reduction.executions
    assert execution.exposure is Exposure.PARTIAL
    assert execution.visible_bytes.value is None  # unattributable, not 5000
    (bundle,) = reduction.bundles
    (chars,) = [d for d in bundle.debits if d.unit.unit == "context_tokens"]
    assert chars.amount.value is None and chars.chars is None and chars.partial


def test_shell_shapes() -> None:
    assert shell_shape('ssgrep search "x"').kind == "pure"
    assert shell_shape('ssgrep search "x" | head -5').kind == "filtered"
    assert shell_shape('ssgrep search "x"; ls').kind == "mixed"
    assert shell_shape('for q in a b; do ssgrep search "$q"; done').kind == "pure"
    assert shell_shape("ssgrep search x | jq .results | wc -l").kind == "filtered"
    assert shell_shape("ssgrep search x && make test").kind == "mixed"


def test_repeated_lookup_costs_twice(tmp_path: Path) -> None:
    body = results_json(card("claude:src:ep:0", "same card"))
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", MCP, {"query": "deploy failure"}, ts=ts(12, 10)),
        claude_tool_result("s1", "t1", body, ts=ts(12, 10, 0, 2)),
        claude_tool_use("s1", "t2", MCP, {"query": "deploy failure again"}, ts=ts(12, 10, 5)),
        claude_tool_result("s1", "t2", body, ts=ts(12, 10, 5, 2)),
    ]
    reduction = reduce_one(tmp_path, records, [])
    (bundle,) = reduction.bundles
    chars = [d.chars for d in bundle.debits if d.unit.unit == "context_tokens"]
    assert chars == [len(body), len(body)]
    assert len({d.debit_id for d in bundle.debits}) == len(bundle.debits)


def test_per_event_scope_follows_cwd_changes_inside_one_session(tmp_path: Path) -> None:
    org = policy("/work/orgrepo", profile="acme")
    records = [
        claude_prompt("s1", "go", ts=ts(12, 9)),
        claude_tool_use(
            "s1", "t1", MCP, {"query": "in repo"}, ts=ts(12, 10), cwd="/work/orgrepo/app"
        ),
        claude_tool_result("s1", "t1", results_json(), ts=ts(12, 10, 0, 2)),
        claude_tool_use(
            "s1", "t2", MCP, {"query": "elsewhere"}, ts=ts(12, 11), cwd="/home/me/side"
        ),
        claude_tool_result("s1", "t2", results_json(), ts=ts(12, 11, 0, 2)),
    ]
    events = [
        *event("n-1", ts(12, 10, 0, 1), text="in repo", cwd="/work/orgrepo/app", profile="acme"),
        *event("n-2", ts(12, 11, 0, 1), text="elsewhere", cwd="/home/me/side", profile="acme"),
    ]
    evidence = claude_evidence(tmp_path, {"-work/s1.jsonl": records})
    reduction = reduce_profile(
        "acme", evidence, events, policy=org, frequency=FrequencyView.unknown(), fingerprint_key=KEY
    )
    assert [e.execution_id for e in reduction.executions] == ["n-1"]
    assert len(reduction.invocations) == 1
    assert reduction.coverage["executions_out_of_scope"] == 1
    assert reduction.coverage["invocations_out_of_scope"] == 1
    assert reduction.executions[0].attribution is Attribution.INFERRED_UNIQUE


def test_verifier_actor_lookups_count_globally_but_never_become_solver_exposure(
    tmp_path: Path,
) -> None:
    body = results_json(card("claude:src:ep:0", "services/billing/limits.py"))
    records = [
        claude_prompt("s1", "verify", ts=ts(12, 9)),
        claude_tool_use("s1", "t1", MCP, {"query": "deploy failure"}, ts=ts(12, 10)),
        claude_tool_result("s1", "t1", body, ts=ts(12, 10, 0, 2)),
    ]
    evidence = claude_evidence(tmp_path, {"-work/s1.jsonl": records})
    reduction = reduce_profile(
        PROFILE,
        evidence,
        event("n-1", ts(12, 10, 0, 1)),
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
        verifier_sessions=frozenset({"claude:s1"}),
    )
    (execution,) = reduction.executions
    assert execution.actor == "verifier"
    assert reduction.observations["searches"].value == 1  # still a real global execution
    assert reduction.bundles == ()  # but neither debit nor credit for any solver
    assert reduction.candidates == ()
