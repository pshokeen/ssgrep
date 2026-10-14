from __future__ import annotations

import json
from dataclasses import replace

from eval.accounting_airline.accounting_oracle import arithmetic_profile, evaluate
from eval.accounting_airline.observer import Frozen
from eval.accounting_airline.oracle_rules import is_mixed
from tests.accounting.airline.oracle_fixtures import (
    TOKEN,
    SessionBuilder,
    consumer_b,
    frozen,
    observed,
    search_call,
    search_json,
    source_a,
    text,
)

PROFILE = arithmetic_profile([("identifier", TOKEN)])


def test_cli_usage_rejection_is_not_an_execution_but_service_error_is() -> None:
    sa, sb = source_a(), consumer_b()
    usage = observed(
        "usage",
        "solver-b",
        ("ssgrep", "search", "--limit", "abc"),
        native="bad",
        exit_code=2,
        stdout=json.dumps({"ok": False, "error": {"type": "BadParameter"}}),
    )
    service_error = observed(
        "where",
        "solver-b",
        ("ssgrep", "search", "--where", "bad predicate"),
        native="bad2",
        exit_code=2,
        stdout=json.dumps({"ok": False, "error": {"type": "ValueError"}}),
    )
    sb.call("bad", "ssgrep search --limit abc")
    sb.result("bad", usage.stdout, error=True)
    sb.call("bad2", "ssgrep search --where 'bad predicate'")
    sb.result("bad2", service_error.stdout, error=True)
    result = evaluate(frozen([sa, sb], [usage, service_error]), PROFILE)
    assert result.counts["executions"] == 1
    assert result.counts["executions_verified"] == 1
    assert result.counts["error"] == 1
    assert result.counts["journaled_search"] == 1
    assert result.counts["exposures"] == 2
    assert {(d.call, d.chars) for d in result.debits} == {
        ("bad", len(usage.stdout)),
        ("bad2", len(service_error.stdout)),
    }


def _loop_world(*, unrelated: bool = False):
    sa = source_a()
    sb = SessionBuilder("sb", "solver-b", "L3", 5000)
    sb.user("Use the connection policy after searching.")
    command = 'for q in "minimum connection" "airport rules"; do ssgrep search "$q" --json; done'
    if unrelated:
        command += "; echo unrelated-output"
    sb.call("loop", command)
    one = search_json(TOKEN, 200)
    two = search_json("MCT-ORD-045", 180)
    visible = one + two
    sb.result("loop", visible)
    sb.call("act", f"psql -c select '{TOKEN}'")
    sb.result("act", text(20, "ok"))
    calls = [
        observed(
            "loop-a",
            "solver-b",
            ("ssgrep", "search", "minimum connection"),
            native="loop",
            stdout=one,
        ),
        observed(
            "loop-b",
            "solver-b",
            ("ssgrep", "search", "airport rules"),
            native="loop",
            stdout=two,
        ),
    ]
    return frozen([sa, sb], calls), visible, command


def test_pure_retrieval_loop_is_one_visible_aggregate_not_mixed_output() -> None:
    world, visible, command = _loop_world()
    assert not is_mixed(command, 2)
    result = evaluate(world, PROFILE)
    assert result.counts["executions"] == 2
    assert result.counts["invocations"] == 1
    assert result.counts["exposures"] == 1
    assert result.debits[0].chars == len(visible)
    assert result.estimates["central"].debit == len(visible) // 4
    assert result.verdicts[0].token == ("identifier", TOKEN)
    credit = result.estimates["central"].credit
    assert credit is not None and credit > 0


def test_retrieval_with_unrelated_output_stays_unattributable() -> None:
    world, _visible, command = _loop_world(unrelated=True)
    assert is_mixed(command, 2)
    result = evaluate(world, PROFILE)
    assert result.counts["invocations"] == 1
    assert result.counts["exposures"] == 0
    assert result.debits[0].chars is None
    assert result.debits[0].partial
    assert result.estimates["central"].net is None
    assert result.estimates["central"].credit == 0


def test_partial_native_result_keeps_measured_debit_and_hides_candidate() -> None:
    sa, sb = source_a(), consumer_b(action=TOKEN)
    visible = '{"ok":true,"data":{"results":[{"ref":"omp:sa:ep:1","response_text":"MCT-'
    sb.records[2] = replace(sb.records[2], text=visible)
    call = search_call(stdout=search_json(TOKEN, 200))
    result = evaluate(frozen([sa, sb], [call]), PROFILE)
    assert result.debits[0].chars == len(visible)
    assert result.estimates["central"].debit == len(visible) // 4
    assert result.counts["truncated_exposures"] == 1
    assert result.verdicts == ()
    assert result.estimates["central"].net == -(len(visible) // 4)


def _relay_world(tmp_path, *, parent_link: bool, payload: bool, sibling: bool = False):
    sa = source_a()
    child = consumer_b(action=None)
    child.sid = "child"
    child.records = [
        replace(r, record_id=r.record_id.replace("sb-", "child-")) for r in child.records
    ]
    parent = SessionBuilder("parent", "solver-b", "L3", 7000)
    parent.user("Delegate lookup and apply the returned policy.")
    parent.call("relay", "run child session")
    parent.result("relay", f"child reported {TOKEN}" if payload else "child finished")
    parent.call("act", f"psql -c select '{TOKEN}'")
    parent.result("act", text(20, "ok"))
    root = tmp_path / f"linked-{parent_link}-{payload}-{sibling}"
    root.mkdir()
    child_path = root / "child.jsonl"
    parent_path = root / "parent.jsonl"
    parent_alias = root / "collect" / "parent.jsonl"
    child_entry = child.entry()
    parent_entry = parent.entry()
    child_native = replace(
        child_entry[2],
        path=str(child_path),
        parent_session=str(parent_alias) if parent_link else None,
    )
    parent_native = replace(
        parent_entry[2], path=str(parent_path), source_aliases=(str(parent_alias),)
    )
    child_entry = (child_entry[0], child_entry[1], child_native)
    parent_entry = (parent_entry[0], parent_entry[1], parent_native)
    sessions = [sa.entry(), child_entry, parent_entry]
    if sibling:
        sib = consumer_b(action=TOKEN)
        sib.sid = "sibling"
        sessions.append(sib.entry())
    dispatch = search_call(native="cs1", stdout=search_json(TOKEN, 200))
    dispatch = replace(dispatch, task="child#0")
    # Parent metadata is part of frozen evidence; the oracle must not reopen native paths.

    return Frozen("run", (dispatch,), {x[2].session_id: x for x in sessions}, (), "head")


def test_explicit_child_relay_credits_parent_but_requires_link_and_payload(tmp_path) -> None:
    valid = evaluate(_relay_world(tmp_path, parent_link=True, payload=True), PROFILE)
    assert valid.verdicts[0].reuse == "action_reference"
    assert valid.verdicts[0].task == "child#0"
    relay_credit = valid.estimates["central"].credit
    assert relay_credit is not None and relay_credit > 0
    missing_payload = evaluate(_relay_world(tmp_path, parent_link=True, payload=False), PROFILE)
    assert missing_payload.verdicts[0].reuse == "none_observed"
    assert missing_payload.estimates["central"].credit == 0
    missing_link = evaluate(_relay_world(tmp_path, parent_link=False, payload=True), PROFILE)
    assert missing_link.verdicts[0].reuse == "none_observed"
    assert missing_link.estimates["central"].credit == 0
    sibling = evaluate(
        _relay_world(tmp_path, parent_link=False, payload=False, sibling=True), PROFILE
    )
    assert not sibling.verdicts or sibling.verdicts[0].reuse == "none_observed"
    assert sibling.estimates["central"].credit == 0
