"""Observed session reductions exercise provenance, inspection, and request boundaries."""

from __future__ import annotations

from pathlib import Path

from ssgrep.accounting.contracts import Attribution, RawEvidence, Reuse, canonical_bytes
from ssgrep.accounting.plain_codec import from_plain, loads
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.sources import Edge, find_cycles
from ssgrep.accounting.valuation_inputs import AtomStatus
from tests.accounting.reduce_builders import (
    CWD,
    KEY,
    MCP,
    PROFILE,
    SHOW,
    card,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    event,
    policy,
    rare_view,
    results_json,
    ts,
)
from tests.accounting.test_reduce_evidence import FACT, PATH, VIEW, consumer, reduce_scene


def test_inspection_without_action_does_not_credit_the_opened_fact(tmp_path: Path) -> None:
    extra = [
        claude_tool_use("cons", "show", SHOW, {"ref": "claude:src1:ep:0"}, ts=ts(12, 10, 1)),
        claude_tool_result("cons", "show", "no new candidate", ts=ts(12, 10, 1, 1)),
    ]
    reduction = reduce_scene(tmp_path, consumer(action=None, extra=extra))
    assert reduction.candidates == ()
    assert reduction.coverage["candidates_unreferenced"] == 2
    assert reduction.bundles[0].debits  # inspection is not free output


def test_reference_in_next_user_task_does_not_retroactively_credit_lookup(tmp_path: Path) -> None:
    extra = [
        claude_prompt("cons", "another task", uuid="u2", ts=ts(12, 10, 1)),
        claude_tool_use("cons", "act", "Bash", {"command": f"cat {PATH}"}, ts=ts(12, 10, 2)),
        claude_tool_result("cons", "act", "ok", ts=ts(12, 10, 2, 1)),
    ]
    reduction = reduce_scene(tmp_path, consumer(action=None, extra=extra))
    assert reduction.candidates == ()
    assert reduction.coverage["candidates_unreferenced"] == 2
    assert reduce_scene(tmp_path / "same-turn", consumer()).candidates


def test_ref_less_visible_fact_is_unresolved_not_free_credit(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(body=FACT))
    assert {row.reason for row in reduction.exclusions} == {"source_unresolved"}
    assert reduction.candidates[0].candidate.immediate[0].status is AtomStatus.UNAVAILABLE
    assert reduction.bundles[0].debits[0].chars == len(FACT)


def test_note_investment_is_accounted_without_creating_retrieval_credit(tmp_path: Path) -> None:
    records = [
        claude_prompt("cons", "store a note", ts=ts()),
        claude_tool_use(
            "cons", "note", "Bash", {"command": 'ssgrep note --title "repair notes"'}, ts=ts(s=1)
        ),
        claude_tool_result("cons", "note", "saved", ts=ts(s=2)),
    ]
    reduction = reduce_scene(tmp_path, records)
    assert reduction.coverage["recognized_invocations"] == 1
    assert reduction.invocations[0].operations == ("note",)
    assert reduction.candidates == ()
    assert all(not bundle.candidates and not bundle.debits for bundle in reduction.bundles)


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


def test_reciprocal_source_reuse_marks_only_returned_work_cyclic(tmp_path: Path) -> None:
    first, second = "services/first/repair.py", "services/second/repair.py"
    a = [
        claude_prompt("a", "repair first issue", ts=ts(2)),
        claude_tool_use("a", "search", MCP, {"query": "earlier solution"}, ts=ts(2, s=1)),
        claude_tool_result(
            "a", "search", results_json(card("claude:b:ep:0", second)), ts=ts(2, s=2)
        ),
        claude_tool_use("a", "act", "Bash", {"command": f"cat {second}"}, ts=ts(2, s=3)),
        claude_tool_result("a", "act", first, ts=ts(2, s=4)),
    ]
    b = [
        claude_prompt("b", "investigate second issue", ts=ts(1)),
        claude_tool_use(
            "b", "witness", "Bash", {"command": "cat deploy.log"}, cwd=CWD, ts=ts(1, s=1)
        ),
        claude_tool_result("b", "witness", second, ts=ts(1, s=2)),
        claude_prompt("b", "reuse later repair", uuid="u2", ts=ts(3)),
        claude_tool_use("b", "search", MCP, {"query": "first issue solution"}, ts=ts(3, s=1)),
        claude_tool_result(
            "b", "search", results_json(card("claude:a:ep:0", first)), ts=ts(3, s=2)
        ),
        claude_tool_use("b", "act", "Bash", {"command": f"cat {first}"}, ts=ts(3, s=3)),
        claude_tool_result("b", "act", "ok", ts=ts(3, s=4)),
    ]
    evidence = claude_evidence(tmp_path, {"project/a.jsonl": a, "project/b.jsonl": b})
    reduced = reduce_profile(
        PROFILE,
        evidence,
        [],
        policy=policy(),
        frequency=rare_view(first, second),
        fingerprint_key=KEY,
    )
    by_root = {row.candidate.task.root: row.candidate for row in reduced.candidates}
    assert set(by_root) == {"claude:a", "claude:b"}
    assert all(atom.status is AtomStatus.OK for atom in by_root["claude:a"].immediate)
    assert all(atom.status is AtomStatus.CYCLE for atom in by_root["claude:b"].immediate)
    assert reduced.coverage["exclusion.provenance_cycle"] == 1
    assert by_root["claude:b"].reuse is Reuse.ACTION_REFERENCE


def test_unknown_exposure_time_cannot_establish_a_provenance_cycle() -> None:
    # B's absorption of A must precede the witness reused by A, not merely exist.
    target = Edge("a", ("b",), 20, 30)
    assert find_cycles([target, Edge("b", ("a",), 5, None)]) == set()
    assert find_cycles([target, Edge("b", ("a",), 5, 10)]) == {0}
    # A self-reference is not evidence of a cross-root cycle; recursive graphs terminate.
    assert find_cycles([target, Edge("b", ("b",), 5, 10)]) == set()
    cyclic = [Edge("a", ("b",), 20, 1), Edge("b", ("c",), 20, 1), Edge("c", ("b",), 20, 1)]
    assert find_cycles(cyclic) == {1, 2}


def test_persisted_mcp_status_is_not_a_lookup_but_following_search_is(tmp_path: Path) -> None:
    search_body = results_json()
    records = [
        claude_prompt("cons", "check index before searching", ts=ts()),
        claude_tool_use("cons", "status", "mcp__renamed_index_status", {}, ts=ts(s=1)),
        claude_tool_result("cons", "status", '{"sessions":7}', ts=ts(s=2)),
        claude_tool_use("cons", "search", MCP, {"query": "deployment"}, ts=ts(s=3)),
        claude_tool_result("cons", "search", search_body, ts=ts(s=4)),
    ]
    native = claude_evidence(tmp_path, {"project/consumer.jsonl": records})
    retained = from_plain(list[RawEvidence], loads(canonical_bytes(native)))
    reduced = reduce_profile(
        PROFILE, retained, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    assert reduced.coverage["classified_tool_calls"] == 2
    assert reduced.coverage["inv_unclassified"] == reduced.coverage["inv_mention"] == 0
    (invocation,) = reduced.invocations
    assert invocation.operations == ("search",)
    assert reduced.candidates == ()
    (bundle,) = reduced.bundles
    assert [cost.chars for cost in bundle.debits if cost.chars is not None] == [len(search_body)]
    assert all(cost.exposure_id == invocation.invocation_id for cost in bundle.debits)
