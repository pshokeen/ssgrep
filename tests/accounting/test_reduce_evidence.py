"""6.1/6.2/6.5: candidate extraction, novelty, relay vs sibling, interpreted outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from ssgrep.accounting.contracts import Reuse
from ssgrep.accounting.evidence_tokens import scan_text, tokens_of
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_types import Reduction
from tests.accounting.reduce_builders import (
    KEY,
    MCP,
    PROFILE,
    card,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
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
    source_session,
    ts,
)

PATH = "services/billing/limits.py"
CODE = "ERR_QUOTA_EXCEEDED"
FACT = f"deploy failed: {CODE} raised in {PATH}"
VIEW = rare_view(CODE, PATH)


def consumer(
    *,
    query: str = "why did the deploy fail",
    action: str | None = f"vim {PATH}",
    action_error: bool = False,
    prompt: str = "fix the deploy",
    before: list[dict] | None = None,
    extra: list[dict] | None = None,
    body: str | None = None,
) -> list[dict]:
    body = body if body is not None else results_json(card("claude:src1:ep:0", FACT))
    records = [
        claude_prompt("cons", prompt, uuid="u1", ts=ts(12, 9)),
        *(before or []),
        claude_tool_use("cons", "t1", MCP, {"query": query}, ts=ts(12, 10), message_id="m1"),
        claude_tool_result("cons", "t1", body, ts=ts(12, 10, 0, 2)),
    ]
    if action is not None:
        records += [
            claude_tool_use(
                "cons", "t2", "Bash", {"command": action}, ts=ts(12, 10, 1), message_id="m2"
            ),
            claude_tool_result("cons", "t2", "ok", is_error=action_error, ts=ts(12, 10, 1, 2)),
        ]
    return records + (extra or [])


def reduce_scene(
    root: Path,
    cons: list[dict],
    *,
    view: FrequencyView = VIEW,
    known_atoms: dict[str, str] | None = None,
    src: list[dict] | None = None,
) -> Reduction:
    files = {
        "-work/src1.jsonl": src if src is not None else source_session("src1", FACT),
        "-work/cons.jsonl": cons,
    }
    return reduce_profile(
        PROFILE,
        claude_evidence(root, files),
        [],
        policy=policy(),
        frequency=view,
        fingerprint_key=KEY,
        known_atoms=known_atoms,
    )


def credited(reduction: Reduction) -> dict[str, Reuse]:
    return {
        c.candidate_type + ":" + str(c.rarity_milli): c.candidate.reuse
        for c in reduction.candidates
    }


def test_extraction_takes_observed_identifiers_paths_flags_and_codes_only() -> None:
    text = (
        f"{CODE} in {PATH} via --retry-budget; at 2026-10-12 see claude:abc:ep:3 "
        "in /Users/someone/code/app/main.py by someone; value is the generic_total"
    )
    kept = {(t.kind, t.text) for t in tokens_of(text)}
    assert ("error_code", CODE) in kept
    assert ("path", PATH) in kept
    assert ("flag", "--retry-budget") in kept
    scan = scan_text(text)
    reasons = set(scan.excluded.values())
    assert "home_path" in reasons
    assert not any("2026-10-12" in t.text for t in scan.kept)  # dates
    assert not any(t.text.startswith("claude:") for t in scan.kept)  # refs
    assert not any("someone" in t.text and t.kind != "identifier" for t in scan.kept)


def test_novel_rare_referenced_fact_becomes_a_candidate_with_nested_atoms(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer())
    (record,) = reduction.candidates
    cand = record.candidate
    assert cand.candidate_type == "path"
    assert cand.reuse is Reuse.ACTION_REFERENCE
    assert cand.novel and not cand.user_supplied and cand.action_outcome_ok
    assert cand.immediate and cand.discovery and cand.episode
    assert all(a.usable for a in cand.immediate + cand.discovery + cand.episode)
    assert cand.task.turn == "turn:u1" and cand.task.root == "claude:cons"
    assert record.association == "association_not_causation"
    assert record.novelty_confidence == "high"
    assert record.rarity_milli is not None and record.rarity_milli >= 8000
    assert cand.witness_locators and cand.action_locators
    # the unreferenced error code is exposure only: no candidate, no credit
    assert reduction.coverage["candidates_unreferenced"] == 1


def test_substring_of_a_candidate_is_not_an_action_reference(tmp_path: Path) -> None:
    for action in ("vim limits.py", f"vim {PATH}.bak.old", "vim services/billing"):
        reduction = reduce_scene(
            tmp_path / action.replace(" ", "_").replace("/", "_"), consumer(action=action)
        )
        assert reduction.candidates == (), action


def test_fact_already_in_the_initiating_query_does_not_qualify(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(query=f"what about {PATH}"))
    assert reduction.candidates == ()
    assert reduction.coverage["exclusion.query_supplied"] >= 1


def test_fact_supplied_by_the_user_prompt_does_not_qualify(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(prompt=f"please look at {PATH}"))
    assert reduction.candidates == ()
    assert reduction.coverage["exclusion.user_supplied"] >= 1


def test_fact_already_visible_in_earlier_context_does_not_qualify(tmp_path: Path) -> None:
    earlier = [
        claude_tool_use("cons", "p1", "Bash", {"command": "cat build.log"}, ts=ts(12, 9, 5)),
        claude_tool_result("cons", "p1", f"saw {PATH} earlier", ts=ts(12, 9, 6)),
    ]
    reduction = reduce_scene(tmp_path, consumer(before=earlier))
    assert reduction.candidates == ()
    assert reduction.coverage["exclusion.already_visible"] >= 1


def test_compaction_or_missing_context_lowers_novelty_confidence_without_dropping_it(
    tmp_path: Path,
) -> None:
    compacted = [
        claude_prompt(
            "cons",
            "This session is being continued from a previous conversation that ran out",
            uuid="u0",
            ts=ts(12, 9, 1),
        )
    ]
    reduction = reduce_scene(tmp_path / "a", consumer(before=compacted))
    (record,) = reduction.candidates
    assert record.novelty_confidence == "reduced"
    assert "compaction_marker_in_context" in record.novelty_reasons
    headless = [
        claude_tool_use("cons", "t1", MCP, {"query": "why"}, ts=ts(12, 10)),
        claude_tool_result(
            "cons", "t1", results_json(card("claude:src1:ep:0", FACT)), ts=ts(12, 10, 0, 2)
        ),
        claude_tool_use("cons", "t2", "Bash", {"command": f"vim {PATH}"}, ts=ts(12, 10, 1)),
        claude_tool_result("cons", "t2", "ok", ts=ts(12, 10, 1, 2)),
    ]
    reduction = reduce_scene(tmp_path / "b", headless)
    (record,) = reduction.candidates
    assert record.novelty_confidence == "reduced"
    assert "session_start_not_observed" in record.novelty_reasons


def test_action_that_errors_is_contradicted_and_earns_no_credit_in_any_scenario(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.reduce import methodology_for
    from ssgrep.accounting.valuation import value_evidence

    reduction = reduce_scene(tmp_path, consumer(action_error=True))
    (record,) = reduction.candidates
    assert record.candidate.reuse is Reuse.CONTRADICTED
    assert record.outcome == "contradicted"
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    assert {a.credit_raw for a in valuation.allocations} == {0}
    assert {a.ineligible for a in valuation.allocations} == {"contradicted"}


def test_transport_success_alone_never_validates_but_a_passing_verifier_does(
    tmp_path: Path,
) -> None:
    plain = reduce_scene(tmp_path / "a", consumer())
    assert plain.candidates[0].candidate.reuse is Reuse.ACTION_REFERENCE
    verified = consumer(
        extra=[
            claude_tool_use(
                "cons", "t3", "Bash", {"command": "uv run pytest -q"}, ts=ts(12, 10, 2)
            ),
            claude_tool_result(
                "cons", "t3", "12 passed in 0.4s", is_error=False, ts=ts(12, 10, 2, 5)
            ),
        ]
    )
    validated = reduce_scene(tmp_path / "b", verified)
    assert validated.candidates[0].candidate.reuse is Reuse.VALIDATED
    failing = consumer(
        extra=[
            claude_tool_use(
                "cons", "t3", "Bash", {"command": "uv run pytest -q"}, ts=ts(12, 10, 2)
            ),
            claude_tool_result(
                "cons", "t3", "3 failed, 9 passed", is_error=False, ts=ts(12, 10, 2, 5)
            ),
        ]
    )
    assert (
        reduce_scene(tmp_path / "c", failing).candidates[0].candidate.reuse
        is Reuse.ACTION_REFERENCE
    )


def test_truncated_show_hides_the_candidate_so_no_exposure_credit(tmp_path: Path) -> None:
    body = json.dumps(
        {
            "episode_id": "claude:src1:ep:0",
            "prompt_text": "intro",
            "response_text": "",
            "truncated": True,
        }
    )  # the fact sat beyond the truncation point and was never shown
    cons = [
        claude_prompt("cons", "fix the deploy", uuid="u1", ts=ts(12, 9)),
        claude_tool_use(
            "cons",
            "t1",
            "mcp__acme_recall__show_session",
            {"ref": "claude:src1:ep:0"},
            ts=ts(12, 10),
        ),
        claude_tool_result("cons", "t1", body, ts=ts(12, 10, 0, 2)),
        claude_tool_use("cons", "t2", "Bash", {"command": f"vim {PATH}"}, ts=ts(12, 10, 1)),
        claude_tool_result("cons", "t2", "ok", ts=ts(12, 10, 1, 2)),
    ]
    reduction = reduce_scene(tmp_path, cons)
    assert reduction.candidates == ()
    assert reduction.invocations[0].exposure.value == "partial"


def test_user_supplied_source_fact_earns_no_discovery_credit(tmp_path: Path) -> None:
    src = [
        claude_prompt("src1", f"I already know {PATH} is broken", uuid="u1", ts=ts(1, 10)),
        claude_tool_use("src1", "w1", "Bash", {"command": "echo hi"}, ts=ts(1, 10, 0, 5)),
        claude_tool_result("src1", "w1", f"confirmed: {PATH}", ts=ts(1, 10, 0, 6)),
    ]
    reduction = reduce_scene(tmp_path, consumer(), src=src)
    (record,) = reduction.candidates
    assert record.candidate.user_supplied is True
    assert ("source_user_supplied", "claude:src1") in {
        (e.reason, e.detail) for e in reduction.exclusions
    }


def omp_relay_scene(
    root: Path, *, relay: bool, sibling_mentions: bool, sibling_report: bool = False
) -> Reduction:
    parent = root / ".omp" / "agent" / "sessions" / "-work" / "root.jsonl"
    src = [
        claude_prompt("src1", "why does the deploy fail", uuid="u1", ts=ts(1, 10)),
        claude_tool_use("src1", "w1", "Bash", {"command": "cat deploy.log"}, ts=ts(1, 10, 0, 5)),
        claude_tool_result("src1", "w1", FACT, ts=ts(1, 10, 0, 6)),
    ]
    claude_ev = claude_evidence(root, {"-work/src1.jsonl": src})
    parent_records = [
        omp_header("root-1", ts(12, 8)),
        omp_user("e-a", "fix the deploy", ts(12, 9)),
        *omp_launch("l1", "c-1", "ScoutA", ts(12, 9, 5), ts(12, 9, 6)),
        *omp_launch("l2", "c-2", "ScoutB", ts(12, 9, 7), ts(12, 9, 8)),
    ]
    if relay:
        parent_records += [
            omp_result(
                "e-rel", "c-1", f"ScoutA reports: {PATH} holds the quota check", ts(12, 10, 5)
            ),
            omp_call("e-act", "c-act", "bash", {"command": f"vim {PATH}"}, ts(12, 10, 6)),
            omp_result("e-actr", "c-act", "ok", ts(12, 10, 7)),
        ]
    if sibling_report:  # the parent hears the fact from the OTHER child, then acts on it
        parent_records += [
            omp_result(
                "e-rel", "c-2", f"ScoutB reports: {PATH} holds the quota check", ts(12, 10, 5)
            ),
            omp_call("e-act", "c-act", "bash", {"command": f"vim {PATH}"}, ts(12, 10, 6)),
            omp_result("e-actr", "c-act", "ok", ts(12, 10, 7)),
        ]
    a = [
        omp_header("scout-a", ts(12, 9, 10), parent=parent),
        *omp_search(
            "e-s", "k1", "deploy", results_json(card("claude:src1:ep:0", FACT)), (12, 10, 0, 0)
        ),
    ]
    b = [omp_header("scout-b", ts(12, 9, 11), parent=parent)]
    if sibling_mentions:
        b += [
            omp_call("e-b1", "kb", "bash", {"command": f"cat {PATH}"}, ts(12, 10, 2)),
            omp_result("e-b1r", "kb", "ok", ts(12, 10, 3)),
        ]
    omp_ev = omp_evidence(
        root,
        {
            "-work/root.jsonl": parent_records,
            "-work/root/ScoutA.jsonl": a,
            "-work/root/ScoutB.jsonl": b,
        },
    )
    return reduce_profile(
        PROFILE, claude_ev + omp_ev, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )


def test_explicit_child_to_parent_relay_with_matching_payload_supports_reuse(
    tmp_path: Path,
) -> None:
    reduction = omp_relay_scene(tmp_path, relay=True, sibling_mentions=True)
    (record,) = reduction.candidates
    assert record.candidate.explicit_relay is True
    assert record.candidate.reuse is Reuse.ACTION_REFERENCE
    assert record.candidate.task.turn == "turn:e-a"  # the launching turn, via the child


def test_sibling_coincidence_alone_is_not_a_relay_and_earns_no_reuse(tmp_path: Path) -> None:
    reduction = omp_relay_scene(tmp_path, relay=False, sibling_mentions=True)
    assert reduction.candidates == ()
    assert reduction.coverage["candidates_unreferenced"] >= 1


def test_parent_acting_on_a_different_childs_report_is_not_a_relay_of_this_childs_exposure(
    tmp_path: Path,
) -> None:
    reduction = omp_relay_scene(tmp_path, relay=False, sibling_mentions=False, sibling_report=True)
    assert reduction.candidates == ()
