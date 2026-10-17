"""6.1/6.2/6.5: candidate extraction, novelty, relay vs sibling, interpreted outcomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.capture import fingerprint, journal
from ssgrep.accounting.collect.frequency import FrequencyView
from ssgrep.accounting.collect.replay_evidence import MAX_SCAN_CHARS, Token, scan_text, tokens_of
from ssgrep.accounting.ledger.contracts import (
    Attribution,
    Basis,
    Quantity,
    RawEvidence,
    Reuse,
    TaskId,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_codec import from_plain
from ssgrep.accounting.ledger.ledger_contracts import valid_profile_id
from ssgrep.accounting.ledger.reduce import load_evidence, reduce_profile, reduce_stored_profile
from ssgrep.accounting.ledger.reduce_records import Reduction
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus, Eligibility
from ssgrep.accounting.privacy.storage import loads
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader
from ssgrep.store.paths import data_dir
from tests.accounting.conftest import REAL_HOME, build_event, build_evidence
from tests.accounting.ledger.reduce_builders import (
    KEY,
    MCP,
    PROFILE,
    SHOW,
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
from tests.accounting.readers.readers_fixtures import build_home, write_records

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


def test_extracted_identifiers_keep_structure_but_not_private_or_generic_tokens() -> None:
    visible = (
        "BUILD_CONFIG AB-123 alpha.beta.gamma module.some_value "
        "/123/456 docs/2026-10-12/log.txt /generic a/.. "
        "/home/alice_user/config.py alice_user --help BUILD-2026-10-12 "
        "a_b " + "long_" + "x" * 170
    )
    scan = scan_text(visible)
    assert scan.kept == {
        Token("identifier", "BUILD_CONFIG"),
        Token("identifier", "AB-123"),
        Token("identifier", "alpha.beta.gamma"),
        Token("identifier", "module.some_value"),
    }
    assert scan.excluded[Token("identifier", "alice_user")] == "username"
    assert scan.excluded[Token("identifier", "BUILD-2026-10-12")] == "date"
    assert scan.excluded[Token("flag", "--help")] == "generic"
    assert scan.excluded[Token("path", "/home/alice_user/config.py")] == "home_path"


def test_scan_bound_does_not_create_candidates_from_unseen_suffix() -> None:
    text = "visible_fact " + " " * MAX_SCAN_CHARS + " hidden_fact"
    assert scan_text(text).kept == {Token("identifier", "visible_fact")}
    assert scan_text("").kept == set()


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
    assert reduction.candidates
    assert all(c.candidate.eligibility is Eligibility.KNOWN_REJECTED for c in reduction.candidates)
    assert reduction.coverage["exclusion.query_supplied"] >= 1


def test_fact_supplied_by_the_user_prompt_does_not_qualify(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(prompt=f"please look at {PATH}"))
    assert reduction.candidates
    assert all(c.candidate.eligibility is Eligibility.KNOWN_REJECTED for c in reduction.candidates)
    assert reduction.coverage["exclusion.user_supplied"] >= 1


def test_fact_already_visible_in_earlier_context_does_not_qualify(tmp_path: Path) -> None:
    earlier = [
        claude_tool_use("cons", "p1", "Bash", {"command": "cat build.log"}, ts=ts(12, 9, 5)),
        claude_tool_result("cons", "p1", f"saw {PATH} earlier", ts=ts(12, 9, 6)),
    ]
    reduction = reduce_scene(tmp_path, consumer(before=earlier))
    assert reduction.candidates
    assert all(c.candidate.eligibility is Eligibility.KNOWN_REJECTED for c in reduction.candidates)
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
    assert record.candidate.eligibility is Eligibility.UNAVAILABLE
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
    assert record.candidate.eligibility is Eligibility.UNAVAILABLE


def test_action_that_errors_is_contradicted_and_earns_no_credit_in_any_scenario(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.ledger.reduce import methodology_for
    from ssgrep.accounting.ledger.valuation import value_evidence

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


def omp_action_scene(root: Path, outcome: str, *, verifier: str | None = None) -> Reduction:
    """Real native calls/results have independent entry IDs, never matching suffixes."""
    source = claude_evidence(root, {"source.jsonl": source_session("src1", FACT)})
    records = [
        omp_header("native-consumer", ts(12, 9)),
        omp_user("user-entry", "repair deployment", ts(12, 9, 1)),
        *omp_search(
            "lookup-entry",
            "lookup-id",
            "deploy",
            results_json(card("claude:src1:ep:0", FACT)),
            (12, 10, 0, 0),
        ),
        omp_call("action-entry", "action-id", "bash", {"command": f"vim {PATH}"}, ts(12, 10, 1)),
    ]
    if outcome == "ambiguous":
        records.append(
            omp_call(
                "other-action-entry",
                "action-id",
                "bash",
                {"command": "echo unrelated"},
                ts(12, 10, 1, 1),
            )
        )
    if outcome != "missing":
        records.append(
            omp_result(
                "independent-result-entry",
                "wrong-id" if outcome == "wrong_call_id" else "action-id",
                "ERROR: action failed" if outcome == "error" else "ok",
                ts(12, 10, 1, 2),
                error=outcome == "error",
            )
        )
    if verifier is not None:
        records.extend(
            [
                omp_call(
                    "verifier-entry",
                    "verify-id",
                    "bash",
                    {"command": "uv run pytest -q"},
                    ts(12, 10, 2),
                ),
                omp_result(
                    "independent-verifier-result",
                    "wrong-verify-id" if verifier == "wrong_call_id" else "verify-id",
                    "12 passed in 0.4s",
                    ts(12, 10, 2, 2),
                ),
            ]
        )
    native = omp_evidence(root, {"consumer.jsonl": records})
    return reduce_profile(
        PROFILE, source + native, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )


@pytest.mark.parametrize(
    "outcome,expected_ok,expected_reuse",
    [
        ("success", True, Reuse.ACTION_REFERENCE),
        ("error", False, Reuse.CONTRADICTED),
        ("missing", False, Reuse.ACTION_REFERENCE),
        ("wrong_call_id", False, Reuse.ACTION_REFERENCE),
        ("ambiguous", False, Reuse.ACTION_REFERENCE),
    ],
)
def test_reuse_outcome_uses_verified_native_result_link(
    tmp_path: Path,
    outcome: str,
    expected_ok: bool,
    expected_reuse: Reuse,
) -> None:
    from ssgrep.accounting.ledger.reduce import methodology_for
    from ssgrep.accounting.ledger.valuation import value_evidence

    reduction = omp_action_scene(tmp_path, outcome)
    (record,) = reduction.candidates
    assert record.candidate.action_outcome_ok is expected_ok
    assert record.candidate.reuse is expected_reuse
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    conservative = [
        row.credit_raw for row in valuation.allocations if row.scenario.value == "conservative"
    ]
    assert any(value > 0 for value in conservative) is expected_ok


@pytest.mark.parametrize(
    "verifier,expected",
    [
        ("success", Reuse.VALIDATED),
        ("wrong_call_id", Reuse.ACTION_REFERENCE),
    ],
)
def test_verifier_requires_its_verified_independent_native_result(
    tmp_path: Path,
    verifier: str,
    expected: Reuse,
) -> None:
    (record,) = omp_action_scene(tmp_path, "success", verifier=verifier).candidates
    assert record.candidate.action_outcome_ok
    assert record.candidate.reuse is expected


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


def _reduce_conflict_rows(
    original: list[RawEvidence], changed: list[RawEvidence], changed_first: bool
) -> tuple[Reduction, Reduction]:
    from ssgrep.accounting.collect.replay import evidence_from_line, evidence_line
    from ssgrep.accounting.readers.base import EvidenceLedger
    from ssgrep.accounting.readers.timeline import build_timelines

    ordered = changed + original if changed_first else original + changed
    ledger = EvidenceLedger()
    retained = []
    for item in ordered:
        if ledger.admit(item):
            retained.append(item)
        elif (claim := ledger.ownership_claim(item)) is not None:
            # Claims survive compact restart without retaining the other body.
            retained.append(evidence_from_line(evidence_line(claim)))
    assert build_timelines(retained).conflicts == 1
    baseline = reduce_profile(
        PROFILE, original, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    conflicted = reduce_profile(
        PROFILE, retained, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    return baseline, conflicted


def _claude_conflict(
    root: Path, records: list[dict], changed_record: dict, changed_first: bool
) -> tuple[Reduction, Reduction]:
    original = claude_evidence(
        root / "original",
        {"source.jsonl": source_session("src1", FACT), "consumer.jsonl": records},
    )
    changed = claude_evidence(root / "changed", {"consumer.jsonl": [changed_record]})
    return _reduce_conflict_rows(original, changed, changed_first)


def _context_estimates(reduction: Reduction):
    from ssgrep.accounting.ledger.reduce import methodology_for
    from ssgrep.accounting.ledger.valuation import rollup, value_evidence
    from tests.accounting.ledger.valuation_builders import CHARS

    leaves = [
        leaf
        for bundle in reduction.bundles
        for leaf in value_evidence(bundle, methodology_for(VIEW)).leaves
        if leaf.unit == CHARS
    ]
    assert leaves
    return rollup(leaves, CHARS)


def _context_debits(reduction: Reduction):
    debits = tuple(
        sorted(
            (
                debit
                for bundle in reduction.bundles
                for debit in bundle.debits
                if debit.unit.unit == "context_tokens"
            ),
            key=lambda debit: debit.debit_id,
        )
    )
    visible_chars = len(results_json(card("claude:src1:ep:0", FACT)))
    assert [debit.chars for debit in debits] == [visible_chars]
    assert [debit.amount.value for debit in debits] == [visible_chars // 4]
    return debits


@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("kind", ["call", "result"])
def test_conflicted_action_body_cannot_authorize_reuse_or_credit(
    tmp_path: Path, changed_first: bool, kind: str
) -> None:
    from copy import deepcopy

    records = consumer()
    changed = deepcopy(records[3 if kind == "call" else 4])
    if kind == "call":
        changed["message"]["content"][0]["input"]["command"] = "echo unrelated"
    else:
        changed["message"]["content"][0].update(content="ERROR: failed", is_error=True)
    baseline, conflicted = _claude_conflict(tmp_path, records, changed, changed_first)
    assert baseline.candidates
    assert any(
        row.credit.value is not None and row.credit.value > 0
        for row in _context_estimates(baseline)
    )
    assert conflicted.invocations  # The supported lookup and its actual exposure survive.
    assert _context_debits(conflicted) == _context_debits(baseline)
    assert not conflicted.candidates
    assert all(row.credit.value == 0 for row in _context_estimates(conflicted))


@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("kind", ["call", "result"])
def test_conflicted_verifier_cannot_validate_an_otherwise_supported_action(
    tmp_path: Path, changed_first: bool, kind: str
) -> None:
    from copy import deepcopy

    records = consumer(
        extra=[
            claude_tool_use(
                "cons", "verify", "Bash", {"command": "uv run pytest -q"}, ts=ts(12, 10, 2)
            ),
            claude_tool_result(
                "cons", "verify", "12 passed in 0.4s", is_error=False, ts=ts(12, 10, 2, 5)
            ),
        ]
    )
    changed = deepcopy(records[5 if kind == "call" else 6])
    if kind == "call":
        changed["message"]["content"][0]["input"]["command"] = "echo not a verifier"
    else:
        changed["message"]["content"][0].update(content="3 failed", is_error=True)
    baseline, conflicted = _claude_conflict(tmp_path, records, changed, changed_first)
    assert baseline.candidates and all(
        row.candidate.reuse is Reuse.VALIDATED for row in baseline.candidates
    )
    assert conflicted.candidates and all(
        row.candidate.reuse is Reuse.ACTION_REFERENCE for row in conflicted.candidates
    )
    assert _context_debits(conflicted) == _context_debits(baseline)


@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("kind", ["user", "call", "result"])
def test_conflicted_prior_context_is_unavailable_not_proven_novel_or_known_zero(
    tmp_path: Path, changed_first: bool, kind: str
) -> None:
    from copy import deepcopy

    prior = [
        claude_tool_use("cons", "prior", "Bash", {"command": "echo unrelated"}, ts=ts(12, 9, 30)),
        claude_tool_result("cons", "prior", "unrelated", ts=ts(12, 9, 30, 1)),
    ]
    records = consumer(before=prior)
    changed = deepcopy(records[{"user": 0, "call": 1, "result": 2}[kind]])
    if kind == "user":
        changed["message"]["content"] = f"fix the deploy using {PATH}"
    elif kind == "call":
        changed["message"]["content"][0]["input"]["command"] = f"cat {PATH}"
    else:
        changed["message"]["content"][0]["content"] = PATH
    baseline, conflicted = _claude_conflict(tmp_path, records, changed, changed_first)
    assert baseline.candidates and all(
        row.candidate.eligibility is Eligibility.SUPPORTED for row in baseline.candidates
    )
    assert conflicted.candidates and all(
        row.candidate.eligibility is Eligibility.UNAVAILABLE
        and row.novelty_confidence == "reduced"
        and "prior_context_unavailable" in row.candidate.eligibility_reasons
        for row in conflicted.candidates
    )
    assert _context_debits(conflicted) == _context_debits(baseline)
    assert all(
        row.credit_missing and row.net.value is None and not row.complete
        for row in _context_estimates(conflicted)
    )


@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("ordering", ["future", "earlier_variant", "unknown_variant"])
def test_context_conflicts_respect_proven_future_but_not_uncertain_variant_time(
    tmp_path: Path, changed_first: bool, ordering: str
) -> None:
    from copy import deepcopy

    records = consumer(
        extra=[
            claude_tool_use("cons", "later", "Bash", {"command": "echo unrelated"}, ts=ts(12, 11))
        ]
    )
    changed = deepcopy(records[-1])
    changed["message"]["content"][0]["input"]["command"] = f"cat {PATH}"
    if ordering != "future":
        changed["timestamp"] = ts(12, 9, 30) if ordering == "earlier_variant" else None
    baseline, conflicted = _claude_conflict(tmp_path, records, changed, changed_first)
    expected = Eligibility.SUPPORTED if ordering == "future" else Eligibility.UNAVAILABLE
    assert baseline.candidates and conflicted.candidates
    assert all(row.candidate.eligibility is expected for row in conflicted.candidates)
    assert _context_debits(conflicted) == _context_debits(baseline)
    if ordering == "future":
        assert _context_estimates(conflicted) == _context_estimates(baseline)
    else:
        assert all(row.net.value is None for row in _context_estimates(conflicted))


@pytest.mark.parametrize("changed_first", [False, True])
def test_trusted_user_supplied_fact_still_proves_rejection_despite_other_context_conflict(
    tmp_path: Path, changed_first: bool
) -> None:
    from copy import deepcopy

    prior = claude_tool_use(
        "cons", "prior", "Bash", {"command": "echo unrelated"}, ts=ts(12, 9, 30)
    )
    records = consumer(prompt=f"repair {PATH}", before=[prior])
    changed = deepcopy(prior)
    changed["message"]["content"][0]["input"]["command"] = f"cat {PATH}"
    baseline, conflicted = _claude_conflict(tmp_path, records, changed, changed_first)
    assert conflicted.candidates and all(
        row.candidate.eligibility is Eligibility.KNOWN_REJECTED
        and "user_supplied" in row.candidate.eligibility_reasons
        for row in conflicted.candidates
    )
    assert _context_debits(conflicted) == _context_debits(baseline)
    assert all(row.credit.value == 0 and row.complete for row in _context_estimates(conflicted))


@pytest.mark.parametrize("changed_first", [False, True])
@pytest.mark.parametrize("position", ["before_launch", "after_launch", "conflicted_launch"])
def test_ancestor_context_conflict_respects_the_actual_child_launch_window(
    tmp_path: Path, changed_first: bool, position: str
) -> None:
    from copy import deepcopy

    root = tmp_path / "original"
    parent_path = root / ".omp" / "agent" / "sessions" / "-work" / "root.jsonl"
    header = omp_header("parent-ctx", ts(12, 8))
    prior = omp_call(
        "prior-entry",
        "prior",
        "bash",
        {"command": "echo unrelated"},
        ts(12, 9 if position == "before_launch" else 10, 0, 0),
    )
    launching = omp_launch("launch-entry", "launch", "ChildCtx", ts(12, 9, 10), ts(12, 9, 11))
    before, after = ([prior], []) if position == "before_launch" else ([], [prior])
    parent_records = [
        header,
        omp_user("parent-user", "repair deployment", ts(12, 8, 1)),
        *before,
        *launching,
        *after,
    ]
    child_records = [
        omp_header("child-ctx", ts(12, 9, 12), parent=parent_path),
        omp_user("child-user", "repair deployment", ts(12, 9, 12, 1)),
        *omp_search(
            "lookup-entry",
            "lookup",
            "deployment",
            results_json(card("claude:src1:ep:0", FACT)),
            (12, 11, 0, 0),
        ),
        omp_call("action-entry", "action", "bash", {"command": f"vim {PATH}"}, ts(12, 11, 1)),
        omp_result("action-result", "action", "ok", ts(12, 11, 1, 1)),
    ]
    original = claude_evidence(
        tmp_path / "source", {"source.jsonl": source_session("src1", FACT)}
    ) + omp_evidence(
        root,
        {"-work/root.jsonl": parent_records, "-work/root/ChildCtx.jsonl": child_records},
    )
    changed_prior = deepcopy(launching[0] if position == "conflicted_launch" else prior)
    if position == "conflicted_launch":
        changed_prior["message"]["content"][0]["arguments"]["tasks"][0]["name"] = "OtherChild"
    else:
        changed_prior["message"]["content"][0]["arguments"]["command"] = f"cat {PATH}"
    changed = omp_evidence(
        tmp_path / "changed", {"-work/parent-copy.jsonl": [header, changed_prior]}
    )
    baseline, conflicted = _reduce_conflict_rows(original, changed, changed_first)
    assert baseline.candidates and conflicted.candidates
    assert all(row.candidate.eligibility is Eligibility.SUPPORTED for row in baseline.candidates)
    expected = Eligibility.SUPPORTED if position == "after_launch" else Eligibility.UNAVAILABLE
    assert all(row.candidate.eligibility is expected for row in conflicted.candidates)
    expected_debits = _context_debits(baseline)
    actual_debits = _context_debits(conflicted)
    if position == "conflicted_launch":
        assert [debit.task for debit in expected_debits] == [
            TaskId("omp:parent-ctx", "turn:parent-user", "main")
        ]
        assert [debit.task for debit in actual_debits] == [TaskId("omp:parent-ctx", None, "main")]
        assert (
            tuple(replace(debit, task=expected_debits[0].task) for debit in actual_debits)
            == expected_debits
        )
    else:
        assert actual_debits == expected_debits
    if position == "after_launch":
        assert _context_estimates(conflicted) == _context_estimates(baseline)
    else:
        assert all(row.net.value is None for row in _context_estimates(conflicted))


"""reduce_io: read-only load of real stores + journal, then one deterministic reduction."""


_IO_PATH = "services/billing/limits.py"
_IO_FACT = f"deploy failed: ERR_QUOTA_EXCEEDED raised in {_IO_PATH}"


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(autouse=True)
def _fresh_state():
    journal.close_all()
    fingerprint.forget_keys()
    yield
    journal.close_all()
    fingerprint.forget_keys()


@pytest.fixture
def stores(tmp_path: Path) -> dict[str, str]:
    claude = tmp_path / ".claude"
    write_records(
        claude / "projects" / "-w" / "src1.jsonl",
        [
            claude_prompt("src1", "why does the deploy fail", uuid="u1", ts=ts(1, 10)),
            claude_tool_use(
                "src1", "w1", "Bash", {"command": "cat deploy.log"}, ts=ts(1, 10, 0, 5)
            ),
            claude_tool_result("src1", "w1", _IO_FACT, ts=ts(1, 10, 0, 6)),
        ],
    )
    write_records(
        claude / "projects" / "-w" / "cons.jsonl",
        [
            claude_prompt("cons", "fix the deploy", uuid="u1", ts=ts(12, 9)),
            claude_tool_use("cons", "t1", MCP, {"query": "why did the deploy fail"}, ts=ts(12, 10)),
            claude_tool_result(
                "cons",
                "t1",
                results_json(card("claude:src1:ep:0", _IO_FACT)),
                ts=ts(12, 10, 0, 2),
            ),
            claude_tool_use("cons", "t2", "Bash", {"command": f"vim {_IO_PATH}"}, ts=ts(12, 10, 1)),
            claude_tool_result("cons", "t2", "ok", ts=ts(12, 10, 1, 2)),
        ],
    )
    return {"HOME": str(tmp_path), "CLAUDE_CONFIG_DIR": str(claude)}


def test_reduce_stored_profile_reads_real_files_and_never_writes(
    stores: dict[str, str], tmp_path: Path
) -> None:
    before = tree_digest(tmp_path)
    reduction, loaded = reduce_stored_profile(PROFILE, stores, frequency=FrequencyView.unknown())
    assert tree_digest(tmp_path) == before  # no checkpoint, quarantine, or key was created
    assert loaded.sources_read["claude"] == 2
    assert loaded.events == []
    assert len(reduction.invocations) == 1
    assert reduction.invocations[0].cls.value == "transcript_only"  # no journal: no join, no guess
    assert reduction.executions == ()
    assert reduction.coverage["recorded_executions"] == 0
    # Retained transcripts never become an implicit rarity corpus.
    assert reduction.candidates
    assert reduction.coverage["exclusion.rarity_unknown"] >= 1


def test_journal_events_written_by_the_real_journal_join_the_transcript(
    stores: dict[str, str], tmp_path: Path
) -> None:
    from tests.accounting.ledger.reduce_builders import event

    key = fingerprint.load_key(PROFILE, create=True)
    text = "why did the deploy fail"
    for item in event("zz-1", ts(12, 10, 0, 1), text=text, cwd="/work/proj"):
        assert journal.append(PROFILE, item)
    # the reducer's fixtures key fingerprints with their own constant; re-key with the real key
    from dataclasses import replace

    from ssgrep.accounting.ledger.contracts import Operation

    real = fingerprint.request_fingerprint(key, Operation.SEARCH, query=text)
    journal.close_all()
    for item in event("zz-2", ts(12, 10, 0, 1), text=text, cwd="/work/proj"):
        assert journal.append(PROFILE, replace(item, fingerprint=real))
    reduction, loaded = reduce_stored_profile(PROFILE, stores)
    by_id = {e.execution_id: e for e in reduction.executions}
    assert loaded.journal_segments >= 1 and len(loaded.events) == 6
    assert by_id["zz-2"].attribution is Attribution.INFERRED_UNIQUE  # real key, real journal
    assert by_id["zz-1"].attribution is Attribution.UNATTRIBUTED  # foreign key never joins
    assert reduction.executions[0].source in ("both", "journal")


def test_missing_index_never_falls_back_to_retained_corpus_rarity(
    stores: dict[str, str],
) -> None:
    reduced, loaded = reduce_stored_profile(PROFILE, stores)
    assert loaded.evidence
    assert reduced.coverage["exclusion.rarity_unknown"] >= 1


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


"""Behavior of the frozen contracts: unknown-vs-zero and deterministic bytes."""


def test_unavailable_quantity_cannot_carry_a_value_or_hide_its_reason() -> None:
    with pytest.raises(ValueError):
        Quantity(0, "bytes", Basis.UNAVAILABLE, reason="no usage record")
    with pytest.raises(ValueError):
        Quantity(None, "bytes", Basis.OBSERVED)
    with pytest.raises(ValueError):
        Quantity(None, "bytes", Basis.UNAVAILABLE)

    missing = Quantity.unavailable("bytes", "usecli owns serialization")
    assert missing.value is None
    assert missing.reason == "usecli owns serialization"


def test_zero_observed_is_distinct_from_unavailable_in_canonical_form() -> None:
    zero = canonical_bytes(Quantity.observed(0, "bytes"))
    unknown = canonical_bytes(Quantity.unavailable("bytes", "n/a"))

    assert zero != unknown
    assert b'"value":0' in zero
    assert b'"value":null' in unknown


def test_canonical_bytes_ignore_construction_order_and_reject_floats() -> None:
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})
    assert canonical_bytes(build_event()) == canonical_bytes(build_event())
    with pytest.raises(TypeError):
        canonical_bytes({"score": 0.5})


def test_event_digest_changes_with_any_semantic_field() -> None:
    base = digest_of(build_event())

    assert digest_of(build_event(truncated=True)) != base
    assert digest_of(build_event(execution_id="n1-0002")) != base


def test_identical_native_events_share_identity_and_edited_ones_do_not() -> None:
    original = build_evidence()
    copy = build_evidence()
    edited = build_evidence(payload={"name": "bash", "arguments": {"command": "rm"}})

    assert original.identity == copy.identity
    assert original.identity != edited.identity


def test_task_without_turn_is_unresolved_not_a_guessed_task() -> None:
    assert not TaskId(root="r1", turn=None).resolved
    assert TaskId(root="r1", turn="t4").resolved


@pytest.mark.parametrize("bad", ["", "../etc", "A", "-x", "a" * 33, "a/b", "a b"])
def test_profile_ids_cannot_act_as_paths(bad: str) -> None:
    assert not valid_profile_id(bad)


def test_profile_id_accepts_bounded_identifiers() -> None:
    assert valid_profile_id("personal-local")
    assert valid_profile_id("acme_eu-1")


def test_tests_cannot_reach_the_real_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Positive evidence: with BOTH isolation layers removed (explicit override
    # and XDG redirect) the resolver points outside tmp_path, so the autouse
    # fixture is what protects the user's real data.
    assert data_dir().is_relative_to(tmp_path)

    monkeypatch.delenv("SSGREP_DATA_DIR")
    assert data_dir().is_relative_to(tmp_path), "XDG_DATA_HOME alone must still isolate"

    monkeypatch.delenv("XDG_DATA_HOME")
    assert not data_dir().is_relative_to(tmp_path)
    assert data_dir().is_relative_to(REAL_HOME)
