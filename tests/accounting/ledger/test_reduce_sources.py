"""6.3/6.4 + 5.5: frequency view, source atoms, drift, scope, cycles, determinism."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.capture import capture, capture_policy, fingerprint, journal, journal_reader
from ssgrep.accounting.collect.frequency import (
    BASIS_CANDIDATES,
    FrequencyView,
    build_view,
    eligible_scenarios,
    lexical_view,
    milli_log2,
)
from ssgrep.accounting.collect.replay_evidence import Token, extract
from ssgrep.accounting.ledger.contracts import (
    Attribution,
    Exposure,
    Operation,
    Outcome,
    Phase,
    RawEvidence,
    Reuse,
    Scenario,
    Transport,
    canonical_bytes,
)
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.ledger.reduce import methodology_for, reduce_profile
from ssgrep.accounting.ledger.reduce_cands import (
    CHARS as REDUCE_CHARS,
    Edge,
    find_cycles,
    validates,
)
from ssgrep.accounting.ledger.reduce_join import (
    Request,
    exposure_of,
    fold_events,
    interpret,
    reconcile,
    shell_shape,
)
from ssgrep.accounting.ledger.reduce_records import Reduction
from ssgrep.accounting.ledger.valuation import rollup, value_evidence
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus
from ssgrep.accounting.privacy import policy as privacy_policy
from ssgrep.accounting.readers.timeline import call_text, ts_ms
from ssgrep.services import api, mcp_server
from ssgrep.utilities.types import SearchResponse
from tests.accounting.ledger.reduce_builders import (
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
    omp_call,
    omp_evidence,
    omp_header,
    omp_result,
    omp_user,
    policy as policy_fixture,
    rare_view,
    results_json,
    source_session,
    ts,
)
from tests.accounting.ledger.test_reduce_evidence import (
    CODE,
    FACT,
    PATH,
    VIEW,
    consumer,
    reduce_scene,
)

PATH_TOKEN = Token("path", PATH)


# ------------------------------------------------------------ frequency (6.3)


def test_log_idf_fixed_point_matches_exact_math_and_is_integer() -> None:
    assert milli_log2(1024, 1) == 10_000
    assert milli_log2(8, 8) == 0
    assert milli_log2(3, 2) == 584
    assert isinstance(milli_log2(1001, 11), int)
    for numerator, denominator in ((0, 0), (2, 3), (-1, 1)):
        with pytest.raises(ValueError):
            milli_log2(numerator, denominator)


def test_thresholds_8_6_4_select_nested_scenarios_in_integer_milli_bits() -> None:
    def view(df: int) -> FrequencyView:
        return FrequencyView("v", BASIS_CANDIDATES, doc_count=1000, df={PATH_TOKEN.key: df})

    assert eligible_scenarios(view(1).rarity_milli(PATH_TOKEN)) == (
        Scenario.CONSERVATIVE,
        Scenario.CENTRAL,
        Scenario.OPTIMISTIC,
    )
    assert eligible_scenarios(view(10).rarity_milli(PATH_TOKEN)) == (
        Scenario.CENTRAL,
        Scenario.OPTIMISTIC,
    )
    assert eligible_scenarios(view(50).rarity_milli(PATH_TOKEN)) == (Scenario.OPTIMISTIC,)
    assert eligible_scenarios(view(300).rarity_milli(PATH_TOKEN)) == ()
    assert eligible_scenarios(None) == ()


def test_missing_frequency_evidence_is_unknown_not_maximal_rarity(tmp_path: Path) -> None:
    for name, view in (
        ("unknown", FrequencyView.unknown()),
        ("lexical", lexical_view({"deploy": 3}, 1000)),
        ("absent", build_view([f"filler {i}" for i in range(1000)])),
        (
            "nonpositive-df",
            FrequencyView("zero", BASIS_CANDIDATES, doc_count=1000, df={PATH_TOKEN.key: 0}),
        ),
    ):
        assert view.rarity_milli(PATH_TOKEN) is None, name
        reduction = reduce_scene(tmp_path / name, consumer(), view=view)
        assert reduction.candidates, name
        assert all(row.candidate.eligibility == "unavailable" for row in reduction.candidates), name
        valued = value_evidence(reduction.bundles[0], methodology_for(view))
        assert valued.leaves and all(s.missing for leaf in valued.leaves for s in leaf.scenarios), (
            name
        )
        assert reduction.coverage["exclusion.rarity_unknown"] >= 1, name


def test_view_is_normalized_like_extraction_and_content_addressed() -> None:
    first = build_view([FACT, "other"])
    second = build_view(["other", FACT])
    assert first.version_id == second.version_id
    assert first.df[PATH_TOKEN.key] == 1
    assert build_view([FACT, "x", "y"]).version_id != first.version_id
    assert methodology_for(first).calibrated is False
    assert dict(methodology_for(first).thresholds) == {
        "conservative": 8,
        "central": 6,
        "optimistic": 4,
    }


def test_rarity_changes_known_credited_cost_by_scenario(tmp_path: Path) -> None:
    def credits(df: int) -> tuple[int, int, int]:
        view = FrequencyView(
            "v", BASIS_CANDIDATES, doc_count=1000, df={PATH_TOKEN.key: df, f"error_code\0{CODE}": 1}
        )
        reduction = reduce_scene(tmp_path / f"df{df}", consumer(), view=view)
        valued = value_evidence(reduction.bundles[0], methodology_for(view))
        estimates = rollup(
            (leaf for leaf in valued.leaves if leaf.unit == REDUCE_CHARS), REDUCE_CHARS
        )
        values = tuple(estimate.credit.value for estimate in estimates)
        assert len(values) == 3
        first, second, third = values
        assert first is not None and second is not None and third is not None
        return first, second, third

    top, mid, low = credits(1), credits(10), credits(50)
    assert all(value > 0 for value in top)
    assert mid[0] == 0 and mid[1] > 0 and mid[2] > 0
    assert low[:2] == (0, 0) and low[2] > 0


# ---------------------------------------------------------------- sources (6.4)


def atom_digests(evidence: list[RawEvidence]) -> dict[str, str]:
    return {
        f"{e.runtime}:{e.session_id}:{e.record_id}": e.digest
        for e in evidence
        if e.session_id == "src1"
    }


def test_atoms_carry_origin_event_identity_and_content_digest(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer())
    (record,) = reduction.candidates
    atoms = record.candidate.immediate
    origins = {a.atom.origin_event for a in atoms}
    assert origins == {"claude:src1:call:w1", "claude:src1:result:w1"}
    assert all(a.atom.digest and len(a.atom.digest) == 64 for a in atoms)
    assert all(a.atom_id == a.atom.origin_event for a in atoms)  # never an episode ordinal


def test_digest_drift_gives_no_credit_and_records_the_reason(tmp_path: Path) -> None:
    clean = reduce_scene(tmp_path / "a", consumer())
    evidence = claude_evidence(
        tmp_path / "probe", {"-work/src1.jsonl": source_session("src1", FACT)}
    )
    known = atom_digests(evidence)
    same = reduce_scene(tmp_path / "b", consumer(), known_atoms=known)
    assert same.candidates[0].candidate.immediate == clean.candidates[0].candidate.immediate
    drifted = {k: "0" * 64 for k in known}
    reduction = reduce_scene(tmp_path / "c", consumer(), known_atoms=drifted)
    (record,) = reduction.candidates
    atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
    assert atoms and all(a.status is AtomStatus.DIGEST_CONFLICT for a in atoms)
    assert {e.reason for e in reduction.exclusions} == {"source_digest_drift"}
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    assert {a.credit_raw for a in valuation.allocations} == {0}
    assert all(a.excluded for a in valuation.allocations if a.scenario is Scenario.CONSERVATIVE)


def test_consumer_own_later_work_can_never_be_its_source(tmp_path: Path) -> None:
    own = claude_evidence(tmp_path / "x", {"-work/cons.jsonl": consumer()})
    assert own
    body = results_json(card("claude:cons:ep:0", FACT))
    reduction = reduce_scene(tmp_path, consumer(body=body), src=source_session("zzz", "nothing"))
    (record,) = reduction.candidates
    atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
    assert atoms and all(a.status is AtomStatus.SELF_LATER_WORK for a in atoms)
    assert "consumer_own_later_work" in {e.reason for e in reduction.exclusions}
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    assert {a.credit_raw for a in valuation.allocations} == {0}


def test_unresolvable_ref_is_missing_coverage_not_zero_cost_proof(tmp_path: Path) -> None:
    body = results_json(card("claude:ghost:ep:0", FACT))
    reduction = reduce_scene(tmp_path, consumer(body=body))
    (record,) = reduction.candidates
    (atom,) = record.candidate.immediate
    assert atom.status is AtomStatus.UNAVAILABLE
    assert "source_unresolved" in {e.reason for e in reduction.exclusions}
    assert reduction.coverage["unavailable_sources"] == 1
    assert reduction.coverage["source_costs_available"] == 0


def test_malformed_card_locator_is_unresolved_despite_visible_action(tmp_path: Path) -> None:
    reduced = reduce_scene(tmp_path, consumer(body=results_json(card("not-a-session-ref", FACT))))
    assert reduced.candidates
    assert {row.reason for row in reduced.exclusions} == {"source_unresolved"}
    assert not any(atom.usable for atom in reduced.candidates[0].candidate.immediate)


def test_source_headers_and_empty_prompt_neither_become_atoms_nor_prior_knowledge(
    tmp_path: Path,
) -> None:
    source = [
        omp_header("source", ts(1)),
        omp_user("prompt", "", ts(1, s=1)),
        omp_call("status", "status", "status", {}, ts(1, s=1)),
        omp_call("call", "witness", "bash", {"command": "cat deploy.log"}, ts(1, s=2)),
        omp_result("result", "witness", FACT, ts(1, s=3)),
    ]
    evidence = omp_evidence(tmp_path, {"project/source.jsonl": source})
    evidence += claude_evidence(
        tmp_path,
        {"project/consumer.jsonl": consumer(body=results_json(card("omp:source:ep:0", FACT)))},
    )
    reduced = reduce_profile(
        PROFILE, evidence, [], policy=policy_fixture(), frequency=VIEW, fingerprint_key=KEY
    )
    (record,) = reduced.candidates
    atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
    assert atoms and all(atom.usable for atom in atoms)
    witnesses = [
        item
        for item in evidence
        if item.runtime == "omp"
        and item.session_id == "source"
        and item.kind in {"tool_call", "tool_result"}
        and item.payload.get("call_id") == "witness"
    ]
    assert {item.kind for item in witnesses} == {"tool_call", "tool_result"}
    assert {atom.atom.origin_event for atom in atoms} == {
        f"{item.runtime}:{item.session_id}:{item.record_id}" for item in witnesses
    }
    assert reduced.exclusions == ()


# -------------------------------------------------------------- scope (5.5)

ORG = "acme"
ORG_CWD = "/work/orgrepo/app"


def org_scene(root: Path, source_cwd: str, *, copy: bool = False) -> Reduction:
    cons = [
        claude_prompt("cons", "fix the deploy", uuid="u1", ts=ts(12, 9)),
        claude_tool_use(
            "cons", "t1", MCP, {"query": "why did the deploy fail"}, ts=ts(12, 10), cwd=ORG_CWD
        ),
        claude_tool_result(
            "cons", "t1", results_json(card("claude:src1:ep:0", FACT)), ts=ts(12, 10, 0, 2)
        ),
        claude_tool_use(
            "cons", "t2", "Bash", {"command": f"vim {PATH}"}, ts=ts(12, 10, 1), cwd=ORG_CWD
        ),
        claude_tool_result("cons", "t2", "ok", ts=ts(12, 10, 1, 2)),
    ]
    files = {
        "-work/src1.jsonl": source_session("src1", FACT, cwd=source_cwd),
        "-work/cons.jsonl": cons,
    }
    if copy:  # a personal-local copy of the very same transcript elsewhere
        files["-home-copy/src1.jsonl"] = source_session("src1", FACT, cwd=source_cwd)
    for path, records in files.items():
        owner = source_cwd if "src1" in path else ORG_CWD
        files[path] = [{**record, "cwd": owner} for record in records]
    events = event(
        "n-1", ts(12, 10, 0, 1), text="why did the deploy fail", cwd=ORG_CWD, profile=ORG
    )
    return reduce_profile(
        ORG,
        claude_evidence(root, files),
        events,
        policy=policy_fixture("/work/orgrepo", profile=ORG),
        frequency=VIEW,
        fingerprint_key=KEY,
    )


def test_out_of_scope_source_cost_is_excluded_with_missing_coverage_consumer_stays_measured(
    tmp_path: Path,
) -> None:
    inside = org_scene(tmp_path / "in", "/work/orgrepo/other")
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside = org_scene(tmp_path / "out", str(outside_root.resolve()))
    (rec_in,) = inside.candidates
    assert rec_in.candidate.immediate and all(a.usable for a in rec_in.candidate.immediate)
    (rec_out,) = outside.candidates
    atoms = rec_out.candidate.immediate + rec_out.candidate.discovery + rec_out.candidate.episode
    assert atoms and all(a.status is AtomStatus.OUT_OF_SCOPE for a in atoms)
    assert "source_out_of_scope" in {e.reason for e in outside.exclusions}
    # consumer usage in scope remains measured (an execution and a debit exist) ...
    assert len(outside.executions) == 1 and outside.executions[0].invocation_id
    assert outside.bundles[0].debits
    # ... but the organization valuation credits nothing and reports the gap
    valuation = value_evidence(outside.bundles[0], methodology_for(VIEW))
    assert {a.credit_raw for a in valuation.allocations} == {0}
    assert any(a.missing for a in valuation.allocations)


def test_personal_local_copy_cannot_bypass_the_organization_scope(tmp_path: Path) -> None:
    outside = org_scene(tmp_path, "/home/me/personal", copy=True)
    (record,) = outside.candidates
    atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
    assert all(a.status is AtomStatus.OUT_OF_SCOPE for a in atoms)


# ------------------------------------------------------------- cycles (6.4)


def test_provenance_cycles_are_detected_transitively() -> None:
    a_from_b = Edge("claude:a", ("claude:b",), witness_ts=2_000, exposure_ts=5_000)
    b_from_a = Edge("claude:b", ("claude:a",), witness_ts=500, exposure_ts=1_500)
    assert find_cycles([a_from_b, b_from_a]) == {0}
    chain = [
        Edge("claude:a", ("claude:b",), 3_000, 9_000),
        Edge("claude:b", ("claude:c",), 2_000, 2_500),
        Edge("claude:c", ("claude:a",), 100, 1_000),
    ]
    assert find_cycles(chain) == {0}
    acyclic = [
        Edge("claude:a", ("claude:b",), 1_000, 5_000),
        Edge("claude:b", ("claude:c",), 100, 500),
    ]
    assert find_cycles(acyclic) == set()
    # B's absorption of A must precede the witness reused by A, not merely exist.
    target = Edge("a", ("b",), 20, 30)
    assert find_cycles([target, Edge("b", ("a",), 5, None)]) == set()
    assert find_cycles([target, Edge("b", ("a",), 5, 10)]) == {0}
    # A self-reference is not evidence of a cross-root cycle; recursive graphs terminate.
    assert find_cycles([target, Edge("b", ("b",), 5, 10)]) == set()
    cyclic = [Edge("a", ("b",), 20, 1), Edge("b", ("c",), 20, 1), Edge("c", ("b",), 20, 1)]
    assert find_cycles(cyclic) == {1, 2}


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
        policy=policy_fixture(),
        frequency=rare_view(first, second),
        fingerprint_key=KEY,
    )
    by_root = {row.candidate.task.root: row.candidate for row in reduced.candidates}
    assert set(by_root) == {"claude:a", "claude:b"}
    assert all(atom.status is AtomStatus.OK for atom in by_root["claude:a"].immediate)
    assert all(atom.status is AtomStatus.CYCLE for atom in by_root["claude:b"].immediate)
    assert reduced.coverage["exclusion.provenance_cycle"] == 1
    assert by_root["claude:b"].reuse is Reuse.ACTION_REFERENCE


# --------------------------------------------------------- determinism (all)


def shuffled(items: list, seed: int) -> list:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def test_reduction_digest_is_independent_of_input_order(tmp_path: Path) -> None:
    files = {
        "-work/src1.jsonl": source_session("src1", FACT),
        "-work/cons.jsonl": consumer(
            extra=[
                claude_tool_use("cons", "t5", MCP, {"query": "dup"}, ts=ts(12, 10, 5)),
                claude_tool_result("cons", "t5", results_json(), ts=ts(12, 10, 5, 2)),
                claude_tool_use("cons", "t6", MCP, {"query": "dup"}, ts=ts(12, 10, 5, 1)),
                claude_tool_result("cons", "t6", results_json(), ts=ts(12, 10, 5, 3)),
            ]
        ),
        "-copy/cons.jsonl": consumer(),
    }
    evidence = claude_evidence(tmp_path, files)
    # the same native record ids with changed content (another store): a conflict that the
    # reducer must resolve identically in any arrival order, never double usage
    evidence += claude_evidence(
        tmp_path / "fork", {"-fork/src1.jsonl": source_session("src1", FACT + " (edited later)")}
    )
    events = [
        *event("n-1", ts(12, 10, 0, 1), text="why did the deploy fail"),
        *event("n-2", ts(12, 10, 5, 2), text="dup"),
        *event("n-3", ts(12, 10, 5, 3), text="dup"),
        *event("n-4", ts(12, 14), text="orphan"),
    ]
    digests = set()
    for seed in range(8):
        reduction = reduce_profile(
            PROFILE,
            shuffled(evidence, seed),
            shuffled(events, seed + 100),
            policy=policy_fixture(),
            frequency=VIEW,
            fingerprint_key=KEY,
        )
        digests.add(reduction.digest)
    assert len(digests) == 1
    assert reduction.coverage["evidence_conflicts"] >= 1
    assert reduction.candidates and reduction.coverage["ambiguous"] == 2
    # a real change in the evidence changes the digest (the digest is not constant)
    changed = reduce_profile(
        PROFILE, evidence, events[:-3], policy=policy_fixture(), frequency=VIEW, fingerprint_key=KEY
    )
    assert changed.digest not in digests
    assert CWD and rare_view  # builders stay importable for sibling suites


"""Missing provenance never becomes free work or a fact shared across aliases."""


@pytest.mark.parametrize(
    "source, reason",
    [
        ([claude_prompt("src1", FACT, ts=ts(1))], "source_not_observed"),
        (
            [
                claude_prompt("src1", "investigate", ts=ts(1)),
                claude_tool_use(
                    "src1", "w1", "Bash", {"command": f"cat {PATH}"}, ts=ts(1, 10, 0, 5)
                ),
                claude_tool_result("src1", "w1", FACT, ts=ts(1, 10, 0, 6)),
            ],
            "source_known_before_witness",
        ),
        (source_session("src1", "no retained witness"), "witness_not_found"),
        (source_session("src1", FACT, when_day=13), "source_postdates_exposure"),
    ],
)
def test_unproven_source_retains_known_consumer_debit_and_unknown_credit(
    tmp_path: Path, source: list[dict], reason: str
) -> None:
    reduction = reduce_scene(tmp_path, consumer(), src=source)
    (record,) = reduction.candidates
    assert reason in {row.reason for row in reduction.exclusions}
    assert record.candidate.immediate
    assert all(atom.status is AtomStatus.UNAVAILABLE for atom in record.candidate.immediate)
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    estimates = rollup(
        (leaf for leaf in valuation.leaves if leaf.unit == REDUCE_CHARS), REDUCE_CHARS
    )
    assert all(estimate.credit.value == 0 for estimate in estimates)
    assert all(estimate.debit.value and estimate.debit.value > 0 for estimate in estimates)
    assert all(estimate.net.value is None for estimate in estimates)
    assert all(estimate.known_net < 0 for estimate in estimates)


def test_legacy_unqualified_ref_resolves_retained_claude_source(tmp_path: Path) -> None:
    body = results_json(card("src1:ep:0", FACT))
    reduction = reduce_scene(tmp_path, consumer(body=body))
    (record,) = reduction.candidates
    assert {atom.atom.origin_event for atom in record.candidate.immediate} == {
        "claude:src1:call:w1",
        "claude:src1:result:w1",
    }
    assert all(atom.usable for atom in record.candidate.immediate)


def test_same_basename_in_another_directory_is_not_a_reference(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(action="vim services/legacy/limits.py"))
    assert reduction.candidates == ()
    assert reduction.coverage["candidates_unreferenced"] == 2
    assert reduction.bundles[0].debits


def test_absolute_action_path_matches_whole_relative_source_path(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(action=f"vim /work/proj/{PATH}"))
    (record,) = reduction.candidates
    assert record.candidate.action_outcome_ok
    assert record.candidate.immediate
    assert all(atom.usable for atom in record.candidate.immediate)


def test_home_identity_in_visible_response_is_charged_but_never_persisted_as_candidate(
    tmp_path: Path,
) -> None:
    secret_path = "/Users/private_operator/services/internal/credentials.py"
    body = results_json(card("claude:src1:ep:0", secret_path))
    reduction = reduce_scene(
        tmp_path,
        consumer(body=body, action=f"cat {secret_path}"),
        src=source_session("src1", secret_path),
    )
    assert reduction.candidates == ()
    assert reduction.coverage["exclusion.home_path"] == 1
    assert reduction.bundles[0].debits[0].chars == len(body)
    persisted = canonical_bytes(reduction)
    assert b"private_operator" not in persisted
    assert b"credentials.py" not in persisted


"""5.4/5.5: result interpretation, exposure, shell mixing, per-event scope."""


INDEX_ERROR = json.dumps({"error": "The global index is incomplete; run `ssgrep index`."})


def reduce_one(root: Path, records: list[dict], events: list, **kw) -> Reduction:
    evidence = claude_evidence(root, {"-work/s1.jsonl": records})
    return reduce_profile(
        PROFILE,
        evidence,
        events,
        policy=kw.pop("policy", policy_fixture()),
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


@pytest.mark.parametrize("separator", ["\n\n", "\n\n\n"])
@pytest.mark.parametrize("duration", ["0", "0.12", "12.500"])
def test_native_timing_footer_preserves_decoded_cards_and_full_visible_cost(duration, separator):
    excerpt = "rule_id minutes\nMCT-DEN-045 45"
    raw = results_json(card("omp:source:ep:0", excerpt))
    visible = raw + f"{separator}Wall time: {duration} seconds\n"
    result = interpret({"output": visible, "is_error": False})
    assert result.outcome is Outcome.OK and result.structured
    assert len(result.cards) == 1
    assert result.cards[0].ref == "omp:source:ep:0"
    assert excerpt in result.cards[0].text
    assert result.text == visible and result.chars == len(visible)


@pytest.mark.parametrize(
    "suffix",
    [
        "\n\narbitrary diagnostic",
        "\n\nWall time: 0.12 seconds\nanother payload",
        '\n{"results": []}',
        "\n\nWall time: -1 seconds",
        "\n\nWall time: 1e2 seconds",
        "\n\nWall time: 0.12 seconds\n[output truncated]",
    ],
)
def test_unknown_trailing_output_cannot_promote_a_structured_result(suffix):
    visible = results_json(card("omp:source:ep:0", "MCT-DEN-045")) + suffix
    result = interpret({"output": visible, "is_error": False})
    assert result.outcome is Outcome.UNKNOWN
    assert not result.structured and result.cards == ()
    assert result.text == visible and result.chars == len(visible)


@pytest.mark.parametrize("override", [{"is_error": True}, {"exit_code": 2}])
def test_native_timing_footer_does_not_hide_failed_result(override):
    visible = results_json(card("omp:source:ep:0", "MCT-DEN-045")) + "\n\nWall time: 0.12 seconds"
    result = interpret({"output": visible, **override})
    assert result.outcome is Outcome.ERROR
    assert result.text == visible and result.chars == len(visible)


def test_native_timing_footer_does_not_hide_a_structured_error_payload():
    visible = INDEX_ERROR + "\n\nWall time: 0.12 seconds"
    result = interpret({"output": visible, "is_error": False})
    assert result.outcome is Outcome.ERROR and result.basis == "error_payload"
    assert result.text == visible and result.chars == len(visible)


def test_deep_tool_output_remains_visible_without_aborting_interpretation():
    visible = "[" * 2000 + "0" + "]" * 2000
    result = interpret({"output": visible, "is_error": False})
    assert result.outcome is Outcome.UNKNOWN and result.cards == ()
    assert result.text == visible and result.chars == len(visible)


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
    assert shell_shape(r'ssgrep search "say \"hello\"" | head -n 2').kind == "filtered"
    shape = shell_shape("if true; then echo unrelated; else ssgrep search repair; fi")
    assert shape.kind == "mixed"
    assert exposure_of(shape.kind, interpret({"output": "visible_fact"})) is Exposure.PARTIAL


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
    org = policy_fixture("/work/orgrepo", profile="acme")
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
        policy=policy_fixture(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
        verifier_sessions=frozenset({"claude:s1"}),
    )
    (execution,) = reduction.executions
    assert execution.actor == "verifier"
    assert reduction.observations["searches"].value == 1  # still a real global execution
    assert reduction.bundles == ()  # but neither debit nor credit for any solver
    assert reduction.candidates == ()


def test_cli_success_envelope_extracts_show_files_not_hidden_envelope_fields() -> None:
    output = json.dumps(
        {
            "ok": True,
            "metadata": {"secret": "hidden_fact"},
            "data": {
                "episode_id": "claude:source:ep:0",
                "response_text": "useful_fact",
                "files_touched": ["src/repair.py", 42],
            },
        }
    )
    result = interpret({"output": output})
    assert result.outcome is Outcome.OK and result.explicit_success
    extracted = extract(result)
    assert {item.token for item in extracted.exposed} == {
        Token("identifier", "useful_fact"),
        Token("path", "src/repair.py"),
    }
    assert {item.refs for item in extracted.exposed} == {("claude:source:ep:0",)}


@pytest.mark.parametrize(
    "body, expected",
    [
        ("{broken json", Outcome.UNKNOWN),
        ('{"ok":false,"message":"incomplete index"}', Outcome.ERROR),
        ('{"error":"Episode not found: gone:ep:0"}', Outcome.EMPTY),
    ],
)
def test_result_semantics_override_transport_success(body: str, expected: Outcome) -> None:
    result = interpret({"output": body, "is_error": False})
    assert result.outcome is expected
    assert result.chars == len(body)
    assert exposure_of("pure", result) is Exposure.OBSERVED
    assert not result.explicit_success


def test_verification_requires_verifier_and_nonerror_result() -> None:
    success = interpret({"output": "8 passed", "exit_code": 0})
    failure = interpret({"output": "8 passed", "is_error": True})
    assert validates("pytest tests/", success)
    assert not validates("cat tests/report", success)
    assert not validates("pytest tests/", failure)


def test_call_text_bounds_nested_arguments_and_accepts_native_string_arguments() -> None:
    nested: object = "hidden_fact"
    for _ in range(10):
        nested = [nested]
    assert call_text({"arguments": {"visible": "known_fact", "nested": nested}}) == "known_fact"
    assert call_text({"arguments": "ssgrep search repair"}) == "ssgrep search repair"
    assert ts_ms("2026-10-12T10:00:00") is None
    assert ts_ms("2026-10-12T10:00:00Z") == ts_ms("2026-10-12T12:00:00+02:00")


"""Safe attribution uses local invocation authority, not client arguments."""


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)
    for name in ("SSGREP_MCP_CALLER_WORKSPACE", "SSGREP_CAPTURE_PROFILE", capture.CORRELATION_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    journal.close_all()


def request(fact, index=0, **overrides):
    base = Request(
        invocation_id=f"native-{index}",
        index=0,
        operation=fact.operation,
        transport=fact.transport,
        fingerprint=fact.fingerprint,
        cwd=fact.cwd,
        correlation=None,
        window=(fact.ts_ms - 1000, fact.ts_ms + 1000),
        capacity=1,
        result_refs=None,
        result_outcome=None,
        fingerprint_key_id=fact.fingerprint_key_id,
        cwd_trusted=True,
    )
    return replace(base, **overrides)


def captured(query="query", transport=Transport.CLI):
    capture.arm(transport)
    execution = capture.begin(Operation.SEARCH, text=query)
    assert execution is not None
    capture.complete(execution, Outcome.EMPTY)
    return execution


def facts(profile="personal-local"):
    return fold_events(journal_reader.read_journal(profile, quarantine=False).events)


def test_seven_unique_cli_mcp_controls_join_once_including_copied_evidence(monkeypatch):
    privacy_policy.enable_collection("personal-local")
    monkeypatch.setenv("SSGREP_MCP_CALLER_WORKSPACE", str(Path.cwd()))
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    for index in range(7):
        if index % 2:
            assert mcp_server.search_sessions(f"unique query {index}")["total_matches"] == 0
        else:
            captured(f"unique query {index}")
    events = journal_reader.read_journal("personal-local", quarantine=False).events
    folded = fold_events(events + events)
    requests = [request(fact, index) for index, fact in enumerate(folded)]
    links = reconcile(folded, requests + requests)
    assert len(links) == 7
    assert {link.attribution for link in links.values()} == {Attribution.INFERRED_UNIQUE}
    assert len({link.requests[0] for link in links.values()}) == 7


def test_parallel_equal_queries_never_choose_first():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()

    link = reconcile([fact], [request(fact), request(fact, 1)])[fact.execution_id]
    assert link.attribution is Attribution.AMBIGUOUS
    assert "equal_query_concurrency" in link.reasons


def test_rotated_key_is_observed_without_process_restart(tmp_path):
    key = fingerprint.load_key("personal-local", create=True)
    assert fingerprint.load_key("personal-local", create=False) == key
    path = tmp_path / "data" / "accounting" / "personal-local" / fingerprint.KEY_FILE
    replacement = b"r" * fingerprint.KEY_BYTES
    path.write_bytes(replacement)
    assert fingerprint.load_key("personal-local", create=False) == replacement
    assert fingerprint.key_id(key) != fingerprint.key_id(replacement)


def test_capacity_collision_has_explicit_reason():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    other = replace(fact, execution_id="second")
    links = reconcile([fact, other], [request(fact)])
    assert all(link.attribution is Attribution.AMBIGUOUS for link in links.values())
    assert all("capacity_ambiguity" in link.reasons for link in links.values())


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"fingerprint_key_id": "rotated-key"}, "key_mismatch"),
        ({"fingerprint": "0" * 64}, "query_mismatch"),
        ({"window": None}, "clock_uncertainty"),
        ({"window": (2000, 1000)}, "clock_uncertainty"),
        ({"cwd_trusted": False}, "untrusted_cwd"),
        ({"cwd": "../outside", "cwd_trusted": True}, "untrusted_cwd"),
        ({"cwd": "/tmp/a\0b", "cwd_trusted": True}, "untrusted_cwd"),
    ],
)
def test_unsafe_join_reports_rejection(override, reason):
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    link = reconcile([fact], [request(fact, **override)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert reason in link.reasons


def test_unknown_execution_clock_cannot_infer():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    link = reconcile([replace(fact, timing_known=False)], [request(fact)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "clock_uncertainty" in link.reasons


def test_mcp_server_cwd_is_not_caller_scope_and_client_cwd_cannot_authorize(monkeypatch):
    privacy_policy.enable_collection("personal-local")
    capture.arm(Transport.MCP)
    execution = capture.begin(Operation.SEARCH, text="q", cwd="/client-supplied")
    assert execution is not None
    (event,) = journal_reader.read_journal("personal-local", quarantine=False).events
    assert event.cwd is None and event.scope_basis == "unknown_scope"
    (fact,) = fold_events([event])
    link = reconcile([fact], [request(fact, cwd=None, cwd_trusted=False)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "unknown_scope" in link.reasons


def test_unknown_execution_scope_cannot_be_rehabilitated_by_matching_request_cwd():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    unknown = replace(fact, cwd="/workspace", scope_basis="unknown_scope")

    link = reconcile([unknown], [request(unknown, cwd="/workspace", cwd_trusted=True)])[
        unknown.execution_id
    ]

    assert link.attribution is Attribution.UNATTRIBUTED
    assert "untrusted_cwd" in link.reasons


def test_trusted_mcp_launch_context_captures_only_workspace_owner(tmp_path, monkeypatch):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    for profile, root in (("org-a", root_a), ("org-b", root_b)):
        privacy_policy.save_profile(
            ProfilePolicy(profile_id=profile, roots=(str(root),), collection_enabled=True)
        )
    monkeypatch.chdir(root_b)
    monkeypatch.setenv("SSGREP_MCP_CALLER_WORKSPACE", str(root_a))
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    assert mcp_server.search_sessions("positive control")["total_matches"] == 0
    events = journal_reader.read_journal("org-a", quarantine=False).events
    assert len(events) == 3
    assert all(
        event.cwd == str(root_a) and event.scope_basis == "trusted_invocation" for event in events
    )
    assert journal_reader.read_journal("org-b", quarantine=False).events == []


def test_unknown_scope_selected_profile_never_broadcasts_or_fabricates_scope(tmp_path, monkeypatch):
    for profile in ("org-a", "org-b"):
        root = tmp_path / profile
        root.mkdir()
        privacy_policy.save_profile(
            ProfilePolicy(profile_id=profile, roots=(str(root),), collection_enabled=True)
        )
    monkeypatch.setenv("SSGREP_CAPTURE_PROFILE", "org-a")
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    assert mcp_server.search_sessions("unknown caller")["total_matches"] == 0
    events = journal_reader.read_journal("org-a", quarantine=False).events
    assert len(events) == 3 and all(event.cwd is None for event in events)
    assert all(event.scope_basis == "unknown_scope" for event in events)
    assert journal_reader.read_journal("org-b", quarantine=False).events == []


def test_profile_selector_cannot_override_known_workspace_owner(tmp_path, monkeypatch):
    for profile in ("org-a", "org-b"):
        root = tmp_path / profile
        root.mkdir()
        privacy_policy.save_profile(
            ProfilePolicy(profile_id=profile, roots=(str(root),), collection_enabled=True)
        )
    monkeypatch.setenv("SSGREP_MCP_CALLER_WORKSPACE", str(tmp_path / "org-a"))
    monkeypatch.setenv("SSGREP_CAPTURE_PROFILE", "org-b")
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    assert mcp_server.search_sessions("wrong tenant")["total_matches"] == 0
    assert journal_reader.read_journal("org-a", quarantine=False).events == []
    assert journal_reader.read_journal("org-b", quarantine=False).events == []


def test_native_seven_lookup_controls_have_independent_expected_counts(tmp_path):
    from tests.accounting.ledger.reduce_builders import (
        CWD,
        KEY,
        MCP,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        event,
        results_json,
        ts,
    )
    from tests.accounting.ledger.test_reduce_reconcile import run

    records = [claude_prompt("s1", "seven controls", ts=ts(12, 9))]
    events = []
    expected_ids = {f"control-{index}" for index in range(7)}
    for index in range(7):
        query = f"independent query {index}"
        transport = "mcp" if index % 2 else "cli"
        name = MCP if transport == "mcp" else "Bash"
        arguments = (
            {"query": query} if transport == "mcp" else {"command": f"ssgrep search '{query}'"}
        )
        records.extend(
            [
                claude_tool_use("s1", f"tool-{index}", name, arguments, ts=ts(12, 10), cwd=CWD),
                claude_tool_result("s1", f"tool-{index}", results_json(), ts=ts(12, 10, 0, 3)),
            ]
        )
        events.extend(
            replace(
                row,
                fingerprint_key_id=fingerprint.key_id(KEY),
                scope_basis="unknown_scope" if transport == "mcp" else "process_cwd",
                cwd=None if transport == "mcp" else CWD,
            )
            for row in event(f"control-{index}", ts(12, 10, 0, 1), text=query, transport=transport)
        )
    reduction = run(tmp_path, records, events + events)
    assert {row.execution_id for row in reduction.executions} == expected_ids
    assert len(reduction.invocations) == 7
    assert all(row.attribution is Attribution.INFERRED_UNIQUE for row in reduction.executions)
    assert {
        identifier for row in reduction.invocations for identifier in row.execution_ids
    } == expected_ids
    assert all(len(row.execution_ids) == 1 for row in reduction.invocations)


def test_conflicting_request_copies_never_pick_supported_variant():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    original = request(fact)
    conflicting = replace(original, fingerprint="0" * 64)
    link = reconcile([fact], [original, conflicting])[fact.execution_id]
    assert link.attribution is Attribution.AMBIGUOUS
    assert link.reasons == ("request_conflict",)


def test_conflicting_journal_phases_never_gain_unique_attribution():
    privacy_policy.enable_collection("personal-local")
    captured()
    events = journal_reader.read_journal("personal-local", quarantine=False).events
    started = next(row for row in events if row.phase is Phase.STARTED)
    folded = fold_events(events + [replace(started, fingerprint="0" * 64)])
    (fact,) = folded
    link = reconcile(folded, [request(fact)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "journal_conflict" in link.reasons


@pytest.mark.parametrize("client_argument", [False, True])
def test_client_cwd_argument_does_not_replace_native_workspace(tmp_path, client_argument):
    from tests.accounting.ledger.reduce_builders import (
        CWD,
        MCP,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        event,
        results_json,
        ts,
    )
    from tests.accounting.ledger.test_reduce_reconcile import run

    arguments = {"query": "deploy failure"}
    if client_argument:
        arguments["cwd"] = "/forged-org"
    records = [
        claude_prompt("s1", "investigate", ts=ts(12, 9)),
        claude_tool_use("s1", "tool", MCP, arguments, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "tool", results_json(), ts=ts(12, 10, 0, 3)),
    ]
    events = [
        replace(row, cwd=None, scope_basis="unknown_scope")
        for row in event("one", ts(12, 10, 0, 1))
    ]
    reduction = run(tmp_path, records, events)
    (row,) = reduction.executions
    if client_argument:
        # cwd is not part of the actual MCP tool schema. Such a transcript
        # cannot manufacture a valid lookup, let alone organization authority.
        assert reduction.invocations == ()
        assert reduction.coverage["inv_unclassified"] == 1
        assert row.attribution is Attribution.UNATTRIBUTED
        assert row.cwd is None
        assert "unknown_scope" in row.join_rejection_reasons
    else:
        assert row.attribution is Attribution.INFERRED_UNIQUE
        assert row.cwd == CWD


@pytest.mark.parametrize("authorized", [True, False])
def test_unknown_org_journal_requires_same_profile_native_workspace(tmp_path, authorized):
    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.ledger.reduce import reduce_profile
    from ssgrep.accounting.privacy.policy import PolicyFile
    from tests.accounting.ledger.reduce_builders import (
        KEY,
        MCP,
        claude_evidence,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        event,
        results_json,
        ts,
    )

    root_a, root_b = str(tmp_path / "org-a"), str(tmp_path / "org-b")
    profiles = {
        "org-a": ProfilePolicy(profile_id="org-a", roots=(root_a,), collection_enabled=True),
        "org-b": ProfilePolicy(profile_id="org-b", roots=(root_b,), collection_enabled=True),
    }
    native_cwd = root_a if authorized else root_b
    records = [
        claude_prompt("s1", "org controls", ts=ts(12, 9)),
        claude_tool_use(
            "s1", "tool", MCP, {"query": "deploy failure"}, ts=ts(12, 10), cwd=native_cwd
        ),
        claude_tool_result("s1", "tool", results_json(), ts=ts(12, 10, 0, 3)),
    ]
    evidence = claude_evidence(tmp_path, {"-work/s1.jsonl": records})
    events = [
        replace(
            row, cwd=None, scope_basis="unknown_scope", fingerprint_key_id=fingerprint.key_id(KEY)
        )
        for row in event("one", ts(12, 10, 0, 1), profile="org-a")
    ]
    reduction = reduce_profile(
        "org-a",
        evidence,
        events,
        policy=profiles["org-a"],
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
        ownership=PolicyFile(profiles=profiles),
    )
    if authorized:
        (row,) = reduction.executions
        assert row.in_scope and row.cwd == root_a
        assert row.attribution is Attribution.INFERRED_UNIQUE
        assert row.scope_basis == "native_call_metadata"
        from ssgrep.accounting.views.filter import Scoper, effective_scoped_journal

        scoped = effective_scoped_journal(
            events,
            reduction.executions,
            Scoper(profiles["org-a"], PolicyFile(profiles=profiles)),
        )
        assert len(scoped) == len(events)
        assert all(
            item.cwd == root_a and item.scope_basis == "native_call_metadata" for item in scoped
        )
        assert all(item.cwd is None and item.scope_basis == "unknown_scope" for item in events)
    else:
        assert not any(row.in_scope for row in reduction.executions)
        assert reduction.candidates == ()


def test_trusted_explicit_correlation_can_replace_clock_but_not_key_or_scope():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    fact = replace(fact, correlation_id="one-local-call", timing_known=False)
    native = request(fact, correlation="one-local-call", window=None)
    link = reconcile([fact], [native, native])[fact.execution_id]
    assert link.attribution is Attribution.EXPLICIT
    assert link.requests == (native.key,)
    bad = replace(native, fingerprint_key_id="rotated-key")
    rejected = reconcile([fact], [bad])[fact.execution_id]
    assert rejected.attribution is Attribution.UNATTRIBUTED
    assert "key_mismatch" in rejected.reasons


@pytest.mark.parametrize(
    ("visible_refs", "expected"),
    [
        (frozenset({"source-a:ep:0"}), Attribution.EXPLICIT),
        (frozenset({"source-b:ep:0"}), Attribution.UNATTRIBUTED),
        (None, Attribution.EXPLICIT),
    ],
)
def test_explicit_correlation_cannot_override_complete_result_contradiction(visible_refs, expected):
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    fact = replace(
        fact,
        correlation_id="one-local-call",
        timing_known=False,
        refs=("source-a:ep:0",),
        refs_complete=True,
        outcome=Outcome.OK,
    )
    native = request(fact, correlation="one-local-call", window=None, result_refs=visible_refs)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is expected
    if expected is Attribution.UNATTRIBUTED:
        assert link.requests == ()
        assert "result_mismatch" in link.reasons
    else:
        assert link.requests == (native.key,)


def test_cwd_and_timing_without_query_or_result_are_not_join_evidence():
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    native = request(fact, fingerprint=None)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "insufficient_evidence" in link.reasons


@pytest.mark.parametrize("profile", ["personal-local", "org-a"])
def test_unjoined_unknown_journal_scope_is_never_company_counter_authority(profile, tmp_path):
    from ssgrep.accounting.views.filter import Scoper, effective_scoped_journal
    from tests.accounting.ledger.reduce_builders import event, ts

    selected = ProfilePolicy(
        profile_id=profile,
        roots=() if profile == "personal-local" else (str(tmp_path / "org"),),
        collection_enabled=True,
    )
    events = [
        replace(item, cwd=None, scope_basis="unknown_scope")
        for item in event("unjoined", ts(12, 10), profile=profile)
    ]
    projected = effective_scoped_journal(events, (), Scoper(selected))
    assert len(projected) == (len(events) if profile == "personal-local" else 0)
    assert all(item.cwd is None and item.scope_basis == "unknown_scope" for item in projected)


@pytest.mark.parametrize(
    ("correlation", "expected"),
    [
        ("recorded-call", Attribution.EXPLICIT),
        ("different-call", Attribution.UNATTRIBUTED),
        (None, Attribution.INFERRED_UNIQUE),
    ],
)
def test_known_correlation_conflict_is_not_an_inference_candidate(correlation, expected):
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    fact = replace(fact, correlation_id="recorded-call")
    native = request(fact, correlation=correlation)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is expected
    if expected is Attribution.UNATTRIBUTED:
        assert not link.requests
        assert "correlation_mismatch" in link.reasons
    else:
        assert link.requests == (native.key,)


def test_workspace_alias_retargeting_is_checked_fresh_between_reconciliations(tmp_path):
    privacy_policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(first, target_is_directory=True)
    fact = replace(fact, cwd=str(first.resolve()), scope_basis="process_cwd")
    native = request(fact, cwd=str(alias))
    assert reconcile([fact], [native])[fact.execution_id].attribution is Attribution.INFERRED_UNIQUE
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED and not link.requests
    assert "cwd_mismatch" in link.reasons
    alias.unlink()
    alias.symlink_to(first, target_is_directory=True)
    assert reconcile([fact], [native])[fact.execution_id].attribution is Attribution.INFERRED_UNIQUE
