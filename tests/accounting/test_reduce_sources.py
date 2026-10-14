"""6.3/6.4 + 5.5: frequency view, source atoms, drift, scope, cycles, determinism."""

from __future__ import annotations

import random
from pathlib import Path

from ssgrep.accounting.contracts import RawEvidence, Scenario
from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.frequency import (
    BASIS_CANDIDATES,
    FrequencyView,
    build_view,
    eligible_scenarios,
    lexical_view,
    milli_log2,
)
from ssgrep.accounting.reduce import methodology_for, reduce_profile
from ssgrep.accounting.reduce_types import Reduction
from ssgrep.accounting.sources import Edge, find_cycles
from ssgrep.accounting.valuation import value_evidence
from ssgrep.accounting.valuation_inputs import AtomStatus
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
    rare_view,
    results_json,
    source_session,
    ts,
)
from tests.accounting.test_reduce_evidence import CODE, FACT, PATH, VIEW, consumer, reduce_scene

PATH_TOKEN = Token("path", PATH)


# ------------------------------------------------------------ frequency (6.3)


def test_log_idf_fixed_point_matches_exact_math_and_is_integer() -> None:
    assert milli_log2(1024, 1) == 10_000
    assert milli_log2(8, 8) == 0
    assert milli_log2(3, 2) == 584
    assert isinstance(milli_log2(1001, 11), int)


def test_thresholds_8_6_4_select_nested_scenarios_in_integer_milli_bits() -> None:
    def view(df: int) -> FrequencyView:
        return FrequencyView("v", BASIS_CANDIDATES, doc_count=1000, df={PATH_TOKEN.key: df})

    assert eligible_scenarios(view(0).rarity_milli(PATH_TOKEN)) == (
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
    ):
        assert view.rarity_milli(PATH_TOKEN) is None, name
        reduction = reduce_scene(tmp_path / name, consumer(), view=view)
        assert reduction.candidates == (), name
        assert reduction.coverage["exclusion.rarity_unknown"] >= 1, name


def test_view_is_normalized_like_extraction_and_content_addressed() -> None:
    first = build_view([FACT, "other"])
    second = build_view(["other", FACT])
    assert first.version_id == second.version_id
    assert first.df[PATH_TOKEN.key] == 1
    assert build_view([FACT, "x", "y"]).version_id != first.version_id
    assert methodology_for(first).frequency_view == first.version_id
    assert methodology_for(first).calibrated is False
    assert dict(methodology_for(first).thresholds) == {
        "conservative": 8,
        "central": 6,
        "optimistic": 4,
    }


def test_rarity_gate_shrinks_the_atom_tiers_by_threshold(tmp_path: Path) -> None:
    def tiers(df: int) -> tuple[int, int, int]:
        view = FrequencyView(
            "v", BASIS_CANDIDATES, doc_count=1000, df={PATH_TOKEN.key: df, f"error_code\0{CODE}": 0}
        )
        reduction = reduce_scene(tmp_path / f"df{df}", consumer(), view=view)
        (record,) = reduction.candidates
        c = record.candidate
        return len(c.immediate), len(c.discovery), len(c.episode)

    top, mid, low = tiers(0), tiers(10), tiers(50)
    assert top[0] > 0 and top[1] > 0 and top[2] > 0
    assert mid[0] == 0 and mid[1] > 0
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
    events = event(
        "n-1", ts(12, 10, 0, 1), text="why did the deploy fail", cwd=ORG_CWD, profile=ORG
    )
    return reduce_profile(
        ORG,
        claude_evidence(root, files),
        events,
        policy=policy("/work/orgrepo", profile=ORG),
        frequency=VIEW,
        fingerprint_key=KEY,
    )


def test_out_of_scope_source_cost_is_excluded_with_missing_coverage_consumer_stays_measured(
    tmp_path: Path,
) -> None:
    inside = org_scene(tmp_path / "in", "/work/orgrepo/other")
    outside = org_scene(tmp_path / "out", "/home/me/personal")
    (rec_in,) = inside.candidates
    assert all(a.usable for a in rec_in.candidate.immediate)
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
            policy=policy(),
            frequency=VIEW,
            fingerprint_key=KEY,
        )
        digests.add(reduction.digest)
    assert len(digests) == 1
    assert reduction.coverage["evidence_conflicts"] >= 1
    assert reduction.candidates and reduction.coverage["ambiguous"] == 2
    # a real change in the evidence changes the digest (the digest is not constant)
    changed = reduce_profile(
        PROFILE, evidence, events[:-3], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    assert changed.digest not in digests
    assert CWD and rare_view  # builders stay importable for sibling suites
