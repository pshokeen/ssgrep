"""Quantified boundary examples for overlap-free, nested, identically-debited valuation."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from ssgrep.accounting.contracts import Basis, Quantity, Reuse, Scenario, TaskId
from ssgrep.accounting.units import CONTEXT_TOKENS, WORK_SECONDS, UnitMismatchError, UnitSpec
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_credit import (
    InvariantError,
    assert_nested,
    credit_scenario,
    scenario_atoms,
)
from ssgrep.accounting.valuation_inputs import AtomStatus, EvidenceBundle
from tests.accounting.valuation_builders import (
    CHARS,
    METHOD,
    TASK,
    atom,
    candidate,
    debit,
)

S = Scenario


def reduce(candidates, debits=()):
    valuation = value_evidence(EvidenceBundle(tuple(candidates), tuple(debits)), METHOD)
    return rollup(valuation.leaves, CHARS), valuation


def by_scenario(estimates):
    return {e.scenario: e for e in estimates}


def test_credit_100_against_debit_160_is_minus_60_in_every_scenario():
    est, _ = reduce([candidate("c1")], [debit("d1", 640)])
    nets = {e.scenario: e.net.value for e in est}
    assert nets == {S.CONSERVATIVE: -60, S.CENTRAL: -60, S.OPTIMISTIC: -60}
    assert {e.credit.value for e in est} == {100}
    assert {e.debit.value for e in est} == {160}  # the identical debit set everywhere


def test_two_facts_sharing_one_discovery_step_credit_once_but_two_exposures_debit_twice():
    facts = [candidate("fact-a"), candidate("fact-b", candidate_type="identifier")]
    exposures = [debit("exp-1", 200, exposure_id="e1"), debit("exp-2", 200, exposure_id="e2")]
    est, _ = reduce(facts, exposures)
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.credit.value == 100  # 400 chars once, not 800
    assert cons.debit.value == 100  # 2 x (200 // 4)... summed chars 400 // 4
    assert cons.known_debits == 2
    assert cons.net.value == 0


def test_sum_before_divide_applies_to_credit_and_debit():
    # 3 atoms of 3 chars each: per-atom division would give 0; sum-then-divide gives 2.
    cands = [
        candidate(f"c{i}", immediate=(atom(f"a{i}", 3, start=10 * i, origin=f"o{i}"),))
        for i in range(3)
    ]
    est, _ = reduce(cands)
    assert by_scenario(est)[S.CONSERVATIVE].credit.value == 2


def test_overlapping_spans_union_by_characters_not_by_atom_count():
    # immediate [100,500), discovery [0,300) (overlaps), episode [0,1000)
    cand = candidate(
        "c1",
        immediate=(atom("imm", 400, start=100),),
        discovery=(atom("disc", 300, start=0),),
        episode=(atom("ep", 1000, start=0),),
    )
    est, _ = reduce([cand])
    got = {e.scenario: e.credit.value for e in est}
    assert got == {S.CONSERVATIVE: 100, S.CENTRAL: 125, S.OPTIMISTIC: 250}


def test_optimistic_retains_lower_scenario_atoms_when_boundaries_differ():
    # The episode does NOT contain the immediate step (disjoint boundaries).
    cand = candidate(
        "c1",
        immediate=(atom("imm", 400, start=5000),),
        discovery=(atom("disc", 400, start=0),),
        episode=(atom("ep", 400, start=1000),),
    )
    sets = {s: {a.atom_id for a in atoms} for s, atoms in scenario_atoms(cand).items()}
    assert sets[S.CONSERVATIVE] == {"imm"}
    assert sets[S.CENTRAL] == {"imm", "disc"}
    assert sets[S.OPTIMISTIC] == {"imm", "disc", "ep"}
    est, _ = reduce([cand])
    assert by_scenario(est)[S.OPTIMISTIC].credit.value == 300


def test_user_supplied_and_contradicted_and_known_candidates_earn_no_credit():
    cands = [
        candidate("user", user_supplied=True),
        candidate("bad", reuse=Reuse.CONTRADICTED),
        candidate("old", novel=False),
    ]
    est, valuation = reduce(cands, [debit("remediation", 80, kind="remediation")])
    assert {e.credit.value for e in est} == {0}
    assert {e.net.value for e in est} == {-20}  # remediation remains a cost
    reasons = {a.candidate_id: a.ineligible for a in valuation.allocations}
    assert reasons == {"user": "user_supplied", "bad": "contradicted", "old": "not_novel"}


def test_conservative_requires_an_interpreted_successful_action():
    failed = candidate("c1", action_outcome_ok=False, discovery=(atom("disc", 200, start=1000),))
    sets = {s: {a.atom_id for a in atoms} for s, atoms in scenario_atoms(failed).items()}
    assert sets[S.CONSERVATIVE] == set()
    assert sets[S.CENTRAL] == {"disc"}


def test_inspected_without_action_only_gets_episode_when_relay_or_report_evidence():
    plain = candidate("c1", reuse=Reuse.INSPECTED, episode=(atom("ep", 800, start=0),))
    assert all(not v for v in scenario_atoms(plain).values())
    relay = replace(plain, explicit_relay=True)
    sets = scenario_atoms(relay)
    assert [a.atom_id for a in sets[S.OPTIMISTIC]] == ["ep"]
    assert sets[S.CENTRAL] == ()


def test_unknown_source_cost_keeps_subtotal_and_suppresses_complete_net():
    cands = [
        candidate("known"),
        candidate(
            "unreadable",
            immediate=(atom("gone", 400, origin="ev-2", status=AtomStatus.UNAVAILABLE),),
        ),
    ]
    est, _ = reduce(cands, [debit("d1", 160)])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.credit.value == 100  # quantified subtotal retained
    assert cons.credit_missing == (("gone", "unavailable"),)
    assert cons.net.basis is Basis.UNAVAILABLE and cons.net.value is None
    assert "unavailable" in cons.net.reason
    assert cons.known_net == 60  # visible as a known subtotal, never as the complete net
    assert not cons.complete


def test_missing_comparable_debit_suppresses_complete_net_but_keeps_known_debit():
    est, _ = reduce([candidate("c1")], [debit("d1", 160), debit("d2", None)])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.debit.value == 40 and cons.known_debits == 1
    assert cons.debit_missing == (("d2", "downstream_exposure_unobserved"),)
    assert cons.net.value is None and cons.net.basis is Basis.UNAVAILABLE


def test_partial_exposure_debit_with_observed_length_is_complete_debit():
    est, _ = reduce([candidate("c1")], [debit("shell", 80, partial=True)])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.debit.value == 20
    assert cons.debit_missing == ()
    assert cons.net.value == 80


def test_invalid_provenance_is_excluded_with_reason_not_counted_as_missing_cost():
    cands = [
        candidate("c1", immediate=(atom("drift", 400, status=AtomStatus.DIGEST_CONFLICT),)),
        candidate("c2", immediate=(atom("loop", 400, origin="e2", status=AtomStatus.CYCLE),)),
    ]
    est, valuation = reduce(cands)
    assert by_scenario(est)[S.CONSERVATIVE].credit.value == 0
    assert by_scenario(est)[S.CONSERVATIVE].net.value == 0
    excluded = {
        a.candidate_id: a.excluded for a in valuation.allocations if a.scenario is S.CENTRAL
    }
    assert excluded == {"c1": (("drift", "digest_conflict"),), "c2": (("loop", "cycle"),)}


def test_out_of_scope_source_is_missing_coverage_while_consumer_debits_stay():
    cand = candidate("c1", immediate=(atom("other-ws", 400, status=AtomStatus.OUT_OF_SCOPE),))
    est, _ = reduce([cand], [debit("d1", 160)])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.debit.value == 40
    assert cons.credit_missing == (("other-ws", "out_of_scope"),)
    assert cons.net.value is None


def test_repeated_identical_inputs_do_not_double_count():
    c = candidate("c1")
    d = debit("d1", 160)
    est, _ = reduce([c, c], [d, d])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert (cons.credit.value, cons.debit.value) == (100, 40)


def test_conflicting_duplicate_debit_ids_become_unknown_not_summed():
    est, _ = reduce([candidate("c1")], [debit("d1", 160), debit("d1", 800)])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.debit_missing == (("d1", "conflicting_duplicate_debit"),)
    assert cons.net.value is None


def test_nested_scenario_invariant_over_randomized_atom_sets():
    rng = random.Random(20261012)
    for trial in range(300):
        cands = []
        for i in range(rng.randint(1, 5)):

            def spans(prefix, n, i=i):
                return tuple(
                    atom(
                        f"{prefix}{i}-{k}",
                        rng.randint(1, 900),
                        start=rng.randint(0, 1500),
                        origin=f"o{rng.randint(0, 2)}",
                        digest=f"dg{rng.randint(0, 1)}",
                    )
                    for k in range(rng.randint(0, n))
                )

            cands.append(
                candidate(
                    f"c{trial}-{i}",
                    reuse=rng.choice(list(Reuse)),
                    novel=rng.random() > 0.2,
                    user_supplied=rng.random() > 0.8,
                    action_outcome_ok=rng.random() > 0.3,
                    explicit_relay=rng.random() > 0.7,
                    report_reference=rng.random() > 0.7,
                    immediate=spans("i", 2),
                    discovery=spans("d", 3),
                    episode=spans("e", 3),
                )
            )
        sets = {s: set() for s in (S.CONSERVATIVE, S.CENTRAL, S.OPTIMISTIC)}
        for c in cands:
            for s, atoms in scenario_atoms(c).items():
                sets[s] |= {a.atom_id for a in atoms}
        assert sets[S.CONSERVATIVE] <= sets[S.CENTRAL] <= sets[S.OPTIMISTIC]
        credits = [credit_scenario(cands, s, CHARS).footprint for s in sets]
        sizes = [sum(e - st for iv in fp.values() for st, e in iv) for fp in credits]
        assert sizes == sorted(sizes), f"trial {trial}: {sizes}"
        est, _ = reduce(cands)
        values = [e.credit.value for e in est]
        assert values == sorted(values)


def test_nesting_assertion_rejects_a_non_nested_selection():
    with pytest.raises(InvariantError):
        assert_nested({S.CONSERVATIVE: {"a"}, S.CENTRAL: {"b"}, S.OPTIMISTIC: {"a", "b"}})


def test_atom_order_and_candidate_order_do_not_change_credit():
    cands = [
        candidate("a", immediate=(atom("x", 300, start=0), atom("y", 300, start=100))),
        candidate("b", immediate=(atom("z", 500, start=200),)),
    ]
    forward, _ = reduce(cands)
    backward, _ = reduce(list(reversed(cands)))
    assert forward == backward
    assert by_scenario(forward)[S.CONSERVATIVE].credit.value == 175  # union [0,700)


def test_mixed_units_methods_tokenizers_and_currencies_are_refused():
    est, valuation = reduce([candidate("c1")], [debit("d1", 80)])
    other = UnitSpec(CONTEXT_TOKENS, "exact", tokenizer="tok==1")
    with pytest.raises(UnitMismatchError):
        rollup(valuation.leaves, other)
    foreign = replace(debit("d2", 80), unit=other)
    with pytest.raises(UnitMismatchError):
        value_evidence(EvidenceBundle((candidate("c1"),), (foreign,)), METHOD)
    secs = UnitSpec(WORK_SECONDS, "monotonic_v1")
    leaf_secs = replace(valuation.leaves[0], unit=secs)
    with pytest.raises(UnitMismatchError):
        rollup([valuation.leaves[0], leaf_secs], CHARS)
    changed = replace(valuation.leaves[0], methodology="other", day="2026-11-02")
    with pytest.raises(UnitMismatchError):
        rollup([valuation.leaves[0], changed], CHARS)


def test_rollup_sums_unique_leaves_and_never_task_plus_root():
    root_only = TaskId("root-1", None, "main")
    second = TaskId("root-1", "turn-2", "main")
    cands = [
        candidate("a", task=TASK),
        candidate("b", task=second, immediate=(atom("other", 800, origin="ev-9"),)),
    ]
    est, valuation = reduce(cands, [debit("d1", 80, task=TASK), debit("d2", 80, task=second)])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert (cons.credit.value, cons.debit.value, cons.net.value) == (300, 40, 260)
    # feeding the same leaves twice (e.g. a task view and a root view) cannot double count
    again = rollup(list(valuation.leaves) + list(valuation.leaves), CHARS)
    assert again == est
    assert all(leaf.task != root_only for leaf in valuation.leaves)


def test_task_unknown_reuse_is_deduplicated_at_the_resolved_root():
    unresolved = TaskId("root-1", None, "main")
    shared = candidate("resolved", task=TASK)
    overlapping = candidate("unknown-same", task=unresolved)  # same 400-char step
    extra = candidate("unknown-extra", task=unresolved, immediate=(atom("new", 200, start=900),))
    est, _ = reduce([shared, overlapping, extra])
    cons = by_scenario(est)[S.CONSERVATIVE]
    assert cons.credit.value == 150  # 100 once at the task + 50 new at the root; not 250


def test_task_unknown_is_full_credit_when_no_resolved_task_overlaps():
    unresolved = TaskId("root-2", None, "main")
    est, _ = reduce([candidate("u", task=unresolved)])
    assert by_scenario(est)[S.CONSERVATIVE].credit.value == 100


def test_parent_relay_and_child_share_one_contribution():
    parent = TaskId("root-1", "turn-1", "main")
    child = TaskId("root-1", "turn-1", "child-a")
    unresolved = TaskId("root-1", None, "main")
    cands = [
        candidate("p", task=parent),
        candidate("c", task=child),
        candidate("u", task=unresolved),
    ]
    est, _ = reduce(cands)
    # parent and child are different consumer tasks with their own credit; the unresolved
    # reuse of the very same evidence adds nothing on top of either
    assert by_scenario(est)[S.CONSERVATIVE].credit.value == 200


def test_credit_across_days_for_one_task_is_counted_once():
    cands = [candidate("d1", day="2026-11-01"), candidate("d2", day="2026-11-02")]
    est, valuation = reduce(cands)
    assert by_scenario(est)[S.CONSERVATIVE].credit.value == 100
    per_day = {leaf.day: leaf.scenarios[0].credit_raw for leaf in valuation.leaves}
    assert per_day == {"2026-11-01": 400, "2026-11-02": 0}


def test_input_order_independence_of_the_whole_valuation():
    rng = random.Random(7)
    cands = [
        candidate(f"c{i}", immediate=(atom(f"a{i}", 100 + i, start=i * 50),)) for i in range(8)
    ]
    debits = [debit(f"d{i}", 40 * i) for i in range(6)]
    baseline = value_evidence(EvidenceBundle(tuple(cands), tuple(debits)), METHOD)
    for _ in range(5):
        rng.shuffle(cands)
        rng.shuffle(debits)
        assert value_evidence(EvidenceBundle(tuple(cands), tuple(debits)), METHOD) == baseline


def test_negative_net_is_preserved_not_clamped():
    est, _ = reduce([candidate("c1")], [debit("big", 4000)])
    assert by_scenario(est)[S.CONSERVATIVE].net == Quantity(
        -900, CONTEXT_TOKENS, Basis.ESTIMATED, by_scenario(est)[S.CONSERVATIVE].net.method
    )
