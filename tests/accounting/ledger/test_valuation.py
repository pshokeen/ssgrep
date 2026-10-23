"""Quantified boundary examples for overlap-free, nested, identically-debited valuation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import socket
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from ssgrep.accounting.collect.frequency import FrequencyView
from ssgrep.accounting.ledger.contracts import Basis, Phase, Quantity, Reuse, Scenario, TaskId
from ssgrep.accounting.ledger.ledger_codec import (
    bundle_from_dict,
    bundle_to_dict,
    from_plain,
    to_plain,
)
from ssgrep.accounting.ledger.leverage import (
    NoteInvestment,
    OverheadEvent,
    Purpose,
    note_leverage,
    overhead_views,
    purpose_of,
)
from ssgrep.accounting.ledger.reduce import methodology_for
from ssgrep.accounting.ledger.reduce_cands import WORK
from ssgrep.accounting.ledger.units import (
    CHARS_DIV4_V1,
    CONTEXT_TOKENS,
    WORK_SECONDS,
    ActorInterval,
    TariffError,
    TokenizerPin,
    UnitMismatchError,
    UnitSpec,
    UsageUpdate,
    capped_gap_v1,
    chars_div4_v1,
    context_estimate,
    measured_duration,
    merge_intervals,
    merged_span_chars,
    modeled_cost,
    normalize_usage,
    reported_cost,
    resolve_tokenizer,
    sum_money,
    sum_quantities,
    union_length,
    validate_tariff,
    work_time,
)
from ssgrep.accounting.ledger.valuation import Leaf, debit_raw, rollup, value_evidence
from ssgrep.accounting.ledger.valuation_inputs import (
    AtomStatus,
    Debit,
    Eligibility,
    EvidenceBundle,
    InvariantError,
    Methodology,
    ScenarioCredit,
    assert_nested,
    credit_scenario,
    scenario_atoms,
)
from ssgrep.accounting.views.lookups import LOOKUP_WORK_ASSUMPTIONS, LookupTiming, journal_timings
from ssgrep.accounting.views.records import build_receipt
from tests.accounting.ledger.reduce_builders import event, ts
from tests.accounting.ledger.test_reduce_evidence import consumer as evidence_consumer, reduce_scene
from tests.accounting.ledger.valuation_builders import (
    CHARS,
    DAY,
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


def by_scenario(rows):
    return {e.scenario: e for e in rows}


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
        est, valuation = reduce(cands)
        values = [e.credit.value for e in est]
        assert values == sorted(values)
        for cand in cands:
            allocated = [
                sum(
                    row.credit_raw
                    for row in valuation.allocations
                    if row.candidate_id == cand.candidate_id and row.scenario is scenario
                )
                for scenario in sets
            ]
            assert allocated == sorted(allocated), (trial, cand.candidate_id, allocated)
        for scenario in sets:
            assert sum(
                row.credit_raw for row in valuation.allocations if row.scenario is scenario
            ) == sum(
                row.credit_raw
                for leaf in valuation.leaves
                for row in leaf.scenarios
                if row.scenario is scenario
            )


def test_nesting_assertion_rejects_a_non_nested_selection():
    with pytest.raises(InvariantError):
        assert_nested({S.CONSERVATIVE: {"a"}, S.CENTRAL: {"b"}, S.OPTIMISTIC: {"a", "b"}})


def test_task_credit_rejects_non_nested_sizes_even_when_atom_ids_are_nested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ssgrep.accounting.ledger import valuation

    def credit(size: int) -> ScenarioCredit:
        return ScenarioCredit({("event", "digest"): ((0, size),)}, (), ("shared",))

    credits = {
        S.CONSERVATIVE: credit(3),
        S.CENTRAL: credit(2),
        S.OPTIMISTIC: credit(1),
    }

    def inconsistent_credit(candidates, scenario, unit, base=None, previous=None):
        return credits[scenario]

    monkeypatch.setattr(valuation, "credit_scenario", inconsistent_credit)

    with pytest.raises(InvariantError, match="scenario credit is not nested"):
        valuation._task_credit([], CHARS, {})


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


def test_public_debit_conversion_rejects_cross_unit_request() -> None:
    with pytest.raises(UnitMismatchError):
        debit_raw(debit("visible", 120), UnitSpec(WORK_SECONDS, "tool_duration_v1"))


@pytest.mark.parametrize(
    "unit, reason",
    [
        (UnitSpec(CONTEXT_TOKENS, "exact:pinned", tokenizer="pinned"), "exact_tokens_unavailable"),
        (UnitSpec(WORK_SECONDS, "tool_duration_v1"), "source_seconds_unavailable"),
    ],
)
def test_missing_source_measurement_prevents_complete_net(unit: UnitSpec, reason: str) -> None:
    bundle = EvidenceBundle(candidates=(candidate("fact", immediate=(atom("source", 40),)),))
    valued = value_evidence(bundle, Methodology("missing-source-cost", (unit,)))
    for estimate in rollup(valued.leaves, unit):
        assert estimate.credit.value == 0
        assert estimate.known_net == 0
        assert estimate.net.value is None
        assert estimate.credit_missing == (("source", reason),)


def test_acceptance_cut_preserves_partial_debit_and_late_candidate_visibility() -> None:
    early = debit("visible", 120, partial=True, accepted_day="2026-11-02")
    late = candidate("late", accepted_day="2026-11-04")
    bundle = EvidenceBundle((late,), (early,))
    assert EvidenceBundle().last_accepted == ""
    assert bundle.last_accepted == "2026-11-04"
    cut = bundle.accepted_through("2026-11-03")
    assert cut.last_accepted == "2026-11-02"
    for estimate in rollup(value_evidence(cut, METHOD).leaves, CHARS):
        assert estimate.credit.value == 0
        assert estimate.debit.value == 30
        assert estimate.net.value == -30
        assert estimate.debit_missing == ()
    assert all(
        row.credit.value == 100 for row in rollup(value_evidence(bundle, METHOD).leaves, CHARS)
    )


def test_bundle_rejects_debits_with_an_unregistered_measurement_method() -> None:
    foreign = replace(debit("exact", 40), unit=UnitSpec(CONTEXT_TOKENS, "exact:other"))
    with pytest.raises(UnitMismatchError):
        value_evidence(EvidenceBundle(debits=(foreign,)), METHOD)


def test_decoded_snapshot_cannot_roll_up_reversed_scenario_credit() -> None:
    valued = value_evidence(EvidenceBundle(candidates=(candidate("fact"),)), METHOD)
    encoded = to_plain(valued.leaves[0])
    restored = from_plain(Leaf, encoded)
    assert all(row.credit.value == 100 for row in rollup([restored], CHARS))
    encoded["scenarios"][0]["credit_raw"] = 800
    corrupted = from_plain(Leaf, encoded)
    with pytest.raises(InvariantError):
        rollup([corrupted], CHARS)


def test_decoded_bundle_rejects_conflicting_exact_counts_across_source_tiers() -> None:
    unit = UnitSpec(CONTEXT_TOKENS, "exact:pinned", tokenizer="pinned")
    method = Methodology("exact-source", (unit,))
    step = atom("step", 120, tokens=30)
    bundle = EvidenceBundle(candidates=(candidate("fact", immediate=(step,), discovery=(step,)),))
    encoded = to_plain(bundle)
    restored = from_plain(EvidenceBundle, encoded)
    estimates = rollup(value_evidence(restored, method).leaves, unit)
    assert all(row.credit.value == 30 for row in estimates)
    encoded["candidates"][0]["discovery"][0]["tokens"] = 1
    corrupted = from_plain(EvidenceBundle, encoded)
    with pytest.raises(InvariantError):
        value_evidence(corrupted, method)


def test_truncated_candidate_absent_from_visible_evidence_earns_no_credit():
    visible = debit("show-truncated", 120, partial=True, exposure_id="show-1")
    hidden = candidate("hidden-fact", immediate=())
    valuation = value_evidence(EvidenceBundle((hidden,), (visible,)), METHOD)
    assert not valuation.leaves[0].scenarios[0].credit_raw
    assert valuation.leaves[0].debit_raw == 120


@pytest.mark.parametrize("reverse", [False, True])
def test_daily_tiers_follow_supported_owner_not_first_optimistic_day(reverse):
    relay = candidate(
        "relay",
        day="2026-11-01",
        reuse=Reuse.INSPECTED,
        explicit_relay=True,
        immediate=(),
        episode=(atom("relay-span", 400),),
    )
    acted = candidate("acted", day="2026-11-02", immediate=(atom("acted-span", 400),))
    candidates = [relay, acted]
    if reverse:
        candidates.reverse()
    costs = [debit("early", 160), debit("late", 80, day="2026-11-02")]
    estimates, valuation = reduce(candidates, costs)
    leaves = {leaf.day: leaf for leaf in valuation.leaves}
    assert [row.credit_raw for row in leaves["2026-11-01"].scenarios] == [0, 0, 0]
    assert [row.credit_raw for row in leaves["2026-11-02"].scenarios] == [400, 400, 400]
    assert (leaves["2026-11-01"].debit_raw, leaves["2026-11-02"].debit_raw) == (160, 80)
    for day, credit, debit_value, net in (
        ("2026-11-01", 0, 40, -40),
        ("2026-11-02", 100, 20, 80),
    ):
        assert [
            (row.credit.value, row.debit.value, row.net.value)
            for row in rollup([leaves[day]], CHARS)
        ] == [(credit, debit_value, net)] * 3
    assert [(row.credit.value, row.debit.value, row.net.value) for row in estimates] == [
        (100, 60, 40)
    ] * 3
    for scenario in S:
        owned = {
            row.candidate_id: row.credit_raw
            for row in valuation.allocations
            if row.scenario is scenario
        }
        assert owned == {"relay": 0, "acted": 400}


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_daily_ownership_preserves_nested_shares_and_sum_before_rounding(reverse):
    relay = candidate(
        "relay",
        day="2026-11-01",
        reuse=Reuse.INSPECTED,
        explicit_relay=True,
        immediate=(),
        episode=(atom("relay-span", 7),),
    )
    acted = candidate("acted", day="2026-11-02", immediate=(atom("acted-span", 3, start=2),))
    candidates = [relay, acted]
    if reverse:
        candidates.reverse()
    estimates, valuation = reduce(candidates, [debit("early", 1), debit("late", 3, day=acted.day)])
    leaves = {leaf.day: leaf for leaf in valuation.leaves}
    assert [row.credit_raw for row in leaves[relay.day].scenarios] == [0, 0, 4]
    assert [row.credit_raw for row in leaves[acted.day].scenarios] == [3, 3, 3]
    assert [row.credit.value for row in estimates] == [0, 0, 1]
    assert [row.debit.value for row in estimates] == [1, 1, 1]
    for leaf in leaves.values():
        for row in leaf.scenarios:
            assert row.credit_raw == sum(
                allocation.credit_raw
                for allocation in valuation.allocations
                if allocation.day == leaf.day and allocation.scenario is row.scenario
            )


def test_rarity_gate_uses_canonical_frequency_threshold_mapping(monkeypatch):
    from ssgrep.accounting.ledger.contracts import THRESHOLD_MILLI

    supported = candidate("supported", rarity_lower_bound_milli=8000)
    assert len(scenario_atoms(supported)[S.CONSERVATIVE]) == 1
    monkeypatch.setitem(THRESHOLD_MILLI, S.CONSERVATIVE, 8001)
    assert scenario_atoms(supported)[S.CONSERVATIVE] == ()
    assert len(scenario_atoms(supported)[S.CENTRAL]) == 1
    assert len(scenario_atoms(supported)[S.OPTIMISTIC]) == 1


"""Public valuation boundaries: subunit rounding, tariff dates, and incompatible snapshots."""


def test_fractional_debits_across_days_round_only_at_consumer_rollup() -> None:
    bundle = EvidenceBundle(
        debits=(
            debit("first", 3, day="2026-11-01", partial=True),
            debit("second", 3, day="2026-11-02", partial=True),
        )
    )
    valuation = value_evidence(bundle, METHOD)
    assert len(valuation.leaves) == 2
    assert all(leaf.debit_raw == 3 for leaf in valuation.leaves)
    estimates = rollup(valuation.leaves, CHARS)
    assert all(estimate.debit.value == 1 for estimate in estimates)
    assert all(estimate.net.value == -1 for estimate in estimates)
    assert all(estimate.debit_missing == () for estimate in estimates)
    assert all(estimate.known_debits == 2 for estimate in estimates)


def test_known_work_seconds_are_not_divided_like_context_characters() -> None:
    unit = UnitSpec(WORK_SECONDS, "tool_duration_v1")
    cost = Debit(
        "work", TASK, DAY, "generation", unit, Quantity.observed(7, WORK_SECONDS, unit.method)
    )
    valuation = value_evidence(EvidenceBundle(debits=(cost,)), Methodology("time", (unit,)))
    estimates = rollup(valuation.leaves, unit)
    assert all(estimate.debit.value == 7 and estimate.net.value == -7 for estimate in estimates)


def test_conflicting_snapshot_leaves_are_rejected_instead_of_first_writer_winning() -> None:
    original = value_evidence(EvidenceBundle(debits=(debit("cost", 40),)), METHOD).leaves[0]
    correction = replace(original, debit_raw=80)
    for leaves in ((original, correction), (correction, original)):
        with pytest.raises(ValueError, match="conflicting leaves"):
            rollup(leaves, CHARS)


def test_tariff_adjacent_intervals_change_at_midnight_and_round_once_per_cost() -> None:
    raw: dict[str, Any] = deepcopy(TARIFF)
    for entry in raw["entries"]:
        entry["micros_per_million_tokens"] = 500_000
        entry["effective_to"] = "2026-11-01"
    raw["entries"] += [
        {
            **entry,
            "effective_from": "2026-11-01",
            "effective_to": None,
            "micros_per_million_tokens": 1_500_000,
        }
        for entry in raw["entries"]
    ]
    tariff = validate_tariff(raw)
    usage = normalize_usage([UsageUpdate("provider", "stream", 1, input_tokens=1, output_tokens=1)])
    before = modeled_cost(tariff, usage["provider"], "m1", "2026-10-31", "USD")
    after = modeled_cost(tariff, usage["provider"], "m1", "2026-11-01", "USD")
    assert before.value == 1  # two half-micro categories, not two rounded zeros
    assert after.value == 3
    assert before.unit == after.unit == "micro_USD"
    assert before.method == after.method


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda raw: raw.update(entries=[]), "at least one entry"),
        (lambda raw: raw.update(entries=[None]), "entry must be an object"),
        (lambda raw: raw["entries"][0].update(model=None), "model is required"),
        (lambda raw: raw["entries"][0].update(effective_from=True), "ISO date"),
        (lambda raw: raw["entries"][0].update(micros_per_million_tokens=True), "integer"),
        (lambda raw: raw["entries"][0].update(micros_per_million_tokens=-1), "integer"),
    ],
)
def test_incomplete_or_unsafe_offline_tariff_cannot_produce_money(mutate, reason: str) -> None:
    raw = deepcopy(TARIFF)
    mutate(raw)
    with pytest.raises(TariffError, match=reason):
        validate_tariff(raw)


"""Unit semantics: chars_div4_v1, provider usage, tokenizer, time, tariffs."""


# ---- chars_div4_v1 -------------------------------------------------------------


def test_chars_div4_sums_characters_before_dividing():
    spans = [("e1", "d", 0, 3), ("e2", "d", 0, 3), ("e3", "d", 0, 3)]
    total = merged_span_chars(spans)
    assert total == 9
    assert chars_div4_v1(total) == 2  # per-span division would give 0
    assert sum(chars_div4_v1(3) for _ in range(3)) == 0


def test_overlapping_spans_are_deduplicated_before_counting():
    spans = [("e1", "d", 0, 100), ("e1", "d", 50, 150), ("e1", "other-digest", 0, 40)]
    assert merged_span_chars(spans) == 150 + 40
    estimate = context_estimate(merged_span_chars(spans))
    assert (estimate.value, estimate.unit, estimate.basis, estimate.method) == (
        47,
        CONTEXT_TOKENS,
        Basis.ESTIMATED,
        CHARS_DIV4_V1,
    )


def test_integer_division_floors_consistently():
    assert [chars_div4_v1(n) for n in (0, 1, 3, 4, 5, 7, 8)] == [0, 0, 0, 1, 1, 1, 2]
    with pytest.raises(ValueError):
        chars_div4_v1(-1)


def test_sum_quantities_refuses_mixed_units_and_methods_and_keeps_missing_reasons():
    a = Quantity(5, CONTEXT_TOKENS, Basis.ESTIMATED, CHARS_DIV4_V1)
    b = Quantity(7, CONTEXT_TOKENS, Basis.ESTIMATED, CHARS_DIV4_V1)
    gone = Quantity.unavailable(CONTEXT_TOKENS, "source_deleted")
    total = sum_quantities([a, b, gone])
    assert (total.known, total.known_count, total.missing_count) == (12, 2, 1)
    assert total.reasons == ("source_deleted",) and not total.complete
    with pytest.raises(UnitMismatchError):
        sum_quantities([a, Quantity(1, CONTEXT_TOKENS, Basis.OBSERVED, "exact:tok")])
    with pytest.raises(UnitMismatchError):
        sum_quantities([a, Quantity(1, "work_seconds", Basis.ESTIMATED, CHARS_DIV4_V1)])


# ---- provider usage ------------------------------------------------------------


def test_streaming_updates_count_only_the_final_report_per_stream():
    updates = [
        UsageUpdate("anthropic", "msg-1", 1, input_tokens=100, output_tokens=1),
        UsageUpdate("anthropic", "msg-1", 2, input_tokens=100, output_tokens=9),
        UsageUpdate("anthropic", "msg-1", 3, input_tokens=100, output_tokens=42),
    ]
    usage = normalize_usage(updates)["anthropic"]
    assert dict(usage.categories)["input_uncached"] == 100
    assert dict(usage.categories)["output"] == 42


def test_cumulative_counters_are_differenced_and_repeats_ignored():
    def cum(seq, inp, cache):
        return UsageUpdate(
            "openai", "turn-1", seq, input_tokens=inp, cache_read_tokens=cache, cumulative=True
        )

    updates = [cum(1, 100, 0), cum(2, 250, 50), cum(2, 250, 50), cum(3, 400, 120)]
    usage = normalize_usage(updates)["openai"]
    assert dict(usage.categories)["input_uncached"] == 400
    assert dict(usage.categories)["cache_read"] == 120
    assert usage.warnings == ()


def test_cumulative_counter_reset_is_flagged_not_negative():
    updates = [
        UsageUpdate("openai", "t", 1, input_tokens=500, cumulative=True),
        UsageUpdate("openai", "t", 2, input_tokens=30, cumulative=True),
    ]
    usage = normalize_usage(updates)["openai"]
    assert dict(usage.categories)["input_uncached"] == 530
    assert usage.warnings == ("counter_reset:t",)


def test_overlapping_categories_are_made_disjoint():
    update = UsageUpdate(
        "openai",
        "t",
        1,
        input_tokens=1000,
        cache_read_tokens=300,
        output_tokens=200,
        reasoning_tokens=80,
        input_includes_cache=True,
        output_includes_reasoning=True,
    )
    cats = dict(normalize_usage([update])["openai"].categories)
    assert cats == {
        "input_uncached": 700,
        "cache_read": 300,
        "cache_write": 0,
        "output": 120,
        "reasoning": 80,
    }
    assert sum(cats.values()) == 1000 + 200  # nothing counted twice or dropped


def test_inconsistent_overlap_is_a_warning_not_negative_usage():
    update = UsageUpdate(
        "openai", "t", 1, input_tokens=10, cache_read_tokens=300, input_includes_cache=True
    )
    usage = normalize_usage([update])["openai"]
    assert dict(usage.categories)["input_uncached"] == 0
    assert usage.warnings == ("inconsistent_usage:t",)


def test_provider_usage_is_a_separate_series_never_mixable_with_chars_estimates():
    usage = normalize_usage([UsageUpdate("anthropic", "m", 1, input_tokens=1000)])["anthropic"]
    reported = usage.quantity("input_uncached")
    estimated = context_estimate(4000)
    assert reported.unit != estimated.unit and reported.basis is Basis.REPORTED
    with pytest.raises(UnitMismatchError):
        sum_quantities([reported, estimated])


# ---- exact tokenizer, offline only ---------------------------------------------


def test_requested_tokenizer_not_installed_is_unavailable_and_never_downloads(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    def absent(name):
        raise ImportError(name)

    pin = TokenizerPin("tokenizers", "0.0.0", "/nonexistent/tok.json", "0" * 64)
    result = resolve_tokenizer(pin, importer=absent)
    assert not result.available
    quantity = result.count("hello world")
    assert quantity.value is None and quantity.basis is Basis.UNAVAILABLE
    assert quantity.reason == "tokenizer_library_not_installed:tokenizers"


def test_tokenizer_version_mismatch_and_missing_file_and_digest_mismatch(tmp_path: Path):
    path = tmp_path / "tok.json"
    path.write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()

    class Fake:
        class Tokenizer:
            @staticmethod
            def from_file(_):
                raise AssertionError("must not load a mismatched pin")

    importer = lambda _: Fake  # noqa: E731
    wrong_version = TokenizerPin("tokenizers", "1.0", str(path), digest)
    assert (
        resolve_tokenizer(wrong_version, importer, lambda _: "2.0").reason
        == "tokenizer_version_mismatch:2.0!=1.0"
    )
    missing = TokenizerPin("tokenizers", "1.0", str(tmp_path / "nope.json"), digest)
    assert resolve_tokenizer(missing, importer, lambda _: "1.0").reason == "tokenizer_file_missing"
    bad = TokenizerPin("tokenizers", "1.0", str(path), "f" * 64)
    assert (
        resolve_tokenizer(bad, importer, lambda _: "1.0").reason == "tokenizer_file_digest_mismatch"
    )
    assert resolve_tokenizer(
        TokenizerPin("tiktoken", "1", str(path), digest), importer
    ).reason.startswith("unsupported_tokenizer_library")


def test_pinned_tokenizer_counts_exactly_and_carries_its_identity(tmp_path: Path):
    path = tmp_path / "tok.json"
    path.write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()

    class Encoding:
        def __init__(self, text):
            self.ids = text.split()

    class Fake:
        class Tokenizer:
            @staticmethod
            def from_file(_):
                return type(
                    "T", (), {"encode": lambda s, text, add_special_tokens: Encoding(text)}
                )()

    pin = TokenizerPin("tokenizers", "1.0", str(path), digest)
    result = resolve_tokenizer(pin, lambda _: Fake, lambda _: "1.0")
    quantity = result.count("a b c d")
    assert quantity.value == 4 and quantity.method == f"exact:{pin.identity}"
    assert quantity.method != CHARS_DIV4_V1
    with pytest.raises(UnitMismatchError):
        sum_quantities([quantity, context_estimate(16)])


def test_corrupt_pinned_tokenizer_is_unavailable_without_character_fallback(tmp_path: Path) -> None:
    version = importlib.metadata.version("tokenizers")
    path = tmp_path / "tokenizer.json"
    payload = b'{"invalid tokenizer":true}'
    path.write_bytes(payload)
    pin = TokenizerPin("tokenizers", version, str(path), hashlib.sha256(payload).hexdigest())
    result = resolve_tokenizer(pin)
    assert not result.available
    assert result.reason.startswith("tokenizer_load_failed:")
    quantity = result.count("four words remain unknown")
    assert quantity.value is None and quantity.reason == result.reason


def test_missing_tokenizer_distribution_metadata_stays_unavailable() -> None:
    def unavailable_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    pin = TokenizerPin("tokenizers", "1", "/not-read/tokenizer.json", "0" * 64)
    result = resolve_tokenizer(pin, version_of=unavailable_version)
    assert result.count("not a zero token count").value is None
    assert result.reason == "tokenizer_library_not_installed:tokenizers"


# ---- time ------------------------------------------------------------------------


def test_two_actors_ten_seconds_each_is_20_work_seconds_but_10_elapsed():
    wt = work_time([ActorInterval("a", 0, 10_000), ActorInterval("b", 0, 10_000)])
    assert wt.work_seconds.value == 20
    assert wt.elapsed_seconds.value == 10
    assert wt.human_time.value is None and wt.human_time.reason == "no_dependency_evidence"


def test_one_actors_overlapping_intervals_are_not_double_counted():
    wt = work_time([ActorInterval("a", 0, 10_000), ActorInterval("a", 5_000, 15_000)])
    assert (wt.work_seconds.value, wt.elapsed_seconds.value) == (15, 15)


def test_work_time_floors_once_after_summing_milliseconds():
    wt = work_time([ActorInterval("a", 0, 700), ActorInterval("b", 1_000, 1_700)])
    assert wt.work_seconds.value == 1  # 1400 ms -> 1 s; per-interval flooring would give 0


def test_reversed_actor_interval_is_dropped_and_counted_never_negative():
    wt = work_time([ActorInterval("a", 10_000, 0), ActorInterval("a", 0, 4_000)])
    assert wt.work_seconds.value == 4 and wt.invalid_intervals == 1


def test_gap_of_900_seconds_is_capped_to_300():
    result = capped_gap_v1([0, 900_000])
    assert result.quantity.value == 300 and result.capped_gaps == 1
    assert result.quantity.method == "capped_gap_v1"


def test_gaps_under_the_cap_pass_through_and_sum():
    assert capped_gap_v1([0, 10_000, 70_000, 370_000]).quantity.value == 10 + 60 + 300


def test_reversed_clock_makes_capped_gap_unknown_not_negative_work():
    result = capped_gap_v1([100_000, 0, 50_000])
    assert result.invalid_gaps == 1
    assert result.quantity.value is None and result.quantity.reason == "invalid_timing_in_gaps"
    assert result.known_ms == 50_000  # the valid part stays visible


def test_monotonic_duration_and_reversal():
    raw = measured_duration(1_000_000, 4_000_000)
    assert raw.value == 3_000_000 and raw.unit == "work_ns"
    reversed_ = measured_duration(5, 1)
    assert reversed_.value is None and reversed_.reason == "monotonic_clock_reversed"


def test_service_millisecond_presentation_requires_observed_monotonic_nanoseconds():
    from ssgrep.accounting.ledger.contracts import Basis, Quantity
    from ssgrep.accounting.ledger.units import service_milliseconds

    measured = Quantity(1_999_999, "work_ns", Basis.OBSERVED, "monotonic_v1")
    assert service_milliseconds(measured) == Quantity(1, "work_ms", Basis.OBSERVED, "monotonic_v1")

    estimated = Quantity(1_999_999, "work_ns", Basis.ESTIMATED, "monotonic_v1")
    unavailable = service_milliseconds(estimated)
    assert unavailable.value is None
    assert unavailable.reason == "nanosecond_precision_unavailable"


@pytest.mark.parametrize("damage", ["complete-flag", "known-subtotal"])
def test_period_estimate_codec_rejects_inconsistent_completeness_metadata(damage: str):
    from ssgrep.accounting.ledger.ledger_codec import period_row_from_dict, period_row_to_dict
    from ssgrep.accounting.ledger.revisions import PeriodRow, unit_key

    valued = value_evidence(
        EvidenceBundle((candidate("codec"),), (debit("codec-cost", 640),)), METHOD
    )
    estimates = tuple(estimate.as_scenario_estimate() for estimate in rollup(valued.leaves, CHARS))
    row = PeriodRow(
        METHOD.hash,
        DAY,
        1,
        None,
        "original",
        False,
        True,
        valued.leaves,
        ((unit_key(CHARS), estimates),),
    )
    raw = period_row_to_dict(row)
    assert period_row_from_dict(raw) == row
    with pytest.raises(ValueError, match="expected an object for PeriodRow"):
        period_row_from_dict(cast(Any, []))
    damaged = deepcopy(raw)
    central = next(
        estimate for estimate in damaged["estimates"][0][1] if estimate["scenario"] == "central"
    )
    if damage == "complete-flag":
        central["complete"] = False
    else:
        central["known_net"] += 1

    with pytest.raises(ValueError, match="complete_net"):
        period_row_from_dict(damaged)


def test_merge_intervals_rejects_inverted_input_and_merges_touching():
    assert merge_intervals([(0, 5), (5, 9), (20, 25)]) == ((0, 9), (20, 25))
    with pytest.raises(ValueError):
        merge_intervals([(5, 1)])


# ---- tariff and provider cost ----------------------------------------------------

TARIFF = {
    "version": "2026-10-offline",
    "entries": [
        {
            "model": "m1",
            "category": "input_uncached",
            "currency": "USD",
            "micros_per_million_tokens": 3_000_000,
            "effective_from": "2026-01-01",
        },
        {
            "model": "m1",
            "category": "output",
            "currency": "USD",
            "micros_per_million_tokens": 15_000_000,
            "effective_from": "2026-01-01",
            "effective_to": "2026-12-01",
        },
    ],
}


def _usage(**kw):
    update = UsageUpdate("anthropic", "s", 1, **kw)
    return normalize_usage([update])["anthropic"]


def test_modeled_cost_uses_the_applicable_entry_with_integer_arithmetic():
    tariff = validate_tariff(TARIFF)
    cost = modeled_cost(
        tariff, _usage(input_tokens=2_000, output_tokens=1_000), "m1", "2026-11-01", "USD"
    )
    assert cost.value == 2_000 * 3 + 1_000 * 15  # micro-USD
    assert cost.unit == "micro_USD" and cost.method.startswith("tariff:2026-10-offline:")


def test_missing_tariff_or_missing_category_is_unknown_not_free():
    usage = _usage(input_tokens=5, output_tokens=5)
    assert modeled_cost(None, usage, "m1", "2026-11-01", "USD").reason == "no_tariff"
    tariff = validate_tariff(TARIFF)
    after_expiry = modeled_cost(tariff, usage, "m1", "2026-12-01", "USD")
    assert after_expiry.value is None and after_expiry.reason == "no_applicable_tariff:output"
    other_model = modeled_cost(tariff, usage, "zz", "2026-11-01", "USD")
    assert other_model.value is None
    other_currency = modeled_cost(tariff, usage, "m1", "2026-11-01", "EUR")
    assert other_currency.value is None


def test_provider_reported_zero_is_distinct_from_missing_and_from_modeled_unknown():
    zero = reported_cost(0, "USD")
    missing = reported_cost(None, "USD")
    assert zero.value == 0 and zero.basis is Basis.REPORTED
    assert missing.value is None and missing.basis is Basis.UNAVAILABLE
    modeled = modeled_cost(None, _usage(input_tokens=10), "m1", "2026-11-01", "USD")
    assert zero != modeled and modeled.value is None  # subscription zero never fills the model


def test_money_sums_refuse_currency_or_basis_mixing_and_propagate_unknown():
    usd, eur = reported_cost(5, "USD"), reported_cost(5, "EUR")
    with pytest.raises(ValueError):
        sum_money([usd, eur])
    tariff = validate_tariff(TARIFF)
    modeled = modeled_cost(tariff, _usage(input_tokens=1_000_000), "m1", "2026-11-01", "USD")
    with pytest.raises(ValueError):
        sum_money([usd, modeled])
    assert sum_money([usd, reported_cost(7, "USD")]).value == 12
    assert sum_money([usd, reported_cost(None, "USD")]).value is None


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda t: t["entries"][0].update(micros_per_million_tokens=1.5), "integer"),
        (lambda t: t["entries"][0].update(currency="usd"), "currency"),
        (lambda t: t["entries"][0].update(category="bogus"), "category"),
        (lambda t: t["entries"][0].update(effective_from="yesterday"), "ISO date"),
        (lambda t: t["entries"][1].update(effective_to="2025-01-01"), "after effective_from"),
        (lambda t: t.pop("version"), "version"),
        (lambda t: t["entries"].append(dict(t["entries"][0])), "overlapping"),
    ],
)
def test_invalid_tariffs_are_rejected(mutate, fragment):
    import copy

    raw = copy.deepcopy(TARIFF)
    mutate(raw)
    with pytest.raises(TariffError, match=fragment):
        validate_tariff(raw)


def test_tariff_identity_changes_with_content():
    import copy

    other = copy.deepcopy(TARIFF)
    assert isinstance(other["entries"], list)
    other["entries"][0]["micros_per_million_tokens"] = 4_000_000
    assert validate_tariff(TARIFF).identity != validate_tariff(other).identity


@pytest.mark.parametrize("intervals", [[(-1, 3)], [(9, 4)]])
def test_invalid_source_ranges_cannot_produce_context_volume(
    intervals: list[tuple[int, int]],
) -> None:
    with pytest.raises(ValueError):
        union_length(intervals)


def test_empty_aggregates_and_unknown_provider_categories_cannot_claim_zero() -> None:
    with pytest.raises(ValueError):
        sum_quantities([])
    with pytest.raises(ValueError):
        sum_money([])
    usage = normalize_usage([UsageUpdate("provider", "request", 0, input_tokens=7)])["provider"]
    assert usage.quantity("input_uncached").value == 7
    with pytest.raises(KeyError):
        usage.quantity("total")
    with pytest.raises(UnitMismatchError):
        sum_quantities([usage.quantity("input_uncached"), context_estimate(28)])
    with pytest.raises(TariffError):
        validate_tariff([])
    with pytest.raises(ValueError):
        reported_cost(0, "USD", trust="guessed")


"""F2: raw journal precision survives caps, uniqueness, bounds and presentation."""


def timings(durations):
    events = tuple(
        replace(event(f"execution-{i}", ts(12), phases=(Phase.COMPLETED,))[0], mono_ns=duration)
        for i, duration in enumerate(durations)
    )
    rows = tuple(journal_timings(events + events).values())
    return rows


def cost_rows(rows):
    return tuple(
        Debit(
            row.execution_id + ":work",
            TASK,
            DAY,
            "exposure",
            WORK,
            row.work,
            work_ns=row.work_ns,
            mapping_version="journal_active_proxy_ns_v2",
            assumptions=LOOKUP_WORK_ASSUMPTIONS,
        )
        for row in rows
    )


def work_estimates(debits, candidates=()):
    valued = value_evidence(
        EvidenceBundle(tuple(candidates), tuple(debits)), Methodology("literal-ns-fixture", (WORK,))
    )
    return rollup(valued.leaves, WORK)


def test_three_333400000ns_operations_debit_one_second_not_zero_in_all_bands():
    rows = timings((333_400_000,) * 3)
    assert [row.measured_service_ms.value for row in rows] == [333] * 3
    assert [row.work_ns for row in rows] == [333_400_000] * 3
    linked = LookupTiming(rows, True, "")
    assert linked.measured_service_ns.value == 1_000_200_000
    assert linked.measured_service_ms.value == 1000 and linked.work.value == 1
    costs = cost_rows(rows)
    for row in work_estimates(costs + costs):
        assert row.credit.value == 0 and row.debit.value == 1 and row.net.value == -1
        assert row.known_debits == 3 and row.complete
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (-2, -1)
    extra = replace(costs[0], debit_id="genuine-extra", work_ns=999_800_000)
    assert all(
        row.debit.value == 2 and row.net.value == -2 for row in work_estimates((*costs, extra))
    )
    assert bundle_from_dict(bundle_to_dict(EvidenceBundle(debits=costs))).debits == costs


def test_nanoseconds_cross_millisecond_boundary_only_after_sum():
    rows = timings((999_999, 1))
    assert [row.measured_service_ms.value for row in rows] == [0, 0]
    assert LookupTiming(rows, True, "").measured_service_ms.value == 1


def test_cap_is_per_operation_in_raw_ns_before_sum_and_seconds_conversion():
    rows = timings((300_000_999_999, 333_400_000, 333_400_000, 333_400_000))
    assert rows[0].measured_service_ns.value == 300_000_999_999
    assert rows[0].work_ns == 300_000_000_000
    assert LookupTiming(rows, True, "").measured_service_ms.value == 301001
    assert all(
        row.debit.value == 301 and row.net.value == -301 for row in work_estimates(cost_rows(rows))
    )


def test_source_native_ms_actor_unions_scale_exactly_when_composing_ns_debits():
    source = candidate(
        "source",
        immediate=(
            atom("a", 4, origin="a", t_start_ms=0, t_end_ms=600, actor="one"),
            atom("b", 4, origin="b", t_start_ms=400, t_end_ms=1000, actor="one"),
            atom("c", 4, origin="c", t_start_ms=0, t_end_ms=600, actor="two"),
        ),
    )
    for row in work_estimates(cost_rows(timings((333_400_000,) * 3)), (source,)):
        assert row.credit.value == 1 and row.debit.value == 1 and row.net.value == 0
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (-1, 1)


def test_missing_raw_ns_has_no_finite_lower_even_if_rounded_seconds_are_present():
    costs = cost_rows(timings((333_400_000,) * 3))
    legacy = Debit(
        "legacy-rounded",
        TASK,
        DAY,
        "exposure",
        WORK,
        Quantity(2, WORK.unit, Basis.ESTIMATED, WORK.method),
    )
    for row in work_estimates((*costs, legacy)):
        assert row.known_net == -1 and row.net.value is None
        assert row.bounds.net_lower.value is None and row.bounds.net_upper.value == -1
        assert row.debit_missing == (("legacy-rounded", "lookup_work_nanoseconds_unavailable"),)


def test_sealed_days_derive_milliseconds_from_raw_ns_across_day_boundaries(tmp_path):
    from ssgrep.accounting.views.history import SealedDay, _observations, seal_days
    from tests.accounting.ledger.reduce_builders import claude_evidence
    from tests.accounting.views.test_lookup_time import records, reduce_timing, timed_events

    evidence = claude_evidence(tmp_path, {"-work/consumer.jsonl": records()})
    template = reduce_timing(tmp_path, evidence, timed_events()).executions[0]
    exact = tuple(
        replace(
            template,
            execution_id=f"exact-{i}",
            day=f"2026-10-{11 + i}",
            measured_service_ns=Quantity.observed(value, "work_ns", "monotonic_v1"),
            measured_service_ms=Quantity.observed(0, "work_ms", "monotonic_v1"),
        )
        for i, value in enumerate((999_999, 1))
    )
    sealed, _ = seal_days(exact, before_day="2026-10-14", already={})
    restored = tuple(SealedDay.from_dict(day.to_dict()) for day in sealed.values())
    assert _observations(restored)["measured_service_ms"]["value"] == 1
    raw = restored[0].to_dict()
    raw.pop("measured_service_known_ns")
    raw["measured_service_known_ms"] = 0  # old rounded artifact, not exact ns evidence
    observed = _observations((SealedDay.from_dict(raw),))
    assert observed["measured_service_ms"]["value"] is None
    assert observed["measured_service_ms_known"]["value"] == 0


def test_published_amount_without_raw_ns_requires_explicit_precision_migration():
    raw = bundle_to_dict(EvidenceBundle(debits=cost_rows(timings((333_400_000,)))))
    published_fields = {
        "debit_id",
        "task",
        "day",
        "kind",
        "unit",
        "amount",
        "partial",
        "exposure_id",
        "chars",
        "accepted_day",
    }
    raw["debits"][0] = {
        key: value for key, value in raw["debits"][0].items() if key in published_fields
    }
    with pytest.raises(ValueError, match="legacy_valuation_migration_required"):
        bundle_from_dict(raw)

    from ssgrep.accounting.collect.replay import LedgerState
    from ssgrep.accounting.ledger.ledger_codec import ledger_state_from_dict, to_plain
    from ssgrep.accounting.ledger.units import UnitMismatchError

    current = Methodology("literal-ns-fixture", (WORK,))
    published = to_plain(LedgerState(current))
    published["methodology"].pop("valuation_version")  # actual 407e3c2 schema
    with pytest.raises(ValueError, match="legacy_valuation_migration_required"):
        ledger_state_from_dict(published)
    valued = value_evidence(EvidenceBundle(debits=cost_rows(timings((333_400_000,) * 3))), current)
    with pytest.raises(UnitMismatchError, match="time_precision_namespace_migration_required"):
        rollup((replace(valued.leaves[0], raw_time_unit=""),), WORK)


def test_closed_company_projection_declares_raw_ns_and_accepts_new_mapping_version():
    from ssgrep.accounting.collect.replay import Ledger
    from ssgrep.accounting.export import records as export_records, registry as export_registry
    from ssgrep.accounting.export.receiver import Key, Outcome, ReceiverState, apply
    from ssgrep.accounting.export.registry import Registry

    ledger = Ledger(Methodology("literal-ns-fixture", (WORK,)))
    ledger.ingest(EvidenceBundle(debits=cost_rows(timings((333_400_000,) * 3))))
    projected, omitted = export_records.project_candidates(
        ledger.rows(), key=bytes(range(32)), destination="test-only"
    )
    assert omitted == 0 and len(projected) == 1
    value = projected[0]
    assert value.core["known_costs"]["raw_unit"] == "nanoseconds"
    assert value.core["known_costs"]["debit_raw"] == 1_000_200_000
    for scenario in value.core["scenarios"].values():
        assert scenario["net"]["value"] == -1 and scenario["complete"]
        assert "journal_active_proxy_ns_v2" in scenario["bounds"]["assumptions"]
    planned = export_registry.plan_revisions(Registry(), projected)
    key = Key("authenticated-test-tenant", "authenticated-test-device", value.record_id)
    assert apply(ReceiverState(), key, planned[0].body)[1] is Outcome.APPLIED_NEW


def test_outward_bounds_do_not_round_a_nearly_two_second_debit_down_at_lower_net():
    source = candidate(
        "three-seconds", immediate=(atom("source", 4, t_start_ms=0, t_end_ms=3000, actor="one"),)
    )
    rows = work_estimates(cost_rows(timings((1_999_999_999,))), (source,))
    for row in rows:
        assert row.net.value == row.known_net == 2  # documented whole-unit point model
        assert (row.bounds.credit_lower.value, row.bounds.credit_upper.value) == (3, 3)
        assert (row.bounds.debit_lower.value, row.bounds.debit_upper.value) == (1, 2)
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (1, 2)


@pytest.mark.parametrize("millisecond_endpoints", [False, True])
def test_reversed_source_clock_is_unknown_credit_for_either_endpoint_precision(
    millisecond_endpoints: bool,
) -> None:
    if millisecond_endpoints:
        source_atom = atom("reversed", 4, t_start_ms=900, t_end_ms=100)
    else:
        source_atom = atom("reversed", 4, t_start=9, t_end=1)
    source = candidate("reversed-source", immediate=(source_atom,))
    rows = work_estimates(cost_rows(timings((1_000_000_000,) * 3)), (source,))

    assert all(row.net.value is None and not row.complete for row in rows)
    assert all(row.credit_missing == (("reversed", "source_clock_reversed"),) for row in rows)


@pytest.mark.parametrize(("raw_min", "raw_max"), [(-1, None), (3, 2)])
def test_debit_rejects_negative_or_inverted_raw_uncertainty_bounds(raw_min, raw_max):
    base = debit("bounded", 100)
    assert base.raw_min >= 0

    with pytest.raises(ValueError, match="invalid nonnegative debit bounds"):
        replace(base, raw_min=raw_min, raw_max=raw_max)


def test_generic_unavailable_debit_preserves_reason_without_turning_into_zero():
    reported = UnitSpec(WORK_SECONDS, "reported_v1")
    missing = Debit(
        "reported-missing",
        TASK,
        DAY,
        "exposure",
        reported,
        Quantity.unavailable(reported.unit, "reported_cost_missing"),
    )
    known = replace(
        missing,
        debit_id="reported-known",
        amount=Quantity(3, reported.unit, Basis.OBSERVED),
    )

    assert debit_raw(known, reported) == (3, "")
    assert debit_raw(missing, reported) == (None, "reported_cost_missing")


def test_explicit_missing_ns_bounds_round_outward_but_unbounded_debit_has_no_lower():
    source = candidate(
        "three-seconds", immediate=(atom("source", 4, t_start_ms=0, t_end_ms=3000, actor="one"),)
    )
    missing = Debit(
        "bounded-missing",
        TASK,
        DAY,
        "exposure",
        WORK,
        Quantity.unavailable(WORK.unit, "missing_duration"),
        raw_min=1,
        raw_max=1_999_999_999,
    )
    bounded = work_estimates((missing,), (source,))
    assert all(row.net.value is None and row.known_net == 3 for row in bounded)
    assert all(
        (row.bounds.net_lower.value, row.bounds.net_upper.value) == (1, 3) for row in bounded
    )
    unbounded = work_estimates((replace(missing, raw_max=None),), (source,))
    assert all(
        row.bounds.net_lower.value is None and row.bounds.net_upper.value == 3 for row in unbounded
    )


def test_character_proxy_bounds_enclose_subtoken_raw_endpoints():
    from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
    from tests.accounting.ledger.valuation_builders import CHARS

    source = candidate("fractional-character-proxy", immediate=(atom("source", 3),))
    cost = Debit(
        "one-character",
        TASK,
        DAY,
        "exposure",
        CHARS,
        Quantity(0, CHARS.unit, Basis.ESTIMATED, CHARS.method),
        chars=1,
    )
    value = value_evidence(EvidenceBundle((source,), (cost,)), Methodology("chars", (CHARS,)))
    for row in rollup(value.leaves, CHARS):
        assert row.net.value == 0
        assert (row.bounds.credit_lower.value, row.bounds.credit_upper.value) == (0, 1)
        assert (row.bounds.debit_lower.value, row.bounds.debit_upper.value) == (0, 1)
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (-1, 1)


def test_actual_native_reducer_keeps_three_fractional_lookup_debits_and_copy_replay(tmp_path):
    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.ledger.reduce import methodology_for, reduce_profile
    from tests.accounting.ledger.reduce_builders import (
        KEY,
        MCP,
        PROFILE,
        claude_evidence,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        policy,
        results_json,
    )

    native = [claude_prompt("consumer", "inspect history", uuid="precision-user", ts=ts(12, 9))]
    events = []
    for i in range(3):
        call_id = f"precise-call-{i}"
        native.extend(
            (
                claude_tool_use(
                    "consumer",
                    call_id,
                    MCP,
                    {"query": f"precision-{i}"},
                    message_id=f"precise-message-{i}",
                    ts=ts(12, 10, i),
                ),
                claude_tool_result("consumer", call_id, results_json(), ts=ts(12, 10, i, 2)),
            )
        )
        events.extend(
            replace(item, mono_ns=333_400_000) if item.phase is Phase.COMPLETED else item
            for item in event(f"precise-execution-{i}", ts(12, 10, i, 1), text=f"precision-{i}")
        )
    evidence = claude_evidence(
        tmp_path, {"-work/original.jsonl": native, "-copy/copied.jsonl": native}
    )
    result = reduce_profile(
        PROFILE,
        evidence,
        events,
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )
    copied = reduce_profile(
        PROFILE,
        evidence + evidence,
        events + events,
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )
    assert copied.executions == result.executions and copied.bundles == result.bundles
    costs = [cost for bundle in result.bundles for cost in bundle.debits if cost.unit == WORK]
    assert len(costs) == 3 and {cost.work_ns for cost in costs} == {333_400_000}
    value = value_evidence(
        EvidenceBundle(debits=tuple(costs)), methodology_for(FrequencyView.unknown())
    )
    rows = rollup((leaf for leaf in value.leaves if leaf.unit == WORK), WORK)
    assert all(row.net.value == -1 and row.debit.value == 1 for row in rows)
    assert all((row.bounds.net_lower.value, row.bounds.net_upper.value) == (-2, -1) for row in rows)


def test_public_receipt_keeps_exact_raw_ns_bound_inputs_and_declared_units(tmp_path):
    import os

    from ssgrep.accounting.collect.cycle_publish import receipt_document as stored_document
    from ssgrep.accounting.ledger.contracts import canonical_bytes
    from ssgrep.accounting.privacy import storage
    from ssgrep.accounting.views.records import build_receipt, read_receipt
    from ssgrep.cli.savings_json import receipt_document as public_document

    source = candidate(
        "three-seconds",
        immediate=(atom("source", 4, t_start_ms=0, t_end_ms=3000, actor="one"),),
        exposure_ids=("lookup",),
    )
    costs = tuple(
        replace(cost, exposure_id="lookup") for cost in cost_rows(timings((1_999_999_999,)))
    )
    method = Methodology("literal-ns-fixture", (WORK,))
    value = value_evidence(EvidenceBundle((source,), costs), method)
    record = build_receipt(source, value.allocations, costs, method)
    stored = stored_document(record)
    assert stored["raw_unit"] == "nanoseconds"
    assert (
        stored["debit_raw"] == stored["debit_raw_min"] == stored["debit_raw_max"] == 1_999_999_999
    )
    assert dict(stored["credit_raw"]) == {
        scenario: 3_000_000_000 for scenario in ("conservative", "central", "optimistic")
    }
    directory = storage.open_profile_dir("personal-local", "receipts")
    try:
        storage.atomic_write(directory, record.receipt_id + ".json", canonical_bytes(stored))
    finally:
        os.close(directory)
    view = read_receipt("personal-local", record.receipt_id)
    assert view is not None
    public = public_document(view, "personal-local")["data"]
    assert isinstance(public, dict)
    assert public["raw_unit"] == "nanoseconds" and public["debit_raw_max"] == 1_999_999_999
    assert public["credit_raw"]["conservative"] == 3_000_000_000
    for scenario in public["scenarios"]:
        assert scenario["net"]["value"] == 2
        assert (
            scenario["bounds"]["net_lower"]["value"],
            scenario["bounds"]["net_upper"]["value"],
        ) == (1, 2)


def test_complete_fractional_point_still_renders_its_distinct_outward_bounds():
    from ssgrep.cli.savings_text import Doc, render_estimate_bounds

    rows = tuple(
        row.as_scenario_estimate() for row in work_estimates(cost_rows(timings((333_400_000,) * 3)))
    )
    assert all(row.complete for row in rows)
    doc = Doc(80)
    render_estimate_bounds(doc, rows)
    text = " ".join(doc.text().plain.split())
    assert "-2 s" in text and "-1 s" in text
    assert "Lower bound" in text and "Upper bound" in text


def test_legacy_sealed_metadata_cannot_manufacture_zero_latency_from_absent_ns():
    from ssgrep.accounting.views.history import SealedDay, _observations

    old = {
        "day": "2026-10-12",
        "counts": {},
        "visible_known": 0,
        "visible_missing": 0,
        "rendered_known": 0,
        "rendered_missing": 0,
    }
    assert _observations((SealedDay.from_dict(old),))["measured_service_ms"]["value"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_daily_partial_source_ownership_keeps_raw_ns_and_receipt_nesting(reverse):
    from ssgrep.accounting.ledger.contracts import Reuse, Scenario
    from ssgrep.accounting.views.records import build_receipts

    relay = candidate(
        "relay",
        day="2026-11-01",
        reuse=Reuse.INSPECTED,
        explicit_relay=True,
        immediate=(),
        episode=(atom("relay-span", 7, t_start_ms=0, t_end_ms=1600, actor="shared"),),
    )
    acted = candidate(
        "acted",
        day="2026-11-02",
        immediate=(atom("acted-span", 3, t_start_ms=200, t_end_ms=1300, actor="shared"),),
    )
    candidates = (acted, relay) if reverse else (relay, acted)
    costs = (
        replace(cost_rows(timings((333_400_000,)))[0], day=relay.day, debit_id="early"),
        replace(cost_rows(timings((666_800_000,)))[0], day=acted.day, debit_id="late"),
    )
    method = Methodology("daily-ns-ownership", (WORK,))
    valued = value_evidence(EvidenceBundle(candidates, costs), method)
    leaves = {leaf.day: leaf for leaf in valued.leaves}
    assert [row.credit_raw for row in leaves[relay.day].scenarios] == [0, 0, 500_000_000]
    assert [row.credit_raw for row in leaves[acted.day].scenarios] == [1_100_000_000] * 3
    assert (leaves[relay.day].debit_raw, leaves[acted.day].debit_raw) == (
        333_400_000,
        666_800_000,
    )
    assert [
        (row.credit.value, row.debit.value, row.net.value) for row in rollup(valued.leaves, WORK)
    ] == [(1, 1, 0)] * 3
    receipts = {
        record.receipt.candidate_fingerprint: record
        for record in build_receipts(candidates, valued.allocations, costs, method).values()
    }
    assert dict(receipts["relay"].credit_raw) == {
        "conservative": 0,
        "central": 0,
        "optimistic": 500_000_000,
    }
    assert dict(receipts["acted"].credit_raw) == {
        scenario.value: 1_100_000_000 for scenario in Scenario
    }
    for leaf in leaves.values():
        for row in leaf.scenarios:
            assert row.credit_raw == sum(
                allocation.credit_raw
                for allocation in valued.allocations
                if allocation.day == leaf.day and allocation.scenario is row.scenario
            )


"""D15/V07 independent interval arithmetic and unknown eligibility regressions."""


def estimates(candidates=(), debits=()):
    valued = value_evidence(EvidenceBundle(tuple(candidates), tuple(debits)), METHOD)
    return rollup(valued.leaves, CHARS), valued


@pytest.mark.parametrize(
    "reason", ["rarity_unknown", "source_missing", "prior_context_unavailable"]
)
def test_unknown_gate_without_source_allocation_never_becomes_complete_zero(reason):
    cand = candidate(
        "unknown",
        immediate=(),
        eligibility=Eligibility.UNAVAILABLE,
        eligibility_reasons=(reason,),
        provenance=(("gate", reason),),
    )
    rows, valued = estimates([cand], [debit("visible", 40)])
    for row in rows:
        assert row.credit.value == 0 and row.known_net == -10
        assert row.net.value is None and not row.complete
        assert row.credit_missing == (("candidate:unknown", reason),)
        assert row.bounds.net_lower.value == -10
        assert row.bounds.net_upper.value is None
        assert row.as_scenario_estimate().known_net == -10
    record = build_receipt(cand, valued.allocations, [debit("visible", 40)], METHOD)
    wire = json.loads(record.canonical_bytes())
    assert wire["receipt"]["eligibility"] == "unavailable"
    assert wire["receipt"]["eligibility_reasons"] == [reason]
    assert all(s["net"]["value"] is None for s in wire["receipt"]["scenarios"])


@pytest.mark.parametrize("reuse", [Reuse.NONE_OBSERVED, Reuse.CONTRADICTED])
def test_known_rejected_reuse_is_zero_credit_with_negative_complete_net(reuse):
    rows, _ = estimates([candidate("rejected", reuse=reuse)], [debit("visible", 40)])
    assert all(r.credit.value == 0 and r.net.value == -10 for r in rows)
    assert all(r.bounds.net_lower.value == r.bounds.net_upper.value == -10 for r in rows)


def test_unknown_unbounded_debit_makes_lower_net_unbounded_not_known_subtotal():
    rows, _ = estimates([candidate("known")], [debit("visible", 40), debit("unknown", None)])
    for row in rows:
        assert row.known_net == 90 and row.net.value is None
        assert row.bounds.credit_lower.value == row.bounds.credit_upper.value == 100
        assert row.bounds.debit_lower.value == 10 and row.bounds.debit_upper.value is None
        assert row.bounds.net_lower.value is None and row.bounds.net_upper.value == 90


def test_bounded_missing_debit_uses_cmin_minus_dmax_and_cmax_minus_dmin():
    cost = replace(debit("bounded", None), raw_min=8, raw_max=32)
    rows, _ = estimates([candidate("known")], [debit("visible", 40), cost])
    assert all(r.bounds.net_lower.value == 82 and r.bounds.net_upper.value == 88 for r in rows)
    assert all(r.net.value is None and r.known_net == 90 for r in rows)


def test_nested_bands_use_same_debits_and_preserve_negative_bounds():
    cand = candidate(
        "known",
        immediate=(atom("low", 4),),
        discovery=(atom("mid", 8, start=4),),
        episode=(atom("high", 12, start=12),),
    )
    rows, _ = estimates([cand], [debit("same", 40), debit("same", 40)])
    assert [r.net.value for r in rows] == [-9, -7, -4]
    assert [r.bounds.net_lower.value for r in rows] == [-9, -7, -4]
    assert [r.debit.value for r in rows] == [10, 10, 10]


@pytest.mark.parametrize("gate", ["rarity", "source", "prior"])
def test_actual_reducer_unknown_gates_survive_preallocation(tmp_path, gate):
    kwargs = {"view": FrequencyView.unknown()} if gate == "rarity" else {}
    if gate == "source":
        kwargs["src"] = []
    records = evidence_consumer()
    if gate == "prior":
        records = records[1:]
    reduction = reduce_scene(tmp_path, records, **kwargs)
    assert reduction.candidates
    cand = reduction.candidates[0].candidate
    if gate in {"rarity", "prior"}:
        assert cand.eligibility is Eligibility.UNAVAILABLE
    valued = value_evidence(
        reduction.bundles[0], methodology_for(kwargs.get("view", FrequencyView.unknown()))
    )
    rows = rollup([leaf for leaf in valued.leaves if leaf.unit == CHARS], CHARS)
    assert all(r.credit_missing and r.net.value is None for r in rows)


def test_mixed_complete_and_incomplete_days_sum_raw_bounds_before_rounding():
    first = candidate("first", immediate=(atom("a", 403),))
    second = candidate(
        "second",
        day="2026-11-02",
        immediate=(),
        eligibility=Eligibility.UNAVAILABLE,
        eligibility_reasons=("source_missing",),
    )
    rows, _ = estimates(
        [first, second],
        [debit("a", 41), replace(debit("b", None, day="2026-11-02"), raw_min=3, raw_max=7)],
    )
    # Cmin=403//4=100, Dmax=(41+7)//4=12; missing credit unbounded.
    assert all(r.bounds.net_lower.value == 88 and r.bounds.net_upper.value is None for r in rows)
    assert all(r.net.value is None and r.known_net == 90 for r in rows)


def test_absence_of_valuation_evidence_is_not_a_complete_zero():
    rows, _ = estimates()
    assert all(r.net.value is None and not r.complete for r in rows)


@pytest.mark.parametrize(
    "foreign",
    [
        UnitSpec(CHARS.unit, "exact:offline", tokenizer="offline"),
        UnitSpec("micro_EUR", "tariff_v1", currency="EUR"),
    ],
)
def test_incomparable_method_or_currency_never_enters_same_net(foreign):
    wrong = Debit(
        "wrong",
        TASK,
        DAY,
        "exposure",
        foreign,
        Quantity(7, foreign.unit, Basis.ESTIMATED, foreign.method),
    )
    with pytest.raises(UnitMismatchError):
        value_evidence(EvidenceBundle(debits=(wrong,)), METHOD)


def test_measured_empty_lookup_is_known_negative_cost_not_unknown_credit():
    rows, _ = estimates(debits=[debit("measured-empty", 40)])
    assert all(r.complete and r.net.value == -10 for r in rows)
    zero, _ = estimates(debits=[debit("observed-zero-chars", 0)])
    assert all(r.complete and r.net.value == 0 and r.known_debits == 1 for r in zero)


def test_capped_gap_source_and_lookup_ns_round_only_after_unique_debit_sum():
    work = UnitSpec("work_seconds", "capped_gap_v1")
    meth = Methodology("time", (work,))
    cand = candidate("time", immediate=(atom("source", 40, t_start=0, t_end=10, actor="src"),))
    costs = tuple(
        Debit(
            f"execution-{i}:work",
            TASK,
            DAY,
            "exposure",
            work,
            Quantity(1, work.unit, Basis.ESTIMATED, work.method),
            work_ns=1_600_000_000,
            mapping_version="journal_active_proxy_ns_v2",
            assumptions=("active_actor_proxy",),
        )
        for i in range(2)
    )
    valued = value_evidence(EvidenceBundle((cand,), costs + costs), meth)
    rows = rollup(valued.leaves, work)
    assert all(r.credit.value == 10 and r.debit.value == 3 and r.net.value == 7 for r in rows)
    assert all(r.known_debits == 2 and "active_actor_proxy" in r.bounds.assumptions for r in rows)


def test_source_work_unions_same_actor_but_sums_parallel_actors():
    work = UnitSpec("work_seconds", "capped_gap_v1")
    meth = Methodology("parallel", (work,))
    for second_actor, expected in (("actor-a", 10), ("actor-b", 20)):
        cand = candidate(
            "parallel",
            immediate=(
                atom("a", 40, origin="event-a", t_start=0, t_end=10, actor="actor-a"),
                atom("b", 40, origin="event-b", t_start=0, t_end=10, actor=second_actor),
            ),
        )
        valued = value_evidence(EvidenceBundle((cand,)), meth)
        assert all(r.credit.value == expected for r in rollup(valued.leaves, work))


def test_wall_gap_source_is_not_combined_with_exact_duration_method():
    exact = UnitSpec("work_seconds", "monotonic_v1")
    meth = Methodology("exact", (exact,))
    cand = candidate("wall-gap", immediate=(atom("source", 40, t_start=0, t_end=10),))
    valued = value_evidence(EvidenceBundle((cand,)), meth)
    assert all(
        r.net.value is None and r.credit_missing == (("source", "source_work_method_incompatible"),)
        for r in rollup(valued.leaves, exact)
    )


def test_source_subsecond_gaps_round_once_after_actor_interval_union():
    work = UnitSpec("work_seconds", "capped_gap_v1")
    meth = Methodology("source-ms", (work,))
    cand = candidate(
        "fractional",
        immediate=(
            atom(
                "a",
                40,
                origin="event-a",
                t_start=0,
                t_end=0,
                t_start_ms=0,
                t_end_ms=600,
                actor="actor-a",
            ),
            atom(
                "b",
                40,
                origin="event-b",
                t_start=1,
                t_end=1,
                t_start_ms=1000,
                t_end_ms=1600,
                actor="actor-a",
            ),
        ),
    )
    valued = value_evidence(EvidenceBundle((cand,)), meth)
    assert all(r.credit.value == 1 and r.net.value == 1 for r in rollup(valued.leaves, work))


def test_partial_source_millisecond_endpoints_remain_unavailable():
    work = UnitSpec("work_seconds", "capped_gap_v1")
    meth = Methodology("source-ms", (work,))
    cand = candidate(
        "partial-ms", immediate=(atom("source", 40, t_start=0, t_end=1, t_start_ms=0),)
    )
    valued = value_evidence(EvidenceBundle((cand,)), meth)
    assert all(
        r.credit_missing == (("source", "source_seconds_unavailable"),) and r.net.value is None
        for r in rollup(valued.leaves, work)
    )


@pytest.mark.parametrize(
    "lower_bound,complete",
    [
        (0, (False, False, False)),
        (4000, (False, False, True)),
        (6000, (False, True, True)),
        (8000, (True, True, True)),
    ],
)
def test_rarity_superset_lower_bound_only_proves_supported_bands(lower_bound, complete):
    cand = candidate(
        "bound",
        rarity_lower_bound_milli=lower_bound,
        discovery=(atom("discovery", 800),),
        episode=(atom("episode", 1200),),
    )
    rows, valued = estimates([cand], [debit("same", 40)])
    assert tuple(r.complete for r in rows) == complete
    for row, proven in zip(rows, complete, strict=True):
        assert row.debit.value == 10
        assert dict(row.eligibility)["bound"] == ("supported" if proven else "unavailable")
        if not proven:
            assert row.net.value is None and row.credit_missing == (
                ("candidate:bound", "rarity_threshold_unproven"),
            )
            assert row.bounds.net_lower.value == -10 and row.bounds.net_upper.value is None
    record = build_receipt(cand, valued.allocations, [debit("same", 40)], METHOD)
    assert tuple(s.complete for s in record.receipt.scenarios) == complete


"""Note leverage, payback coverage, and overhead purpose tagging."""


NOTE_ORIGIN = "note-event-1"


def cost(value):
    if value is None:
        return Quantity.unavailable(CONTEXT_TOKENS, "creation_cost_unobserved")
    return Quantity(value, CONTEXT_TOKENS, Basis.ESTIMATED, "chars_div4_v1")


def note_consumer(cid, task, chars=400, start=0, origin=NOTE_ORIGIN):
    return candidate(
        cid, task=task, immediate=(atom(f"{cid}-a", chars, start=start, origin=origin),)
    )


def test_overlapping_consumers_of_a_note_are_deduplicated_per_task_and_summed_across_tasks():
    t1, t2 = TaskId("r1", "t1", "main"), TaskId("r2", "t1", "main")
    consumers = [
        note_consumer("c1", t1, 400, 0),
        note_consumer("c2", t1, 400, 200),  # overlaps c1 inside one task: union [0,600)
        note_consumer("c3", t2, 400, 0),  # a different consumer task: credited again
    ]
    result = note_leverage(
        "n1",
        frozenset({NOTE_ORIGIN}),
        consumers,
        [NoteInvestment("n1", "creation", cost(100))],
        METHOD,
        CHARS,
    )
    cons = next(s for s in result.scenarios if s.scenario is Scenario.CONSERVATIVE)
    assert cons.credit.value == (600 + 400) // 4
    assert cons.consumers == 2
    assert result.known_cost == 100 and result.payback_complete
    assert result.payback_permille(Scenario.CONSERVATIVE) == 250 * 1000 // 100


def test_atoms_from_other_sources_do_not_count_toward_this_notes_leverage():
    task = TaskId("r1", "t1", "main")
    consumers = [
        note_consumer("c1", task, 400, 0),
        note_consumer("c2", task, 800, 0, origin="elsewhere"),
    ]
    result = note_leverage(
        "n1",
        frozenset({NOTE_ORIGIN}),
        consumers,
        [NoteInvestment("n1", "creation", cost(10))],
        METHOD,
        CHARS,
    )
    assert result.scenarios[0].credit.value == 100


def test_partly_unavailable_creation_cost_shows_known_investment_and_denies_payback():
    task = TaskId("r1", "t1", "main")
    investments = [
        NoteInvestment("n1", "creation", cost(100)),
        NoteInvestment("n1", "index", cost(None)),
    ]
    result = note_leverage(
        "n1", frozenset({NOTE_ORIGIN}), [note_consumer("c1", task)], investments, METHOD, CHARS
    )
    assert result.known_cost == 100
    assert (result.known_components, result.cost_components) == (1, 2)
    assert not result.payback_complete
    assert result.missing_reasons == ("creation_cost_unobserved",)
    assert all(result.payback_permille(s.scenario) is None for s in result.scenarios)


def test_no_observed_creation_cost_has_unknown_known_cost_not_zero():
    task = TaskId("r1", "t1", "main")
    result = note_leverage(
        "n1", frozenset({NOTE_ORIGIN}), [note_consumer("c1", task)], [], METHOD, CHARS
    )
    assert result.known_cost is None and result.cost_components == 0
    assert not result.payback_complete
    assert result.missing_reasons == ("note_creation_cost_unobserved",)


def test_unavailable_source_atom_blocks_payback_even_with_complete_cost():
    from ssgrep.accounting.ledger.valuation_inputs import AtomStatus

    task = TaskId("r1", "t1", "main")
    gone = candidate(
        "c1",
        task=task,
        immediate=(atom("gone", 400, origin=NOTE_ORIGIN, status=AtomStatus.UNAVAILABLE),),
    )
    result = note_leverage(
        "n1",
        frozenset({NOTE_ORIGIN}),
        [gone],
        [NoteInvestment("n1", "creation", cost(50))],
        METHOD,
        CHARS,
    )
    assert result.payback_complete and result.scenarios[0].credit_missing == 1
    assert result.payback_permille(Scenario.CONSERVATIVE) is None


# ---- overhead ---------------------------------------------------------------------


def test_protocol_purpose_requires_explicit_tag_or_sequence_evidence():
    tagged = OverheadEvent("e1", "protocol", cost(30), explicit_tag="protocol")
    sequenced = OverheadEvent("e2", "protocol", cost(30), sequence_evidence=True)
    assert purpose_of(tagged) is Purpose.PROTOCOL
    assert purpose_of(sequenced) is Purpose.PROTOCOL
    assert purpose_of(OverheadEvent("e3", "protocol", cost(30))) is Purpose.UNKNOWN
    # an explicit tag on a non-protocol kind does not manufacture protocol purpose
    assert (
        purpose_of(OverheadEvent("e4", "index", cost(5), explicit_tag="protocol"))
        is Purpose.UNKNOWN
    )


def test_overhead_views_group_by_kind_and_purpose_and_keep_missing_visible():
    events = [
        OverheadEvent("e1", "protocol", cost(30), explicit_tag="protocol"),
        OverheadEvent("e2", "protocol", cost(12)),  # purpose unknown stays a cost
        OverheadEvent("e3", "protocol", cost(None)),
        OverheadEvent("e4", "index", cost(200)),
        OverheadEvent("e5", "collector", cost(7)),
    ]
    views = {(v.kind, v.purpose): v for v in overhead_views(events)}
    assert views[("protocol", Purpose.PROTOCOL)].known == 30
    unknown = views[("protocol", Purpose.UNKNOWN)]
    assert (unknown.known, unknown.events, unknown.missing) == (12, 2, 1)
    assert views[("index", Purpose.UNKNOWN)].known == 200
    assert views[("collector", Purpose.UNKNOWN)].known == 7
    # input order does not matter
    assert overhead_views(list(reversed(events))) == overhead_views(events)
