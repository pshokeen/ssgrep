"""D15/V07 independent interval arithmetic and unknown eligibility regressions."""

import json
from dataclasses import replace

import pytest

from ssgrep.accounting.contracts import Basis, Quantity, Reuse
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.receipts import build_receipt
from ssgrep.accounting.reduce import methodology_for
from ssgrep.accounting.units import UnitMismatchError, UnitSpec
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import Debit, Eligibility, EvidenceBundle, Methodology
from tests.accounting.test_reduce_evidence import consumer, reduce_scene
from tests.accounting.valuation_builders import CHARS, DAY, METHOD, TASK, atom, candidate, debit


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
    records = consumer()
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
