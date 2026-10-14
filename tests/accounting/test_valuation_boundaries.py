"""Public valuation boundaries: subunit rounding, tariff dates, and incompatible snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.tariff import TariffError, modeled_cost, validate_tariff
from ssgrep.accounting.units import WORK_SECONDS, UnitSpec, UsageUpdate, normalize_usage
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import Debit, EvidenceBundle, Methodology
from tests.accounting.test_units import TARIFF
from tests.accounting.valuation_builders import CHARS, DAY, METHOD, TASK, debit


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
