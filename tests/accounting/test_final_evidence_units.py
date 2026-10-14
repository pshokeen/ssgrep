"""Incomplete costs stay unknown across real bundle and offline-tokenizer boundaries."""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.plain_codec import from_plain, to_plain
from ssgrep.accounting.tariff import TariffError, reported_cost, sum_money, validate_tariff
from ssgrep.accounting.units import (
    CONTEXT_TOKENS,
    WORK_SECONDS,
    TokenizerPin,
    UnitMismatchError,
    UnitSpec,
    UsageUpdate,
    context_estimate,
    normalize_usage,
    resolve_tokenizer,
    sum_quantities,
    union_length,
)
from ssgrep.accounting.valuation import Leaf, rollup, value_evidence
from ssgrep.accounting.valuation_credit import InvariantError
from ssgrep.accounting.valuation_inputs import EvidenceBundle, Methodology
from tests.accounting.valuation_builders import CHARS, METHOD, atom, candidate, debit


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
