"""Unit semantics: chars_div4_v1, provider usage, tokenizer, time, tariffs."""

from __future__ import annotations

import hashlib
import socket
from pathlib import Path

import pytest

from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.tariff import (
    TariffError,
    modeled_cost,
    reported_cost,
    sum_money,
    validate_tariff,
)
from ssgrep.accounting.timeunits import (
    ActorInterval,
    capped_gap_v1,
    measured_duration,
    merge_intervals,
    work_time,
)
from ssgrep.accounting.units import (
    CHARS_DIV4_V1,
    CONTEXT_TOKENS,
    TokenizerPin,
    UnitMismatchError,
    UsageUpdate,
    chars_div4_v1,
    context_estimate,
    merged_span_chars,
    normalize_usage,
    resolve_tokenizer,
    sum_quantities,
)

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


def test_capped_gap_cannot_be_mixed_with_exact_tool_durations():
    exact = work_time([ActorInterval("a", 0, 5_000)]).work_seconds
    capped = capped_gap_v1([0, 5_000]).quantity
    assert exact.method != capped.method
    with pytest.raises(UnitMismatchError):
        sum_quantities([exact, capped])


def test_monotonic_duration_and_reversal():
    assert measured_duration(1_000_000, 4_000_000).value == 3
    reversed_ = measured_duration(5, 1)
    assert reversed_.value is None and reversed_.reason == "monotonic_clock_reversed"


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
