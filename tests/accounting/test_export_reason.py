"""Quantity reasons leave the machine only as members of a closed vocabulary."""

from __future__ import annotations

import pytest

from ssgrep.accounting.export import REASON_CODES, safe_reason


@pytest.mark.parametrize("code", sorted(REASON_CODES))
def test_every_known_code_passes_unchanged(code: str) -> None:
    assert safe_reason(code) == code
    assert safe_reason(f"incomplete:{code}") == f"incomplete:{code}"


@pytest.mark.parametrize(
    "shipped",
    [
        "downstream_exposure_unobserved",
        "mixed_shell_output_unattributable",
        "lookup_work_seconds_not_modeled",
        "output_shared_by_several_executions",
        "no_runtime_exposure_joined",
        "no_render_event",
        "exact_tokens_unavailable",
        "source_seconds_unavailable",
        "debit_chars_unavailable",
        "conflicting_duplicate_debit",
        "unit_not_in_period",
        "no_closed_periods",
        "history_unreadable",
        "malformed_quantity",
        "no_tariff",
        "provider_cost_missing",
        "monotonic_clock_reversed",
        "no_dependency_evidence",
        "invalid_timing_in_gaps",
        "tokenizer_file_missing",
        "tokenizer_file_digest_mismatch",
        "visible_bytes_missing_for_some_executions",
        "rendered_bytes_missing_for_some_executions",
        "not_estimated",
        "scenario_missing",
    ],
)
def test_reasons_the_reducers_actually_produce_are_in_the_vocabulary(shipped: str) -> None:
    assert safe_reason(shipped) == shipped


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("no_applicable_tariff:input_uncached", "no_applicable_tariff"),
        ("tokenizer_load_failed:ValueError", "tokenizer_load_failed"),
        ("tokenizer_library_not_installed:tokenizers", "tokenizer_library_not_installed"),
        ("tokenizer_version_mismatch:0.1!=0.2", "tokenizer_version_mismatch"),
        ("unsupported_tokenizer_library:mylib", "unsupported_tokenizer_library"),
    ],
)
def test_known_codes_with_free_value_suffixes_drop_the_suffix(raw: str, expected: str) -> None:
    assert safe_reason(raw) == expected


@pytest.mark.parametrize(
    "free",
    [
        "sk-canary-7f3a91c2",
        "deadbeef" * 5,
        "Secret Reason! /home/u",
        "incomplete:sk-canary-7f3a91c2",
        "no_tariff:sk-canary",  # only suffix-family codes may carry a suffix
        "",
        None,
        7,
    ],
)
def test_anything_outside_the_vocabulary_collapses_to_unspecified(free: object) -> None:
    out = safe_reason(free)
    assert "canary" not in out and "deadbeef" not in out and "home" not in out
    assert out == "unspecified" or out in REASON_CODES or out.startswith("incomplete:")


def test_incomplete_members_are_normalized_sorted_and_deduplicated() -> None:
    known = (
        "incomplete:source_seconds_unavailable,exact_tokens_unavailable,source_seconds_unavailable"
    )
    assert safe_reason(known) == "incomplete:exact_tokens_unavailable,source_seconds_unavailable"
    suffixed = "incomplete:no_applicable_tariff:output,no_applicable_tariff:input_uncached"
    assert safe_reason(suffixed) == "incomplete:no_applicable_tariff"


def test_one_unknown_member_collapses_the_whole_incomplete_reason() -> None:
    mixed = "incomplete:source_seconds_unavailable,sk-canary-1,exact_tokens_unavailable"
    assert safe_reason(mixed) == "unspecified"
    assert safe_reason("incomplete:") == "unspecified"


@pytest.mark.parametrize("bad", ["sk-CANARY-1234", "2026-13-40T00:00:00Z", "x", "2026-10-12"])
def test_unix_nano_errors_never_quote_the_value(bad: str) -> None:
    from ssgrep.accounting.export import unix_nano

    with pytest.raises(ValueError, match="invalid_timestamp") as caught:
        unix_nano(bad)
    assert bad not in str(caught.value) and caught.value.__suppress_context__
