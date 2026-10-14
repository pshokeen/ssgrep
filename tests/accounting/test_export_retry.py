"""Backoff and outcome classification are pure; every property is checked on real values."""

from __future__ import annotations

import pytest

from ssgrep.accounting import export_retry as retry
from ssgrep.accounting.export_http import Outcome


def _always(value: float):
    return lambda: value


def test_growth_is_monotone_exponential_with_full_jitter_top():
    delays = [retry.backoff_seconds(n, None, rand=_always(1.0)) for n in range(1, 12)]
    assert delays[:9] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert delays == sorted(delays)


def test_cap_is_five_minutes_even_for_absurd_attempts():
    for attempt in (10, 11, 50, 10_000):
        assert retry.backoff_seconds(attempt, None, rand=_always(1.0)) == 300


def test_jitter_stays_within_half_and_full_delay():
    for attempt, delay in ((3, 4), (6, 32), (9, 256), (12, 300)):
        low = retry.backoff_seconds(attempt, None, rand=_always(0.0))
        high = retry.backoff_seconds(attempt, None, rand=_always(1.0))
        mid = retry.backoff_seconds(attempt, None, rand=_always(0.5))
        assert (low, high) == (delay // 2, delay)
        assert low <= mid <= high


def test_first_attempt_is_at_least_one_second():
    assert retry.backoff_seconds(1, None, rand=_always(0.0)) == 1
    assert retry.backoff_seconds(2, None, rand=_always(0.0)) == 1


def test_out_of_range_random_values_cannot_escape_the_bounds():
    assert retry.backoff_seconds(4, None, rand=_always(7.5)) == 8
    assert retry.backoff_seconds(4, None, rand=_always(-3.0)) == 4


def test_retry_after_is_a_floor_not_a_replacement():
    assert retry.backoff_seconds(1, 30, rand=_always(1.0)) == 30
    assert retry.backoff_seconds(8, 5, rand=_always(1.0)) == 128  # computed delay is larger
    assert retry.backoff_seconds(2, 0, rand=_always(1.0)) == 2


def test_final_delay_is_clamped_to_one_hour():
    assert retry.backoff_seconds(1, 86_400, rand=_always(0.0)) == 3600
    assert retry.backoff_seconds(1, 3600, rand=_always(0.0)) == 3600


@pytest.mark.parametrize("attempt", [0, -1, -100])
def test_non_positive_attempt_is_rejected(attempt):
    with pytest.raises(ValueError):
        retry.backoff_seconds(attempt, None)


def test_default_randomness_stays_in_bounds():
    for _ in range(200):
        assert 4 <= retry.backoff_seconds(4, None) <= 8


def _response(status: int) -> Outcome:
    return Outcome("response", status, None, None, "x")


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (_response(200), "ack_candidate"),
        (Outcome("network_error"), "retry"),
        (Outcome("timeout"), "retry"),
        (_response(429), "retry"),
        (_response(502), "retry"),
        (_response(503), "retry"),
        (_response(504), "retry"),
        (_response(401), "blocked"),
        (_response(403), "blocked"),
        (_response(400), "permanent"),
        (_response(404), "permanent"),
        (_response(413), "permanent"),
        (_response(500), "permanent"),
        (_response(501), "permanent"),
        (Outcome("tls_error"), "refused"),
        (Outcome("redirect_refused", 307), "refused"),
        (Outcome("oversize_response"), "refused"),
        (Outcome("malformed_response"), "refused"),
        (Outcome("transport_refused"), "refused"),
    ],
)
def test_classification_table(outcome, expected):
    assert retry.classify(outcome) == expected


def test_is_retryable_agrees_with_classification_for_every_status():
    for status in range(100, 600):
        outcome = _response(status)
        assert retry.is_retryable("response", status) == (retry.classify(outcome) == "retry")
    assert not retry.is_retryable("tls_error", None)
    assert retry.is_retryable("timeout", None)
