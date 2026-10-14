"""Pure retry/backoff policy and outcome classification for organization export.

No I/O, no clock, no global state: callers inject ``rand`` and persist the result.

Backoff: ``attempt`` is 1-based. The un-jittered delay is ``2 ** (attempt - 1)`` seconds
capped at ``MAX_BACKOFF_SECONDS`` (300). Jitter is "full-ish": the delay is drawn uniformly
from ``[delay / 2, delay]`` so a fleet never retries in lockstep while the growth stays
monotone in expectation and bounded below by half the cap. The result is a whole number of
seconds, at least 1. ``Retry-After`` is honoured as a *floor* (``max(jittered, retry_after)``),
so a receiver can slow a client down but never speed it up; the final delay is clamped to
``MAX_DELAY_SECONDS`` (3600) so a hostile header cannot park a destination for days.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssgrep.accounting.export_http import Outcome

BASE_SECONDS = 1
MAX_BACKOFF_SECONDS = 300
MAX_DELAY_SECONDS = 3600
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
BLOCKING_STATUSES = frozenset({401, 403})
RETRYABLE_KINDS = frozenset({"network_error", "timeout"})

ACK_CANDIDATE = "ack_candidate"
RETRY = "retry"
BLOCKED = "blocked"
PERMANENT = "permanent"
REFUSED = "refused"


def backoff_seconds(
    attempt: int, retry_after: int | None, *, rand: Callable[[], float] = random.random
) -> int:
    """Seconds to wait before retry number ``attempt`` (1-based); see the module docstring."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be an integer >= 1")
    exponent = min(attempt - 1, 20)  # 2**20 s is far beyond the cap; avoids huge shifts
    delay = min(BASE_SECONDS << exponent, MAX_BACKOFF_SECONDS)
    fraction = min(max(rand(), 0.0), 1.0)
    low = delay / 2
    jittered = int(low + (delay - low) * fraction)
    result = max(jittered, 1)
    if retry_after is not None and retry_after > 0:
        result = max(result, retry_after)
    return min(result, MAX_DELAY_SECONDS)


def is_retryable(kind: str, status: int | None) -> bool:
    """True for transient failures: disconnects, timeouts, and HTTP 429/502/503/504."""
    if kind in RETRYABLE_KINDS:
        return True
    return kind == "response" and status in RETRYABLE_STATUSES


def classify(outcome: Outcome) -> str:
    """One of ``ack_candidate | retry | blocked | permanent | refused``.

    ``ack_candidate``: HTTP 200 (the body still has to be parsed for partial success).
    ``retry``: transient, same immutable batch. ``blocked``: 401/403, the destination stays
    blocked until credentials/config change. ``permanent``: other 4xx/5xx, quarantine.
    ``refused``: local policy/TLS/redirect/size/format failures that need a config change.
    """
    if outcome.kind == "response":
        status = outcome.status
        if status == 200:
            return ACK_CANDIDATE
        if status in BLOCKING_STATUSES:
            return BLOCKED
        if is_retryable(outcome.kind, status):
            return RETRY
        return PERMANENT
    if is_retryable(outcome.kind, outcome.status):
        return RETRY
    return REFUSED
