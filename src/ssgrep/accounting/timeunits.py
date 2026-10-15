"""Time units: monotonic durations, per-actor work vs elapsed, ``capped_gap_v1``.

Monotonic durations retain integer nanoseconds. Native transcript intervals
retain their declared millisecond precision. Derived units floor only after
compatible raw quantities are summed; invalid timing remains unknown. Exact
tool-duration sums and capped-gap estimates remain different methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.units import WORK_SECONDS, union_length

WORK_MS = "work_ms"
WORK_NS = "work_ns"
NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
MONOTONIC_V1 = "monotonic_v1"
CAPPED_GAP_V1 = "capped_gap_v1"
TRANSCRIPT_INTERVALS_V1 = "transcript_intervals_v1"
GAP_CAP_MS = 300_000
GAP_CAP_NS = GAP_CAP_MS * NS_PER_MS
HUMAN_TIME = "human_seconds"


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Sorted disjoint union of half-open integer intervals (empty ones dropped)."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError(f"invalid interval ({start}, {end})")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def subtract_intervals(
    intervals: Sequence[tuple[int, int]], removed: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """``intervals`` minus ``removed`` (both merged on entry)."""
    out: list[tuple[int, int]] = []
    holes = merge_intervals(removed)
    for start, end in merge_intervals(intervals):
        cursor = start
        for h_start, h_end in holes:
            if h_end <= cursor or h_start >= end:
                continue
            if h_start > cursor:
                out.append((cursor, h_start))
            cursor = max(cursor, h_end)
        if cursor < end:
            out.append((cursor, end))
    return tuple(out)


def measured_duration(start_ns: int, end_ns: int) -> Quantity:
    """One raw monotonic duration in ns; no intermediate quantization."""
    if end_ns < start_ns:
        return Quantity.unavailable(WORK_NS, "monotonic_clock_reversed")
    return Quantity(end_ns - start_ns, WORK_NS, Basis.OBSERVED, MONOTONIC_V1)


def seconds_quantity(total_ms: int, method: str, basis: Basis = Basis.OBSERVED) -> Quantity:
    """Whole seconds from an already-summed millisecond total (floor once)."""
    return Quantity(total_ms // 1000, WORK_SECONDS, basis, method)


def service_milliseconds(duration_ns: Quantity) -> Quantity:
    """Presentation only: the input must retain actual observed raw ns."""
    if duration_ns.value is None:
        return Quantity.unavailable(WORK_MS, duration_ns.reason)
    if (
        duration_ns.unit != WORK_NS
        or duration_ns.basis is not Basis.OBSERVED
        or duration_ns.method != MONOTONIC_V1
    ):
        return Quantity.unavailable(WORK_MS, "nanosecond_precision_unavailable")
    return Quantity(duration_ns.value // NS_PER_MS, WORK_MS, Basis.OBSERVED, duration_ns.method)


def ns_seconds_quantity(total_ns: int, method: str, basis: Basis = Basis.OBSERVED) -> Quantity:
    """Whole seconds from an already-summed nanosecond quantity."""
    return Quantity(total_ns // NS_PER_SECOND, WORK_SECONDS, basis, method)


@dataclass(frozen=True)
class ActorInterval:
    """One observed work interval of one actor; times are wall-clock epoch ms."""

    actor: str
    start_ms: int
    end_ms: int

    @property
    def valid(self) -> bool:
        return self.end_ms >= self.start_ms


@dataclass(frozen=True)
class WorkTime:
    """Work-seconds (per-actor sum) and elapsed (interval union) kept separate."""

    work_ms: int
    elapsed_ms: int
    invalid_intervals: int

    @property
    def work_seconds(self) -> Quantity:
        return seconds_quantity(self.work_ms, TRANSCRIPT_INTERVALS_V1)

    @property
    def elapsed_seconds(self) -> Quantity:
        return seconds_quantity(self.elapsed_ms, TRANSCRIPT_INTERVALS_V1)

    @property
    def human_time(self) -> Quantity:
        """Parallel work is not waiting time; without dependency evidence: unknown."""
        return Quantity.unavailable(HUMAN_TIME, "no_dependency_evidence")


def work_time(intervals: Iterable[ActorInterval]) -> WorkTime:
    """Per-actor union summed (work) vs union over all actors (elapsed)."""
    by_actor: dict[str, list[tuple[int, int]]] = {}
    invalid = 0
    for interval in intervals:
        if not interval.valid:
            invalid += 1
            continue
        by_actor.setdefault(interval.actor, []).append((interval.start_ms, interval.end_ms))
    work = sum(union_length(spans) for spans in by_actor.values())
    elapsed = union_length(span for spans in by_actor.values() for span in spans)
    return WorkTime(work, elapsed, invalid)


@dataclass(frozen=True)
class CappedGap:
    """``capped_gap_v1`` result: known subtotal plus the count of unknown gaps."""

    known_ms: int
    invalid_gaps: int
    capped_gaps: int

    @property
    def quantity(self) -> Quantity:
        if self.invalid_gaps:
            return Quantity.unavailable(WORK_SECONDS, "invalid_timing_in_gaps")
        return seconds_quantity(self.known_ms, CAPPED_GAP_V1, Basis.ESTIMATED)


def capped_gap_v1(timestamps_ms: Sequence[int | None]) -> CappedGap:
    """Cap adjacent gaps; absent/reversed clocks or no bounded pair remain unknown."""
    known = capped = 0
    invalid = int(len(timestamps_ms) < 2)
    for before, after in zip(timestamps_ms, timestamps_ms[1:], strict=False):
        if before is None or after is None or after < before:
            invalid += 1
            continue
        gap = after - before
        if gap > GAP_CAP_MS:
            known += GAP_CAP_MS
            capped += 1
        else:
            known += gap
    return CappedGap(known, invalid, capped)
