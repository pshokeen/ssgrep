"""Fixed-bucket integer-microsecond duration histogram and strict JSON field checks.

Bucket upper bounds are Prometheus ``le`` bounds: inclusive. ``buckets`` are
NON-cumulative (the last one is ``+Inf``); ``count`` always equals their sum and
``sum_us`` is the exact integer sum of the observed durations. No floats.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BUCKET_BOUNDS_US = (
    1_000,
    5_000,
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    30_000_000,
)
BUCKET_COUNT = len(BUCKET_BOUNDS_US) + 1

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def bucket_index(duration_us: int) -> int:
    """Index of the first bucket whose inclusive bound holds ``duration_us``."""
    return bisect_left(BUCKET_BOUNDS_US, duration_us)


def as_int(value: Any, name: str) -> int:
    """A non-negative real int (bool and float are garbage)."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: expected a non-negative integer")
    return value


def as_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}: expected a non-empty string")
    return value


def as_list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name}: expected a list")
    return value


def as_map(value: Any, name: str, fields: set[str]) -> Mapping[str, Any]:
    """A dict with EXACTLY ``fields`` as keys."""
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name}: expected an object with fields {sorted(fields)}")
    return value


def as_day(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DAY.fullmatch(value):
        raise ValueError(f"{name}: expected YYYY-MM-DD")
    return value


def as_utc(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        raise ValueError(f"{name}: expected YYYY-MM-DDTHH:MM:SSZ")
    return value


@dataclass(frozen=True)
class Histogram:
    buckets: tuple[int, ...] = (0,) * BUCKET_COUNT
    count: int = 0
    sum_us: int = 0

    def __post_init__(self) -> None:
        if len(self.buckets) != BUCKET_COUNT or any(b < 0 for b in self.buckets):
            raise ValueError(f"a histogram has {BUCKET_COUNT} non-negative buckets")
        if self.count != sum(self.buckets) or self.sum_us < 0:
            raise ValueError("histogram count must equal the bucket total and sum_us be >= 0")

    def add(self, duration_us: int) -> Histogram:
        index = bucket_index(duration_us)
        buckets = tuple(b + (i == index) for i, b in enumerate(self.buckets))
        return Histogram(buckets, self.count + 1, self.sum_us + duration_us)

    def cumulative(self) -> tuple[int, ...]:
        """Prometheus ``le`` counts: bucket i holds every observation <= bound i."""
        total, out = 0, []
        for bucket in self.buckets:
            total += bucket
            out.append(total)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"buckets": list(self.buckets), "count": self.count, "sum_us": self.sum_us}

    @classmethod
    def from_dict(cls, raw: Any) -> Histogram:
        data = as_map(raw, "histogram", {"buckets", "count", "sum_us"})
        buckets = tuple(as_int(b, "bucket") for b in as_list(data["buckets"], "buckets"))
        return cls(buckets, as_int(data["count"], "count"), as_int(data["sum_us"], "sum_us"))
