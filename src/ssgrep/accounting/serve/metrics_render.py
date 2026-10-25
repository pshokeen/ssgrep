"""OpenMetrics 1.0 text exposition for integer samples (no timestamps, no floats).

Counter samples are named ``<family>_total`` and typed ``counter`` on ``<family>``; the info
series is ``<family>_info`` typed ``info``; histogram samples are ``_bucket``/``_count``/
``_sum`` (``_sum`` arrives as integer microseconds and prints as decimal seconds). A family
whose name ends in seconds/bytes/tokens/milliseconds declares that ``# UNIT``. The document ends
with ``# EOF``. No sample carries an explicit timestamp: a stale snapshot is shown stale by
its collected-timestamp gauge, never by a fabricated event time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

CONTENT_TYPE = "application/openmetrics-text; version=1.0.0; charset=utf-8"
_UNITS = ("seconds", "bytes", "tokens", "milliseconds")
_SUFFIXES = {
    "counter": "_total",
    "info": "_info",
    "histogram_bucket": "_bucket",
    "histogram_count": "_count",
    "histogram_sum": "_sum",
}
_TYPES = {
    "counter": "counter",
    "gauge": "gauge",
    "info": "info",
    "histogram_bucket": "histogram",
    "histogram_count": "histogram",
    "histogram_sum": "histogram",
}


class ValidationError(ValueError):
    """Explicit rejection of a dimension, label value, or limit (never silently coerced)."""

    def __init__(self, condition: str, message: str) -> None:
        super().__init__(message)
        self.condition = condition


@dataclass(frozen=True)
class Sample:
    """One series point. ``family`` is the full sample name (``..._total``, ``..._bucket``)."""

    family: str
    kind: str  # counter|gauge|info|histogram_bucket|histogram_count|histogram_sum
    labels: tuple[tuple[str, str], ...]
    value: int


def family_name(sample: Sample) -> str:
    suffix = _SUFFIXES.get(sample.kind, "")
    if suffix and not sample.family.endswith(suffix):
        raise ValidationError("sample_name", f"{sample.family} must end in {suffix}")
    return sample.family[: len(sample.family) - len(suffix)] if suffix else sample.family


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(sample: Sample) -> str:
    if sample.kind == "histogram_sum":
        seconds, micros = divmod(sample.value, 10**6)
        return f"{seconds}.{micros:06d}"
    return str(sample.value)


def render_openmetrics(samples: Sequence[Sample]) -> bytes:
    """The exposition for ``samples`` (families in first-seen order, series kept contiguous)."""
    order: list[str] = []
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        name = family_name(sample)
        if name not in grouped:
            order.append(name)
        grouped.setdefault(name, []).append(sample)
    lines: list[str] = []
    for name in order:
        group = grouped[name]
        lines.append(f"# TYPE {name} {_TYPES[group[0].kind]}")
        unit = next((u for u in _UNITS if name.endswith(f"_{u}")), None)
        if unit:
            lines.append(f"# UNIT {name} {unit}")
        for sample in group:
            labels = ",".join(f'{n}="{_escape(v)}"' for n, v in sample.labels)
            lines.append(
                f"{sample.family}{{{labels}}} {_number(sample)}"
                if labels
                else f"{sample.family} {_number(sample)}"
            )
    lines.append("# EOF")
    return ("\n".join(lines) + "\n").encode("ascii")
