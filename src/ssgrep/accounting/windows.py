"""UTC event-time windows (24h/7d/30d/lifetime) with sealed-day lifetime aggregates (D8).

Pure functions: no clocks (callers pass ``now``), no I/O, no floats.

* Rolling windows (24h/7d/30d) use EXACT event time: ``start <= ts_utc <= now``. A
  future event (clock skew) is outside every rolling window but still counts in
  ``lifetime``. Advancing only ``now`` changes rolling membership; nothing else.
* ``lifetime`` = sealed days UNION retained executions of days that are not sealed, so
  counts never decrease when old evidence expires. A sealed day stands in for its
  executions: later executions for it are reported by ``seal_days`` as late and ignored.
* Estimates are daily (``windows_estimates``): current period revisions only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ssgrep.accounting.collector_types import RetentionCoverage
from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.ledger_contracts import WINDOWS
from ssgrep.accounting.reduce_types import ExecutionRecord, summarize
from ssgrep.accounting.replay import LedgerRows
from ssgrep.accounting.windows_estimates import quantity_dict, window_estimates

WINDOW_NAMES = WINDOWS
_WINDOW_DAYS = {"24h": 1, "7d": 7, "30d": 30}
BASIS = "event_time_utc"
ESTIMATE_GRANULARITY = "utc_day"
SHORT_RETENTION = "evidence_retention_shorter_than_window"
_BYTE_LABELS = ("visible_bytes", "rendered_bytes")
_OUTCOME_PREFIX = "outcome."


def window_start(name: str, now: datetime) -> datetime | None:
    """``now`` minus the window length; ``None`` for ``lifetime`` (unbounded)."""
    if name == "lifetime":
        return None
    if name not in _WINDOW_DAYS:
        raise ValueError(f"unknown window {name!r}")
    return _utc(now) - timedelta(days=_WINDOW_DAYS[name])


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("window clock values must be timezone-aware")
    return moment.astimezone(UTC)


def _event_time(ts_utc: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts_utc)
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed.astimezone(UTC)


def in_window(ts_utc: str, name: str, now: datetime) -> bool:
    """Inclusive on both ends; unparseable or future timestamps are outside rolling windows."""
    event = _event_time(ts_utc)
    if event is None:
        return False
    start = window_start(name, now)
    return start is None or start <= event <= _utc(now)


# ---- sealed days ---------------------------------------------------------------------


def _int(raw: object, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{label} must be an integer")
    return raw


@dataclass(frozen=True)
class SealedDay:
    """Closed aggregate of one UTC day; keeps counts after its executions expire."""

    day: str
    counts: tuple[tuple[str, int], ...]
    visible_known: int
    visible_missing: int
    rendered_known: int
    rendered_missing: int

    @property
    def executions(self) -> int:
        """Every execution has exactly one outcome, so outcome counts total the day."""
        return sum(v for k, v in self.counts if k.startswith(_OUTCOME_PREFIX))

    def to_dict(self) -> dict[str, object]:
        return {
            "day": self.day,
            "counts": dict(self.counts),
            "visible_known": self.visible_known,
            "visible_missing": self.visible_missing,
            "rendered_known": self.rendered_known,
            "rendered_missing": self.rendered_missing,
        }

    @classmethod
    def from_dict(cls, raw: object) -> SealedDay:
        if not isinstance(raw, dict) or not isinstance(raw.get("day"), str):
            raise ValueError("malformed sealed day")
        counts = raw.get("counts")
        if not isinstance(counts, dict):
            raise ValueError("malformed sealed day counts")
        return cls(
            raw["day"],
            tuple(sorted((str(k), _int(v, "count")) for k, v in counts.items())),
            _int(raw.get("visible_known"), "visible_known"),
            _int(raw.get("visible_missing"), "visible_missing"),
            _int(raw.get("rendered_known"), "rendered_known"),
            _int(raw.get("rendered_missing"), "rendered_missing"),
        )


def _seal_one(day: str, executions: Sequence[ExecutionRecord]) -> SealedDay:
    summary = summarize(executions)
    counts = tuple(
        (name, quantity.value or 0)
        for name, quantity in sorted(summary.items())
        if quantity.unit == "count"
    )
    known = {label: summary[f"{label}_known"].value or 0 for label in _BYTE_LABELS}
    missing = {
        label: sum(1 for e in executions if getattr(e, label).value is None)
        for label in _BYTE_LABELS
    }
    return SealedDay(
        day,
        counts,
        known["visible_bytes"],
        missing["visible_bytes"],
        known["rendered_bytes"],
        missing["rendered_bytes"],
    )


def _by_day(executions: Iterable[ExecutionRecord]) -> dict[str, list[ExecutionRecord]]:
    days: dict[str, list[ExecutionRecord]] = {}
    for execution in executions:
        days.setdefault(execution.day, []).append(execution)
    return days


def seal_days(
    executions: Sequence[ExecutionRecord], *, before_day: str, already: Mapping[str, SealedDay]
) -> tuple[dict[str, SealedDay], int]:
    """Seal every unsealed UTC day ``< before_day`` that has executions.

    Returns all sealed days (old ones untouched) and the number of LATE executions:
    retained executions of an already sealed day beyond that day's sealed total. They
    are not added (a sealed day is closed). Idempotent and order independent. Late
    detection is count based, so it cannot see a late execution after the day's
    originals expired and were dropped from ``executions`` at the same time.
    """
    sealed = dict(already)
    late = 0
    for day, rows in sorted(_by_day(executions).items()):
        if day in already:
            late += max(0, len(rows) - already[day].executions)
        elif day < before_day:
            sealed[day] = _seal_one(day, rows)
    return dict(sorted(sealed.items())), late


# ---- observations --------------------------------------------------------------------


def _observations(parts: Sequence[SealedDay]) -> dict[str, dict[str, object]]:
    """Sum closed parts; bytes stay unavailable if ANY part lacks them (subtotal kept)."""
    totals: dict[str, int] = {}
    for part in parts:
        for name, value in part.counts:
            totals[name] = totals.get(name, 0) + value
    for name in ("searches", "shows", "notes", "failed"):
        totals.setdefault(name, 0)
    out = {name: Quantity(value, "count", Basis.OBSERVED, name) for name, value in totals.items()}
    for label in _BYTE_LABELS:
        stem = label.removesuffix("_bytes")
        known = sum(getattr(p, f"{stem}_known") for p in parts)
        missing = sum(getattr(p, f"{stem}_missing") for p in parts)
        out[f"{label}_known"] = Quantity(known, "bytes", Basis.OBSERVED, label)
        out[label] = (
            Quantity.unavailable("bytes", f"{label}_missing_for_some_executions")
            if missing
            else Quantity(known, "bytes", Basis.OBSERVED, label)
        )
    return {name: quantity_dict(q) for name, q in sorted(out.items())}


def _rolling_observations(
    executions: Sequence[ExecutionRecord], name: str, now: datetime
) -> dict[str, dict[str, object]]:
    chosen = [e for e in executions if in_window(e.ts_utc, name, now)]
    return _observations([_seal_one("", chosen)])


def _lifetime_observations(
    executions: Sequence[ExecutionRecord], sealed: Mapping[str, SealedDay]
) -> dict[str, dict[str, object]]:
    open_days = [e for e in executions if e.day not in sealed]
    parts = [sealed[day] for day in sorted(sealed)]
    return _observations([*parts, _seal_one("", open_days)])


# ---- coverage ------------------------------------------------------------------------


def _rolling_coverage(name: str, horizon_days: int) -> dict[str, object]:
    short = horizon_days < _WINDOW_DAYS[name]
    coverage: dict[str, object] = {"partial": short, "evidence_horizon_days": horizon_days}
    if short:
        coverage["reason"] = SHORT_RETENTION
    return coverage


def _lifetime_coverage(
    executions: Sequence[ExecutionRecord],
    sealed: Mapping[str, SealedDay],
    retention: RetentionCoverage,
    since: str | None,
) -> dict[str, object]:
    reasons = []
    if retention.evidence_expired_before:
        reasons.append("evidence_expired")
    if retention.aggregates_expired_before:
        reasons.append("aggregates_expired")
    if retention.frozen_roots:
        reasons.append("frozen_roots")
    earliest = min([*sealed, *(e.day for e in executions)], default=None)
    if since and earliest and since > earliest:
        reasons.append("data_predates_collection")
    return {
        "partial": bool(reasons),
        "reasons": reasons,
        "retention": {
            "evidence_expired_before": retention.evidence_expired_before,
            "aggregates_expired_before": retention.aggregates_expired_before,
            "frozen_roots": retention.frozen_roots,
            "complete_since": since,
            "sealed_days": len(sealed),
        },
    }


# ---- assembly ------------------------------------------------------------------------


def build_windows(
    *,
    executions: Sequence[ExecutionRecord],
    sealed: Mapping[str, SealedDay],
    rows: LedgerRows | None,
    retention: RetentionCoverage,
    now: datetime,
    collection_since_day: str | None,
    evidence_horizon_days: int,
) -> dict[str, dict[str, object]]:
    """Per window: observations, estimates, coverage (JSON-safe, deterministic, float-free)."""
    result: dict[str, dict[str, object]] = {}
    for name in WINDOW_NAMES:
        start = window_start(name, now)
        if start is None:
            observations = _lifetime_observations(executions, sealed)
            coverage = _lifetime_coverage(executions, sealed, retention, collection_since_day)
        else:
            observations = _rolling_observations(executions, name, now)
            coverage = _rolling_coverage(name, evidence_horizon_days)
        result[name] = {
            "observations": observations,
            "estimates": window_estimates(
                rows,
                start_day=None if start is None else start.date().isoformat(),
                end_day=None if start is None else _utc(now).date().isoformat(),
            ),
            "coverage": coverage,
            "basis": BASIS,
            "estimate_granularity": ESTIMATE_GRANULARITY,
        }
    return result
