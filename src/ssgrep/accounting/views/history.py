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

from ssgrep.accounting.collect.collector import RetentionCoverage
from ssgrep.accounting.collect.replay import LedgerRows
from ssgrep.accounting.ledger.contracts import SCHEMA_VERSION, Basis, Quantity, Scenario
from ssgrep.accounting.ledger.ledger_contracts import WINDOWS, EstimateBounds, ScenarioEstimate
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord, summarize
from ssgrep.accounting.ledger.revisions import PeriodRow, current_rows, history, unit_key
from ssgrep.accounting.ledger.units import NS_PER_MS, UnitSpec
from ssgrep.accounting.ledger.valuation import Leaf, rollup
from ssgrep.accounting.privacy.storage import from_plain, to_plain
from ssgrep.accounting.views.types import ScenarioRow, parse_quantity

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
    measured_service_known_ns: int | None = (
        None  # missing legacy precision is never inferred from ms
    )
    measured_service_missing: int | None = None  # legacy sealed timing is unknown

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
            "measured_service_known_ns": self.measured_service_known_ns,
            "measured_service_missing": self.measured_service_missing,
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
            None
            if raw.get("measured_service_known_ns") is None
            else _int(raw["measured_service_known_ns"], "measured_service_known_ns"),
            None
            if raw.get("measured_service_missing") is None
            else _int(raw["measured_service_missing"], "measured_service_missing"),
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
    measured = [
        e.measured_service_ns.value
        for e in executions
        if (
            e.measured_service_ns.basis is Basis.OBSERVED
            and e.measured_service_ns.method == "monotonic_v1"
            and e.measured_service_ns.unit == "work_ns"
            and type(e.measured_service_ns.value) is int
            and e.measured_service_ns.value >= 0
        )
    ]
    return SealedDay(
        day,
        counts,
        known["visible_bytes"],
        missing["visible_bytes"],
        known["rendered_bytes"],
        missing["rendered_bytes"],
        sum(measured),
        len(executions) - len(measured),
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
    service_known = sum(part.measured_service_known_ns or 0 for part in parts) // NS_PER_MS
    service_missing = sum(
        part.executions
        if part.measured_service_missing is None or part.measured_service_known_ns is None
        else part.measured_service_missing
        for part in parts
    )
    service_coverage_unknown = any(
        part.measured_service_known_ns is None or part.measured_service_missing is None
        for part in parts
    )
    out["measured_service_ms_known"] = Quantity.observed(service_known, "work_ms", "monotonic_v1")
    out["measured_service_ms"] = (
        Quantity.unavailable(
            "work_ms",
            "journal_nanosecond_precision_unavailable"
            if service_coverage_unknown
            else "measured_service_ms_missing_for_some_executions",
        )
        if service_missing or service_coverage_unknown
        else Quantity.observed(service_known, "work_ms", "monotonic_v1")
    )
    out["measured_service_missing"] = Quantity.observed(service_missing, "count", "timing_coverage")
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


def history_documents(rows, unit, today):
    result = []
    versions = [("", rows.namespace, rows.periods, True)]
    versions.extend((a.revision_name, a.namespace, a.periods, False) for a in rows.archives)
    for revision_name, namespace, periods, active in versions:
        by_key = {(p.period, p.revision): p for p in periods if p.namespace == namespace}
        for record in history(periods, namespace, unit, today):
            period = by_key[(record.period, record.revision)]
            estimates = next(
                (values for key, values in period.estimates if key == unit_key(unit)), ()
            )
            scenarios = [to_plain(e) for e in estimates]
            result.append(
                {
                    "format": "history_v2",
                    "schema": SCHEMA_VERSION,
                    "namespace": namespace,
                    "revision_name": revision_name,
                    "period": record.period,
                    "revision": record.revision,
                    "status": record.status if active else "superseded",
                    "current": active and record.current,
                    "supersedes": record.supersedes,
                    "net": to_plain(record.net),
                    "kind": record.kind,
                    "recomputable": record.recomputable,
                    "scenarios": scenarios,
                }
            )
    return sorted(
        result, key=lambda r: (r["period"], r["namespace"], r["revision_name"], r["revision"])
    )


Estimates = dict[str, dict[str, dict[str, object]]]


def quantity_dict(quantity: Quantity) -> dict[str, object]:
    """``service_types.parse_quantity``-readable form; ``reason`` is "" when available."""
    return {
        "value": quantity.value,
        "unit": quantity.unit,
        "basis": quantity.basis.value,
        "method": quantity.method,
        "reason": quantity.reason,
    }


def _unit_sort(unit: UnitSpec) -> tuple[str, str, str, str]:
    return (unit.unit, unit.method, unit.tokenizer or "", unit.currency or "")


def unit_keys(periods: Sequence[PeriodRow]) -> dict[UnitSpec, str]:
    """Stable key per unit across windows: the first unit of a name gets the bare name.

    "First" is the deterministic order of ``(unit, method, tokenizer, currency)``; the
    ledger rows do not carry methodology order. Other units sharing the name get
    ``unit:method`` (extended with tokenizer/currency only if that still collides).
    Computed from every current period so a key never depends on the window.
    """
    units = sorted({leaf.unit for row in periods for leaf in row.leaves}, key=_unit_sort)
    keys: dict[UnitSpec, str] = {}
    used: set[str] = set()
    for unit in units:
        key = unit.unit if unit.unit not in used else f"{unit.unit}:{unit.method}"
        if key in used:
            key = f"{key}:{unit.tokenizer or ''}:{unit.currency or ''}"
        used.add(key)
        keys[unit] = key
    return keys


def window_estimates(
    rows: LedgerRows | None, *, start_day: str | None, end_day: str | None
) -> Estimates:
    """Scenario credit/debit/net per unit key over current periods in ``[start_day, end_day]``.

    Both bounds are UTC days (``None`` = unbounded, as for lifetime). A period after
    ``end_day`` (clock skew) is outside rolling windows, like a future event.
    """
    if rows is None:
        return {}
    current = current_rows(rows.periods, rows.namespace)
    keys = unit_keys(current)
    by_unit: dict[UnitSpec, list[Leaf]] = {}
    for row in current:
        if (start_day is None or row.period >= start_day) and (
            end_day is None or row.period <= end_day
        ):
            for leaf in row.leaves:
                by_unit.setdefault(leaf.unit, []).append(leaf)
    out: Estimates = {}
    for unit in sorted(by_unit, key=_unit_sort):
        out[keys[unit]] = {
            e.scenario.value: to_plain(e.as_scenario_estimate())
            for e in rollup(by_unit[unit], unit)
        }
    return out


SCENARIOS = (Scenario.CONSERVATIVE, Scenario.CENTRAL, Scenario.OPTIMISTIC)
LEGACY_REASON = "legacy_valuation_uncertainty_unavailable"
REQUIRED = frozenset(
    {
        "known_net",
        "complete",
        "credit_missing",
        "debit_missing",
        "known_atoms",
        "known_debits",
        "bounds",
        "eligibility",
        "provenance",
        "unit",
    }
)


def parse_estimate(
    raw: object, scenario: Scenario, unit: str, missing_reason: str = "scenario_missing"
) -> ScenarioEstimate:
    entry = raw if isinstance(raw, dict) else {}
    if (
        REQUIRED <= entry.keys()
        and entry.get("bounds") is not None
        and entry.get("unit") is not None
    ):
        try:
            estimate = from_plain(ScenarioEstimate, {"scenario": scenario.value, **entry})
            if estimate.scenario is not scenario:
                raise ValueError("scenario mismatch")
            if estimate.complete != (not estimate.credit_missing and not estimate.debit_missing):
                raise ValueError("completeness mismatch")
            if estimate.complete != (estimate.net.value is not None):
                raise ValueError("complete net mismatch")
            if any(
                q.unit != estimate.unit.unit
                for q in (estimate.credit, estimate.debit, estimate.net)
            ):
                raise ValueError("estimate unit mismatch")
            return estimate
        except (ValueError, TypeError):
            reason = "malformed_valuation_uncertainty"
    else:
        reason = LEGACY_REASON if entry else missing_reason
    credit = parse_quantity(entry.get("credit"), unit, reason)
    debit = parse_quantity(entry.get("debit"), unit, reason)
    prior = parse_quantity(entry.get("net"), unit, reason)
    known = (
        credit.value - debit.value
        if credit.value is not None and debit.value is not None and credit.unit == debit.unit
        else prior.value
    )
    return ScenarioEstimate(
        scenario,
        credit,
        debit,
        Quantity.unavailable(credit.unit, reason),
        known,
        False,
        (("coverage", reason),),
        (("coverage", reason),),
    )


def scenario_row(unit_key: str, raw: object) -> ScenarioRow:
    entry = raw if isinstance(raw, dict) else {}
    unit = unit_key.split(":", 1)[0]
    missing_reason = "not_estimated" if raw is None else "scenario_missing"
    estimates = tuple(
        parse_estimate(entry.get(s.value), s, unit, missing_reason) for s in SCENARIOS
    )
    return ScenarioRow(
        unit_key,
        tuple((e.scenario, e.credit) for e in estimates),
        tuple((e.scenario, e.debit) for e in estimates),
        tuple((e.scenario, e.net) for e in estimates),
        estimates,
    )


def parse_scenarios(raw: object, *, current: bool = True) -> tuple[ScenarioEstimate, ...]:
    if not isinstance(raw, list):
        return ()
    result = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("scenario must be an object")
        scenario = Scenario(entry.get("scenario"))
        unit = entry.get("credit", {}).get("unit", "unknown")
        if not current:
            entry = {name: entry.get(name) for name in ("credit", "debit", "net")}
        result.append(parse_estimate(entry, scenario, unit))
    if len({e.scenario for e in result}) != len(result):
        raise ValueError("duplicate scenario")
    return tuple(result)


def sum_scenarios(
    estimates: Iterable[tuple[ScenarioEstimate, ...]],
) -> tuple[ScenarioEstimate, ...]:
    """Sum current period endpoints only when full UnitSpec identities agree.

    Period amounts have already been converted to integer units; their interval
    endpoints remain sound when summed. Known subtotals do not supply missing bounds.
    """
    groups: dict[Scenario, list[ScenarioEstimate]] = {}
    for rows in estimates:
        if not rows:
            missing = Quantity.unavailable("unknown", LEGACY_REASON)
            rows = tuple(ScenarioEstimate(s, missing, missing, missing) for s in SCENARIOS)
        for estimate in rows:
            groups.setdefault(estimate.scenario, []).append(estimate)
    result = []
    for scenario in SCENARIOS:
        rows = groups.get(scenario, [])
        if not rows:
            continue
        units = {r.unit for r in rows}
        methods = {q.method for r in rows for q in (r.credit, r.debit) if q.method}
        if len(units) != 1 or None in units or len(methods) > 1:
            missing = Quantity.unavailable("mixed", "incompatible_or_legacy_methods")
            gap = (("aggregate_identity", "incompatible_or_legacy_methods"),)
            result.append(
                ScenarioEstimate(
                    scenario, missing, missing, missing, credit_missing=gap, debit_missing=gap
                )
            )
            continue
        unit = rows[0].unit
        assert unit is not None  # The identity gate above excludes legacy units.

        def total(values, reason, unit=unit):
            values = tuple(values)
            if any(q.value is None for q in values):
                return Quantity.unavailable(unit.unit, reason)
            return Quantity(
                sum(q.value for q in values if q.value is not None),
                unit.unit,
                Basis.ESTIMATED,
                "sum_of_current_revisions",
            )

        credit = total((r.credit for r in rows), "credit_subtotal_unavailable")
        debit = total((r.debit for r in rows), "debit_subtotal_unavailable")
        complete = all(r.complete for r in rows)
        net = total((r.net for r in rows), "incomplete_current_revisions")
        bounds = None
        if all(r.bounds is not None for r in rows):
            bounds = EstimateBounds(
                credit_lower=total(
                    (r.bounds.credit_lower for r in rows if r.bounds is not None),
                    "credit_lower_unavailable",
                ),
                credit_upper=total(
                    (r.bounds.credit_upper for r in rows if r.bounds is not None),
                    "credit_upper_unavailable",
                ),
                debit_lower=total(
                    (r.bounds.debit_lower for r in rows if r.bounds is not None),
                    "debit_lower_unavailable",
                ),
                debit_upper=total(
                    (r.bounds.debit_upper for r in rows if r.bounds is not None),
                    "debit_upper_unavailable",
                ),
                net_lower=total(
                    (r.bounds.net_lower for r in rows if r.bounds is not None),
                    "net_lower_unavailable",
                ),
                net_upper=total(
                    (r.bounds.net_upper for r in rows if r.bounds is not None),
                    "net_upper_unavailable",
                ),
                assumptions=tuple(
                    sorted({a for r in rows if r.bounds is not None for a in r.bounds.assumptions})
                ),
            )
        result.append(
            ScenarioEstimate(
                scenario,
                credit,
                debit,
                net,
                sum(r.known_net for r in rows if r.known_net is not None)
                if all(r.known_net is not None for r in rows)
                else None,
                complete,
                tuple(sorted({m for r in rows for m in r.credit_missing})),
                tuple(sorted({m for r in rows for m in r.debit_missing})),
                sum(r.known_atoms for r in rows),
                sum(r.known_debits for r in rows),
                bounds,
                tuple(sorted({m for r in rows for m in r.eligibility})),
                tuple(sorted({m for r in rows for m in r.provenance})),
                unit,
            )
        )
    return tuple(result)
