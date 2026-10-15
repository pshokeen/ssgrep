"""Per-window estimate rollups from CURRENT period revisions only (pure; ints/strings).

Estimates are daily: a period (one UTC day) is in a rolling window when its day is
on or after the window's start day, so a window's estimate covers whole UTC days
even though its observations use exact event time. Superseded revisions never
contribute (``revisions.current_rows``); nets keep negative values and the
unavailable reasons ``valuation.rollup`` produced.
"""

from __future__ import annotations

from collections.abc import Sequence

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.plain_codec import to_plain
from ssgrep.accounting.replay import LedgerRows
from ssgrep.accounting.revisions import PeriodRow, current_rows
from ssgrep.accounting.units import UnitSpec
from ssgrep.accounting.valuation import Leaf, rollup

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
