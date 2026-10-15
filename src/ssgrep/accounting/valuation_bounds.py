"""Sound deterministic bounds in one method/unit; known subtotals are not bounds.

All unsupported credit/costs are nonnegative. Missing credit has no asserted
finite maximum. Missing debit may have caller-evidenced raw bounds, otherwise
its maximum is unbounded. Raw lower endpoints round down and raw upper endpoints
round up after compatible sums, so integer intervals enclose the underlying raw
interval. Point/known-subtotal values remain differences of floored component
totals: a whole-unit model, not a claim that the raw interval is exact.
"""

from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.ledger_contracts import EstimateBounds
from ssgrep.accounting.timeunits import NS_PER_SECOND
from ssgrep.accounting.units import CHARS_DIV4_V1, WORK_SECONDS, UnitSpec

BOUND_ASSUMPTIONS = (
    "credit_and_debit_nonnegative",
    "missing_credit_has_no_asserted_finite_maximum",
    "missing_debit_unbounded_unless_explicit_raw_bounds",
    "same_unique_compatible_debit_set_in_all_scenarios",
    "sum_raw_before_monotone_unit_conversion",
    "known_net_is_subtotal_not_complete_net_or_lower_bound",
    "outward_round_lower_floor_upper_ceil",
    "point_net_uses_floored_component_totals",
)


def to_unit_value(raw: int, unit: UnitSpec) -> int:
    if unit.method == CHARS_DIV4_V1:
        return raw // 4
    if unit.unit == WORK_SECONDS and unit.method == "capped_gap_v1":
        return raw // NS_PER_SECOND
    return raw


def _upper_unit_value(raw: int, unit: UnitSpec) -> int:
    """Ceiling using the exact raw denominator, with no float conversion."""
    denominator = (
        4
        if unit.method == CHARS_DIV4_V1
        else NS_PER_SECOND
        if unit.unit == WORK_SECONDS and unit.method == "capped_gap_v1"
        else 1
    )
    return -((-raw) // denominator)


def estimate_bounds(
    credit_raw: int,
    credit_missing: bool,
    debit_min: int,
    debit_max: int | None,
    unit: UnitSpec,
    label: str,
    assumptions: tuple[str, ...] = (),
) -> EstimateBounds:
    def quantity(value: int | None, reason: str) -> Quantity:
        if value is None:
            return Quantity.unavailable(unit.unit, reason)
        return Quantity(value, unit.unit, Basis.ESTIMATED, f"estimate:{label}")

    cmin = to_unit_value(credit_raw, unit)
    cmax = None if credit_missing else _upper_unit_value(credit_raw, unit)
    dmin = to_unit_value(debit_min, unit)
    dmax = None if debit_max is None else _upper_unit_value(debit_max, unit)
    return EstimateBounds(
        quantity(cmin, "credit_lower_unavailable"),
        quantity(cmax, "credit_upper_unbounded"),
        quantity(dmin, "debit_lower_unavailable"),
        quantity(dmax, "debit_upper_unbounded"),
        quantity(None if dmax is None else cmin - dmax, "net_lower_unbounded_debit"),
        quantity(None if cmax is None else cmax - dmin, "net_upper_unbounded_credit"),
        tuple(sorted(set(BOUND_ASSUMPTIONS) | set(assumptions))),
    )
