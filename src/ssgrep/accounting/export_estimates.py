"""Positive JSON projection of uncertainty; private evidence identities never copied."""

from collections import Counter

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.export import safe_method, safe_reason
from ssgrep.accounting.lookup_time import LOOKUP_WORK_ASSUMPTIONS, LOOKUP_WORK_MAPPING
from ssgrep.accounting.valuation_bounds import BOUND_ASSUMPTIONS

BOUND_FIELDS = (
    "credit_lower",
    "credit_upper",
    "debit_lower",
    "debit_upper",
    "net_lower",
    "net_upper",
)
_ALLOWED_ASSUMPTIONS = frozenset(BOUND_ASSUMPTIONS)
_LIMIT = 2**63 - 1


def quantity(raw: Quantity, unit: str) -> dict:
    if raw.unit != unit:
        raise ValueError("incompatible_quantity_unit")
    if raw.value is not None and (
        type(raw.value) is not int or not -_LIMIT - 1 <= raw.value <= _LIMIT
    ):
        raise ValueError("quantity_out_of_range")
    return {
        "value": raw.value,
        "unit": unit,
        "basis": raw.basis.value,
        "method": safe_method(raw.method) if raw.method else "",
        "reason": safe_reason(raw.reason) if raw.reason else "",
    }


def missing_counts(missing) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, reason in missing:
        code = safe_reason(reason)
        counts[code if len(code) <= 64 else "unspecified"] += 1
    ordered = sorted(counts)
    out = {code: counts[code] for code in ordered[:63]}
    if len(ordered) > 63:
        out["other_missing_reasons"] = sum(counts[code] for code in ordered[63:])
    return out


def _assumptions(values) -> list[str]:
    out = set()
    for value in values:
        if value in _ALLOWED_ASSUMPTIONS:
            out.add(value)
        elif value in LOOKUP_WORK_ASSUMPTIONS:
            out.add(LOOKUP_WORK_MAPPING)
        else:
            out.add("additional_assumptions_not_exported")
    return sorted(out)


def project_scenario(estimate) -> dict:
    unit = estimate.unit.unit
    known = estimate.known_net
    if type(known) is not int or not -_LIMIT - 1 <= known <= _LIMIT:
        raise ValueError("known_subtotal_out_of_range")
    bounds = estimate.bounds
    return {
        "credit": quantity(estimate.credit, unit),
        "debit": quantity(estimate.debit, unit),
        "net": quantity(estimate.net, unit),
        "known_net": known,
        "complete": estimate.complete,
        "credit_missing": missing_counts(estimate.credit_missing),
        "debit_missing": missing_counts(estimate.debit_missing),
        "known_atoms": estimate.known_atoms,
        "known_debits": estimate.known_debits,
        "bounds": (
            {
                **{field: quantity(getattr(bounds, field), unit) for field in BOUND_FIELDS},
                "assumptions": _assumptions(bounds.assumptions),
            }
            if bounds is not None
            else None
        ),
    }
