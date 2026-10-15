"""Strict bounded receiver values; identity/authentication remains in receiver_contract."""

import re
from collections.abc import Mapping
from typing import Any

from ssgrep.accounting.contracts import Basis

MAX_STRING = 1024
MAX_ENTRIES = 64
MAX_KEY_CHARS = 64
MAX_INT = 2**63 - 1
SCENARIOS = ("conservative", "central", "optimistic")
_QUANTITY_KEYS = frozenset(("value", "unit", "basis", "method", "reason"))
_AMOUNT_KEYS = frozenset(("credit", "debit", "net"))
_SCENARIO_KEYS = _AMOUNT_KEYS | {
    "known_net",
    "complete",
    "bounds",
    "credit_missing",
    "debit_missing",
    "known_atoms",
    "known_debits",
}
_BOUND_NAMES = (
    "credit_lower",
    "credit_upper",
    "debit_lower",
    "debit_upper",
    "net_lower",
    "net_upper",
)
_ASSUMPTIONS = frozenset(
    {
        "credit_and_debit_nonnegative",
        "missing_credit_has_no_asserted_finite_maximum",
        "missing_debit_unbounded_unless_explicit_raw_bounds",
        "same_unique_compatible_debit_set_in_all_scenarios",
        "journal_active_proxy_ns_v2",
        "sum_raw_before_monotone_unit_conversion",
        "known_net_is_subtotal_not_complete_net_or_lower_bound",
        "outward_round_lower_floor_upper_ceil",
        "point_net_uses_floored_component_totals",
        "additional_assumptions_not_exported",
    }
)
_CODE = re.compile(r"[a-z][a-z0-9_:,]{0,63}")


class _Invalid(Exception):
    pass


def _fail():
    return _Invalid()


def _int(value: Any, low: int = -MAX_INT - 1) -> int:
    if type(value) is not int or not low <= value <= MAX_INT:
        raise _fail()
    return value


def _text(value: Any, *, empty: bool = False, limit: int = MAX_STRING) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value):
        raise _fail()
    return value


def _mapping(value: Any, keys=None) -> Mapping:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise _fail()
    if keys is not None and set(value) != keys:
        raise _fail()
    return value


def _quantity(raw):
    q = _mapping(raw, _QUANTITY_KEYS)
    value = q["value"]
    if value is not None:
        _int(value)
    try:
        basis = Basis(q["basis"])
    except (ValueError, TypeError):
        raise _fail() from None
    reason = _text(q["reason"], empty=True)
    if (value is None) != (basis is Basis.UNAVAILABLE) or (value is None and not reason):
        raise _fail()
    return {
        "value": value,
        "unit": _text(q["unit"]),
        "basis": basis.value,
        "method": _text(q["method"], empty=True),
        "reason": reason,
    }


def _missing(raw):
    values = _mapping(raw)
    if len(values) > MAX_ENTRIES:
        raise _fail()
    out = {}
    for code, count in values.items():
        if not _CODE.fullmatch(code):
            raise _fail()
        out[code] = _int(count, 1)
    return out


def _bounds(raw, unit):
    if raw is None:
        return None
    values = _mapping(raw, frozenset((*_BOUND_NAMES, "assumptions")))
    out = {name: _quantity(values[name]) for name in _BOUND_NAMES}
    assumptions = values["assumptions"]
    if (
        not isinstance(assumptions, list)
        or len(assumptions) > 16
        or any(not isinstance(a, str) or a not in _ASSUMPTIONS for a in assumptions)
    ):
        raise _fail()
    out["assumptions"] = list(assumptions)
    if any(out[name]["unit"] != unit for name in _BOUND_NAMES):
        raise _fail()
    for part in ("credit", "debit", "net"):
        lower, upper = out[f"{part}_lower"]["value"], out[f"{part}_upper"]["value"]
        if part != "net" and any(v is not None and v < 0 for v in (lower, upper)):
            raise _fail()
        if lower is not None and upper is not None and lower > upper:
            raise _fail()
    cmin, cmax = out["credit_lower"]["value"], out["credit_upper"]["value"]
    dmin, dmax = out["debit_lower"]["value"], out["debit_upper"]["value"]
    lower = None if cmin is None or dmax is None else cmin - dmax
    upper = None if cmax is None or dmin is None else cmax - dmin
    if out["net_lower"]["value"] != lower or out["net_upper"]["value"] != upper:
        raise _fail()
    return out


def _scenarios(raw):
    out = {}
    for name, amounts in _mapping(raw).items():
        if name not in SCENARIOS:
            raise _fail()
        a = _mapping(amounts)
        if set(a) not in (_AMOUNT_KEYS, _SCENARIO_KEYS):
            raise _fail()
        row = {part: _quantity(a[part]) for part in sorted(_AMOUNT_KEYS)}
        if set(a) == _AMOUNT_KEYS:
            out[name] = row
            continue
        unit = row["credit"]["unit"]
        if any(q["unit"] != unit for q in row.values()) or type(a["complete"]) is not bool:
            raise _fail()
        row.update(
            {
                "known_net": None if a["known_net"] is None else _int(a["known_net"]),
                "complete": a["complete"],
                "credit_missing": _missing(a["credit_missing"]),
                "debit_missing": _missing(a["debit_missing"]),
                "known_atoms": _int(a["known_atoms"], 0),
                "known_debits": _int(a["known_debits"], 0),
                "bounds": _bounds(a["bounds"], unit),
            }
        )
        credit, debit, net = (row[p]["value"] for p in ("credit", "debit", "net"))
        if credit is not None and debit is not None and row["known_net"] != credit - debit:
            raise _fail()
        if row["complete"]:
            if (
                credit is None
                or debit is None
                or row["credit_missing"]
                or row["debit_missing"]
                or net != credit - debit
            ):
                raise _fail()
        elif net is not None:
            raise _fail()
        out[name] = row
    return out


def _flat(raw: Any, *, allow_bool: bool) -> dict:
    m = _mapping(raw)
    if len(m) > MAX_ENTRIES:
        raise _fail()
    out = {}
    for key, value in m.items():
        _text(key, limit=MAX_KEY_CHARS)
        if isinstance(value, str):
            _text(value, empty=True)
        elif type(value) is bool:
            if not allow_bool:
                raise _fail()
        else:
            _int(value)
        out[key] = value
    return out


def _unit(raw):
    values = _mapping(raw, frozenset(("unit", "method", "tokenizer", "currency")))
    out = {"unit": _text(values["unit"]), "method": _text(values["method"])}
    for name in ("tokenizer", "currency"):
        out[name] = None if values[name] is None else _text(values[name])
    return out
