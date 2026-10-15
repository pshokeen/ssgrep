"""Independent receipt arithmetic/coverage checks, bounded by observer-derived totals.

Receipt allocations are shares, never additional rollup credit. They may not exceed
an independently recomputed scenario union. Raw receipt numbers are subjects here,
not a replacement for the native/controller evidence used by the main oracle.
"""

from collections.abc import Mapping


def compare_receipt(result, document):
    body = document
    while isinstance(body, Mapping) and isinstance(body.get("data"), Mapping):
        body = body["data"]
    if not isinstance(body, Mapping) or not isinstance(body.get("scenarios"), list):
        return ["receipt.scenarios: missing"]
    unit_key = body.get("unit") or ""
    unit, _, method = unit_key.partition(":")
    estimates = result.time_estimates if unit == "work_seconds" else result.estimates
    if unit not in ("context_tokens", "work_seconds"):
        return ["receipt.unit: unknown/incomparable"]
    if method != {"context_tokens": "chars_div4_v1", "work_seconds": "capped_gap_v1"}[unit]:
        return ["receipt.unit: incompatible method"]
    raw_unit, divisor = (
        ("characters", 4) if unit == "context_tokens" else ("nanoseconds", 1000000000)
    )
    raw_debit, raw_min, raw_max = (
        body.get(field) for field in ("debit_raw", "debit_raw_min", "debit_raw_max")
    )
    credits = body.get("credit_raw")
    if (
        body.get("raw_unit") != raw_unit
        or not isinstance(credits, Mapping)
        or type(raw_debit) is not int
        or raw_debit < 0
        or type(raw_min) is not int
        or raw_min < raw_debit
        or "debit_raw_max" not in body
        or (raw_max is not None and (type(raw_max) is not int or raw_max < raw_min))
    ):
        return ["receipt.raw_inputs: unavailable or invalid raw precision; no rounded fallback"]
    problems = []
    rows = body["scenarios"]
    if {row.get("scenario") for row in rows if isinstance(row, Mapping)} != set(estimates):
        problems.append("receipt.scenarios: incomplete scenario coverage")
    for row in rows:
        if not isinstance(row, Mapping) or row.get("scenario") not in estimates:
            problems.append("receipt.scenario: unknown")
            continue
        name = row["scenario"]
        label = f"receipt.{unit}.{name}"
        expected = estimates[name]
        raw_credit = credits.get(name)
        if type(raw_credit) is not int or raw_credit < 0:
            problems.append(f"{label}.raw_credit: unavailable or invalid")
            continue
        if (
            expected.raw_credit is None
            or expected.raw_debit is None
            or raw_credit > expected.raw_credit
            or raw_debit > expected.raw_debit
        ):
            problems.append(f"{label}: raw allocation exceeds independently observed totals")

        def value(field, mapping=row, label=label):
            quantity = mapping.get(field)
            if not isinstance(quantity, Mapping) or "value" not in quantity:
                problems.append(f"{label}.{field}: missing quantity")
                return None
            raw = quantity["value"]
            if raw is not None and type(raw) is not int:
                problems.append(f"{label}.{field}: non-integer")
                return None
            if raw is None and not quantity.get("reason"):
                problems.append(f"{label}.{field}: unavailable without reason")
            if quantity.get("unit") != unit:
                problems.append(f"{label}.{field}: incomparable unit")
            return raw

        credit, debit, net = (value(field) for field in ("credit", "debit", "net"))
        if credit is None or debit is None or min(credit, debit) < 0:
            problems.append(f"{label}: known subtotal missing or negative")
            continue
        if credit > expected.credit or debit > expected.debit:
            problems.append(f"{label}: allocation exceeds independently observed totals")
        if credit != raw_credit // divisor or debit != raw_debit // divisor:
            problems.append(f"{label}: point subtotal differs from declared raw inputs")
        if type(row.get("known_net")) is not int or row["known_net"] != credit - debit:
            problems.append(f"{label}.known_net: incorrect subtotal")
        credit_missing, debit_missing = row.get("credit_missing"), row.get("debit_missing")
        if not isinstance(credit_missing, list) or not isinstance(debit_missing, list):
            problems.append(f"{label}: missing explicit token/time coverage")
        complete = not credit_missing and not debit_missing
        if (
            row.get("complete") is not complete
            or net != (credit - debit if complete else None)
            or (not debit_missing and (raw_min != raw_debit or raw_max != raw_debit))
        ):
            problems.append(f"{label}.net: incomplete evidence exposed as complete")
        bounds = row.get("bounds") if isinstance(row.get("bounds"), Mapping) else {}
        assumptions = bounds.get("assumptions")
        if (
            not isinstance(assumptions, (list, tuple))
            or any(not isinstance(item, str) for item in assumptions)
            or not set(expected.bound_assumptions).issubset(assumptions)
        ):
            problems.append(
                f"{label}.bounds.assumptions: missing outward/point precision methodology"
            )
        cmin, cmax, dmin, dmax, lower, upper = (
            value(field, bounds)
            for field in (
                "credit_lower",
                "credit_upper",
                "debit_lower",
                "debit_upper",
                "net_lower",
                "net_upper",
            )
        )
        credit_upper = None if credit_missing else (raw_credit + divisor - 1) // divisor
        debit_upper = None if raw_max is None else (raw_max + divisor - 1) // divisor
        if cmin != raw_credit // divisor or cmax != credit_upper:
            problems.append(f"{label}.bounds.credit: incorrect outward endpoints")
        if dmin != raw_min // divisor or dmax != debit_upper:
            problems.append(f"{label}.bounds.debit: incorrect outward endpoints")
        if debit_missing and dmax is None and lower is not None:
            problems.append(f"{label}.bounds.net_lower: unbounded debit presented as bound")
        wanted_lower = None if cmin is None or dmax is None else cmin - dmax
        wanted_upper = None if cmax is None or dmin is None else cmax - dmin
        if lower != wanted_lower or upper != wanted_upper:
            problems.append(f"{label}.bounds.net: incorrect endpoint subtraction")
    return problems
