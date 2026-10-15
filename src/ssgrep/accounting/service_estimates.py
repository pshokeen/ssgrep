"""Lossless current estimate projections; legacy values are subtotals, never proof.

This read adapter does not migrate or rewrite history. Missing v2 metadata is a
named unavailable state; the explicit collector migration owns new revisions.
"""

from __future__ import annotations

from collections.abc import Iterable

from ssgrep.accounting.contracts import Basis, Quantity, Scenario
from ssgrep.accounting.ledger_contracts import EstimateBounds, ScenarioEstimate
from ssgrep.accounting.plain_codec import from_plain
from ssgrep.accounting.service_types import ScenarioRow, parse_quantity

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
