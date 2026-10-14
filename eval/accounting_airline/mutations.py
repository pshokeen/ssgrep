"""Domain mutation catalogue: wrong formulations that MUST be rejected on actual values.

Each entry runs a deliberately wrong query on the real database (as the read-only role)
and the independent oracle must report violations whose text names the actual wrong
values. ``CORRECT`` lists legitimate alternative formulations the oracle must accept
(equivalent SQL is judged on values, never wording).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import dataset, domain_oracle as oracle, queries as q
from .database import READER, LabInstance

Check = Callable[[Sequence[Mapping[str, Any]], oracle.State], list[str]]


def _check_plan(rows: Sequence[Mapping[str, Any]], state: oracle.State) -> list[str]:
    plan = {r["booking_id"]: tuple(r["legs"]) for r in rows}
    return oracle.plan_violations(state, plan)


CHECKS: Mapping[str, Check] = {
    "cash": lambda rows, state: oracle.verify_cash(rows, state),
    "seats": lambda rows, state: oracle.verify_seats(rows, state),
    "affected": lambda rows, state: oracle.verify_affected(rows, state),
    "connections": lambda rows, state: oracle.verify_connections(rows, state),
    "instants": lambda rows, state: oracle.verify_flight_instants(rows, state),
    "routes": lambda rows, state: oracle.verify_routes(rows, state),
    "local_clock": lambda rows, state: oracle.verify_local_clock(rows, state),
    "connectable": lambda rows, state: oracle.verify_connectable(rows, state),
    "plan": _check_plan,
}


@dataclass(frozen=True)
class Formulation:
    name: str
    check: str
    sql: str
    # Fragments each violation list must contain: the oracle names the actual wrong value.
    must_flag: tuple[str, ...] = ()
    trap: str = ""

    @property
    def wrong(self) -> bool:
        return bool(self.must_flag)


MUTATIONS: tuple[Formulation, ...] = (
    Formulation(
        "cash.join_fanout",
        "cash",
        q.CASH_JOIN_FANOUT,
        (
            "cash ('B1', 'USD'): expected 45000 actual 180000",
            "cash ('B2', 'USD'): expected 50000 actual 200000",
        ),
        "payment x coupon x codeshare fan-out; B3/B5 coincidentally stay right",
    ),
    Formulation(
        "cash.face_value",
        "cash",
        q.CASH_FACE_VALUE,
        (
            "cash ('B1', 'USD'): expected 45000 actual 50000",
            "cash ('B3', 'USD'): expected 0 actual 20000",
        ),
        "face value is not settled cash",
    ),
    Formulation(
        "cash.failed_payment_counted",
        "cash",
        q.CASH_INCLUDES_FAILED,
        ("cash ('B1', 'USD'): expected 45000 actual 145000",),
        "failed capture P4 must not count",
    ),
    Formulation(
        "cash.currency_merged",
        "cash",
        q.CASH_CURRENCY_MERGED,
        ("missing cash row ('B5', 'EUR')", "unexpected cash row ('B5', 'USD')"),
        "USD and EUR never merged",
    ),
    Formulation(
        "cash.null_metadata_drops_booking",
        "cash",
        q.CASH_DROPS_NULL_REF,
        ("missing cash row ('B6', 'USD')",),
        "B6 has NULL external_customer_ref",
    ),
    Formulation(
        "seats.hold_expiry_inclusive",
        "seats",
        q.seats_hold_expiry_inclusive(),
        ("seats F7: expected (1, 0, 1, 1) actual (1, 1, 0, 0)",),
        "expires_at >= as_of: H4 expiring exactly at as_of must not consume F7",
    ),
    Formulation(
        "seats.void_coupon_counted",
        "seats",
        q.seats_counting_void(),
        ("seats F1: expected (2, 0, 1, 0) actual (3, 0, 0, 0)",),
        "void coupon C5 must not occupy F1",
    ),
    Formulation(
        "seats.marketing_join_drops_flights",
        "seats",
        q.seats_via_marketing_join(),
        ("missing seat row F7", "missing seat row F8"),
        "F7/F8 have no marketing row",
    ),
    Formulation(
        "seats.cancelled_flight_sellable",
        "seats",
        q.seats_raw_free_as_sellable(),
        ("seats F1: expected (2, 0, 1, 0) actual (2, 0, 1, 1)",),
        "cancelled F1 is not sellable although it has raw free capacity",
    ),
    Formulation(
        "l2.group_by_name",
        "affected",
        q.affected_grouped_by_name(),
        ("affected:", "('B2', 'V2')"),
        "V1 and V2 share a display name",
    ),
    Formulation(
        "l2.cancelled_booking_counted",
        "affected",
        q.affected_ignoring_status(),
        ("('B3', 'V3')",),
        "B3 is cancelled/void",
    ),
    Formulation(
        "l2.exact_minimum_invalid",
        "affected",
        q.affected_strict_connection(),
        ("('B4', 'V4')", "('B6', 'V6')"),
        "an exactly-45-minute connection is valid",
    ),
    Formulation(
        "l2.connections_strict",
        "connections",
        q.connections_strict(),
        (
            "missing connection ('B4', 'F3', 'F4', 'ORD', 45, 'MCT-ORD-045', 45, True)",
            "unexpected connection ('B4', 'F3', 'F4', 'ORD', 45, 'MCT-ORD-045', 45, False)",
        ),
        "gap == minimum is valid",
    ),
    Formulation(
        "l2.local_clock_merges_flights",
        "instants",
        q.flight_instants_by_local_clock(),
        ("missing flight F8",),
        "F1 and F8 share displayed SEA 01:30 but are one hour apart",
    ),
    Formulation(
        "l2.local_clock_connectability",
        "connectable",
        q.connectable_pairs_by_local_clock(),
        ("unexpected connectable pair ('F8', 'F2')", "unexpected connectable pair ('F1', 'F2')"),
        "ordering flights by displayed local clock lets F8 connect to the earlier F2 departure",
    ),
    Formulation(
        "l3.capacity_blind_routes",
        "routes",
        q.candidate_routes_ignoring_capacity(),
        ("unexpected candidate route ('F6', 'F7')",),
        "H3 consumes the last F6 seat",
    ),
    Formulation(
        "l3.independent_f5_oversell",
        "plan",
        q.independent_plan(),
        ("F5: 2 travellers assigned for 1 sellable",),
        "both disrupted travellers independently offered the single remaining F5 seat",
    ),
)

# Legitimate alternatives the oracle must ACCEPT (values, not wording).
CORRECT: tuple[Formulation, ...] = (
    Formulation("cash.preaggregated", "cash", q.CASH_PREAGGREGATED),
    Formulation("cash.cte_alternative", "cash", q.CASH_CTE_ALTERNATIVE),
    Formulation("seats.canonical", "seats", q.seats()),
    Formulation("l2.connections", "connections", q.connections()),
    Formulation("l2.affected", "affected", q.affected()),
    Formulation("l2.instants", "instants", q.flight_instants()),
    Formulation("l2.local_clock", "local_clock", q.local_clock_listing()),
    Formulation("l2.connectable", "connectable", q.connectable_pairs()),
    Formulation("l3.routes", "routes", q.candidate_routes()),
)


def evaluate(
    lab: LabInstance, database: str, formulation: Formulation, state: oracle.State
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the SQL as the read-only role; return its ACTUAL rows and the oracle's verdict."""
    rows = lab.rows(database, formulation.sql, role=READER)
    return rows, CHECKS[formulation.check](rows, state)


def missing_flags(formulation: Formulation, violations: Sequence[str]) -> list[str]:
    joined = "\n".join(violations)
    return [frag for frag in formulation.must_flag if frag not in joined]


def clock_advanced_seats(as_of: str) -> str:
    """Seats recomputed at another frozen application clock (runtime clocks untouched)."""
    return q.seats(as_of)


ADVANCED_AS_OF = "2026-11-01T07:20:00Z"  # past H1/H3 (07:10); H2/H4 were already dead
assert dataset.AS_OF < ADVANCED_AS_OF
