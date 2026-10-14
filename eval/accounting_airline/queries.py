"""Lab-owned SQL formulations: correct alternatives and deliberately wrong ones.

These stand in for what a solver might write. The oracle NEVER reads this text: tests and
the lane run each query on the real database as the read-only role and hand the ACTUAL
result rows to ``domain_oracle``. A wrong formulation is rejected on its values, not
because its SQL differs from a reference string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import dataset


def _as_of(as_of: str) -> str:
    return f"'{as_of}'::timestamptz"


# ----------------------------------------------------------------------------- L1 cash

CASH_PREAGGREGATED = """
SELECT t.booking_id, t.currency, coalesce(sum(p.signed_amount_minor), 0)::bigint AS settled_minor
  FROM tickets t
  LEFT JOIN payment_events p
    ON p.booking_id = t.booking_id AND p.currency = t.currency AND p.status = 'settled'
 GROUP BY t.booking_id, t.currency
 ORDER BY t.booking_id
"""

CASH_CTE_ALTERNATIVE = """
WITH cash AS (
  SELECT booking_id, currency, sum(signed_amount_minor) AS minor
    FROM payment_events WHERE status = 'settled' GROUP BY booking_id, currency)
SELECT t.booking_id, t.currency, coalesce(c.minor, 0)::bigint AS settled_minor
  FROM tickets t LEFT JOIN cash c ON c.booking_id = t.booking_id AND c.currency = t.currency
 ORDER BY t.booking_id
"""

CASH_JOIN_FANOUT = """
SELECT p.booking_id, p.currency, sum(p.signed_amount_minor)::bigint AS settled_minor
  FROM payment_events p
  JOIN tickets t ON t.booking_id = p.booking_id
  JOIN coupons c ON c.ticket_id = t.ticket_id
  LEFT JOIN marketing_flights m ON m.flight_id = c.flight_id
 WHERE p.status = 'settled'
 GROUP BY p.booking_id, p.currency
 ORDER BY p.booking_id
"""

CASH_DROPS_NULL_REF = """
SELECT b.booking_id, t.currency, coalesce(sum(p.signed_amount_minor), 0)::bigint AS settled_minor
  FROM bookings b JOIN tickets t ON t.booking_id = b.booking_id
  LEFT JOIN payment_events p
    ON p.booking_id = b.booking_id AND p.currency = t.currency AND p.status = 'settled'
 WHERE b.external_customer_ref <> ''
 GROUP BY b.booking_id, t.currency ORDER BY b.booking_id
"""

CASH_FACE_VALUE = """
SELECT booking_id, currency, face_value_minor AS settled_minor FROM tickets ORDER BY booking_id
"""

CASH_INCLUDES_FAILED = """
SELECT t.booking_id, t.currency, coalesce(sum(p.signed_amount_minor), 0)::bigint AS settled_minor
  FROM tickets t
  LEFT JOIN payment_events p ON p.booking_id = t.booking_id AND p.currency = t.currency
 GROUP BY t.booking_id, t.currency ORDER BY t.booking_id
"""

CASH_CURRENCY_MERGED = """
SELECT t.booking_id, 'USD' AS currency,
       coalesce(sum(p.signed_amount_minor), 0)::bigint AS settled_minor
  FROM tickets t LEFT JOIN payment_events p
    ON p.booking_id = t.booking_id AND p.status = 'settled'
 GROUP BY t.booking_id ORDER BY t.booking_id
"""


# ----------------------------------------------------------------------------- L1 seats


def _seats_cte(as_of: str, *, hold_op: str = ">", ignore_status: bool = False) -> str:
    coupon_filter = (
        ""
        if ignore_status
        else "c.status = 'reserved' AND t.status = 'open' AND b.status = 'confirmed' AND "
    )
    return f"""
occ AS (
  SELECT c.flight_id, c.cabin, count(*) AS n
    FROM coupons c JOIN tickets t ON t.ticket_id = c.ticket_id
    JOIN bookings b ON b.booking_id = t.booking_id
   WHERE {coupon_filter}c.cabin = 'economy'
   GROUP BY c.flight_id, c.cabin),
hld AS (
  SELECT flight_id, cabin, sum(quantity) AS n FROM seat_holds
   WHERE status = 'active' AND expires_at {hold_op} {_as_of(as_of)}
   GROUP BY flight_id, cabin),
seats AS (
  SELECT i.flight_id,
         coalesce(o.n, 0)::int AS occupied,
         coalesce(h.n, 0)::int AS live_holds,
         (i.capacity - coalesce(o.n, 0) - coalesce(h.n, 0))::int AS raw_free,
         (CASE WHEN f.status <> 'scheduled' THEN 0
               ELSE i.capacity - coalesce(o.n, 0) - coalesce(h.n, 0) END)::int AS sellable_free
    FROM inventory i JOIN flight_instances f ON f.flight_id = i.flight_id
    LEFT JOIN occ o ON o.flight_id = i.flight_id AND o.cabin = i.cabin
    LEFT JOIN hld h ON h.flight_id = i.flight_id AND h.cabin = i.cabin
   WHERE i.cabin = 'economy')"""


def seats(as_of: str = dataset.AS_OF) -> str:
    return f"WITH {_seats_cte(as_of)} SELECT * FROM seats ORDER BY flight_id"


def seats_hold_expiry_inclusive(as_of: str = dataset.AS_OF) -> str:
    """Wrong: ``expires_at >= as_of`` lets H4 (expiring exactly at as_of) consume F7."""
    return f"WITH {_seats_cte(as_of, hold_op='>=')} SELECT * FROM seats ORDER BY flight_id"


def seats_counting_void(as_of: str = dataset.AS_OF) -> str:
    """Wrong: void coupon C5 and cancelled booking B3 still occupy F1."""
    return f"WITH {_seats_cte(as_of, ignore_status=True)} SELECT * FROM seats ORDER BY flight_id"


def seats_via_marketing_join(as_of: str = dataset.AS_OF) -> str:
    """Wrong: availability for flights that have a marketing row only (F7/F8 vanish)."""
    return (
        f"WITH {_seats_cte(as_of)} SELECT s.* FROM seats s "
        "WHERE EXISTS (SELECT 1 FROM marketing_flights m WHERE m.flight_id = s.flight_id) "
        "ORDER BY s.flight_id"
    )


def seats_raw_free_as_sellable(as_of: str = dataset.AS_OF) -> str:
    """Wrong: cancelled F1 treated as sellable (and holds ignored for sellable)."""
    return (
        f"WITH {_seats_cte(as_of)} SELECT flight_id, occupied, live_holds, raw_free, "
        "raw_free AS sellable_free FROM seats ORDER BY flight_id"
    )


# ----------------------------------------------------------------------------- L2

_LEGS = """
legs AS (
  SELECT t.booking_id, t.traveller_id, c.leg_no, c.flight_id, f.origin, f.destination,
         f.departure_utc, f.arrival_utc, f.status AS flight_status
    FROM tickets t JOIN coupons c ON c.ticket_id = t.ticket_id
    JOIN flight_instances f ON f.flight_id = c.flight_id
   WHERE t.status = 'open' AND c.status = 'reserved')"""


def connections(*, inclusive: bool = True) -> str:
    """Per-booking connections with gap, policy ID and validity (equality-inclusive)."""
    op = ">=" if inclusive else ">"
    return f"""
WITH {_LEGS}
SELECT a.booking_id, a.flight_id AS from_flight, b.flight_id AS to_flight,
       a.destination AS airport,
       (extract(epoch FROM (b.departure_utc - a.arrival_utc)) / 60)::int AS gap_minutes,
       r.rule_id, r.minimum_minutes,
       (b.departure_utc - a.arrival_utc {op} make_interval(mins => r.minimum_minutes)) AS valid
  FROM legs a JOIN legs b ON b.booking_id = a.booking_id AND b.leg_no = a.leg_no + 1
  JOIN connection_rules r ON r.airport_code = a.destination
 ORDER BY a.booking_id"""


def connections_strict() -> str:
    """Wrong: exact-minimum connection is treated as invalid (B4/B6 flip)."""
    return connections(inclusive=False)


def _affected_body(*, inclusive: bool = True) -> str:
    op = ">=" if inclusive else ">"
    return f"""
WITH {_LEGS},
bad_conn AS (
  SELECT a.booking_id FROM legs a JOIN legs b ON b.booking_id = a.booking_id
   AND b.leg_no = a.leg_no + 1
  JOIN connection_rules r ON r.airport_code = a.destination
   WHERE NOT (b.departure_utc - a.arrival_utc {op} make_interval(mins => r.minimum_minutes)))
SELECT DISTINCT b.booking_id, t.traveller_id, b.service_priority
  FROM bookings b JOIN tickets t ON t.booking_id = b.booking_id
 WHERE b.status = 'confirmed' AND t.status = 'open'
   AND (b.booking_id IN (SELECT booking_id FROM legs WHERE flight_status = 'cancelled')
     OR b.booking_id IN (SELECT booking_id FROM bad_conn))
 ORDER BY b.booking_id"""


def affected() -> str:
    return _affected_body()


def affected_strict_connection() -> str:
    """Wrong: exact-45 connections invalid, so B4/B6 become 'affected'."""
    return _affected_body(inclusive=False)


def affected_ignoring_status() -> str:
    """Wrong: cancelled booking/void ticket counted (B3 appears)."""
    return """
WITH legs AS (
  SELECT t.booking_id, t.traveller_id, c.flight_id, f.status AS flight_status
    FROM tickets t JOIN coupons c ON c.ticket_id = t.ticket_id
    JOIN flight_instances f ON f.flight_id = c.flight_id)
SELECT DISTINCT l.booking_id, l.traveller_id, b.service_priority
  FROM legs l JOIN bookings b ON b.booking_id = l.booking_id
 WHERE l.flight_status = 'cancelled' ORDER BY l.booking_id"""


def affected_grouped_by_name() -> str:
    """Wrong: collapses V1/V2 (same display name) into one displaced person."""
    return """
WITH legs AS (
  SELECT tv.display_name, min(t.booking_id) AS booking_id, min(t.traveller_id) AS traveller_id
    FROM travellers tv JOIN tickets t ON t.traveller_id = tv.traveller_id
    JOIN bookings b ON b.booking_id = t.booking_id
    JOIN coupons c ON c.ticket_id = t.ticket_id AND c.status = 'reserved'
    JOIN flight_instances f ON f.flight_id = c.flight_id
   WHERE b.status = 'confirmed' AND f.status = 'cancelled' GROUP BY tv.display_name)
SELECT booking_id, traveller_id, 0 AS service_priority FROM legs ORDER BY booking_id"""


def flight_instants() -> str:
    """Correct: one row per physical flight with its own UTC departure instant."""
    return """
SELECT flight_id,
       to_char(departure_utc AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS departure_utc
  FROM flight_instances ORDER BY flight_id"""


def flight_instants_by_local_clock() -> str:
    """Wrong: de-duplicates by displayed local departure time, merging F1 and F8."""
    return """
SELECT DISTINCT ON (f.origin, to_char(f.departure_utc AT TIME ZONE a.time_zone, 'HH24:MI'))
       f.flight_id,
       to_char(f.departure_utc AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS departure_utc
  FROM flight_instances f JOIN airports a ON a.airport_code = f.origin
 ORDER BY f.origin, to_char(f.departure_utc AT TIME ZONE a.time_zone, 'HH24:MI'), f.flight_id"""


def local_clock_listing() -> str:
    """Evidence: displayed local departure clock and UTC offset (hours) per flight."""
    return """
SELECT f.flight_id,
       to_char(f.departure_utc AT TIME ZONE a.time_zone, 'HH24:MI') AS local_clock,
       (extract(epoch FROM ((f.departure_utc AT TIME ZONE a.time_zone)
                          - (f.departure_utc AT TIME ZONE 'UTC'))) / 3600)::int AS utc_offset_hours
  FROM flight_instances f JOIN airports a ON a.airport_code = f.origin ORDER BY f.flight_id"""


def connectable_pairs() -> str:
    """Correct: pairs that connect on instants with the airport minimum (>=)."""
    return """
SELECT a.flight_id AS from_flight, b.flight_id AS to_flight
  FROM flight_instances a JOIN flight_instances b ON b.origin = a.destination
  JOIN connection_rules r ON r.airport_code = a.destination
 WHERE b.departure_utc - a.arrival_utc >= make_interval(mins => r.minimum_minutes)
 ORDER BY 1, 2"""


def connectable_pairs_by_local_clock() -> str:
    """Wrong: orders flights by their displayed local departure clock strings."""
    return """
SELECT a.flight_id AS from_flight, b.flight_id AS to_flight
  FROM flight_instances a JOIN airports za ON za.airport_code = a.origin
  JOIN flight_instances b ON b.origin = a.destination
  JOIN airports zb ON zb.airport_code = b.origin
 WHERE to_char(a.departure_utc AT TIME ZONE za.time_zone, 'HH24:MI')
     < to_char(b.departure_utc AT TIME ZONE zb.time_zone, 'HH24:MI')
 ORDER BY 1, 2"""


# ----------------------------------------------------------------------------- L3


def _routes_body(as_of: str, *, with_capacity: bool = True) -> str:
    capacity = "AND s.sellable_free >= 1" if with_capacity else ""
    return f"""
WITH {_seats_cte(as_of)},
usable AS (
  SELECT f.flight_id, f.origin, f.destination, f.departure_utc, f.arrival_utc
    FROM flight_instances f JOIN seats s ON s.flight_id = f.flight_id
   WHERE f.status = 'scheduled' AND f.departure_utc >= {_as_of(as_of)} {capacity}),
paths AS (
  SELECT ARRAY[a.flight_id] AS legs, a.arrival_utc AS arrival FROM usable a
   WHERE a.origin = 'SEA' AND a.destination = 'JFK'
  UNION ALL
  SELECT ARRAY[a.flight_id, b.flight_id], b.arrival_utc FROM usable a
    JOIN usable b ON b.origin = a.destination AND a.flight_id <> b.flight_id
    JOIN connection_rules r ON r.airport_code = a.destination
   WHERE a.origin = 'SEA' AND b.destination = 'JFK'
     AND b.departure_utc - a.arrival_utc >= make_interval(mins => r.minimum_minutes)),
routes AS (
  SELECT legs, arrival FROM paths WHERE arrival <= '2026-11-01T18:00:00Z'::timestamptz)"""


def candidate_routes(as_of: str = dataset.AS_OF) -> str:
    return (
        _routes_body(as_of)
        + " SELECT legs, to_char(arrival AT TIME ZONE 'UTC', 'HH24:MI') AS arrival_utc "
        "FROM routes ORDER BY arrival, array_length(legs, 1), legs"
    )


def candidate_routes_ignoring_capacity(as_of: str = dataset.AS_OF) -> str:
    """Wrong: capacity-blind enumeration offers F6->F7 (blocked by hold H3)."""
    return (
        _routes_body(as_of, with_capacity=False)
        + " SELECT legs, to_char(arrival AT TIME ZONE 'UTC', 'HH24:MI') AS arrival_utc "
        "FROM routes ORDER BY arrival, array_length(legs, 1), legs"
    )


def independent_plan(as_of: str = dataset.AS_OF) -> str:
    """Wrong: each disrupted traveller independently takes the earliest route (both F5)."""
    return f"""
WITH aff AS ({affected()}),
rts AS ({candidate_routes(as_of)})
SELECT DISTINCT ON (a.booking_id) a.booking_id, r.legs
  FROM aff a CROSS JOIN rts r
 ORDER BY a.booking_id, r.arrival_utc, array_length(r.legs, 1), r.legs"""


def greedy_priority_plan(
    affected_rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    seat_rows_: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Priority-ordered allocation over ACTUAL query rows with shared remaining capacity.

    Higher priority first, earliest arrival, fewer legs, lexicographic route; every flight
    leg decrements shared sellable seats, so two travellers cannot both take the last F5
    seat. (A planning helper; correctness is judged by ``domain_oracle.plan_violations``.)
    """
    remaining = {r["flight_id"]: int(r["sellable_free"]) for r in seat_rows_}
    ordered = sorted(affected_rows, key=lambda r: (-int(r["service_priority"]), r["booking_id"]))
    plan: dict[str, tuple[str, ...]] = {}
    for person in ordered:
        for route in route_rows:  # already sorted by arrival, leg count, legs
            legs = tuple(route["legs"])
            if all(remaining[leg] >= 1 for leg in legs):
                for leg in legs:
                    remaining[leg] -= 1
                plan[person["booking_id"]] = legs
                break
    return plan


def policy_ids_used(legs: Sequence[str]) -> str:
    """Validate the minimum-connection policy ID(s) for a multi-leg route from the DB."""
    flight_list = ", ".join("'" + leg + "'" for leg in legs)
    return f"""
SELECT DISTINCT r.rule_id FROM flight_instances f
  JOIN connection_rules r ON r.airport_code = f.destination
 WHERE f.flight_id IN ({flight_list}) AND f.flight_id <> '{legs[-1]}' ORDER BY r.rule_id"""
