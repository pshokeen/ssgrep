"""Independent reference model for the airline fixture (pure Python, no SQL).

Everything here is computed from fixture rows (``dataset.seed_state()`` or a state read
back from the database) with separately written logic. It never inspects an agent's SQL
text: callers hand it the *actual result rows* of a query and it returns violations.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from . import dataset

State = Mapping[str, Sequence[Mapping[str, Any]]]
AS_OF = datetime(2026, 11, 1, 7, 0, tzinfo=UTC)
ARRIVE_BY = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
ORIGIN, DESTINATION = "SEA", "JFK"

# Canonical answers transcribed from validation-lab.md section 3 (L1/L2/L3). Tests assert
# that the *computed* model below agrees with these literals, and verifiers compare actual
# database results with the computed values.
CANON_CASH = {
    ("B1", "USD"): 45000,
    ("B2", "USD"): 50000,
    ("B3", "USD"): 0,
    ("B4", "USD"): 50000,
    ("B5", "EUR"): 45000,
    ("B6", "USD"): 50000,
}
CANON_SELLABLE = {"F1": 0, "F2": 0, "F3": 1, "F4": 1, "F5": 1, "F6": 0, "F7": 1, "F8": 2}
CANON_SEATS = {
    "F1": (2, 0, 1, 0),
    "F2": (2, 1, 0, 0),
    "F3": (1, 0, 1, 1),
    "F4": (1, 0, 1, 1),
    "F5": (1, 0, 1, 1),
    "F6": (1, 1, 0, 0),
    "F7": (1, 0, 1, 1),
    "F8": (0, 0, 2, 2),
}  # occupied, live holds, raw free, sellable free
CANON_AFFECTED = (("B1", "V1"), ("B2", "V2"))
CANON_ROUTES = (("F5",), ("F3", "F4"))
CANON_PLAN = {"B1": ("F5",), "B2": ("F3", "F4")}


class IntegrityError(AssertionError):
    """The state violates an invariant the lab refuses to hide (e.g. negative raw free)."""


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    return {r[key]: r for r in rows}


# ---------------------------------------------------------------- L1 cash and seats


def settled_cash(state: State) -> dict[tuple[str, str], int]:
    """Settled minor units per (booking, currency); the ticket currency always appears."""
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for ticket in state["tickets"]:
        totals[(ticket["booking_id"], ticket["currency"])] += 0
    for pay in state["payment_events"]:
        if pay["status"] == "settled":
            totals[(pay["booking_id"], pay["currency"])] += pay["signed_amount_minor"]
    return dict(totals)


def currency_totals(cash: Mapping[tuple[str, str], int]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for (_, currency), minor in cash.items():
        out[currency] += minor
    return dict(out)


def _occupying_coupons(state: State) -> list[Mapping[str, Any]]:
    tickets = _by(state["tickets"], "ticket_id")
    bookings = _by(state["bookings"], "booking_id")
    return [
        c
        for c in state["coupons"]
        if c["status"] == "reserved"
        and tickets[c["ticket_id"]]["status"] == "open"
        and bookings[tickets[c["ticket_id"]]["booking_id"]]["status"] == "confirmed"
    ]


def live_holds(state: State, flight_id: str, cabin: str, as_of: datetime) -> int:
    """A hold counts only while ``active`` and strictly before expiry (equality is expired)."""
    return sum(
        h["quantity"]
        for h in state["seat_holds"]
        if h["flight_id"] == flight_id
        and h["cabin"] == cabin
        and h["status"] == "active"
        and parse_utc(h["expires_at"]) > as_of
    )


@dataclass(frozen=True)
class SeatRow:
    flight_id: str
    capacity: int
    occupied: int
    live_holds: int
    raw_free: int  # capacity - occupied - live holds; negative is an integrity failure
    sellable_free: int

    def key(self) -> tuple[int, int, int, int]:
        return (self.occupied, self.live_holds, self.raw_free, self.sellable_free)


def seat_rows(state: State, as_of: datetime = AS_OF, cabin: str = dataset.CABIN) -> list[SeatRow]:
    occupancy: dict[tuple[str, str], int] = defaultdict(int)
    for coupon in _occupying_coupons(state):
        occupancy[(coupon["flight_id"], coupon["cabin"])] += 1
    flights = _by(state["flight_instances"], "flight_id")
    rows = []
    for inv in sorted(state["inventory"], key=lambda r: r["flight_id"]):
        if inv["cabin"] != cabin:
            continue
        fid = inv["flight_id"]
        occupied = occupancy[(fid, cabin)]
        holds = live_holds(state, fid, cabin, as_of)
        raw_free = inv["capacity"] - occupied - holds
        if raw_free < 0:
            raise IntegrityError(f"{fid}: negative raw capacity {raw_free} (oversold)")
        sellable = 0 if flights[fid]["status"] != "scheduled" else raw_free
        rows.append(SeatRow(fid, inv["capacity"], occupied, holds, raw_free, sellable))
    return sorted(rows, key=lambda r: int(r.flight_id[1:]))


def sellable_map(state: State, as_of: datetime = AS_OF) -> dict[str, int]:
    return {r.flight_id: r.sellable_free for r in seat_rows(state, as_of)}


# ---------------------------------------------------------------- L2 itineraries


def _flight_times(state: State) -> dict[str, tuple[datetime, datetime]]:
    return {
        f["flight_id"]: (parse_utc(f["departure_utc"]), parse_utc(f["arrival_utc"]))
        for f in state["flight_instances"]
    }


def minimum_connection(state: State) -> dict[str, tuple[str, int]]:
    """airport -> (rule_id, minimum minutes)."""
    return {
        r["airport_code"]: (r["rule_id"], r["minimum_minutes"]) for r in state["connection_rules"]
    }


def gap_minutes(arrival: datetime, departure: datetime) -> int:
    delta = (departure - arrival).total_seconds()
    if delta % 60:
        raise IntegrityError("fractional-minute connection")
    return int(delta // 60)


def connection_valid(gap: int, minimum: int) -> bool:
    """Equality is valid: exactly the minimum connects."""
    return gap >= minimum


@dataclass(frozen=True)
class Connection:
    booking_id: str
    from_flight: str
    to_flight: str
    airport: str
    gap_minutes: int
    rule_id: str
    minimum_minutes: int
    valid: bool

    def key(self) -> tuple[Any, ...]:
        return (self.booking_id, self.from_flight, self.to_flight, self.airport,
                self.gap_minutes, self.rule_id, self.minimum_minutes, self.valid)  # fmt: skip


def connections(state: State) -> list[Connection]:
    """Connections between consecutive *reserved* coupons of every open ticket."""
    times = _flight_times(state)
    flights = _by(state["flight_instances"], "flight_id")
    mct = minimum_connection(state)
    tickets = _by(state["tickets"], "ticket_id")
    per_ticket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for coupon in state["coupons"]:
        if coupon["status"] == "reserved":
            per_ticket[coupon["ticket_id"]].append(coupon)
    out = []
    for ticket_id, coupons in sorted(per_ticket.items()):
        coupons.sort(key=lambda c: c["leg_no"])
        for first, second in itertools.pairwise(coupons):
            if tickets[ticket_id]["status"] != "open":
                continue
            airport = flights[first["flight_id"]]["destination"]
            if flights[second["flight_id"]]["origin"] != airport:
                raise IntegrityError(f"{ticket_id}: legs do not connect at {airport}")
            gap = gap_minutes(times[first["flight_id"]][1], times[second["flight_id"]][0])
            rule, minimum = mct[airport]
            out.append(
                Connection(
                    tickets[ticket_id]["booking_id"],
                    first["flight_id"],
                    second["flight_id"],
                    airport,
                    gap,
                    rule,
                    minimum,
                    connection_valid(gap, minimum),
                )  # fmt: skip
            )
    return sorted(out, key=lambda c: c.booking_id)


def affected_travellers(state: State) -> list[tuple[str, str]]:
    """Confirmed, open-ticket travellers with a cancelled flight or a broken connection."""
    flights = _by(state["flight_instances"], "flight_id")
    tickets = {t["booking_id"]: t for t in state["tickets"]}
    bad_connection = {c.booking_id for c in connections(state) if not c.valid}
    out = []
    for booking in sorted(state["bookings"], key=lambda b: b["booking_id"]):
        ticket = tickets[booking["booking_id"]]
        if booking["status"] != "confirmed" or ticket["status"] != "open":
            continue
        coupons = [
            c for c in state["coupons"] if c["ticket_id"] == ticket["ticket_id"]
            if c["status"] == "reserved"
        ]  # fmt: skip
        cancelled = any(flights[c["flight_id"]]["status"] == "cancelled" for c in coupons)
        if cancelled or booking["booking_id"] in bad_connection:
            out.append((booking["booking_id"], ticket["traveller_id"]))
    return out


def local_departure(state: State, flight_id: str) -> tuple[str, str]:
    """(displayed local wall clock, UTC offset) of a flight's departure."""
    flights = _by(state["flight_instances"], "flight_id")
    zones = _by(state["airports"], "airport_code")
    flight = flights[flight_id]
    zone = ZoneInfo(zones[flight["origin"]]["time_zone"])
    local = parse_utc(flight["departure_utc"]).astimezone(zone)
    return local.strftime("%H:%M"), local.strftime("%z")


def local_offset_hours(state: State, flight_id: str) -> int:
    """Whole-hour UTC offset of the flight's departure airport at its departure instant."""
    flights = _by(state["flight_instances"], "flight_id")
    zones = _by(state["airports"], "airport_code")
    zone = ZoneInfo(zones[flights[flight_id]["origin"]]["time_zone"])
    offset = parse_utc(flights[flight_id]["departure_utc"]).astimezone(zone).utcoffset()
    assert offset is not None
    return int(offset.total_seconds() // 3600)


def connectable_pairs(state: State) -> set[tuple[str, str]]:
    """Every (inbound, outbound) physical pair meeting airport and minimum rules."""
    ids = sorted(f["flight_id"] for f in state["flight_instances"])
    return {(a, b) for a in ids for b in ids if a != b and can_connect(state, a, b)}


def instant_gap_minutes(state: State, first: str, second: str) -> int:
    """Actual elapsed minutes between two departures (instants, never local strings)."""
    times = _flight_times(state)
    return int((times[second][0] - times[first][0]).total_seconds() // 60)


def policy_ids(state: State, legs: Sequence[str]) -> list[str]:
    """Connection-policy IDs a multi-leg route must be validated against (sorted)."""
    flights = _by(state["flight_instances"], "flight_id")
    mct = minimum_connection(state)
    return sorted({mct[flights[leg]["destination"]][0] for leg in legs[:-1]})


def can_connect(state: State, inbound: str, outbound: str) -> bool:
    flights = _by(state["flight_instances"], "flight_id")
    times = _flight_times(state)
    if flights[inbound]["destination"] != flights[outbound]["origin"]:
        return False
    airport = flights[inbound]["destination"]
    minimum = minimum_connection(state)[airport][1]
    return connection_valid(gap_minutes(times[inbound][1], times[outbound][0]), minimum)


# ---------------------------------------------------------------- L3 planning


@dataclass(frozen=True)
class Route:
    legs: tuple[str, ...]
    arrival: datetime

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self.arrival, len(self.legs), self.legs)


def _route_rejection(
    state: State, legs: tuple[str, ...], sellable: Mapping[str, int]
) -> str | None:
    """First reason a path is not bookable; structural/temporal rules before capacity."""
    flights = _by(state["flight_instances"], "flight_id")
    times = _flight_times(state)
    airports = [flights[legs[0]]["origin"]]
    for leg in legs:
        if flights[leg]["status"] != "scheduled":
            return f"{leg}: not scheduled"
        airports.append(flights[leg]["destination"])
        if times[leg][0] < AS_OF:
            return f"{leg}: departs before as_of"
    if len(set(airports)) != len(airports):
        return "repeated airport"
    for inbound, outbound in itertools.pairwise(legs):
        if not can_connect(state, inbound, outbound):
            return f"{inbound}->{outbound}: invalid connection"
    if times[legs[-1]][1] > ARRIVE_BY:
        return "arrives after 18:00Z"
    for leg in legs:
        if sellable[leg] < 1:
            return f"{leg}: no sellable seat"
    return None


def route_candidates(state: State) -> tuple[list[Route], dict[tuple[str, ...], str]]:
    """Feasible SEA->JFK routes of at most two physical legs and every rejected path."""
    flights = _by(state["flight_instances"], "flight_id")
    times = _flight_times(state)
    sellable = sellable_map(state)
    ids = sorted(flights)
    paths: list[tuple[str, ...]] = []
    for first in ids:
        if flights[first]["origin"] != ORIGIN:
            continue
        if flights[first]["destination"] == DESTINATION:
            paths.append((first,))
        for second in ids:
            if (
                second != first
                and flights[second]["origin"] == flights[first]["destination"]
                and flights[second]["destination"] == DESTINATION
            ):
                paths.append((first, second))
    feasible: list[Route] = []
    rejected: dict[tuple[str, ...], str] = {}
    for legs in paths:
        reason = _route_rejection(state, legs, sellable)
        if reason:
            rejected[legs] = reason
        else:
            feasible.append(Route(legs, times[legs[-1]][1]))
    return sorted(feasible, key=lambda r: r.sort_key), rejected


def joint_plan(state: State) -> dict[str, tuple[str, ...]]:
    """Best shared-capacity assignment for the disrupted travellers.

    Objective in order: most travellers served, priority-respecting arrivals (higher
    priority never arrives later than a lower one), minimum aggregate arrival, fewest
    legs, then lexicographic flight tuples. Seats are allocated jointly: a route that is
    individually feasible is rejected when the shared flight is already spoken for.
    """
    routes, _ = route_candidates(state)
    sellable = sellable_map(state)
    bookings = _by(state["bookings"], "booking_id")
    people = [b for b, _ in affected_travellers(state)]
    options: list[Route | None] = [None, *routes]
    best_key = None
    best: dict[str, tuple[str, ...]] = {}
    for combo in itertools.product(options, repeat=len(people)):
        used: dict[str, int] = defaultdict(int)
        for route in combo:
            for leg in route.legs if route else ():
                used[leg] += 1
        if any(used[f] > sellable[f] for f in used):
            continue
        served = [(p, r) for p, r in zip(people, combo, strict=True) if r is not None]
        violations = sum(
            1
            for (pa, ra), (pb, rb) in itertools.permutations(served, 2)
            if bookings[pa]["service_priority"] > bookings[pb]["service_priority"]
            and ra.arrival > rb.arrival
        )
        key = (
            -len(served),
            violations,
            sum(r.arrival.timestamp() for _, r in served),
            sum(len(r.legs) for _, r in served),
            tuple((p, r.legs) for p, r in served),
        )
        if best_key is None or key < best_key:
            best_key, best = key, {p: r.legs for p, r in served}
    return best


def plan_violations(state: State, plan: Mapping[str, Sequence[str]]) -> list[str]:
    """Reject plans that oversell shared capacity or use an infeasible route."""
    problems = []
    sellable = sellable_map(state)
    routes, _ = route_candidates(state)
    feasible = {r.legs for r in routes}
    used: dict[str, int] = defaultdict(int)
    for booking, legs in plan.items():
        if tuple(legs) not in feasible:
            problems.append(
                f"{booking}: route {tuple(legs)} is not a feasible single-traveller path"
            )
        for leg in legs:
            used[leg] += 1
    for flight, count in sorted(used.items()):
        if count > sellable[flight]:
            problems.append(
                f"{flight}: {count} travellers assigned for {sellable[flight]} sellable"
            )
    expected = joint_plan(state)
    if not problems and {k: tuple(v) for k, v in plan.items()} != expected:
        problems.append(f"plan {dict(plan)} is not the priority-respecting optimum {expected}")
    return problems


# ---------------------------------------------------------------- result verifiers


def verify_cash(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    """Compare actual rows ``booking_id, currency, settled_minor`` with the oracle."""
    expected = settled_cash(state)
    actual: dict[tuple[str, str], int] = {}
    problems = []
    for row in rows:
        key = (row["booking_id"], row["currency"])
        if key in actual:
            problems.append(f"duplicate cash row {key}")
        actual[key] = int(row["settled_minor"])
    for key in sorted(expected.keys() - actual.keys()):
        problems.append(f"missing cash row {key}")
    for key in sorted(actual.keys() - expected.keys()):
        problems.append(f"unexpected cash row {key}")
    for key in sorted(expected.keys() & actual.keys()):
        if expected[key] != actual[key]:
            problems.append(f"cash {key}: expected {expected[key]} actual {actual[key]}")
    return problems


def verify_seats(
    rows: Iterable[Mapping[str, Any]], state: State, as_of: datetime = AS_OF
) -> list[str]:
    expected = {r.flight_id: r.key() for r in seat_rows(state, as_of)}
    actual: dict[str, tuple[int, ...]] = {}
    problems = []
    for row in rows:
        fid = row["flight_id"]
        if fid in actual:
            problems.append(f"duplicate seat row {fid}")
        actual[fid] = (
            int(row["occupied"]), int(row["live_holds"]), int(row["raw_free"]),
            int(row["sellable_free"]),
        )  # fmt: skip
    for fid in sorted(expected.keys() - actual.keys()):
        problems.append(f"missing seat row {fid}")
    for fid in sorted(actual.keys() - expected.keys()):
        problems.append(f"unexpected seat row {fid}")
    for fid in sorted(expected.keys() & actual.keys()):
        if tuple(expected[fid]) != actual[fid]:
            problems.append(f"seats {fid}: expected {expected[fid]} actual {actual[fid]}")
    return problems


def verify_affected(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    expected = affected_travellers(state)
    actual = sorted((r["booking_id"], r["traveller_id"]) for r in rows)
    if actual == sorted(expected):
        return []
    return [f"affected: expected {expected} actual {actual}"]


def verify_connections(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    expected = {c.key() for c in connections(state)}
    actual = {
        (r["booking_id"], r["from_flight"], r["to_flight"], r["airport"],
         int(r["gap_minutes"]), r["rule_id"], int(r["minimum_minutes"]), bool(r["valid"]))
        for r in rows
    }  # fmt: skip
    problems = [f"missing connection {c}" for c in sorted(expected - actual)]
    problems += [f"unexpected connection {c}" for c in sorted(actual - expected)]
    return problems


def verify_flight_instants(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    """Every physical flight must appear once with its own UTC departure instant."""
    expected = {f["flight_id"]: f["departure_utc"] for f in state["flight_instances"]}
    actual = {r["flight_id"]: r["departure_utc"] for r in rows}
    problems = [f"missing flight {f}" for f in sorted(expected.keys() - actual.keys())]
    problems += [
        f"flight {f}: expected {expected[f]} actual {actual[f]}"
        for f in sorted(expected.keys() & actual.keys())
        if expected[f] != actual[f]
    ]
    return problems


def verify_local_clock(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    """Displayed local clock plus UTC offset per flight (F1 -7h vs F8 -8h at SEA 01:30)."""
    expected = {
        f["flight_id"]: (*local_departure(state, f["flight_id"])[:1],
                         local_offset_hours(state, f["flight_id"]))
        for f in state["flight_instances"]
    }  # fmt: skip
    actual = {r["flight_id"]: (r["local_clock"], int(r["utc_offset_hours"])) for r in rows}
    problems = [f"missing flight {f}" for f in sorted(expected.keys() - actual.keys())]
    problems += [
        f"local clock {f}: expected {expected[f]} actual {actual[f]}"
        for f in sorted(expected.keys() & actual.keys())
        if expected[f] != actual[f]
    ]
    return problems


def verify_connectable(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    expected = connectable_pairs(state)
    actual = {(r["from_flight"], r["to_flight"]) for r in rows}
    problems = [f"missing connectable pair {p}" for p in sorted(expected - actual)]
    problems += [f"unexpected connectable pair {p}" for p in sorted(actual - expected)]
    return problems


def verify_routes(rows: Iterable[Mapping[str, Any]], state: State) -> list[str]:
    routes, _ = route_candidates(state)
    expected = {r.legs for r in routes}
    actual = {tuple(r["legs"]) for r in rows}
    problems = [f"missing candidate route {r}" for r in sorted(expected - actual)]
    problems += [f"unexpected candidate route {r}" for r in sorted(actual - expected)]
    return problems


# ---------------------------------------------------------------- L4 state oracle


@dataclass(frozen=True)
class MoveRequest:
    """One rebooking request in oracle terms (mirrors the lab's request payload)."""

    request_key: str
    booking_id: str
    itinerary: tuple[str, ...]


def apply_moves(state: State, moves: Iterable[MoveRequest]) -> dict[str, list[dict[str, Any]]]:
    """Business state after the given *successful* moves, computed without SQL.

    Each move re-points the booking's leg-n coupon to the n-th itinerary flight and voids
    leftover legs (the direct reissue voids C2). Versions bump by one; payments untouched.
    """
    new = {t: [dict(r) for r in rows] for t, rows in state.items()}
    tickets = {t["booking_id"]: t["ticket_id"] for t in state["tickets"]}
    for move in moves:
        coupons = sorted(
            (c for c in new["coupons"] if c["ticket_id"] == tickets[move.booking_id]),
            key=lambda c: c["leg_no"],
        )
        for coupon in coupons:
            index = coupon["leg_no"] - 1
            coupon["version"] += 1
            if index < len(move.itinerary):
                coupon["flight_id"] = move.itinerary[index]
            else:
                coupon["status"] = "void"
    return new


def state_after_joint_plan(state: State) -> dict[str, list[dict[str, Any]]]:
    """The canonical L4 end state, derived from the oracle's own plan (not hardcoded)."""
    plan = joint_plan(state)
    return apply_moves(
        state, [MoveRequest(f"req-{b}", b, legs) for b, legs in sorted(plan.items())]
    )


def verify_state_delta(
    actual: State,
    expected: State,
    *,
    request_tables: str = "rebooking_requests",
) -> list[str]:
    """Row-level comparison; the request log is compared separately by the L4 verifiers."""
    problems = []
    for table in dataset.TABLES:
        if table.name == request_tables:
            continue
        digest_actual = dataset.table_digest(table.name, actual[table.name])
        if digest_actual != dataset.table_digest(table.name, expected[table.name]):
            problems.append(f"table {table.name}: state differs from oracle")
    return problems
