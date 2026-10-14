"""Public business brief and the four user-turn prompts (L1-L4).

Shared by the scripted stand-in and the live omp lane so one set of hashes pins both. The text
carries objectives and the public schema only: no answer rows, no solved SQL, no policy
identifiers, no receipts, no rarity settings, and nothing about reuse of earlier history beyond
the instruction to investigate ssgrep history first (L3).
"""

from __future__ import annotations

from collections.abc import Mapping

TABLES = (
    "airports(airport_code, time_zone)",
    "flight_instances(flight_id, origin, destination, departure_utc, arrival_utc, status)",
    "marketing_flights(marketing_id, carrier, number, flight_id)",
    "bookings(booking_id, pnr, status, service_priority, external_customer_ref)",
    "travellers(traveller_id, display_name)",
    "tickets(ticket_id, booking_id, traveller_id, status, currency, face_value_minor)",
    "coupons(coupon_id, ticket_id, flight_id, leg_no, cabin, status, version)",
    "inventory(flight_id, cabin, capacity)",
    "payment_events(event_id, booking_id, currency, signed_amount_minor, type, status,"
    " provider_event_id)",
    "seat_holds(hold_id, flight_id, cabin, quantity, status, expires_at)",
    "connection_rules(rule_id, airport_code, minimum_minutes)",
    "rebooking_requests(request_key, booking_id, payload_digest, state, response_json, created_at)",
)

BRIEF = (
    "You are an operations analyst for a small airline. The reservation database is a "
    "PostgreSQL schema `airline` (search_path is already set). Tables:\n"
    + "\n".join(f"- {t}" for t in TABLES)
    + "\n\nConventions: money is integer minor units per currency (never merge currencies); "
    "all instants are timestamptz; the application clock is fixed at 2026-11-01T07:00:00Z "
    "(use it instead of now()); seats are economy only and are counted per PHYSICAL flight "
    "(marketing_flights are codeshare labels, not separate capacity). Run SQL with the "
    "`sql` tool: `sql --role reader --database <db> <<'EOF' ... EOF` (read-only). "
    "Answer with the exact values you verified by executing SQL."
)

L1 = (
    "Explain settled cash for each reservation and sellable economy inventory for every "
    "physical flight at the supplied clock. Demonstrate why joining payments, travellers, "
    "coupons, and codeshares directly can inflate the result. Include cancelled/void cases "
    "and currencies."
)

L2 = (
    "Identify confirmed travellers requiring reaccommodation, distinguish cancelled "
    "operations from invalid connection times, and establish valid minimum-connection "
    "boundaries with policy IDs. Explain the duplicated local departure time without "
    "treating it as a duplicate flight."
)

L3 = (
    "Use available ssgrep history to investigate established reservation rules (start with "
    '`ssgrep search "<question>" --json` and open relevant results with `ssgrep show '
    "<ref> --json`), then produce a joint plan for the disrupted confirmed bookings from "
    "SEA to JFK arriving by 18:00Z. Use at most two legs, scheduled physical flights only, "
    "no repeated airport, departure at/after the application clock, valid minimum "
    "connections, and available capacity on every new leg. Validate candidate itineraries, "
    "policy references, shared capacity, and priority (higher service_priority first, then "
    "earlier arrival, fewer legs, lexicographic flight ids); do not modify reservations yet. "
    "Validate any rule you reuse against the current database before relying on it."
)

L4 = (
    "Apply the joint plan from the previous turn atomically and idempotently as the "
    "restricted writer (`sql --role writer --database <db>`), using the installed operation "
    "`airline.rebook_itinerary(request_key text, payload_digest text, booking_id text, "
    "itinerary text[], expected_coupons jsonb, as_of timestamptz, cabin text)`. Use a new "
    "immutable request key per booking, carry the expected original coupon state/version, "
    "run each request in its own READ COMMITTED transaction, then verify the final rows "
    "(coupons, occupancy, request log) with SQL. Do not touch payments."
)

PROMPTS: Mapping[str, str] = {"brief": BRIEF, "L1": L1, "L2": L2, "L3": L3, "L4": L4}

# Strings that must never occur in any prompt (answer leakage guard, checked by tests).
FORBIDDEN_FRAGMENTS = ("MCT-", "45000", "195000", "B1/USD", "F5", "capacity_conflict")
