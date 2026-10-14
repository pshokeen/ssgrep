"""Reference transactional rebooking operation (L4) installed by the lab as plpgsql.

Protocol (READ COMMITTED; every writer, including the hold writer, follows it):
  1. transaction-scoped advisory locks, always request key then booking;
  2. idempotency lookup: same key + same digest returns the original outcome, same key
     + different digest is refused; neither touches business rows;
  3. inventory rows of every old *and* new flight/cabin locked ``FOR NO KEY UPDATE`` in
     one deterministic ``ORDER BY flight_id, cabin``;
  4. a FRESH capacity check (capacity - other occupants - live holds) *after* the locks,
     so a competitor that committed while we waited is seen;
  5. every coupon move applied in the one transaction, leftover legs voided, the request
     outcome logged in that same transaction; payments are never written.

Deliberately broken variants exist ONLY so the verifiers can prove they notice them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from . import dataset
from .database import WRITER

SCHEMA = dataset.SCHEMA
FUNCTION = f"{SCHEMA}.rebook_itinerary"
LOG_FUNCTION = f"{SCHEMA}.rebook_log"
HOLD_FUNCTION = f"{SCHEMA}.place_hold"
SIGNATURE = "text, text, text, text[], jsonb, timestamptz, text"

VARIANTS = (
    "correct", "no_lock", "no_revalidation", "no_capacity_check", "half_applied", "swallow_failure",
)  # fmt: skip


def quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class RebookRequest:
    """Client payload: booking, expected original coupon state/version, chosen itinerary."""

    request_key: str
    booking_id: str
    itinerary: tuple[str, ...]
    expected: tuple[dict[str, Any], ...]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "itinerary": list(self.itinerary),
            "expected": list(self.expected),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(dataset.canonical_json(self.payload)).hexdigest()

    def with_itinerary(self, itinerary: tuple[str, ...]) -> RebookRequest:
        return RebookRequest(self.request_key, self.booking_id, itinerary, self.expected)

    def call(self, *, digest: str | None = None, as_of: str = dataset.AS_OF) -> str:
        """The single SELECT a client runs inside its READ COMMITTED transaction."""
        itinerary = ", ".join(quote(f) for f in self.itinerary)
        expected = quote(json.dumps(list(self.expected), sort_keys=True, separators=(",", ":")))
        return (
            f"SELECT {FUNCTION}({quote(self.request_key)}, {quote(digest or self.digest)}, "
            f"{quote(self.booking_id)}, ARRAY[{itinerary}]::text[], {expected}::jsonb, "
            f"{quote(as_of)}::timestamptz, 'economy');"
        )


def expected_from_coupons(
    coupons: list[dict[str, Any]], ticket_id: str
) -> tuple[dict[str, Any], ...]:
    """The expected original coupon state a request must carry (ordered by leg)."""
    mine = sorted((c for c in coupons if c["ticket_id"] == ticket_id), key=lambda c: c["leg_no"])
    return tuple(
        {"coupon_id": c["coupon_id"], "flight_id": c["flight_id"], "status": c["status"],
         "version": c["version"]}
        for c in mine
    )  # fmt: skip


def request_for(
    state: dict[str, list[dict[str, Any]]],
    request_key: str,
    booking_id: str,
    itinerary: tuple[str, ...],
) -> RebookRequest:
    ticket = next(t for t in state["tickets"] if t["booking_id"] == booking_id)
    expected = expected_from_coupons(state["coupons"], ticket["ticket_id"])
    return RebookRequest(request_key, booking_id, itinerary, expected)


def place_hold_call(hold_id: str, flight_id: str, quantity: int, expires: str) -> str:
    return (
        f"SELECT {HOLD_FUNCTION}({quote(hold_id)}, {quote(flight_id)}, 'economy', "
        f"{int(quantity)}, {quote(expires)}::timestamptz, {quote(dataset.AS_OF)}::timestamptz);"
    )


_LOG_SQL = f"""
CREATE OR REPLACE FUNCTION {LOG_FUNCTION}(
  p_key text, p_booking text, p_digest text, p_state text, p_resp jsonb, p_at timestamptz
) RETURNS jsonb LANGUAGE plpgsql SET search_path = {SCHEMA}, pg_catalog AS $fn$
BEGIN
  INSERT INTO rebooking_requests (request_key, booking_id, payload_digest, state,
                                  response_json, created_at)
  VALUES (p_key, p_booking, p_digest, p_state, p_resp, p_at);
  RETURN p_resp || jsonb_build_object('replayed', false);
END $fn$;
"""

_HOLD_SQL = f"""
CREATE OR REPLACE FUNCTION {HOLD_FUNCTION}(
  p_hold text, p_flight text, p_cabin text, p_qty integer, p_expires timestamptz,
  p_as_of timestamptz
) RETURNS jsonb LANGUAGE plpgsql SET search_path = {SCHEMA}, pg_catalog AS $fn$
DECLARE
  v_capacity integer;
  v_occupied integer;
  v_holds integer;
BEGIN
  -- Hold writers obey the same inventory-row lock protocol as the rebooking operation.
  SELECT capacity INTO v_capacity FROM inventory
   WHERE flight_id = p_flight AND cabin = p_cabin FOR NO KEY UPDATE;
  SELECT count(*) INTO v_occupied FROM coupons c
    JOIN tickets t ON t.ticket_id = c.ticket_id JOIN bookings b ON b.booking_id = t.booking_id
   WHERE c.flight_id = p_flight AND c.cabin = p_cabin AND c.status = 'reserved'
     AND t.status = 'open' AND b.status = 'confirmed';
  SELECT coalesce(sum(quantity), 0) INTO v_holds FROM seat_holds
   WHERE flight_id = p_flight AND cabin = p_cabin AND status = 'active' AND expires_at > p_as_of;
  IF v_capacity - v_occupied - v_holds < p_qty THEN
    RETURN jsonb_build_object('outcome', 'rejected', 'reason', 'capacity_conflict');
  END IF;
  INSERT INTO seat_holds (hold_id, flight_id, cabin, quantity, status, expires_at)
  VALUES (p_hold, p_flight, p_cabin, p_qty, 'active', p_expires);
  RETURN jsonb_build_object('outcome', 'accepted', 'hold_id', p_hold);
END $fn$;
"""

_CAPACITY_CHECK = """
  v_full := NULL;
  SELECT t.fid INTO v_full FROM unnest(p_itinerary) AS t(fid)
   WHERE (SELECT i.capacity FROM inventory i WHERE i.flight_id = t.fid AND i.cabin = p_cabin)
       - (SELECT count(*) FROM coupons c
            JOIN tickets tk ON tk.ticket_id = c.ticket_id
            JOIN bookings b ON b.booking_id = tk.booking_id
           WHERE c.flight_id = t.fid AND c.cabin = p_cabin AND c.status = 'reserved'
             AND tk.status = 'open' AND b.status = 'confirmed'
             AND c.ticket_id <> v_ticket.ticket_id)
       - (SELECT coalesce(sum(h.quantity), 0) FROM seat_holds h
           WHERE h.flight_id = t.fid AND h.cabin = p_cabin AND h.status = 'active'
             AND h.expires_at > p_as_of) < 1
   ORDER BY t.fid LIMIT 1;
  IF v_full IS NOT NULL THEN
    RETURN rebook_log(p_request_key, p_booking_id, p_payload_digest, 'rejected',
      jsonb_build_object('outcome', 'rejected', 'reason', 'capacity_conflict',
                         'flight_id', v_full), p_as_of);
  END IF;
"""

_LOCK_INVENTORY = """
  PERFORM 1 FROM inventory
   WHERE (flight_id, cabin) IN (SELECT u.fid, p_cabin FROM unnest(v_lock_set) AS u(fid))
   ORDER BY flight_id, cabin
     FOR NO KEY UPDATE;
"""

_MOVE = """
  FOR i IN 1..{legs} LOOP
    {open}UPDATE coupons SET flight_id = p_itinerary[i], status = 'reserved', version = version + 1
     WHERE ticket_id = v_ticket.ticket_id AND leg_no = i;
    {close}
  END LOOP;"""

_VOID = """
  UPDATE coupons SET status = 'void', version = version + 1
   WHERE ticket_id = v_ticket.ticket_id AND leg_no > array_length(p_itinerary, 1)
     AND status = 'reserved';"""


def _move_sql(variant: str) -> str:
    if variant == "half_applied":  # moves only the first leg, never voids the leftover
        return _MOVE.format(legs="1", open="", close="") + "\n"
    if variant == "swallow_failure":  # catch-and-ignore around each leg update
        return (
            _MOVE.format(
                legs="array_length(p_itinerary, 1)",
                open="BEGIN\n    ",
                close="EXCEPTION WHEN OTHERS THEN NULL;\n    END;",
            )
            + _VOID
        )
    return _MOVE.format(legs="array_length(p_itinerary, 1)", open="", close="") + _VOID


def _operation_sql(variant: str) -> str:
    lock = "" if variant == "no_lock" else _LOCK_INVENTORY
    before, after = ("", _CAPACITY_CHECK)
    if variant == "no_revalidation":  # checked before the locks, never again after them
        before, after = _CAPACITY_CHECK, ""
    elif variant == "no_capacity_check":
        before = after = ""
    return f"""
CREATE OR REPLACE FUNCTION {FUNCTION}(
  p_request_key text, p_payload_digest text, p_booking_id text, p_itinerary text[],
  p_expected jsonb, p_as_of timestamptz, p_cabin text DEFAULT 'economy'
) RETURNS jsonb LANGUAGE plpgsql SET search_path = {SCHEMA}, pg_catalog AS $fn$
DECLARE
  v_existing rebooking_requests%ROWTYPE;
  v_booking bookings%ROWTYPE;
  v_ticket tickets%ROWTYPE;
  v_current jsonb;
  v_lock_set text[];
  v_full text;
  v_found integer;
  v_bad integer;
  v_resp jsonb;
  v_origin text;
  v_dest text;
BEGIN
  -- 1. request coordination, always in the same order: request key, then booking
  PERFORM pg_advisory_xact_lock(hashtextextended('rebook-request:' || p_request_key, 0));
  PERFORM pg_advisory_xact_lock(hashtextextended('rebook-booking:' || p_booking_id, 0));

  -- 2. idempotency: same key+digest replays the original outcome, other digests refused
  SELECT * INTO v_existing FROM rebooking_requests WHERE request_key = p_request_key;
  IF FOUND THEN
    IF v_existing.payload_digest = p_payload_digest THEN
      RETURN v_existing.response_json || jsonb_build_object('replayed', true);
    END IF;
    RETURN jsonb_build_object('outcome', 'rejected', 'reason', 'idempotency_key_reuse',
                              'replayed', false);
  END IF;

  -- 3. booking, ticket and the expected original coupon state/version
  SELECT * INTO v_booking FROM bookings WHERE booking_id = p_booking_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'unknown booking %', p_booking_id USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_ticket FROM tickets WHERE booking_id = p_booking_id;
  IF v_booking.status <> 'confirmed' OR v_ticket.status <> 'open' THEN
    RETURN rebook_log(p_request_key, p_booking_id, p_payload_digest, 'rejected',
      jsonb_build_object('outcome', 'rejected', 'reason', 'booking_cancelled'), p_as_of);
  END IF;
  SELECT coalesce(jsonb_agg(jsonb_build_object('coupon_id', coupon_id, 'flight_id', flight_id,
           'status', status, 'version', version) ORDER BY leg_no), '[]'::jsonb)
    INTO v_current FROM coupons WHERE ticket_id = v_ticket.ticket_id;
  IF v_current IS DISTINCT FROM p_expected THEN
    RETURN rebook_log(p_request_key, p_booking_id, p_payload_digest, 'rejected',
      jsonb_build_object('outcome', 'rejected', 'reason', 'coupon_state_mismatch'), p_as_of);
  END IF;

  -- 4. itinerary validity on instants: scheduled, departs at/after as_of, same endpoints,
  --    every connection at least the airport minimum (exactly the minimum is valid)
  SELECT f.origin INTO v_origin FROM coupons c JOIN flight_instances f ON f.flight_id = c.flight_id
   WHERE c.ticket_id = v_ticket.ticket_id AND c.status = 'reserved' ORDER BY c.leg_no LIMIT 1;
  SELECT f.destination INTO v_dest FROM coupons c JOIN flight_instances f
      ON f.flight_id = c.flight_id
   WHERE c.ticket_id = v_ticket.ticket_id AND c.status = 'reserved'
   ORDER BY c.leg_no DESC LIMIT 1;
  SELECT count(*) INTO v_found FROM unnest(p_itinerary) AS u(fid)
    JOIN flight_instances f ON f.flight_id = u.fid;
  IF array_length(p_itinerary, 1) IS NULL
     OR array_length(p_itinerary, 1) NOT BETWEEN 1 AND 2
     OR array_length(p_itinerary, 1) > jsonb_array_length(v_current)
     OR v_found <> array_length(p_itinerary, 1) THEN
    RETURN rebook_log(p_request_key, p_booking_id, p_payload_digest, 'rejected',
      jsonb_build_object('outcome', 'rejected', 'reason', 'invalid_itinerary'), p_as_of);
  END IF;
  SELECT count(*) INTO v_bad FROM (
    SELECT 1 FROM unnest(p_itinerary) AS u(fid) JOIN flight_instances f ON f.flight_id = u.fid
     WHERE f.status <> 'scheduled' OR f.departure_utc < p_as_of
    UNION ALL
    SELECT 1
      FROM unnest(p_itinerary) WITH ORDINALITY AS a(fid, n)
      JOIN unnest(p_itinerary) WITH ORDINALITY AS b(fid, n) ON b.n = a.n + 1
      JOIN flight_instances fa ON fa.flight_id = a.fid
      JOIN flight_instances fb ON fb.flight_id = b.fid
      JOIN connection_rules r ON r.airport_code = fa.destination
     WHERE fa.destination <> fb.origin
        OR fb.departure_utc - fa.arrival_utc < make_interval(mins => r.minimum_minutes)
    UNION ALL
    SELECT 1 FROM flight_instances f WHERE f.flight_id = p_itinerary[1] AND f.origin <> v_origin
    UNION ALL
    SELECT 1 FROM flight_instances f
     WHERE f.flight_id = p_itinerary[array_length(p_itinerary, 1)] AND f.destination <> v_dest
  ) bad;
  IF v_bad > 0 THEN
    RETURN rebook_log(p_request_key, p_booking_id, p_payload_digest, 'rejected',
      jsonb_build_object('outcome', 'rejected', 'reason', 'invalid_itinerary'), p_as_of);
  END IF;
{before}
  -- 5. deterministic inventory-row locks over old and new flight/cabin keys
  SELECT array_agg(fid ORDER BY fid) INTO v_lock_set FROM (
    SELECT c.flight_id AS fid FROM coupons c WHERE c.ticket_id = v_ticket.ticket_id
    UNION SELECT unnest(p_itinerary)) s;
{lock}
  -- 6. fresh capacity check under the locks
{after}
  -- 7. atomic multi-leg move, request outcome logged in the same transaction
{_move_sql(variant)}

  SELECT jsonb_build_object('outcome', 'accepted', 'booking_id', p_booking_id,
           'itinerary', to_jsonb(p_itinerary),
           'coupons', coalesce(jsonb_agg(jsonb_build_object('coupon_id', coupon_id,
              'flight_id', flight_id, 'status', status, 'version', version) ORDER BY leg_no),
              '[]'::jsonb))
    INTO v_resp FROM coupons WHERE ticket_id = v_ticket.ticket_id;
  RETURN rebook_log(p_request_key, p_booking_id, p_payload_digest, 'accepted', v_resp, p_as_of);
END $fn$;
"""


def install_sql(variant: str = "correct") -> str:
    """SQL (run as the verifier) installing the operation, its helpers and grants."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown L4 variant {variant!r}")
    grants = f"""
REVOKE ALL ON FUNCTION {FUNCTION}({SIGNATURE}) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION {FUNCTION}({SIGNATURE}) TO {WRITER};
REVOKE ALL ON FUNCTION {LOG_FUNCTION}(text, text, text, text, jsonb, timestamptz) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION {LOG_FUNCTION}(text, text, text, text, jsonb, timestamptz) TO {WRITER};
REVOKE ALL ON FUNCTION {HOLD_FUNCTION}(text, text, text, integer, timestamptz, timestamptz)
  FROM PUBLIC;
"""
    return "\n".join((_LOG_SQL, _HOLD_SQL, _operation_sql(variant), grants))
