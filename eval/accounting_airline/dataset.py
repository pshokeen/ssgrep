"""Canonical synthetic airline fixture: 12 tables, 82 rows, DDL, seed SQL, digests.

The rows below are a literal transcription of ``validation-lab.md`` section 2. They
are hand-authored synthetic data (no real people, no real history). Everything the
oracle checks is derived from these tuples, never from agent SQL.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

IMAGE_DIGEST = "sha256:2d2b8998d31037bf721cfdf764d76ba74171b4fab3431b7f72c27c56ddbdf9e3"
IMAGE_REF = f"postgres@{IMAGE_DIGEST}"
SCHEMA = "airline"
AS_OF = "2026-11-01T07:00:00Z"
CABIN = "economy"

EXPECTED_COUNTS: dict[str, int] = {
    "airports": 5,
    "flight_instances": 8,
    "marketing_flights": 8,
    "bookings": 6,
    "travellers": 6,
    "tickets": 6,
    "coupons": 10,
    "inventory": 8,
    "payment_events": 10,
    "seat_holds": 5,
    "connection_rules": 5,
    "rebooking_requests": 5,
}
TOTAL_ROWS = 82

TS = "timestamptz"


def ts(hhmm: str) -> str:
    """UTC instant on the lab day, e.g. ``ts('08:30')``."""
    return f"2026-11-01T{hhmm}:00Z"


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[tuple[str, str], ...]  # (column, kind) kind in text/int/timestamptz/jsonb
    pk: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    def row_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.column_names, row, strict=True)) for row in self.rows]


def _rejection(reason: str) -> dict[str, str]:
    return {"outcome": "rejected", "reason": reason}


_SEED_REQUESTS = (
    ("R1", "B3", "booking_cancelled", "06:01"),
    ("R2", "B4", "no_matching_itinerary", "06:02"),
    ("R3", "B5", "currency_mismatch", "06:03"),
    ("R4", "B6", "hold_conflict", "06:04"),
    ("R5", "B1", "request_expired", "06:05"),
)


def seed_request_key(label: str) -> str:
    """Seed request keys live in their own ``seed:`` namespace; L4 keys never use it."""
    return f"seed:{label}"


def seed_request_digest(label: str) -> str:
    return hashlib.sha256(f"seed-payload-{label}".encode()).hexdigest()


TABLES: tuple[Table, ...] = (
    Table(
        "airports",
        (("airport_code", "text"), ("time_zone", "text")),
        ("airport_code",),
        (
            ("SEA", "America/Los_Angeles"),
            ("DEN", "America/Denver"),
            ("ORD", "America/Chicago"),
            ("JFK", "America/New_York"),
            ("SFO", "America/Los_Angeles"),
        ),
    ),
    Table(
        "flight_instances",
        (
            ("flight_id", "text"),
            ("origin", "text"),
            ("destination", "text"),
            ("departure_utc", TS),
            ("arrival_utc", TS),
            ("status", "text"),
        ),
        ("flight_id",),
        (
            ("F1", "SEA", "DEN", ts("08:30"), ts("11:00"), "cancelled"),
            ("F2", "DEN", "JFK", ts("11:35"), ts("15:00"), "scheduled"),
            ("F3", "SEA", "ORD", ts("09:00"), ts("13:00"), "scheduled"),
            ("F4", "ORD", "JFK", ts("13:45"), ts("16:00"), "scheduled"),
            ("F5", "SEA", "JFK", ts("10:00"), ts("15:30"), "scheduled"),
            ("F6", "SEA", "SFO", ts("09:30"), ts("11:30"), "scheduled"),
            ("F7", "SFO", "JFK", ts("12:15"), ts("18:00"), "scheduled"),
            ("F8", "SEA", "DEN", ts("09:30"), ts("12:00"), "scheduled"),
        ),
    ),
    Table(
        "marketing_flights",
        (
            ("marketing_id", "text"),
            ("carrier", "text"),
            ("flight_number", "int"),
            ("flight_id", "text"),
        ),
        ("marketing_id",),
        (
            ("M1", "AX", 101, "F1"),
            ("M2", "BX", 9101, "F1"),
            ("M3", "AX", 202, "F2"),
            ("M4", "BX", 9202, "F2"),
            ("M5", "AX", 303, "F3"),
            ("M6", "AX", 404, "F4"),
            ("M7", "AX", 505, "F5"),
            ("M8", "AX", 606, "F6"),
        ),
    ),
    Table(
        "bookings",
        (
            ("booking_id", "text"),
            ("pnr", "text"),
            ("status", "text"),
            ("service_priority", "int"),
            ("external_customer_ref", "text"),
        ),
        ("booking_id",),
        (
            ("B1", "QXA1B1", "confirmed", 3, "ext-cust-0001"),
            ("B2", "QXA2B2", "confirmed", 2, "ext-cust-0002"),
            ("B3", "QXA3B3", "cancelled", 1, "ext-cust-0003"),
            ("B4", "QXA4B4", "confirmed", 1, "ext-cust-0004"),
            ("B5", "QXA5B5", "confirmed", 1, "ext-cust-0005"),
            ("B6", "QXA6B6", "confirmed", 1, None),
        ),
    ),
    Table(
        "travellers",
        (("traveller_id", "text"), ("display_name", "text")),
        ("traveller_id",),
        (
            ("V1", "Sam Lee"),
            ("V2", "Sam Lee"),
            ("V3", "Ana Ruiz"),
            ("V4", "Kai Okafor"),
            ("V5", "Lena Voss"),
            ("V6", "Ravi Menon"),
        ),
    ),
    Table(
        "tickets",
        (
            ("ticket_id", "text"),
            ("booking_id", "text"),
            ("traveller_id", "text"),
            ("status", "text"),
            ("currency", "text"),
            ("face_value_minor", "int"),
        ),
        ("ticket_id",),
        (
            ("T1", "B1", "V1", "open", "USD", 50000),
            ("T2", "B2", "V2", "open", "USD", 50000),
            ("T3", "B3", "V3", "void", "USD", 20000),
            ("T4", "B4", "V4", "open", "USD", 50000),
            ("T5", "B5", "V5", "open", "EUR", 45000),
            ("T6", "B6", "V6", "open", "USD", 50000),
        ),
    ),
    Table(
        "coupons",
        (
            ("coupon_id", "text"),
            ("ticket_id", "text"),
            ("flight_id", "text"),
            ("leg_no", "int"),
            ("cabin", "text"),
            ("status", "text"),
            ("version", "int"),
        ),
        ("coupon_id",),
        (
            ("C1", "T1", "F1", 1, CABIN, "reserved", 1),
            ("C2", "T1", "F2", 2, CABIN, "reserved", 1),
            ("C3", "T2", "F1", 1, CABIN, "reserved", 1),
            ("C4", "T2", "F2", 2, CABIN, "reserved", 1),
            ("C5", "T3", "F1", 1, CABIN, "void", 1),
            ("C6", "T4", "F3", 1, CABIN, "reserved", 1),
            ("C7", "T4", "F4", 2, CABIN, "reserved", 1),
            ("C8", "T5", "F5", 1, CABIN, "reserved", 1),
            ("C9", "T6", "F6", 1, CABIN, "reserved", 1),
            ("C10", "T6", "F7", 2, CABIN, "reserved", 1),
        ),
    ),
    Table(
        "inventory",
        (("flight_id", "text"), ("cabin", "text"), ("capacity", "int")),
        ("flight_id", "cabin"),
        tuple((f"F{i}", CABIN, cap) for i, cap in enumerate((3, 3, 2, 2, 2, 2, 2, 2), start=1)),
    ),
    Table(
        "payment_events",
        (
            ("event_id", "text"),
            ("booking_id", "text"),
            ("currency", "text"),
            ("signed_amount_minor", "int"),
            ("type", "text"),
            ("status", "text"),
            ("provider_event_id", "text"),
        ),
        ("event_id",),
        (
            ("P1", "B1", "USD", 30000, "capture", "settled", "prov-evt-0001"),
            ("P2", "B1", "USD", 20000, "capture", "settled", "prov-evt-0002"),
            ("P3", "B1", "USD", -5000, "refund", "settled", "prov-evt-0003"),
            ("P4", "B1", "USD", 100000, "capture", "failed", "prov-evt-0004"),
            ("P5", "B2", "USD", 50000, "capture", "settled", "prov-evt-0005"),
            ("P6", "B3", "USD", 20000, "capture", "settled", "prov-evt-0006"),
            ("P7", "B3", "USD", -20000, "refund", "settled", "prov-evt-0007"),
            ("P8", "B4", "USD", 50000, "capture", "settled", "prov-evt-0008"),
            ("P9", "B5", "EUR", 45000, "capture", "settled", "prov-evt-0009"),
            ("P10", "B6", "USD", 50000, "capture", "settled", "prov-evt-0010"),
        ),
    ),
    Table(
        "seat_holds",
        (
            ("hold_id", "text"),
            ("flight_id", "text"),
            ("cabin", "text"),
            ("quantity", "int"),
            ("status", "text"),
            ("expires_at", TS),
        ),
        ("hold_id",),
        (
            ("H1", "F2", CABIN, 1, "active", ts("07:10")),
            ("H2", "F5", CABIN, 1, "active", ts("06:59")),
            ("H3", "F6", CABIN, 1, "active", ts("07:10")),
            ("H4", "F7", CABIN, 1, "active", ts("07:00")),
            ("H5", "F8", CABIN, 1, "released", ts("07:10")),
        ),
    ),
    Table(
        "connection_rules",
        (("rule_id", "text"), ("airport_code", "text"), ("minimum_minutes", "int")),
        ("rule_id",),
        (
            ("MCT-SEA-045", "SEA", 45),
            ("MCT-DEN-045", "DEN", 45),
            ("MCT-ORD-045", "ORD", 45),
            ("MCT-JFK-060", "JFK", 60),
            ("MCT-SFO-045", "SFO", 45),
        ),
    ),
    Table(
        "rebooking_requests",
        (
            ("request_key", "text"),
            ("booking_id", "text"),
            ("payload_digest", "text"),
            ("state", "text"),
            ("response_json", "jsonb"),
            ("created_at", TS),
        ),
        ("request_key",),
        tuple(
            (
                seed_request_key(label),
                booking,
                seed_request_digest(label),
                "rejected",
                _rejection(reason),
                ts(created),
            )
            for label, booking, reason, created in _SEED_REQUESTS
        ),
    ),
)

TABLES_BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}

# Parents before children: foreign keys are enforced while seeding.
INSERT_ORDER = (
    "airports", "flight_instances", "marketing_flights", "bookings", "travellers", "tickets",
    "inventory", "coupons", "payment_events", "seat_holds", "connection_rules",
    "rebooking_requests",
)  # fmt: skip

DDL = f"""
CREATE SCHEMA {SCHEMA};
SET search_path = {SCHEMA}, pg_catalog;

CREATE TABLE airports (
  airport_code text PRIMARY KEY CHECK (airport_code ~ '^[A-Z]{{3}}$'),
  time_zone text NOT NULL
);
CREATE TABLE flight_instances (
  flight_id text PRIMARY KEY,
  origin text NOT NULL REFERENCES airports (airport_code),
  destination text NOT NULL REFERENCES airports (airport_code),
  departure_utc timestamptz NOT NULL,
  arrival_utc timestamptz NOT NULL,
  status text NOT NULL CHECK (status IN ('scheduled', 'cancelled')),
  CHECK (origin <> destination),
  CHECK (arrival_utc > departure_utc)
);
CREATE TABLE marketing_flights (
  marketing_id text PRIMARY KEY,
  carrier text NOT NULL CHECK (carrier ~ '^[A-Z0-9]{{2}}$'),
  flight_number integer NOT NULL CHECK (flight_number > 0),
  flight_id text NOT NULL REFERENCES flight_instances (flight_id),
  UNIQUE (carrier, flight_number)
);
CREATE TABLE bookings (
  booking_id text PRIMARY KEY,
  pnr text NOT NULL UNIQUE,
  status text NOT NULL CHECK (status IN ('confirmed', 'cancelled')),
  service_priority integer NOT NULL CHECK (service_priority BETWEEN 1 AND 9),
  external_customer_ref text UNIQUE
);
CREATE TABLE travellers (
  traveller_id text PRIMARY KEY,
  display_name text NOT NULL
);
CREATE TABLE tickets (
  ticket_id text PRIMARY KEY,
  booking_id text NOT NULL UNIQUE REFERENCES bookings (booking_id),
  traveller_id text NOT NULL UNIQUE REFERENCES travellers (traveller_id),
  status text NOT NULL CHECK (status IN ('open', 'void')),
  currency text NOT NULL CHECK (currency IN ('USD', 'EUR')),
  face_value_minor bigint NOT NULL CHECK (face_value_minor >= 0)
);
CREATE TABLE inventory (
  flight_id text NOT NULL REFERENCES flight_instances (flight_id),
  cabin text NOT NULL CHECK (cabin IN ('economy', 'business')),
  capacity integer NOT NULL CHECK (capacity >= 0),
  PRIMARY KEY (flight_id, cabin)
);
CREATE TABLE coupons (
  coupon_id text PRIMARY KEY,
  ticket_id text NOT NULL REFERENCES tickets (ticket_id),
  flight_id text NOT NULL REFERENCES flight_instances (flight_id),
  leg_no integer NOT NULL CHECK (leg_no BETWEEN 1 AND 2),
  cabin text NOT NULL,
  status text NOT NULL CHECK (status IN ('reserved', 'void')),
  version integer NOT NULL DEFAULT 1 CHECK (version >= 1),
  UNIQUE (ticket_id, leg_no),
  FOREIGN KEY (flight_id, cabin) REFERENCES inventory (flight_id, cabin)
);
CREATE TABLE payment_events (
  event_id text PRIMARY KEY,
  booking_id text NOT NULL REFERENCES bookings (booking_id),
  currency text NOT NULL CHECK (currency IN ('USD', 'EUR')),
  signed_amount_minor bigint NOT NULL,
  type text NOT NULL CHECK (type IN ('capture', 'refund')),
  status text NOT NULL CHECK (status IN ('settled', 'failed')),
  provider_event_id text NOT NULL UNIQUE,
  CHECK ((type = 'capture' AND signed_amount_minor > 0)
      OR (type = 'refund' AND signed_amount_minor < 0))
);
CREATE TABLE seat_holds (
  hold_id text PRIMARY KEY,
  flight_id text NOT NULL,
  cabin text NOT NULL,
  quantity integer NOT NULL CHECK (quantity > 0),
  status text NOT NULL CHECK (status IN ('active', 'released')),
  expires_at timestamptz NOT NULL,
  FOREIGN KEY (flight_id, cabin) REFERENCES inventory (flight_id, cabin)
);
CREATE TABLE connection_rules (
  rule_id text PRIMARY KEY,
  airport_code text NOT NULL UNIQUE REFERENCES airports (airport_code),
  minimum_minutes integer NOT NULL CHECK (minimum_minutes > 0)
);
CREATE TABLE rebooking_requests (
  request_key text PRIMARY KEY,
  booking_id text NOT NULL REFERENCES bookings (booking_id),
  payload_digest text NOT NULL CHECK (payload_digest ~ '^[0-9a-f]{{64}}$'),
  state text NOT NULL CHECK (state IN ('accepted', 'rejected')),
  response_json jsonb NOT NULL,
  created_at timestamptz NOT NULL
);
"""

# Payload used by the capacity-changes-after-planning/competing hold scenarios.
FUTURE_HOLD_EXPIRY = ts("23:00")


def sql_literal(value: Any, kind: str) -> str:
    """Render one synthetic value as a SQL literal (fixture data only, never user input)."""
    if value is None:
        return "NULL"
    if kind == "int":
        return str(int(value))
    text = json.dumps(value, sort_keys=True, separators=(",", ":")) if kind == "jsonb" else value
    quoted = "'" + str(text).replace("'", "''") + "'"
    if kind in ("timestamptz", "jsonb"):
        return f"{quoted}::{kind}"
    return quoted


def insert_statements() -> str:
    parts: list[str] = []
    for table in (TABLES_BY_NAME[name] for name in INSERT_ORDER):
        cols = ", ".join(table.column_names)
        for row in table.rows:
            values = ", ".join(
                sql_literal(v, kind) for v, (_, kind) in zip(row, table.columns, strict=True)
            )
            parts.append(f"INSERT INTO {table.name} ({cols}) VALUES ({values});")
    return "\n".join(parts) + "\n"


def seed_sql() -> str:
    """Whole schema plus seed rows as one transaction (generated, not hand-pasted)."""
    return f"BEGIN;\n{DDL}\n{insert_statements()}COMMIT;\n"


def schema_digest() -> str:
    return hashlib.sha256(DDL.encode()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def state_select_sql() -> str:
    """One query returning every table as JSON rows in canonical column form."""
    pieces = []
    for table in TABLES:
        exprs = ", ".join(
            f"to_char({name} AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"') AS {name}"
            if kind == "timestamptz"
            else name
            for name, kind in table.columns
        )
        pieces.append(
            f"'{table.name}', (SELECT coalesce(jsonb_agg(to_jsonb(q)), '[]'::jsonb) "
            f"FROM (SELECT {exprs} FROM {SCHEMA}.{table.name}) q)"
        )
    return "SELECT jsonb_build_object(" + ", ".join(pieces) + ") AS state"


def seed_state() -> dict[str, list[dict[str, Any]]]:
    return {t.name: t.row_dicts() for t in TABLES}


def table_digest(table_name: str, rows: Iterable[Mapping[str, Any]]) -> str:
    """Logical digest of one table: rows sorted by primary key, canonical JSON, sha256."""
    table = TABLES_BY_NAME[table_name]
    ordered = sorted((dict(r) for r in rows), key=lambda r: tuple(str(r[c]) for c in table.pk))
    return hashlib.sha256(canonical_json(ordered)).hexdigest()


def logical_digests(state: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, str]:
    return {t.name: table_digest(t.name, state[t.name]) for t in TABLES}


def overall_digest(digests: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_json(dict(digests))).hexdigest()


def seed_digests() -> dict[str, str]:
    return logical_digests(seed_state())


def row_counts(state: Mapping[str, Iterable[Any]]) -> dict[str, int]:
    return {t.name: len(list(state[t.name])) for t in TABLES}


@dataclass(frozen=True)
class ConstraintProbe:
    """A write the real constraints must refuse (never run against the recorded baseline)."""

    name: str
    sql: str
    sqlstate: str


UNIQUE, FOREIGN_KEY, CHECK = "23505", "23503", "23514"
CONSTRAINT_PROBES: tuple[ConstraintProbe, ...] = (
    ConstraintProbe(
        "pk.duplicate_airport",
        f"INSERT INTO {SCHEMA}.airports VALUES ('SEA', 'America/Los_Angeles');",
        UNIQUE,
    ),
    ConstraintProbe(
        "pk.duplicate_inventory_pair",
        f"INSERT INTO {SCHEMA}.inventory VALUES ('F3', 'economy', 9);",
        UNIQUE,
    ),
    ConstraintProbe(
        "fk.unknown_flight_origin",
        f"INSERT INTO {SCHEMA}.flight_instances VALUES "
        "('F9', 'XXX', 'JFK', '2026-11-01T20:00:00Z', '2026-11-01T22:00:00Z', 'scheduled');",
        FOREIGN_KEY,
    ),
    ConstraintProbe(
        "fk.coupon_needs_inventory_row",
        f"INSERT INTO {SCHEMA}.coupons VALUES ('C99', 'T3', 'F3', 2, 'business', 'reserved', 1);",
        FOREIGN_KEY,
    ),
    ConstraintProbe(
        "check.booking_status",
        f"UPDATE {SCHEMA}.bookings SET status = 'pending' WHERE booking_id = 'B1';",
        CHECK,
    ),
    ConstraintProbe(
        "check.flight_arrives_after_departure",
        f"UPDATE {SCHEMA}.flight_instances SET arrival_utc = departure_utc WHERE flight_id = 'F5';",
        CHECK,
    ),
    ConstraintProbe(
        "check.capture_must_be_positive",
        f"INSERT INTO {SCHEMA}.payment_events VALUES "
        "('P99', 'B1', 'USD', -1, 'capture', 'settled', 'prov-evt-0099');",
        CHECK,
    ),
    ConstraintProbe(
        "unique.provider_event_id",
        f"INSERT INTO {SCHEMA}.payment_events VALUES "
        "('P98', 'B1', 'USD', 1, 'capture', 'settled', 'prov-evt-0001');",
        UNIQUE,
    ),
    ConstraintProbe(
        "unique.ticket_leg",
        f"INSERT INTO {SCHEMA}.coupons VALUES ('C98', 'T4', 'F5', 1, 'economy', 'reserved', 1);",
        UNIQUE,
    ),
    ConstraintProbe(
        "unique.pnr",
        f"UPDATE {SCHEMA}.bookings SET pnr = 'QXA1B1' WHERE booking_id = 'B2';",
        UNIQUE,
    ),
    ConstraintProbe(
        "unique.connection_rule_airport",
        f"INSERT INTO {SCHEMA}.connection_rules VALUES ('MCT-SEA-099', 'SEA', 99);",
        UNIQUE,
    ),
    ConstraintProbe(
        "fk.request_unknown_booking",
        f"INSERT INTO {SCHEMA}.rebooking_requests VALUES "
        f"('k', 'B99', '{'0' * 64}', 'accepted', '{{}}'::jsonb, now());",
        FOREIGN_KEY,
    ),
)
