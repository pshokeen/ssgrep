"""Offline checks of the canonical fixture definition (no database needed)."""

from __future__ import annotations

import copy

import pytest

from eval.accounting_airline import dataset

STATE = dataset.seed_state()


def test_counts_match_spec_and_total_82() -> None:
    counts = dataset.row_counts(STATE)
    assert list(counts.values()) == [5, 8, 8, 6, 6, 6, 10, 8, 10, 5, 5, 5]
    assert [t.name for t in dataset.TABLES] == list(dataset.EXPECTED_COUNTS)
    assert counts == dataset.EXPECTED_COUNTS
    assert sum(counts.values()) == dataset.TOTAL_ROWS == 82
    assert all(5 <= n <= 10 for n in counts.values())


def test_primary_keys_unique_and_references_resolve_in_fixture_rows() -> None:
    for table in dataset.TABLES:
        keys = [tuple(r[c] for c in table.pk) for r in STATE[table.name]]
        assert len(keys) == len(set(keys)), table.name
    ids = lambda t, c: {r[c] for r in STATE[t]}  # noqa: E731
    assert {r["origin"] for r in STATE["flight_instances"]} <= ids("airports", "airport_code")
    assert {r["flight_id"] for r in STATE["marketing_flights"]} <= ids(
        "flight_instances", "flight_id"
    )
    assert {r["ticket_id"] for r in STATE["coupons"]} <= ids("tickets", "ticket_id")
    inventory = {(r["flight_id"], r["cabin"]) for r in STATE["inventory"]}
    assert {(r["flight_id"], r["cabin"]) for r in STATE["coupons"]} <= inventory
    assert {(r["flight_id"], r["cabin"]) for r in STATE["seat_holds"]} <= inventory
    assert {r["booking_id"] for r in STATE["payment_events"]} <= ids("bookings", "booking_id")
    provider_ids = [r["provider_event_id"] for r in STATE["payment_events"]]
    assert len(provider_ids) == len(set(provider_ids)) == 10


def test_traps_are_present_in_the_data_not_just_named() -> None:
    names = [r["display_name"] for r in STATE["travellers"]]
    assert names.count("Sam Lee") == 2
    by_booking = {r["booking_id"]: r for r in STATE["bookings"]}
    assert by_booking["B6"]["external_customer_ref"] is None
    assert sum(r["external_customer_ref"] is None for r in STATE["bookings"]) == 1
    assert by_booking["B3"]["status"] == "cancelled"
    assert {r["coupon_id"] for r in STATE["coupons"] if r["status"] == "void"} == {"C5"}
    marketed = {r["flight_id"] for r in STATE["marketing_flights"]}
    assert marketed == {"F1", "F2", "F3", "F4", "F5", "F6"}  # F7/F8 have no marketing row
    assert sorted(
        r["marketing_id"] for r in STATE["marketing_flights"] if r["flight_id"] == "F1"
    ) == [
        "M1",
        "M2",
    ]
    failed = [r for r in STATE["payment_events"] if r["status"] == "failed"]
    assert [(r["event_id"], r["signed_amount_minor"]) for r in failed] == [("P4", 100000)]
    assert {r["currency"] for r in STATE["payment_events"]} == {"USD", "EUR"}
    holds = {r["hold_id"]: (r["status"], r["expires_at"]) for r in STATE["seat_holds"]}
    assert holds["H4"] == ("active", "2026-11-01T07:00:00Z")  # equality boundary
    assert holds["H5"][0] == "released"
    mct = {
        r["airport_code"]: (r["rule_id"], r["minimum_minutes"]) for r in STATE["connection_rules"]
    }
    assert mct["JFK"] == ("MCT-JFK-060", 60)
    assert all(rule == f"MCT-{code}-{minutes:03d}" for code, (rule, minutes) in mct.items())


def test_seed_requests_use_their_own_namespace_and_predate_as_of() -> None:
    rows = STATE["rebooking_requests"]
    assert all(r["request_key"].startswith("seed:") for r in rows)
    assert all(r["state"] == "rejected" and r["created_at"] < dataset.AS_OF for r in rows)
    assert {(r["booking_id"], r["response_json"]["reason"]) for r in rows} == {
        ("B3", "booking_cancelled"),
        ("B4", "no_matching_itinerary"),
        ("B5", "currency_mismatch"),
        ("B6", "hold_conflict"),
        ("B1", "request_expired"),
    }
    assert len({r["payload_digest"] for r in rows}) == 5


def test_ddl_declares_real_constraints_for_all_twelve_tables() -> None:
    ddl = dataset.DDL
    assert ddl.count("CREATE TABLE") == 12
    assert ddl.count("PRIMARY KEY") >= 12
    assert ddl.count("REFERENCES") >= 13
    assert "UNIQUE (ticket_id, leg_no)" in ddl
    assert "provider_event_id text NOT NULL UNIQUE" in ddl
    assert "timestamptz" in ddl and "timestamp without" not in ddl
    assert ddl.count("CHECK") >= 18


def test_seed_sql_is_generated_deterministically_from_the_fixture() -> None:
    first, second = dataset.seed_sql(), dataset.seed_sql()
    assert first == second
    assert first.startswith("BEGIN;") and first.rstrip().endswith("COMMIT;")
    assert first.count("INSERT INTO") == 82
    # parents precede children so foreign keys are enforced while seeding
    order = [first.index(f"INSERT INTO {t} ") for t in dataset.INSERT_ORDER]
    assert order == sorted(order)
    assert (
        "'Sam Lee'" in first and "NULL" in first and "'2026-11-01T07:00:00Z'::timestamptz" in first
    )
    assert "''" not in first.replace("''::", "")  # no accidental empty-string values


def test_digests_are_stable_order_independent_and_value_sensitive() -> None:
    base = dataset.seed_digests()
    assert base == dataset.logical_digests(STATE)
    shuffled = copy.deepcopy(STATE)
    for rows in shuffled.values():
        rows.reverse()
    assert dataset.logical_digests(shuffled) == base
    assert dataset.overall_digest(base) == dataset.overall_digest(dict(reversed(base.items())))

    mutated = copy.deepcopy(STATE)
    next(r for r in mutated["coupons"] if r["coupon_id"] == "C1")["flight_id"] = "F5"
    changed = dataset.logical_digests(mutated)
    assert [t for t in base if base[t] != changed[t]] == ["coupons"]
    assert dataset.overall_digest(changed) != dataset.overall_digest(base)


def test_sql_literal_escapes_and_types() -> None:
    assert dataset.sql_literal(None, "text") == "NULL"
    assert dataset.sql_literal("O'Hare", "text") == "'O''Hare'"
    assert dataset.sql_literal(7, "int") == "7"
    assert dataset.sql_literal("2026-11-01T07:00:00Z", "timestamptz") == (
        "'2026-11-01T07:00:00Z'::timestamptz"
    )
    assert dataset.sql_literal({"b": 1, "a": 2}, "jsonb") == '\'{"a":2,"b":1}\'::jsonb'


@pytest.mark.parametrize("probe", dataset.CONSTRAINT_PROBES, ids=lambda p: p.name)
def test_every_constraint_probe_targets_a_declared_constraint(
    probe: dataset.ConstraintProbe,
) -> None:
    assert probe.sqlstate in (dataset.UNIQUE, dataset.FOREIGN_KEY, dataset.CHECK)
    assert probe.sql.endswith(";") and dataset.SCHEMA in probe.sql
