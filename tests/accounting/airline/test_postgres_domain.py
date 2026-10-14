"""Domain oracle vs the REAL PostgreSQL database (seed, L1, L2, L3, sensitivity, roles)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from eval.accounting_airline import dataset, domain_oracle as oracle, mutations, queries as q
from eval.accounting_airline.database import ADMIN_DB, READER, ROLE_LIMITS, WRITER

pytestmark = pytest.mark.postgres


def rows(lab, database: str, sql: str) -> list[dict]:
    return lab.instance.rows(database, sql, role=READER)


# ------------------------------------------------------------------------------ seed


def test_ephemeral_instance_is_labelled_loopback_and_pinned(lab) -> None:
    info = lab.instance.inspect()
    labels = info["Config"]["Labels"]
    assert labels["ssgrep-lab-run"] == lab.instance.run_id and labels["ssgrep-lab"] == "airline"
    assert info["Name"] == f"/ssgrep-airline-{lab.instance.run_id}"
    assert info["Config"]["Image"] == dataset.IMAGE_REF
    assert info["Image"] == dataset.IMAGE_DIGEST
    bound = info["NetworkSettings"]["Ports"]["5432/tcp"]
    assert {b["HostIp"] for b in bound} == {"127.0.0.1"} and int(bound[0]["HostPort"]) > 1023
    assert lab.instance.manifest()["bind"]["host_ips"] == ["127.0.0.1"]
    version = lab.instance.value(ADMIN_DB, "SELECT current_setting('server_version_num')::int AS v")
    assert 170000 <= version["v"] < 180000


def test_seed_counts_total_and_digest_match_the_fixture_definition(lab, scratch) -> None:
    assert lab.base.counts == dataset.EXPECTED_COUNTS
    assert sum(lab.base.counts.values()) == 82
    # the sealed baseline refuses connections; read it through a clone of itself
    actual = lab.instance.read_state(scratch)
    assert dataset.logical_digests(actual) == dataset.seed_digests() == lab.base.digests
    assert lab.base.digest == dataset.overall_digest(dataset.seed_digests())
    for table, expected in dataset.EXPECTED_COUNTS.items():
        n = lab.instance.rows(scratch, f"SELECT count(*) AS n FROM {dataset.SCHEMA}.{table}")[0][
            "n"
        ]
        assert n == expected, table
    refused = lab.instance.psql(lab.base.database, "SELECT 1;", check=False)
    assert not refused.ok and "not currently accepting connections" in refused.stderr


def test_constraints_are_real_and_refuse_each_probe(lab, scratch) -> None:
    catalog = lab.instance.rows(
        scratch,
        "SELECT contype::text AS kind, count(*) AS n FROM pg_constraint c "
        f"JOIN pg_namespace s ON s.oid = c.connamespace WHERE s.nspname = '{dataset.SCHEMA}' "
        "GROUP BY contype ORDER BY contype",
    )
    kinds = {r["kind"]: r["n"] for r in catalog}
    assert kinds["p"] == 12 and kinds["f"] == 13 and kinds["u"] == 8 and kinds["c"] >= 20
    for probe in dataset.CONSTRAINT_PROBES:
        res = lab.instance.psql(scratch, f"BEGIN;\n{probe.sql}\nROLLBACK;\n", check=False)
        assert not res.ok and res.sqlstate == probe.sqlstate, (probe.name, res.stderr[:200])
    assert dataset.logical_digests(lab.instance.read_state(scratch)) == lab.base.digests


def test_frozen_clock_is_recorded_in_the_manifest(lab) -> None:
    manifest = lab.instance.manifest()
    assert manifest["as_of"] == dataset.AS_OF == "2026-11-01T07:00:00Z"
    assert manifest["schema_digest"] == dataset.schema_digest()
    assert "password" not in str(manifest).lower()
    for password in lab.instance._passwords.values():
        assert password not in str(manifest)


def test_database_roles_are_least_privilege_with_recorded_timeouts(lab, scratch) -> None:
    s = dataset.SCHEMA
    probes = [
        (READER, f"UPDATE {s}.coupons SET status = 'void';", "25006"),
        (READER, f"INSERT INTO {s}.airports VALUES ('XXX', 'UTC');", "25006"),
        (WRITER, f"UPDATE {s}.payment_events SET status = 'failed';", "42501"),
        (WRITER, f"UPDATE {s}.tickets SET face_value_minor = 1;", "42501"),
        (WRITER, f"DELETE FROM {s}.coupons;", "42501"),
        (WRITER, f"UPDATE {s}.inventory SET capacity = 99;", "42501"),
        (WRITER, f"CREATE TABLE {s}.x (a int);", "42501"),
    ]
    for role, sql, state in probes:
        res = lab.instance.psql(scratch, sql, role=role, check=False)
        assert not res.ok and res.sqlstate == state, (role, sql, res.stderr[:200])
    for role in (READER, WRITER):
        res = lab.instance.psql(ADMIN_DB, "SELECT 1;", role=role, check=False)
        assert not res.ok and "permission denied for database" in res.stderr
        for key, value in ROLE_LIMITS[role].items():
            assert lab.instance.psql(scratch, f"SHOW {key};", role=role).stdout.strip() == value
    assert rows(lab, scratch, f"SELECT count(*) AS n FROM {s}.coupons")[0]["n"] == 10
    assert dataset.logical_digests(lab.instance.read_state(scratch)) == lab.base.digests


def test_lock_timeout_bounds_a_blocked_reader_writer(lab, scratch) -> None:
    """The writer's lock timeout is real: a verifier-held row lock makes it fail fast."""
    holder = lab.instance.popen(scratch, app="lab-lock-holder")
    assert holder.stdin is not None
    holder.stdin.write(
        f"BEGIN;\nSELECT 1 FROM {dataset.SCHEMA}.coupons WHERE coupon_id='C8' FOR UPDATE;\n"
    )
    holder.stdin.flush()
    from eval.accounting_airline.faults import wait_until

    try:
        wait_until(
            lambda: bool(
                lab.instance.rows(
                    ADMIN_DB,
                    "SELECT 1 AS n FROM pg_stat_activity WHERE application_name='lab-lock-holder' "
                    "AND state='idle in transaction'",
                )
            ),
            "the holder to hold its row lock",
        )
        res = lab.instance.psql(
            scratch,
            f"SET lock_timeout='300ms'; UPDATE {dataset.SCHEMA}.coupons SET version=version+1 "
            "WHERE coupon_id='C8';",
            role=WRITER, check=False,
        )  # fmt: skip
        assert not res.ok and res.sqlstate == "55P03"
    finally:
        holder.kill()
        holder.communicate()


# ---------------------------------------------------------------------- L1 / L2 / L3


def test_l1_cash_matches_oracle_and_canonical_numbers(lab, scratch) -> None:
    state = lab.base.state
    for sql in (q.CASH_PREAGGREGATED, q.CASH_CTE_ALTERNATIVE):  # other formulations accepted
        actual = rows(lab, scratch, sql)
        assert oracle.verify_cash(actual, state) == []
        got = {(r["booking_id"], r["currency"]): r["settled_minor"] for r in actual}
        assert got == {
            ("B1", "USD"): 45000, ("B2", "USD"): 50000, ("B3", "USD"): 0,
            ("B4", "USD"): 50000, ("B5", "EUR"): 45000, ("B6", "USD"): 50000,
        }  # fmt: skip
        totals: dict[str, int] = {}
        for (_, cur), v in got.items():
            totals[cur] = totals.get(cur, 0) + v
        assert totals == {"USD": 195000, "EUR": 45000}


def test_l1_seats_match_oracle_for_all_eight_physical_flights(lab, scratch) -> None:
    actual = rows(lab, scratch, q.seats())
    assert [r["flight_id"] for r in actual] == [f"F{i}" for i in range(1, 9)]
    assert [r["sellable_free"] for r in actual] == [0, 0, 1, 1, 1, 0, 1, 2]
    assert oracle.verify_seats(actual, lab.base.state) == []
    by = {r["flight_id"]: r for r in actual}
    assert (by["F1"]["raw_free"], by["F1"]["sellable_free"]) == (1, 0)  # cancelled stays diagnostic


def test_clock_advance_changes_actual_database_seats_and_the_oracle_agrees(lab, scratch) -> None:
    later = rows(lab, scratch, mutations.clock_advanced_seats(mutations.ADVANCED_AS_OF))
    when = datetime(2026, 11, 1, 7, 20, tzinfo=UTC)
    assert oracle.verify_seats(later, lab.base.state, when) == []
    assert {r["flight_id"]: r["sellable_free"] for r in later}["F2"] == 1
    # the same rows are WRONG at the canonical clock
    assert oracle.verify_seats(later, lab.base.state) != []


def test_l2_affected_connections_and_dst_instants_match_oracle(lab, scratch) -> None:
    state = lab.base.state
    affected = rows(lab, scratch, q.affected())
    assert [(r["booking_id"], r["traveller_id"]) for r in affected] == [("B1", "V1"), ("B2", "V2")]
    assert oracle.verify_affected(affected, state) == []
    conns = rows(lab, scratch, q.connections())
    assert oracle.verify_connections(conns, state) == []
    by = {r["booking_id"]: r for r in conns}
    assert (by["B1"]["gap_minutes"], by["B1"]["valid"], by["B1"]["rule_id"]) == (
        35,
        False,
        "MCT-DEN-045",
    )
    assert (by["B4"]["gap_minutes"], by["B4"]["valid"]) == (45, True)
    assert (by["B6"]["gap_minutes"], by["B6"]["valid"]) == (45, True)
    clocks = {r["flight_id"]: r for r in rows(lab, scratch, q.local_clock_listing())}
    assert (clocks["F1"]["local_clock"], clocks["F1"]["utc_offset_hours"]) == ("01:30", -7)
    assert (clocks["F8"]["local_clock"], clocks["F8"]["utc_offset_hours"]) == ("01:30", -8)
    assert oracle.verify_local_clock(list(clocks.values()), state) == []
    pairs = {(r["from_flight"], r["to_flight"]) for r in rows(lab, scratch, q.connectable_pairs())}
    assert ("F8", "F2") not in pairs and ("F3", "F4") in pairs and ("F6", "F7") in pairs


def test_l3_routes_and_joint_plan_from_real_rows(lab, scratch) -> None:
    state = lab.base.state
    routes = rows(lab, scratch, q.candidate_routes())
    assert [tuple(r["legs"]) for r in routes] == [("F5",), ("F3", "F4")]
    assert oracle.verify_routes(routes, state) == []
    affected = rows(lab, scratch, q.affected())
    seats = rows(lab, scratch, q.seats())
    plan = q.greedy_priority_plan(affected, routes, seats)
    assert plan == {"B1": ("F5",), "B2": ("F3", "F4")}
    assert oracle.plan_violations(state, plan) == []
    policy = rows(lab, scratch, q.policy_ids_used(("F3", "F4")))
    assert (
        [r["rule_id"] for r in policy] == oracle.policy_ids(state, ("F3", "F4")) == ["MCT-ORD-045"]
    )


# ------------------------------------------------------------------- SENSITIVITY


@pytest.mark.parametrize("formulation", mutations.MUTATIONS, ids=lambda f: f.name)
def test_wrong_formulation_is_rejected_on_its_actual_values(lab, scratch, formulation) -> None:
    actual, violations = mutations.evaluate(lab.instance, scratch, formulation, lab.base.state)
    assert actual, "the wrong query still returned plausible rows"
    assert violations, f"{formulation.name}: oracle accepted a wrong formulation"
    assert mutations.missing_flags(formulation, violations) == []


@pytest.mark.parametrize("formulation", mutations.CORRECT, ids=lambda f: f.name)
def test_equivalent_correct_formulation_is_accepted(lab, scratch, formulation) -> None:
    _, violations = mutations.evaluate(lab.instance, scratch, formulation, lab.base.state)
    assert violations == []


def test_fanout_cash_is_wrong_for_b1_even_though_b3_and_b5_coincidentally_pass(
    lab, scratch
) -> None:
    actual = rows(lab, scratch, q.CASH_JOIN_FANOUT)
    got = {(r["booking_id"], r["currency"]): r["settled_minor"] for r in actual}
    assert got[("B1", "USD")] == 4 * 45000  # 2 coupons x 2 marketing codeshares
    assert ("B3", "USD") not in got or got[("B3", "USD")] == 0
    assert got[("B5", "EUR")] == 45000  # a single-coupon, single-marketing booking looks right
    assert "cash ('B1', 'USD'): expected 45000 actual 180000" in oracle.verify_cash(
        actual, lab.base.state
    )


def test_independent_planning_oversells_f5_and_the_joint_oracle_rejects_it(lab, scratch) -> None:
    actual = rows(lab, scratch, q.independent_plan())
    plan = {r["booking_id"]: tuple(r["legs"]) for r in actual}
    assert plan == {"B1": ("F5",), "B2": ("F5",)}
    problems = oracle.plan_violations(lab.base.state, plan)
    assert any("F5: 2 travellers assigned for 1 sellable" in p for p in problems)


def test_sensitivity_inclusive_hold_expiry_flips_f7_only(lab, scratch) -> None:
    good = {r["flight_id"]: r for r in rows(lab, scratch, q.seats())}
    bad = {r["flight_id"]: r for r in rows(lab, scratch, q.seats_hold_expiry_inclusive())}
    diff = [f for f in good if good[f] != bad[f]]
    assert diff == ["F7"]  # H2 (06:59) is dead either way; only the 07:00 equality differs
    assert bad["F7"]["live_holds"] == 1 and good["F7"]["live_holds"] == 0
