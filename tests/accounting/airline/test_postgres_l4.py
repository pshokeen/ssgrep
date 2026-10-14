"""L4 reference operation on the real database; broken variants must be caught."""

from __future__ import annotations

import pytest

from eval.accounting_airline import dataset, domain_oracle as oracle, faults, l4_reference as l4
from eval.accounting_airline.database import ADMIN_DB, READER, WRITER

pytestmark = pytest.mark.postgres

S = dataset.SCHEMA


def clone_state(lab, scenario_db: str) -> dict:
    return lab.instance.read_state(scenario_db)


def test_successful_joint_application_final_state_oracle(lab) -> None:
    scenario = faults.run(lab.prep, "joint_apply")
    assert scenario.violations == []
    assert [a["returncode"] for a in scenario.attempts] == [0, 0]
    assert all(a["response"]["outcome"] == "accepted" for a in scenario.attempts)


def test_joint_application_leaves_the_exact_rows_the_spec_names(lab) -> None:
    database, before = lab.prep.clone("joint_direct")
    try:
        plan = oracle.joint_plan(before)
        for b, legs in sorted(plan.items()):
            res = faults.apply_request(
                lab.instance, database, l4.request_for(before, f"direct:{b}", b, legs), app=f"d-{b}"
            )
            assert res.response is not None and res.returncode == 0
            assert res.response["outcome"] == "accepted"
        rows = lab.instance.rows(
            database,
            f"SELECT coupon_id, flight_id, status, version FROM {S}.coupons ORDER BY 1",
            role=READER,
        )
        by = {r["coupon_id"]: r for r in rows}
        assert (by["C1"]["flight_id"], by["C1"]["status"]) == ("F5", "reserved")
        assert by["C2"]["status"] == "void"  # direct reissue voids the old connection coupon
        assert (by["C3"]["flight_id"], by["C4"]["flight_id"]) == ("F3", "F4")
        for untouched in ("C5", "C6", "C7", "C8", "C9", "C10"):
            assert by[untouched]["version"] == 1, untouched
        after = lab.instance.read_state(database)
        occupied = {r.flight_id: r.occupied for r in oracle.seat_rows(after)}
        assert [occupied[f] for f in ("F3", "F4", "F5")] == [2, 2, 2]
        assert occupied["F1"] == occupied["F2"] == 0
        assert oracle.sellable_map(after)["F2"] == 2
        sellable = lab.instance.rows(
            database,
            "WITH occ AS (SELECT flight_id, count(*) n FROM airline.coupons "
            "WHERE status='reserved' GROUP BY 1) "
            "SELECT i.flight_id, i.capacity - coalesce(o.n,0) AS raw_free "
            "FROM airline.inventory i LEFT JOIN occ o USING (flight_id) ORDER BY 1",
            role=READER,
        )
        assert all(r["raw_free"] >= 0 for r in sellable)  # no negative capacity anywhere
        for table in ("payment_events", "tickets", "seat_holds", "bookings", "travellers"):
            assert dataset.table_digest(table, after[table]) == dataset.table_digest(
                table, before[table]
            ), table
        assert [r["state"] for r in after["rebooking_requests"]].count("accepted") == 2
    finally:
        lab.instance.drop_clone("joint_direct")
        lab.prep.created.clear()


@pytest.mark.parametrize("repeat", range(3))
def test_last_seat_race_exactly_one_commit_one_conflict(lab, repeat) -> None:
    scenario = faults.run(lab.prep, "race")
    assert scenario.violations == []
    sync = scenario.evidence["synchronisation"]
    assert sync["clients_blocked_at_start_gate"] == 2  # observed at the gate, not slept for
    assert sorted(sync["clients_mid_transaction_waiting"]) == ["lab-race-0", "lab-race-1"]
    outcomes = sorted(r["outcome"] for r in scenario.evidence["responses"].values())
    assert outcomes == ["accepted", "rejected"]
    rejected = next(
        r for r in scenario.evidence["responses"].values() if r["outcome"] == "rejected"
    )
    assert rejected["reason"] == "capacity_conflict"
    assert scenario.evidence["final_occupancy"] == {"F3": 2, "F4": 2}
    assert [a["returncode"] for a in scenario.attempts] == [0, 0]


def test_race_loser_keeps_its_original_coupons(lab) -> None:
    database, before = lab.prep.clone("race_loser")
    try:
        requests = [l4.request_for(before, f"loser:{b}", b, ("F3", "F4")) for b in ("B1", "B2")]
        results, _ = faults.run_barriered_pair(lab.instance, database, requests)
        after = lab.instance.read_state(database)
        by_key = {
            r.request_key: res.response or {} for r, res in zip(requests, results, strict=True)
        }
        loser_request = next(
            r for r in requests if by_key[r.request_key].get("outcome") == "rejected"
        )
        winner_request = next(
            r for r in requests if by_key[r.request_key].get("outcome") == "accepted"
        )
        ticket = next(t for t in after["tickets"] if t["booking_id"] == loser_request.booking_id)
        mine = sorted(
            (c for c in after["coupons"] if c["ticket_id"] == ticket["ticket_id"]),
            key=lambda c: c["leg_no"],
        )
        original = sorted(
            (c for c in before["coupons"] if c["ticket_id"] == ticket["ticket_id"]),
            key=lambda c: c["leg_no"],
        )
        assert mine == original  # loser: every coupon identical, version included
        won = next(t for t in after["tickets"] if t["booking_id"] == winner_request.booking_id)
        moved = sorted(
            c["flight_id"] for c in after["coupons"] if c["ticket_id"] == won["ticket_id"]
        )
        assert moved == ["F3", "F4"]
        log = {r["request_key"]: r["state"] for r in after["rebooking_requests"]}
        assert log[loser_request.request_key] == "rejected"
        assert log[winner_request.request_key] == "accepted"
        assert len(after["rebooking_requests"]) == 7
    finally:
        lab.instance.drop_clone("race_loser")
        lab.prep.created.clear()


def test_replay_returns_original_outcome_without_any_extra_mutation(lab) -> None:
    scenario = faults.run(lab.prep, "replay")
    assert scenario.violations == []
    first, replayed = scenario.attempts
    assert first["response"]["replayed"] is False and replayed["response"]["replayed"] is True
    assert replayed["response"]["coupons"] == first["response"]["coupons"]
    assert replayed["duration_ns"] > 0  # the repeated invocation is a real, measurable call


def test_same_key_different_digest_is_rejected_without_state_change(lab) -> None:
    scenario = faults.run(lab.prep, "key_conflict")
    assert scenario.violations == []
    assert scenario.attempts[1]["response"] == {
        "outcome": "rejected", "reason": "idempotency_key_reuse", "replayed": False,
    }  # fmt: skip


def test_mid_transaction_failure_rolls_everything_back_then_retry_commits_once(lab) -> None:
    scenario = faults.run(lab.prep, "rollback")
    assert scenario.violations == []
    failed, retry, again = scenario.attempts
    assert failed["returncode"] != 0 and "injected failure" in failed["error"]
    assert retry["response"]["outcome"] == "accepted" and retry["response"]["replayed"] is False
    assert again["response"]["replayed"] is True  # the retry committed exactly once


def test_failed_attempt_leaves_no_request_row_and_no_coupon_change(lab) -> None:
    database, before = lab.prep.clone("rollback_direct")
    try:
        request = l4.request_for(before, "direct:rb", "B1", ("F3", "F4"))
        res = faults.apply_request(lab.instance, database, request, app="rb-fail", fail_coupon="C2")
        assert res.returncode != 0 and res.sqlstate == "P0001"
        assert lab.instance.read_state(database) == before  # byte-for-byte logical equality
        keys = lab.instance.rows(
            database, f"SELECT request_key FROM {S}.rebooking_requests", role=READER
        )
        assert request.request_key not in {r["request_key"] for r in keys}
    finally:
        lab.instance.drop_clone("rollback_direct")
        lab.prep.created.clear()


def test_stale_plan_is_refused_after_a_competing_hold_takes_the_last_seat(lab) -> None:
    scenario = faults.run(lab.prep, "stale_plan")
    assert scenario.violations == []
    assert scenario.attempts[0]["response"]["reason"] == "capacity_conflict"
    assert scenario.evidence["baseline_digest"] != scenario.evidence["changed_baseline_digest"]
    assert scenario.evidence["expected_delta"] == {"seat_holds": "+H-STALE-1 on F5"}


def test_writer_role_runs_the_operation_but_cannot_place_holds_or_book_unknowns(
    lab, scratch
) -> None:
    request = l4.request_for(lab.base.state, "role:check", "B2", ("F5",))
    res = lab.instance.psql(scratch, request.call(), role=WRITER)
    assert '"outcome": "accepted"' in res.stdout
    direct_hold = lab.instance.psql(
        scratch,
        l4.place_hold_call("HX", "F5", 1, dataset.FUTURE_HOLD_EXPIRY),
        role=WRITER,
        check=False,
    )
    assert not direct_hold.ok and direct_hold.sqlstate == "42501"  # only the verifier places holds
    ghost = l4.RebookRequest("role:unknown", "B9", ("F5",), ())
    unknown = lab.instance.psql(scratch, ghost.call(), role=WRITER, check=False)
    assert not unknown.ok and unknown.sqlstate == "P0002"  # unknown booking aborts, no row


# ------------------------------------------------------------- broken L4 is caught


@pytest.mark.parametrize(
    ("variant", "scenario"),
    [
        ("no_lock", "race"),
        ("no_revalidation", "race"),
        ("no_capacity_check", "race"),
        ("no_capacity_check", "stale_plan"),
        ("half_applied", "joint_apply"),
        ("half_applied", "replay"),
        ("swallow_failure", "rollback"),
    ],
)
def test_broken_l4_is_rejected_by_the_verifiers(lab, variant, scenario) -> None:
    result = faults.run(lab.prep, scenario, variant)
    assert result.violations, f"{variant} passed {scenario}: the verifier is blind to it"


def test_broken_variants_really_differ_from_the_reference_in_the_database(lab) -> None:
    sources = {}
    for variant in ("correct", "no_lock", "no_revalidation", "half_applied"):
        database = lab.instance.clone(f"src_{variant}")
        try:
            lab.instance.psql(database, l4.install_sql(variant))
            sources[variant] = lab.instance.psql(
                database, f"SELECT pg_get_functiondef('{l4.FUNCTION}'::regproc);"
            ).stdout
        finally:
            lab.instance.drop_clone(f"src_{variant}")
    assert (
        "FOR NO KEY UPDATE" in sources["correct"] and "FOR NO KEY UPDATE" not in sources["no_lock"]
    )
    revalidating = sources["correct"]
    broken = sources["no_revalidation"]
    # reference: locks are taken BEFORE the capacity check; the broken variant checks first
    assert revalidating.index("FOR NO KEY UPDATE") < revalidating.index("v_full := NULL")
    assert broken.index("v_full := NULL") < broken.index("FOR NO KEY UPDATE")
    assert "FOR i IN 1..1 LOOP" in sources["half_applied"]
    assert "FOR i IN 1..array_length(p_itinerary, 1) LOOP" in sources["correct"]
    assert lab.instance.psql(ADMIN_DB, "SELECT 1;").ok


def test_race_with_the_correct_operation_passes_where_no_lock_fails_on_the_same_inputs(lab) -> None:
    good = faults.run(lab.prep, "race", "correct")
    bad = faults.run(lab.prep, "race", "no_lock")
    assert good.violations == []
    assert any("expected 1 accepted + 1 rejected, got 2/0" in v for v in bad.violations)
    assert bad.evidence["final_occupancy"].startswith("integrity failure")  # oversell is surfaced
