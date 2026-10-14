"""Pure-Python reference model checks (offline). The model is independent of any SQL."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest

from eval.accounting_airline import dataset, domain_oracle as oracle

STATE = dataset.seed_state()


def mutate(fn) -> dict:
    state = copy.deepcopy(STATE)
    fn(state)
    return state


def test_l1_cash_by_booking_and_currency() -> None:
    cash = oracle.settled_cash(STATE)
    assert cash == {
        ("B1", "USD"): 45000, ("B2", "USD"): 50000, ("B3", "USD"): 0,
        ("B4", "USD"): 50000, ("B5", "EUR"): 45000, ("B6", "USD"): 50000,
    }  # fmt: skip
    assert cash == oracle.CANON_CASH
    assert oracle.currency_totals(cash) == {"USD": 195000, "EUR": 45000}
    face = {t["booking_id"]: t["face_value_minor"] for t in STATE["tickets"]}
    assert face["B1"] == 50000 != cash[("B1", "USD")]  # face value is not settled cash


def test_l1_seat_rows_include_live_hold_equality_and_cancelled_flight() -> None:
    rows = {r.flight_id: r for r in oracle.seat_rows(STATE)}
    assert [rows[f"F{i}"].sellable_free for i in range(1, 9)] == [0, 0, 1, 1, 1, 0, 1, 2]
    assert {f: r.key() for f, r in rows.items()} == oracle.CANON_SEATS
    assert rows["F1"].raw_free == 1 and rows["F1"].sellable_free == 0  # cancelled but occupied
    assert rows["F7"].live_holds == 0  # H4 expires exactly at as_of: not live
    assert rows["F5"].live_holds == 0  # H2 expired at 06:59
    assert rows["F8"].live_holds == 0  # H5 released
    assert rows["F2"].live_holds == rows["F6"].live_holds == 1


def test_clock_advance_recomputes_without_touching_runtime_clock() -> None:
    later = datetime(2026, 11, 1, 7, 20, tzinfo=UTC)
    rows = {r.flight_id: r for r in oracle.seat_rows(STATE, later)}
    assert rows["F2"].sellable_free == 1 and rows["F6"].sellable_free == 1  # H1/H3 expired
    earlier = datetime(2026, 11, 1, 6, 58, tzinfo=UTC)
    assert {r.flight_id: r for r in oracle.seat_rows(STATE, earlier)}["F5"].live_holds == 1


def test_oversold_state_is_an_integrity_failure_not_hidden() -> None:
    state = mutate(
        lambda s: next(c for c in s["coupons"] if c["coupon_id"] == "C9").update(flight_id="F3")
    )
    # F3 now holds C6 and C9 (capacity 2): fine. A third occupant oversells it.
    state["coupons"].append(
        {"coupon_id": "C11", "ticket_id": "T5", "flight_id": "F3", "leg_no": 2, "cabin": "economy",
         "status": "reserved", "version": 1}
    )  # fmt: skip
    with pytest.raises(oracle.IntegrityError, match="F3"):
        oracle.seat_rows(state)


def test_l2_affected_connections_and_boundaries() -> None:
    assert oracle.affected_travellers(STATE) == [("B1", "V1"), ("B2", "V2")]
    conns = {c.booking_id: c for c in oracle.connections(STATE)}
    assert (conns["B1"].gap_minutes, conns["B1"].rule_id, conns["B1"].valid) == (
        35,
        "MCT-DEN-045",
        False,
    )
    assert conns["B2"].gap_minutes == 35 and not conns["B2"].valid
    assert (conns["B4"].gap_minutes, conns["B4"].valid) == (45, True)
    assert (conns["B6"].gap_minutes, conns["B6"].valid) == (45, True)
    assert "B3" not in conns  # void coupon never forms a connection
    assert oracle.connection_valid(45, 45) and not oracle.connection_valid(44, 45)


def test_l2_names_do_not_collapse_travellers() -> None:
    affected = oracle.affected_travellers(STATE)
    assert len({t for _, t in affected}) == 2
    names = {t["traveller_id"]: t["display_name"] for t in STATE["travellers"]}
    assert names["V1"] == names["V2"]


def test_repeated_local_clock_is_two_instants_one_hour_apart() -> None:
    assert oracle.local_departure(STATE, "F1") == ("01:30", "-0700")
    assert oracle.local_departure(STATE, "F8") == ("01:30", "-0800")
    assert oracle.instant_gap_minutes(STATE, "F1", "F8") == 60
    assert oracle.can_connect(STATE, "F8", "F2") is False  # F8 lands after F2 departs
    assert oracle.can_connect(STATE, "F3", "F4") and oracle.can_connect(STATE, "F6", "F7")
    assert ("F8", "F2") not in oracle.connectable_pairs(STATE)


def test_l3_candidates_are_exactly_f5_and_f3_f4_with_reasoned_rejections() -> None:
    routes, rejected = oracle.route_candidates(STATE)
    assert [r.legs for r in routes] == [("F5",), ("F3", "F4")]
    assert [r.arrival.strftime("%H:%M") for r in routes] == ["15:30", "16:00"]
    assert "no sellable seat" in rejected[("F6", "F7")]
    assert "not scheduled" in rejected[("F1", "F2")]
    assert "invalid connection" in rejected[("F8", "F2")]
    assert not any("F1" in legs for legs in {r.legs for r in routes})


def test_l3_joint_plan_respects_priority_and_shared_capacity() -> None:
    assert oracle.joint_plan(STATE) == {"B1": ("F5",), "B2": ("F3", "F4")}
    assert oracle.plan_violations(STATE, oracle.CANON_PLAN) == []
    oversell = {"B1": ("F5",), "B2": ("F5",)}
    problems = oracle.plan_violations(STATE, oversell)
    assert any("F5: 2 travellers assigned for 1 sellable" in p for p in problems)
    # individually valid routes, but priorities inverted: higher priority B1 gets the later arrival
    inverted = {"B1": ("F3", "F4"), "B2": ("F5",)}
    assert any(
        "not the priority-respecting optimum" in p for p in oracle.plan_violations(STATE, inverted)
    )
    assert any("not a feasible" in p for p in oracle.plan_violations(STATE, {"B1": ("F6", "F7")}))


def test_oracle_follows_the_data_not_the_canonical_literals() -> None:
    """Change fixture rows and the model must change: it is a computation, not a table."""
    freed = mutate(
        lambda s: next(h for h in s["seat_holds"] if h["hold_id"] == "H3").update(status="released")
    )
    assert oracle.sellable_map(freed)["F6"] == 1
    assert ("F6", "F7") in {r.legs for r in oracle.route_candidates(freed)[0]}
    # with F6->F7 now bookable, B2 can arrive earlier than F3->F4 only if it beats 16:00Z
    assert oracle.joint_plan(freed) == {"B1": ("F5",), "B2": ("F3", "F4")}

    tighter = mutate(
        lambda s: next(r for r in s["connection_rules"] if r["airport_code"] == "ORD").update(
            minimum_minutes=46
        )
    )
    assert ("F3", "F4") not in {r.legs for r in oracle.route_candidates(tighter)[0]}
    assert any(c.booking_id == "B4" and not c.valid for c in oracle.connections(tighter))

    refunded = mutate(
        lambda s: next(p for p in s["payment_events"] if p["event_id"] == "P4").update(
            status="settled"
        )
    )
    assert oracle.settled_cash(refunded)[("B1", "USD")] == 145000


def test_verifiers_reject_wrong_rows_on_values() -> None:
    good_cash = [
        {"booking_id": b, "currency": c, "settled_minor": v}
        for (b, c), v in oracle.CANON_CASH.items()
    ]
    assert oracle.verify_cash(good_cash, STATE) == []
    fanout = [
        dict(r, settled_minor=r["settled_minor"] * 4) if r["booking_id"] == "B1" else r
        for r in good_cash
    ]
    assert oracle.verify_cash(fanout, STATE) == ["cash ('B1', 'USD'): expected 45000 actual 180000"]
    assert oracle.verify_cash(good_cash[1:], STATE) == ["missing cash row ('B1', 'USD')"]
    assert oracle.verify_cash(good_cash + good_cash[:1], STATE) == [
        "duplicate cash row ('B1', 'USD')"
    ]
    merged = [dict(r, currency="USD") for r in good_cash]
    problems = oracle.verify_cash(merged, STATE)
    assert (
        "missing cash row ('B5', 'EUR')" in problems
        and "unexpected cash row ('B5', 'USD')" in problems
    )

    good_seats = [
        {"flight_id": f, "occupied": o, "live_holds": h, "raw_free": r, "sellable_free": s}
        for f, (o, h, r, s) in oracle.CANON_SEATS.items()
    ]
    assert oracle.verify_seats(good_seats, STATE) == []
    inclusive_expiry = [dict(r) for r in good_seats]
    inclusive_expiry[6].update(live_holds=1, raw_free=0, sellable_free=0)  # H4 consumed F7
    assert oracle.verify_seats(inclusive_expiry, STATE) == [
        "seats F7: expected (1, 0, 1, 1) actual (1, 1, 0, 0)"
    ]
    assert oracle.verify_seats(good_seats[:7], STATE) == ["missing seat row F8"]

    assert oracle.verify_affected([{"booking_id": "B1", "traveller_id": "V1"}], STATE) != []
    assert oracle.verify_routes([{"legs": ["F5"]}], STATE) == [
        "missing candidate route ('F3', 'F4')"
    ]


def test_apply_moves_models_direct_reissue_and_leaves_payments_alone() -> None:
    after = oracle.state_after_joint_plan(STATE)
    coupons = {c["coupon_id"]: c for c in after["coupons"]}
    assert (coupons["C1"]["flight_id"], coupons["C1"]["status"], coupons["C1"]["version"]) == (
        "F5",
        "reserved",
        2,
    )
    assert (coupons["C2"]["status"], coupons["C2"]["version"]) == ("void", 2)
    assert (coupons["C3"]["flight_id"], coupons["C4"]["flight_id"]) == ("F3", "F4")
    assert coupons["C6"] == next(c for c in STATE["coupons"] if c["coupon_id"] == "C6")
    assert after["payment_events"] == STATE["payment_events"]
    assert after["tickets"] == STATE["tickets"]
    occupied = {r.flight_id: r.occupied for r in oracle.seat_rows(after)}
    assert [occupied[f] for f in ("F3", "F4", "F5")] == [2, 2, 2]
    assert occupied["F1"] == occupied["F2"] == 0
    assert oracle.sellable_map(after)["F2"] == 2
    assert STATE["coupons"][0]["flight_id"] == "F1"  # the input state is never mutated
