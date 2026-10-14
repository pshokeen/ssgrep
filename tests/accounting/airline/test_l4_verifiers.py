"""The L4 verifiers must catch each broken end state (offline, over synthetic states)."""

from __future__ import annotations

import copy

import pytest

from eval.accounting_airline import dataset, domain_oracle as oracle, l4_reference as l4, l4_verify

BEFORE = dataset.seed_state()


def clone() -> dict:
    return copy.deepcopy(BEFORE)


def coupon(state: dict, coupon_id: str) -> dict:
    return next(c for c in state["coupons"] if c["coupon_id"] == coupon_id)


def accepted(request: l4.RebookRequest, state: dict, response: dict | None = None) -> dict:
    row = {
        "request_key": request.request_key, "booking_id": request.booking_id,
        "payload_digest": request.digest, "state": "accepted",
        "response_json": response or {"outcome": "accepted"}, "created_at": dataset.AS_OF,
    }  # fmt: skip
    state["rebooking_requests"].append(row)
    return row


def rejected(request: l4.RebookRequest, state: dict) -> None:
    state["rebooking_requests"].append(
        {
            "request_key": request.request_key,
            "booking_id": request.booking_id,
            "payload_digest": request.digest,
            "state": "rejected",
            "response_json": {"outcome": "rejected", "reason": "capacity_conflict"},
            "created_at": dataset.AS_OF,
        }  # fmt: skip
    )


def joint_final() -> tuple[dict, list[l4.RebookRequest]]:
    plan = oracle.joint_plan(BEFORE)
    requests = [l4.request_for(BEFORE, f"k:{b}", b, legs) for b, legs in sorted(plan.items())]
    after = oracle.state_after_joint_plan(BEFORE)
    for request in requests:
        accepted(request, after)
    return after, requests


def test_correct_joint_end_state_passes() -> None:
    after, requests = joint_final()
    assert l4_verify.verify_joint_apply(BEFORE, after, requests) == []


def test_joint_end_state_rejects_each_broken_variant_on_actual_rows() -> None:
    after, requests = joint_final()

    half = copy.deepcopy(after)
    coupon(half, "C4")["flight_id"] = "F2"  # B2's second leg never moved
    problems = l4_verify.verify_joint_apply(BEFORE, half, requests)
    assert "table coupons: state differs from oracle" in problems
    assert any("F4 occupied" in p for p in problems)

    no_void = copy.deepcopy(after)
    coupon(no_void, "C2")["status"] = "reserved"  # C2 not voided on the direct reissue
    assert "table coupons: state differs from oracle" in l4_verify.verify_joint_apply(
        BEFORE, no_void, requests
    )

    paid = copy.deepcopy(after)
    paid["payment_events"][0]["signed_amount_minor"] += 1
    assert "payment_events changed" in l4_verify.verify_joint_apply(BEFORE, paid, requests)

    refunded = copy.deepcopy(after)
    next(t for t in refunded["tickets"] if t["ticket_id"] == "T1")["face_value_minor"] = 0
    assert "ticket face values/status changed" in l4_verify.verify_joint_apply(
        BEFORE, refunded, requests
    )

    meddled = copy.deepcopy(after)
    coupon(meddled, "C9")["version"] += 1  # an unrelated row moved
    assert "table coupons: state differs from oracle" in l4_verify.verify_joint_apply(
        BEFORE, meddled, requests
    )

    held = copy.deepcopy(after)
    held["seat_holds"][0]["status"] = "released"
    assert "table seat_holds: state differs from oracle" in l4_verify.verify_joint_apply(
        BEFORE, held, requests
    )

    missing_log = copy.deepcopy(after)
    missing_log["rebooking_requests"].pop()
    assert any(
        "request log delta" in p
        for p in l4_verify.verify_joint_apply(BEFORE, missing_log, requests)
    )


def race_requests() -> list[l4.RebookRequest]:
    return [l4.request_for(BEFORE, f"race:{b}", b, ("F3", "F4")) for b in ("B1", "B2")]


@pytest.mark.parametrize("winner", [0, 1])
def test_race_accepts_either_single_winner(winner: int) -> None:
    requests = race_requests()
    move = oracle.MoveRequest(
        requests[winner].request_key, requests[winner].booking_id, ("F3", "F4")
    )
    after = oracle.apply_moves(BEFORE, [move])
    accepted(requests[winner], after)
    rejected(requests[1 - winner], after)
    responses = {
        requests[winner].request_key: {"outcome": "accepted"},
        requests[1 - winner].request_key: {"outcome": "rejected", "reason": "capacity_conflict"},
    }
    assert l4_verify.verify_race(BEFORE, after, requests, responses) == []


def test_race_rejects_oversell_half_move_and_missing_outcomes() -> None:
    requests = race_requests()
    both = oracle.apply_moves(
        BEFORE, [oracle.MoveRequest(r.request_key, r.booking_id, ("F3", "F4")) for r in requests]
    )
    for request in requests:
        accepted(request, both)
    problems = l4_verify.verify_race(
        BEFORE, both, requests, {r.request_key: {"outcome": "accepted"} for r in requests}
    )
    assert "expected 1 accepted + 1 rejected, got 2/0" in problems
    assert any(
        p.startswith("integrity failure") and "F3" in p for p in problems
    )  # oversold raw capacity

    half = oracle.apply_moves(BEFORE, [oracle.MoveRequest("race:B1", "B1", ("F3", "F4"))])
    coupon(half, "C2")["flight_id"] = "F2"  # second leg of the winner not applied
    accepted(requests[0], half)
    rejected(requests[1], half)
    responses = {
        "race:B1": {"outcome": "accepted"},
        "race:B2": {"outcome": "rejected", "reason": "capacity_conflict"},
    }
    problems = l4_verify.verify_race(BEFORE, half, requests, responses)
    assert "final business state matches neither single-winner oracle outcome" in problems

    one_outcome = oracle.apply_moves(BEFORE, [oracle.MoveRequest("race:B1", "B1", ("F3", "F4"))])
    accepted(requests[0], one_outcome)
    problems = l4_verify.verify_race(BEFORE, one_outcome, requests, responses)
    assert "expected exactly 2 new request outcomes, found 1" in problems

    wrong_reason = {**responses, "race:B2": {"outcome": "rejected", "reason": "error"}}
    healthy = oracle.apply_moves(BEFORE, [oracle.MoveRequest("race:B1", "B1", ("F3", "F4"))])
    accepted(requests[0], healthy)
    rejected(requests[1], healthy)
    assert "race:B2 rejected as 'error'" in l4_verify.verify_race(
        BEFORE, healthy, requests, wrong_reason
    )


def test_replay_and_key_conflict_and_rollback_verifiers() -> None:
    request = l4.request_for(BEFORE, "k:replay", "B1", ("F5",))
    after = oracle.apply_moves(BEFORE, [oracle.MoveRequest("k:replay", "B1", ("F5",))])
    accepted(
        request,
        after,
        {"outcome": "accepted", "itinerary": ["F5"], "coupons": [], "booking_id": "B1"},
    )
    first = {
        "outcome": "accepted",
        "itinerary": ["F5"],
        "coupons": [],
        "booking_id": "B1",
        "replayed": False,
    }
    replayed = {**first, "replayed": True}
    assert l4_verify.verify_replay(after, after, first, replayed) == []

    duplicated = copy.deepcopy(after)
    accepted(l4.request_for(BEFORE, "k:replay2", "B1", ("F5",)), duplicated)
    mutated = copy.deepcopy(after)
    coupon(mutated, "C1")["version"] += 1
    problems = l4_verify.verify_replay(
        after, duplicated, first, {**replayed, "itinerary": ["F3", "F4"]}
    )
    assert (
        "replay added a request row" in problems
        and "rebooking_requests changed by replay" in problems
    )
    assert any("replay itinerary" in p for p in problems)
    assert "coupons changed by replay" in l4_verify.verify_replay(after, mutated, first, replayed)
    assert "replay not flagged as replayed" in l4_verify.verify_replay(after, after, first, first)

    conflict = {"outcome": "rejected", "reason": "idempotency_key_reuse"}
    assert l4_verify.verify_key_conflict(after, after, conflict) == []
    assert l4_verify.verify_key_conflict(after, mutated, conflict) == [
        "coupons changed by conflicting-digest request"
    ]
    assert l4_verify.verify_key_conflict(after, after, {"outcome": "accepted"}) != []

    interrupted = clone()
    coupon(interrupted, "C1")["flight_id"] = "F3"  # first coupon committed, second not
    problems = l4_verify.verify_rollback(BEFORE, interrupted)
    assert problems == ["coupons not restored"]
    stray = clone()
    accepted(request, stray)
    assert "interrupted attempt left a committed request outcome" in l4_verify.verify_rollback(
        BEFORE, stray
    )
    assert l4_verify.verify_rollback(BEFORE, clone()) == []


def test_clean_retry_and_stale_plan_verifiers() -> None:
    request = l4.request_for(BEFORE, "k:retry", "B1", ("F3", "F4"))
    after = oracle.apply_moves(BEFORE, [oracle.MoveRequest("k:retry", "B1", ("F3", "F4"))])
    accepted(request, after)
    assert l4_verify.verify_clean_retry(BEFORE, after, request, {"outcome": "accepted"}) == []
    assert l4_verify.verify_clean_retry(BEFORE, BEFORE, request, {"outcome": "accepted"}) != []
    twice = copy.deepcopy(after)
    coupon(twice, "C1")["version"] += 1
    assert "table coupons: state differs from oracle" in l4_verify.verify_clean_retry(
        BEFORE, twice, request, {"outcome": "accepted"}
    )

    held = clone()
    held["seat_holds"].append(
        {"hold_id": "HX", "flight_id": "F5", "cabin": "economy", "quantity": 1, "status": "active",
         "expires_at": dataset.FUTURE_HOLD_EXPIRY}
    )  # fmt: skip
    assert l4_verify.verify_hold_delta(BEFORE, held, "HX", "F5") == []
    assert l4_verify.verify_hold_delta(BEFORE, BEFORE, "HX", "F5") != []
    stale = l4.request_for(held, "k:stale", "B1", ("F5",))
    refused = copy.deepcopy(held)
    rejected(stale, refused)
    refusal = {"outcome": "rejected", "reason": "capacity_conflict"}
    assert l4_verify.verify_stale_plan(held, refused, stale, refusal, "HX") == []
    applied = oracle.apply_moves(held, [oracle.MoveRequest("k:stale", "B1", ("F5",))])
    accepted(stale, applied)
    problems = l4_verify.verify_stale_plan(held, applied, stale, {"outcome": "accepted"}, "HX")
    assert "coupons changed by refused stale plan" in problems
