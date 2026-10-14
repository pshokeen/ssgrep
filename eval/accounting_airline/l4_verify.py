"""Independent post-state verifiers for the L4 scenarios.

Pure functions over states read back from the real database. They share no code with the
plpgsql operation under test: expected states come from ``domain_oracle`` (a Python
reference model) and the request log is compared by value. Each returns a list of
violation strings; an empty list means the scenario's invariants hold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import dataset, domain_oracle as oracle
from .domain_oracle import MoveRequest, State

BUSINESS_TABLES = tuple(t.name for t in dataset.TABLES if t.name != "rebooking_requests")


def digests(state: State) -> dict[str, str]:
    return dataset.logical_digests(state)


def changed_tables(before: State, after: State) -> list[str]:
    left, right = digests(before), digests(after)
    return [t for t in left if left[t] != right[t]]


def new_requests(before: State, after: State) -> list[dict[str, Any]]:
    seen = {r["request_key"] for r in before["rebooking_requests"]}
    return [dict(r) for r in after["rebooking_requests"] if r["request_key"] not in seen]


def _guarded(check: Callable[[], list[str]]) -> list[str]:
    try:
        return check()
    except oracle.IntegrityError as exc:
        return [f"integrity failure: {exc}"]


def payments_unchanged(before: State, after: State) -> list[str]:
    same = dataset.table_digest("payment_events", before["payment_events"]) == (
        dataset.table_digest("payment_events", after["payment_events"])
    )
    tickets_same = dataset.table_digest("tickets", before["tickets"]) == (
        dataset.table_digest("tickets", after["tickets"])
    )
    problems = [] if same else ["payment_events changed"]
    return problems + ([] if tickets_same else ["ticket face values/status changed"])


def occupancy(state: State) -> dict[str, int]:
    return {r.flight_id: r.occupied for r in oracle.seat_rows(state)}


def verify_joint_apply(before: State, after: State, requests: Sequence[Any]) -> list[str]:
    """Final-state oracle for the canonical joint plan (B1->F5, B2->F3+F4)."""
    problems = payments_unchanged(before, after)
    expected = oracle.state_after_joint_plan(before)
    problems += oracle.verify_state_delta(after, expected)

    def seats() -> list[str]:
        occ = occupancy(after)
        out = [f"{f} occupied {occ[f]} != 2" for f in ("F3", "F4", "F5") if occ[f] != 2]
        out += [f"{f} occupied {occ[f]} != 0" for f in ("F1", "F2") if occ[f] != 0]
        sellable = oracle.sellable_map(after)
        if sellable["F2"] != 2:
            out.append(f"F2 sellable {sellable['F2']} != 2")
        return out

    problems += _guarded(seats)
    keys = {r.request_key: r for r in requests}
    added = new_requests(before, after)
    if sorted(r["request_key"] for r in added) != sorted(keys):
        problems.append(f"request log delta {[r['request_key'] for r in added]} != {sorted(keys)}")
    for row in added:
        if row["state"] != "accepted":
            problems.append(f"{row['request_key']} not accepted")
        if row["payload_digest"] != keys[row["request_key"]].digest:
            problems.append(f"{row['request_key']} digest mismatch")
    return problems


def _expected_after_winner(before: State, winner: MoveRequest) -> dict[str, list[dict[str, Any]]]:
    return oracle.apply_moves(before, [winner])


def verify_race(
    before: State,
    after: State,
    requests: Sequence[Any],
    responses: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Exactly one complete two-leg commit, one capacity_conflict, no oversell."""
    problems = payments_unchanged(before, after)
    accepted = [k for k, r in responses.items() if r.get("outcome") == "accepted"]
    rejected = [k for k, r in responses.items() if r.get("outcome") == "rejected"]
    if len(accepted) != 1 or len(rejected) != 1:
        problems.append(f"expected 1 accepted + 1 rejected, got {len(accepted)}/{len(rejected)}")
    for key in rejected:
        if responses[key].get("reason") != "capacity_conflict":
            problems.append(f"{key} rejected as {responses[key].get('reason')!r}")
    by_key = {r.request_key: r for r in requests}
    outcomes: list[list[str]] = []
    for winner in requests:
        move = MoveRequest(winner.request_key, winner.booking_id, winner.itinerary)
        expected = _expected_after_winner(before, move)
        outcomes.append(oracle.verify_state_delta(after, expected))
    if all(outcomes):
        problems.append("final business state matches neither single-winner oracle outcome")
        problems += outcomes[0]
    for t in changed_tables(before, after):
        if t not in ("coupons", "rebooking_requests"):
            problems.append(f"unexpected change in {t}")

    def seats() -> list[str]:
        occ = occupancy(after)
        return [f"{f} occupied {occ[f]} != 2" for f in ("F3", "F4") if occ[f] != 2]

    problems += _guarded(seats)
    added = new_requests(before, after)
    if len(added) != 2:
        problems.append(f"expected exactly 2 new request outcomes, found {len(added)}")
    states = sorted(r["state"] for r in added)
    if states != ["accepted", "rejected"]:
        problems.append(f"request outcome states {states}")
    for row in added:
        request = by_key.get(row["request_key"])
        if request is None or row["payload_digest"] != request.digest:
            problems.append(f"request row {row['request_key']} digest not the client's")
    return problems


def verify_replay(
    before_replay: State,
    after_replay: State,
    first: Mapping[str, Any],
    replayed: Mapping[str, Any],
) -> list[str]:
    problems = [f"{t} changed by replay" for t in changed_tables(before_replay, after_replay)]
    if len(before_replay["rebooking_requests"]) != len(after_replay["rebooking_requests"]):
        problems.append("replay added a request row")
    if replayed.get("replayed") is not True:
        problems.append("replay not flagged as replayed")
    if first.get("replayed") is not False:
        problems.append("first response unexpectedly flagged replayed")
    for field in ("outcome", "itinerary", "coupons", "booking_id"):
        if replayed.get(field) != first.get(field):
            problems.append(
                f"replay {field} {replayed.get(field)!r} != original {first.get(field)!r}"
            )
    return problems


def verify_key_conflict(before: State, after: State, response: Mapping[str, Any]) -> list[str]:
    problems = [f"{t} changed by conflicting-digest request" for t in changed_tables(before, after)]
    if response.get("outcome") != "rejected" or response.get("reason") != "idempotency_key_reuse":
        problems.append(f"same key/different digest returned {dict(response)}")
    return problems


def verify_rollback(before: State, after_failure: State) -> list[str]:
    """Every business row and the request log are back to the baseline digests."""
    problems = [f"{t} not restored" for t in changed_tables(before, after_failure)]
    if new_requests(before, after_failure):
        problems.append("interrupted attempt left a committed request outcome")
    return problems


def verify_clean_retry(
    before: State, after_retry: State, request: Any, response: Mapping[str, Any]
) -> list[str]:
    move = MoveRequest(request.request_key, request.booking_id, request.itinerary)
    expected = oracle.apply_moves(before, [move])
    problems = payments_unchanged(before, after_retry)
    problems += oracle.verify_state_delta(after_retry, expected)
    added = new_requests(before, after_retry)
    if [r["request_key"] for r in added] != [request.request_key]:
        problems.append(f"retry request delta {[r['request_key'] for r in added]}")
    elif added[0]["state"] != "accepted":
        problems.append("retry request not accepted")
    if response.get("outcome") != "accepted":
        problems.append(f"retry response {dict(response)}")
    return problems


def verify_stale_plan(
    baseline_with_hold: State,
    after_apply: State,
    request: Any,
    response: Mapping[str, Any],
    hold_id: str,
) -> list[str]:
    """Stale plan refused: coupons/payments untouched, one rejected capacity_conflict row."""
    problems = payments_unchanged(baseline_with_hold, after_apply)
    for table in BUSINESS_TABLES:
        if dataset.table_digest(table, baseline_with_hold[table]) != dataset.table_digest(
            table, after_apply[table]
        ):
            problems.append(f"{table} changed by refused stale plan")
    if response.get("outcome") != "rejected" or response.get("reason") != "capacity_conflict":
        problems.append(f"stale apply returned {dict(response)}")
    added = new_requests(baseline_with_hold, after_apply)
    if [(r["request_key"], r["state"]) for r in added] != [(request.request_key, "rejected")]:
        problems.append(
            f"stale-plan request delta {[(r['request_key'], r['state']) for r in added]}"
        )
    if not any(h["hold_id"] == hold_id for h in after_apply["seat_holds"]):
        problems.append("competing hold missing from final state")
    return problems


def verify_hold_delta(before: State, with_hold: State, hold_id: str, flight_id: str) -> list[str]:
    """The recorded changed baseline differs from the original by exactly one hold."""
    changed = changed_tables(before, with_hold)
    problems = [] if changed == ["seat_holds"] else [f"hold step changed {changed}"]
    added = [h for h in with_hold["seat_holds"] if h["hold_id"] == hold_id]
    if len(added) != 1 or added[0]["flight_id"] != flight_id or added[0]["status"] != "active":
        problems.append(f"hold {hold_id} not recorded as an active hold on {flight_id}")
    return problems
