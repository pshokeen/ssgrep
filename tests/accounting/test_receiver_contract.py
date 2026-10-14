"""Behavior tests for the organization receiver deduplication/tombstone contract.

The document ``docs/examples/receiver-conformance.json`` is replayed through
``apply`` (it is the executable fixture an organization runs against its own
receiver); the remaining tests assert each rule with exact state values.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.contracts import canonical_bytes, digest_of
from ssgrep.accounting.receiver_contract import (
    Applied,
    Key,
    Outcome,
    Quarantined,
    ReceiverState,
    apply,
    current_totals,
)

FIXTURE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "receiver-conformance.json"
RID1 = "0" * 31 + "1"
RID2 = "0" * 31 + "2"
METH = "methodology-hash-1"
UNIT = {"unit": "tokens", "method": "estimate", "tokenizer": None, "currency": None}
UK = ("tokens", "estimate", None, None)
KA = Key("tenant-a", "device-1", RID1)


def q(value: int | None) -> dict[str, Any]:
    if value is None:
        return {
            "value": None,
            "unit": "tokens",
            "basis": "unavailable",
            "method": "",
            "reason": "no_evidence",
        }
    return {"value": value, "unit": "tokens", "basis": "estimated", "method": "m", "reason": ""}


def mk(
    rid: str,
    revision: int,
    *,
    net: int | None = 100,
    rtype: str | None = None,
    methodology: str = METH,
    unit: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kind = rtype or ("contribution" if revision == 1 else "adjustment")
    body: dict[str, Any] = {
        "schema": 1,
        "record_id": rid,
        "revision": revision,
        "supersedes_revision": None if revision == 1 else revision - 1,
        "record_type": kind,
        "task_id": "synthetic-task-1",
        "period": "2026-01-02",
        "unit": dict(unit or UNIT),
        "scenarios": {},
        "known_costs": {},
        "coverage": {},
        "methodology": methodology,
    }
    if kind != "deletion":
        credit = None if net is None else max(net, 0)
        debit = None if net is None else max(-net, 0)
        body["scenarios"] = {"central": {"credit": q(credit), "debit": q(debit), "net": q(net)}}
        body["known_costs"] = {"output_tokens": 5}
        body["coverage"] = {"complete": True, "sources": 1}
    body.update(extra or {})
    return body


def run(steps: list[tuple[Key, dict[str, Any]]], state: ReceiverState | None = None):
    state = state or ReceiverState()
    outcomes = []
    for key, body in steps:
        state, outcome = apply(state, key, body)
        outcomes.append(outcome)
    return state, outcomes


def net(state: ReceiverState, scenario: str = "central") -> int | None:
    return current_totals(state, methodology=METH, unit_key=UK)["scenarios"][scenario]["net"][
        "total"
    ]


# --- executable conformance fixture ---------------------------------------


def _load() -> dict[str, Any]:
    def no_float(text: str) -> float:
        raise AssertionError(f"float in fixture: {text}")

    return json.loads(FIXTURE.read_text(), parse_float=no_float)


def _case_ids() -> list[str]:
    return [c["name"] for c in _load()["cases"]]


def test_fixture_covers_required_scenarios() -> None:
    names = set(_case_ids())
    assert {
        "retry_and_reorder",
        "conflicting_duplicate_quarantine",
        "tombstone_blocks_replay",
        "tenant_isolation",
        "negative_net_preserved",
        "unknown_net_not_zero",
    } <= names


def test_fixture_is_synthetic_only() -> None:
    doc = _load()
    text = FIXTURE.read_text()
    assert "/Users" not in text and "/home" not in text and "secret" not in text.lower()
    for case in doc["cases"]:
        for step in case["steps"]:
            body = step["body"]
            assert re.fullmatch(r"0{31}\d", body["record_id"])
            assert re.fullmatch(r"synthetic-task-\d+", body["task_id"])


@pytest.mark.parametrize("name", _case_ids())
def test_fixture_case_replays(name: str) -> None:
    doc = _load()
    case = next(c for c in doc["cases"] if c["name"] == name)
    state = ReceiverState()
    for i, step in enumerate(case["steps"]):
        k = step["key"]
        state, outcome = apply(state, Key(k["tenant"], k["device"], k["record_id"]), step["body"])
        assert outcome is Outcome(step["expect"]), f"{name} step {i}"
    current = {f"{k.tenant}/{k.device}/{k.record_id}": a.revision for k, a in state.current.items()}
    assert current == case["expect_current"]
    tombs = {f"{k.tenant}/{k.device}/{k.record_id}": r for k, r in state.tombstones.items()}
    assert tombs == case["expect_tombstones"]
    assert len(state.quarantine) == case.get("expect_quarantined", 0)
    totals = current_totals(state, methodology=doc["methodology"], unit_key=tuple(doc["unit_key"]))
    assert totals["scenarios"]["central"]["net"]["total"] == case["expect_net_central"]
    if "expect_unknown_central" in case:
        assert totals["scenarios"]["central"]["net"]["unknown"] == case["expect_unknown_central"]


# --- revision rules ---------------------------------------------------------


def test_first_sight_is_applied_new_with_exact_state() -> None:
    body = mk(RID1, 1)
    state, outcome = apply(ReceiverState(), KA, body)
    assert outcome is Outcome.APPLIED_NEW
    assert dict(state.current) == {KA: Applied(1, digest_of(body), "contribution", body)}
    assert dict(state.tombstones) == {}
    assert state.quarantine == () and state.rejected == ()
    assert net(state) == 100


def test_higher_revision_replaces_current_row() -> None:
    state, outcomes = run([(KA, mk(RID1, 1, net=100)), (KA, mk(RID1, 2, net=-40))])
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.APPLIED_UPSERT]
    new = mk(RID1, 2, net=-40)
    assert dict(state.current) == {KA: Applied(2, digest_of(new), "adjustment", new)}
    assert net(state) == -40


def test_identical_duplicate_is_noop_and_state_equal() -> None:
    before, _ = apply(ReceiverState(), KA, mk(RID1, 3))
    after, outcome = apply(before, KA, mk(RID1, 3))
    assert outcome is Outcome.DUPLICATE_NOOP
    assert after == before
    assert net(after) == 100


def test_retry_then_reordered_older_revision() -> None:
    state, outcomes = run(
        [(KA, mk(RID1, 3, net=300)), (KA, mk(RID1, 3, net=300)), (KA, mk(RID1, 2, net=200))]
    )
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.DUPLICATE_NOOP, Outcome.REJECTED_OLD_REVISION]
    assert state.current[KA].revision == 3
    assert net(state) == 300  # rev 3 contributes once, rev 2 adds nothing
    assert dict(state.tombstones) == {}


def test_same_revision_different_digest_quarantines_without_replacing() -> None:
    held = mk(RID1, 2, net=100)
    clash = mk(RID1, 2, net=999)
    before, _ = apply(ReceiverState(), KA, held)
    after, outcome = apply(before, KA, clash)
    assert outcome is Outcome.QUARANTINED
    assert after.current == before.current
    assert net(after) == 100
    assert after.quarantine == (
        Quarantined(KA, 2, digest_of(clash), digest_of(held), "same_revision_different_digest"),
    )
    again, outcome2 = apply(after, KA, clash)
    assert outcome2 is Outcome.QUARANTINED and again == after  # retries do not grow state


def test_quarantine_records_digests_never_bodies() -> None:
    state, _ = run([(KA, mk(RID1, 2)), (KA, mk(RID1, 2, net=7))])
    entry = state.quarantine[0]
    assert set(vars(entry)) == {"key", "revision", "digest", "held_digest", "reason"}
    assert "synthetic-task-1" not in repr(state.quarantine)


def test_lower_revision_rejected_and_recorded() -> None:
    older = mk(RID1, 1)
    state, outcomes = run([(KA, mk(RID1, 4)), (KA, older)])
    assert outcomes[1] is Outcome.REJECTED_OLD_REVISION
    assert state.current[KA].revision == 4
    assert [(r.key, r.revision, r.digest) for r in state.rejected] == [(KA, 1, digest_of(older))]


# --- deletion / tombstones --------------------------------------------------


def test_deletion_removes_row_and_retains_tombstone() -> None:
    state, outcomes = run([(KA, mk(RID1, 1)), (KA, mk(RID1, 2, rtype="deletion"))])
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.APPLIED_TOMBSTONE]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {KA: 2}
    assert net(state) == 0


def test_tombstone_blocks_replay_of_older_contribution() -> None:
    state, outcomes = run(
        [
            (KA, mk(RID1, 1)),
            (KA, mk(RID1, 2, rtype="deletion")),
            (KA, mk(RID1, 1)),
            (KA, mk(RID1, 2, net=55)),
        ]
    )
    assert outcomes[2:] == [Outcome.REJECTED_TOMBSTONED, Outcome.REJECTED_TOMBSTONED]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {KA: 2}
    assert net(state) == 0


def test_recreation_above_tombstone_is_new_and_tombstone_kept() -> None:
    state, outcomes = run(
        [(KA, mk(RID1, 1)), (KA, mk(RID1, 2, rtype="deletion")), (KA, mk(RID1, 3, net=60))]
    )
    assert outcomes[2] is Outcome.APPLIED_NEW
    assert state.current[KA].revision == 3
    assert dict(state.tombstones) == {KA: 2}
    assert net(state) == 60
    replay, outcome = apply(state, KA, mk(RID1, 2, net=1))
    assert outcome is Outcome.REJECTED_TOMBSTONED and replay.current == state.current
    stale, outcome = apply(state, KA, mk(RID1, 3, net=1))
    assert outcome is Outcome.QUARANTINED and stale.current == state.current


@pytest.mark.parametrize(("del_rev", "tombs"), [(3, {}), (2, {KA: 2})])
def test_deletion_not_above_current_revision_cannot_delete_newer_row(
    del_rev: int, tombs: dict[Key, int]
) -> None:
    before, _ = run([(KA, mk(RID1, 3))])
    after, outcome = apply(before, KA, mk(RID1, del_rev, rtype="deletion"))
    assert outcome is Outcome.REJECTED_OLD_REVISION
    assert after.current == before.current
    assert dict(after.tombstones) == tombs
    assert net(after) == 100


def test_deletion_of_unseen_key_still_records_tombstone() -> None:
    state, outcomes = run([(KA, mk(RID1, 5, rtype="deletion")), (KA, mk(RID1, 1))])
    assert outcomes == [Outcome.APPLIED_TOMBSTONE, Outcome.REJECTED_TOMBSTONED]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {KA: 5}


def test_repeated_identical_deletion_is_noop() -> None:
    before, _ = run([(KA, mk(RID1, 1)), (KA, mk(RID1, 2, rtype="deletion"))])
    after, outcome = apply(before, KA, mk(RID1, 2, rtype="deletion"))
    assert outcome is Outcome.DUPLICATE_NOOP
    assert after == before


def test_deletion_below_tombstone_is_tombstoned_and_conflicting_deletion_quarantined() -> None:
    first = mk(RID1, 4, rtype="deletion")
    before, _ = run([(KA, first)])
    low, outcome = apply(before, KA, mk(RID1, 3, rtype="deletion"))
    assert outcome is Outcome.REJECTED_TOMBSTONED
    assert dict(low.tombstones) == {KA: 4}
    clash = mk(RID1, 4, rtype="deletion", extra={"task_id": "synthetic-task-2"})
    after, outcome = apply(before, KA, clash)
    assert outcome is Outcome.QUARANTINED
    assert dict(after.tombstones) == {KA: 4}
    assert after.quarantine[0].held_digest == digest_of(first)


def test_later_deletion_advances_tombstone_and_clears_recreated_row() -> None:
    state, outcomes = run(
        [
            (KA, mk(RID1, 2, rtype="deletion")),
            (KA, mk(RID1, 3)),
            (KA, mk(RID1, 4, rtype="deletion")),
        ]
    )
    assert outcomes == [Outcome.APPLIED_TOMBSTONE, Outcome.APPLIED_NEW, Outcome.APPLIED_TOMBSTONE]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {KA: 4}


# --- isolation / authenticated identity --------------------------------------


def test_tenant_and_device_isolation() -> None:
    kb = Key("tenant-b", "device-1", RID1)
    kc = Key("tenant-a", "device-2", RID1)
    state, outcomes = run(
        [
            (KA, mk(RID1, 3, rtype="deletion")),
            (kb, mk(RID1, 1)),
            (kc, mk(RID1, 1)),
            (KA, mk(RID1, 1)),
        ]
    )
    assert outcomes == [
        Outcome.APPLIED_TOMBSTONE,
        Outcome.APPLIED_NEW,
        Outcome.APPLIED_NEW,
        Outcome.REJECTED_TOMBSTONED,
    ]
    assert dict(state.tombstones) == {KA: 3}
    assert set(state.current) == {kb, kc}


def test_tenant_claim_in_body_is_not_accepted_or_used() -> None:
    kb = Key("tenant-b", "device-1", RID1)
    for field in ("tenant", "device", "attributes"):
        state, outcome = apply(ReceiverState(), kb, mk(RID1, 1, extra={field: "tenant-a"}))
        assert outcome is Outcome.REJECTED_INVALID
        assert state == ReceiverState()
    state, _ = apply(ReceiverState(), kb, mk(RID1, 1))
    assert set(state.current) == {kb}


# --- strict validation --------------------------------------------------------


def _bad_bodies() -> dict[str, dict[str, Any]]:
    float_net = mk(RID1, 1)
    float_net["scenarios"]["central"]["net"]["value"] = 1.5
    unavailable_with_value = mk(RID1, 1)
    unavailable_with_value["scenarios"]["central"]["net"]["basis"] = "unavailable"
    bad_unit = mk(RID1, 1)
    bad_unit["unit"].pop("currency")
    missing = mk(RID1, 1)
    del missing["period"]
    deletion_with_scenarios = mk(RID1, 2, rtype="deletion")
    deletion_with_scenarios["scenarios"] = mk(RID1, 1)["scenarios"]
    huge = mk(RID1, 1, extra={"known_costs": {f"k{i}": "x" * 500 for i in range(60)}})
    huge["coverage"] = {f"c{i}": "y" * 500 for i in range(60)}
    huge["scenarios"]["central"]["net"]["method"] = "z" * 100_000
    return {
        "bool_revision": mk(RID1, 1, extra={"revision": True}),
        "zero_revision": mk(RID1, 0),
        "float_revision": mk(RID1, 1, extra={"revision": 1.0}),
        "float_net": float_net,
        "string_revision": mk(RID1, 1, extra={"revision": "1"}),
        "unknown_key": mk(RID1, 1, extra={"extra": 1}),
        "missing_key": missing,
        "wrong_schema": mk(RID1, 1, extra={"schema": 2}),
        "record_id_mismatch": mk(RID2, 1),
        "bad_record_type": mk(RID1, 1, extra={"record_type": "refund"}),
        "bad_period": mk(RID1, 1, extra={"period": "2026-13-40"}),
        "bad_unit": bad_unit,
        "bad_scenario": mk(
            RID1, 1, extra={"scenarios": {"best": mk(RID1, 1)["scenarios"]["central"]}}
        ),
        "unavailable_inconsistent": unavailable_with_value,
        "supersedes_on_first": mk(RID1, 1, extra={"supersedes_revision": 1}),
        "supersedes_not_lower": mk(RID1, 3, extra={"supersedes_revision": 3}),
        "deletion_with_scenarios": deletion_with_scenarios,
        "float_known_cost": mk(RID1, 1, extra={"known_costs": {"cost": 0.5}}),
        "nested_coverage": mk(RID1, 1, extra={"coverage": {"a": {"b": 1}}}),
        "empty_methodology": mk(RID1, 1, extra={"methodology": ""}),
        "oversize": huge,
    }


@pytest.mark.parametrize("name", sorted(_bad_bodies()))
def test_invalid_bodies_rejected_without_state_change(name: str) -> None:
    before, _ = run([(KA, mk(RID1, 1))])
    after, outcome = apply(before, KA, _bad_bodies()[name])
    assert outcome is Outcome.REJECTED_INVALID
    assert after is before or after == before


def test_non_mapping_body_and_bad_key_are_invalid() -> None:
    empty = ReceiverState()
    for body in (None, [], "x", 3):
        assert apply(empty, KA, body) == (empty, Outcome.REJECTED_INVALID)  # ty: ignore[invalid-argument-type]
    for key in (Key("", "d", RID1), Key("t", "", RID1), Key("t", "d", "")):
        assert apply(empty, key, mk(RID1, 1)) == (empty, Outcome.REJECTED_INVALID)


def test_body_size_limit_is_64_kib_of_canonical_bytes() -> None:
    def sized(entries: int) -> dict[str, Any]:
        costs = {f"k{i}": "x" * 1000 for i in range(entries)}
        return mk(
            RID1,
            1,
            extra={"known_costs": costs, "coverage": {f"c{i}": "y" * 1000 for i in range(entries)}},
        )

    under, over = sized(31), sized(33)
    assert len(canonical_bytes(under)) < 64 * 1024 < len(canonical_bytes(over))
    assert apply(ReceiverState(), KA, under)[1] is Outcome.APPLIED_NEW
    assert apply(ReceiverState(), KA, over) == (ReceiverState(), Outcome.REJECTED_INVALID)


def test_apply_does_not_mutate_inputs_and_state_is_immutable() -> None:
    body = mk(RID1, 1)
    snapshot = json.loads(json.dumps(body))
    before = ReceiverState()
    state, _ = apply(before, KA, body)
    assert body == snapshot
    assert before == ReceiverState() and dict(before.current) == {}
    body["scenarios"]["central"]["net"]["value"] = 12345  # caller mutates afterwards
    assert state.current[KA].body == snapshot
    with pytest.raises(TypeError):
        state.current[KA] = state.current[KA]  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        state.tombstones[KA] = 9  # ty: ignore[invalid-assignment]


def test_rejected_log_is_bounded_to_newest_1000() -> None:
    state, _ = apply(ReceiverState(), KA, mk(RID1, 2000))
    for rev in range(1, 1051):
        state, outcome = apply(state, KA, mk(RID1, rev))
        assert outcome is Outcome.REJECTED_OLD_REVISION
    assert len(state.rejected) == 1000
    assert state.rejected[-1].revision == 1050 and state.rejected[0].revision == 51


def test_quarantine_is_bounded_to_newest_1000() -> None:
    state, _ = apply(ReceiverState(), KA, mk(RID1, 1, net=0))
    for n in range(1, 1051):
        state, outcome = apply(state, KA, mk(RID1, 1, net=n))
        assert outcome is Outcome.QUARANTINED
    assert len(state.quarantine) == 1000
    assert state.quarantine[-1].digest == digest_of(mk(RID1, 1, net=1050))
    assert net(state) == 0


# --- permutation / order insensitivity ----------------------------------------


@pytest.mark.parametrize(
    "bodies",
    [
        [mk(RID1, 1, net=10), mk(RID1, 2, net=20), mk(RID1, 3, rtype="deletion")],
        [mk(RID1, 1, net=10), mk(RID1, 2, rtype="deletion"), mk(RID1, 3, net=30)],
        [mk(RID1, 1, net=10), mk(RID1, 2, net=20), mk(RID1, 3, net=30), mk(RID1, 4, net=-5)],
        [
            mk(RID1, 1, net=10),
            mk(RID1, 2, rtype="deletion"),
            mk(RID1, 4, rtype="deletion"),
            mk(RID1, 3, net=30),
        ],
    ],
    ids=["delete-last", "delete-then-recreate", "four-adjustments", "two-deletions-between"],
)
def test_every_permutation_converges_without_double_counting(bodies: list[dict[str, Any]]) -> None:
    finals = set()
    for order in itertools.permutations(bodies):
        state, _ = run([(KA, b) for b in order])
        # replaying the whole set again (retry storm) must not change anything
        again, outcomes = run([(KA, b) for b in order], state)
        assert again.current == state.current and again.tombstones == state.tombstones
        assert Outcome.APPLIED_NEW not in outcomes and Outcome.APPLIED_UPSERT not in outcomes
        finals.add(
            (
                tuple(sorted((repr(k), a.revision, a.digest) for k, a in state.current.items())),
                tuple(sorted(state.tombstones.items(), key=repr)),
                net(state),
            )
        )
    assert len(finals) == 1


# --- totals --------------------------------------------------------------------


def test_totals_sum_current_rows_per_scenario_with_negatives_preserved() -> None:
    state, _ = run(
        [
            (KA, mk(RID1, 1, net=100)),
            (KA, mk(RID1, 2, net=-250)),
            (Key("t", "d", RID2), mk(RID2, 1, net=100)),
        ]
    )
    totals = current_totals(state, methodology=METH, unit_key=UK)
    central = totals["scenarios"]["central"]["net"]
    assert central == {"total": -150, "known": 2, "unknown": 0}
    assert totals["scenarios"]["central"]["debit"]["total"] == 250
    assert totals["scenarios"]["central"]["credit"]["total"] == 100
    assert totals["rows"] == 2
    assert totals["methodology"] == METH
    assert totals["unit"] == UNIT
    assert all(type(v["total"]) is int for v in totals["scenarios"]["central"].values())


def test_scenario_absent_from_row_counts_as_unknown() -> None:
    state, _ = run([(KA, mk(RID1, 1))])
    opt = current_totals(state, methodology=METH, unit_key=UK)["scenarios"]["optimistic"]["net"]
    assert opt == {"total": None, "known": 0, "unknown": 1}


def test_unknown_net_is_counted_not_zero() -> None:
    only, _ = run([(KA, mk(RID1, 1, net=None))])
    assert current_totals(only, methodology=METH, unit_key=UK)["scenarios"]["central"]["net"] == {
        "total": None,
        "known": 0,
        "unknown": 1,
    }
    mixed, _ = run([(Key("t", "d", RID2), mk(RID2, 1, net=40))], only)
    assert current_totals(mixed, methodology=METH, unit_key=UK)["scenarios"]["central"]["net"] == {
        "total": 40,
        "known": 1,
        "unknown": 1,
    }


def test_empty_state_totals_are_exact_zero() -> None:
    totals = current_totals(ReceiverState(), methodology=METH, unit_key=UK)
    assert totals["rows"] == 0
    assert totals["scenarios"]["central"]["net"] == {"total": 0, "known": 0, "unknown": 0}


def test_mixing_methodologies_or_units_raises() -> None:
    other_unit = {"unit": "usd_micros", "method": "tariff", "tokenizer": None, "currency": "USD"}
    rid3 = "0" * 31 + "3"
    base, _ = run([(KA, mk(RID1, 1, net=10))])
    assert net(base) == 10
    by_method, _ = run(
        [(Key("t", "d", RID2), mk(RID2, 1, net=20, methodology="methodology-hash-2"))], base
    )
    by_unit, _ = run([(Key("t", "d", rid3), mk(rid3, 1, net=30, unit=other_unit))], base)
    for mixed in (by_method, by_unit):
        with pytest.raises(ValueError):
            current_totals(mixed, methodology=METH, unit_key=UK)
    with pytest.raises(ValueError):
        current_totals(base, methodology="methodology-hash-2", unit_key=UK)
    with pytest.raises(ValueError):
        current_totals(base, methodology=METH, unit_key=("usd_micros", "tariff", None, "USD"))
    with pytest.raises(ValueError):
        current_totals(base, methodology=METH, unit_key=("tokens", "estimate"))  # ty: ignore[invalid-argument-type]
