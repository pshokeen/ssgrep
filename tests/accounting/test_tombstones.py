"""Remote-deletion signalling: plans, revocation-safe spooling, acknowledgement bookkeeping.

Everything is real: ledger rows come from the real ``Ledger`` reducers and are projected with the
install's real pseudonym key into a registry written by ``mark_queued``/``mark_acked`` exactly as
delivery does; policies are saved with ``policy.save_profile`` and judged by the real
``export_policy.gate``; spool batches and delivery documents are real files under ``tmp_path``;
every tombstone body is fed through the reference receiver after the contributions it deletes.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import acks, export_ids, export_spool, policy
from ssgrep.accounting.contracts import TaskId, digest_of
from ssgrep.accounting.export_records import (
    Planned,
    Registry,
    load_registry,
    mark_acked,
    mark_queued,
    plan_revisions,
    project_candidates,
    root_key_of,
)
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.receiver_contract import (
    Key,
    Outcome,
    ReceiverState,
    apply,
    current_totals,
)
from ssgrep.accounting.replay import Ledger
from ssgrep.accounting.tombstones import (
    MAX_OBLIGATIONS,
    TombstonePlan,
    enqueue_tombstones,
    forget_acknowledged_deletions,
    pending_tombstones,
    plan_tombstones,
)
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.test_export_records import CANARY, QUANTITY, canary_rows
from tests.accounting.valuation_builders import METHOD, atom, candidate

PROFILE = "personal-local"
DEST = "acme-org"
OTHER = "other-org"
ENDPOINT = "https://collector.example.test"
NOW = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)
STAMP = int(NOW.timestamp())
T_ID = re.compile(r"^t[0-9]{10}-[0-9a-f]{8}-[0-9]{4}$")
UNIT_KEY = ("context_tokens", "chars_div4_v1", None, None)
ROOT1, ROOT2, ROOT3 = (TaskId(f"root-{n}", "turn-1", "main") for n in (1, 2, 3))
DAY1, DAY2, DAY3 = "2026-11-01", "2026-11-02", "2026-11-03"


# ---- builders --------------------------------------------------------------------------------


def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def tree(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def make_policy(
    *, enabled: tuple[str, ...] = (DEST, OTHER), logs: tuple[str, ...] = (DEST, OTHER)
) -> ProfilePolicy:
    """Both destinations known; ``logs`` may carry logs; ``enabled`` have logs switched on."""
    destinations = tuple(
        DestinationPolicy(
            name,
            ENDPOINT,
            (Signal.METRICS, Signal.LOGS) if name in logs else (Signal.METRICS,),
            logs_permitted=name in logs,
        )
        for name in (DEST, OTHER)
    )
    exports = tuple((name, "logs") for name in enabled if name in logs)
    return policy.save_profile(
        ProfilePolicy(destinations=destinations, enabled_exports=exports, collection_enabled=True)
    )


def key() -> bytes:
    minted = export_ids.load_key(PROFILE, create=True)
    assert minted is not None
    return minted


def ledger_rows() -> Any:
    ledger = Ledger(METHOD)
    ledger.ingest(
        EvidenceBundle(
            (
                candidate("a", task=ROOT1, day=DAY1),
                candidate("b", task=ROOT2, day=DAY3, immediate=(atom("s2", 80, 0, "e2"),)),
                candidate("c", task=ROOT3, day=DAY2, immediate=(atom("s3", 120, 0, "e3"),)),
            ),
            (),
        )
    )
    return ledger.rows()


def plan_for(destination: str, rows: Any = None) -> list[Planned]:
    candidates, omitted = project_candidates(
        rows if rows is not None else ledger_rows(), key=key(), destination=destination
    )
    assert omitted == 0
    return plan_revisions(Registry(), candidates)


def seed(destination: str) -> dict[str, Planned]:
    """Root-1 and root-2 records are SENT (acked); root-3's only ever sat queued."""
    planned = plan_for(destination)
    by_root = {p.root_key: p for p in planned}
    assert set(by_root) == {root_key_of(t.root) for t in (ROOT1, ROOT2, ROOT3)}
    sent = [by_root[root_key_of(ROOT1.root)], by_root[root_key_of(ROOT2.root)]]
    mark_queued(PROFILE, destination, sent, "b-sent-0001")
    mark_acked(PROFILE, destination, "b-sent-0001")
    mark_queued(PROFILE, destination, [by_root[root_key_of(ROOT3.root)]], "b-unsent-0001")
    return by_root


def ids_of(by_root: dict[str, Planned], *roots: TaskId) -> tuple[str, ...]:
    return tuple(sorted(by_root[root_key_of(r.root)].record_id for r in roots))


def plan_of(plans: list[TombstonePlan], destination: str) -> TombstonePlan:
    (found,) = [p for p in plans if p.destination_id == destination]
    return found


def logs_bodies(request: bytes) -> list[dict[str, Any]]:
    """Record bodies of one serialized OTLP logs request, in request order."""
    document = json.loads(request)
    found = []
    for resource in document["resourceLogs"]:
        for scope in resource["scopeLogs"]:
            found += [json.loads(r["body"]["stringValue"]) for r in scope["logRecords"]]
    return found


def spooled(destination: str) -> list[export_spool.Meta]:
    return export_spool.list_batches(PROFILE, destination, Signal.LOGS)


def tombstone_bodies(destination: str) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for meta in spooled(destination):
        if meta.kind == "tombstone":
            bodies += logs_bodies(
                export_spool.read_body(PROFILE, destination, Signal.LOGS, meta.batch_id)
            )
    return bodies


def receive(
    state: ReceiverState, bodies: list[dict[str, Any]]
) -> tuple[ReceiverState, list[Outcome]]:
    outcomes = []
    for body in bodies:
        state, outcome = apply(state, Key("t", "dev", body["record_id"]), body)
        outcomes.append(outcome)
    return state, outcomes


def rows_total(state: ReceiverState) -> int:
    return current_totals(state, methodology=METHOD.hash, unit_key=UNIT_KEY)["rows"]


def synthetic(index: int, destination: str = DEST) -> Planned:
    """A receiver-valid revision-1 body that needs no ledger."""
    record_id = export_ids.pseudonym(key(), "record", destination, f"i{index}")
    body: dict[str, Any] = {
        "schema": 1, "record_id": record_id, "revision": 1, "supersedes_revision": None,
        "record_type": "contribution", "task_id": record_id, "period": DAY1,
        "unit": {"unit": "context_tokens", "method": "chars_div4_v1", "tokenizer": None,
                 "currency": None},
        "scenarios": {"central": {k: dict(QUANTITY) for k in ("credit", "debit", "net")}},
        "known_costs": {}, "coverage": {}, "methodology": METHOD.hash,
    }  # fmt: skip
    return Planned(
        record_id, 1, None, "contribution", body, digest_of(body), DAY1, record_id,
        root_key_of(f"w{index}"), digest_of(["content", index]),
    )  # fmt: skip


# ---- plan selection --------------------------------------------------------------------------


def test_plan_selects_sent_records_by_day_roots_and_their_intersection() -> None:
    current = make_policy()
    by_root = {d: seed(d) for d in (DEST, OTHER)}
    one, two = ids_of(by_root[DEST], ROOT1), ids_of(by_root[DEST], ROOT2)
    both = ids_of(by_root[DEST], ROOT1, ROOT2)
    r1, r2, r3 = (frozenset({root_key_of(t.root)}) for t in (ROOT1, ROOT2, ROOT3))

    everything = plan_of(plan_tombstones(PROFILE, current, before_day=None, root_keys=None), DEST)
    assert everything.record_ids == both and everything.record_count == 2  # root-3 never sent

    def by_day(day: str) -> tuple[str, ...]:
        plans = plan_tombstones(PROFILE, current, before_day=day, root_keys=None)
        return plan_of(plans, DEST).record_ids

    assert by_day("2026-11-02") == one  # strictly before the day
    assert by_day("2026-11-04") == both
    assert by_day(DAY1) == ()

    def by_roots(roots: frozenset[str], day: str | None = None) -> tuple[str, ...]:
        plans = plan_tombstones(PROFILE, current, before_day=day, root_keys=roots)
        return plan_of(plans, DEST).record_ids

    assert by_roots(r2) == two
    assert by_roots(r1 | r2) == both
    assert by_roots(r1, "2026-11-04") == one  # intersection: root-1 AND before day
    assert by_roots(r2, "2026-11-02") == ()  # root-2 is dated after the cutoff
    assert by_roots(r3) == ()  # queued-never-sent: nothing to retract
    assert by_roots(frozenset({"f" * 16})) == ()


def test_each_destination_is_planned_from_its_own_registry() -> None:
    current = make_policy()
    by_root = {d: seed(d) for d in (DEST, OTHER)}
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert [p.destination_id for p in plans] == [DEST, OTHER]
    for destination in (DEST, OTHER):
        found = plan_of(plans, destination)
        assert found.record_ids == ids_of(by_root[destination], ROOT1, ROOT2)
    assert not set(plans[0].record_ids) & set(plans[1].record_ids)  # unrelated pseudonyms


def test_queued_only_registry_has_nothing_sent_to_retract() -> None:
    current = make_policy()
    planned = plan_for(DEST)
    mark_queued(PROFILE, DEST, planned, "b-unsent-0001")
    found = plan_of(plan_tombstones(PROFILE, current, before_day=None, root_keys=None), DEST)
    assert (found.permitted, found.reason, found.record_ids, found.record_count) == (
        True,
        "no_exported_records",
        (),
        0,
    )


def test_destination_without_any_registry_or_metrics_ack_gets_no_plan() -> None:
    current = make_policy()
    seed(DEST)  # OTHER is in the policy but never exported anything
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert [p.destination_id for p in plans] == [DEST]


# ---- permission ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("saved", "permitted", "reason"),
    [
        (lambda: make_policy(), True, None),
        (lambda: make_policy(enabled=()), False, "not_enabled"),
        (lambda: make_policy(enabled=(), logs=()), False, "signal_not_permitted"),
    ],
    ids=["enabled", "not_enabled", "logs_not_allowed"],
)
def test_permission_follows_current_policy_with_stable_codes(
    saved: Any, permitted: bool, reason: str | None
) -> None:
    seed(DEST)
    current = saved()
    found = plan_of(plan_tombstones(PROFILE, current, before_day=None, root_keys=None), DEST)
    assert (found.permitted, found.reason) == (permitted, reason)
    assert found.record_count == 2 and len(found.record_ids) == 2
    assert (found.obligations == ()) is permitted
    assert len(found.obligations) == (0 if permitted else 2)


def test_unreadable_policy_file_denies_and_unknown_destination_cannot_be_planned(
    tmp_path: Path,
) -> None:
    current = make_policy()
    seed(DEST)
    (data_root(tmp_path) / "accounting" / "policy.json").write_text("{not json")
    found = plan_of(plan_tombstones(PROFILE, current, before_day=None, root_keys=None), DEST)
    assert (found.permitted, found.reason) == (False, "policy_unreadable")
    assert len(found.obligations) == 2


def test_paused_profile_still_permits_spooling_but_not_unpaused_sending() -> None:
    make_policy()
    seed(DEST)
    policy.pause(PROFILE)
    current = policy.load_profile_policy(PROFILE)
    (found,) = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert found.permitted and found.reason is None
    (result,) = enqueue_tombstones(PROFILE, current, [found], now=NOW)
    assert result.batches == 1 and pending_tombstones(PROFILE, DEST) == 1


def test_non_reversible_metrics_only_when_a_metrics_ack_exists() -> None:
    current = make_policy()
    seed(DEST)
    seed(OTHER)
    assert [
        p.non_reversible_metrics
        for p in plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    ] == [False, False]
    acks.update_delivery(
        PROFILE,
        DEST,
        Signal.METRICS,
        lambda d: acks.ack_vacuous(d, 1, "a" * 16, "2026-11-01T00:00:00Z"),
    )
    acks.update_delivery(PROFILE, OTHER, Signal.LOGS, lambda d: acks.set_counts(d, 1, 0))
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert plan_of(plans, DEST).non_reversible_metrics is True
    assert plan_of(plans, OTHER).non_reversible_metrics is False  # a logs document is not metrics


def test_metrics_ack_alone_reports_the_limit_even_without_per_task_records() -> None:
    current = make_policy()
    acks.update_delivery(
        PROFILE,
        DEST,
        Signal.METRICS,
        lambda d: acks.ack_vacuous(d, 1, "a" * 16, "2026-11-01T00:00:00Z"),
    )
    (found,) = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert (found.destination_id, found.record_count, found.reason) == (
        DEST,
        0,
        "no_exported_records",
    )
    assert found.non_reversible_metrics is True


# ---- plan is read-only -----------------------------------------------------------------------


def test_plan_creates_and_changes_nothing(tmp_path: Path) -> None:
    current = make_policy()
    seed(DEST)
    before = tree(data_root(tmp_path))
    r1 = frozenset({root_key_of(ROOT1.root)})
    for day, roots in ((None, None), ("2026-11-09", r1)):
        plans = plan_tombstones(PROFILE, current, before_day=day, root_keys=roots)
        assert plan_of(plans, DEST).record_count >= 1  # positive twin: it really planned
    assert tree(data_root(tmp_path)) == before


# ---- enqueue ---------------------------------------------------------------------------------


def test_enqueue_spools_receiver_valid_tombstones_and_queues_deletion_revisions() -> None:
    current = make_policy()
    by_root = seed(DEST)
    sent = [by_root[root_key_of(t.root)] for t in (ROOT1, ROOT2)]
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    before = load_registry(PROFILE, DEST)

    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert (result.destination_id, result.queued_records, result.batches) == (DEST, 2, 1)
    assert result.skipped_reason is None and result.obligations == ()
    (batch_id,) = result.batch_ids
    assert T_ID.fullmatch(batch_id) and batch_id.startswith(f"t{STAMP:010d}-")

    (meta,) = spooled(DEST)
    assert (meta.batch_id, meta.kind, meta.state, meta.generation, meta.snapshot_id) == (
        batch_id,
        "tombstone",
        "queued",
        0,
        "",
    )
    assert meta.created_utc == "2026-11-05T12:00:00Z" and meta.items == 2
    assert meta.record_ids == tuple(sorted(p.record_id for p in sent))
    assert meta.scope["endpoint"] == ENDPOINT and meta.scope["logs_permitted"] is True

    after = load_registry(PROFILE, DEST)
    for item in sent:
        entry = after.entries[item.record_id]
        assert entry.acked == before.entries[item.record_id].acked  # the contribution stays acked
        (queued,) = entry.queued
        assert (queued.revision, queued.type, queued.batch) == (2, "deletion", batch_id)
        assert queued.revision == item.revision + 1
    unsent = by_root[root_key_of(ROOT3.root)]
    assert after.entries[unsent.record_id].queued[0].type == "contribution"  # untouched

    delivery = acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    assert delivery is not None and delivery.pending_batches == 1  # the one tombstone batch
    assert delivery.unknown_batches == 0
    assert pending_tombstones(PROFILE, DEST) == 1
    assert pending_tombstones(PROFILE, OTHER) == 0


def test_tombstone_bodies_delete_at_the_receiver_and_replays_are_refused() -> None:
    current = make_policy()
    by_root = seed(DEST)
    sent = [by_root[root_key_of(t.root)] for t in (ROOT1, ROOT2)]
    state, outcomes = receive(ReceiverState(), [p.body for p in sent])
    assert outcomes == [Outcome.APPLIED_NEW] * 2 and rows_total(state) == 2

    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    enqueue_tombstones(PROFILE, current, plans, now=NOW)
    bodies = tombstone_bodies(DEST)
    assert len(bodies) == 2
    state, outcomes = receive(state, bodies)
    assert outcomes == [Outcome.APPLIED_TOMBSTONE] * 2
    assert rows_total(state) == 0  # current_totals dropped them

    state, replays = receive(state, [p.body for p in sent])
    assert replays == [Outcome.REJECTED_TOMBSTONED] * 2
    assert rows_total(state) == 0
    state, again = receive(state, bodies)
    assert again == [Outcome.DUPLICATE_NOOP] * 2  # re-delivery of the tombstone is harmless


def test_tombstone_batches_contain_no_contribution_data() -> None:
    current = make_policy()
    by_root = seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    enqueue_tombstones(PROFILE, current, plans, now=NOW)
    bodies = tombstone_bodies(DEST)
    assert len(bodies) == 2
    originals = {p.record_id: p for p in by_root.values()}
    for body in bodies:
        assert (body["scenarios"], body["known_costs"], body["coverage"]) == ({}, {}, {})
        assert body["record_type"] == "deletion" and body["revision"] == 2
        assert body["supersedes_revision"] == 1
        original = originals[body["record_id"]]
        assert body["task_id"] == original.task and body["period"] == original.period


def test_second_enqueue_is_a_no_op() -> None:
    current = make_policy()
    seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (first,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    registry = load_registry(PROFILE, DEST)
    files = sorted(m.batch_id for m in spooled(DEST))

    later = datetime(2026, 11, 6, 9, 0, 0, tzinfo=UTC)
    (second,) = enqueue_tombstones(PROFILE, current, plans, now=later)
    assert (second.queued_records, second.batches, second.batch_ids) == (0, 0, ())
    assert second.skipped_reason is None
    assert sorted(m.batch_id for m in spooled(DEST)) == files == list(first.batch_ids)
    assert load_registry(PROFILE, DEST) == registry
    fresh = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (third,) = enqueue_tombstones(PROFILE, current, fresh, now=later)  # a fresh plan too
    assert third.batches == 0 and len(spooled(DEST)) == 1


def test_enqueue_signals_only_the_planned_selection() -> None:
    current = make_policy()
    by_root = seed(DEST)
    r1 = frozenset({root_key_of(ROOT1.root)})
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=r1)
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert result.queued_records == 1
    (body,) = tombstone_bodies(DEST)
    assert body["record_id"] == by_root[root_key_of(ROOT1.root)].record_id
    other = load_registry(PROFILE, DEST).entries[by_root[root_key_of(ROOT2.root)].record_id]
    assert other.queued == ()


def test_destinations_are_enqueued_independently() -> None:
    current = make_policy(enabled=(DEST,))  # OTHER logs permitted but not switched on
    for destination in (DEST, OTHER):
        seed(destination)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    results = {r.destination_id: r for r in enqueue_tombstones(PROFILE, current, plans, now=NOW)}
    assert results[DEST].batches == 1 and results[DEST].skipped_reason is None
    assert (results[OTHER].batches, results[OTHER].skipped_reason) == (0, "not_enabled")
    assert len(results[OTHER].obligations) == 2
    assert spooled(OTHER) == [] and pending_tombstones(PROFILE, OTHER) == 0
    assert all(
        not e.queued or e.queued[0].type == "contribution"
        for e in load_registry(PROFILE, OTHER).entries.values()
    )


# ---- revocation is never bypassed ------------------------------------------------------------


def test_not_permitted_enqueue_writes_nothing_and_reports_obligations(tmp_path: Path) -> None:
    current = make_policy(enabled=())
    by_root = seed(DEST)
    before = tree(data_root(tmp_path))
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert result.skipped_reason == "not_enabled"
    assert (result.queued_records, result.batches, result.batch_ids) == (0, 0, ())
    assert result.obligations == ids_of(by_root, ROOT1, ROOT2) == plans[0].obligations
    assert tree(data_root(tmp_path)) == before  # no spool, no registry, no delivery document


def test_obligations_are_bounded_to_a_thousand_ids() -> None:
    current = make_policy(enabled=())
    planned = [synthetic(i) for i in range(1500)]
    mark_queued(PROFILE, DEST, planned, "b-big-0001")
    mark_acked(PROFILE, DEST, "b-big-0001")
    (found,) = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    ordered = tuple(sorted(p.record_id for p in planned))
    assert found.record_count == 1500 and len(found.record_ids) == 1500
    assert MAX_OBLIGATIONS == 1000 and found.obligations == ordered[:1000]
    (result,) = enqueue_tombstones(PROFILE, current, [found], now=NOW)
    assert result.obligations == ordered[:1000] and result.batches == 0


def test_permitted_plan_batches_all_records_beyond_the_obligation_bound() -> None:
    current = make_policy()
    planned = [synthetic(i) for i in range(1500)]
    mark_queued(PROFILE, DEST, planned, "b-big-0001")
    mark_acked(PROFILE, DEST, "b-big-0001")
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert plans[0].obligations == ()
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert result.queued_records == 1500 and result.batches >= 1
    assert len({b["record_id"] for b in tombstone_bodies(DEST)}) == 1500
    assert all(T_ID.fullmatch(b) for b in result.batch_ids)


@pytest.mark.parametrize(
    ("revoke", "reason"),
    [
        ("disable", "not_enabled"),
        ("remove_destination", "unknown_destination"),
        ("drop_logs", "signal_not_permitted"),
    ],
)
def test_revoked_after_plan_writes_nothing(tmp_path: Path, revoke: str, reason: str) -> None:
    current = make_policy()
    seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    assert plans[0].permitted  # the plan itself was fine
    if revoke == "disable":
        policy.save_profile(replace(current, enabled_exports=()))
    elif revoke == "remove_destination":
        policy.save_profile(
            replace(
                current,
                destinations=tuple(d for d in current.destinations if d.destination_id != DEST),
                enabled_exports=(),
            )
        )
    else:
        narrowed = tuple(
            replace(d, signals=(Signal.METRICS,), logs_permitted=False)
            for d in current.destinations
        )
        policy.save_profile(replace(current, destinations=narrowed, enabled_exports=()))
    before = tree(data_root(tmp_path))
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert (result.batches, result.queued_records, result.skipped_reason) == (0, 0, reason)
    assert len(result.obligations) == 2
    assert tree(data_root(tmp_path)) == before


def test_unreadable_policy_at_enqueue_time_blocks_spooling(tmp_path: Path) -> None:
    current = make_policy()
    seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (data_root(tmp_path) / "accounting" / "policy.json").write_text("{broken")
    before = tree(data_root(tmp_path))
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert (result.skipped_reason, result.batches) == ("policy_unreadable", 0)
    assert tree(data_root(tmp_path)) == before


def test_missing_pseudonym_key_is_never_minted_by_enqueue(tmp_path: Path) -> None:
    current = make_policy()
    seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    key_file = data_root(tmp_path) / "accounting" / PROFILE / "export" / "pseudonym.key"
    key_file.unlink()
    before = tree(data_root(tmp_path))
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert (result.skipped_reason, result.batches) == ("no_pseudonym_key", 0)
    assert tree(data_root(tmp_path)) == before and not key_file.exists()


# ---- acknowledgement bookkeeping -------------------------------------------------------------


def test_forget_removes_exactly_the_acknowledged_deletions() -> None:
    current = make_policy()
    by_root = seed(DEST)
    unsent = by_root[root_key_of(ROOT3.root)]
    assert forget_acknowledged_deletions(PROFILE, DEST) == 0  # nothing deleted yet
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert forget_acknowledged_deletions(PROFILE, DEST) == 0  # queued is not accepted
    assert set(load_registry(PROFILE, DEST).entries) == {p.record_id for p in by_root.values()}

    mark_acked(PROFILE, DEST, result.batch_ids[0])
    acked = load_registry(PROFILE, DEST)
    assert all(
        (state := acked.entries[i].acked) is not None and state.type == "deletion"
        for i in ids_of(by_root, ROOT1, ROOT2)
    )
    assert forget_acknowledged_deletions(PROFILE, DEST) == 2
    left = load_registry(PROFILE, DEST)
    assert set(left.entries) == {unsent.record_id}  # unrelated records keep their linkage
    assert left.entries[unsent.record_id].queued[0].type == "contribution"
    assert forget_acknowledged_deletions(PROFILE, DEST) == 0  # idempotent


def test_forget_keeps_a_record_recreated_after_its_deletion() -> None:
    current = make_policy()
    by_root = seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    mark_acked(PROFILE, DEST, result.batch_ids[0])
    # changed content re-creates root-1's record above the deletion (adjustment at revision 3)
    more = Ledger(METHOD)
    more.ingest(
        EvidenceBundle(
            (
                candidate("a", task=ROOT1, day=DAY1),
                candidate("z", task=ROOT1, day=DAY1, immediate=(atom("s9", 200, 0, "e9"),)),
            ),
            (),
        )
    )
    candidates, _ = project_candidates(more.rows(), key=key(), destination=DEST)
    (recreated,) = plan_revisions(load_registry(PROFILE, DEST), candidates)
    assert (recreated.revision, recreated.record_type) == (3, "adjustment")
    mark_queued(PROFILE, DEST, [recreated], "b-new-0001")
    assert forget_acknowledged_deletions(PROFILE, DEST) == 1  # only root-2's
    kept = load_registry(PROFILE, DEST).entries
    assert by_root[root_key_of(ROOT1.root)].record_id in kept
    assert by_root[root_key_of(ROOT2.root)].record_id not in kept


def test_forget_never_creates_a_registry(tmp_path: Path) -> None:
    assert forget_acknowledged_deletions(PROFILE, DEST) == 0
    assert tree(data_root(tmp_path)) == {}


def test_pending_counts_only_live_tombstone_batches() -> None:
    from ssgrep.accounting.export_spool import update_meta

    current = make_policy()
    seed(DEST)
    assert pending_tombstones(PROFILE, DEST) == 0
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    (batch_id,) = result.batch_ids
    assert pending_tombstones(PROFILE, DEST) == 1
    update_meta(PROFILE, DEST, Signal.LOGS, batch_id, lambda m: replace(m, state="in_flight"))
    assert pending_tombstones(PROFILE, DEST) == 1
    update_meta(PROFILE, DEST, Signal.LOGS, batch_id, lambda m: replace(m, state="acked"))
    assert pending_tombstones(PROFILE, DEST) == 0


def test_delivery_counts_include_unresolved_attempts() -> None:
    from ssgrep.accounting.export_spool import update_meta

    current = make_policy()
    seed(DEST)
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (first,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    update_meta(PROFILE, DEST, Signal.LOGS, first.batch_ids[0], lambda m: replace(m, attempts=2))
    # a second selection later spools a second tombstone batch and refreshes the counts
    more = Ledger(METHOD)
    more.ingest(
        EvidenceBundle(
            (candidate("y", task=ROOT2, day=DAY3, immediate=(atom("s7", 40, 0, "e7"),)),), ()
        )
    )
    candidates, _ = project_candidates(more.rows(), key=key(), destination=DEST)
    (follow,) = plan_revisions(load_registry(PROFILE, DEST), candidates)
    mark_queued(PROFILE, DEST, [follow], "b-new-0001")
    mark_acked(PROFILE, DEST, "b-new-0001")
    again = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (second,) = enqueue_tombstones(PROFILE, current, again, now=datetime(2026, 11, 7, tzinfo=UTC))
    assert second.batches == 1
    delivery = acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    assert delivery is not None
    assert (delivery.pending_batches, delivery.unknown_batches) == (2, 1)


# ---- canary ----------------------------------------------------------------------------------


def test_no_root_task_or_path_string_reaches_a_tombstone_batch(tmp_path: Path) -> None:
    current = make_policy()
    rows = canary_rows()
    candidates, omitted = project_candidates(rows, key=key(), destination=DEST)
    assert omitted == 0 and CANARY.encode() in rows.canonical_bytes()
    planned = plan_revisions(Registry(), candidates)
    mark_queued(PROFILE, DEST, planned, "b-sent-0001")
    mark_acked(PROFILE, DEST, "b-sent-0001")
    plans = plan_tombstones(PROFILE, current, before_day=None, root_keys=None)
    (result,) = enqueue_tombstones(PROFILE, current, plans, now=NOW)
    assert result.queued_records == 1
    spool_dir = data_root(tmp_path) / "accounting" / PROFILE / "spool" / DEST / "logs"
    files = [p for p in spool_dir.iterdir() if p.name.startswith("t")]
    assert len(files) == 2  # body + meta
    hay = " ".join(p.read_text().lower() for p in files)
    for fragment in (CANARY, "secret-project", "/home/", "witness", "action:", "exposure"):
        assert fragment not in hay, fragment
    for token in ("origin-", "atom-", "cand-", "debit-", "turn-", "branch-"):
        assert token not in hay, token
    assert tombstone_bodies(DEST)[0]["scenarios"] == {}
