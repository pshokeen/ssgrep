"""Ledger-log records: read-only ledger access, positive projection, revisions, the registry.

Rows come from the real ``Ledger``/``VersionedLedger`` reducers (and, for ``ledger_rows``, from
real persisted collector state: ``collector_state.save_state`` + ``commit_checkpoint`` exactly as
a cycle writes them). Every projected body must pass the reference receiver (``receiver_contract``)
and the registry is real files under the isolated data dir.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_state, collector_store, export_ids
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.contracts import Basis, Quantity, TaskId, digest_of
from ssgrep.accounting.export_records import (
    Candidate,
    Planned,
    RecordsUnavailable,
    Registry,
    RegistryCorrupt,
    build_deletions,
    ledger_rows,
    load_registry,
    log_records,
    mark_acked,
    mark_partial,
    mark_quarantined,
    mark_queued,
    mark_unknown,
    plan_revisions,
    project_candidates,
    records_before,
    records_for_roots,
    reset_queued,
    root_key_of,
)
from ssgrep.accounting.export_records_model import MAX_REGISTRY_BYTES
from ssgrep.accounting.receiver_contract import (
    Key,
    Outcome,
    ReceiverState,
    apply,
    current_totals,
)
from ssgrep.accounting.replay import Ledger, LedgerRows
from ssgrep.accounting.revisions import seal_day
from ssgrep.accounting.units import UnitSpec
from ssgrep.accounting.valuation_inputs import (
    Debit,
    EvidenceBundle,
)
from tests.accounting.valuation_builders import CHARS, DAY, METHOD, TASK, atom, candidate, debit

PROFILE = "personal-local"
DEST = "acme-org"
OTHER = "other-org"
KEY = bytes(range(32))
UNIT_KEY = ("context_tokens", "chars_div4_v1", None, None)
SEAL = seal_day(DAY)
NANOS = 1_794_000_000_000_000_000


# ---- builders --------------------------------------------------------------------------------


def rows_of(cands: list[Any], debits: list[Any] | None = None) -> LedgerRows:
    ledger = Ledger(METHOD)
    ledger.ingest(EvidenceBundle(tuple(cands), tuple(debits or ())))
    return ledger.rows()


def rows_a() -> LedgerRows:
    """One task, one day: credit 100 tokens, debit 160 tokens -> central net -60."""
    return rows_of([candidate("a")], [debit("x", 640)])


def rows_b() -> LedgerRows:
    """Same leaf after more evidence: credit 150, debit 160 -> central net -10."""
    return rows_of(
        [candidate("a"), candidate("b", immediate=(atom("s2", 200, 0, "ev-2"),))], [debit("x", 640)]
    )


def project(rows: LedgerRows, destination: str = DEST, key: bytes = KEY) -> list[Candidate]:
    candidates, omitted = project_candidates(rows, key=key, destination=destination)
    assert omitted == 0
    return candidates


def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def registry_path(tmp_path: Path, destination: str = DEST) -> Path:
    return data_root(tmp_path) / "accounting" / PROFILE / "export" / destination / "records.json"


def tree(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def receive(
    state: ReceiverState, planned: list[Planned], tenant: str = "t"
) -> tuple[ReceiverState, list[Outcome]]:
    outcomes = []
    for item in planned:
        state, outcome = apply(state, Key(tenant, "dev", item.record_id), item.body)
        outcomes.append(outcome)
    return state, outcomes


def totals(state: ReceiverState) -> dict[str, Any]:
    return current_totals(state, methodology=METHOD.hash, unit_key=UNIT_KEY)


def central_net(state: ReceiverState) -> int | None:
    return totals(state)["scenarios"]["central"]["net"]["total"]


def ship(profile_dest: str, planned: list[Planned], batch: str) -> None:
    mark_queued(PROFILE, profile_dest, planned, batch)
    mark_acked(PROFILE, profile_dest, batch)


# ---- A. ledger access ------------------------------------------------------------------------


def test_ledger_rows_is_none_without_a_checkpoint_and_creates_nothing(tmp_path: Path) -> None:
    assert ledger_rows(PROFILE) is None
    assert tree(data_root(tmp_path)) == {}


def commit_real_state(ledger: VersionedLedger) -> None:
    """Persist ledger state + checkpoint through the collector's own writers."""
    blobs: dict[str, str] = {}
    state = collector_state.State(ledger=ledger.state(), superseded=dict(ledger.superseded))
    collector_state.save_state(PROFILE, state, METHOD.hash, blobs)
    checkpoint = collector_store.Checkpoint(PROFILE, generation=1, methodology=METHOD.hash)
    checkpoint.blobs = blobs
    collector_store.commit_checkpoint(PROFILE, checkpoint)


def versioned() -> VersionedLedger:
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("a"),), (debit("x", None),))}, today="2026-11-02"
    )
    # refinement after the seal: revision 2 (so a superseded revision exists on disk)
    ledger.reconcile({"root-1": EvidenceBundle((candidate("a"),), (debit("x", 640),))}, today=SEAL)
    return ledger


def test_ledger_rows_equals_the_rows_of_the_persisted_collector_ledger(tmp_path: Path) -> None:
    ledger = versioned()
    expected = ledger.rows()
    assert [p.revision for p in expected.periods] == [1, 2]
    commit_real_state(ledger)
    before = tree(data_root(tmp_path))

    rows = ledger_rows(PROFILE)

    assert rows == expected and rows is not None
    assert rows.canonical_bytes() == expected.canonical_bytes()
    assert tree(data_root(tmp_path)) == before  # read-only: nothing written, no lease taken
    candidates, omitted = project_candidates(rows, key=KEY, destination=DEST)
    assert omitted == 0 and len(candidates) == 1  # only the CURRENT period revision


def test_ledger_rows_is_none_when_the_checkpoint_has_no_ledger(tmp_path: Path) -> None:
    checkpoint = collector_store.Checkpoint(PROFILE, generation=1, methodology=METHOD.hash)
    blobs: dict[str, str] = {}
    collector_state.save_state(PROFILE, collector_state.State(), METHOD.hash, blobs)
    checkpoint.blobs = blobs
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    assert ledger_rows(PROFILE) is None


def test_corrupt_state_is_a_stable_condition_without_raw_text(tmp_path: Path) -> None:
    commit_real_state(versioned())
    state_dir = data_root(tmp_path) / "accounting" / PROFILE / "collector" / "state"
    (blob,) = state_dir.glob("ledger-*")
    blob.write_bytes(b"{ not json " + str(tmp_path).encode())
    with pytest.raises(RecordsUnavailable) as raised:
        ledger_rows(PROFILE)
    assert raised.value.condition == "state_corrupt"
    assert str(tmp_path) not in str(raised.value) and "json" not in str(raised.value).lower()


def test_unsafe_collector_storage_is_a_stable_condition(tmp_path: Path) -> None:
    profile = data_root(tmp_path) / "accounting" / PROFILE
    profile.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (profile / "collector").symlink_to(elsewhere)
    with pytest.raises(RecordsUnavailable) as raised:
        ledger_rows(PROFILE)
    assert raised.value.condition == "storage_unsafe"
    assert str(tmp_path) not in str(raised.value)


# ---- B. projection ---------------------------------------------------------------------------


def test_body_has_exactly_the_receiver_keys_and_ints_for_one_leaf() -> None:
    (cand,) = project(rows_a())
    core = cand.core
    assert set(core) == {
        "schema", "record_id", "task_id", "period", "unit", "scenarios", "known_costs",
        "coverage", "methodology",
    }  # fmt: skip
    assert core["period"] == DAY and core["methodology"] == METHOD.hash
    assert core["unit"] == {
        "unit": "context_tokens", "method": "chars_div4_v1", "tokenizer": None, "currency": None
    }  # fmt: skip
    assert set(core["scenarios"]) == {"conservative", "central", "optimistic"}
    central = core["scenarios"]["central"]
    assert central["credit"]["value"] == 100 and central["debit"]["value"] == 160
    assert central["net"] == {
        "value": -60,
        "unit": "context_tokens",
        "basis": "estimated",
        "method": f"estimate:{METHOD.hash}",
        "reason": "",
    }
    assert core["known_costs"] == {
        "debit_raw": 640, "debit_known": 1, "debit_missing_count": 0,
        "credit_raw_conservative": 400, "credit_raw_central": 400, "credit_raw_optimistic": 400,
    }  # fmt: skip
    assert core["coverage"] == {
        "recomputable": True, "debit_missing": 0,
        "known_atoms_conservative": 1, "known_atoms_central": 1, "known_atoms_optimistic": 1,
        "missing_conservative": 0, "missing_central": 0, "missing_optimistic": 0,
    }  # fmt: skip
    assert cand.content_digest == digest_of(core)
    assert (
        cand.root_key
        == hashlib.sha256(TASK.root.encode()).hexdigest()[:16]
        == root_key_of(TASK.root)
    )


def test_every_projected_body_is_applied_new_by_the_receiver() -> None:
    cands = project(rows_a())
    planned = plan_revisions(Registry(), cands)
    state, outcomes = receive(ReceiverState(), planned)
    assert outcomes == [Outcome.APPLIED_NEW] * len(planned) == [Outcome.APPLIED_NEW]
    assert central_net(state) == -60
    for item in planned:  # also a hand-checkable invariant of the wire body
        assert item.body["record_type"] == "contribution" and item.body["revision"] == 1
        assert item.body["supersedes_revision"] is None
        assert item.body["record_id"] == item.record_id == cands[0].record_id


def test_unavailable_net_stays_null_with_a_reason_and_negative_nets_survive() -> None:
    unknown_debit = rows_of([candidate("a")], [debit("x", None)])
    (cand,) = project(unknown_debit)
    net = cand.core["scenarios"]["central"]["net"]
    assert net["value"] is None and net["basis"] == "unavailable"
    assert net["reason"].startswith("incomplete:") and net["reason"] != ""
    assert cand.core["scenarios"]["central"]["credit"]["value"] == 100  # known part kept
    assert cand.core["scenarios"]["central"]["debit"]["value"] == 0
    assert cand.core["coverage"]["debit_missing"] == 1
    assert cand.core["known_costs"]["debit_missing_count"] == 1
    state, outcomes = receive(ReceiverState(), plan_revisions(Registry(), [cand]))
    assert outcomes == [Outcome.APPLIED_NEW]
    assert totals(state)["scenarios"]["central"]["net"] == {"total": None, "known": 0, "unknown": 1}

    (negative,) = project(rows_a())
    assert negative.core["scenarios"]["central"]["net"]["value"] == -60  # never clamped


def test_free_text_reasons_collapse_to_a_constant() -> None:
    free = Quantity(None, "context_tokens", Basis.UNAVAILABLE, reason="Secret Reason! /home/u")
    rows = rows_of([candidate("a")], [Debit("x", TASK, DAY, "exposure", CHARS, free, chars=None)])
    (cand,) = project(rows)
    net = cand.core["scenarios"]["central"]["net"]
    assert net["value"] is None and net["basis"] == "unavailable"
    assert net["reason"] == "unspecified"
    assert "secret" not in json.dumps(cand.core).lower()


def test_only_current_period_revisions_contribute() -> None:
    rows = versioned().rows()  # revision 1: unknown debit; revision 2 (after the seal): 640 chars
    assert [p.revision for p in rows.periods] == [1, 2]
    (cand,) = project(rows)
    central = cand.core["scenarios"]["central"]
    assert central["net"]["value"] == -60 and cand.core["coverage"]["debit_missing"] == 0
    rev1 = LedgerRows(rows.namespace, rows.periods[:1], ())
    (old,) = project(rev1)
    assert old.core["scenarios"]["central"]["net"]["value"] is None  # what rev 1 would have said
    assert old.record_id == cand.record_id and old.content_digest != cand.content_digest


def test_candidates_are_sorted_deterministic_and_order_independent() -> None:
    day2 = "2026-11-03"
    cands = [candidate("a"), candidate("c", day=day2, immediate=(atom("s3", 80, 0, "ev-3"),))]
    cands.append(
        candidate(
            "t2", task=TaskId("root-2", "turn-9", "main"), immediate=(atom("s4", 120, 0, "ev-4"),)
        )
    )
    first = project(rows_of(cands, [debit("x", 640)]))
    again = project(rows_of(list(reversed(cands)), [debit("x", 640)]))
    assert first == again and len(first) == 3
    assert [(c.period, c.record_id) for c in first] == sorted(
        (c.period, c.record_id) for c in first
    )
    assert len({c.record_id for c in first}) == 3 and len({c.task_id for c in first}) == 2


def test_destinations_get_unrelated_ids_and_keys_matter() -> None:
    rows = rows_a()
    (one,) = project(rows, DEST)
    (two,) = project(rows, OTHER)
    (rekeyed,) = project(rows, DEST, key=bytes(reversed(KEY)))
    assert one.record_id != two.record_id and one.task_id != two.task_id
    assert one.record_id != rekeyed.record_id and one.task_id != rekeyed.task_id
    assert one.content_digest != two.content_digest  # ids are part of the body


def test_pseudonyms_are_not_unkeyed_hashes_of_their_inputs() -> None:
    (cand,) = project(rows_a())
    parts = [TASK.root, TASK.turn or "", TASK.branch, DAY, *UNIT_KEY[:2], "", ""]
    for joiner in ("", "|", ":", "\0", "/"):
        for prefix in ("", DEST + joiner, "record" + joiner + DEST + joiner):
            plain = hashlib.sha256((prefix + joiner.join(parts)).encode()).hexdigest()
            assert cand.record_id != plain[:32] and cand.task_id != plain[:32]
    root_hash = hashlib.sha256(TASK.root.encode()).hexdigest()
    assert cand.task_id not in root_hash and cand.record_id not in root_hash
    assert cand.task_id == export_ids.pseudonym(
        KEY, "task", DEST, TASK.root, TASK.turn or "", TASK.branch
    )


def test_unresolved_task_fallback_roots_get_their_own_ids() -> None:
    resolved = TaskId("root-1", "turn-1", "main")
    unknown = TaskId("root-1", None, "main")
    cands = [
        candidate("a", task=resolved),
        candidate("u", task=unknown, immediate=(atom("s9", 80, 0, "ev-9"),)),
    ]
    projected = project(rows_of(cands))
    assert len(projected) == 2
    assert len({c.record_id for c in projected}) == 2 and len({c.task_id for c in projected}) == 2
    assert len({c.root_key for c in projected}) == 1


def test_leaves_with_hostile_labels_are_omitted_and_counted_not_quoted() -> None:
    rows = rows_a()
    (row,) = rows.periods
    (leaf,) = row.leaves
    variants = [
        replace(leaf, unit=UnitSpec("bad unit /home/u", leaf.unit.method)),
        replace(leaf, unit=UnitSpec(leaf.unit.unit, "m /home/u")),
        replace(leaf, unit=UnitSpec(leaf.unit.unit, leaf.unit.method, currency="usd /home")),
        replace(leaf, methodology="NOT-HEX /home/u"),
    ]
    for bad in variants:
        hostile = LedgerRows(rows.namespace, (replace(row, leaves=(bad,)),), ())
        candidates, omitted = project_candidates(hostile, key=KEY, destination=DEST)
        assert candidates == [] and omitted == 1
    mixed = LedgerRows(rows.namespace, (replace(row, leaves=(leaf, variants[0])),), ())
    candidates, omitted = project_candidates(mixed, key=KEY, destination=DEST)
    assert len(candidates) == 1 and omitted == 1


# ---- canary ----------------------------------------------------------------------------------

CANARY = "canary7f3a9"


def canary_rows() -> LedgerRows:
    root = f"/home/{CANARY}/secret-project-{CANARY}"
    task = TaskId(root, f"turn-{CANARY}", f"branch-{CANARY}")
    cand = candidate(
        f"cand-{CANARY}",
        task=task,
        immediate=(atom(f"atom-{CANARY}", 400, 0, f"origin-{CANARY}", digest=f"dg-{CANARY}"),),
        witness_locators=(f"witness:{CANARY}/file#L7",),
        action_locators=(f"action:{CANARY}",),
        exposure_ids=(f"exp-{CANARY}",),
        execution_ids=(f"exec-{CANARY}",),
    )
    free = Quantity(None, "context_tokens", Basis.UNAVAILABLE, reason=f"Reason {CANARY}!")
    hostile = Debit(f"debit-{CANARY}", task, DAY, "exposure", CHARS, free, chars=None)
    return rows_of([cand], [hostile])


def test_canaries_never_reach_bodies_ids_log_records_or_the_registry_file(tmp_path: Path) -> None:
    rows = canary_rows()
    # the canary really is in the ledger the projection reads from
    assert CANARY.encode() in rows.canonical_bytes()
    candidates = project(rows)
    planned = plan_revisions(Registry(), candidates)
    mark_queued(PROFILE, DEST, planned, "b-0001")
    records = log_records(planned, time_unix_nano=NANOS)
    deletions = build_deletions(
        load_registry(PROFILE, DEST), [p.record_id for p in planned], key=KEY, destination=DEST
    )
    hay = json.dumps([p.body for p in planned] + [d.body for d in deletions]).lower()
    hay += " ".join(r.body + repr(r.attributes) for r in records).lower()
    hay += repr(candidates).lower() + registry_path(tmp_path).read_text().lower()
    for fragment in (CANARY, "secret-project", "/home/", "witness", "action:", "exposure"):
        assert fragment not in hay, fragment
    for item in planned:
        assert CANARY not in item.record_id + item.task + item.root_key
    for token in ("origin-", "atom-", "cand-", "debit-", "turn-", "branch-"):
        assert token not in hay, token
    state, outcomes = receive(ReceiverState(), planned)
    assert outcomes == [Outcome.APPLIED_NEW]


# ---- revisions -------------------------------------------------------------------------------


def test_first_projection_is_revision_one_contribution() -> None:
    (item,) = plan_revisions(Registry(), project(rows_a()))
    assert (item.revision, item.supersedes, item.record_type) == (1, None, "contribution")
    assert item.body_digest == digest_of(item.body) and item.content_digest != item.body_digest
    assert item.period == DAY and item.root_key == root_key_of(TASK.root)


def test_same_content_again_plans_nothing_once_the_registry_knows_it(tmp_path: Path) -> None:
    cands = project(rows_a())
    first = plan_revisions(load_registry(PROFILE, DEST), cands)
    assert len(first) == 1
    mark_queued(PROFILE, DEST, first, "b-0001")
    assert plan_revisions(load_registry(PROFILE, DEST), cands) == []  # queued counts as known
    mark_acked(PROFILE, DEST, "b-0001")
    assert plan_revisions(load_registry(PROFILE, DEST), cands) == []  # and acked, too
    assert plan_revisions(load_registry(PROFILE, DEST), cands + cands) == []


def test_changed_content_is_revision_two_adjustment_and_never_double_counts() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    ship(DEST, first, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    assert [(p.revision, p.supersedes, p.record_type) for p in second] == [(2, 1, "adjustment")]
    assert second[0].record_id == first[0].record_id and second[0].task == first[0].task

    state, _ = receive(ReceiverState(), first)
    assert central_net(state) == -60
    state, outcomes = receive(state, second)
    assert outcomes == [Outcome.APPLIED_UPSERT]
    assert central_net(state) == -10 and totals(state)["rows"] == 1  # replaced, not added
    state, outcomes = receive(state, first + second)  # replays are duplicates / old revisions
    assert outcomes == [Outcome.REJECTED_OLD_REVISION, Outcome.DUPLICATE_NOOP]
    assert central_net(state) == -10


def test_reordered_delivery_converges_to_the_latest_revision() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    ship(DEST, first, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    state, outcomes = receive(ReceiverState(), second + first)
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.REJECTED_OLD_REVISION]
    assert central_net(state) == -10 and totals(state)["rows"] == 1


def test_superseded_period_revisions_never_export() -> None:
    rows = versioned().rows()
    (item,) = plan_revisions(Registry(), project(rows))
    assert item.body["scenarios"]["central"]["net"]["value"] == -60
    assert "unavailable" not in json.dumps(item.body)  # revision 1's unknown net is not exported


def test_planning_is_idempotent_and_independent_of_input_order() -> None:
    cands = project(
        rows_of(
            [
                candidate("a"),
                candidate("c", day="2026-11-03", immediate=(atom("s3", 80, 0, "e3"),)),
            ],
            [debit("x", 640)],
        )
    )
    shuffled = list(cands)
    random.Random(7).shuffle(shuffled)
    registry = Registry()
    assert plan_revisions(registry, cands) == plan_revisions(registry, shuffled)
    assert plan_revisions(registry, cands) == plan_revisions(registry, cands + cands)


def test_retracted_records_are_not_planned() -> None:
    cands = project(rows_a())
    assert plan_revisions(Registry(), cands, retracted=frozenset({cands[0].record_id})) == []


def test_reset_queued_then_replan_reuses_the_same_revision_number() -> None:
    cands = project(rows_a())
    first = plan_revisions(load_registry(PROFILE, DEST), cands)
    mark_queued(PROFILE, DEST, first, "b-0001")
    reset_queued(PROFILE, DEST, "b-0001")
    assert load_registry(PROFILE, DEST).entries == {}  # nothing ever acked: forgotten entirely
    assert plan_revisions(load_registry(PROFILE, DEST), cands) == first

    ship(DEST, first, "b-0002")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    mark_queued(PROFILE, DEST, second, "b-0003")
    reset_queued(PROFILE, DEST, "b-0003")
    entry = load_registry(PROFILE, DEST).entries[first[0].record_id]
    assert entry.acked is not None and entry.acked.revision == 1 and entry.queued == ()
    again = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    assert again == second and again[0].revision == 2


@pytest.mark.parametrize("settle", [mark_unknown, mark_partial, mark_quarantined])
def test_unresolved_batches_block_duplicate_additive_revisions(settle: Any) -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    ship(DEST, first, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    mark_queued(PROFILE, DEST, second, "b-0002")
    settle(PROFILE, DEST, "b-0002")
    entry = load_registry(PROFILE, DEST).entries[second[0].record_id]
    assert entry.unknown is not None and entry.unknown.revision == 2
    assert entry.acked is not None and entry.acked.revision == 1 and entry.queued == ()

    # same content: nothing (the receiver may already hold revision 2)
    assert plan_revisions(load_registry(PROFILE, DEST), project(rows_b())) == []
    # changed content: revision 3 supersedes 2 -- never a second revision 2
    more = rows_of(
        [candidate("a"), candidate("b", immediate=(atom("s2", 200, 0, "ev-2"),))],
        [debit("x", 1280)],
    )
    third = plan_revisions(load_registry(PROFILE, DEST), project(more))
    assert [(p.revision, p.supersedes, p.record_type) for p in third] == [(3, 2, "adjustment")]
    state, outcomes = receive(ReceiverState(), first + second + third)
    assert outcomes == [Outcome.APPLIED_NEW, Outcome.APPLIED_UPSERT, Outcome.APPLIED_UPSERT]
    assert totals(state)["rows"] == 1


def test_ack_of_revision_two_before_revision_one_keeps_revision_two() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    mark_queued(PROFILE, DEST, first, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    assert second[0].revision == 2
    mark_queued(PROFILE, DEST, second, "b-0002")
    mark_acked(PROFILE, DEST, "b-0002")
    mark_acked(PROFILE, DEST, "b-0001")  # the older batch is acknowledged LAST
    entry = load_registry(PROFILE, DEST).entries[first[0].record_id]
    assert entry.acked is not None and entry.acked.revision == 2
    assert entry.acked.digest == second[0].body_digest and entry.queued == ()
    assert entry.acked.batch is None


def test_older_unknown_never_replaces_a_newer_ack() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    mark_queued(PROFILE, DEST, first, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    mark_queued(PROFILE, DEST, second, "b-0002")
    mark_acked(PROFILE, DEST, "b-0002")
    mark_unknown(PROFILE, DEST, "b-0001")
    entry = load_registry(PROFILE, DEST).entries[first[0].record_id]
    assert entry.acked is not None and entry.acked.revision == 2 and entry.unknown is None


def test_ack_clears_an_older_unknown_state() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    mark_queued(PROFILE, DEST, first, "b-0001")
    mark_unknown(PROFILE, DEST, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    mark_queued(PROFILE, DEST, second, "b-0002")
    mark_acked(PROFILE, DEST, "b-0002")
    entry = load_registry(PROFILE, DEST).entries[first[0].record_id]
    assert entry.unknown is None and entry.acked is not None and entry.acked.revision == 2


def test_transitions_on_an_absent_registry_create_nothing(tmp_path: Path) -> None:
    for transition in (mark_acked, mark_unknown, mark_partial, mark_quarantined, reset_queued):
        transition(PROFILE, DEST, "b-0001")
    assert tree(data_root(tmp_path)) == {}


def test_load_registry_is_read_only_and_creates_nothing(tmp_path: Path) -> None:
    assert load_registry(PROFILE, DEST) == Registry()
    assert records_for_roots(PROFILE, DEST, ["0" * 16]) == []
    assert records_before(PROFILE, DEST, "2030-01-01") == []
    assert tree(data_root(tmp_path)) == {}
    ship(DEST, plan_revisions(Registry(), project(rows_a())), "b-0001")
    before = tree(data_root(tmp_path))
    load_registry(PROFILE, DEST)
    assert tree(data_root(tmp_path)) == before


# ---- deletions and purge lookups -------------------------------------------------------------


def test_deletion_bodies_come_from_the_registry_and_tombstone_the_receiver_row() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    ship(DEST, first, "b-0001")
    second = plan_revisions(load_registry(PROFILE, DEST), project(rows_b()))
    ship(DEST, second, "b-0002")
    registry = load_registry(PROFILE, DEST)

    (deletion,) = build_deletions(registry, [first[0].record_id], key=KEY, destination=DEST)
    body = deletion.body
    assert (deletion.revision, deletion.supersedes, deletion.record_type) == (3, 2, "deletion")
    assert body["scenarios"] == {} and body["known_costs"] == {} and body["coverage"] == {}
    assert body["task_id"] == first[0].task and body["period"] == DAY
    assert body["unit"] == first[0].body["unit"] and body["methodology"] == METHOD.hash
    assert set(body) == set(first[0].body)

    state, _ = receive(ReceiverState(), first + second)
    state, outcomes = receive(state, [deletion])
    assert outcomes == [Outcome.APPLIED_TOMBSTONE] and totals(state)["rows"] == 0

    assert build_deletions(
        registry, ["f" * 32, first[0].record_id, first[0].record_id], key=KEY, destination=DEST
    ) == [deletion]


def test_deleted_records_are_not_recreated_unless_content_changes() -> None:
    first = plan_revisions(Registry(), project(rows_a()))
    ship(DEST, first, "b-0001")
    (deletion,) = build_deletions(
        load_registry(PROFILE, DEST), [first[0].record_id], key=KEY, destination=DEST
    )
    ship(DEST, [deletion], "b-0002")
    registry = load_registry(PROFILE, DEST)
    acked = registry.entries[first[0].record_id].acked
    assert acked is not None and acked.type == "deletion" and acked.revision == 2
    assert build_deletions(registry, [first[0].record_id], key=KEY, destination=DEST) == []
    assert plan_revisions(registry, project(rows_a())) == []  # identical content stays deleted

    (recreated,) = plan_revisions(registry, project(rows_b()))
    assert (recreated.revision, recreated.supersedes, recreated.record_type) == (3, 2, "adjustment")
    state, _ = receive(ReceiverState(), first + [deletion])
    state, outcomes = receive(state, [recreated])
    assert outcomes == [Outcome.APPLIED_NEW] and central_net(state) == -10


def test_purge_lookups_find_records_by_root_key_and_day() -> None:
    other_task = TaskId("root-2", "turn-1", "main")
    cands = project(
        rows_of(
            [
                candidate("a"),
                candidate(
                    "o", task=other_task, day="2026-11-03", immediate=(atom("s5", 80, 0, "ev-5"),)
                ),
            ]
        )
    )
    planned = plan_revisions(Registry(), cands)
    ship(DEST, planned, "b-0001")
    by_root = {p.root_key: p.record_id for p in planned}
    assert set(by_root) == {root_key_of("root-1"), root_key_of("root-2")}
    assert records_for_roots(PROFILE, DEST, [root_key_of("root-1")]) == [
        by_root[root_key_of("root-1")]
    ]
    assert records_for_roots(PROFILE, DEST, ["f" * 16]) == []
    assert records_before(PROFILE, DEST, "2026-11-02") == [by_root[root_key_of("root-1")]]
    assert records_before(PROFILE, DEST, "2026-11-04") == sorted(by_root.values())
    assert records_before(PROFILE, DEST, "2026-11-04", [root_key_of("root-2")]) == [
        by_root[root_key_of("root-2")]
    ]
    assert records_before(PROFILE, DEST, DAY) == []  # strictly before the day


def test_destinations_keep_independent_registries(tmp_path: Path) -> None:
    one = plan_revisions(Registry(), project(rows_a(), DEST))
    two = plan_revisions(Registry(), project(rows_a(), OTHER))
    ship(DEST, one, "b-0001")
    assert load_registry(PROFILE, OTHER) == Registry()
    ship(OTHER, two, "b-0001")
    assert set(load_registry(PROFILE, DEST).entries) == {one[0].record_id}
    assert set(load_registry(PROFILE, OTHER).entries) == {two[0].record_id}
    assert registry_path(tmp_path).exists() and registry_path(tmp_path, OTHER).exists()


# ---- log records -----------------------------------------------------------------------------


def test_log_records_are_canonical_json_with_exactly_four_attributes() -> None:
    planned = plan_revisions(Registry(), project(rows_a()))
    (record,) = log_records(planned, time_unix_nano=NANOS)
    assert record.time_unix_nano == NANOS
    assert record.body == json.dumps(planned[0].body, sort_keys=True, separators=(",", ":"))
    assert record.body.isascii() and ", " not in record.body and ": " not in record.body
    parsed = json.loads(record.body, parse_float=lambda text: pytest.fail(f"float {text} in body"))
    assert parsed == planned[0].body
    assert dict(record.attributes) == {
        "ssgrep.record.id": planned[0].record_id,
        "ssgrep.record.revision": 1,
        "ssgrep.record.type": "contribution",
        "ssgrep.record.digest": planned[0].body_digest,
    }
    assert (
        len(record.attributes) == 4
        and type(dict(record.attributes)["ssgrep.record.revision"]) is int
    )
    assert digest_of(parsed) == dict(record.attributes)["ssgrep.record.digest"]


# ---- registry file: strict, bounded, never rewritten when damaged ----------------------------


def damaged_registry_cases(tmp_path: Path) -> dict[str, bytes]:
    valid = json.dumps({"schema": 1, "records": {}}).encode()
    entry = {
        "task": "a" * 32, "root_key": "b" * 16, "period": DAY, "methodology": "c" * 32,
        "unit": {"unit": "context_tokens", "method": "chars_div4_v1", "tokenizer": None,
                 "currency": None},
        "acked": {"revision": 1, "digest": "d" * 64, "content": "e" * 64, "type": "contribution"},
        "unknown": None, "queued": [],
    }  # fmt: skip

    def doc(**changes: Any) -> bytes:
        record = {**entry, **changes}
        return json.dumps({"schema": 1, "records": {"f" * 32: record}}).encode()

    assert valid  # a well-formed empty registry is NOT among the cases
    return {
        "garbage": b"\x00\xff not json",
        "float": b'{"schema":1,"records":{},"x":1.5}',
        "float_revision": doc(
            acked={"revision": 1.0, "digest": "d" * 64, "content": "e" * 64, "type": "contribution"}
        ),
        "schema": b'{"schema":2,"records":{}}',
        "extra_top": b'{"schema":1,"records":{},"extra":1}',
        "bad_record_id": json.dumps({"schema": 1, "records": {"XYZ": entry}}).encode(),
        "bad_task": doc(task="not-hex"),
        "bad_type": doc(
            acked={"revision": 1, "digest": "d" * 64, "content": "e" * 64, "type": "x"}
        ),
        "bad_period": doc(period="2026-13-45"),
        "extra_entry_key": doc(path="/home/user/project"),
        "negative_revision": doc(
            acked={"revision": 0, "digest": "d" * 64, "content": "e" * 64, "type": "contribution"}
        ),
        "oversize": b" " * (MAX_REGISTRY_BYTES + 1),
        "nan": b'{"schema":1,"records":{},"x":NaN}',
    }


def test_damaged_registries_raise_and_are_never_rewritten(tmp_path: Path) -> None:
    ship(DEST, plan_revisions(Registry(), project(rows_a())), "b-0001")
    path = registry_path(tmp_path)
    planned = plan_revisions(Registry(), project(rows_b()))
    for name, content in damaged_registry_cases(tmp_path).items():
        path.write_bytes(content)
        before = tree(data_root(tmp_path))
        with pytest.raises(RegistryCorrupt):
            load_registry(PROFILE, DEST)
        for write in (
            lambda: mark_queued(PROFILE, DEST, planned, "b-0009"),
            lambda: mark_acked(PROFILE, DEST, "b-0001"),
            lambda: reset_queued(PROFILE, DEST, "b-0001"),
        ):
            with pytest.raises(RegistryCorrupt):
                write()
        assert tree(data_root(tmp_path)) == before, name


def test_symlinked_registry_is_unsafe_and_untouched(tmp_path: Path) -> None:
    ship(DEST, plan_revisions(Registry(), project(rows_a())), "b-0001")
    path = registry_path(tmp_path)
    target = tmp_path / "elsewhere.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(RegistryCorrupt) as raised:
        load_registry(PROFILE, DEST)
    assert raised.value.condition == "registry_unsafe" and str(tmp_path) not in str(raised.value)
    with pytest.raises(RegistryCorrupt):
        mark_queued(PROFILE, DEST, plan_revisions(Registry(), project(rows_b())), "b-0002")
    assert path.is_symlink() and target.read_bytes() != b""


def test_registry_with_too_many_records_is_refused_on_read_and_write(tmp_path: Path) -> None:
    from ssgrep.accounting import export_registry
    from ssgrep.accounting.export_records_model import MAX_RECORDS, Entry

    entry = Entry("a" * 32, "b" * 16, DAY, UNIT_KEY, "c" * 32)
    too_many = Registry({f"{i:032x}": entry for i in range(MAX_RECORDS + 1)})
    with pytest.raises(RegistryCorrupt) as raised:
        export_registry.encode(too_many)
    assert raised.value.condition == "registry_oversize"
    document = (
        b'{"schema":1,"records":{'
        + b",".join(b'"%032x":{}' % i for i in range(MAX_RECORDS + 1))
        + b"}}"
    )
    with pytest.raises(RegistryCorrupt) as again:
        export_registry.decode(document)
    assert again.value.condition == "registry_oversize"


def test_registry_file_is_private_float_free_and_round_trips(tmp_path: Path) -> None:
    planned = plan_revisions(Registry(), project(rows_a()))
    mark_queued(PROFILE, DEST, planned, "b-0001")
    path = registry_path(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    raw = json.loads(path.read_text(), parse_float=lambda t: pytest.fail("float"))
    entry = raw["records"][planned[0].record_id]
    assert entry["queued"] == [
        {
            "revision": 1,
            "digest": planned[0].body_digest,
            "content": planned[0].content_digest,
            "type": "contribution",
            "batch": "b-0001",
        }
    ]
    assert entry["acked"] is None and entry["unknown"] is None
    assert entry["task"] == planned[0].task and entry["root_key"] == planned[0].root_key
    assert entry["unit"]["unit"] == "context_tokens" and entry["methodology"] == METHOD.hash
    before = path.read_bytes()
    mark_queued(PROFILE, DEST, planned, "b-0001")  # idempotent re-queue: byte-identical file
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        mark_queued(PROFILE, DEST, planned, "../escape")


# ---- concurrency -----------------------------------------------------------------------------


QUANTITY = {"value": 1, "unit": "context_tokens", "basis": "estimated", "method": "", "reason": ""}


def synthetic(worker: int, index: int) -> Planned:
    """A receiver-valid revision-1 body that needs no ledger (unique per worker/index)."""
    record_id = export_ids.pseudonym(KEY, "record", DEST, f"w{worker}", f"i{index}")
    body: dict[str, Any] = {
        "schema": 1, "record_id": record_id, "revision": 1, "supersedes_revision": None,
        "record_type": "contribution", "task_id": record_id, "period": DAY,
        "unit": {"unit": "context_tokens", "method": "chars_div4_v1", "tokenizer": None,
                 "currency": None},
        "scenarios": {"central": {k: dict(QUANTITY) for k in ("credit", "debit", "net")}},
        "known_costs": {}, "coverage": {}, "methodology": METHOD.hash,
    }  # fmt: skip
    return Planned(
        record_id, 1, None, "contribution", body, digest_of(body), DAY, record_id,
        root_key_of(f"w{worker}"), digest_of(["content", worker, index]),
    )  # fmt: skip


def hammer(worker: int, count: int) -> None:
    for index in range(count):
        batch = f"w{worker}-{index:04d}"
        mark_queued(PROFILE, DEST, [synthetic(worker, index)], batch)
        if index % 3 == 0:
            mark_unknown(PROFILE, DEST, batch)
        elif index % 3 == 1:
            mark_acked(PROFILE, DEST, batch)
        else:
            reset_queued(PROFILE, DEST, batch)


def test_two_processes_updating_concurrently_stay_consistent(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=hammer, args=(w, 25)) for w in (1, 2)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(120)
        assert process.exitcode == 0
    entries = load_registry(PROFILE, DEST).entries
    expected = {
        synthetic(w, i).record_id: ("unknown" if i % 3 == 0 else "acked")
        for w in (1, 2)
        for i in range(25)
        if i % 3 != 2
    }
    assert set(entries) == set(expected)  # no lost update, no resurrected discarded batch
    for record_id, kind in expected.items():
        entry = entries[record_id]
        assert entry.queued == () and getattr(entry, kind) is not None
        assert getattr(entry, "unknown" if kind == "acked" else "acked") is None
    assert not [p for p in registry_path(tmp_path).parent.iterdir() if p.name.endswith(".tmp")]
    assert registry_path(tmp_path).stat().st_mode & 0o777 == 0o600
