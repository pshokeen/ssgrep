"""VersionedLedger: one active version per evidence identity, sealed periods append-only."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest

from ssgrep.accounting import collector_state, collector_store as store
from ssgrep.accounting.collector_ledger import ReconcileResult, VersionedLedger
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import Scenario, TaskId, digest_of
from ssgrep.accounting.ledger_codec import ledger_state_from_dict, ledger_state_to_dict
from ssgrep.accounting.replay import Ledger, full_replay
from ssgrep.accounting.revisions import PeriodRow, current_rows, current_total, history, seal_day
from ssgrep.accounting.valuation_inputs import Candidate, Debit, Eligibility, EvidenceBundle
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_replay import corpus
from tests.accounting.valuation_builders import CHARS, DAY, METHOD, atom, candidate, debit

NS = METHOD.hash
SEAL = seal_day(DAY)  # first UTC day on which acceptance of a DAY period counts as late
LAST_PROVISIONAL = "2026-11-08"


def bundle(cands: Sequence[Candidate], debits: Sequence[Debit]) -> EvidenceBundle:
    return EvidenceBundle(tuple(cands), tuple(debits))


def central(row: PeriodRow, part: str = "net") -> int | None:
    for _, estimates in row.estimates:
        for estimate in estimates:
            if estimate.scenario is Scenario.CENTRAL:
                return getattr(estimate, part).value
    raise AssertionError("no central estimate")


def cand_b() -> Candidate:
    return candidate("b", immediate=(atom("s2", 200, 0, "ev-2"),))  # 50 tokens of credit


# ---- (a) refinement inside the provisional window ---------------------------------


def test_naive_ingest_of_both_debit_versions_is_the_conflict_this_module_prevents():
    naive = Ledger(METHOD)
    naive.ingest(bundle([candidate("a")], [debit("x", None)]))
    naive.ingest(bundle([], [debit("x", 640)]))
    assert b"conflicting_duplicate_debit" in naive.rows().canonical_bytes()


def test_debit_refined_inside_provisional_window_replaces_in_place_without_a_revision():
    ledger = VersionedLedger(METHOD)
    first = ledger.reconcile(
        {"root-1": bundle([candidate("a")], [debit("x", None)])}, today="2026-11-02"
    )
    assert first == ReconcileResult(2, 0, 0, 0, 0, ())
    before = ledger.rows()
    assert central(before.periods[0], "debit") == 0 and central(before.periods[0]) is None

    second = ledger.reconcile(
        {"root-1": bundle([candidate("a")], [debit("x", 640)])}, today="2026-11-03"
    )

    assert second == ReconcileResult(0, 1, 0, 1, 0, ())
    state = ledger.state()
    assert [(d.debit_id, d.chars, d.accepted_day) for d in state.debits] == [
        ("x", 640, "2026-11-02")
    ]
    assert ledger.superseded == {}
    rows = ledger.rows()
    assert b"conflicting_duplicate_debit" not in rows.canonical_bytes()
    assert [p.revision for p in rows.periods] == [1]
    leaf = rows.periods[0].leaves[0]
    assert (leaf.debit_raw, leaf.debit_known, leaf.debit_missing) == (640, 1, ())
    assert central(rows.periods[0]) == 100 - 160


def test_reconciling_the_same_projection_twice_the_same_day_changes_nothing():
    ledger = VersionedLedger(METHOD)
    desired = {"root-1": bundle([candidate("a"), cand_b()], [debit("x", 640)])}
    ledger.reconcile(desired, today="2026-11-02")
    snapshot = ledger.rows().canonical_bytes(), ledger.state(), dict(ledger.superseded)
    again = ledger.reconcile(desired, today="2026-11-02")
    assert again == ReconcileResult(0, 0, 0, 3, 0, ())
    assert (ledger.rows().canonical_bytes(), ledger.state(), ledger.superseded) == snapshot


# ---- (b) refinement after the seal -------------------------------------------------


def test_refinement_after_the_seal_adds_revision_two_and_keeps_revision_one_intact():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", None)])}, today="2026-11-02")
    (old_digest,) = ledger._debits
    before = ledger.rows()

    result = ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", 640)])}, today=SEAL)

    assert result == ReconcileResult(0, 1, 0, 1, 1, ())
    assert ledger.superseded == {old_digest: SEAL}
    assert len(ledger._debits) == 2  # old version retained for history
    rows = ledger.rows()
    assert rows.periods[0] == before.periods[0]  # revision 1 byte-for-byte unchanged
    rev1, rev2 = rows.periods
    assert (rev2.revision, rev2.supersedes, rev2.kind, rev2.late) == (2, 1, "adjustment", True)
    assert central(rev1) is None and central(rev2) == -60
    shown = history(rows.periods, NS, CHARS, today=SEAL)
    assert [(h.revision, h.current) for h in shown] == [(1, False), (2, True)]
    total, _ = current_total(rows.periods, NS, CHARS)
    assert total is not None and total.value == -60  # only the current revision counts


@pytest.mark.parametrize(("today", "revisions"), [(LAST_PROVISIONAL, 1), (SEAL, 2)])
def test_the_seal_boundary_decides_between_rewrite_and_revision(today, revisions):
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", None)])}, today="2026-11-02")
    ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", 640)])}, today=today)
    assert len(ledger.rows().periods) == revisions


# ---- (c) retraction -----------------------------------------------------------------


def test_sealed_retraction_supersedes_without_touching_revision_one():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": bundle([candidate("a"), cand_b()], [debit("x", 640)])}, today="2026-11-02"
    )
    before = ledger.rows()
    assert central(before.periods[0], "credit") == 150

    result = ledger.reconcile({"root-1": bundle([candidate("a")], [debit("x", 640)])}, today=SEAL)

    assert result == ReconcileResult(0, 0, 1, 2, 0, ())
    assert list(ledger.superseded.values()) == [SEAL]
    rows = ledger.rows()
    assert rows.periods[0] == before.periods[0]
    assert [p.revision for p in rows.periods] == [1, 2]
    assert central(rows.periods[1], "credit") == 100
    assert central(rows.periods[1]) == 100 - 160
    # the retracted candidate stays in the ledger as history
    assert len(ledger.state().candidates) == 2


def test_provisional_retraction_simply_removes_the_item():
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": bundle([candidate("a"), cand_b()], [debit("x", 640)])}, today="2026-11-02"
    )
    result = ledger.reconcile(
        {"root-1": bundle([candidate("a")], [debit("x", 640)])}, today="2026-11-03"
    )
    assert result == ReconcileResult(0, 0, 1, 2, 0, ())
    assert [c.candidate_id for c in ledger.state().candidates] == ["a"]
    assert ledger.superseded == {}
    rows = ledger.rows()
    assert [p.revision for p in rows.periods] == [1]
    assert central(rows.periods[0], "credit") == 100


def test_roots_not_mentioned_in_desired_are_left_untouched():
    ledger = VersionedLedger(METHOD)
    other = TaskId("root-2", "turn-1", "main")
    ledger.reconcile(
        {
            "root-1": bundle([candidate("a")], []),
            "root-2": bundle([candidate("z", task=other)], []),
        },
        today="2026-11-02",
    )
    result = ledger.reconcile({"root-1": bundle([candidate("a")], [])}, today="2026-11-03")
    assert result == ReconcileResult(0, 0, 0, 1, 0, ())
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}


def test_conflicting_variants_of_one_identity_in_a_projection_are_rejected_deterministically():
    ledger = VersionedLedger(METHOD)
    pair = [debit("x", 640), debit("x", 800)]
    result = ledger.reconcile({"root-1": bundle([], pair)}, today="2026-11-02")
    other = VersionedLedger(METHOD)
    other.reconcile({"root-1": bundle([], pair[::-1])}, today="2026-11-02")
    assert result.rejected == (("x", "conflicting_desired_variant"),)
    assert ledger.state() == other.state() and len(ledger.state().debits) == 1


@pytest.mark.parametrize("second_supported", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_lookups_keep_supported_reuse_and_every_debit_without_double_credit(
    second_supported: bool, reverse: bool
) -> None:
    first = candidate("same-fingerprint", exposure_ids=("lookup-1",))
    second = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2",),
        novel=second_supported,
        eligibility=Eligibility.SUPPORTED if second_supported else Eligibility.KNOWN_REJECTED,
        eligibility_reasons=() if second_supported else ("already_visible",),
        immediate=first.immediate if second_supported else (),
    )
    candidates = (second, first) if reverse else (first, second)
    debits = (
        debit("debit-1", 640, exposure_id="lookup-1"),
        debit("debit-2", 640, exposure_id="lookup-2"),
    )
    ledger = VersionedLedger(METHOD)
    result = ledger.reconcile({"root-1": bundle(candidates, debits)}, today=DAY)
    assert result.rejected == ()
    assert {row.exposure_ids for row in ledger.state().candidates} == {("lookup-1",), ("lookup-2",)}
    assert len(ledger.state().debits) == 2
    rows = ledger.rows()
    assert central(rows.periods[0], "credit") == 100  # same 400-char source union, once
    assert central(rows.periods[0], "debit") == 320  # two distinct 640-char exposures
    assert central(rows.periods[0]) == -220
    assert (
        rows.canonical_bytes() == full_replay(bundle(candidates, debits), METHOD).canonical_bytes()
    )
    restored = VersionedLedger.restore_versioned(ledger.state(), ledger.superseded_state())
    assert restored.rows().canonical_bytes() == rows.canonical_bytes()


def test_conflicting_candidate_variants_of_the_same_lookup_remain_rejected() -> None:
    first = candidate("same-fingerprint", exposure_ids=("lookup-1", "lookup-2"))
    variant = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2", "lookup-1"),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
    )
    ledger = VersionedLedger(METHOD)
    result = ledger.reconcile({"root-1": bundle((first, variant), ())}, today=DAY)
    assert result.rejected == (("same-fingerprint", "conflicting_desired_variant"),)
    assert len(ledger.state().candidates) == 1
    reversed_ledger = VersionedLedger(METHOD)
    reversed_ledger.reconcile({"root-1": bundle((variant, first), ())}, today=DAY)
    assert reversed_ledger.state() == ledger.state()


def test_refining_one_lookup_preserves_other_observations_of_the_same_fingerprint() -> None:
    first = candidate("same-fingerprint", exposure_ids=("lookup-1",))
    second = candidate("same-fingerprint", exposure_ids=("lookup-2",))
    ledger = VersionedLedger(METHOD)
    ledger.reconcile({"root-1": bundle((first, second), ())}, today=DAY)
    held_first = next(row for row in ledger.state().candidates if row.exposure_ids == ("lookup-1",))
    rejected_second = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2",),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
        immediate=(),
    )
    result = ledger.reconcile({"root-1": bundle((first, rejected_second), ())}, today=DAY)
    assert (result.replaced, result.unchanged, result.retracted, result.rejected) == (1, 1, 0, ())
    current = {row.exposure_ids: row for row in ledger.state().candidates}
    assert current[("lookup-1",)] == held_first
    assert current[("lookup-2",)].eligibility is Eligibility.KNOWN_REJECTED
    assert central(ledger.rows().periods[0], "credit") == 100


def test_restored_collapsed_projection_recovers_first_valid_credit_without_recounting_costs():
    first = candidate("same-fingerprint", exposure_ids=("lookup-1",))
    later_known = candidate(
        "same-fingerprint",
        exposure_ids=("lookup-2",),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
        immediate=(),
    )
    costs = (
        debit("debit-1", 640, exposure_id="lookup-1"),
        debit("debit-2", 640, exposure_id="lookup-2"),
    )
    # A legitimate old digest-selected projection could retain only the later
    # known-rejected observation despite accepting both distinct lookup costs.
    old = VersionedLedger(METHOD)
    old.reconcile({"root-1": bundle((later_known,), costs)}, today=DAY)
    assert central(old.rows().periods[0], "credit") == 0
    assert central(old.rows().periods[0], "debit") == 320
    state = ledger_state_from_dict(json.loads(json.dumps(ledger_state_to_dict(old.state()))))
    restored = VersionedLedger.restore_versioned(state, old.superseded_state())
    result = restored.reconcile({"root-1": bundle((first, later_known), costs)}, today=DAY)
    assert (result.added, result.unchanged, result.rejected) == (1, 3, ())
    assert central(restored.rows().periods[0], "credit") == 100
    assert central(restored.rows().periods[0], "debit") == 320
    assert (
        restored.rows().canonical_bytes()
        == full_replay(bundle((first, later_known), costs), METHOD).canonical_bytes()
    )


# ---- (e) frozen roots ----------------------------------------------------------------


def test_frozen_root_items_are_rejected_while_other_roots_proceed():
    ledger = VersionedLedger(METHOD)
    other = TaskId("root-2", "turn-1", "main")
    ledger.reconcile(
        {
            "root-1": bundle([candidate("a")], [debit("x", 640)]),
            "root-2": bundle([candidate("z", task=other)], []),
        },
        today="2026-11-02",
    )
    assert ledger.expire_roots(["root-1"]) == ["root-1"]
    frozen_rows = [p for p in ledger.rows().periods if p.period == DAY]

    result = ledger.reconcile(
        {
            "root-1": bundle([candidate("a"), cand_b()], [debit("x", 800)]),
            "root-2": bundle([candidate("z", task=other), candidate("y", task=other)], []),
        },
        today="2026-11-03",
    )

    assert result.rejected == (
        ("a", "evidence_expired_root"),
        ("b", "evidence_expired_root"),
        ("x", "evidence_expired_root"),
    )
    assert (result.added, result.unchanged) == (1, 1)
    state = ledger.state()
    assert [f.root for f in state.frozen] == ["root-1"]
    assert sorted(c.candidate_id for c in state.candidates) == ["y", "z"]
    assert state.debits == ()
    assert [p for p in ledger.rows().periods if p.period == DAY][0].leaves != ()
    assert frozen_rows  # root-1 totals survived the expiry


# ---- (f) equivalence with Ledger, pruning, restore ---------------------------------------


def test_with_no_supersession_rows_are_byte_identical_to_the_base_ledger():
    versioned = VersionedLedger(METHOD)
    versioned.ingest(corpus())
    assert versioned.superseded == {}
    reference = full_replay(corpus(), METHOD)
    assert versioned.rows().canonical_bytes() == reference.canonical_bytes()
    assert versioned.rows().digest == reference.digest


def _sealed_two_root_ledger() -> VersionedLedger:
    ledger = VersionedLedger(METHOD)
    other = TaskId("root-2", "turn-1", "main")
    first = {
        "root-1": bundle([candidate("a")], [debit("x", None)]),
        "root-2": bundle([candidate("z", task=other)], [debit("w", None, task=other)]),
    }
    second = {
        "root-1": bundle([candidate("a")], [debit("x", 640)]),
        "root-2": bundle([candidate("z", task=other)], [debit("w", 80, task=other)]),
    }
    ledger.reconcile(first, today="2026-11-02")
    ledger.reconcile(second, today=SEAL)
    return ledger


def test_expire_roots_and_purge_prune_superseded_entries_of_removed_items():
    ledger = _sealed_two_root_ledger()
    assert len(ledger.superseded) == 2
    ledger.expire_roots(["root-1"])
    assert len(ledger.superseded) == 1
    kept = next(iter(ledger.superseded))
    assert kept in {digest_of(d) for d in ledger.state().debits}
    ledger.purge(roots=["root-2"])
    assert ledger.superseded == {}
    assert ledger.state().debits == ()


def test_restore_versioned_round_trips_through_json_and_drops_unknown_digests():
    ledger = _sealed_two_root_ledger()
    state = json.loads(json.dumps(ledger_state_to_dict(ledger.state())))
    persisted = {**ledger.superseded_state(), "f" * 64: "2026-11-30"}
    restored = VersionedLedger.restore_versioned(ledger_state_from_dict(state), persisted)
    assert restored.superseded == ledger.superseded
    assert restored.superseded_state() == dict(sorted(ledger.superseded.items()))
    assert restored.rows().canonical_bytes() == ledger.rows().canonical_bytes()
    plain = Ledger.restore(ledger.state())  # forgetting supersession double counts: it must differ
    assert plain.rows().canonical_bytes() != ledger.rows().canonical_bytes()


# ---- (d)/(g) simulated collector runs -----------------------------------------------------

STEPS = ["2026-11-02", "2026-11-04", "2026-11-07", "2026-11-09", "2026-11-10", "2026-11-11"]
EVENT_DAYS = ["2026-11-01", "2026-11-02", "2026-11-03"]
ROOTS = ["root-1", "root-2", "root-3"]


def simulate(seed: int) -> list[dict[str, EvidenceBundle]]:
    """Six full projections: anchors per (root, day) never vanish; the rest churn."""
    rng = random.Random(seed)
    truth: dict[str, dict[str, Candidate | Debit]] = {r: {} for r in ROOTS}
    out = []
    for step in range(len(STEPS)):
        for root in ROOTS:
            task = TaskId(root, "turn-1", "main")
            items = truth[root]
            if step == 0:
                for n, day in enumerate(EVENT_DAYS):
                    items[f"c-anchor-{n}"] = candidate(
                        f"{root}-ca{n}",
                        task=task,
                        day=day,
                        immediate=(atom(f"{root}-s{n}", 400, 0, f"ev-{root}-{n}"),),
                    )
                    items[f"d-anchor-{n}"] = debit(f"{root}-da{n}", None, task=task, day=day)
                continue
            for key in sorted(items):
                roll = rng.random()
                item = items[key]
                if isinstance(item, Debit) and roll < 0.5:
                    chars = (item.chars or 0) + rng.choice([40, 120, 400])
                    items[key] = debit(item.debit_id, chars, task=task, day=item.day)
                elif isinstance(item, Candidate) and roll < 0.2:
                    size = rng.choice([200, 600, 800])
                    atom_id = f"{item.candidate_id}-a{size}"
                    items[key] = candidate(
                        item.candidate_id,
                        task=task,
                        day=item.day,
                        immediate=(atom(atom_id, size, 0, f"ev-{item.candidate_id}"),),
                    )
                elif "anchor" not in key and roll > 0.85:
                    del items[key]
            if rng.random() < 0.7:
                tag, day = f"{root}-n{step}", rng.choice(EVENT_DAYS)
                items[f"c-{tag}"] = candidate(
                    tag,
                    task=task,
                    day=day,
                    immediate=(atom(f"{tag}-s", 300, 0, f"ev-{tag}"),),
                )
                items[f"d-{tag}"] = debit(f"{tag}-d", rng.choice([None, 200]), task=task, day=day)
        out.append(
            {
                root: bundle(
                    [i for i in items.values() if isinstance(i, Candidate)],
                    [i for i in items.values() if isinstance(i, Debit)],
                )
                for root, items in truth.items()
            }
        )
    return out


def persist(ledger: VersionedLedger) -> VersionedLedger:
    state = json.loads(json.dumps(ledger_state_to_dict(ledger.state())))
    superseded = json.loads(json.dumps(ledger.superseded_state()))
    return VersionedLedger.restore_versioned(ledger_state_from_dict(state), superseded)


def shuffled(desired: dict[str, EvidenceBundle], rng: random.Random) -> dict[str, EvidenceBundle]:
    out = {}
    for root, b in sorted(desired.items(), reverse=True):
        cands, debits = list(b.candidates), list(b.debits)
        rng.shuffle(cands)
        rng.shuffle(debits)
        out[root] = bundle(cands, debits)
    return out


def test_incremental_runs_with_restarts_match_continuous_runs_and_final_totals():
    projections = simulate(seed=20261013)
    continuous, restarted = VersionedLedger(METHOD), VersionedLedger(METHOD)
    fresh = VersionedLedger(METHOD)
    totals = {"replaced": 0, "retracted": 0, "superseded": 0, "added": 0}
    for projection, today in zip(projections, STEPS, strict=True):
        result = continuous.reconcile(projection, today=today)
        for name in totals:
            totals[name] += getattr(result, name)
        restarted = persist(restarted)
        assert restarted.reconcile(projection, today=today) == result
        assert restarted.rows().canonical_bytes() == continuous.rows().canonical_bytes()
        assert result.rejected == ()
    assert all(totals.values()), totals  # the run exercised every reconcile branch
    assert continuous.superseded  # at least one sealed period was revised

    # A fresh ledger given only the FINAL projection stamps everything with a single day, so its
    # revision rows are NOT required to equal the incremental history; the current totals must.
    fresh.reconcile(projections[-1], today=STEPS[-1])
    final, current = fresh.rows(), continuous.rows()
    assert current_total(current.periods, NS, CHARS) == current_total(final.periods, NS, CHARS)
    assert [(r.period, r.leaves) for r in current_rows(current.periods, NS)] == [
        (r.period, r.leaves) for r in current_rows(final.periods, NS)
    ]
    # no identity ever has two active versions (the conflict reason cannot appear)
    assert b"conflicting_duplicate_debit" not in current.canonical_bytes()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_item_order_inside_desired_bundles_never_changes_rows(seed):
    projections = simulate(seed=99)
    ordered, scrambled = VersionedLedger(METHOD), VersionedLedger(METHOD)
    rng = random.Random(seed)
    for projection, today in zip(projections, STEPS, strict=True):
        ordered.reconcile(projection, today=today)
        scrambled.reconcile(shuffled(projection, rng), today=today)
        assert scrambled.rows().canonical_bytes() == ordered.rows().canonical_bytes()
        assert scrambled.state() == ordered.state()
        assert scrambled.superseded == ordered.superseded


# ---- restored conflicts, foreign roots, exhaustive reconcile ----------------------------------


def test_restored_conflicting_variants_retire_the_stale_copy_when_one_matches() -> None:
    old = VersionedLedger(METHOD)
    old.ingest(EvidenceBundle((candidate("one"),), (debit("lookup", None),)))
    old.ingest(EvidenceBundle((), (debit("lookup", 640),)))
    assert b"conflicting_duplicate_debit" in old.rows().canonical_bytes()
    restored = VersionedLedger.restore_versioned(old.state(), {})
    result = restored.reconcile(
        {"root-1": EvidenceBundle((candidate("one"),), (debit("lookup", 640),))},
        today="2026-11-02",
    )
    assert result.replaced == 1
    assert len(restored.state().debits) == 1
    assert restored.state().debits[0].chars == 640
    assert b"conflicting_duplicate_debit" not in restored.rows().canonical_bytes()


def test_projection_rejects_candidates_and_debits_owned_by_a_different_root() -> None:
    ledger = VersionedLedger(METHOD)
    foreign = TaskId("other-root", "turn", "main")
    result = ledger.reconcile(
        {
            "root-1": EvidenceBundle(
                (candidate("good"), candidate("foreign", task=foreign)),
                (debit("foreign-debit", 400, task=foreign),),
            )
        },
        today="2026-11-02",
    )
    assert set(result.rejected) == {
        ("foreign", "root_mismatch"),
        ("foreign-debit", "root_mismatch"),
    }
    assert [item.candidate_id for item in ledger.state().candidates] == ["good"]
    assert ledger.state().debits == ()


def test_exhaustive_reconcile_retracts_roots_the_projection_no_longer_mentions() -> None:
    other = TaskId("root-2", "turn-1", "main")

    def projection(root_task: TaskId, ident: str) -> EvidenceBundle:
        return EvidenceBundle(
            (
                candidate(
                    f"c-{ident}", task=root_task, immediate=(atom(f"s-{ident}", 400, 0, ident),)
                ),
            ),
            (debit(f"d-{ident}", 4000, task=root_task, exposure_id=f"e-{ident}"),),
        )

    first = TaskId("root-1", "turn-1", "main")
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": projection(first, "1"), "root-2": projection(other, "2")}, today="2026-11-02"
    )
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}

    ledger.reconcile({"root-2": projection(other, "2")}, today="2026-11-03")  # not mentioned
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}  # untouched

    result = ledger.reconcile(
        {"root-2": projection(other, "2")}, today="2026-11-03", exhaustive=True
    )
    assert result.retracted >= 2  # root-1's candidate and debit
    assert {c.task.root for c in ledger.state().candidates} == {"root-2"}
    assert {d.task.root for d in ledger.state().debits} == {"root-2"}
    again = ledger.reconcile(
        {"root-2": projection(other, "2")}, today="2026-11-03", exhaustive=True
    )
    assert (again.added, again.replaced, again.retracted) == (0, 0, 0)  # idempotent


def test_exhaustive_reconcile_never_touches_frozen_roots() -> None:
    ledger = VersionedLedger(METHOD)
    ledger.reconcile(
        {"root-1": EvidenceBundle((candidate("c1", immediate=(atom("s1", 400, 0, "e1"),)),), ())},
        today="2026-11-02",
    )
    before = ledger.expire_roots(["root-1"])
    assert before == ["root-1"]
    frozen = ledger.rows().digest
    ledger.reconcile({}, today="2026-11-03", exhaustive=True)
    assert ledger.rows().digest == frozen  # closed totals survive an exhaustive reconcile


# ---- a committed ledger blob of a foreign methodology is corruption, not data ---------------

NOW = scene.at(hour=20)


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def test_cross_namespace_ledger_reference_is_not_published(tmp_path: Path) -> None:
    seeded(tmp_path)
    checkpoint = scene.committed()
    original = checkpoint.to_dict()
    key = collector_state.ledger_blob_name(checkpoint.methodology)
    original_path = vs.accounting_dir() / "collector" / "state" / checkpoint.blobs[key]
    original_bytes = original_path.read_bytes()
    assert METHOD.hash != checkpoint.methodology
    checkpoint.blobs[key] = store.write_blob(
        PROFILE,
        key,
        {
            "state": ledger_state_to_dict(VersionedLedger(METHOD).state()),
            "superseded": {},
            "collection_scope": checkpoint.collection_scope,
        },
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    before = scene.current()
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW)
    assert failed.value.condition == "state_corrupt"
    assert scene.current() == before
    original_path.write_bytes(original_bytes)
    original_path.chmod(0o600)
    store.commit_checkpoint(PROFILE, store.Checkpoint.from_dict(original))
    assert scene.report(scene.collect(NOW)).generation == 2
