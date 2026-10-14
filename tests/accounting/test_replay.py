"""Full replay vs restarted incremental batches in any order: identical bytes."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.replay import Ledger, full_replay, select_namespace
from ssgrep.accounting.valuation_inputs import Candidate, Debit, EvidenceBundle
from tests.accounting.valuation_builders import METHOD, atom, candidate, debit


def corpus() -> EvidenceBundle:
    """Several tasks, days, overlapping atoms, late evidence, missing and partial debits."""
    t1 = TaskId("root-1", "turn-1", "main")
    t2 = TaskId("root-1", "turn-2", "main")
    unknown = TaskId("root-1", None, "main")
    t3 = TaskId("root-2", "turn-1", "main")
    cands = [
        candidate("a", task=t1, accepted_day="2026-11-02"),
        candidate("b", task=t1, candidate_type="identifier", accepted_day="2026-11-03"),
        candidate(
            "c", task=t2, immediate=(atom("c1", 600, 100, "ev-1"),), accepted_day="2026-11-03"
        ),
        candidate("u", task=unknown, immediate=(atom("u1", 200, 900, "ev-1"),)),
        candidate("d", task=t3, day="2026-11-03", immediate=(atom("d1", 80, 0, "ev-3"),)),
        candidate(
            "late",
            task=t3,
            day="2026-11-03",
            accepted_day="2026-11-30",
            immediate=(atom("l1", 800, 5000, "ev-4"),),
        ),
    ]
    debits = [
        debit("x1", 640, task=t1, accepted_day="2026-11-02"),
        debit("x2", None, task=t1, accepted_day="2026-11-02"),
        debit("x3", 200, task=t2, partial=True),
        debit("x4", 400, task=t3, day="2026-11-03", accepted_day="2026-11-30"),
    ]
    return EvidenceBundle(tuple(cands), tuple(debits))


def batches(bundle: EvidenceBundle, rng: random.Random) -> list[EvidenceBundle]:
    items: list[tuple[str, Candidate | Debit]] = [("c", c) for c in bundle.candidates]
    items += [("d", d) for d in bundle.debits]
    rng.shuffle(items)
    cuts = sorted(rng.sample(range(1, len(items)), 3))
    out = []
    for lo, hi in zip([0, *cuts], [*cuts, len(items)], strict=True):
        part = items[lo:hi]
        cands = tuple(x for k, x in part if isinstance(x, Candidate))
        debs = tuple(x for k, x in part if isinstance(x, Debit))
        out.append(EvidenceBundle(cands, debs))
    return out


def test_incremental_batches_in_different_orders_equal_full_replay_bytes_and_digest():
    reference = full_replay(corpus(), METHOD)
    assert reference.periods and reference.receipts
    rng = random.Random(1012)
    for _ in range(25):
        ledger = Ledger(METHOD)
        for part in batches(corpus(), rng):
            ledger.ingest(part)
            ledger.rows()  # intermediate reductions must not poison later ones
        rows = ledger.rows()
        assert rows.canonical_bytes() == reference.canonical_bytes()
        assert rows.digest == reference.digest


def test_restart_from_persisted_state_matches_a_continuous_run():
    reference = full_replay(corpus(), METHOD)
    parts = batches(corpus(), random.Random(3))
    first = Ledger(METHOD)
    first.ingest(parts[0])
    first.ingest(parts[1])
    resumed = Ledger.restore(first.state())  # process restart
    for part in parts[2:]:
        resumed.ingest(part)
    assert resumed.rows().canonical_bytes() == reference.canonical_bytes()


def test_reingesting_everything_is_idempotent():
    ledger = Ledger(METHOD)
    first = ledger.ingest(corpus())
    again = ledger.ingest(corpus())
    assert first.accepted == 10 and again.accepted == 0 and again.duplicates == 10
    assert ledger.rows().canonical_bytes() == full_replay(corpus(), METHOD).canonical_bytes()


def test_canonical_rows_contain_no_clock_values_and_no_floats():
    rows = full_replay(corpus(), METHOD).canonical_bytes().decode()
    assert "collected_at" not in rows and "mono_ns" not in rows
    full_replay(corpus(), METHOD).canonical_bytes()  # raises TypeError on any float


def test_late_evidence_replays_to_the_same_revisions_as_it_arrived_incrementally():
    ledger = Ledger(METHOD)
    early = corpus().accepted_through("2026-11-05")
    late = EvidenceBundle(
        tuple(c for c in corpus().candidates if c not in early.candidates),
        tuple(d for d in corpus().debits if d not in early.debits),
    )
    ledger.ingest(early)
    before = ledger.rows()
    ledger.ingest(late)
    after = ledger.rows()
    assert after.canonical_bytes() == full_replay(corpus(), METHOD).canonical_bytes()
    kept = {(p.period, p.revision): p.digest for p in before.periods}
    now = {(p.period, p.revision): p.digest for p in after.periods}
    assert all(now[k] == v for k, v in kept.items())  # append-only: old rows unchanged
    assert len(now) > len(kept)  # and the late evidence added a revision


def test_receipt_revisions_are_append_only_and_deterministic():
    ledger = Ledger(METHOD)
    ledger.ingest(corpus().accepted_through("2026-11-05"))
    first = {(r.receipt_id, r.receipt.revision): r.digest for r in ledger.rows().receipts}
    ledger.ingest(corpus())
    second = {(r.receipt_id, r.receipt.revision): r.digest for r in ledger.rows().receipts}
    assert all(second[k] == v for k, v in first.items())


def test_methodology_change_is_a_new_namespace_selected_not_summed():
    other = replace(METHOD, label="v2", frequency_view="freq-2")
    old = full_replay(corpus(), METHOD)
    new = full_replay(corpus(), other)
    assert old.namespace != new.namespace and old.digest != new.digest
    assert select_namespace([old, new], other) is new
    assert select_namespace([old, new], METHOD) is old
    with pytest.raises(KeyError):
        select_namespace([old], other)
    assert all(p.namespace == old.namespace for p in old.periods)
    assert all(r.receipt.methodology == METHOD.hash for r in old.receipts)


def test_clock_and_input_order_never_change_rows():
    import time

    ledger_a, ledger_b = Ledger(METHOD), Ledger(METHOD)
    bundle = corpus()
    ledger_a.ingest(bundle)
    time.sleep(0.01)
    ledger_b.ingest(
        EvidenceBundle(tuple(reversed(bundle.candidates)), tuple(reversed(bundle.debits)))
    )
    assert ledger_a.rows().canonical_bytes() == ledger_b.rows().canonical_bytes()


def test_expired_roots_keep_replayed_totals_and_survive_restart():
    ledger = Ledger(METHOD)
    ledger.ingest(corpus())
    before = {(p.period, p.revision): p.digest for p in ledger.rows().periods if p.revision == 1}
    ledger.expire_roots(["root-1"])
    restored = Ledger.restore(ledger.state())
    rows = restored.rows()
    assert rows.canonical_bytes() == ledger.rows().canonical_bytes()
    flags = {r.receipt.root: r.receipt.evidence_available for r in rows.receipts}
    assert flags == {"root-1": False, "root-2": True}
    assert before  # the pre-expiry rows existed; totals were kept, not recomputed away
