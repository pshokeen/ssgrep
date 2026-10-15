"""Revision, sealing, expiry and purge behavior with quantified boundary examples."""

from __future__ import annotations

from ssgrep.accounting.contracts import Basis, TaskId
from ssgrep.accounting.replay import Ledger
from ssgrep.accounting.revisions import (
    current_rows,
    current_total,
    history,
    last_provisional_day,
    seal_day,
    status_at,
)
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.valuation_builders import CHARS, METHOD, atom, candidate, debit

PERIOD = "2026-11-01"


def early(**kw):
    """100-token credit and 40-token debit on PERIOD, accepted inside the window."""
    return EvidenceBundle(
        (candidate("early", accepted_day="2026-11-02", **kw),),
        (debit("d-early", 160, accepted_day="2026-11-02"),),
    )


def late_bundle():
    """More evidence for the SAME period, accepted long after the seal day."""
    extra = candidate(
        "late",
        accepted_day="2026-11-20",
        immediate=(atom("late-step", 800, start=5000, origin="ev-late"),),
    )
    return EvidenceBundle((extra,), (debit("d-late", 400, accepted_day="2026-11-20"),))


def central_net(ledger):
    total, _ = current_total(ledger.rows().periods, ledger.namespace, CHARS)
    return total.value if total else None


def test_seal_boundary_is_seven_utc_days_after_period_end():
    assert last_provisional_day(PERIOD) == "2026-11-08"
    assert seal_day(PERIOD) == "2026-11-09"
    assert status_at(PERIOD, "2026-11-08") == "provisional"
    assert status_at(PERIOD, "2026-11-09") == "sealed"


def test_evidence_inside_the_provisional_window_folds_into_one_revision():
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    ledger.ingest(
        EvidenceBundle(
            (
                candidate(
                    "same-window", accepted_day="2026-11-08", immediate=(atom("w", 400, 9000),)
                ),
            )
        )
    )
    rows = [p for p in ledger.rows().periods if p.period == PERIOD]
    assert [(p.revision, p.kind, p.late) for p in rows] == [(1, "original", False)]
    assert central_net(ledger) == (100 + 100) - 40


def test_late_evidence_after_the_seal_supersedes_and_history_shows_both_without_double_counting():
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    assert central_net(ledger) == 100 - 40
    ledger.ingest(late_bundle())
    rows = ledger.rows().periods
    mine = [p for p in rows if p.period == PERIOD]
    assert [(p.revision, p.supersedes, p.kind, p.late) for p in mine] == [
        (1, None, "original", False),
        (2, 1, "adjustment", True),
    ]
    # revision 1 and revision 2 both exist in history, but only revision 2 is current
    records = history(rows, ledger.namespace, CHARS, today="2026-12-01")
    assert [(r.revision, r.current, r.status) for r in records] == [
        (1, False, "superseded"),
        (2, True, "sealed"),
    ]
    assert records[0].net.value == 60 and records[1].net.value == (100 + 200) - (40 + 100)
    total, _ = current_total(rows, ledger.namespace, CHARS)
    assert (
        total is not None and total.value == 160
    )  # the replacement alone: 60 + 160 would double count
    assert [p.revision for p in current_rows(rows, ledger.namespace)] == [2]


def test_current_total_is_independent_of_how_many_late_revisions_exist():
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    for day, cid in (("2026-11-15", "l1"), ("2026-11-20", "l2")):
        ledger.ingest(
            EvidenceBundle(
                (
                    candidate(
                        cid,
                        accepted_day=day,
                        immediate=(atom(cid, 400, start=7000 + len(cid) + int(day[-2:]) * 1000),),
                    ),
                )
            )
        )
    revisions = sorted(p.revision for p in ledger.rows().periods if p.period == PERIOD)
    assert revisions == [1, 2, 3]
    assert central_net(ledger) == 100 + 100 + 100 - 40


def test_unavailable_net_stays_unavailable_in_history():
    ledger = Ledger(METHOD)
    ledger.ingest(EvidenceBundle((candidate("c"),), (debit("d", None),)))
    (record,) = history(ledger.rows().periods, ledger.namespace, CHARS, today="2026-11-02")
    assert record.net.basis is Basis.UNAVAILABLE and record.status == "provisional"


def test_methodology_change_is_a_new_namespace_and_never_additive():
    from dataclasses import replace

    other = replace(METHOD, label="test-v2", frequency_view="freq-2")
    old, new = Ledger(METHOD), Ledger(other)
    for ledger in (old, new):
        ledger.ingest(early())
    assert old.namespace != new.namespace
    combined = list(old.rows().periods) + list(new.rows().periods)
    # selecting one namespace gives exactly that namespace's current total, not the sum
    old_total = current_total(combined, old.namespace, CHARS)[0]
    new_total = current_total(combined, new.namespace, CHARS)[0]
    assert old_total is not None and old_total.value == 60
    assert new_total is not None and new_total.value == 60
    assert {p.namespace for p in current_rows(combined, new.namespace)} == {new.namespace}


def test_expired_evidence_keeps_closed_totals_marks_receipts_and_refuses_rebuilds():
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    ledger.ingest(late_bundle())
    net_before = central_net(ledger)
    assert ledger.expirable_roots("2026-12-01") == ["root-1"]
    assert ledger.expire_roots(["root-1"]) == ["root-1"]
    rows = ledger.rows()
    assert central_net(ledger) == net_before == 160  # the total does not silently shrink
    assert all(not p.recomputable for p in rows.periods)
    assert rows.receipts and all(not r.receipt.evidence_available for r in rows.receipts)
    assert all(("recomputation", "evidence_expired") in r.missing_costs for r in rows.receipts)
    # a changed live source cannot rebuild the old digest: new evidence for it is rejected
    result = ledger.ingest(EvidenceBundle((candidate("revived", accepted_day="2026-12-01"),)))
    assert (result.accepted, result.rejected) == (0, (("revived", "evidence_expired_root"),))
    assert central_net(ledger) == net_before


def test_expiry_leaves_other_roots_recomputable():
    other = TaskId("root-2", "turn-1", "main")
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    ledger.ingest(
        EvidenceBundle((candidate("o", task=other, day="2026-11-03", accepted_day="2026-11-04"),))
    )
    ledger.expire_roots(["root-1"])
    flags = {p.period: p.recomputable for p in ledger.rows().periods}
    assert flags == {"2026-11-01": False, "2026-11-03": True}


def test_privacy_purge_removes_totals_and_receipts_even_after_expiry():
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    ledger.expire_roots(["root-1"])
    assert central_net(ledger) == 60 and ledger.rows().receipts
    ledger.purge(roots=["root-1"])
    rows = ledger.rows()
    assert rows.periods == () and rows.receipts == ()
    assert central_net(ledger) is None


def test_purge_before_a_day_removes_only_older_periods():
    ledger = Ledger(METHOD)
    ledger.ingest(early())
    ledger.ingest(
        EvidenceBundle(
            (
                candidate(
                    "new",
                    day="2026-11-10",
                    accepted_day="2026-11-11",
                    immediate=(atom("n", 400, 9000),),
                ),
            )
        )
    )
    ledger.purge(before_day="2026-11-05")
    rows = ledger.rows()
    assert {p.period for p in rows.periods} == {"2026-11-10"}
    assert {r.period for r in rows.receipts} == {"2026-11-10"}
    assert central_net(ledger) == 100
