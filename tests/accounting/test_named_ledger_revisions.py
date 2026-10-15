"""Explicit same-day corrections preserve history without adding historical debit."""

from __future__ import annotations

from dataclasses import replace

from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.contracts import Basis, Quantity, TaskId
from ssgrep.accounting.units import CHARS_DIV4_V1, CONTEXT_TOKENS, UnitSpec
from ssgrep.accounting.valuation import rollup
from ssgrep.accounting.valuation_inputs import Debit, EvidenceBundle, Methodology

DAY = "2026-10-01"
UNIT = UnitSpec(CONTEXT_TOKENS, CHARS_DIV4_V1)
TASK = TaskId("omp:native-session", "user:one")


def debit(identity: str) -> Debit:
    return Debit(
        identity,
        TASK,
        DAY,
        "exposure",
        UNIT,
        Quantity(39, CONTEXT_TOKENS, Basis.ESTIMATED, CHARS_DIV4_V1),
        exposure_id=identity,
        chars=156,
    )


def nets(rows) -> tuple[int | None, ...]:
    return tuple(e.net.value for e in rollup(rows.periods[-1].leaves, UNIT))


def test_same_day_named_correction_keeps_78_history_and_only_39_current() -> None:
    ledger = VersionedLedger(Methodology("independent-correction", units=(UNIT,)))
    ledger.reconcile(
        {TASK.root: EvidenceBundle((), (debit("native"), debit("legacy-wrapper-copy")))},
        today=DAY,
    )
    before = ledger.rows().canonical_bytes()
    ledger.archive_current("native_identity_v2:1")
    ledger.reconcile({TASK.root: EvidenceBundle((), (debit("native"),))}, today=DAY)
    current = ledger.rows()
    assert nets(current) == (-39, -39, -39)
    assert len(current.archives) == 1
    archived = current.archives[0]
    assert archived.revision_name == "native_identity_v2:1"
    assert archived.as_rows().canonical_bytes() == before
    assert nets(archived.as_rows()) == (-78, -78, -78)

    restored = VersionedLedger.restore_versioned(
        ledger.state(), ledger.superseded_state(), ledger.archives_state()
    )
    assert restored.rows().canonical_bytes() == current.canonical_bytes()
    restored.archive_current("native_identity_v2:1")
    assert restored.rows().canonical_bytes() == current.canonical_bytes()


def test_distinct_same_day_recomputations_keep_their_own_frozen_rows() -> None:
    ledger = VersionedLedger(Methodology("independent-correction", units=(UNIT,)))
    ledger.reconcile({TASK.root: EvidenceBundle((), (debit("native"),))}, today=DAY)
    first = ledger.rows().canonical_bytes()
    ledger.archive_current("frequency-recompute:1")
    ledger.reconcile(
        {TASK.root: EvidenceBundle((), (debit("native"), debit("new-execution")))},
        today=DAY,
    )
    ledger.archive_current("frequency-recompute:2")
    ledger.reconcile({TASK.root: EvidenceBundle((), (debit("native"),))}, today=DAY)
    current = ledger.rows()
    assert nets(current) == (-39, -39, -39)
    assert tuple(a.revision_name for a in current.archives) == (
        "frequency-recompute:1",
        "frequency-recompute:2",
    )
    assert current.archives[0].as_rows().canonical_bytes() == first
    assert tuple(nets(a.as_rows()) for a in current.archives) == ((-39, -39, -39), (-78, -78, -78))


def test_scoped_purge_removes_archived_contribution_without_other_workspace_loss() -> None:
    other = TaskId("omp:other-workspace", "user:other")
    ledger = VersionedLedger(Methodology("independent-correction", units=(UNIT,)))
    ledger.reconcile(
        {
            TASK.root: EvidenceBundle((), (debit("native"),)),
            other.root: EvidenceBundle((), (replace(debit("other"), task=other),)),
        },
        today=DAY,
    )
    ledger.archive_current("before-workspace-purge")
    ledger.purge(roots=(root for root in (TASK.root,)))
    current = ledger.rows()
    assert nets(current) == (-39, -39, -39)
    assert len(current.archives) == 1
    historical = current.archives[0].as_rows()
    assert nets(historical) == (-39, -39, -39)
    assert {leaf.task.root for period in historical.periods for leaf in period.leaves} == {
        other.root
    }


def test_scoped_date_purge_uses_intersection_for_active_and_archived_rows() -> None:
    other = TaskId("omp:other-workspace", "user:other")
    later = "2026-10-10"
    ledger = VersionedLedger(Methodology("independent-correction", units=(UNIT,)))
    ledger.reconcile(
        {
            TASK.root: EvidenceBundle((), (debit("old"), replace(debit("new"), day=later))),
            other.root: EvidenceBundle((), (replace(debit("other"), task=other),)),
        },
        today=later,
    )
    ledger.archive_current("before-scoped-date-purge")
    ledger.purge_scoped(before_day="2026-10-05", roots=(TASK.root,))
    rows = ledger.rows()
    expected = {(TASK.root, later), (other.root, DAY)}
    for actual in (rows, rows.archives[0].as_rows()):
        assert {
            (leaf.task.root, leaf.day) for period in actual.periods for leaf in period.leaves
        } == expected
        values = rollup([leaf for period in actual.periods for leaf in period.leaves], UNIT)
        assert tuple(value.net.value for value in values) == (-78, -78, -78)
