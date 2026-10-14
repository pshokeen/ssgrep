"""A conversation whose parent arrives later must not leave its old root behind in the ledger.

A child transcript read BEFORE its parent has no resolvable lineage, so it is its own root. When
the parent file appears, the whole conversation is one root. Full replay of the retained evidence
yields only that one root; the incremental ledger has to converge to the same rows, never keep the
child's earlier root (which would count every one of its lookups twice).
"""

from __future__ import annotations

import collections
import os
from pathlib import Path

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import load_state
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.readers_fixtures import write_records
from tests.accounting.reduce_builders import (
    PROFILE,
    omp_header,
    omp_launch,
    omp_search,
    omp_user,
    results_json,
    ts,
)

NOW = scene.at(hour=20)
STAMP = scene.at(hour=10).timestamp()


def sessions(root: Path) -> Path:
    return root / "omp-sessions" / "-work"


def write_child(root: Path) -> None:
    parent = sessions(root) / "root.jsonl"
    child = sessions(root) / "root" / "ChildAlpha.jsonl"
    write_records(
        child,
        [
            omp_header("child-a", ts(12, 9, 12), parent=parent),
            *omp_search("c1", "call-1", "deploy", results_json(), (12, 10, 0, 0)),
            *omp_search("c2", "call-2", "billing", results_json(), (12, 10, 1, 0)),
        ],
    )
    os.utime(child, (STAMP, STAMP))


def write_parent(root: Path) -> None:
    parent = sessions(root) / "root.jsonl"
    write_records(
        parent,
        [
            omp_header("root-1", ts(12, 8)),
            omp_user("e-a", "investigate the failing deploy", ts(12, 9)),
            *omp_launch("e-l1", "c-launch", "ChildAlpha", ts(12, 9, 10), ts(12, 9, 11)),
        ],
    )
    os.utime(parent, (STAMP, STAMP))


def ledger() -> VersionedLedger:
    committed = vs.checkpoint()
    assert committed is not None
    state = load_state(PROFILE, committed, committed.methodology)
    assert state.ledger is not None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded)


def live_debit_roots() -> dict[str, set[str]]:
    by_id: dict[str, set[str]] = collections.defaultdict(set)
    for debit in ledger()._debits.values():
        by_id[debit.debit_id].add(debit.task.root)
    return dict(by_id)


def run(root: Path, *, parent_later: bool) -> vs.Facts:
    vs.enable(pin=True)
    vs.start_capture(day=12)
    write_child(root)
    if parent_later:
        first = scene.collect(NOW)
        assert first.state == "published"
        assert {r for roots in live_debit_roots().values() for r in roots} == {"omp:child-a"}
    write_parent(root)
    assert scene.collect(NOW).state == "published"
    return vs.facts()


def test_the_scene_really_resolves_the_child_under_the_parent_root_in_one_shot(
    tmp_path: Path,
) -> None:
    run(tmp_path, parent_later=False)
    assert {r for roots in live_debit_roots().values() for r in roots} == {"omp:root-1"}


def test_a_late_parent_leaves_no_debit_under_the_childs_earlier_root(tmp_path: Path) -> None:
    run(tmp_path, parent_later=True)
    owners = live_debit_roots()
    assert owners, "the scene must produce lookup debits"
    assert all(roots == {"omp:root-1"} for roots in owners.values()), owners  # one root each
    assert len(ledger()._debits) == len(owners)  # and no id is held twice


def test_incremental_arrival_of_the_parent_equals_the_single_shot_ledger(tmp_path: Path) -> None:
    incremental = run(tmp_path, parent_later=True)
    vs.wipe(tmp_path)
    for leftover in ("omp-sessions",):
        __import__("shutil").rmtree(tmp_path / leftover, ignore_errors=True)
    single = run(tmp_path, parent_later=False)
    assert incremental.evidence == single.evidence
    assert incremental.ledger == single.ledger  # canonical rows identical to full replay
    assert incremental.snapshot_id == single.snapshot_id


def test_receipt_files_follow_the_ledger_when_a_root_disappears(tmp_path: Path) -> None:
    run(tmp_path, parent_later=True)
    receipts = tmp_path / "data" / "accounting" / PROFILE / "receipts"
    on_disk = {p.stem for p in receipts.glob("*.json")} if receipts.exists() else set()
    in_ledger = {r.receipt_id for r in ledger().rows().receipts}
    assert on_disk == in_ledger  # no ghost receipt of a retracted root stays explainable
    assert store.load_checkpoint(PROFILE) is not None


# ---- the ledger operation itself ---------------------------------------------------------------


def test_exhaustive_reconcile_retracts_roots_the_projection_no_longer_mentions() -> None:
    from ssgrep.accounting.contracts import TaskId
    from ssgrep.accounting.valuation_inputs import EvidenceBundle
    from tests.accounting.valuation_builders import METHOD, atom, candidate, debit

    other = TaskId("root-2", "turn-1", "main")

    def bundle(root_task: TaskId, ident: str) -> EvidenceBundle:
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
        {"root-1": bundle(first, "1"), "root-2": bundle(other, "2")}, today="2026-11-02"
    )
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}

    ledger.reconcile({"root-2": bundle(other, "2")}, today="2026-11-03")  # default: not mentioned
    assert {c.task.root for c in ledger.state().candidates} == {"root-1", "root-2"}  # untouched

    result = ledger.reconcile({"root-2": bundle(other, "2")}, today="2026-11-03", exhaustive=True)
    assert result.retracted >= 2  # root-1's candidate and debit
    assert {c.task.root for c in ledger.state().candidates} == {"root-2"}
    assert {d.task.root for d in ledger.state().debits} == {"root-2"}
    again = ledger.reconcile({"root-2": bundle(other, "2")}, today="2026-11-03", exhaustive=True)
    assert (again.added, again.replaced, again.retracted) == (0, 0, 0)  # idempotent


def test_exhaustive_reconcile_never_touches_frozen_roots() -> None:
    from ssgrep.accounting.valuation_inputs import EvidenceBundle
    from tests.accounting.valuation_builders import METHOD, atom, candidate

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
