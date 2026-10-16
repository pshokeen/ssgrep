"""Explicit same-day corrections preserve history without adding historical debit."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from ssgrep.accounting.collect import (
    cycle_publish as publish,
    state as collector_state,
    store as store,
)
from ssgrep.accounting.collect.replay import Ledger, LedgerRows, LedgerState, select_namespace
from ssgrep.accounting.collect.state import VersionedLedger
from ssgrep.accounting.collect.store import CollectError
from ssgrep.accounting.ledger.contracts import Basis, Reuse, TaskId
from ssgrep.accounting.ledger.ledger_codec import from_plain, to_plain
from ssgrep.accounting.ledger.revisions import (
    current_rows,
    current_total,
    history,
    last_provisional_day,
    seal_day,
    status_at,
)
from ssgrep.accounting.ledger.units import CHARS_DIV4_V1, CONTEXT_TOKENS, UnitSpec
from ssgrep.accounting.ledger.valuation import Leaf, rollup, value_evidence
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus, Debit, EvidenceBundle, Methodology
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.privacy.storage import loads
from ssgrep.accounting.views import service
from ssgrep.accounting.views.records import (
    EXPIRED_REASON,
    ReceiptRecord,
    build_receipt,
    build_receipts,
    default_view,
    receipt_id_for,
)
from tests.accounting.collect.test_publication import PROFILE, ledger as make_ledger
from tests.accounting.ledger.valuation_builders import (
    CHARS,
    DAY as RECEIPT_DAY,
    METHOD,
    TASK as RECEIPT_TASK,
    atom,
    candidate,
    debit as make_debit,
)
from tests.accounting.views.snapshot_builders import RECEIPT_ID, write_receipt
from tests.accounting.views.test_ux_support import fingerprint

DAY = "2026-10-01"
UNIT = UnitSpec(CONTEXT_TOKENS, CHARS_DIV4_V1)
TASK = TaskId("omp:native-session", "user:one")


def debit(
    identity: str,
    chars: int | None = 156,
    **kwargs,
) -> Debit:
    kwargs.setdefault("task", TASK)
    kwargs.setdefault("day", DAY)
    return make_debit(identity, chars, **kwargs)


def receipt_debit(identity: str, chars: int | None, **kwargs) -> Debit:
    kwargs.setdefault("task", RECEIPT_TASK)
    kwargs.setdefault("day", RECEIPT_DAY)
    return make_debit(identity, chars, **kwargs)


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


def test_current_daily_ownership_replay_cannot_reuse_the_v3_namespace() -> None:
    from ssgrep.accounting.ledger.valuation_inputs import Candidate
    from tests.accounting.ledger.valuation_builders import atom, candidate

    method = Methodology("daily-owner-cutover", units=(UNIT,))
    previous = replace(method, valuation_version="honest_bounds_ns_v3")
    early: Candidate = candidate("early", task=TASK, day=DAY, immediate=(atom("shared", 400),))
    late = replace(early, candidate_id="late", day="2026-10-02")
    ledger = VersionedLedger(method)
    ledger.reconcile(
        {TASK.root: EvidenceBundle((late, early), (debit("native"),))},
        today="2026-10-02",
    )
    rows = ledger.rows()
    assert rows.namespace != previous.hash
    assert method.valuation_version == "honest_bounds_ns_v4"
    assert {leaf.methodology for row in rows.periods for leaf in row.leaves} == {method.hash}
    assert {
        row.period: tuple(estimate.credit.value for estimate in rollup(row.leaves, UNIT))
        for row in rows.periods
    } == {DAY: (100, 100, 100), "2026-10-02": (0, 0, 0)}


@pytest.mark.parametrize("version", ("honest_bounds_ns_v3", "honest_bounds_v1"))
def test_explicit_retired_method_cannot_label_current_attribution_math(version: str) -> None:
    from ssgrep.accounting.ledger.valuation import value_evidence

    retired = Methodology("retired-evaluation", units=(UNIT,), valuation_version=version)
    with pytest.raises(ValueError, match="methodology_read_only"):
        value_evidence(EvidenceBundle((), (debit("native"),)), retired)


@dataclass(frozen=True)
class AcceptedEvaluation:
    root: str
    keys: tuple[str, ...]
    leaves: tuple[Leaf, ...]
    receipts: dict[str, ReceiptRecord]


@dataclass(frozen=True)
class AcceptedV3:
    source_commit: str
    source_valuation: str
    source_valuation_sha256: str
    state: LedgerState
    superseded: dict[str, str]
    rows: LedgerRows
    canonical_rows_utf8: str
    evaluations: tuple[AcceptedEvaluation, ...]
    inputs: dict[str, EvidenceBundle]
    expired_state: LedgerState
    expired_rows: LedgerRows


def accepted_v3() -> AcceptedV3:
    document = loads((Path(__file__).parent.parent / "v3_retirement_fixture.json").read_bytes())
    proof = from_plain(AcceptedV3, document)
    assert proof.source_commit == "2f0f620dcf212381e179a4fffd52bd4c277034fb"
    assert proof.source_valuation == "src/ssgrep/accounting/valuation.py"
    assert len(proof.source_valuation_sha256) == 64
    assert proof.state.methodology.valuation_version == "honest_bounds_ns_v3"
    assert proof.rows.canonical_bytes() == proof.canonical_rows_utf8.encode()
    return proof


def install_v3(*, expired: bool = False, cache: bool = True) -> tuple[store.Checkpoint, AcceptedV3]:
    from ssgrep.accounting.collect.store import ROOT_EVALUATIONS_BLOB as BLOB
    from ssgrep.accounting.readers.identity_migration import prepare
    from tests.accounting.collect.collector_scene import at
    from tests.accounting.ledger.reduce_builders import PROFILE

    proof = accepted_v3()
    before = proof.expired_state if expired else proof.state
    checkpoint = store.Checkpoint(
        PROFILE,
        baseline_ts="2026-10-12T07:55:00Z",
        methodology=before.methodology.hash,
    )
    bootstrap = prepare(PROFILE, checkpoint, at(hour=8))
    assert bootstrap.complete
    collector_state.save_state(
        PROFILE,
        collector_state.State(ledger=before, superseded=proof.superseded),
        before.methodology.hash,
        checkpoint.blobs,
    )
    if cache:
        roots = {}
        for root in sorted({row.root for row in proof.evaluations}):
            rows = [
                {
                    "keys": list(row.keys),
                    "leaves": to_plain(row.leaves),
                    "receipts": to_plain(row.receipts),
                }
                for row in proof.evaluations
                if row.root == root
            ]
            roots[root] = store.write_blob(PROFILE, "root-evaluation", rows)
        checkpoint.blobs[BLOB] = store.write_blob(
            PROFILE,
            BLOB,
            {"schema": 1, "methodology": before.methodology.hash, "roots": roots},
        )
    return checkpoint, proof


def test_retirement_keeps_authentic_v3_wrong_day_rows_without_running_new_math() -> None:
    from ssgrep.accounting.collect.steps import retire_methodology
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3()
    current = replace(proof.state.methodology, valuation_version="honest_bounds_ns_v4")
    retire_methodology(PROFILE, checkpoint, current.hash)
    old = collector_state.load_state(PROFILE, checkpoint, proof.state.methodology.hash)
    assert old.ledger is not None
    restored = VersionedLedger.restore_versioned(old.ledger, old.superseded, old.archives)
    rows = restored.rows()
    original = old.archives["methodology-retired:" + proof.state.methodology.hash]
    assert original.as_rows().canonical_bytes() == proof.rows.canonical_bytes()
    assert rows.namespace == proof.state.methodology.hash
    assert tuple(row.leaves for row in rows.periods) == tuple(
        row.leaves for row in proof.rows.periods
    )
    assert rows.periods and all(not row.recomputable for row in rows.periods)
    assert rows.receipts == proof.rows.receipts
    assert tuple(row.digest for row in original.periods) == tuple(
        row.digest for row in proof.rows.periods
    )
    before = dict(checkpoint.blobs)
    retire_methodology(PROFILE, checkpoint, current.hash)
    assert checkpoint.blobs == before


def test_expired_v3_accepted_totals_survive_cutover_without_source_or_evaluation_cache() -> None:
    from ssgrep.accounting.collect.steps import retire_methodology
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3(expired=True, cache=False)
    current = replace(proof.state.methodology, valuation_version="honest_bounds_ns_v4")
    retire_methodology(PROFILE, checkpoint, current.hash)
    old = collector_state.load_state(PROFILE, checkpoint, proof.state.methodology.hash)
    assert old.ledger is not None
    rows = VersionedLedger.restore_versioned(old.ledger, old.superseded, old.archives).rows()
    assert rows.periods and all(not row.recomputable for row in rows.periods)
    assert tuple(row.leaves for row in rows.periods) == tuple(
        row.leaves for row in proof.expired_rows.periods
    )
    original = old.archives["methodology-retired:" + proof.state.methodology.hash]
    assert original.as_rows().canonical_bytes() == proof.expired_rows.canonical_bytes()


def test_missing_accepted_v3_cache_fails_closed_without_relabeling_or_dropping_history() -> None:
    from ssgrep.accounting.collect.steps import retire_methodology
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3(cache=False)
    current = replace(proof.state.methodology, valuation_version="honest_bounds_ns_v4")
    before = dict(checkpoint.blobs)
    with pytest.raises(CollectError) as caught:
        retire_methodology(PROFILE, checkpoint, current.hash)
    assert caught.value.condition == "methodology_history_unavailable"
    assert checkpoint.blobs == before
    old = collector_state.load_state(PROFILE, checkpoint, proof.state.methodology.hash)
    assert old.ledger == proof.state


def test_retired_v3_scoped_purge_uses_accepted_leaves_without_new_attribution() -> None:
    from ssgrep.accounting.collect.steps import retire_methodology
    from ssgrep.accounting.privacy.purge_state import _namespace
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3()
    current = replace(proof.state.methodology, valuation_version="honest_bounds_ns_v4")
    retire_methodology(PROFILE, checkpoint, current.hash)
    old = collector_state.load_state(PROFILE, checkpoint, proof.state.methodology.hash)
    assert old.ledger is not None
    purged = _namespace(
        collector_state.ledger_blob_name(proof.state.methodology.hash),
        old.ledger,
        old.superseded,
        "2026-10-03",
        frozenset({"root-1"}),
        old.archives,
    )
    rows = purged.rows_after
    assert {leaf.task.root for row in rows.periods for leaf in row.leaves} == {"root-2"}
    assert tuple(row.leaves for row in rows.periods) == tuple(
        tuple(leaf for leaf in row.leaves if leaf.task.root == "root-2")
        for row in proof.rows.periods
    )
    assert rows.periods and all(not row.recomputable for row in rows.periods)
    assert purged.changed


def test_current_replay_selects_one_namespace_instead_of_adding_retired_usage() -> None:
    from ssgrep.accounting.collect.steps import retire_methodology
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3()
    current = replace(proof.state.methodology, valuation_version="honest_bounds_ns_v4")
    retire_methodology(PROFILE, checkpoint, current.hash)
    ledger = VersionedLedger(current)
    ledger.reconcile(proof.inputs, today="2026-10-02")
    collector_state.save_state(
        PROFILE, collector_state.State(ledger=ledger.state()), current.hash, checkpoint.blobs
    )
    active = collector_state.load_state(PROFILE, checkpoint, current.hash)
    old = collector_state.load_state(PROFILE, checkpoint, proof.state.methodology.hash)
    assert active.ledger is not None and old.ledger is not None
    current_rows = VersionedLedger.restore_versioned(active.ledger, active.superseded).rows()
    old_rows = VersionedLedger.restore_versioned(old.ledger, old.superseded, old.archives).rows()
    chosen = select_namespace((old_rows, current_rows), current)
    assert chosen == current_rows
    assert chosen.namespace != old_rows.namespace
    assert {
        row.period: tuple(estimate.credit.value for estimate in rollup(row.leaves, UNIT))
        for row in chosen.periods
    } == {DAY: (0, 0, 0), "2026-10-02": (200, 200, 200)}
    assert tuple(
        estimate.net.value
        for estimate in rollup((leaf for row in chosen.periods for leaf in row.leaves), UNIT)
    ) == (122, 122, 122)
    assert (
        old.archives["methodology-retired:" + proof.state.methodology.hash]
        .as_rows()
        .canonical_bytes()
        == proof.rows.canonical_bytes()
    )


def test_collection_rebuilds_current_method_boundedly_and_recompute_does_not_recount_v3(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.collect.collector import CollectOptions, CycleBudget
    from ssgrep.accounting.privacy.policy import enable_collection
    from tests.accounting.collect import collector_scene as scene
    from tests.accounting.ledger.reduce_builders import PROFILE

    scene.install_key()
    enable_collection(PROFILE)
    checkpoint, proof = install_v3()
    store.commit_checkpoint(PROFILE, checkpoint)
    scene.journal_events(scene.start_marker())
    pending = scene.collect(scene.at(hour=8, minute=5), budget=CycleBudget(seconds=0))
    assert not scene.report(pending).complete
    completed = scene.collect(scene.at(hour=8, minute=6))
    assert scene.report(completed).complete
    scene.write_transcript(
        tmp_path,
        "fresh",
        scene.search_records("fresh", "fresh-first", "fresh-query", (10, 0, 0)),
        scene.at(),
    )
    result = scene.collect(scene.at())
    assert scene.report(result).complete
    committed = scene.committed()
    active = collector_state.load_state(PROFILE, committed, committed.methodology)
    assert active.ledger is not None
    assert active.ledger.methodology.valuation_version == "honest_bounds_ns_v4"
    current = VersionedLedger.restore_versioned(active.ledger, active.superseded, active.archives)
    roots = {leaf.task.root for row in current.rows().periods for leaf in row.leaves}
    assert roots and not roots.intersection({"root-1", "root-2"})
    before = tuple(
        estimate.net.value
        for estimate in rollup(
            (leaf for row in current.rows().periods for leaf in row.leaves if leaf.unit == UNIT),
            UNIT,
        )
    )
    counters = scene.counter_total(scene.current())
    assert counters > 0
    replaced = scene.collect(scene.at(minute=1), CollectOptions(recompute=True))
    assert scene.report(replaced).complete
    checkpoint = scene.committed()
    active = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert active.ledger is not None
    rows = VersionedLedger.restore_versioned(
        active.ledger, active.superseded, active.archives
    ).rows()
    assert (
        tuple(
            estimate.net.value
            for estimate in rollup(
                (leaf for row in rows.periods for leaf in row.leaves if leaf.unit == UNIT), UNIT
            )
        )
        == before
    )
    assert scene.counter_total(scene.current()) == counters
    old = collector_state.load_state(PROFILE, checkpoint, proof.state.methodology.hash)
    archived = old.archives["methodology-retired:" + proof.state.methodology.hash]
    assert archived.as_rows().canonical_bytes() == proof.rows.canonical_bytes()


def test_public_v3_whole_view_reads_accepted_cache_without_current_revaluation() -> None:
    from ssgrep.accounting.views.ledger import load_ledger
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3()
    store.commit_checkpoint(PROFILE, checkpoint)
    data, reason = load_ledger(PROFILE)
    assert reason is None and data is not None
    assert data.methodology == proof.state.methodology.hash
    assert tuple(row.leaves for row in data.rows.periods) == tuple(
        row.leaves for row in proof.rows.periods
    )
    assert data.rows.periods and all(not row.recomputable for row in data.rows.periods)
    assert data.rows.receipts == proof.rows.receipts
    assert data.candidates == proof.state.candidates
    assert data.debits == proof.state.debits


def historical_data():
    from ssgrep.accounting.views.ledger import LedgerData

    proof = accepted_v3()
    return LedgerData(
        generation=1,
        methodology=proof.state.methodology.hash,
        rows=proof.rows,
        doc=collector_state.ReductionDoc(
            session_roots=(("claude", "parent", "root-1"), ("claude", "child", "root-1")),
            session_parents=(("claude", "parent", None), ("claude", "child", "parent")),
            invocation_sessions=(("inv1", "claude", "child"),),
            lookup_schema="lookups_v1",
        ),
        sealed_days=(),
        methodology_spec=proof.state.methodology,
        candidates=proof.state.candidates,
        debits=proof.state.debits,
        evidence_expired_before=None,
    )


def test_partial_retired_session_is_typed_unavailable_instead_of_revalued_or_zero() -> None:
    from ssgrep.accounting.views.filter import evaluate
    from ssgrep.accounting.views.types import build_scope
    from tests.accounting.ledger.reduce_builders import PROFILE

    data = historical_data()
    result = evaluate(
        PROFILE,
        build_scope(session="child", since="2026-10-01", until="2026-10-03"),
        data,
        "accepted-v3",
        "2026-10-12T20:00:00Z",
        50,
        None,
    )
    assert result.estimates_reason == "methodology_retired"
    assert result.scenarios
    assert all(
        quantity.value is None
        for row in result.scenarios
        for _, quantity in (*row.credit, *row.debit, *row.net)
    )
    assert data.candidates and data.debits


def test_retired_lookup_subselection_is_unavailable_without_losing_accepted_receipts() -> None:
    from ssgrep.accounting.views.lookups import project_lookup
    from tests.accounting.collect.test_collect_suites import invocation

    data = historical_data()
    assert data.doc is not None
    row = replace(invocation(1, "2026-10-02T10:00:00Z"), session_id="child")
    inputs = replace(
        data,
        candidates=tuple(replace(c, exposure_ids=("inv1",)) for c in data.candidates),
        debits=tuple(replace(d, exposure_id="inv1") for d in data.debits),
    )
    projected = project_lookup("invocation", "inv1", row, data.doc, inputs)
    assert projected.eligibility == "unavailable"
    assert "methodology_retired" in projected.missing_evidence
    assert projected.scenarios == ()
    assert projected.known_costs == ()
    assert inputs.rows.receipts == data.rows.receipts
    assert inputs.candidates and inputs.debits


def test_pre_cutover_v3_purge_reads_accepted_cache_without_mutating_checkpoint() -> None:
    from ssgrep.accounting.privacy.purge_state import _namespace, _read_namespaces
    from tests.accounting.ledger.reduce_builders import PROFILE

    checkpoint, proof = install_v3()
    before = dict(checkpoint.blobs)
    namespaces = _read_namespaces(PROFILE, checkpoint)
    assert len(namespaces) == 1
    name, state, superseded, archives = namespaces[0]
    purged = _namespace(name, state, superseded, "2026-10-03", frozenset({"root-1"}), archives)
    assert checkpoint.blobs == before
    assert purged.changed
    assert {leaf.task.root for row in purged.rows_after.periods for leaf in row.leaves} == {
        "root-2"
    }
    assert purged.rows_after.periods and all(
        not row.recomputable for row in purged.rows_after.periods
    )
    original = archives["methodology-retired:" + state.methodology.hash]
    assert original.as_rows().canonical_bytes() == proof.rows.canonical_bytes()


"""Canonical receipt records: stable ids, allocations, exclusions, costs, no clocks."""


def make(candidates, debits=(), methodology=METHOD):
    valuation = value_evidence(EvidenceBundle(tuple(candidates), tuple(debits)), methodology)
    return build_receipts(candidates, valuation.allocations, debits, methodology)


def test_two_facts_sharing_a_step_allocate_it_once_and_show_the_same_atom_eligible_twice():
    a, b = candidate("fact-a"), candidate("fact-b")
    receipts = make([a, b])
    ra = receipts[receipt_id_for(RECEIPT_TASK, "fact-a", METHOD)]
    rb = receipts[receipt_id_for(RECEIPT_TASK, "fact-b", METHOD)]
    assert dict(ra.receipt.allocations)["conservative"] == ("step-1",)
    assert dict(rb.receipt.allocations)["conservative"] == ()  # already claimed by fact-a
    assert ra.receipt.atoms == rb.receipt.atoms == ("step-1",)
    assert ra.atom_digests == (("step-1", "d-ev-1"),)
    assert dict(ra.credit_raw)["conservative"] == 400  # chars, undivided
    assert dict(rb.credit_raw)["conservative"] == 0  # credited once, not twice


def test_broader_candidate_does_not_take_another_receipts_lower_scenario_allocation():
    shared = atom("shared-step", 400, origin="shared-source")
    first = candidate(
        "fact-a",
        immediate=(atom("own-step", 100, origin="other-source"),),
        discovery=(atom("broader-step", 600, origin="shared-source"),),
    )
    second = candidate("fact-b", immediate=(shared,))
    receipts = make([first, second])
    a = receipts[receipt_id_for(RECEIPT_TASK, "fact-a", METHOD)]
    b = receipts[receipt_id_for(RECEIPT_TASK, "fact-b", METHOD)]
    assert dict(a.credit_raw) == {"conservative": 100, "central": 300, "optimistic": 300}
    assert dict(b.credit_raw) == dict.fromkeys(("conservative", "central", "optimistic"), 400)
    assert [row.credit.value for row in a.receipt.scenarios] == [25, 75, 75]
    assert [row.credit.value for row in b.receipt.scenarios] == [100, 100, 100]
    assert [
        dict(a.credit_raw)[scenario] + dict(b.credit_raw)[scenario]
        for scenario in ("conservative", "central", "optimistic")
    ] == [500, 700, 700]
    assert all(
        row.net.value == row.credit.value for row in (*a.receipt.scenarios, *b.receipt.scenarios)
    )


def test_nested_receipt_allocations_keep_raw_source_nanoseconds_and_outward_bounds():
    method = Methodology("time-allocation", (UnitSpec("work_seconds", "capped_gap_v1"),))
    shared = atom(
        "shared-step",
        400,
        origin="shared-source",
        t_start_ms=0,
        t_end_ms=1100,
        actor="shared-actor",
    )
    broader = atom(
        "broader-step",
        600,
        origin="shared-source",
        t_start_ms=0,
        t_end_ms=1600,
        actor="shared-actor",
    )
    own = atom(
        "own-step",
        100,
        origin="other-source",
        t_start_ms=0,
        t_end_ms=700,
        actor="other-actor",
    )
    candidates = [
        candidate("fact-a", immediate=(own,), discovery=(broader,)),
        candidate("fact-b", immediate=(shared,)),
    ]
    receipts = make(candidates, methodology=method)
    a = receipts[receipt_id_for(RECEIPT_TASK, "fact-a", method)]
    b = receipts[receipt_id_for(RECEIPT_TASK, "fact-b", method)]
    assert dict(a.credit_raw) == {
        "conservative": 700_000_000,
        "central": 1_200_000_000,
        "optimistic": 1_200_000_000,
    }
    assert set(dict(b.credit_raw).values()) == {1_100_000_000}
    assert [row.credit.value for row in a.receipt.scenarios] == [0, 1, 1]
    assert [row.bounds.credit_upper.value for row in a.receipt.scenarios] == [1, 2, 2]
    assert all(
        row.credit.value == 1 and row.bounds.credit_upper.value == 2 for row in b.receipt.scenarios
    )


def test_receipt_carries_lineage_identities_locators_and_methodology():
    cand = candidate(
        "c1",
        execution_ids=("n1-0002", "n1-0001"),
        exposure_ids=("exp-2", "exp-1"),
        witness_locators=("claude:s1:7",),
        action_locators=("claude:s1:19",),
    )
    meth = Methodology(
        "m", (CHARS,), frequency_view="freq-2026-10", tokenizer="", pricing="tariff-1"
    )
    record = make([cand], methodology=meth)[receipt_id_for(RECEIPT_TASK, "c1", meth)]
    r = record.receipt
    assert (r.root, r.task_turn, r.revision) == ("root-1", "turn-1", 1)
    assert r.execution_ids == ("n1-0001", "n1-0002") and r.exposure_ids == ("exp-1", "exp-2")
    assert r.candidate_fingerprint == "c1" and r.candidate_type == "path"
    assert r.methodology == meth.hash and r.evidence_available
    assert record.witness_locators == ("claude:s1:7",) and record.action_locators == (
        "claude:s1:19",
    )
    assert (record.frequency_identity, record.pricing_identity) == ("freq-2026-10", "tariff-1")


def test_exclusions_state_why_nothing_was_credited():
    receipts = make(
        [
            candidate("user", user_supplied=True),
            candidate("bad", reuse=Reuse.CONTRADICTED),
            candidate("drift", immediate=(atom("x", 40, status=AtomStatus.DIGEST_CONFLICT),)),
            candidate(
                "gone", immediate=(atom("y", 40, origin="e2", status=AtomStatus.UNAVAILABLE),)
            ),
        ]
    )
    by_fp = {r.receipt.candidate_fingerprint: r.receipt for r in receipts.values()}
    assert by_fp["user"].exclusions == ("candidate:user_supplied",)
    assert by_fp["bad"].exclusions == ("candidate:contradicted",)
    assert by_fp["drift"].exclusions == ("atom:x:digest_conflict",)
    assert by_fp["gone"].exclusions == ("missing:y:unavailable",)
    assert all(dict(r.allocations)["optimistic"] == () for r in by_fp.values())


def test_costs_list_known_and_missing_reasons_for_this_candidates_exposures():
    cand = candidate("c1", exposure_ids=("e1", "e2", "e3"))
    debits = [
        receipt_debit("d1", 160, exposure_id="e1"),
        receipt_debit("d2", None, exposure_id="e2"),
        receipt_debit("d3", 80, exposure_id="e3", partial=True),
        receipt_debit("other", 999, exposure_id="unrelated"),
    ]
    record = make([cand], debits)[receipt_id_for(RECEIPT_TASK, "c1", METHOD)]

    names = [name for name, _ in record.receipt.costs]
    assert names == ["exposure:d1", "exposure:d2", "exposure:d3"]
    assert record.missing_costs == (("exposure:d2", "downstream_exposure_unobserved"),)


def test_receipt_id_is_stable_and_changes_with_task_candidate_or_methodology():
    base = receipt_id_for(RECEIPT_TASK, "c1", METHOD)
    assert base == receipt_id_for(RECEIPT_TASK, "c1", METHOD)
    assert base != receipt_id_for(TaskId("root-1", "turn-2", "main"), "c1", METHOD)
    assert base != receipt_id_for(RECEIPT_TASK, "c2", METHOD)
    assert base != receipt_id_for(RECEIPT_TASK, "c1", Methodology("other", (CHARS,)))
    assert len(base) == 24 and base.isalnum()


def test_canonical_bytes_are_deterministic_clock_free_and_content_free():
    cand = candidate("c1", execution_ids=("n1-0001",))
    first = make([cand])[receipt_id_for(RECEIPT_TASK, "c1", METHOD)]
    second = make([cand])[receipt_id_for(RECEIPT_TASK, "c1", METHOD)]
    assert first.canonical_bytes() == second.canonical_bytes() and first.digest == second.digest
    text = first.canonical_bytes().decode()
    assert json.loads(text)["format"] == "receipt_v2"
    assert "collected_at" not in text and "ts_utc" not in text
    local = first.with_content("SECRET-PATH /home/user/x.py")
    assert local.canonical_bytes() == first.canonical_bytes()  # content is never canonical
    assert "SECRET" not in json.dumps(default_view(local), default=str)
    assert local.content == "SECRET-PATH /home/user/x.py"


def test_digest_changes_when_any_allocation_or_cost_changes():
    plain = make([candidate("c1")])
    changed = make([candidate("c1", immediate=(atom("step-1", 800),))])
    rid = receipt_id_for(RECEIPT_TASK, "c1", METHOD)
    assert plain[rid].digest == plain[rid].digest != changed[rid].digest


def test_expiry_marks_receipt_non_recomputable_and_drops_content():
    record = make([candidate("c1")])[receipt_id_for(RECEIPT_TASK, "c1", METHOD)].with_content(
        "secret"
    )
    expired = record.mark_expired()
    assert not expired.receipt.evidence_available and expired.content is None
    assert ("recomputation", EXPIRED_REASON) in expired.missing_costs
    assert expired.digest != record.digest
    assert expired.receipt.allocations == record.receipt.allocations  # total explanation kept


def test_build_receipt_uses_only_allocations_of_its_own_task_and_unit():
    other_task = TaskId("root-9", "turn-1", "main")
    both = [candidate("c1"), candidate("c1", task=other_task)]
    valuation = value_evidence(EvidenceBundle(tuple(both)), METHOD)
    first = build_receipt(both[0], valuation.allocations, [], METHOD)
    second = build_receipt(both[1], valuation.allocations, [], METHOD)
    assert first.receipt.root == "root-1" and second.receipt.root == "root-9"
    assert dict(first.receipt.allocations)["conservative"] == ("step-1",)
    assert dict(second.receipt.allocations)["conservative"] == ("step-1",)  # separate consumer
    secs = Methodology("t", (UnitSpec("work_seconds", "monotonic_v1"),))
    assert build_receipt(both[0], valuation.allocations, [], secs).receipt.allocations == ()
    assert replace(first.receipt, revision=2).revision == 2


def test_receipt_identity_mismatch_is_unreadable_not_another_receipt(tmp_path: Path):
    root = tmp_path / "data"
    write_receipt(root, "acme")
    path = root / "accounting" / "acme" / "receipts" / f"{RECEIPT_ID}.json"
    original = path.read_bytes()
    raw = json.loads(original)
    raw["receipt_id"] = "rcpt-other"
    path.write_text(json.dumps(raw))
    before = fingerprint(root)
    with pytest.raises(PolicyError) as caught:
        service.read_receipt("acme", RECEIPT_ID)
    assert caught.value.condition == "receipt_unreadable" and caught.value.exit_code == 1
    assert service.read_receipt("personal-local", RECEIPT_ID) is None
    assert fingerprint(root) == before
    path.write_bytes(original)
    assert service.read_receipt("acme", RECEIPT_ID) is not None


def test_stored_receipt_stays_readable_by_the_facade_with_the_stamp(tmp_path: Path) -> None:
    rows = make_ledger().rows()
    publish.write_receipts(PROFILE, rows)
    record = rows.receipts[0]
    stored = json.loads(
        (
            tmp_path / "data" / "accounting" / PROFILE / "receipts" / f"{record.receipt_id}.json"
        ).read_bytes()
    )
    assert stored["format"] == "receipt_v2"
    assert stored["receipt_id"] == record.receipt_id
    view = service.read_receipt(PROFILE, record.receipt_id)
    assert view is not None and view.receipt_id == record.receipt_id


"""Revision, sealing, expiry and purge behavior with quantified boundary examples."""


PERIOD = "2026-11-01"


def early(**kw):
    """100-token credit and 40-token debit on PERIOD, accepted inside the window."""
    return EvidenceBundle(
        (candidate("early", accepted_day="2026-11-02", **kw),),
        (receipt_debit("d-early", 160, accepted_day="2026-11-02"),),
    )


def late_bundle():
    """More evidence for the SAME period, accepted long after the seal day."""
    extra = candidate(
        "late",
        accepted_day="2026-11-20",
        immediate=(atom("late-step", 800, start=5000, origin="ev-late"),),
    )
    return EvidenceBundle((extra,), (receipt_debit("d-late", 400, accepted_day="2026-11-20"),))


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
    ledger.ingest(EvidenceBundle((candidate("c"),), (receipt_debit("d", None),)))
    (record,) = history(ledger.rows().periods, ledger.namespace, CHARS, today="2026-11-02")
    assert record.net.basis is Basis.UNAVAILABLE and record.status == "provisional"


def test_methodology_change_is_a_new_namespace_and_never_additive():
    from dataclasses import replace

    other = replace(METHOD, label="test-v2", frequency_view="freq-2")
    old, new = Ledger(METHOD), Ledger(other)
    for current_ledger in (old, new):
        current_ledger.ingest(early())
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
