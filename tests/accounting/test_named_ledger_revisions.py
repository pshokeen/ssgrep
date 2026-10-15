"""Explicit same-day corrections preserve history without adding historical debit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from ssgrep.accounting import collector_state, collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import Basis, Quantity, TaskId
from ssgrep.accounting.plain_codec import from_plain, loads, to_plain
from ssgrep.accounting.receipts import ReceiptRecord
from ssgrep.accounting.replay import LedgerRows, LedgerState, select_namespace
from ssgrep.accounting.units import CHARS_DIV4_V1, CONTEXT_TOKENS, UnitSpec
from ssgrep.accounting.valuation import Leaf, rollup
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


def test_current_daily_ownership_replay_cannot_reuse_the_v3_namespace() -> None:
    from ssgrep.accounting.valuation_inputs import Candidate
    from tests.accounting.valuation_builders import atom, candidate

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
    from ssgrep.accounting.valuation import value_evidence

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
    document = loads(Path(__file__).with_name("v3_retirement_fixture.json").read_bytes())
    proof = from_plain(AcceptedV3, document)
    assert proof.source_commit == "2f0f620dcf212381e179a4fffd52bd4c277034fb"
    assert proof.source_valuation == "src/ssgrep/accounting/valuation.py"
    assert len(proof.source_valuation_sha256) == 64
    assert proof.state.methodology.valuation_version == "honest_bounds_ns_v3"
    assert proof.rows.canonical_bytes() == proof.canonical_rows_utf8.encode()
    return proof


def install_v3(*, expired: bool = False, cache: bool = True) -> tuple[store.Checkpoint, AcceptedV3]:
    from ssgrep.accounting.collector_root_cache import BLOB
    from ssgrep.accounting.identity_migration import prepare
    from tests.accounting.collector_scene import at
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.collector_state import retire_methodology
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.collector_state import retire_methodology
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.collector_state import retire_methodology
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.collector_state import retire_methodology
    from ssgrep.accounting.purge_state import _namespace
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.collector_state import retire_methodology
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.collector_types import CollectOptions, CycleBudget
    from ssgrep.accounting.policy import enable_collection
    from tests.accounting import collector_scene as scene
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.service_ledger import load_ledger
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.service_ledger import LedgerData

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
    from ssgrep.accounting.scope import build_scope
    from ssgrep.accounting.service_filter import evaluate
    from tests.accounting.reduce_builders import PROFILE

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
    from ssgrep.accounting.lookup_projection import project_lookup
    from tests.accounting.test_collect_counters import invocation

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
    from ssgrep.accounting.purge_state import _namespace, _read_namespaces
    from tests.accounting.reduce_builders import PROFILE

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
