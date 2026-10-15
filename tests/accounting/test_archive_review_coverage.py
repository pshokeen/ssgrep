"""Cross-format archive erasure with real canonical rows and accepted original JSON."""

import json

import pytest

from ssgrep.accounting import collector_store as store, purge, storage
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import ReductionDoc, State, load_state, save_state
from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.published_documents import read_document, restore_receipts
from ssgrep.accounting.published_retention import stage_children
from ssgrep.accounting.purge_select import Selection
from ssgrep.accounting.valuation import rollup
from ssgrep.accounting.valuation_inputs import EvidenceBundle, Methodology
from tests.accounting.test_published_archive_purge import seed
from tests.accounting.test_published_documents import ARCHIVE, PROFILE
from tests.accounting.valuation_builders import CHARS, atom, candidate, debit


def typed_archive():
    method = Methodology("archive-root-erasure", units=(CHARS,))
    ledger = VersionedLedger(method)
    desired = {}
    for identity, root, day in (
        ("selected", "claude:alpha", "2026-10-13"),
        ("other", "claude:beta", "2026-10-14"),
        ("boundary", "claude:alpha", "2026-10-15"),
    ):
        task = TaskId(root, "turn:" + identity)
        item = candidate(
            identity,
            task=task,
            day=day,
            immediate=(atom("atom:" + identity, 400, origin="evidence:" + identity),),
        )
        prior = desired.get(root, EvidenceBundle())
        desired[root] = EvidenceBundle(
            (*prior.candidates, item),
            (*prior.debits, debit("cost:" + identity, 156, task=task, day=day)),
        )
    ledger.reconcile(desired, today="2026-10-15")
    ledger.archive_current("typed-before-erasure")
    return method, ledger.archives_state()


def test_scoped_erasure_commits_typed_receipt_roots_and_preserves_other_original_publication():
    checkpoint, _, original = seed()
    method, archives = typed_archive()
    before = archives["typed-before-erasure"]
    assert {(row.receipt.root, row.period) for row in before.receipts} == {
        ("claude:alpha", "2026-10-13"),
        ("claude:beta", "2026-10-14"),
        ("claude:alpha", "2026-10-15"),
    }
    # No active candidate/debit can supply an archived receipt's root or value.
    state = State(
        ledger=VersionedLedger(method).state(),
        archives=archives,
        reduction=ReductionDoc(
            session_roots=(("claude", "alpha", "claude:alpha"), ("claude", "beta", "claude:beta")),
            session_cwd=(("claude", "alpha", "/alpha"), ("claude", "beta", "/beta")),
        ),
    )
    checkpoint.methodology = method.hash
    save_state(PROFILE, state, method.hash, checkpoint.blobs)
    store.commit_checkpoint(PROFILE, checkpoint)
    restore_receipts(PROFILE, checkpoint.blobs, set())
    public = storage.data_root() / "accounting" / PROFILE / "receipts"
    selected_file = public / "original-0.json"
    other_file = public / "original-1.json"
    assert selected_file.exists()
    other_bytes = other_file.read_bytes()
    other_ref = original["published_documents"]["receipt_blobs"]["original-1"]
    _, other_receipt = read_document(PROFILE, other_ref, "receipt")
    assert other_receipt["root"] == "claude:beta" and other_receipt["period"] == "2026-10-14"

    purge.execute(Selection(PROFILE, "2026-10-15T00:00:00Z", "/alpha"))

    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    recovered = load_state(PROFILE, restarted, method.hash)
    assert recovered.ledger is not None
    assert recovered.ledger.candidates == () and recovered.ledger.debits == ()
    surviving = recovered.archives["typed-before-erasure"]
    assert {(row.receipt.root, row.period) for row in surviving.receipts} == {
        ("claude:beta", "2026-10-14"),
        ("claude:alpha", "2026-10-15"),
    }
    for row in surviving.receipts:
        assert row.canonical_bytes() == next(
            old.canonical_bytes() for old in before.receipts if old.receipt_id == row.receipt_id
        )
        assert dict(row.credit_raw) == {"conservative": 400, "central": 400, "optimistic": 400}
    values = rollup([leaf for period in surviving.periods for leaf in period.leaves], CHARS)
    assert tuple(value.net.value for value in values) == (122, 122, 122)
    assert not selected_file.exists() and other_file.read_bytes() == other_bytes
    archived = store.read_blob(PROFILE, restarted.blobs[ARCHIVE])
    assert archived is not None
    assert archived["published_documents"]["receipt_blobs"] == {"original-1": other_ref}
    _, history = read_document(PROFILE, archived["published_documents"]["history_blob"], "history")
    assert [(row["period"], row["net"]["value"]) for row in history] == [("2026-10-14", -39)]
    ledger = store.read_blob(PROFILE, archived["blobs"]["ledger-original"])
    assert ledger is not None
    assert [(row["task"]["root"], row["amount"]["value"]) for row in ledger["state"]["debits"]] == [
        ("claude:beta", 39)
    ]


@pytest.mark.parametrize("receipt_id", ["original-0", "original-1"])
def test_unindexed_receipt_expiry_reads_selected_original_and_preserves_surviving_values(
    receipt_id,
):
    checkpoint, _, manifest = seed()
    old_ref = manifest["published_documents"]["receipt_blobs"]["original-0"]
    kept_ref = manifest["published_documents"]["receipt_blobs"]["original-1"]
    state_dir = storage.data_root() / "accounting" / PROFILE / "collector" / "state"
    kept_bytes = (state_dir / kept_ref).read_bytes()
    result = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        receipt_ids=(receipt_id, "already-absent"),
    )
    assert result.changed and result.manifest is not None
    documents = result.manifest["published_documents"]
    expected_periods = {"original-1": "2026-10-14"} if receipt_id == "original-1" else {}
    assert documents["receipt_periods"] == expected_periods
    if receipt_id == "original-0":
        assert result.expired_receipts == ("original-0",)
        assert documents["receipt_blobs"] == {"original-1": kept_ref}
        assert documents["receipt_min_period"] is None  # The unselected body was not indexed.
    else:
        assert result.expired_receipts == ()
        assert documents["receipt_roots"] == {"original-1": "claude:beta"}
        assert documents["receipt_blobs"] == manifest["published_documents"]["receipt_blobs"]
    assert (state_dir / kept_ref).read_bytes() == kept_bytes
    _, survivor = read_document(PROFILE, kept_ref, "receipt")
    assert (survivor["receipt_id"], survivor["period"], survivor["root"]) == (
        "original-1",
        "2026-10-14",
        "claude:beta",
    )
    # Staging alone cannot retire the committed original; only the new checkpoint can.
    assert (state_dir / old_ref).exists()
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    assert (state_dir / old_ref).exists() is (receipt_id != "original-0")
    assert (state_dir / kept_ref).read_bytes() == kept_bytes


def test_restoring_original_receipt_refuses_current_identity_collision_without_overwriting():
    checkpoint, _, manifest = seed()
    original_ref = manifest["published_documents"]["receipt_blobs"]["original-0"]
    envelope, original = read_document(PROFILE, original_ref, "receipt")
    restore_receipts(PROFILE, checkpoint.blobs, set())
    path = storage.data_root() / "accounting" / PROFILE / "receipts" / "original-0.json"
    # Matching immutable bytes are safe even when a current publication owns the ID.
    assert restore_receipts(PROFILE, checkpoint.blobs, {"original-0"}) == 0
    assert path.read_bytes() == envelope["original_utf8"].encode()
    current = {**original, "revision": 2, "root": "claude:current"}
    current_bytes = (json.dumps(current) + "\n").encode()
    path.write_bytes(current_bytes)
    with pytest.raises(ValueError, match="identities conflict"):
        restore_receipts(PROFILE, checkpoint.blobs, {"original-0"})
    assert path.read_bytes() == current_bytes
    assert store.load_checkpoint(PROFILE) == checkpoint
    _, retained = read_document(PROFILE, original_ref, "receipt")
    assert retained["root"] == "claude:alpha" and retained["revision"] == 1


def test_postcommit_original_cleanup_protects_current_and_typed_archive_receipt_authority():
    from ssgrep.accounting import collector_published_retention as maintenance
    from ssgrep.accounting.collector_publish import write_receipts

    checkpoint, _, manifest = seed()
    restore_receipts(PROFILE, checkpoint.blobs, set())
    expired = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        receipt_ids=("original-0",),
    )
    assert expired.expired_receipts == ("original-0",)
    method, archives = typed_archive()
    ledger = VersionedLedger(method)
    current = candidate(
        "current",
        task=TaskId("claude:current", "turn:current"),
        day="2026-10-15",
        immediate=(atom("current-atom", 800, origin="current-evidence"),),
    )
    ledger.reconcile({current.task.root: EvidenceBundle((current,), ())}, today="2026-10-15")
    ledger.archives = archives
    rows = ledger.rows()
    assert len(rows.receipts) == 1 and len(rows.archives[0].receipts) == 3
    write_receipts(PROFILE, rows)
    write_receipts(PROFILE, rows.archives[0].as_rows())
    checkpoint.methodology = method.hash
    save_state(
        PROFILE,
        State(ledger=ledger.state(), archives=ledger.archives_state()),
        method.hash,
        checkpoint.blobs,
    )
    protected_ids = (rows.receipts[0].receipt_id, rows.archives[0].receipts[0].receipt_id)
    public = storage.data_root() / "accounting" / PROFILE / "receipts"
    protected_bytes = {name: (public / (name + ".json")).read_bytes() for name in protected_ids}
    other_bytes = (public / "original-1.json").read_bytes()
    checkpoint.retention[maintenance.KEY] = {
        "cleanup_ids": [*protected_ids, "original-0", "already-absent"]
    }
    store.commit_checkpoint(PROFILE, checkpoint)
    restarted = store.load_checkpoint(PROFILE)
    assert restarted is not None
    blocked = maintenance.prepare(
        PROFILE,
        restarted,
        rows=rows,
        before_day="2026-10-14",
        evidence_before_day="2026-10-14",
        clock=lambda: 0,
        deadline=0,
        byte_budget=1024 * 1024,
    )
    assert blocked.pending and blocked.delete_receipts == ()
    assert (public / "original-0.json").exists()
    assert restarted.retention[maintenance.KEY]["cleanup_ids"] == ["original-0"]
    ready = maintenance.prepare(
        PROFILE,
        restarted,
        rows=rows,
        before_day="2026-10-14",
        evidence_before_day="2026-10-14",
        clock=lambda: 0,
        deadline=1,
        byte_budget=1024 * 1024,
    )
    assert ready.delete_receipts == ("original-0",)
    store.commit_checkpoint(PROFILE, restarted)
    assert maintenance.finish(PROFILE, ready) == 1
    assert not (public / "original-0.json").exists()
    assert {
        name: (public / (name + ".json")).read_bytes() for name in protected_ids
    } == protected_bytes
    assert (public / "original-1.json").read_bytes() == other_bytes
    assert rows.receipts[0].receipt.root == "claude:current"
    assert dict(rows.receipts[0].credit_raw) == {
        "conservative": 800,
        "central": 800,
        "optimistic": 800,
    }


def test_original_frozen_child_expires_selected_dates_without_revaluing_surviving_receipts():
    from ssgrep.accounting.plain_codec import from_plain, to_plain
    from ssgrep.accounting.replay import FrozenRoot, LedgerState

    checkpoint, _, manifest = seed()
    method, archives = typed_archive()
    before = archives["typed-before-erasure"]
    roots = ("claude:alpha", "claude:beta")
    frozen = tuple(
        FrozenRoot(
            root,
            tuple(
                leaf
                for period in before.periods
                for leaf in period.leaves
                if leaf.task.root == root
            ),
            tuple(row.mark_expired() for row in before.receipts if row.receipt.root == root),
        )
        for root in roots
    )
    raw = {"state": to_plain(LedgerState(method, frozen=frozen)), "superseded": {}}
    original_ref = store.write_blob(PROFILE, "ledger-original", raw)
    manifest["blobs"]["ledger-original"] = original_ref
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    store.commit_checkpoint(PROFILE, checkpoint)
    original_publication = manifest["published_documents"]["history_blob"]
    publication_bytes = store.read_blob(PROFILE, original_publication)

    result = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        blob_names=("ledger-original",),
    )

    assert result.changed and result.manifest is not None
    retained_ref = result.manifest["blobs"]["ledger-original"]
    retained = store.read_blob(PROFILE, retained_ref)
    assert retained is not None
    decoded = from_plain(LedgerState, retained["state"])
    assert decoded.candidates == () and decoded.debits == ()
    leaves = tuple(leaf for root in decoded.frozen for leaf in root.leaves)
    receipts = tuple(row for root in decoded.frozen for row in root.receipts)
    assert {(leaf.task.root, leaf.day) for leaf in leaves} == {
        ("claude:alpha", "2026-10-15"),
        ("claude:beta", "2026-10-14"),
    }
    assert {(row.receipt.root, row.period) for row in receipts} == {
        ("claude:alpha", "2026-10-15"),
        ("claude:beta", "2026-10-14"),
    }
    for row in receipts:
        previous = next(
            old for root in frozen for old in root.receipts if old.receipt_id == row.receipt_id
        )
        assert row.canonical_bytes() == previous.canonical_bytes()
        assert not row.receipt.evidence_available
        assert ("recomputation", "evidence_expired") in row.missing_costs
    assert tuple(value.net.value for value in rollup(leaves, CHARS)) == (122, 122, 122)
    index = result.manifest["published_documents"]["retention_child_dates"]["blob:ledger-original"]
    assert index["minimum"] == "2026-10-14" and index["complete"] is True
    assert result.manifest["published_documents"]["history_blob"] == original_publication
    assert store.read_blob(PROFILE, original_publication) == publication_bytes
    store.commit_checkpoint(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    assert store.read_blob(PROFILE, retained_ref) == retained


@pytest.mark.parametrize("expiry", ["whole-profile-child", "scoped-purge"])
def test_original_sealed_aggregate_expiry_respects_unsplittable_workspace_authority(expiry):
    from ssgrep.accounting.windows import SealedDay

    checkpoint, _, manifest = seed()
    old = SealedDay(
        "2026-10-13",
        (("outcome.ok", 2),),
        312,
        0,
        812,
        1,
        measured_service_known_ns=1_234_567,
        measured_service_missing=0,
    )
    boundary = SealedDay(
        "2026-10-14",
        (("outcome.ok", 3),),
        624,
        1,
        1624,
        0,
        measured_service_known_ns=9_876_543,
        measured_service_missing=1,
    )
    raw = {row.day: row.to_dict() for row in (old, boundary)}
    reference = store.write_blob(PROFILE, "sealed", raw)
    manifest["blobs"]["sealed"] = reference
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    store.commit_checkpoint(PROFILE, checkpoint)
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "state"
    original_bytes = (directory / reference).read_bytes()
    if expiry == "whole-profile-child":
        staged = stage_children(
            PROFILE,
            checkpoint,
            name=ARCHIVE,
            manifest=manifest,
            before_day="2026-10-14",
            blob_names=("sealed",),
        )
        assert staged.changed and staged.manifest is not None
        result = staged.manifest
        index = result["published_documents"]["retention_child_dates"]["blob:sealed"]
        assert index["minimum"] == "2026-10-14" and index["complete"] is True
        store.commit_checkpoint(PROFILE, checkpoint)
    else:
        purge.execute(Selection(PROFILE, "2026-10-14T00:00:00Z", "/alpha"))
        checkpoint = store.load_checkpoint(PROFILE)
        assert checkpoint is not None
        result = store.read_blob(PROFILE, checkpoint.blobs[ARCHIVE])
        assert result is not None
    retained = store.read_blob(PROFILE, result["blobs"]["sealed"])
    assert retained is not None
    assert SealedDay.from_dict(retained["2026-10-14"]) == boundary
    assert retained["2026-10-14"]["measured_service_known_ns"] == 9_876_543
    if expiry == "whole-profile-child":
        assert set(retained) == {"2026-10-14"}
    else:
        assert SealedDay.from_dict(retained["2026-10-13"]) == old
        assert (directory / reference).read_bytes() == original_bytes
