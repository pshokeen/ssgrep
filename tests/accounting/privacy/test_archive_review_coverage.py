"""Cross-format archive erasure with real canonical rows and accepted original JSON."""

import hashlib
import json

import pytest

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.state import (
    ReductionDoc,
    State,
    VersionedLedger,
    load_state,
    save_state,
)
from ssgrep.accounting.ledger.contracts import TaskId
from ssgrep.accounting.ledger.valuation import rollup
from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle, Methodology
from ssgrep.accounting.privacy import purge, storage
from ssgrep.accounting.privacy.published import read_document, restore_receipts
from ssgrep.accounting.privacy.purge import Selection
from ssgrep.accounting.privacy.retention import stage_children
from tests.accounting.ledger.valuation_builders import CHARS, atom, candidate, debit
from tests.accounting.privacy.test_published_archive_purge import seed
from tests.accounting.privacy.test_published_documents import ARCHIVE, PROFILE


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


def test_receipt_restoration_rejects_archive_reference_with_mismatched_identity() -> None:
    checkpoint, _, manifest = seed()
    receipt_id = "original-0"
    original_ref = manifest["published_documents"]["receipt_blobs"][receipt_id]
    envelope = store.read_blob(PROFILE, original_ref)
    assert envelope is not None
    receipt = json.loads(envelope["original_utf8"])
    receipt["receipt_id"] = "different-id"
    text = json.dumps(receipt) + "\n"
    corrupted_ref = store.write_blob(
        PROFILE,
        "published-original",
        {
            **envelope,
            "original_utf8": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )
    manifest["published_documents"]["receipt_blobs"][receipt_id] = corrupted_ref
    checkpoint.blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    store.commit_checkpoint(PROFILE, checkpoint)

    with pytest.raises(ValueError, match="identity does not match"):
        restore_receipts(PROFILE, checkpoint.blobs, set())

    public = storage.data_root() / "accounting" / PROFILE / "receipts" / f"{receipt_id}.json"
    assert not public.exists()
    assert store.load_checkpoint(PROFILE) == checkpoint


def test_postcommit_original_cleanup_protects_current_and_typed_archive_receipt_authority():
    from ssgrep.accounting.collect import collector as maintenance
    from ssgrep.accounting.collect.cycle_publish import write_receipts

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
    from ssgrep.accounting.collect.replay import FrozenRoot, LedgerState
    from ssgrep.accounting.privacy.storage import from_plain, to_plain

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
    from ssgrep.accounting.views.history import SealedDay

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


def test_retention_indexes_legacy_timestamp_and_reports_unknown_child_incomplete():
    from ssgrep.accounting.privacy import retention

    state = {
        "candidates": [
            {"ts_utc": "2026-10-12T09:30:00Z"},
            {"ts_utc": "2026-10-15T10:00:00Z"},
        ],
        "debits": [],
        "frozen": [],
    }

    assert retention._raw_minimum("ledger-original", {"state": state}) == (
        "2026-10-12",
        True,
    )
    assert retention._raw_minimum("unrecognized-child", {}) == (None, False)


def test_empty_published_retention_request_preserves_committed_archive():
    checkpoint, _, manifest = seed()
    before_checkpoint = store.dumps(checkpoint.to_dict())
    before_manifest = store.dumps(manifest)

    result = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
    )

    assert not result.changed and result.manifest is manifest
    assert store.dumps(checkpoint.to_dict()) == before_checkpoint
    assert store.dumps(manifest) == before_manifest


def test_original_receipt_expiry_rejects_index_entry_with_wrong_body_identity():
    checkpoint, _, manifest = seed()
    receipt_id = "original-0"
    old_ref = manifest["published_documents"]["receipt_blobs"][receipt_id]
    envelope = store.read_blob(PROFILE, old_ref)
    assert envelope is not None
    document = json.loads(envelope["original_utf8"])
    document["receipt_id"] = "different-id"
    text = json.dumps(document) + "\n"
    wrong_ref = store.write_blob(
        PROFILE,
        "published-original",
        {
            **envelope,
            "original_utf8": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )
    manifest["published_documents"]["receipt_blobs"][receipt_id] = wrong_ref
    manifest["published_documents"]["receipt_periods"] = {}
    manifest["published_documents"]["receipt_roots"] = {}
    before_manifest = store.dumps(manifest)

    with pytest.raises(ValueError, match="identity"):
        stage_children(
            PROFILE,
            checkpoint,
            name=ARCHIVE,
            manifest=manifest,
            before_day="2026-10-14",
            receipt_ids=(receipt_id,),
        )

    assert store.dumps(manifest) == before_manifest
    assert store.read_blob(PROFILE, old_ref) == envelope


def test_expiring_published_cache_child_drops_only_disposable_cache_reference():
    checkpoint, _, manifest = seed()
    cache = store.write_blob(PROFILE, "partitions", {"cache": "disposable"})
    manifest["blobs"]["partitions"] = cache

    result = stage_children(
        PROFILE,
        checkpoint,
        name=ARCHIVE,
        manifest=manifest,
        before_day="2026-10-14",
        blob_names=("partitions",),
    )

    assert result.changed and result.manifest is not None
    assert "partitions" not in result.manifest["blobs"]
    assert (
        result.manifest["published_documents"]["receipt_blobs"]
        == manifest["published_documents"]["receipt_blobs"]
    )
    assert store.read_blob(PROFILE, cache) == {"cache": "disposable"}


def test_published_retention_rejects_unsupported_manifest_without_mutation():
    from ssgrep.accounting.privacy.purge import PurgeError

    checkpoint, _, manifest = seed()
    manifest["schema"] = 99
    before = store.dumps(checkpoint.to_dict())
    manifest_before = store.dumps(manifest)

    with pytest.raises(PurgeError) as caught:
        stage_children(
            PROFILE,
            checkpoint,
            name=ARCHIVE,
            manifest=manifest,
            before_day="2026-10-14",
            history=True,
        )

    assert caught.value.condition == "unsupported_schema"
    assert store.dumps(checkpoint.to_dict()) == before
    assert store.dumps(manifest) == manifest_before


def test_receipt_restoration_refuses_manifest_identity_change_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, manifest = seed()
    archive_ref = checkpoint.blobs[ARCHIVE]
    original_read = store.read_blob
    archive_reads = 0

    def changed_after_authorization(profile, reference):
        nonlocal archive_reads
        document = original_read(profile, reference)
        if reference == archive_ref:
            archive_reads += 1
            if archive_reads == 2:
                assert isinstance(document, dict)
                document = dict(document)
                documents = dict(document["published_documents"])
                documents["receipt_blobs"] = {
                    "new-unapproved-id": manifest["published_documents"]["receipt_blobs"][
                        "original-0"
                    ]
                }
                document["published_documents"] = documents
        return document

    monkeypatch.setattr(store, "read_blob", changed_after_authorization)
    before = store.dumps(checkpoint.to_dict())

    with pytest.raises(ValueError, match="identity"):
        restore_receipts(PROFILE, checkpoint.blobs, set())

    public = storage.data_root() / "accounting" / PROFILE / "receipts"
    assert archive_reads == 2
    assert not public.exists() or not list(public.iterdir())
    assert store.dumps(checkpoint.to_dict()) == before


def test_stale_receipt_scan_preserves_committed_archive_ids_and_returns_orphans():
    from ssgrep.accounting.privacy.purge_files import stale_receipts

    checkpoint, _, manifest = seed()
    assert restore_receipts(PROFILE, checkpoint.blobs, set()) == 2
    public = storage.data_root() / "accounting" / PROFILE / "receipts"
    (public / "orphan-receipt.json").write_bytes(b"orphan")

    stale_with_checkpoint = stale_receipts(PROFILE, None, checkpoint=checkpoint)
    stale_from_committed_checkpoint = stale_receipts(PROFILE, None)

    assert stale_with_checkpoint == {"orphan-receipt"}
    assert stale_from_committed_checkpoint == stale_with_checkpoint
    assert all(
        (public / f"{receipt_id}.json").is_file()
        for receipt_id in manifest["published_documents"]["receipt_blobs"]
    )
