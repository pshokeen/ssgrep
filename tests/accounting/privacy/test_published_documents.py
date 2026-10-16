"""Published-407 quantities remain original inactive values, never v3 scenarios."""

import hashlib
import json

import pytest

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.cycle_publish import history_document, write_history
from ssgrep.accounting.collect.state import VersionedLedger
from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle, Methodology
from tests.accounting.ledger.test_named_ledger_revisions import DAY, TASK, UNIT, debit

PROFILE = "personal-local"
OLD_NAMESPACE = "a" * 64
ARCHIVE = "identity-archive-native_identity_v2"
ORIGINAL = (
    '[\n {"schema":1,"period":"2026-10-05","revision":1,"status":"closed",'
    '"supersedes":null,"net":{"value":-78,"unit":"context_tokens",'
    '"basis":"estimated","method":"chars_div4_v1","reason":""},'
    '"kind":"original","recomputable":true}\n]\n'
)


def publication(text=ORIGINAL, *, available=True):
    document = {
        "original_utf8": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "kind": "history",
        "source_generation": 1,
        "original_collected_at": "2026-10-06T00:00:00Z",
    }
    history = store.write_blob(PROFILE, "published-history", document) if available else None
    manifest = {
        "schema": 1,
        "methodology": OLD_NAMESPACE,
        "root_mapping_status": "unavailable",
        "segments": [],
        "blobs": {"published-history": history} if history else {},
        "published_documents": {
            "format": "published-session-v1",
            "history_blob": history,
            "receipt_blobs": {},
            "source_generation": 1,
            "original_collected_at": "2026-10-06T00:00:00Z",
            "roots_by_period": {"2026-10-05": ["claude:old"]},
        },
    }
    blobs = {ARCHIVE: store.write_blob(PROFILE, ARCHIVE, manifest)}
    store.commit_checkpoint(PROFILE, store.Checkpoint(PROFILE, blobs=blobs))
    return blobs, history


def test_original_net_is_visible_inactive_and_not_added_to_current_scenarios():
    from ssgrep.accounting.privacy.published import merge_history
    from ssgrep.accounting.views.records import read_history

    blobs, original = publication()
    ledger = VersionedLedger(Methodology("published-original-reader", units=(UNIT,)))
    ledger.reconcile({TASK.root: EvidenceBundle((), (debit("current-native"),))}, today=DAY)
    current = history_document(ledger.rows(), UNIT, DAY)
    merged, warnings = merge_history(PROFILE, current, blobs)
    assert warnings == ()
    write_history(PROFILE, merged)
    result = read_history(PROFILE)
    old = next(row for row in result.entries if row.namespace == OLD_NAMESPACE)
    assert old.net.value == -78 and old.net.method == "chars_div4_v1"
    assert old.net.reason == "original_published_untrusted_model"
    assert not old.current and not old.recomputable and old.scenarios == ()
    assert result.current_total.value == -39
    archived = store.read_blob(PROFILE, original)
    assert archived is not None
    assert archived["original_utf8"] == ORIGINAL


def test_original_history_remains_readable_before_republication_and_after_file_loss():
    from ssgrep.accounting.privacy import storage
    from ssgrep.accounting.views.records import read_history

    publication()
    write_history(PROFILE, json.loads(ORIGINAL))
    first = read_history(PROFILE)
    assert len(first.entries) == 1 and first.entries[0].net.value == -78
    assert not first.entries[0].current and first.current_total.value is None
    directory = storage.open_profile_dir(PROFILE, "history", create=False)
    try:
        storage.delete_file(directory, "periods.json")
    finally:
        import os

        os.close(directory)
    assert read_history(PROFILE).entries == first.entries


def test_missing_original_history_is_unavailable_not_zero():
    from ssgrep.accounting.views.records import read_history

    publication(available=False)
    result = read_history(PROFILE)
    assert result.entries == () and result.current_total.value is None
    assert result.reason == "published_history_unavailable"


def test_pending_capture_and_committed_document_refs_survive_orphan_collection():
    from ssgrep.accounting.collect.store import referenced_files

    blobs, history = publication()
    manifest = store.read_blob(PROFILE, blobs[ARCHIVE])
    assert manifest is not None
    # The typed publication map is authority even when it is not duplicated in blobs.
    manifest["blobs"] = {}
    blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    capture = store.write_blob(
        PROFILE,
        "published-capture",
        {
            **manifest["published_documents"],
            "source_namespace": OLD_NAMESPACE,
            "after": "",
            "complete": False,
        },
    )
    checkpoint = store.Checkpoint(
        PROFILE, blobs=blobs, identity_migration={"published_capture": capture}
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    _, kept = referenced_files(PROFILE, checkpoint)
    assert {capture, history} <= kept
    store.sweep_orphans(PROFILE, checkpoint)
    archived = store.read_blob(PROFILE, history)
    assert archived is not None
    assert archived["original_utf8"] == ORIGINAL


def test_published_reference_maps_fail_closed_without_accepting_malformed_references():
    import pytest

    from ssgrep.accounting.collect.collector import CollectError
    from ssgrep.accounting.privacy.published import references

    valid = {
        "format": "published-session-v1",
        "history_blob": "history-1",
        "receipt_blobs": {"receipt-1": "receipt-blob-1"},
    }
    assert references(valid) == {"history-1", "receipt-blob-1"}
    for malformed in (
        {"format": "wrong", "history_blob": None, "receipt_blobs": {}},
        {"format": "published-session-v1", "history_blob": [], "receipt_blobs": {}},
        {"format": "published-session-v1", "receipt_blobs": []},
        {
            "format": "published-session-v1",
            "history_blob": None,
            "receipt_blobs": {"receipt-1": None},
        },
    ):
        with pytest.raises(CollectError) as caught:
            references(malformed)
        assert caught.value.condition == "state_corrupt"


def test_original_document_digest_tampering_is_rejected_without_rewriting_archive():
    import pytest

    from ssgrep.accounting.privacy.published import read_document

    blobs, original = publication()
    before = store.read_blob(PROFILE, original)
    assert before is not None
    corrupted = {**before, "sha256": "0" * 64}
    replacement = store.write_blob(PROFILE, "published-history", corrupted)
    manifest = store.read_blob(PROFILE, blobs[ARCHIVE])
    assert manifest is not None
    manifest["published_documents"]["history_blob"] = replacement
    blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    store.commit_checkpoint(PROFILE, store.Checkpoint(PROFILE, blobs=blobs))
    unchanged = store.read_blob(PROFILE, replacement)

    with pytest.raises(ValueError, match="digest"):
        read_document(PROFILE, replacement, "history")

    assert store.read_blob(PROFILE, replacement) == unchanged


@pytest.mark.parametrize("namespace", [OLD_NAMESPACE, None])
def test_stored_history_recovery_uses_committed_capture_or_reports_unavailable(namespace):
    from ssgrep.accounting.privacy.published import merge_stored_history

    _, history = publication()
    capture = store.write_blob(
        PROFILE,
        "published-capture",
        {
            "format": "published-session-v1",
            "source_namespace": namespace,
            "history_blob": history,
            "receipt_blobs": {},
        },
    )
    checkpoint = store.Checkpoint(PROFILE, identity_migration={"published_capture": capture})
    store.commit_checkpoint(PROFILE, checkpoint)

    rows, warnings = merge_stored_history(PROFILE, [])

    if namespace is None:
        assert rows == [] and warnings == ("published_history_unavailable",)
    else:
        assert warnings == ()
        assert len(rows) == 1
        assert rows[0]["namespace"] == OLD_NAMESPACE
        assert rows[0]["net"]["value"] == -78
        assert rows[0]["current"] is False


def test_original_history_requires_committed_namespace_and_valid_envelope_kind():
    import pytest

    from ssgrep.accounting.privacy.published import merge_history, read_document

    blobs, history = publication()
    manifest = store.read_blob(PROFILE, blobs[ARCHIVE])
    assert manifest is not None
    original = store.read_blob(PROFILE, history)
    assert original is not None

    with pytest.raises(ValueError, match="namespace"):
        merge_history(
            PROFILE,
            [],
            {
                **blobs,
                ARCHIVE: store.write_blob(
                    PROFILE,
                    ARCHIVE,
                    {**manifest, "methodology": None},
                ),
            },
        )
    with pytest.raises(ValueError, match="envelope"):
        read_document(PROFILE, history, "receipt")
    assert store.read_blob(PROFILE, history) == original


def test_unreadable_committed_capture_is_wrapped_as_history_unavailable():
    import pytest

    from ssgrep.accounting.privacy.published import merge_stored_history

    checkpoint = store.Checkpoint(
        PROFILE, identity_migration={"published_capture": "missing-capture-reference"}
    )
    store.commit_checkpoint(PROFILE, checkpoint)

    with pytest.raises(ValueError, match="state cannot be read"):
        merge_stored_history(PROFILE, [])

    assert store.load_checkpoint(PROFILE) == checkpoint


def test_archived_receipt_identifiers_must_be_safe_before_restoration():
    import pytest

    from ssgrep.accounting.privacy import storage
    from ssgrep.accounting.privacy.published import archived_receipt_ids, restore_receipts

    blobs, history = publication()
    manifest = store.read_blob(PROFILE, blobs[ARCHIVE])
    assert manifest is not None
    manifest["published_documents"]["receipt_blobs"] = {"../escape": history}
    blobs[ARCHIVE] = store.write_blob(PROFILE, ARCHIVE, manifest)
    store.commit_checkpoint(PROFILE, store.Checkpoint(PROFILE, blobs=blobs))
    before = store.load_checkpoint(PROFILE)

    with pytest.raises(ValueError, match="identity"):
        archived_receipt_ids(PROFILE, blobs)
    with pytest.raises(ValueError, match="identity"):
        restore_receipts(PROFILE, blobs, set())

    assert store.load_checkpoint(PROFILE) == before
    assert not (storage.data_root() / "accounting" / PROFILE / "receipts").exists()
