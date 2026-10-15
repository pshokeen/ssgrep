"""Published-407 quantities remain original inactive values, never v3 scenarios."""

import hashlib
import json

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_publish import history_document, write_history
from ssgrep.accounting.valuation_inputs import EvidenceBundle, Methodology
from tests.accounting.test_named_ledger_revisions import DAY, TASK, UNIT, debit

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
    from ssgrep.accounting.published_documents import merge_history
    from ssgrep.accounting.service_records import read_history

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
    from ssgrep.accounting import storage
    from ssgrep.accounting.service_records import read_history

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
    from ssgrep.accounting.service_records import read_history

    publication(available=False)
    result = read_history(PROFILE)
    assert result.entries == () and result.current_total.value is None
    assert result.reason == "published_history_unavailable"


def test_pending_capture_and_committed_document_refs_survive_orphan_collection():
    from ssgrep.accounting.evidence_references import referenced_files

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
