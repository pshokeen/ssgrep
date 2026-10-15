"""Purge expands and rewrites sharded derived documents without retaining erased rows."""

from dataclasses import replace

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_state import REDUCTION, ReductionDoc, ledger_blob_name
from ssgrep.accounting.ledger_codec import ledger_state_to_dict
from ssgrep.accounting.plain_codec import to_plain
from ssgrep.accounting.purge_select import Selection
from ssgrep.accounting.purge_state import plan_state, write_purged_state
from tests.accounting.test_minimal_evidence import PROFILE


def test_purge_reads_and_rewrites_sharded_current_and_archived_ledger():
    from ssgrep.accounting import collector_ledger_store
    from ssgrep.accounting.collector_ledger import VersionedLedger
    from ssgrep.accounting.valuation_inputs import EvidenceBundle, Methodology
    from tests.accounting.test_named_ledger_revisions import DAY, TASK, UNIT, debit

    ledger = VersionedLedger(Methodology("private-shard-purge", units=(UNIT,)))
    ledger.reconcile({TASK.root: EvidenceBundle((), (debit("erased-shard-debit"),))}, today=DAY)
    ledger.archive_current("before-purge")
    name = ledger_blob_name(ledger.methodology.hash)
    logical = {
        "state": ledger_state_to_dict(ledger.state()),
        "superseded": ledger.superseded_state(),
        "archives": {key: to_plain(value) for key, value in ledger.archives_state().items()},
        "collection_scope": {},
    }
    filename = collector_ledger_store.write_document(PROFILE, name, logical)
    cp = store.Checkpoint(PROFILE, methodology=ledger.methodology.hash, blobs={name: filename})
    selection = Selection(PROFILE, "2026-10-15T00:00:00Z", None)
    planned = plan_state(PROFILE, selection, cp)
    assert planned is not None
    assert planned.changed
    purged = write_purged_state(PROFILE, planned, selection)
    expanded = collector_ledger_store.expand_document(
        PROFILE, store.read_blob(PROFILE, purged.blobs[name])
    )
    assert expanded["state"]["debits"] == []
    assert expanded["archives"] == {}


def test_purge_rewrites_sharded_reduction_and_keeps_other_workspace_metadata():
    from ssgrep.accounting import collector_assignment_store, collector_reduction_store
    from tests.accounting.test_collect_counters import invocation

    mine = replace(invocation(1, "2026-10-01T00:00:00Z"), cwd="/workspace")
    other = replace(invocation(2, "2026-10-01T00:00:00Z"), cwd="/other", session_id="other")
    logical = ReductionDoc(
        invocations=(mine, other),
        lookup_schema="lookups_v1",
        session_cwd=(("claude", "s", "/workspace"), ("claude", "other", "/other")),
        invocation_sessions=(
            (mine.invocation_id, "claude", "s"),
            (other.invocation_id, "claude", "other"),
        ),
    )
    filename = collector_reduction_store.write_document(PROFILE, REDUCTION, logical.to_dict())
    rows = [
        {
            "invocation_id": row.invocation_id,
            "view_id": "frequency-unknown",
            "unavailable_reason": "frequency_unavailable",
            "profile_scope": PROFILE,
            "revision_name": "",
        }
        for row in (mine, other)
    ]
    child = store.write_blob(PROFILE, "assignment-shard", rows)
    assignments = store.write_blob(
        PROFILE,
        "frequency-assignments",
        {
            "schema": 1,
            "format": "frequency-assignments-v2",
            "shards": {"00": child},
            "unavailable_probe": None,
        },
    )
    cp = store.Checkpoint(
        PROFILE, blobs={REDUCTION: filename, "frequency-assignments": assignments}
    )
    selection = Selection(PROFILE, "2026-10-15T00:00:00Z", "/workspace")
    planned = plan_state(PROFILE, selection, cp)
    assert planned is not None
    assert planned.invocations_removed == 1
    purged = write_purged_state(PROFILE, planned, selection)
    expanded = collector_reduction_store.expand_document(
        PROFILE, store.read_blob(PROFILE, purged.blobs[REDUCTION])
    )
    restored = ReductionDoc.from_dict(expanded)
    assert restored.invocations == (other,)
    assert restored.lookup_schema == "lookups_v1"
    retained = collector_assignment_store.expand_document(
        PROFILE, store.read_blob(PROFILE, purged.blobs["frequency-assignments"])
    )
    assert retained["assignments"] == rows[1:]
