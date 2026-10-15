"""Per-root persistence retains canonical rows and migrates legacy state resumably."""

from ssgrep.accounting import collector_state, collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import State
from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.ledger_codec import ledger_state_to_dict
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.valuation_builders import METHOD, candidate, debit

PROFILE = "personal-local"


def ledger_two_roots():
    ledger = VersionedLedger(METHOD)
    for root in ("root-1", "root-2"):
        task = TaskId(root, "turn", "main")
        ledger.reconcile(
            {root: EvidenceBundle((candidate(root, task=task),), (debit(root, 40, task=task),))},
            today="2026-11-02",
        )
    return ledger


def test_single_root_change_reuses_other_canonical_root_blob():
    ledger = ledger_two_roots()
    before = State(ledger=ledger.state(), superseded=ledger.superseded_state())
    blobs = {}
    collector_state.save_state(PROFILE, before, METHOD.hash, blobs)
    name = collector_state.ledger_blob_name(METHOD.hash)
    first = store.read_blob(PROFILE, blobs[name])
    ledger.reconcile(
        {
            "root-1": EvidenceBundle(
                (candidate("root-1", task=TaskId("root-1", "turn")),),
                (debit("root-1", 80, task=TaskId("root-1", "turn")),),
            )
        },
        today="2026-11-03",
    )
    after = State(ledger=ledger.state(), superseded=ledger.superseded_state())
    collector_state.save_state(PROFILE, after, METHOD.hash, blobs, previous=before)
    second = store.read_blob(PROFILE, blobs[name])
    assert first is not None and second is not None
    assert first["roots"]["root-2"]["blob"] == second["roots"]["root-2"]["blob"]
    assert first["roots"]["root-1"]["blob"] != second["roots"]["root-1"]["blob"]
    checkpoint = store.Checkpoint(PROFILE, blobs=blobs)
    restored = collector_state.load_state(PROFILE, checkpoint, METHOD.hash)
    assert restored.ledger is not None
    actual = VersionedLedger.restore_versioned(restored.ledger, restored.superseded)
    assert actual.rows().canonical_bytes() == ledger.rows().canonical_bytes()
    store.sweep_orphans(PROFILE, checkpoint)
    assert collector_state.load_state(PROFILE, checkpoint, METHOD.hash).ledger == restored.ledger


def test_legacy_root_conversion_is_budgeted_without_losing_closed_rows():
    from ssgrep.accounting.collector_ledger_store import expand_document, save_incremental

    ledger = ledger_two_roots()
    state = State(ledger=ledger.state())
    name = collector_state.ledger_blob_name(METHOD.hash)
    assert state.ledger is not None
    blobs = {
        name: store.write_blob(
            PROFILE,
            name,
            {"state": ledger_state_to_dict(state.ledger), "superseded": {}, "archives": {}},
        )
    }
    remaining = save_incremental(PROFILE, name, state, blobs, previous=state, migration_limit=1)
    assert remaining == 1
    raw = store.read_blob(PROFILE, blobs[name])
    assert raw is not None
    assert raw["base_blob"] and raw["migration_pending_roots"] == 1
    expanded = expand_document(PROFILE, raw)
    assert collector_state.decode_ledger(expanded)[0] == state.ledger
    remaining = save_incremental(PROFILE, name, state, blobs, previous=state, migration_limit=1)
    assert remaining == 0
    final = store.read_blob(PROFILE, blobs[name])
    assert final is not None and final["base_blob"] is None
    restored = collector_state.decode_ledger(
        store.read_blob(PROFILE, blobs[name]), profile=PROFILE
    )[0]
    assert restored == state.ledger
