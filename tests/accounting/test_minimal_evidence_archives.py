"""Identity migration archives cannot evade scoped deletion or orphan recovery."""

from dataclasses import replace

import pytest

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_state import REDUCTION, ReductionDoc
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.purge_select import Selection
from ssgrep.accounting.purge_state import plan_state, write_purged_state
from tests.accounting.test_minimal_evidence import PROFILE, evidence


def archived_checkpoint(*, mapping="available"):
    mine = replace(
        evidence(),
        record_id="private-erased-native-id",
        origin_id="private-erased-native-id",
        session_id="mine",
        payload={"cwd": "/workspace", "text": "private-prompt"},
    )
    other = replace(
        evidence(),
        record_id="retained-native-id",
        origin_id="retained-native-id",
        session_id="other",
        payload={"cwd": "/other", "text": "other-prompt"},
    )
    refs = store.write_segments(PROFILE, [mine, other])
    reduction = ReductionDoc(
        session_roots=(("claude", "mine", "claude:mine"), ("claude", "other", "claude:other")),
        session_cwd=(("claude", "mine", "/workspace"), ("claude", "other", "/other")),
    )
    blobs = {REDUCTION: store.write_blob(PROFILE, REDUCTION, reduction.to_dict())}
    name = "identity-archive-native_identity_v2"
    manifest = {
        "schema": 1,
        "methodology": "",
        "root_mapping_status": mapping,
        "segments": [ref.to_dict() for ref in refs],
        "blobs": blobs,
    }
    cp = store.Checkpoint(
        PROFILE,
        blobs={name: store.write_blob(PROFILE, name, manifest)},
        identity_method="native_identity_v2",
        identity_migration={
            "status": "complete",
            "target": "native_identity_v2",
            "superseded_blobs": blobs,
        },
    )
    store.commit_checkpoint(PROFILE, cp)
    return cp, refs, name


def test_orphan_sweep_preserves_manifest_reachable_segments_and_blobs():
    cp, refs, _ = archived_checkpoint()
    store.sweep_orphans(PROFILE, cp)
    assert len(list(store.iter_evidence(PROFILE, refs))) == 2
    for filename in cp.identity_migration["superseded_blobs"].values():
        assert store.read_blob(PROFILE, filename) is not None


def test_scoped_purge_filters_legacy_archive_and_all_alias_pointers():
    cp, _, name = archived_checkpoint()
    sel = Selection(PROFILE, "2026-10-15T00:00:00Z", "/workspace")
    planned = plan_state(PROFILE, sel, cp)
    assert planned is not None
    assert planned.changed
    new = write_purged_state(PROFILE, planned, sel)
    store.commit_checkpoint(PROFILE, new)
    store.sweep_orphans(PROFILE, new)
    manifest = store.read_blob(PROFILE, new.blobs[name])
    assert manifest is not None
    records = list(
        store.iter_evidence(
            PROFILE, [store.SegmentRef.from_dict(ref) for ref in manifest["segments"]]
        )
    )
    assert [record.record_id for record in records] == ["retained-native-id"]
    assert new.identity_migration["superseded_blobs"] == manifest["blobs"]


def test_unknown_legacy_mapping_blocks_scoped_purge_but_not_whole_profile():
    cp, _, _ = archived_checkpoint(mapping="unavailable")
    with pytest.raises(PurgeError) as failed:
        plan_state(PROFILE, Selection(PROFILE, "2026-10-15T00:00:00Z", "/workspace"), cp)
    assert failed.value.condition == "archive_scope_unresolved"
    assert store.load_checkpoint(PROFILE) == cp
    sel = Selection(PROFILE, "2026-10-15T00:00:00Z", None)
    planned = plan_state(PROFILE, sel, cp)
    assert planned is not None
    new = write_purged_state(PROFILE, planned, sel)
    assert not any(name.startswith("identity-archive-") for name in new.blobs)
    assert new.identity_migration["superseded_blobs"] == {}
