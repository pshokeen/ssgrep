"""Scoped erasure removes caches and shared rarity authority without source mutation."""

import os
from dataclasses import replace

from ssgrep.accounting import collector_store as store, policy, storage
from ssgrep.accounting.evidence_cache import cache_evidence
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.purge_derived import purge_derived
from ssgrep.accounting.purge_select import Selection
from tests.accounting.test_minimal_evidence import CANARY, PROFILE, evidence


def test_scoped_purge_erases_selected_cache_and_all_superseded_vocab(tmp_path):
    selected = str(tmp_path / "workspace")
    other = str(tmp_path / "other")
    p = ProfilePolicy(
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=8192,
    )
    policy.save_profile(p)
    mine = replace(evidence(), payload={**evidence().payload, "cwd": selected})
    theirs = replace(
        evidence(),
        record_id="call:b",
        origin_id="call:b",
        payload={"cwd": other, "output": "other-private-canary"},
    )
    cache_evidence(PROFILE, mine, p)
    cache_evidence(PROFILE, theirs, p)
    fd = storage.open_profile_dir(PROFILE, "collector")
    try:
        for name in (
            "frequency.json",
            "frequency-old.json",
            "frequency-new.json",
            ".frequency.private.tmp",
        ):
            storage.atomic_write(fd, name, CANARY.encode())
    finally:
        os.close(fd)
    result = purge_derived(Selection(PROFILE, "2026-10-15T00:00:00Z", selected))
    assert result.cache_files == 1
    assert result.frequency_files == 4
    root = storage.data_root() / "accounting" / PROFILE
    data = b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    assert CANARY.encode() not in data
    assert b"other-private-canary" in data
    assert (root / "collector" / "frequency-purged.json").is_file()


def test_full_empty_profile_erases_fingerprint_key():
    from ssgrep.accounting import fingerprint

    fingerprint.load_key(PROFILE, create=True)
    store.commit_checkpoint(PROFILE, store.Checkpoint(PROFILE))
    purge_derived(Selection(PROFILE, "2026-10-15T00:00:00Z", None))
    assert fingerprint.load_key(PROFILE, create=False) is None


def test_scoped_purge_removes_old_verdict_pointers_without_erasing_newer_same_session():
    from ssgrep.accounting.collector_state import ReductionDoc
    from ssgrep.accounting.purge_select import Scopes
    from ssgrep.accounting.purge_state import _reduction_after
    from tests.accounting.test_collect_counters import invocation

    old = replace(invocation(1, "2026-10-01T00:00:00Z"), cwd="/workspace")
    new = replace(invocation(2, "2026-10-14T00:00:00Z"), cwd="/workspace")
    other = replace(invocation(3, "2026-10-01T00:00:00Z"), cwd="/other", session_id="other")
    doc = ReductionDoc(
        invocations=(old, new, other),
        lookup_schema="lookups_v1",
        invocation_sessions=(
            (old.invocation_id, "claude", "s"),
            (new.invocation_id, "claude", "s"),
            (other.invocation_id, "claude", "other"),
        ),
    )
    result = _reduction_after(
        doc,
        (),
        frozenset(),
        Selection(PROFILE, "2026-10-10T00:00:00Z", "/workspace"),
        Scopes(frozenset({"claude:s"})),
    )
    assert result is not None
    assert result.invocations == (new, other)
    assert result.invocation_sessions == doc.invocation_sessions[1:]
    assert result.lookup_schema == "lookups_v1"


def test_repeated_scoped_erasure_merges_exclusions_and_revokes_old_reset_authority():
    import json

    fd = storage.open_profile_dir(PROFILE, "collector")
    try:
        storage.atomic_write(
            fd,
            "frequency-purged.json",
            store.dumps(
                {
                    "schema": 1,
                    "scoped": True,
                    "erased_workspaces": ["/first"],
                    "erased_generation": "generation-1",
                    "authorized_generation": "generation-2",
                    "authorized_scope_epoch": 3,
                }
            ),
        )
    finally:
        os.close(fd)
    purge_derived(Selection(PROFILE, "2026-10-15T00:00:00Z", "/second"), invalidate_frequency=True)
    path = storage.data_root() / "accounting" / PROFILE / "collector" / "frequency-purged.json"
    marker = json.loads(path.read_bytes())
    assert marker["erased_workspaces"] == ["/first", "/second"]
    assert marker["erased_generation"] == "generation-2"
    assert marker["authorized_generation"] is None
    assert marker["authorized_scope_epoch"] is None


def test_restart_resumes_staged_purge_even_if_checkpoint_was_not_committed():
    from ssgrep.accounting.lease import Lease
    from ssgrep.accounting.purge_derived import recover_pending_purge, stage_erasure

    item = replace(evidence(), payload={"cwd": "/workspace", "text": CANARY})
    cp = store.Checkpoint(PROFILE, segments=store.write_segments(PROFILE, [item]))
    store.commit_checkpoint(PROFILE, cp)
    sel = Selection(PROFILE, "2026-10-15T00:00:00Z", "/workspace")
    with Lease.acquire(PROFILE, "purge"):
        stage_erasure(sel)
        committed = store.load_checkpoint(PROFILE)
        assert committed is not None
        assert committed.segments == cp.segments
        result = recover_pending_purge(PROFILE)
        assert result is not None
        assert result.performed
        committed = store.load_checkpoint(PROFILE)
        assert committed is not None
        assert committed.segments == []
        assert recover_pending_purge(PROFILE) is None


def test_purge_invalidates_partition_journal_and_root_evaluation_cache_pointers():
    from ssgrep.accounting.purge_state import plan_state, write_purged_state

    bases = ("partitions", "journal-cache", "root-evaluations", "frequency-maintenance")
    keys = (
        *bases,
        *(f"{key}-scope-1" for key in bases),
        *(f"{key}-scope-1-scope-2" for key in bases),
    )
    blobs = {
        key: store.write_blob(PROFILE, key, {"private_id": "erased-derived-id"}) for key in keys
    }
    blobs["unrelated-metadata"] = store.write_blob(PROFILE, "unrelated-metadata", {"count": 17})
    eligible = replace(evidence(), payload={"cwd": "/workspace", "call_id": "a"})
    checkpoint = store.Checkpoint(
        PROFILE, blobs=blobs, segments=store.write_segments(PROFILE, [eligible])
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    selection = Selection(PROFILE, "2026-10-15T00:00:00Z", "/workspace")
    planned = plan_state(PROFILE, selection, checkpoint)
    assert planned is not None
    assert planned.changed
    purged = write_purged_state(PROFILE, planned, selection)
    assert all(key not in purged.blobs for key in keys)
    assert purged.blobs["unrelated-metadata"] == checkpoint.blobs["unrelated-metadata"]
    store.commit_checkpoint(PROFILE, purged)
    store.sweep_orphans(PROFILE, purged)
    for key in keys:
        fd = storage.open_profile_dir(PROFILE, "collector", "state", create=False)
        try:
            assert blobs[key] not in storage.list_names(fd)
        finally:
            os.close(fd)
