"""Journal phase identities live in existing immutable blobs, not a giant checkpoint."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from ssgrep.accounting import collector_store as store, journal, journal_retention as retention
from ssgrep.accounting.journal_reader import read_journal
from tests.accounting.test_journal_retention import AGGREGATE, FLOOR, PROFILE, event, segment


@pytest.fixture(autouse=True)
def fresh_writer():
    journal.close_all()
    yield
    journal.close_all()


def record(count=2000):
    keys = [hashlib.sha256(str(n).encode()).hexdigest() for n in range(count)]
    return {
        "version": 1,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "retired": {key: "2026-11-09" for key in keys},
        "bridges": sorted(keys[: count // 2]),
        "proofs": [],
    }


def commit(expanded, generation=1):
    checkpoint = store.Checkpoint(profile=PROFILE, generation=generation)
    checkpoint.retention["journal"] = retention.persist_journal_retention(
        PROFILE, expanded, blobs=checkpoint.blobs
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    return checkpoint


def test_checkpoint_has_bounded_handles_not_phase_id_maps_and_refs_survive_sweep():
    expanded = record()
    checkpoint = commit(expanded)
    persisted = checkpoint.retention["journal"]
    body = json.dumps(checkpoint.to_dict())
    assert "retired" not in persisted and "bridges" not in persisted
    assert len(persisted["identity_shards"]) <= 256 and len(body) < 128 * 1024
    assert all(key not in body for key in expanded["retired"])
    assert retention.journal_retention_refs(persisted) == set(checkpoint.blobs.values())
    store.sweep_orphans(PROFILE, checkpoint)
    loaded = retention.load_journal_retention(PROFILE)
    assert loaded is not None
    assert loaded["retired"] == expanded["retired"] and loaded["bridges"] == expanded["bridges"]


def test_old_inline_checkpoint_migrates_without_losing_floor_or_replay_identities():
    expanded = record(20)
    checkpoint = store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.retention["journal"] = expanded
    store.commit_checkpoint(PROFILE, checkpoint)
    old = retention.load_journal_retention(PROFILE)
    assert old is not None
    assert old["retired"] == expanded["retired"]
    migrated = commit(old, generation=2)
    assert migrated.retention["journal"]["version"] == 2
    restored = retention.load_journal_retention(PROFILE)
    assert restored is not None
    assert restored["retired"] == expanded["retired"]
    assert restored["horizon_day"] == FLOOR and restored["key"] == expanded["key"]


def test_expand_uses_supplied_checkpoint_snapshot_not_newer_mutable_current():
    first = commit(record(4))
    snapshot = deepcopy(first.retention["journal"])
    second = record(8)
    second["horizon_day"] = "2026-11-11"
    commit(second, generation=2)
    expanded = retention.expand_journal_retention(PROFILE, snapshot)
    assert expanded["horizon_day"] == FLOOR and len(expanded["retired"]) == 4


def test_clock_only_prepare_reuses_identity_map_and_existing_blob_handles():
    original = record(20)
    original["bridges"] = []
    commit(original)
    previous = retention.load_journal_retention(PROFILE)
    current = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=previous,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert current is not None and previous is not None
    assert current["retired"] is previous["retired"]
    blobs = {}
    persisted = retention.persist_journal_retention(PROFILE, current, blobs=blobs)
    assert persisted is not None
    assert persisted["identity_shards"] == previous["_identity_shards"]


def test_uncommitted_identity_blobs_cannot_authorize_physical_retirement():
    from ssgrep.accounting.journal_codec import serialize_event

    source = segment(
        "seg-v2-uncommitted-pointer.jsonl", serialize_event(event("old", "2026-11-09"))
    )
    expanded = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    blobs = {}
    retention.persist_journal_retention(PROFILE, expanded, blobs=blobs)
    assert retention.finish_journal_retention(PROFILE, generation=1).status == "no_checkpoint"
    assert source.exists()
    commit(expanded)
    retention.finish_journal_retention(PROFILE, generation=1)
    assert not source.exists()


def test_corrupt_or_escaping_identity_reference_is_rejected_before_source_access():
    escaped = {
        "version": 2,
        "key": "11" * 32,
        "horizon_day": FLOOR,
        "aggregate_horizon_day": AGGREGATE,
        "proofs": [],
        "identity_shards": {"00": "../../original-source.json"},
    }
    with pytest.raises(ValueError, match="identity"):
        retention.expand_journal_retention(PROFILE, escaped)


def test_bounded_proof_batch_stays_pending_then_next_accepted_batch_progresses():
    from ssgrep.accounting.journal_codec import serialize_event

    for n in range(35):
        segment(f"seg-v2-batch-{n:03}.jsonl", serialize_event(event(f"old-{n}", "2026-11-09")))
    expanded = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=None,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(expanded["proofs"]) == 32 and expanded["queued_segments"] == 3
    commit(expanded)
    out = retention.finish_journal_retention(PROFILE, generation=1)
    assert out.status == "pending" and out.removed_segments == 32
    later = retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=retention.load_journal_retention(PROFILE),
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert len(later["proofs"]) == 3 and later["queued_segments"] == 0
    commit(later, generation=2)
    assert retention.finish_journal_retention(PROFILE, generation=2).removed_segments == 3


def test_missing_committed_shard_is_not_hidden_by_warm_parse_cache():
    from ssgrep.accounting import storage
    from ssgrep.accounting.collector_types import CollectError

    checkpoint = commit(record(4))
    retained = retention.load_journal_retention(PROFILE)
    assert retained is not None and retained["retired"]
    name = next(iter(retention.journal_retention_refs(checkpoint.retention["journal"])))
    directory = storage.open_profile_dir(PROFILE, "collector", "state")
    try:
        storage.delete_file(directory, name)
    finally:
        import os

        os.close(directory)
    with pytest.raises(CollectError) as failure:
        retention.load_journal_retention(PROFILE)
    assert failure.value.condition == "state_corrupt"


def test_same_name_shard_corruption_invalidates_cached_identity_payload():
    from ssgrep.accounting import storage
    from ssgrep.accounting.collector_types import CollectError

    checkpoint = commit(record(4))
    retained = retention.load_journal_retention(PROFILE)
    assert retained is not None and retained["retired"]
    name = next(iter(retention.journal_retention_refs(checkpoint.retention["journal"])))
    path = storage.data_root() / "accounting" / PROFILE / "collector" / "state" / name
    body = json.loads(path.read_bytes())
    body["bridges"] = []
    body["retired"] = {}
    path.write_text(json.dumps(body))
    with pytest.raises(CollectError) as failure:
        retention.load_journal_retention(PROFILE)
    assert failure.value.condition == "state_corrupt"
