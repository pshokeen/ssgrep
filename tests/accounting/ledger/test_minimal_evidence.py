"""Private canaries exercise compact evidence, consent and recoverable migration."""

import json
import os
from dataclasses import replace

import pytest

from ssgrep.accounting.collect import store as store
from ssgrep.accounting.collect.replay import EvidenceReplay, evidence_from_line, evidence_line
from ssgrep.accounting.collect.replay_evidence import cache_collected_evidence
from ssgrep.accounting.collect.state import REDUCTION, ReductionDoc, ledger_blob_name
from ssgrep.accounting.ledger.contracts import RawEvidence
from ssgrep.accounting.ledger.ledger_codec import ledger_state_to_dict, to_plain
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.privacy import policy, storage
from ssgrep.accounting.privacy.purge import PurgeError, Selection
from ssgrep.accounting.privacy.purge_files import purge_derived
from ssgrep.accounting.privacy.purge_state import plan_state, write_purged_state
from ssgrep.cli.metrics_json import lifecycle_document, retained_data
from tests.accounting.readers.test_native_source import old_source

PROFILE = "personal-local"
CANARY = "private-prompt-query-args-output-7fd204"


def evidence():
    return RawEvidence(
        "claude",
        "claude:/missing/source.jsonl",
        "main",
        "session",
        "call:a",
        "byte:0",
        "digest",
        "2026-10-14T00:00:00Z",
        "tool_call",
        {
            "arguments": {"query": CANARY},
            "output": CANARY,
            "text": CANARY,
            "cwd": "/workspace",
            "call_id": "a",
        },
        origin_domain="claude:session",
        origin_id="call:a",
        origin_status="native",
    )


def disk_bytes():
    root = storage.data_root() / "accounting"
    return b"".join(p.read_bytes() for p in root.rglob("*") if p.is_file())


def test_default_segments_are_compact_and_preserve_native_origin():
    item = evidence()
    refs = store.write_segments(PROFILE, [item])
    assert CANARY.encode() not in disk_bytes()
    restored = list(store.iter_evidence(PROFILE, refs))[0]
    assert restored.native_key == item.native_key
    assert restored.origin_status == "native"
    assert restored.digest == item.digest
    assert restored.payload["_evidence_format"] == "compact-v1"
    assert restored.payload["cwd"] == "/workspace"
    assert "arguments" not in restored.payload


def test_legacy_origin_is_explicit_not_silently_native():
    raw = json.loads(evidence_line(evidence()))
    for key in ("evidence_format", "origin_domain", "origin_id", "origin_status"):
        raw.pop(key)
    restored = evidence_from_line(json.dumps(raw).encode())
    assert restored.origin_status == "legacy_unverified"
    assert restored.origin_domain is None


def test_cache_opt_in_is_bounded_and_revocation_erases_plaintext():
    from ssgrep.accounting.collect.replay_evidence import maintain_cache

    enabled = ProfilePolicy(
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=4096,
    )
    policy.save_profile(enabled)
    cache_collected_evidence(PROFILE, [evidence()])
    assert CANARY.encode() in disk_bytes()
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    assert sum(p.stat().st_size for p in directory.iterdir()) <= 4096
    policy.save_profile(
        replace(
            enabled,
            evidence_cache_enabled=False,
            evidence_cache_retention_days=0,
            evidence_cache_max_bytes=0,
        )
    )
    maintain_cache(PROFILE)
    assert CANARY.encode() not in disk_bytes()


def test_migration_commits_references_before_removing_legacy_plaintext(monkeypatch):
    from ssgrep.accounting.collect.store import migrate_evidence

    raw = json.loads(evidence_line(evidence()))
    raw.pop("evidence_format")
    raw["payload"] = evidence().payload
    body = json.dumps(raw).encode() + b"\n"
    fd = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        storage.write_once(fd, "seg-legacy.jsonl", body)
    finally:
        os.close(fd)
    checkpoint = store.Checkpoint(
        PROFILE, generation=7, segments=[store.SegmentRef("seg-legacy.jsonl", 1, len(body))]
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    original = store.commit_checkpoint

    def crash(*args):
        raise RuntimeError("before checkpoint")

    monkeypatch.setattr(store, "commit_checkpoint", crash)
    with pytest.raises(RuntimeError):
        migrate_evidence(PROFILE, checkpoint)
    assert CANARY.encode() in disk_bytes()
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    assert committed.generation == 7
    monkeypatch.setattr(store, "commit_checkpoint", original)
    migrated = migrate_evidence(PROFILE, committed)
    assert migrated.generation == 7
    assert CANARY.encode() not in disk_bytes()
    assert len(list(store.iter_evidence(PROFILE, migrated.segments))) == 1


def test_cache_oversize_record_is_not_written_and_expiration_survives_reload():
    from ssgrep.accounting.collect.replay_evidence import maintain_cache

    enabled = ProfilePolicy(
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=2048,
    )
    policy.save_profile(enabled)
    huge = replace(evidence(), payload={"cwd": "/workspace", "output": CANARY * 1024})
    cache_collected_evidence(PROFILE, [huge])
    assert CANARY.encode() not in disk_bytes()
    cache_collected_evidence(PROFILE, [evidence()])
    assert CANARY.encode() in disk_bytes()
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    for path in directory.iterdir():
        os.utime(path, (1, 1))
    maintain_cache(PROFILE, policy.load_profile_policy(PROFILE), now=86402)
    assert CANARY.encode() not in disk_bytes()


def test_retained_usage_measurements_preserve_streaming_and_cost_context():
    payload = {
        "message_id": "m",
        "model": None,
        "provider": "native-provider",
        "tokens": {"input_tokens": 12, "output_tokens": 8, CANARY: 99},
        "cost_nano_usd": 512,
        "streaming_updates": 3,
        "completed_ms": 42,
    }
    item = replace(evidence(), kind="usage", payload=payload)
    restored = evidence_from_line(evidence_line(item))
    assert restored.payload["tokens"] == {"input_tokens": 12, "output_tokens": 8}
    assert restored.payload["model"] is None
    assert restored.payload["cost_nano_usd"] == 512
    assert restored.payload["streaming_updates"] == 3
    assert restored.payload["provider"] == "native-provider"
    assert CANARY.encode() not in evidence_line(item)


def test_cache_needs_separate_consent_and_valid_bounds_before_any_policy_write():
    from ssgrep.accounting.privacy.policy import PolicyError

    with pytest.raises(PolicyError):
        policy.preview_enable(PROFILE, evidence_cache=(0, 4096))
    assert not policy.load_profile_policy(PROFILE).collection_enabled
    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    assert enabled.evidence_cache_enabled
    assert enabled.evidence_cache_retention_days == 1
    with pytest.raises(PolicyError):
        policy.save_profile(replace(enabled, evidence_cache_max_bytes=32 * 1024 * 1024 + 1))
    assert policy.load_profile_policy(PROFILE).evidence_cache_max_bytes == 4096


def test_revocation_crash_restart_uses_persisted_denial_before_erasing_cache(monkeypatch):
    from ssgrep.accounting.collect import replay_evidence as evidence_cache

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    cache_collected_evidence(PROFILE, [evidence()])
    refs = store.write_segments(PROFILE, [evidence()])
    assert CANARY.encode() in disk_bytes()
    maintain = evidence_cache.maintain_cache

    def crash(*args, **kwargs):
        raise RuntimeError("after consent commit")

    monkeypatch.setattr(evidence_cache, "maintain_cache", crash)
    revoked = replace(
        enabled,
        evidence_cache_enabled=False,
        evidence_cache_retention_days=0,
        evidence_cache_max_bytes=0,
    )
    with pytest.raises(RuntimeError):
        policy.save_profile(revoked)
    assert not policy.load_profile_policy(PROFILE).evidence_cache_enabled
    monkeypatch.setattr(evidence_cache, "maintain_cache", maintain)
    maintain(PROFILE)
    assert CANARY.encode() not in disk_bytes()
    assert len(list(store.iter_evidence(PROFILE, refs))) == 1


def test_compact_ownership_claim_retains_conflict_quarantine_without_payload():
    claim = replace(
        evidence(),
        kind="ownership_claim",
        payload={
            "target_native_key": ["claude", "domain", "call:a"],
            "target_digest": "other-digest",
            "target_record_id": "result:copied",
            "call_record_id": "call:known",
            "call_link_status": None,
            "origin_conflict": True,
            "original_kind": "tool_call",
            "claimant_cwd": "/workspace",
            "claimant_parent_session": None,
            "arguments": {"query": CANARY},
            "output": CANARY,
        },
    )
    restored = evidence_from_line(evidence_line(claim))
    assert restored.payload["target_native_key"] == ["claude", "domain", "call:a"]
    assert restored.payload["target_digest"] == "other-digest"
    assert restored.payload["target_record_id"] == "result:copied"
    assert restored.payload["call_record_id"] == "call:known"
    assert restored.payload["call_link_status"] is None
    assert restored.payload["origin_conflict"] is True
    assert restored.payload["claimant_parent_session"] is None
    assert CANARY.encode() not in evidence_line(claim)


def test_cache_revocation_is_linearized_after_inflight_plaintext_writer():
    from threading import Event, Thread

    from ssgrep.accounting.collect.replay_evidence import CacheWriter

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    writer = CacheWriter(PROFILE, enabled)
    writer.add(evidence())
    started, finished = Event(), Event()

    def revoke():
        started.set()
        policy.save_profile(
            replace(
                enabled,
                evidence_cache_enabled=False,
                evidence_cache_retention_days=0,
                evidence_cache_max_bytes=0,
            )
        )
        finished.set()

    thread = Thread(target=revoke)
    thread.start()
    try:
        assert started.wait(1)
        assert not finished.wait(0.1)
    finally:
        writer.close()
        thread.join(timeout=5)
    assert not thread.is_alive() and finished.is_set()
    assert CANARY.encode() not in disk_bytes()


def test_migration_restart_finishes_post_commit_plaintext_cleanup_once(monkeypatch):
    from ssgrep.accounting.collect.store import migrate_evidence

    body = evidence_line(evidence(), plaintext=True) + b"\n"
    fd = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        storage.write_once(fd, "seg-obsolete-plaintext.jsonl", body)
    finally:
        os.close(fd)
    cp = store.Checkpoint(
        PROFILE, segments=[store.SegmentRef("seg-obsolete-plaintext.jsonl", 1, len(body))]
    )
    store.commit_checkpoint(PROFILE, cp)
    sweep = store.sweep_orphans

    def crash(*args):
        raise RuntimeError("after compact checkpoint")

    monkeypatch.setattr(store, "sweep_orphans", crash)
    with pytest.raises(RuntimeError):
        migrate_evidence(PROFILE, cp)
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    assert committed.retention["evidence_cleanup_pending"] is True
    assert CANARY.encode() in disk_bytes()
    monkeypatch.setattr(store, "sweep_orphans", sweep)
    cleaned = migrate_evidence(PROFILE, committed)
    assert CANARY.encode() not in disk_bytes()
    assert "evidence_cleanup_pending" not in cleaned.retention
    monkeypatch.setattr(store, "sweep_orphans", crash)
    assert migrate_evidence(PROFILE, cleaned) is cleaned


def test_opted_in_segment_batch_reads_each_existing_plaintext_cache_file_once(monkeypatch):
    from ssgrep.accounting.collect import replay_evidence as evidence_cache
    from ssgrep.accounting.readers.native_identity import with_origin

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 16384))

    def item(number):
        return with_origin(
            replace(
                evidence(),
                record_id=f"call:{number}",
                payload={**evidence().payload, "output": f"{CANARY}-{number}"},
            ),
            "claude:inventory-test",
            f"call:{number}",
            "session_record",
        )

    first, second, external, last = (item(number) for number in range(4))
    cache_collected_evidence(PROFILE, [first])
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    assert CANARY.encode() in b"".join(path.read_bytes() for path in directory.iterdir())
    identity = directory.stat()
    real_read = storage.read_file
    reads = []

    def measured_read(fd, name, maximum):
        data = real_read(fd, name, maximum)
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino) and data is not None:
            reads.append((name, len(data)))
        return data

    before = {path.name: path.stat().st_size for path in directory.iterdir()}
    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_file", measured_read)
        cache_collected_evidence(PROFILE, [second])
    assert sorted(reads) == sorted(before.items())
    cached = evidence_cache.cached_evidence(PROFILE, second)
    assert cached is not None
    assert cached.payload["output"] == f"{CANARY}-1"
    # A different writer changes the directory between batches: no global/stale inventory.
    cache_collected_evidence(PROFILE, [external])
    reads.clear()
    before = {path.name: path.stat().st_size for path in directory.iterdir()}
    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_file", measured_read)
        cache_collected_evidence(PROFILE, [last])
    assert sorted(reads) == sorted(before.items())
    assert (
        sum(path.stat().st_size for path in directory.iterdir()) <= enabled.evidence_cache_max_bytes
    )
    cached = evidence_cache.cached_evidence(PROFILE, external)
    assert cached is not None
    assert cached.payload["output"] == f"{CANARY}-2"


def test_shared_cache_writer_updates_inventory_without_rereading_its_own_writes(monkeypatch):
    from ssgrep.accounting.collect.replay_evidence import CacheWriter

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    seed = evidence()
    cache_collected_evidence(PROFILE, [seed])
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    expected_bytes = sum(path.stat().st_size for path in directory.iterdir())
    real_read, observed_bytes = storage.read_file, []
    identity = directory.stat()

    def measured_read(fd, name, maximum):
        data = real_read(fd, name, maximum)
        info = os.fstat(fd)
        if data is not None and (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            observed_bytes.append(len(data))
        return data

    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_file", measured_read)
        with CacheWriter(PROFILE, enabled) as writer:
            for number in range(6):
                writer.add(replace(seed, record_id=f"call:{number}", origin_id=f"call:{number}"))
            assert writer.size == sum(path.stat().st_size for path in directory.iterdir())
            assert writer.names == {path.name for path in directory.iterdir()}
            assert writer.size <= enabled.evidence_cache_max_bytes
    assert sum(observed_bytes) == expected_bytes


def test_new_cache_batch_rechecks_revocation_even_with_old_policy_argument():
    from ssgrep.accounting.collect.replay_evidence import CacheWriter

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    cache_collected_evidence(PROFILE, [evidence()])
    policy.save_profile(
        replace(
            enabled,
            evidence_cache_enabled=False,
            evidence_cache_retention_days=0,
            evidence_cache_max_bytes=0,
        )
    )
    with CacheWriter(PROFILE, enabled) as writer:
        writer.add(evidence())
        assert not writer.policy.evidence_cache_enabled
        assert writer.size == 0
    assert CANARY.encode() not in disk_bytes()


def test_cache_batch_expires_entries_without_inventory_content_reread(monkeypatch):
    from ssgrep.accounting.collect import replay_evidence as evidence_cache
    from ssgrep.accounting.privacy import policy, storage

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    cache_collected_evidence(PROFILE, [evidence()])
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    original = next(directory.iterdir())
    with evidence_cache.CacheWriter(PROFILE, enabled) as writer:
        cutoff = original.stat().st_mtime_ns + 86400 * 1_000_000_000 + 1
        with monkeypatch.context() as patch:
            patch.setattr(evidence_cache.time, "time_ns", lambda: cutoff)
            writer.add(
                replace(evidence(), record_id="call:after-expiry", origin_id="call:after-expiry")
            )
        assert not original.exists()
        assert writer.size == sum(path.stat().st_size for path in directory.iterdir())
        assert writer.names == {path.name for path in directory.iterdir()}
    assert CANARY.encode() in b"".join(path.read_bytes() for path in directory.iterdir())


def test_narrowed_cache_cap_trims_metadata_before_reading_remaining_plaintext(monkeypatch):
    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 16384))
    cache_collected_evidence(
        PROFILE,
        [
            replace(evidence(), record_id=f"call:{number}", origin_id=f"call:{number}")
            for number in range(3)
        ],
    )
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    maximum = max(path.stat().st_size for path in directory.iterdir())
    assert len(tuple(directory.iterdir())) == 3
    identity = directory.stat()
    real_read, observed_bytes = storage.read_file, []

    def measured_read(fd, name, bound):
        data = real_read(fd, name, bound)
        info = os.fstat(fd)
        if data is not None and (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            observed_bytes.append(len(data))
        return data

    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_file", measured_read)
        policy.save_profile(replace(enabled, evidence_cache_max_bytes=maximum))
    assert 0 < sum(observed_bytes) <= maximum
    assert sum(path.stat().st_size for path in directory.iterdir()) <= maximum
    assert CANARY.encode() in b"".join(path.read_bytes() for path in directory.iterdir())


def test_compaction_resumes_across_active_and_archived_mirrors_checkpoint_first(tmp_path):
    from ssgrep.accounting.collect.store import migrate_evidence

    native = tmp_path / "original-native.jsonl"
    native.write_bytes(CANARY.encode())
    body = evidence_line(replace(evidence(), source_id=f"claude:{native}"), plaintext=True) + b"\n"
    fd = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        for name in ("seg-old-active.jsonl", "seg-old-archive.jsonl"):
            storage.write_once(fd, name, body)
    finally:
        os.close(fd)
    archived_ref = store.SegmentRef("seg-old-archive.jsonl", 1, len(body))
    archive = "identity-archive-native_identity_v2"
    manifest = {"schema": 1, "segments": [archived_ref.to_dict()], "blobs": {}}
    checkpoint = store.Checkpoint(
        PROFILE,
        segments=[store.SegmentRef("seg-old-active.jsonl", 1, len(body))],
        blobs={archive: store.write_blob(PROFILE, archive, manifest)},
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    first = migrate_evidence(PROFILE, checkpoint)
    assert first.retention["evidence_migration_pending"]
    assert CANARY.encode() in disk_bytes()  # recovery obligation is not falsely complete
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    second = migrate_evidence(PROFILE, committed)
    assert not second.retention.get("evidence_migration_pending")
    assert CANARY.encode() not in disk_bytes()
    assert native.read_bytes() == CANARY.encode()
    assert len(list(store.iter_evidence(PROFILE, second.segments))) == 1
    archived = store.read_blob(PROFILE, second.blobs[archive])
    assert archived is not None
    assert (
        len(
            list(
                store.iter_evidence(
                    PROFILE, [store.SegmentRef.from_dict(ref) for ref in archived["segments"]]
                )
            )
        )
        == 1
    )


def test_compaction_byte_budget_blocks_before_read_and_retries_only_on_larger_budget(monkeypatch):
    from ssgrep.accounting.collect.store import migrate_evidence

    body = evidence_line(evidence(), plaintext=True) + b"\n"
    fd = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        storage.write_once(fd, "seg-budget.jsonl", body)
    finally:
        os.close(fd)
    checkpoint = store.Checkpoint(
        PROFILE, segments=[store.SegmentRef("seg-budget.jsonl", 1, len(body))]
    )
    store.commit_checkpoint(PROFILE, checkpoint)

    def forbidden(*args):
        raise AssertionError("over-budget legacy segment must not be parsed")

    with monkeypatch.context() as instrument:
        instrument.setattr(store, "read_segment", forbidden)
        blocked = migrate_evidence(PROFILE, checkpoint, byte_budget=len(body) - 1)
        assert blocked.retention["evidence_migration_pending"]
        assert blocked.retention["evidence_compaction"]["blocked_child_bytes"] == len(body)
        assert migrate_evidence(PROFILE, blocked, byte_budget=len(body) - 1) is blocked
    complete = migrate_evidence(PROFILE, blocked, byte_budget=len(body))
    assert not complete.retention.get("evidence_migration_pending")
    assert CANARY.encode() not in disk_bytes()


@pytest.mark.parametrize("archived", [False, True])
def test_privacy_compaction_never_renews_expired_plaintext_cache(tmp_path, archived):
    from ssgrep.accounting.collect.store import migrate_evidence

    policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    old = replace(evidence(), ts_utc="2000-01-01T00:00:00Z")
    cache_collected_evidence(PROFILE, [old])
    cache_dir = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    (cached,) = tuple(cache_dir.iterdir())
    assert CANARY.encode() in cached.read_bytes()
    os.utime(cached, (0, 0))
    body = evidence_line(old, plaintext=True) + b"\n"
    directory = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        storage.write_once(directory, "seg-expired-legacy.jsonl", body)
    finally:
        os.close(directory)
    ref = store.SegmentRef("seg-expired-legacy.jsonl", 1, len(body))
    archive = "identity-archive-native_identity_v2"
    checkpoint = store.Checkpoint(PROFILE, generation=7, segments=[] if archived else [ref])
    if archived:
        checkpoint.blobs[archive] = store.write_blob(
            PROFILE, archive, {"schema": 1, "segments": [ref.to_dict()], "blobs": {}}
        )
    store.commit_checkpoint(PROFILE, checkpoint)
    for _ in range(8):
        checkpoint = migrate_evidence(PROFILE, checkpoint)
        if not checkpoint.retention.get("evidence_migration_pending"):
            break
    assert not checkpoint.retention.get("evidence_migration_pending")
    assert CANARY.encode() not in disk_bytes()
    if archived:
        manifest = store.read_blob(PROFILE, checkpoint.blobs[archive])
        assert manifest is not None
        refs = [store.SegmentRef.from_dict(raw) for raw in manifest["segments"]]
    else:
        refs = checkpoint.segments
    (compact,) = tuple(store.iter_evidence(PROFILE, refs))
    assert compact.native_key == old.native_key and compact.digest == old.digest
    assert compact.payload["_evidence_format"] == "compact-v1"
    fresh = replace(evidence(), record_id="call:new", origin_id="call:new")
    cache_collected_evidence(PROFILE, [fresh])
    assert CANARY.encode() in b"".join(path.read_bytes() for path in cache_dir.iterdir())


@pytest.mark.parametrize("boundary", ["root_narrowing", "revocation", "mtime_expiry", "malformed"])
def test_cached_plaintext_privacy_boundaries_do_not_renew_from_compact_replay(
    tmp_path, monkeypatch, boundary
):
    from ssgrep.accounting.collect import replay_evidence as evidence_cache
    from ssgrep.accounting.readers.native_identity import with_origin

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    approved, removed = tmp_path / "approved", tmp_path / "removed"
    approved.mkdir()
    removed.mkdir()
    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    enabled = policy.save_profile(replace(enabled, roots=(str(approved), str(removed))))
    original = with_origin(
        replace(evidence(), payload={**evidence().payload, "cwd": str(removed)}),
        "claude:cache-privacy",
        "call:a",
        "native",
    )
    cache_collected_evidence(PROFILE, [original])
    refs = store.write_segments(PROFILE, [original])
    (compact,) = tuple(store.iter_evidence(PROFILE, refs))
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    (cached_path,) = tuple(directory.iterdir())
    cached = evidence_cache.cached_evidence(PROFILE, compact)
    assert cached is not None and cached.payload["output"] == CANARY
    assert CANARY.encode() in cached_path.read_bytes()
    segment_bytes = {
        ref.name: (
            storage.data_root() / "accounting" / PROFILE / "collector" / "evidence" / ref.name
        ).read_bytes()
        for ref in refs
    }

    if boundary == "root_narrowing":
        policy.save_profile(replace(enabled, roots=(str(approved),)))
    elif boundary == "revocation":
        policy.save_profile(
            replace(
                enabled,
                evidence_cache_enabled=False,
                evidence_cache_retention_days=0,
                evidence_cache_max_bytes=0,
            )
        )
    elif boundary == "mtime_expiry":
        os.utime(cached_path, (0, 0))
    else:
        cached_path.write_bytes(b'{"payload":"' + CANARY.encode() + b'"\n')
    assert evidence_cache.cached_evidence(PROFILE, compact) is None
    assert not cached_path.exists()
    assert CANARY.encode() not in disk_bytes()
    store.write_segments(PROFILE, [compact])
    evidence_cache.maintain_cache(PROFILE)
    assert CANARY.encode() not in disk_bytes()
    assert {
        ref.name: (
            storage.data_root() / "accounting" / PROFILE / "collector" / "evidence" / ref.name
        ).read_bytes()
        for ref in refs
    } == segment_bytes

    if boundary == "revocation":
        policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    fresh = with_origin(
        replace(
            original,
            record_id="call:fresh",
            payload={**original.payload, "cwd": str(approved)},
        ),
        "claude:cache-privacy",
        "call:fresh",
        "native",
    )
    cache_collected_evidence(PROFILE, [fresh])
    fresh_refs = store.write_segments(PROFILE, [fresh])
    (fresh_compact,) = tuple(store.iter_evidence(PROFILE, fresh_refs))
    recovered = evidence_cache.cached_evidence(PROFILE, fresh_compact)
    assert recovered is not None
    assert recovered.native_key == fresh.native_key and recovered.digest == fresh.digest
    assert recovered.payload["output"] == CANARY
    assert CANARY.encode() in b"".join(path.read_bytes() for path in directory.iterdir())


def legacy_segment(items, name="seg-administrative-legacy.jsonl"):
    """An actual pre-format persisted segment, not a compact record relabeled in memory."""
    rows = []
    for item in items:
        raw = json.loads(evidence_line(item, plaintext=True))
        raw.pop("evidence_format")
        rows.append(json.dumps(raw).encode() + b"\n")
    body = b"".join(rows)
    directory = storage.open_profile_dir(PROFILE, "collector", "evidence")
    try:
        assert storage.write_once(directory, name, body)
    finally:
        os.close(directory)
    ref = store.SegmentRef(name, len(rows), len(body))
    assert all(
        item.payload["_evidence_format"] == "legacy-plaintext"
        for item in store.read_segment(PROFILE, ref)
    )
    return ref


def expired_cache(item):
    cache_collected_evidence(PROFILE, [item])
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    (path,) = tuple(directory.iterdir())
    assert CANARY.encode() in path.read_bytes()
    os.utime(path, (0, 0))
    return directory


@pytest.mark.parametrize("operation", ["maintenance", "expiry", "scoped_purge"])
def test_administrative_legacy_rewrites_do_not_readmit_expired_cache(operation):
    from datetime import UTC, datetime

    from ssgrep.accounting.collect.collector import RetentionCoverage
    from ssgrep.accounting.collect.cycle import apply_expiry
    from ssgrep.accounting.collect.replay_evidence import cached_evidence, maintain_cache
    from ssgrep.accounting.collect.state import ReductionDoc, VersionedLedger
    from ssgrep.accounting.collect.steps import maintain_evidence
    from ssgrep.accounting.privacy.purge import Selection
    from ssgrep.accounting.privacy.purge_state import plan_state, write_purged_state
    from tests.accounting.ledger.valuation_builders import METHOD

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    kept = evidence()
    removed = replace(
        kept,
        session_id="expired",
        record_id="call:expired",
        ts_utc="2000-01-01T00:00:00Z",
        payload={"cwd": "/removed", "output": "removed-private"},
    )
    directory = expired_cache(kept)
    ref = legacy_segment([kept, removed])
    checkpoint = store.Checkpoint(PROFILE, generation=1, segments=[ref])
    store.commit_checkpoint(PROFILE, checkpoint)
    if operation == "maintenance":
        refs, job, used, dropped = maintain_evidence(
            PROFILE,
            [ref],
            expired_sessions={("claude", "expired")},
            previous=None,
            clock=lambda: 0,
            deadline=1,
            byte_budget=ref.size,
            should_stop=lambda: False,
        )
        assert dropped == 1 and used == ref.size and job["pending"] == []
    elif operation == "expiry":
        doc = ReductionDoc(
            session_roots=(
                ("claude", "session", "kept-root"),
                ("claude", "expired", "expired-root"),
            ),
        )
        outcome, _ = apply_expiry(
            PROFILE,
            now=datetime(2026, 10, 15, tzinfo=UTC),
            policy=enabled,
            ledger=VersionedLedger(METHOD),
            doc=doc,
            session_last={
                ("claude", "session"): kept.ts_utc,
                ("claude", "expired"): removed.ts_utc,
            },
            already=frozenset(),
            previous=RetentionCoverage(),
            segments=[ref],
            sealed_days={},
        )
        assert outcome.dropped_records == 1 and outcome.segments is not None
        refs = outcome.segments
    else:
        selection = Selection(PROFILE, "2026-10-15T00:00:00Z", "/removed")
        plan = plan_state(PROFILE, selection, checkpoint)
        assert plan is not None and plan.evidence_records == 1
        refs = write_purged_state(PROFILE, plan, selection).segments
    (retained,) = tuple(store.iter_evidence(PROFILE, refs))
    assert retained.session_id == "session"
    assert retained.payload["_evidence_format"] == "compact-v1"
    maintain_cache(PROFILE)
    assert cached_evidence(PROFILE, retained) is None
    assert tuple(directory.iterdir()) == ()


def test_identity_conversion_native_hydration_does_not_readmit_expired_cache(tmp_path, monkeypatch):
    from ssgrep.accounting.collect.replay_evidence import maintain_cache
    from ssgrep.accounting.readers.identity_migration import prepare
    from tests.accounting.readers.test_identity import NOW, legacy_pair, setup

    root, env = setup(tmp_path, monkeypatch)
    items = legacy_pair(root / "cache-conversion.jsonl", "first")
    # Match a real native record/digest, including its private content.
    native = root / "cache-conversion.jsonl"
    raw = native.read_bytes().replace(b"ssgrep search x", CANARY.encode())
    native.write_bytes(raw)
    items[0] = replace(
        items[0],
        payload={**items[0].payload, "arguments": {"command": CANARY}},
    )
    from ssgrep.accounting.readers.base import evidence_digest

    items[0] = replace(items[0], digest=evidence_digest(items[0].kind, items[0].payload))
    items[1] = replace(items[1], locator=f"offset:{len(raw.splitlines(keepends=True)[0])}")
    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 8192))
    directory = expired_cache(items[0])
    ref = legacy_segment(items)
    checkpoint = store.Checkpoint(PROFILE, generation=1, segments=[ref])
    result = prepare(PROFILE, checkpoint, NOW, env=env, policy=enabled)
    assert result.complete and result.changed
    retained = list(store.iter_evidence(PROFILE, checkpoint.segments))
    assert len(retained) == 2 and all(item.native_key is not None for item in retained)
    maintain_cache(PROFILE)
    assert tuple(directory.iterdir()) == ()
    assert CANARY.encode() in native.read_bytes()


def test_cache_existing_file_outside_inventory_does_not_inflate_or_evict(tmp_path):
    from ssgrep.accounting.collect.replay_evidence import CacheWriter, _name

    seed = evidence()
    other = replace(seed, record_id="call:external", origin_id="call:external")
    last = replace(seed, record_id="call:last", origin_id="call:last")
    body = evidence_line(seed, plaintext=True) + b"\n"
    external_body = evidence_line(other, plaintext=True) + b" " * 64 + b"\n"
    last_body = evidence_line(last, plaintext=True) + b"\n"
    maximum = len(body) + len(external_body) + len(last_body) - min(len(body), len(external_body))
    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, maximum))
    with CacheWriter(PROFILE, enabled) as writer:
        writer.add(seed)
        directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
        external = directory / _name(other)
        assert writer.fd is not None
        assert storage.write_once(writer.fd, _name(other), external_body)
        seed_stamp = (directory / _name(seed)).stat().st_mtime_ns
        os.utime(external, ns=(seed_stamp - 1_000_000_000, seed_stamp - 1_000_000_000))
        stamp = external.stat().st_mtime_ns
        writer.add(other)  # Real write_once(False), not a mocked return value.
        assert writer.size == sum(path.stat().st_size for path in directory.iterdir())
        assert writer.names == {path.name for path in directory.iterdir()}
        assert external.stat().st_mtime_ns == stamp
        writer.add(last)
        assert {path.name for path in directory.iterdir()} == {
            _name(seed),
            _name(last),
        }
        assert writer.size == sum(path.stat().st_size for path in directory.iterdir())
        assert writer.size <= enabled.evidence_cache_max_bytes


@pytest.mark.parametrize("cache_on", [False, True])
@pytest.mark.parametrize("backfill", [False, True])
def test_real_native_read_boundary_caches_only_fresh_authorized_evidence(
    tmp_path, cache_on, backfill
):
    from datetime import UTC, datetime

    from ssgrep.accounting.collect.collector import CollectOptions
    from ssgrep.accounting.collect.read import LazyLedger, read_stage
    from ssgrep.accounting.collect.replay_evidence import cached_evidence
    from tests.accounting.collect.test_collect_suites import claude_dir, env_for
    from tests.accounting.readers.readers_fixtures import claude_tool_use, write_records

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096) if cache_on else None)
    write_records(
        claude_dir(tmp_path) / "fresh.jsonl",
        [
            claude_tool_use(
                "fresh",
                "fresh-call",
                "Bash",
                {"command": CANARY},
                ts="2026-10-14T00:00:00Z",
                cwd="/workspace",
            ),
            claude_tool_use(
                "fresh",
                "oversized-call",
                "Bash",
                {"command": CANARY * 1024},
                ts="2026-10-14T00:00:00Z",
                cwd="/workspace",
            ),
        ],
    )
    checkpoint = store.Checkpoint(PROFILE, baseline_ts="2026-10-01T00:00:00Z")
    options = CollectOptions(
        backfill=backfill,
        since="2026-10-01T00:00:00Z" if backfill else None,
        until="2026-10-15T00:00:00Z" if backfill else None,
    )
    outcome, items, _ = read_stage(
        profile=PROFILE,
        options=options,
        now=datetime(2026, 10, 15, tzinfo=UTC),
        clock=lambda: 0,
        should_stop=lambda: False,
        env=env_for(tmp_path),
        checkpoint=checkpoint,
        ledger=LazyLedger(PROFILE, checkpoint),
        expired=frozenset(),
        policy=enabled,
        horizon="2026-10-01",
        phase=lambda *args, **kwargs: None,
    )
    calls = [item for item in items if item.kind == "tool_call"]
    assert len(calls) == 2 and outcome.bytes_read > 0
    refs = store.write_segments(PROFILE, items)
    compact = [item for item in store.iter_evidence(PROFILE, refs) if item.kind == "tool_call"]
    cached = cached_evidence(PROFILE, compact[0])
    assert (cached is not None) is cache_on
    if cached is not None:
        assert cached.payload["arguments"]["command"] == CANARY
    assert cached_evidence(PROFILE, compact[1]) is None
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    assert sum(path.stat().st_size for path in directory.glob("*")) <= (
        enabled.evidence_cache_max_bytes if cache_on else 0
    )


def test_segment_serialization_does_not_authorize_cache_admission():
    from ssgrep.accounting.collect.replay_evidence import cached_evidence

    policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    (ref,) = store.write_segments(PROFILE, [evidence()])
    (compact,) = tuple(store.read_segment(PROFILE, ref))
    assert compact.native_key == evidence().native_key
    assert cached_evidence(PROFILE, compact) is None
    assert CANARY.encode() not in disk_bytes()


"""Identity migration archives cannot evade scoped deletion or orphan recovery."""


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


"""Consent JSON describes persisted classes without opening collector state."""


def test_default_disclosure_has_no_plaintext_consent_and_explicit_replay_limits():
    policy = ProfilePolicy()
    data = retained_data(policy)
    assert data["schema"] == 1
    assert data["evidence_format"] == "compact-v1"
    assert data["plaintext_cache"] == {
        "enabled": False,
        "retention_days": 0,
        "max_bytes": 0,
        "fields": ["prompts", "queries", "tool_arguments", "tool_outputs"],
    }
    replay = data["replay"]
    assert isinstance(replay, dict)
    assert replay["missing_or_changed_original"] == "unavailable_not_zero"
    assert replay["accepted_totals"] == "preserved"
    classes = data["classes"]
    assert isinstance(classes, dict)
    assert set(classes) >= {
        "journal",
        "compact_evidence",
        "rarity_metadata",
        "receipts",
        "accepted_totals",
        "queues",
        "temporary_files",
        "fingerprint_keys",
    }


def test_lifecycle_json_discloses_separate_cache_limits():
    policy = replace(
        ProfilePolicy(),
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=2,
        evidence_cache_max_bytes=8192,
    )
    result = lifecycle_document("enabled", policy, policy.profile_id, "metrics collect", "Collect.")
    data = result["data"]
    assert isinstance(data, dict)
    disclosed_policy = data["policy"]
    assert isinstance(disclosed_policy, dict)
    assert disclosed_policy["evidence_cache_enabled"] is True
    retained = data["retained_data"]
    assert isinstance(retained, dict)
    cache = retained["plaintext_cache"]
    assert isinstance(cache, dict)
    assert cache["retention_days"] == 2
    assert cache["max_bytes"] == 8192


"""Compact references replay exact originals, never changed text or zeroed accounting."""


def target(adapter):
    line = adapter.resolve("claude:src1:ep:0").line
    return next(record.ev for record in line.records if record.kind == "tool_result")


def test_compact_replay_reads_exact_authorized_original_and_keeps_call_link(tmp_path):
    path, _, adapter = old_source(tmp_path)
    item = target(adapter)
    compact = evidence_from_line(evidence_line(item))
    before = path.read_bytes()
    adapter.bytes_read = 0
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    restored = replay.hydrate(compact)
    assert restored.payload["output"] == item.payload["output"]
    assert restored.payload.get("call_record_id") == item.payload.get("call_record_id")
    assert restored.identity == item.identity
    assert replay.unavailable == {}
    assert path.read_bytes() == before


def test_deleted_original_preserves_compact_checkpoint_and_reports_replay_limit(tmp_path):
    path, _, adapter = old_source(tmp_path)
    item = target(adapter)
    refs = store.write_segments(PROFILE, [item])
    checkpoint = store.Checkpoint(
        PROFILE,
        generation=8,
        segments=refs,
        blobs={"accepted": store.write_blob(PROFILE, "accepted", {"tokens": 729})},
    )
    store.commit_checkpoint(PROFILE, checkpoint)
    path.unlink()
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    restored = replay.hydrate(next(store.iter_evidence(PROFILE, refs)))
    assert restored.payload["_replay_unavailable"] == "source_missing"
    assert replay.unavailable[(item.runtime, item.session_id)] == "source_missing"
    committed = store.load_checkpoint(PROFILE)
    assert committed is not None
    assert committed.generation == 8
    assert store.read_blob(PROFILE, committed.blobs["accepted"]) == {"tokens": 729}


def test_changed_original_never_substitutes_text_under_old_digest(tmp_path):
    path, _, adapter = old_source(tmp_path)
    item = target(adapter)
    compact = evidence_from_line(evidence_line(item))
    path.write_bytes(
        path.read_bytes().replace(item.payload["output"].encode(), b"changed-private-source")
    )
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    restored = replay.hydrate(compact)
    assert restored.digest == item.digest
    assert "output" not in restored.payload
    assert restored.payload["_replay_unavailable"] == "source_digest_drift"


def test_actual_cycle_keeps_accepted_rows_after_original_deletion(tmp_path):
    from pathlib import Path

    from ssgrep.accounting.collect.state import load_state
    from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
    from tests.accounting.collect.test_collect_suites import NOW, seeded

    seeded(tmp_path)
    checkpoint = scene.committed()
    before = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert before.ledger is not None and before.ledger.debits
    for source in checkpoint.sources.values():
        path = Path(source.source_id.split(":", 1)[1])
        assert path.is_relative_to(tmp_path)
        path.unlink()
    vs.filler(vs.projects(tmp_path), "new-input", NOW, 1)
    scene.collect(NOW)
    after_checkpoint = scene.committed()
    after = load_state(PROFILE, after_checkpoint, after_checkpoint.methodology)
    assert after.ledger is not None
    assert after.reduction is not None
    assert after.ledger.candidates == before.ledger.candidates
    assert after.ledger.debits == before.ledger.debits
    assert any("replay" in key or "source_missing" in key for key in after.reduction.coverage)


def test_optional_cache_verifies_content_not_only_stored_digest_field(tmp_path):
    import json

    from ssgrep.accounting.collect.replay_evidence import cache_collected_evidence, cached_evidence
    from ssgrep.accounting.privacy import policy, storage

    _, _, adapter = old_source(tmp_path)
    item = target(adapter)
    policy.enable_collection(PROFILE, evidence_cache=(1, 1024 * 1024))
    cache_collected_evidence(PROFILE, [item])
    refs = store.write_segments(PROFILE, [item])
    compact = next(store.iter_evidence(PROFILE, refs))
    cached = cached_evidence(PROFILE, compact)
    assert cached is not None
    assert cached.payload["output"] == item.payload["output"]
    directory = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    path = next(directory.iterdir())
    raw = json.loads(path.read_bytes())
    raw["payload"]["output"] = "changed-cache-private-canary"
    path.write_text(json.dumps(raw))
    assert cached_evidence(PROFILE, compact) is None
    replay = EvidenceReplay(PROFILE, adapter=adapter)
    assert replay.hydrate(compact).payload["output"] == item.payload["output"]


"""Scoped erasure removes caches and shared rarity authority without source mutation."""


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
    cache_collected_evidence(PROFILE, [mine, theirs])
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
    from ssgrep.accounting.capture import fingerprint

    fingerprint.load_key(PROFILE, create=True)
    store.commit_checkpoint(PROFILE, store.Checkpoint(PROFILE))
    purge_derived(Selection(PROFILE, "2026-10-15T00:00:00Z", None))
    assert fingerprint.load_key(PROFILE, create=False) is None


def test_scoped_purge_removes_old_verdict_pointers_without_erasing_newer_same_session():
    from ssgrep.accounting.collect.state import ReductionDoc
    from ssgrep.accounting.privacy.purge import Scopes
    from ssgrep.accounting.privacy.purge_state import _reduction_after
    from tests.accounting.collect.test_collect_suites import invocation

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
    from ssgrep.accounting.privacy.purge_recovery import recover_pending_purge, stage_erasure
    from ssgrep.accounting.serve.lease import Lease

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
    from ssgrep.accounting.privacy.purge_state import plan_state, write_purged_state

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


"""Purge expands and rewrites sharded derived documents without retaining erased rows."""


def test_purge_reads_and_rewrites_sharded_current_and_archived_ledger():
    from ssgrep.accounting.collect import state as collector_ledger_store
    from ssgrep.accounting.collect.state import VersionedLedger
    from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle, Methodology
    from tests.accounting.ledger.test_named_ledger_revisions import DAY, TASK, UNIT, debit

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
    filename = collector_ledger_store.write_ledger_document(PROFILE, name, logical)
    cp = store.Checkpoint(PROFILE, methodology=ledger.methodology.hash, blobs={name: filename})
    selection = Selection(PROFILE, "2026-10-15T00:00:00Z", None)
    planned = plan_state(PROFILE, selection, cp)
    assert planned is not None
    assert planned.changed
    purged = write_purged_state(PROFILE, planned, selection)
    expanded = collector_ledger_store.expand_ledger_document(
        PROFILE, store.read_blob(PROFILE, purged.blobs[name])
    )
    assert expanded["state"]["debits"] == []
    assert expanded["archives"] == {}


def test_purge_rewrites_sharded_reduction_and_keeps_other_workspace_metadata():
    from ssgrep.accounting.collect import (
        partitions as collector_reduction_store,
        state as collector_assignment_store,
    )
    from tests.accounting.collect.test_collect_suites import invocation

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
    filename = collector_reduction_store.write_reduction_document(
        PROFILE, REDUCTION, logical.to_dict()
    )
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
    expanded = collector_reduction_store.expand_reduction_document(
        PROFILE, store.read_blob(PROFILE, purged.blobs[REDUCTION])
    )
    restored = ReductionDoc.from_dict(expanded)
    assert restored.invocations == (other,)
    assert restored.lookup_schema == "lookups_v1"
    retained = collector_assignment_store.expand_assignments_document(
        PROFILE, store.read_blob(PROFILE, purged.blobs["frequency-assignments"])
    )
    assert retained["assignments"] == rows[1:]
