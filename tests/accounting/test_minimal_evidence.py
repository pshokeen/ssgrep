"""Private canaries exercise compact evidence, consent and recoverable migration."""

import json
import os
from dataclasses import replace

import pytest

from ssgrep.accounting import collector_store as store, policy, storage
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.ledger_contracts import ProfilePolicy

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
    raw = json.loads(store.evidence_line(evidence()))
    for key in ("evidence_format", "origin_domain", "origin_id", "origin_status"):
        raw.pop(key)
    restored = store.evidence_from_line(json.dumps(raw).encode())
    assert restored.origin_status == "legacy_unverified"
    assert restored.origin_domain is None


def test_cache_opt_in_is_bounded_and_revocation_erases_plaintext():
    from ssgrep.accounting.evidence_cache import maintain_cache

    enabled = ProfilePolicy(
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=4096,
    )
    policy.save_profile(enabled)
    store.write_segments(PROFILE, [evidence()])
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
    from ssgrep.accounting.evidence_migration import migrate_evidence

    raw = json.loads(store.evidence_line(evidence()))
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
    from ssgrep.accounting.evidence_cache import maintain_cache

    enabled = ProfilePolicy(
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=2048,
    )
    policy.save_profile(enabled)
    huge = replace(evidence(), payload={"cwd": "/workspace", "output": CANARY * 1024})
    store.write_segments(PROFILE, [huge])
    assert CANARY.encode() not in disk_bytes()
    store.write_segments(PROFILE, [evidence()])
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
    restored = store.evidence_from_line(store.evidence_line(item))
    assert restored.payload["tokens"] == {"input_tokens": 12, "output_tokens": 8}
    assert restored.payload["model"] is None
    assert restored.payload["cost_nano_usd"] == 512
    assert restored.payload["streaming_updates"] == 3
    assert restored.payload["provider"] == "native-provider"
    assert CANARY.encode() not in store.evidence_line(item)


def test_cache_needs_separate_consent_and_valid_bounds_before_any_policy_write():
    from ssgrep.accounting.policy_rules import PolicyError

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
    from ssgrep.accounting import evidence_cache

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
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
    restored = store.evidence_from_line(store.evidence_line(claim))
    assert restored.payload["target_native_key"] == ["claude", "domain", "call:a"]
    assert restored.payload["target_digest"] == "other-digest"
    assert restored.payload["target_record_id"] == "result:copied"
    assert restored.payload["call_record_id"] == "call:known"
    assert restored.payload["call_link_status"] is None
    assert restored.payload["origin_conflict"] is True
    assert restored.payload["claimant_parent_session"] is None
    assert CANARY.encode() not in store.evidence_line(claim)


def test_cache_revocation_is_linearized_after_inflight_plaintext_writer():
    from threading import Event, Thread

    from ssgrep.accounting.evidence_cache import CacheWriter

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
    from ssgrep.accounting.evidence_migration import migrate_evidence

    body = store.evidence_line(evidence(), plaintext=True) + b"\n"
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
    from ssgrep.accounting import evidence_cache
    from ssgrep.accounting.native_identity import with_origin

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
    store.write_segments(PROFILE, [first])
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
        store.write_segments(PROFILE, [second])
    assert sorted(reads) == sorted(before.items())
    cached = evidence_cache.cached_evidence(PROFILE, second)
    assert cached is not None
    assert cached.payload["output"] == f"{CANARY}-1"
    # A different writer changes the directory between batches: no global/stale inventory.
    evidence_cache.cache_evidence(PROFILE, external, enabled)
    reads.clear()
    before = {path.name: path.stat().st_size for path in directory.iterdir()}
    with monkeypatch.context() as patch:
        patch.setattr(storage, "read_file", measured_read)
        store.write_segments(PROFILE, [last])
    assert sorted(reads) == sorted(before.items())
    assert (
        sum(path.stat().st_size for path in directory.iterdir()) <= enabled.evidence_cache_max_bytes
    )
    cached = evidence_cache.cached_evidence(PROFILE, external)
    assert cached is not None
    assert cached.payload["output"] == f"{CANARY}-2"


def test_shared_cache_writer_updates_inventory_without_rereading_its_own_writes(monkeypatch):
    from ssgrep.accounting.evidence_cache import CacheWriter

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    seed = evidence()
    store.write_segments(PROFILE, [seed])
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
    from ssgrep.accounting.evidence_cache import CacheWriter

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    store.write_segments(PROFILE, [evidence()])
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
    from ssgrep.accounting import evidence_cache

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    store.write_segments(PROFILE, [evidence()])
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
    store.write_segments(
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
    from ssgrep.accounting.evidence_migration import migrate_evidence

    native = tmp_path / "original-native.jsonl"
    native.write_bytes(CANARY.encode())
    body = (
        store.evidence_line(replace(evidence(), source_id=f"claude:{native}"), plaintext=True)
        + b"\n"
    )
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
    from ssgrep.accounting.evidence_migration import migrate_evidence

    body = store.evidence_line(evidence(), plaintext=True) + b"\n"
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
    from ssgrep.accounting.evidence_cache import cache_evidence
    from ssgrep.accounting.evidence_migration import migrate_evidence

    enabled = policy.enable_collection(PROFILE, evidence_cache=(1, 4096))
    old = replace(evidence(), ts_utc="2000-01-01T00:00:00Z")
    cache_evidence(PROFILE, old, enabled)
    cache_dir = storage.data_root() / "accounting" / PROFILE / "collector" / "cache"
    (cached,) = tuple(cache_dir.iterdir())
    assert CANARY.encode() in cached.read_bytes()
    os.utime(cached, (0, 0))
    body = store.evidence_line(old, plaintext=True) + b"\n"
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
    store.write_segments(PROFILE, [fresh])
    assert CANARY.encode() in b"".join(path.read_bytes() for path in cache_dir.iterdir())


@pytest.mark.parametrize("boundary", ["root_narrowing", "revocation", "mtime_expiry", "malformed"])
def test_cached_plaintext_privacy_boundaries_do_not_renew_from_compact_replay(
    tmp_path, monkeypatch, boundary
):
    from ssgrep.accounting import evidence_cache
    from ssgrep.accounting.native_identity import with_origin

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
    fresh_refs = store.write_segments(PROFILE, [fresh])
    (fresh_compact,) = tuple(store.iter_evidence(PROFILE, fresh_refs))
    recovered = evidence_cache.cached_evidence(PROFILE, fresh_compact)
    assert recovered is not None
    assert recovered.native_key == fresh.native_key and recovered.digest == fresh.digest
    assert recovered.payload["output"] == CANARY
    assert CANARY.encode() in b"".join(path.read_bytes() for path in directory.iterdir())
