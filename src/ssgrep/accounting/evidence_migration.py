"""Checkpoint-first, resumable removal of obsolete mirrored accounting evidence."""

from __future__ import annotations

import os
from dataclasses import replace

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.evidence_cache import maintain_cache
from ssgrep.accounting.evidence_codec import COMPACT, compact_payload

# Legacy accounting segments are atomic JSONL objects, not native-source scans.
# The original writer targeted 32 MiB; the existing store hard ceiling is explicit.
MAX_ATOMIC_SEGMENT_BYTES = store.MAX_BLOB_BYTES


def _compact_one(profile, ref):
    if not 0 <= ref.size <= MAX_ATOMIC_SEGMENT_BYTES:
        raise CollectError(
            "evidence_migration_child_bound",
            "Legacy evidence segment exceeds the atomic migration limit.",
        )
    compact = (
        replace(item, payload=compact_payload(item.payload))
        for item in store.read_segment(profile, ref)
    )
    return store.write_segments(profile, compact, origin=ref.origin)


def _manifest_size(profile, filename):
    directory = storage.open_profile_dir(profile, "collector", "state", create=False)
    try:
        descriptor = storage.open_regular(directory, filename)
        try:
            return os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def migrate_evidence(
    profile: str, checkpoint: store.Checkpoint, *, byte_budget: int = MAX_ATOMIC_SEGMENT_BYTES
) -> store.Checkpoint:
    """One original segment or archive-manifest quantum; no native reads or revaluation.

    Every replacement commits before deletion. A pending return must stop the
    collection cycle before native work/current-codec decoding. The last quantum
    retains a cleanup obligation until orphan recovery has removed old plaintext.
    Accepted ledger blobs, generation, and original event dates never change.
    """
    maintain_cache(profile)
    retention = dict(checkpoint.retention)
    ready = (
        retention.get("evidence_format") == COMPACT
        and retention.get("archive_evidence_format") == COMPACT
    )
    if ready:
        if not retention.get("evidence_cleanup_pending"):
            return checkpoint
        store.sweep_orphans(profile, checkpoint)
        retention.pop("evidence_cleanup_pending", None)
        cleaned = replace(checkpoint, retention=retention)
        store.commit_checkpoint(profile, cleaned)
        return cleaned
    job = dict(retention.get("evidence_compaction", {}))
    job.setdefault("phase", "archives" if retention.get("evidence_format") == COMPACT else "active")
    refs, blobs = list(checkpoint.segments), dict(checkpoint.blobs)
    worked = False
    last_bytes = 0

    def blocked(required):
        if job.get("blocked_child_bytes") == required and retention.get(
            "evidence_migration_pending"
        ):
            return checkpoint
        job["blocked_child_bytes"] = required
        retention.update(
            evidence_compaction=job,
            evidence_migration_pending=True,
            evidence_migration_last_bytes=last_bytes,
        )
        result = replace(checkpoint, retention=retention)
        store.commit_checkpoint(profile, result)
        return result

    if int(job.get("blocked_child_bytes", 0)) > byte_budget:
        return checkpoint
    job.pop("blocked_child_bytes", None)
    if job["phase"] == "active":
        cursor = int(job.get("index", 0))
        if cursor < len(refs):
            old = refs[cursor]
            if old.size > byte_budget:
                return blocked(old.size)
            compact = _compact_one(profile, old)
            refs[cursor : cursor + 1] = compact
            job["index"] = cursor + len(compact)
            job["cleanup"] = bool(job.get("cleanup")) or compact != [old]
            last_bytes, worked = old.size, True
        if int(job.get("index", 0)) >= len(refs):
            retention["evidence_format"] = COMPACT
            job.update(phase="archives", index=0)
    archives = sorted(name for name in blobs if name.startswith("identity-archive-"))
    after = str(job.get("after", ""))
    pending_archives = [name for name in archives if name > after]
    if job["phase"] == "archives" and not worked and pending_archives:
        name = pending_archives[0]
        size = _manifest_size(profile, blobs[name])
        if size > byte_budget:
            return blocked(size)
        manifest = store.read_blob(profile, blobs[name])
        last_bytes = size
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            raise CollectError("state_corrupt", "Identity archive cannot be compacted safely.")
        old_refs = [store.SegmentRef.from_dict(raw) for raw in manifest.get("segments", [])]
        cursor = int(job.get("index", 0))
        if cursor < len(old_refs):
            old = old_refs[cursor]
            if old.size + size > byte_budget:
                return blocked(old.size + size)
            compact = _compact_one(profile, old)
            old_refs[cursor : cursor + 1] = compact
            job["index"] = cursor + len(compact)
            if compact != [old]:
                blobs[name] = store.write_blob(
                    profile, name, {**manifest, "segments": [ref.to_dict() for ref in old_refs]}
                )
                job["cleanup"] = True
            last_bytes += old.size
        if int(job.get("index", 0)) >= len(old_refs):
            job.update(after=name, index=0)
        worked = True
    done = job["phase"] == "archives" and not any(
        name > str(job.get("after", "")) for name in archives
    )
    retention["evidence_migration_last_bytes"] = last_bytes
    if done:
        retention.update(evidence_format=COMPACT, archive_evidence_format=COMPACT)
        retention.pop("evidence_compaction", None)
        retention.pop("evidence_migration_pending", None)
        if job.get("cleanup") or retention.get("evidence_cleanup_pending"):
            retention["evidence_cleanup_pending"] = True
    else:
        retention["evidence_compaction"] = job
        retention["evidence_migration_pending"] = True
    migrated = replace(checkpoint, segments=refs, blobs=blobs, retention=retention)
    store.commit_checkpoint(profile, migrated)
    return (
        migrate_evidence(profile, migrated, byte_budget=byte_budget)
        if done and retention.get("evidence_cleanup_pending")
        else migrated
    )
