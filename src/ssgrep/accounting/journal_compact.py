"""Nonblocking, checkpoint-proven replacement of private journal segments.

Publish retained-only bytes before removing the original. A crash exposes both
copies (reader dedupes) or the new complete copy. Descriptors from cooperative
writers are checked for unlink before any append; no file name is reused.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.journal_codec import MalformedEvent, parse_event, serialize_event
from ssgrep.accounting.journal_reader import SegmentProof, _lines
from ssgrep.accounting.journal_retention import expand_journal_retention, retained_event


@dataclass
class JournalMaintenance:
    status: str = "complete"
    removed_segments: int = 0
    retained_segments: int = 0
    deferred_segments: int = 0
    legacy_segments: int = 0
    bytes_read: int = 0
    quarantine_removed: int = 0
    remnants_removed: int = 0
    recovery_action: str | None = None
    warnings: tuple[str, ...] = ()
    reasons: dict[str, int] = field(default_factory=dict)

    def defer(self, reason: str) -> None:
        self.deferred_segments += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _safe_segment(name: str) -> bool:
    return (
        isinstance(name, str)
        and name.startswith("seg-")
        and name.endswith(".jsonl")
        and "/" not in name
        and "\0" not in name
    )


def _retained_bytes(
    fd: int, proof: SegmentProof, record: dict[str, Any]
) -> tuple[bytes | None, str | None]:
    kept = bytearray()
    changed = proof.size == 0
    modified_day = (
        datetime.fromtimestamp(proof.modified_ns // 1_000_000_000, UTC).date().isoformat()
    )
    stale_damage = modified_day < record["horizon_day"]
    for _, line, terminated, _ in _lines(fd, proof.size):
        try:
            if not terminated or line is None:
                raise MalformedEvent("partial or oversized record")
            event = parse_event(line)
        except MalformedEvent:
            if not stale_damage:
                return None, "recent_damage"
            changed = True
            continue
        retained = retained_event(event, record)
        if retained is None:
            changed = True
        else:
            kept.extend(serialize_event(retained))
            changed = changed or retained != event
    return (bytes(kept) if changed else None), None


def _publish(
    directory: int,
    proof: SegmentProof,
    data: bytes,
    record: dict[str, Any],
    result: JournalMaintenance,
    remaining: int,
) -> bool:
    content_digest = hashlib.sha256(data).hexdigest()
    identity = json.dumps(
        [proof.to_dict(), record["horizon_day"], content_digest], sort_keys=True
    ).encode()
    nonce = hashlib.sha256(identity).hexdigest()
    destination = f"seg-v2-compact-{nonce}.jsonl"
    temporary = f".retention-{nonce}.tmp"
    try:
        existing = storage.open_regular(directory, destination)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        try:
            if os.fstat(existing).st_size != len(data):
                raise ValueError("retention replacement has conflicting size")
            if len(data) > remaining:
                return False
            digest = hashlib.sha256()
            read = 0
            while read < len(data):
                chunk = os.read(existing, min(1 << 20, len(data) - read))
                if not chunk:
                    raise ValueError("retention replacement was truncated")
                read += len(chunk)
                result.bytes_read += len(chunk)
                digest.update(chunk)
            if digest.hexdigest() != content_digest:
                raise ValueError("retention replacement has conflicting content")
            return True
        finally:
            os.close(existing)
    # A previous attempt can leave only a nonauthoritative retained-only staging file.
    storage.delete_file(directory, temporary)
    fd = storage.create_exclusive(directory, temporary)
    try:
        storage.write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    # Exclusive creation means a replacement never reuses a writer's segment name.
    os.link(
        temporary,
        destination,
        src_dir_fd=directory,
        dst_dir_fd=directory,
        follow_symlinks=False,
    )
    storage.fsync_dir(directory)
    storage.delete_file(directory, temporary)
    storage.fsync_dir(directory)
    return True


def _compact_one(
    directory: int,
    proof: SegmentProof,
    record: dict[str, Any],
    result: JournalMaintenance,
    *,
    legacy_writers_stopped: bool,
    byte_budget: int,
) -> None:
    if not _safe_segment(proof.name):
        raise ValueError("unsafe segment in journal retention checkpoint")
    try:
        fd = storage.open_regular(directory, proof.name)
    except FileNotFoundError:
        return
    try:
        if not proof.name.startswith("seg-v2-") and legacy_writers_stopped is not True:
            result.legacy_segments += 1
            result.defer("legacy_writer_ack_required")
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.defer("writer_busy")
            return
        if SegmentProof.capture(proof.name, fd) != proof:
            result.defer("changed_after_checkpoint")
            return
        data, reason = _retained_bytes(fd, proof, record)
        result.bytes_read += os.lseek(fd, 0, os.SEEK_CUR)
        if reason:
            result.defer(reason)
            return
        if data is None:
            return
        if data:
            if not _publish(
                directory, proof, data, record, result, byte_budget - result.bytes_read
            ):
                result.defer("maintenance_budget")
                return
            result.retained_segments += 1
        # Only a proof-matching, locked source is unlinked, after replacement durability.
        storage.delete_file(directory, proof.name)
        storage.fsync_dir(directory)
        result.removed_segments += 1
    finally:
        os.close(fd)


def compact_committed_journal(
    profile: str,
    *,
    generation: int,
    byte_budget: int,
    segment_budget: int,
    legacy_writers_stopped: bool,
) -> JournalMaintenance:
    from ssgrep.accounting.collector_store import load_checkpoint
    from ssgrep.accounting.journal_remnants import sweep_remnants

    if type(byte_budget) is not int or byte_budget < 0:
        raise ValueError("byte_budget must be a nonnegative integer")
    if type(segment_budget) is not int or segment_budget < 0:
        raise ValueError("segment_budget must be a nonnegative integer")
    checkpoint = load_checkpoint(profile)
    if checkpoint is None:
        return JournalMaintenance(status="no_checkpoint")
    if checkpoint.generation != generation:
        return JournalMaintenance(status="generation_mismatch")
    record = checkpoint.retention.get("journal")
    if record is None:
        return JournalMaintenance(status="no_retention_checkpoint")
    record = expand_journal_retention(profile, record)
    result = JournalMaintenance()
    try:
        directory = storage.open_profile_dir(profile, "journal", create=False)
    except FileNotFoundError:
        return result
    try:
        visited = 0
        for raw in record["proofs"]:
            proof = SegmentProof(**raw)
            if not _safe_segment(proof.name):
                raise ValueError("unsafe segment in journal retention checkpoint")
            if visited >= segment_budget or proof.size > byte_budget - result.bytes_read:
                # Already removed proofs cost no work and must not starve later entries.
                try:
                    os.stat(proof.name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                result.defer("maintenance_budget")
                continue
            try:
                os.stat(proof.name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            visited += 1
            _compact_one(
                directory,
                proof,
                record,
                result,
                legacy_writers_stopped=legacy_writers_stopped,
                byte_budget=byte_budget,
            )
        sweep_remnants(
            profile,
            directory,
            record,
            result,
            file_budget=max(0, segment_budget - visited),
            byte_budget=max(0, byte_budget - result.bytes_read),
        )
        queued = record.get("queued_segments", 0)
        if queued:
            result.deferred_segments += queued
            result.reasons["queued_maintenance"] = queued
    finally:
        os.close(directory)
    if result.legacy_segments:
        result.status = "blocked_legacy"
        result.recovery_action = (
            "Stop every pre-v2 journal capture process, then run "
            f"ssgrep metrics collect --profile {profile} --journal-legacy-writers-stopped "
            "to acknowledge the offline migration."
        )
        result.warnings = (*result.warnings, "journal_retention_blocked_legacy_writers")
    elif result.deferred_segments or result.reasons.get("remnants_pending"):
        result.status = "pending"
        result.warnings = (*result.warnings, "journal_retention_pending")
    return result
