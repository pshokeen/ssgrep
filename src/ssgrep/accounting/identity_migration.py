"""Explicit native_identity_v2 replacement; never combine legacy and corrected totals."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_state import COUNTERS, REDUCTION
from ssgrep.accounting.identity_migration_record import _context, migrate_record
from ssgrep.accounting.native_identity import CONTEXT_BYTES, IDENTITY_METHOD
from ssgrep.accounting.native_source import SourceReadError, open_source
from ssgrep.accounting.readers.base import WINDOW, SourceState
from ssgrep.accounting.scope_events import Scoper

ARCHIVE = "identity-archive-" + IDENTITY_METHOD


@dataclass(frozen=True)
class MigrationResult:
    changed: bool = False
    complete: bool = True
    warnings: tuple[str, ...] = ()
    unavailable_sessions: dict[tuple[str, str], str] = field(default_factory=dict)
    bytes_read: int = 0
    epoch_reason: str = "unchanged_journal_identity"
    source_counts: dict[str, int] = field(default_factory=dict)


def _active(name: str) -> bool:
    return not name.startswith("identity-archive-") and "-scope-" not in name


def archive_current(
    profile: str,
    checkpoint: store.Checkpoint,
    now: datetime,
    legacy_segments: list[store.SegmentRef],
) -> dict[str, str]:
    """Private replay/purge manifest owns old current rows, never active totals."""
    from ssgrep.accounting.published_capture import captured_documents

    published = captured_documents(profile, checkpoint)
    # Actual published state predates the current codecs, including inactive
    # scope aliases. Move their existing references atomically into the same
    # opaque archive; leaving one top-level would route it to a new decoder.
    previous = {
        k: v
        for k, v in checkpoint.blobs.items()
        if _active(k) or (published is not None and not k.startswith("identity-archive-"))
    }
    reduction = store.read_blob(profile, previous.get(REDUCTION))
    if isinstance(reduction, dict) and reduction.get("format") == "root-reduction-v1":
        from ssgrep.accounting.collector_reduction_store import expand_document

        reduction = expand_document(profile, reduction)
    roots = reduction.get("session_roots", []) if isinstance(reduction, dict) else []
    cwds = reduction.get("session_cwd", []) if isinstance(reduction, dict) else []
    cwd_keys = {(r[0], r[1]) for r in cwds if isinstance(r, (list, tuple)) and len(r) == 3}
    mapped = bool(roots) and all(
        isinstance(r, (list, tuple)) and len(r) == 3 and (r[0], r[1]) in cwd_keys for r in roots
    )
    manifest = {
        "schema": 1,
        "identity_method": checkpoint.identity_method or "legacy_session_scoped",
        "replacement_method": IDENTITY_METHOD,
        "superseded_at": now.isoformat(),
        "methodology": checkpoint.methodology,
        "collection_scope": checkpoint.collection_scope,
        "session_roots": roots,
        "session_cwd": cwds,
        "root_mapping_status": "available" if mapped else "unavailable",
        "blobs": previous,
        "segments": [r.to_dict() for r in legacy_segments],
        "counter_policy": "preserve_journal_observations",
    }
    if published is not None:
        manifest["published_documents"] = published
        if any("-scope-" in name for name in previous):
            # Current-only session metadata cannot prove a prior scope's cwd.
            # Preserve every old alias, but do not guess workspace ownership.
            manifest["root_mapping_status"] = "unavailable"
    if previous or legacy_segments:
        checkpoint.blobs[ARCHIVE] = store.write_blob(profile, ARCHIVE, manifest)
    for name in previous:
        checkpoint.blobs.pop(name, None)
    return previous


def _failures(job: dict) -> dict[tuple[str, str], str]:
    return {(r, s): why for r, s, why in job.get("unavailable", [])}


def _repair_source(record: Any, readers: dict) -> int:
    reader = readers.get(record.runtime)
    if reader is None or record.runtime not in {"claude", "pi", "omp", "prime-agent"}:
        return 0
    prefix = record.runtime + ":"
    if not record.source_id.startswith(prefix):
        return 0
    from pathlib import Path

    source = reader.source_for(Path(record.source_id[len(prefix) :]))
    state = SourceState.from_dict(record.state)
    if not state.offset:
        return 0
    descriptor = open_source(source.path, tuple(reader.roots()))
    try:
        info = os.fstat(descriptor)
        if info.st_size < state.offset:
            raise OSError("legacy committed prefix truncated")
        context, context_bytes = _context(reader, source, descriptor, state.offset)
        context["native_identity_method"] = IDENTITY_METHOD
        context["baseline_context"] = True
        head = os.pread(descriptor, min(state.offset, WINDOW), 0)
        tail_start = max(0, state.offset - WINDOW)
        tail = os.pread(descriptor, state.offset - tail_start, tail_start)
        repaired = SourceState(
            state.offset,
            info.st_dev,
            info.st_ino,
            hashlib.sha256(head).hexdigest(),
            hashlib.sha256(tail).hexdigest(),
            context,
        )
        record.state = repaired.to_dict()
        return context_bytes + len(head) + len(tail)
    finally:
        os.close(descriptor)


def prepare(
    profile: str,
    checkpoint: store.Checkpoint,
    now: datetime,
    *,
    env=None,
    policy=None,
    journal=None,
    byte_budget: int = 32 * 1024 * 1024,
    max_records: int = 256,
) -> MigrationResult:
    """Mutate staged progress; switch active namespace only when all originals verify.

    The caller commits staged checkpoints and stops before reading/reducing when
    complete=False. Missing originals leave the prior authoritative legacy state
    intact and explicitly unavailable for corrected reporting/export.
    """
    if checkpoint.identity_method == IDENTITY_METHOD:
        proof = checkpoint.identity_migration
        if proof.get("target") != IDENTITY_METHOD or proof.get("status") != "complete":
            return MigrationResult(
                complete=False, warnings=("identity_migration_proof_unavailable",)
            )
        return MigrationResult()
    if (
        not checkpoint.segments
        and checkpoint.generation == 0
        and not checkpoint.blobs
        and not checkpoint.sources
    ):
        checkpoint.identity_method = IDENTITY_METHOD
        checkpoint.identity_migration = {"target": IDENTITY_METHOD, "status": "complete"}
        return MigrationResult(changed=True)
    from ssgrep.accounting.collector_read import make_readers

    if policy is None:
        from ssgrep.accounting.policy import load_profile_policy

        policy = load_profile_policy(profile)
    readers = make_readers(env)
    scoper = Scoper(policy)
    job: dict[str, Any] = checkpoint.identity_migration
    if job.get("target") != IDENTITY_METHOD or job.get("status") == "unavailable":
        published_capture = job.get("published_capture")
        job = {
            "target": IDENTITY_METHOD,
            "status": "pending",
            "segment": 0,
            "record": 0,
            "staged_segments": [],
            "compact_legacy_segments": [],
            "unavailable": [],
            "source": 0,
        }
        if published_capture is not None:
            job["published_capture"] = published_capture
        checkpoint.identity_migration = job
    from ssgrep.accounting.published_capture import stage as capture_publication

    captured, publication_bytes = capture_publication(
        profile, checkpoint, limit=min(32, max_records), byte_budget=byte_budget
    )
    if not captured or publication_bytes >= byte_budget:
        return MigrationResult(
            True, False, ("identity_migration_pending",), bytes_read=publication_bytes
        )
    failures = _failures(job)
    staged: dict[str, list] = {}
    legacy: dict[str, list] = {}
    count, consumed = 0, publication_bytes
    while job["segment"] < len(checkpoint.segments):
        ref = checkpoint.segments[job["segment"]]
        items = list(store.read_segment(profile, ref))
        cursor = job["record"]
        while cursor < len(items):
            if count >= max_records or (count and consumed >= byte_budget):
                break
            item = items[cursor]
            result = migrate_record(
                item,
                readers,
                scoper,
                max(byte_budget - consumed, 8 * 1024 * 1024 + 2 * CONTEXT_BYTES),
            )
            consumed += result.bytes_read
            if result.evidence is None:
                failures[(item.runtime, item.session_id)] = result.reason
            else:
                staged.setdefault(ref.origin, []).append(result.evidence)
            legacy.setdefault(ref.origin, []).append(item)
            cursor += 1
            count += 1
        job["record"] = cursor
        if cursor < len(items):
            break
        job["segment"] += 1
        job["record"] = 0
    for origin, items in staged.items():
        job["staged_segments"].extend(
            r.to_dict() for r in store.write_segments(profile, items, origin=origin)
        )
    for origin, items in legacy.items():
        job["compact_legacy_segments"].extend(
            r.to_dict() for r in store.write_segments(profile, items, origin=origin)
        )
    job["unavailable"] = [[r, s, why] for (r, s), why in sorted(failures.items())]
    source_keys = sorted(checkpoint.sources)
    if job["segment"] == len(checkpoint.segments) and not failures:
        while job["source"] < len(source_keys) and (not count or consumed < byte_budget):
            record = checkpoint.sources[source_keys[job["source"]]]
            try:
                consumed += _repair_source(record, readers)
            except (SourceReadError, OSError):
                # A disappeared source without retained rows does not erase accepted history.
                record.status, record.complete = "unreadable", False
            job["source"] += 1
            count += 1
            if count >= max_records:
                break
    done = job["segment"] == len(checkpoint.segments) and job["source"] == len(source_keys)
    if failures and job["segment"] == len(checkpoint.segments):
        job["status"] = "unavailable"
    if not done or failures:
        warning = "identity_migration_unavailable" if failures else "identity_migration_pending"
        return MigrationResult(True, False, (warning,), failures, consumed)
    previous_counters = checkpoint.blobs.get(COUNTERS)
    archived = archive_current(
        profile,
        checkpoint,
        now,
        [store.SegmentRef.from_dict(r) for r in job["compact_legacy_segments"]],
    )
    if previous_counters is not None:
        checkpoint.blobs[COUNTERS] = previous_counters
    checkpoint.segments = [store.SegmentRef.from_dict(r) for r in job["staged_segments"]]
    checkpoint.identity_method = IDENTITY_METHOD
    checkpoint.methodology, checkpoint.input_digest, checkpoint.journal_sig = "", "", ""
    checkpoint.compacted_day = None
    checkpoint.backfill_job = {}
    checkpoint.identity_migration = {
        "target": IDENTITY_METHOD,
        "status": "complete",
        "superseded_blobs": archived,
        "superseded_at": now.isoformat(),
        "counter_policy": "preserve_journal_observations",
    }
    return MigrationResult(
        True,
        True,
        ("native_identity_v2_replaces_legacy",),
        bytes_read=consumed,
        source_counts={
            "repaired_sources": len(source_keys),
            "verified_records": sum(r["records"] for r in job["staged_segments"]),
            **(
                {"known_journal_executions": len({e.execution_id for e in journal.events})}
                if journal is not None
                else {}
            ),
        },
    )


def reset_after_purge(checkpoint: store.Checkpoint, new_segments, new_blobs) -> None:
    """Only surviving refs can restart migration; discarded originals cannot resurrect."""
    checkpoint.segments = list(new_segments)
    checkpoint.blobs = dict(new_blobs)
    checkpoint.input_digest, checkpoint.journal_sig = "", ""
    if checkpoint.identity_method != IDENTITY_METHOD:
        checkpoint.identity_migration = {
            "target": IDENTITY_METHOD,
            "status": "pending",
            "segment": 0,
            "record": 0,
            "source": 0,
            "staged_segments": [],
            "compact_legacy_segments": [],
            "unavailable": [],
        }
        return
    surviving: dict[str, str] = {}
    for name, file_name in new_blobs.items():
        if name.startswith("identity-archive-"):
            manifest = store.read_blob(checkpoint.profile, file_name)
            if isinstance(manifest, dict) and isinstance(manifest.get("blobs"), dict):
                surviving.update(manifest["blobs"])
    checkpoint.identity_migration = {
        "target": IDENTITY_METHOD,
        "status": "complete",
        "superseded_blobs": surviving,
        "counter_policy": "preserve_journal_observations",
    }
