"""Resumable original-publication retention over existing immutable children.

An old monolithic child is atomic, not constant-cost: an oversized child stays an
explicit byte-budget backlog. No native input, old runtime or valuation is used.
"""

from __future__ import annotations

import os
from bisect import bisect_left
from dataclasses import dataclass

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of

KEY = "published_retention"
PREFIX = "identity-archive-"


@dataclass(frozen=True)
class Work:
    pending: bool = False
    changed: bool = False
    delete_receipts: tuple[str, ...] = ()


def _signature(checkpoint):
    return digest_of(
        sorted((name, value) for name, value in checkpoint.blobs.items() if name.startswith(PREFIX))
    )


def _size(profile, name, sub=store.STATE_DIR):
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\0" in name:
        raise CollectError("state_corrupt", "Original retention child reference is invalid.")
    directory = storage.open_profile_dir(profile, "collector", sub, create=False)
    try:
        descriptor = storage.open_regular(directory, name)
        try:
            return os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _remaining_public(profile, rows, receipt_ids):
    """Check <=32 names; canonical receipt tuples are ordered by ID and revision."""
    from ssgrep.accounting.service_records import RECEIPT_ID_RE

    if not isinstance(receipt_ids, (tuple, list)) or len(receipt_ids) > 32:
        raise CollectError("state_corrupt", "Original receipt cleanup intent is invalid.")
    if not receipt_ids:
        return ()
    groups = (
        (rows.receipts, *(archive.receipts for archive in rows.archives))
        if rows is not None
        else ()
    )
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return ()
    remaining = []
    try:
        for receipt in receipt_ids:
            if not isinstance(receipt, str) or RECEIPT_ID_RE.fullmatch(receipt) is None:
                raise CollectError("state_corrupt", "Original receipt cleanup identity is invalid.")
            protected = False
            for group in groups:
                position = bisect_left(group, receipt, key=lambda row: row.receipt_id)
                if position < len(group) and group[position].receipt_id == receipt:
                    protected = True
                    break
            if protected:
                continue
            try:
                descriptor = storage.open_regular(directory, receipt + ".json")
            except FileNotFoundError:
                continue
            os.close(descriptor)
            remaining.append(receipt)
    finally:
        os.close(directory)
    return tuple(remaining)


def _clean(documents, key, source, cutoff):
    from ssgrep.accounting.published_documents import day

    indexes = documents.get("retention_child_dates")
    index = indexes.get(key) if isinstance(indexes, dict) else None
    if not isinstance(index, dict) or index.get("schema") != 1 or index.get("source") != source:
        return False
    checked = day(index.get("clean_before"))
    if checked is not None and cutoff <= checked:
        return True
    minimum = index.get("minimum")
    return index.get("complete") is True and (
        minimum is None or day(minimum) is not None and cutoff <= minimum
    )


def _next(manifest, name, job):
    """Advance one empty/clean metadata edge, or return the next data quantum."""
    from ssgrep.accounting.published_documents import day

    documents, phase, after = manifest["published_documents"], job["phase"], job["after"]
    if phase == "raw":
        public_receipts = documents.get("receipt_blobs", {}).values()
        candidates = sorted(
            (key, value, store.STATE_DIR)
            for key, value in manifest.get("blobs", {}).items()
            if key > after
            and not key.startswith("published-")
            and value != documents.get("history_blob")
            and value not in public_receipts
        )
        field, following = "blob_names", "segments"
    elif phase == "segments":
        candidates = sorted(
            (ref["name"], ref["name"], store.EVIDENCE_DIR)
            for ref in manifest.get("segments", [])
            if ref["name"] > after
        )
        field, following = "segment_names", "history"
    elif phase == "history":
        candidates = (
            [("history", documents["history_blob"], store.STATE_DIR)]
            if documents.get("history_blob")
            else []
        )
        field, following = "history", "receipts"
    else:
        minimum = day(documents.get("receipt_min_period"))
        receipts = documents.get("receipt_blobs", {})
        if (
            minimum is not None
            and minimum >= job["before_day"]
            and len(documents.get("receipt_periods", {})) == len(receipts)
        ):
            job.update(archive_after=name, phase="raw", after="")
            return None
        candidates = sorted(
            (key, value, store.STATE_DIR) for key, value in receipts.items() if key > after
        )
        field, following = "receipt_ids", None
    if not candidates:
        if following:
            job.update(phase=following, after="")
        else:
            job.update(archive_after=name, phase="raw", after="")
        return None
    first, source, _ = candidates[0]
    cutoff = (
        job["evidence_before_day"]
        if phase == "raw" and first.split("-scope-", 1)[0] == "counters"
        else job["before_day"]
    )
    cache_key = (
        "blob:" + first
        if phase == "raw"
        else "segment:" + first
        if phase == "segments"
        else "history"
    )
    if phase != "receipts" and _clean(documents, cache_key, source, cutoff):
        if phase == "history":
            job.update(phase="receipts", after="")
        else:
            job["after"] = first
        return None
    return field, candidates[: 32 if phase == "receipts" else 1], cutoff


def prepare(
    profile, checkpoint, *, rows, before_day, evidence_before_day, clock, deadline, byte_budget
):
    """One data quantum, <=64 metadata edges, and a <=32-ID postcommit cleanup intent."""
    from ssgrep.accounting.published_documents import day
    from ssgrep.accounting.published_retention import stage_children

    names = sorted(name for name in checkpoint.blobs if name.startswith(PREFIX))
    old = checkpoint.retention.get(KEY, {})
    if not names and not old.get("cleanup_ids"):
        return Work()
    signature = _signature(checkpoint)
    if (
        old.get("before_day") == before_day
        and old.get("evidence_before_day") == evidence_before_day
        and old.get("signature") == signature
    ):
        job = dict(old)
        if not job.get("pending") and not job.get("cleanup_ids"):
            return Work()
    else:
        job = {
            "before_day": before_day,
            "evidence_before_day": evidence_before_day,
            "signature": signature,
            "archive_after": "",
            "phase": "raw",
            "after": "",
            "pending": bool(names),
            "cleanup_ids": old.get("cleanup_ids", []),
        }
    queued = _remaining_public(profile, rows, job.get("cleanup_ids", []))
    job["cleanup_ids"] = list(queued)
    if queued:
        ready = clock() < deadline
        checkpoint.retention[KEY] = job
        return Work(bool(job["pending"] or not ready), False, queued if ready else ())
    changed, consumed, expired_receipts, used_data, edges = False, 0, (), False, 0
    job.pop("blocked_child_bytes", None)
    stop = False
    while job["pending"] and edges < 64 and clock() < deadline and not stop:
        name = next((name for name in names if name > job["archive_after"]), None)
        if name is None:
            job["pending"] = False
            break
        size = _size(profile, checkpoint.blobs[name])
        if consumed + size > byte_budget:
            job["blocked_child_bytes"] = size
            break
        consumed += size
        manifest = store.read_blob(profile, checkpoint.blobs[name])
        if not isinstance(manifest, dict):
            raise CollectError("state_corrupt", "Original publication manifest is invalid.")
        edges += 1
        if manifest.get("published_documents") is None:
            job.update(archive_after=name, phase="raw", after="")
            continue
        while edges < 64 and clock() < deadline and job["archive_after"] != name:
            candidate = _next(manifest, name, job)
            edges += 1
            if candidate is None:
                continue
            if used_data:
                stop = True
                break
            field, candidates, cutoff = candidate
            selected = []
            documents = manifest["published_documents"]
            for key, filename, sub in candidates:
                indexed = (
                    field == "receipt_ids"
                    and day(documents.get("receipt_periods", {}).get(key)) is not None
                    and isinstance(documents.get("receipt_roots", {}).get(key), str)
                )
                size = (
                    0
                    if indexed
                    else _size(profile, filename, sub) * (2 if field == "segment_names" else 1)
                )
                if consumed + size > byte_budget or clock() >= deadline:
                    job["blocked_child_bytes"] = size
                    break
                selected.append(key)
                consumed += size
            if not selected:
                stop = True
                break
            result = stage_children(
                profile,
                checkpoint,
                name=name,
                manifest=manifest,
                before_day=cutoff,
                **{field: True if field == "history" else tuple(selected)},
            )
            manifest = result.manifest
            changed = changed or result.changed
            expired_receipts = result.expired_receipts
            used_data = True
            if field == "history":
                job.update(phase="receipts", after="")
            else:
                job["after"] = selected[-1]
        if job["archive_after"] == name and not any(other > name for other in names):
            job["pending"] = False
    deletions = _remaining_public(profile, rows, expired_receipts)
    job.update(
        cleanup_ids=list(deletions),
        signature=_signature(checkpoint),
        bytes_read_bound=consumed,
        metadata_edges=edges,
    )
    checkpoint.retention[KEY] = job
    return Work(bool(job["pending"]), changed, deletions)


def finish(profile, work):
    """Only explicitly proven stale public receipt files are unlinked after commit."""
    from ssgrep.accounting.purge_files import delete_receipts

    return delete_receipts(profile, set(work.delete_receipts)) if work.delete_receipts else 0
