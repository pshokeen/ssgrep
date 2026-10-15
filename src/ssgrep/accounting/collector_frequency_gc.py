"""Checkpoint-authorized, resumable frozen-view reclamation (metadata-only inventory).

Inventory runs only when scheduled, never scans transcripts, and stores filename
queues in the existing immutable blob store. Reference proof is compiled from
manifest indexes and surviving receipts in bounded quanta. Unknown legacy proof
protects every view instead of guessing that a missing index means no references.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.frequency_retention import prune_views

BLOB = "frequency-maintenance"
VIEW = re.compile(r"frequency-(freq-[0-9a-f]{24})\.json$")
MAX_REFERENCE_BYTES = 4 * 1024 * 1024
REFERENCE_BYTE_BUDGET = 16 * 1024 * 1024
MAX_INVENTORY_ENTRIES = 250_000


@dataclass(frozen=True)
class Outcome:
    pending: bool = False
    removed: int = 0
    warnings: tuple[str, ...] = ()


def _read(profile, filename, sub=store.STATE_DIR):
    directory = (
        storage.open_profile_dir(profile, "collector", sub, create=False)
        if sub == store.STATE_DIR
        else storage.open_profile_dir(profile, sub, create=False)
    )
    try:
        descriptor = storage.open_regular(directory, filename)
        try:
            size = os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
        if size > MAX_REFERENCE_BYTES:
            return None, size
        body = storage.read_file(directory, filename, MAX_REFERENCE_BYTES)
        return store.loads(body) if body is not None else None, size
    finally:
        os.close(directory)


def _receipt_inventory(profile):
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return [], None
    try:
        info = os.fstat(directory)
        names = [name for name in storage.list_names(directory) if name.endswith(".json")]
        return names, [info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns]
    finally:
        os.close(directory)


def _reference_sources(blobs):
    return sorted(
        (name, filename)
        for name, filename in blobs.items()
        if name.startswith("ledger-")
        or name.startswith("identity-archive-")
        or name.split("-scope-", 1)[0] in {"frequency-assignments", "reduction", "partitions"}
    )


def _receipt_stamp(profile):
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(directory)
        return [info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns]
    finally:
        os.close(directory)


def stage(profile, checkpoint, blobs, *, current_refs, before_day, today):
    """Stage queues/reference handles before the acceptance checkpoint, never erase."""
    raw = store.read_blob(profile, checkpoint.blobs.get(BLOB))
    if raw is not None and (not isinstance(raw, dict) or raw.get("schema") != 1):
        raise CollectError("state_corrupt", "Frequency maintenance job is invalid.")
    sources = _reference_sources(blobs)
    # A completed epoch does no directory/body inventory again on the same day.
    source_version = digest_of(sources)
    if raw and raw["today"] == today and raw["phase"] == "complete":
        blobs[BLOB] = checkpoint.blobs[BLOB]
        return False
    if (
        raw
        and raw["phase"] not in {"complete", "blocked"}
        and raw["source_version"] == source_version
        and raw.get("receipt_stamp") == _receipt_stamp(profile)
    ):
        protected = sorted(set(raw["protected"]) | set(current_refs))
        if protected != raw["protected"]:
            raw["protected"] = protected
            blobs[BLOB] = store.write_blob(profile, BLOB, raw)
        else:
            blobs[BLOB] = checkpoint.blobs[BLOB]
        return bool(raw["phase"] != "complete")
    directory = storage.open_profile_dir(profile, "collector")
    try:
        candidates = []
        for name in storage.list_names(directory):
            if VIEW.fullmatch(name) is None:
                continue
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if datetime.fromtimestamp(info.st_mtime, UTC).date().isoformat() < before_day:
                candidates.append(name)
    finally:
        os.close(directory)
    receipts, receipt_stamp = _receipt_inventory(profile) if candidates else ([], None)
    bounded = len(candidates) + len(receipts) + len(sources) <= MAX_INVENTORY_ENTRIES
    if raw and raw["phase"] not in {"complete", "blocked"} and bounded:
        known = {filename for _, filename in raw["sources"]}
        raw["sources"].extend(pair for pair in sources if pair[1] not in known)
        known_receipts = set(raw["receipts"])
        raw["receipts"].extend(name for name in receipts if name not in known_receipts)
        known_candidates = set(raw["candidates"])
        raw["candidates"].extend(
            name for name in sorted(candidates) if name not in known_candidates
        )
        raw["protected"] = sorted(set(raw["protected"]) | set(current_refs))
        raw["source_version"] = source_version
        raw["receipt_stamp"] = receipt_stamp
        raw["phase"] = "references"
        blobs[BLOB] = store.write_blob(profile, BLOB, raw)
        return True
    job = {
        "schema": 1,
        "today": today,
        "before_day": before_day,
        "source_version": source_version,
        "sources": sources,
        "source_cursor": 0,
        "receipt_cursor": 0,
        "receipts": receipts,
        "receipt_stamp": receipt_stamp,
        "candidates": sorted(candidates),
        "candidate_cursor": 0,
        "protected": sorted(set(current_refs)),
        "phase": "references"
        if candidates and bounded
        else "blocked"
        if not bounded
        else "complete",
        "warnings": [] if bounded else ["frequency_inventory_bound"],
    }
    blobs[BLOB] = store.write_blob(profile, BLOB, job)
    return job["phase"] != "complete"


def _collect_refs(name, raw):
    if not isinstance(raw, dict):
        return None, []
    if name.startswith("identity-archive-"):
        if raw.get("schema") != 1:
            return None, []
        return set(), _reference_sources(raw.get("blobs", {}))
    if raw.get("reference_index_complete") is False:
        return None, []
    refs = raw.get("frequency_refs")
    if isinstance(refs, list) and all(isinstance(view, str) for view in refs):
        return set(refs), []
    # No full legacy ledger/reduction/assignment traversal during collection.
    return None, []


def finish(profile, *, generation, limit=32):
    """Resume only accepted work, then durably commit progress at the same generation."""
    checkpoint = store.load_checkpoint(profile)
    if checkpoint is None or checkpoint.generation != generation or BLOB not in checkpoint.blobs:
        return Outcome()
    job = store.read_blob(profile, checkpoint.blobs[BLOB])
    if not isinstance(job, dict) or job.get("schema") != 1:
        raise CollectError("state_corrupt", "Frequency maintenance job is invalid.")
    if job["phase"] == "complete":
        return Outcome()
    if job["phase"] == "blocked":
        return Outcome(True, warnings=tuple(job["warnings"]))
    protected = set(job["protected"])
    charged = visited = 0
    while job["phase"] == "references" and visited < limit:
        if job["source_cursor"] < len(job["sources"]):
            name, filename = job["sources"][job["source_cursor"]]
            raw, size = _read(profile, filename)
            if charged and charged + size > REFERENCE_BYTE_BUDGET:
                break
            refs, children = _collect_refs(name, raw)
            if refs is None:
                job["phase"] = "blocked"
                job["warnings"] = ["frequency_references_unknown"]
                break
            protected.update(refs)
            known = {filename for _, filename in job["sources"]}
            job["sources"].extend(pair for pair in children if pair[1] not in known)
            job["source_cursor"] += 1
        elif job["receipt_cursor"] < len(job["receipts"]):
            filename = job["receipts"][job["receipt_cursor"]]
            try:
                raw, size = _read(profile, filename, "receipts")
            except FileNotFoundError:
                raw, size = {}, 0  # Physical committed retention already removed this receipt.
            if charged and charged + size > REFERENCE_BYTE_BUDGET:
                break
            if raw is None:
                job["phase"] = "blocked"
                job["warnings"] = ["frequency_references_unknown"]
                break
            from ssgrep.accounting.collector_ledger_store import frequency_refs

            protected.update(frequency_refs(raw))
            job["receipt_cursor"] += 1
        else:
            job["phase"] = "pruning"
            break
        charged += size
        visited += 1
    removed = 0
    if job["phase"] == "pruning":
        # A changed receipt inventory needs a new accepted reference-proof queue.
        if job.get("receipt_stamp") != _receipt_stamp(profile):
            return Outcome(True, warnings=("frequency_reference_inventory_changed",))
        start = job["candidate_cursor"]
        names = job["candidates"][start : start + limit]
        result = prune_views(
            profile,
            candidates=names,
            referenced=protected,
            before_day=job["before_day"],
            limit=limit,
        )
        removed = result.removed
        job["candidate_cursor"] += result.scanned
        if job["candidate_cursor"] >= len(job["candidates"]):
            job["phase"] = "complete"
            job["sources"] = []  # Completed proof no longer pins obsolete namespace children.
            job["receipts"] = []
            job["candidates"] = []
    job["protected"] = sorted(protected)
    checkpoint.blobs[BLOB] = store.write_blob(profile, BLOB, job)
    store.commit_checkpoint(profile, checkpoint)
    return Outcome(job["phase"] != "complete", removed, tuple(job["warnings"]))


def references(raw):
    return {filename for _, filename in raw.get("sources", ())}
