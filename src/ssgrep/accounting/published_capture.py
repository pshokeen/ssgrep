"""Bounded byte-preserving capture of already published original documents."""

from __future__ import annotations

import hashlib
import os
from typing import Any, cast

from ssgrep.accounting import collector_store as store, snapshots, storage
from ssgrep.accounting.collector_state import ledger_blob_name
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.json_io import MAX_SNAPSHOT_BYTES
from ssgrep.accounting.service_records import MAX_RECEIPT_BYTES, RECEIPT_ID_RE


def _read_bounded(profile, sub, name, *, limit, budget):
    """Pin, stat and gate a document before reading any of its body bytes."""
    try:
        directory = storage.open_profile_dir(profile, sub, create=False)
        try:
            descriptor = storage.open_regular(directory, name)
            try:
                before = os.fstat(descriptor)
                size = before.st_size
                if size > limit:
                    raise CollectError(
                        "migration_history_unavailable",
                        "An original published document exceeds its supported size.",
                    )
                if size > budget:
                    return None, size
                chunks, remaining = [], size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 64 * 1024))
                    if not chunk:
                        raise CollectError(
                            "migration_history_unavailable",
                            "An original published document was truncated.",
                        )
                    chunks.append(chunk)
                    remaining -= len(chunk)
                after = os.fstat(descriptor)
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise CollectError(
                        "migration_history_unavailable",
                        "An original published document changed during capture.",
                    )
                return b"".join(chunks), size
            finally:
                os.close(descriptor)
        finally:
            os.close(directory)
    except (ValueError, OSError) as error:
        raise CollectError(
            "migration_history_unavailable", "Original published documents cannot be verified."
        ) from error


def _envelope(profile, raw, kind, generation, stamp):
    from ssgrep.accounting.published_documents import decode_original

    try:
        text = raw.decode("utf-8")
        decode_original(raw, kind)
    except (ValueError, UnicodeError) as error:
        raise CollectError(
            "migration_history_unavailable", "An original published document is damaged."
        ) from error
    return store.write_blob(
        profile,
        "published-original",
        {
            "original_utf8": text,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "kind": kind,
            "source_generation": generation,
            "original_collected_at": stamp,
        },
    )


def _roots_by_period(state):
    roots = {}
    for row in (*state.get("candidates", []), *state.get("debits", [])):
        day, root = row.get("day"), row.get("task", {}).get("root")
        if not isinstance(day, str) or not isinstance(root, str):
            return None
        roots.setdefault(day, set()).add(root)
    for frozen in state.get("frozen", []):
        root = frozen.get("root")
        for leaf in frozen.get("leaves", []):
            day = leaf.get("day")
            if not isinstance(day, str) or not isinstance(root, str):
                return None
            roots.setdefault(day, set()).add(root)
    return {day: sorted(values) for day, values in sorted(roots.items())}


def stage(profile, checkpoint, *, limit=32, byte_budget=32 * 1024 * 1024):
    """Return (complete, bytes read); history and receipts share both budgets."""
    limit = max(0, min(limit, 32))
    job = checkpoint.identity_migration
    ref = job.get("published_capture")
    document: dict[str, Any] | None = store.read_blob(profile, ref) if ref else None
    if document is None:
        original = store.read_blob(
            profile, checkpoint.blobs.get(ledger_blob_name(checkpoint.methodology))
        )
        method = (
            original.get("state", {}).get("methodology", {}) if isinstance(original, dict) else {}
        )
        if not isinstance(method, dict) or "valuation_version" in method or not method:
            return True, 0
        snapshot = snapshots.read_generation(profile, checkpoint.generation)
        if snapshot is None:
            raise CollectError(
                "migration_history_unavailable", "Original publication is unavailable."
            )
        document = {
            "format": "published-session-v1",
            "source_generation": checkpoint.generation,
            "source_namespace": checkpoint.methodology,
            "original_collected_at": snapshot.collected_at,
            "history_blob": None,
            "roots_by_period": _roots_by_period(cast(dict[str, Any], original)["state"]),
            "receipt_periods": {},
            "receipt_roots": {},
            "receipt_min_period": None,
            "receipt_blobs": {},
            "after": "",
            "complete": False,
        }
    if document.get("complete"):
        return True, 0
    consumed, processed = 0, 0
    document.pop("blocked_child_bytes", None)
    document.pop("blocked_by", None)
    if document["history_blob"] is None:
        raw, size = (
            (None, 0)
            if not limit
            else _read_bounded(
                profile, "history", "periods.json", limit=MAX_SNAPSHOT_BYTES, budget=byte_budget
            )
        )
        if raw is None:
            document.update(
                blocked_child_bytes=size, blocked_by="byte_budget" if limit else "document_budget"
            )
            job["published_capture"] = store.write_blob(profile, "published-capture", document)
            return False, 0
        document["history_blob"] = _envelope(
            profile,
            raw,
            "history",
            document["source_generation"],
            document["original_collected_at"],
        )
        consumed, processed = len(raw), 1
    directory = storage.open_profile_dir(profile, "receipts")
    try:
        names = storage.list_names(directory)
    finally:
        os.close(directory)
    if len(names) > 250_000:
        raise CollectError(
            "migration_history_unavailable",
            "Original receipt inventory exceeds the migration bound.",
        )
    pending = False
    for name in names:
        if name <= document["after"]:
            continue
        if processed >= limit:
            pending = True
            document["blocked_by"] = "document_budget"
            break
        if not name.endswith(".json") or not RECEIPT_ID_RE.fullmatch(name[:-5]):
            document["after"] = name
            processed += 1
            continue
        raw, size = _read_bounded(
            profile, "receipts", name, limit=MAX_RECEIPT_BYTES, budget=byte_budget - consumed
        )
        if raw is None:
            document.update(blocked_child_bytes=size, blocked_by="byte_budget")
            pending = True
            break
        consumed += len(raw)
        processed += 1
        try:
            receipt = store.loads(raw)
        except ValueError as error:
            raise CollectError(
                "migration_history_unavailable",
                "An original receipt cannot be decoded or safely assigned to another namespace.",
            ) from error
        if not isinstance(receipt, dict):
            raise CollectError("migration_history_unavailable", "An original receipt is invalid.")
        if receipt.get("methodology") == checkpoint.methodology:
            if receipt.get("receipt_id") != name[:-5]:
                raise CollectError(
                    "migration_history_unavailable",
                    "Original receipt identity differs from its file.",
                )
            document["receipt_blobs"][name[:-5]] = _envelope(
                profile,
                raw,
                "receipt",
                document["source_generation"],
                document["original_collected_at"],
            )
            document["receipt_periods"][name[:-5]] = receipt["period"]
            document["receipt_roots"][name[:-5]] = receipt["root"]
            minimum = document.get("receipt_min_period")
            document["receipt_min_period"] = (
                min(minimum, receipt["period"]) if minimum else receipt["period"]
            )
        document["after"] = name
    document["complete"] = not pending
    job["published_capture"] = store.write_blob(profile, "published-capture", document)
    return not pending, consumed


def captured_documents(profile, checkpoint):
    ref = checkpoint.identity_migration.get("published_capture")
    document = store.read_blob(profile, ref) if ref else None
    if not document:
        return None
    if not document.get("complete"):
        raise CollectError(
            "migration_history_unavailable", "Original publication capture is incomplete."
        )
    return {key: value for key, value in document.items() if key not in {"after", "complete"}}
