"""One-way reading of actual published-session-v1 documents; never current valuation."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any, cast

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.collector_types import CollectError

FORMAT = "published-session-v1"
HISTORY_UNAVAILABLE = "published_history_unavailable"
ORIGINAL_REASON = "original_published_untrusted_model"
ARCHIVE_PREFIX = "identity-archive-"


def day(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return value if date.fromisoformat(value).isoformat() == value else None
    except ValueError:
        return None


def references(document: object) -> set[str]:
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise CollectError("state_corrupt", "Original publication reference map is invalid.")
    history, receipts = document.get("history_blob"), document.get("receipt_blobs", {})
    if history is not None and not isinstance(history, str) or not isinstance(receipts, dict):
        raise CollectError("state_corrupt", "Original publication references are invalid.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in receipts.items()):
        raise CollectError("state_corrupt", "Original receipt references are invalid.")
    return ({history} if history else set()) | set(receipts.values())


def decode_original(raw: bytes, kind: str) -> Any:
    """Admit only closed actual407 publications, before any original-byte envelope write."""
    from ssgrep.accounting.published_schema import decode_published

    return decode_published(raw, kind)


def read_document(profile: str, filename: str, kind: str) -> tuple[dict[str, Any], Any]:
    envelope = store.read_blob(profile, filename)
    if not isinstance(envelope, dict) or envelope.get("kind") != kind:
        raise ValueError("Original publication envelope is invalid.")
    text, expected = envelope.get("original_utf8"), envelope.get("sha256")
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError("Original publication digest does not match.")
    return envelope, decode_original(text.encode(), kind)


def manifests(profile: str, blobs: dict[str, str]):
    for name, filename in sorted(blobs.items()):
        if not name.startswith(ARCHIVE_PREFIX):
            continue
        manifest = store.read_blob(profile, filename)
        documents = manifest.get("published_documents") if isinstance(manifest, dict) else None
        if documents is not None:
            references(documents)
            yield name, manifest, documents


def _original_rows(profile: str, namespace: str, documents: dict[str, Any]):
    filename = documents.get("history_blob")
    if filename is None:
        return [], (HISTORY_UNAVAILABLE,)
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Original publication namespace is unavailable.")
    envelope, originals = read_document(profile, filename, "history")
    return [
        {
            **original,
            "namespace": namespace,
            "current": False,
            "status": "superseded",
            "published_status": original["status"],
            "published_format": FORMAT,
            "revision_name": "published-original:" + namespace,
            "published_sha256": envelope["sha256"],
        }
        for original in originals
    ], ()


def archived_history(
    profile: str, blobs: dict[str, str]
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows, warnings = [], set()
    for _, manifest, documents in manifests(profile, blobs):
        original, missing = _original_rows(profile, manifest.get("methodology"), documents)
        rows.extend(original)
        warnings.update(missing)
    return rows, tuple(sorted(warnings))


def _merge_rows(current, originals, warnings):
    # Bare genuine-v1 rows are replaced only by their verified archive copy, never
    # promoted into a current scenario. Modern current rows retain their identity.
    keys = ("schema", "period", "revision", "supersedes", "net", "kind", "recomputable")

    def key(row):
        return store.dumps({name: row.get(name) for name in keys})

    original_keys = {key(row) for row in originals}
    remaining = [
        row
        for row in current
        if row.get("published_format") != FORMAT
        and not ("format" not in row and key(row) in original_keys)
    ]
    if warnings:
        remaining = [{**row, "published_warnings": list(warnings)} for row in remaining]
    return [*remaining, *originals], warnings


def merge_history(profile: str, current: list[dict[str, Any]], blobs: dict[str, str]):
    return _merge_rows(current, *archived_history(profile, blobs))


def merge_stored_history(profile: str, raw: Any):
    rows = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
    # Current-only publications need no legacy archive lookup. Original wrappers,
    # however, are a derived cache: only the committed surviving archive authorizes
    # them, including during recovery after a purge committed without an active ledger.
    if rows and all(
        row.get("format") == "history_v2"
        and row.get("published_format") != FORMAT
        and not row.get("published_warnings")
        for row in rows
    ):
        return rows, ()
    try:
        checkpoint = store.load_checkpoint(profile)
        if checkpoint is None:
            remaining = [row for row in rows if row.get("published_format") != FORMAT]
            return remaining, (HISTORY_UNAVAILABLE,) if len(remaining) != len(rows) else ()
        merged, warnings = merge_history(profile, rows, checkpoint.blobs)
        capture = checkpoint.identity_migration.get("published_capture")
        if isinstance(capture, str) and not any(
            row.get("published_format") == FORMAT for row in merged
        ):
            document = store.read_blob(profile, capture)
            references(document)
            document = cast(dict[str, Any], document)  # references validated the document mapping.
            namespace = document.get("source_namespace")
            if namespace:
                originals, missing = _original_rows(profile, namespace, document)
                return _merge_rows(rows, originals, tuple(sorted(set(warnings) | set(missing))))
            return merged, (HISTORY_UNAVAILABLE,)
        return merged, warnings
    except CollectError as error:
        raise ValueError("Original publication state cannot be read.") from error


def archived_receipt_ids(profile: str, blobs: dict[str, str]) -> set[str]:
    from ssgrep.accounting.service_records import RECEIPT_ID_RE

    ids = {
        receipt
        for _, _, documents in manifests(profile, blobs)
        for receipt in documents["receipt_blobs"]
    }
    if any(RECEIPT_ID_RE.fullmatch(receipt) is None for receipt in ids):
        raise ValueError("Original receipt identity is invalid.")
    return ids


def restore_receipts(profile: str, blobs: dict[str, str], active: set[str]) -> int:
    written = 0
    allowed = archived_receipt_ids(profile, blobs)
    if not allowed:
        return 0
    directory = storage.open_profile_dir(profile, "receipts")
    try:
        for _, _, documents in manifests(profile, blobs):
            for receipt_id, filename in documents["receipt_blobs"].items():
                if receipt_id not in allowed:
                    raise ValueError("Original receipt identity is invalid.")
                envelope, receipt = read_document(profile, filename, "receipt")
                if receipt.get("receipt_id") != receipt_id:
                    raise ValueError("Original receipt identity does not match.")
                body = envelope["original_utf8"].encode()
                existing = storage.read_file(directory, receipt_id + ".json", len(body) + 1)
                if receipt_id in active and existing != body:
                    raise ValueError("Original and current receipt identities conflict.")
                if existing != body:
                    storage.atomic_write(directory, receipt_id + ".json", body)
                    written += 1
    finally:
        import os

        os.close(directory)
    return written
