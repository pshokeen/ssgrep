"""Legacy identity manifests are purgeable state, never untracked privacy archives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.evidence_references import ARCHIVE_PREFIX
from ssgrep.accounting.purge_model import PurgeError
from ssgrep.accounting.purge_select import Selection


@dataclass
class LegacyArchive:
    name: str
    raw: dict[str, Any]
    plan: Any


def plan_archives(
    profile: str, checkpoint: store.Checkpoint, sel: Selection
) -> list[LegacyArchive]:
    from ssgrep.accounting.purge_state import plan_state

    found = []
    for name, filename in checkpoint.blobs.items():
        if not name.startswith(ARCHIVE_PREFIX):
            continue
        raw = store.read_blob(profile, filename)
        if not isinstance(raw, dict) or raw.get("schema") != 1:
            raise PurgeError("state_corrupt", "Identity archive manifest is invalid.")
        if any(key.startswith(ARCHIVE_PREFIX) for key in raw.get("blobs", {})):
            raise PurgeError(
                "archive_scope_unresolved",
                "Nested identity archives require recovery before purge.",
            )
        if sel.workspace is not None and raw.get("root_mapping_status") != "available":
            raise PurgeError(
                "archive_scope_unresolved",
                "Legacy identity archive has no verified workspace/root mapping. "
                "Recover its mapping or request a whole-profile purge; nothing was changed.",
            )
        if raw.get("published_documents") is not None:
            from ssgrep.accounting.purge_published import plan_published

            found.append(LegacyArchive(name, raw, plan_published(profile, raw, sel)))
            continue
        blobs = dict(raw.get("blobs", {}))
        archived = store.Checkpoint(
            profile,
            methodology=str(raw.get("methodology", "")),
            blobs=blobs,
            segments=[store.SegmentRef.from_dict(ref) for ref in raw.get("segments", [])],
            collection_scope=dict(raw.get("collection_scope", {})),
        )
        planned = plan_state(profile, sel, archived)
        if planned is not None:
            found.append(LegacyArchive(name, raw, planned))
    return found


def write_archives(
    profile: str, archives: list[LegacyArchive], sel: Selection, blobs: dict[str, str]
) -> dict[str, str]:
    from ssgrep.accounting.purge_state import write_purged_state

    retained = {}
    for archive in archives:
        from ssgrep.accounting.purge_published import PublishedPlan, write_published

        if isinstance(archive.plan, PublishedPlan):
            retained.update(write_published(profile, archive.name, archive.plan, sel, blobs))
            continue
        if not archive.plan.changed:
            retained.update(archive.raw.get("blobs", {}))
            continue
        new = write_purged_state(profile, archive.plan, sel)
        has_rows = any(
            namespace.rows_after.periods
            or namespace.rows_after.receipts
            or namespace.archives_after
            for namespace in archive.plan.namespaces
        )
        has_invocations = bool(
            archive.plan.reduction_after and archive.plan.reduction_after.invocations
        )
        if not new.segments and not has_rows and not has_invocations:
            blobs.pop(archive.name, None)
            continue
        document = {
            **archive.raw,
            "blobs": new.blobs,
            "segments": [ref.to_dict() for ref in new.segments],
        }
        if archive.plan.reduction_after is not None:
            document["session_roots"] = list(archive.plan.reduction_after.session_roots)
            document["session_cwd"] = list(archive.plan.reduction_after.session_cwd)
        blobs[archive.name] = store.write_blob(profile, archive.name, document)
        retained.update(new.blobs)
    return retained


def row_receipt_ids(rows) -> set[str]:
    return {receipt.receipt_id for receipt in rows.receipts} | {
        receipt.receipt_id for archive in rows.archives for receipt in archive.receipts
    }


def retained_receipt_ids(archives: list[LegacyArchive]) -> set[str]:
    """Only original or canonical receipts surviving the selection remain public."""
    from ssgrep.accounting.purge_published import PublishedPlan

    kept = set()
    for archive in archives:
        if isinstance(archive.plan, PublishedPlan):
            kept.update(archive.plan.documents["receipt_blobs"])
        else:
            for namespace in archive.plan.namespaces:
                kept.update(row_receipt_ids(namespace.rows_after))
    return kept
