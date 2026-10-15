"""Prune retained original publication inputs without decoding current valuation types."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.published_documents import day
from ssgrep.accounting.purge_select import Selection

if TYPE_CHECKING:
    from ssgrep.accounting.purge_published import PublishedPlan


def prune_inputs(
    profile: str,
    plan: PublishedPlan,
    sel: Selection,
    *,
    atomic: bool,
    selected_record,
    note_remaining,
) -> dict[str, tuple[str, dict[str, Any]]]:
    from ssgrep.accounting.purge_published import _unresolved

    manifest = plan.manifest
    history = plan.documents.get("history_blob")
    selected = plan.scopes.selected
    reductions = {}
    for name, filename in manifest.get("blobs", {}).items():
        role = name.split("-scope-", 1)[0]
        alias = "-scope-" in name
        if (
            name.startswith("published-")
            or filename == history
            or filename in manifest["published_documents"].get("receipt_blobs", {}).values()
        ):
            continue
        if role.startswith("ledger-"):
            from ssgrep.accounting.collector_ledger_store import expand_document

            source = store.read_blob(profile, filename)
            if not isinstance(source, dict):
                raise ValueError("Original ledger input is invalid.")
            if atomic and source.get("format") is not None:
                _unresolved("Original sharded input needs child-index maintenance before expiry.")
            raw = expand_document(profile, source)
            if atomic:
                plan.selected_inputs[filename] = raw
            state = raw["state"]
            if raw.get("archives"):
                _unresolved("Original extra revision archives require published-format recovery.")
            updated = dict(state)
            deleted_hashes = set()
            for kind in ("candidates", "debits"):
                kept = []
                for record in state.get(kind, []):
                    if selected_record(record, scope_alias=alias):
                        deleted_hashes.add(hashlib.sha256(store.dumps(record)).hexdigest())
                        plan.ledger_items += 1
                    else:
                        kept.append(record)
                updated[kind] = kept
                plan.remaining_records += len(kept)
                for row in kept:
                    note_remaining(row)
            frozen = []
            for root in state.get("frozen", []):
                leaves = [
                    row
                    for row in root.get("leaves", [])
                    if not selected_record(row, scope_alias=alias)
                ]
                records = [
                    row
                    for row in root.get("receipts", [])
                    if not selected_record(row, scope_alias=alias)
                ]
                plan.ledger_items += len(root.get("leaves", [])) - len(leaves)
                plan.remaining_records += len(leaves) + len(records)
                for row in (*leaves, *records):
                    note_remaining(row)
                if leaves or records:
                    frozen.append({**root, "leaves": leaves, "receipts": records})
            updated["frozen"] = frozen
            if updated != state:
                plan.updates[filename] = (
                    name,
                    {
                        **raw,
                        "state": updated,
                        "superseded": {
                            key: value
                            for key, value in raw.get("superseded", {}).items()
                            if key not in deleted_hashes
                        },
                    },
                )
        elif role == "reduction":
            from ssgrep.accounting.collector_reduction_store import expand_document

            source = store.read_blob(profile, filename)
            if not isinstance(source, dict):
                raise ValueError("Original reduction input is invalid.")
            if atomic and source.get("format") is not None:
                _unresolved(
                    "Original sharded reduction needs child-index maintenance before expiry."
                )
            raw = expand_document(profile, source)
            if atomic:
                plan.selected_inputs[filename] = raw
            local_roots = {
                (runtime, session): root for runtime, session, root in raw.get("session_roots", [])
            }
            updated = dict(raw)
            for key, counter in (
                ("executions", "executions_removed"),
                ("invocations", "invocations_removed"),
            ):
                rows = raw.get(key, [])
                kept = [row for row in rows if not selected_record(row, scope_alias=alias)]
                setattr(plan, counter, getattr(plan, counter) + len(rows) - len(kept))
                if key in raw:
                    updated[key] = kept
                plan.remaining_records += len(kept)
                for row in kept:
                    note_remaining(row)
                    local_root = local_roots.get((row.get("runtime"), row.get("session_id")))
                    if local_root is not None:
                        plan.kept_roots.add(local_root)
            if "session_last" in raw:
                kept_last = []
                for runtime, session, stamp in raw["session_last"]:
                    root = local_roots.get((runtime, session))
                    old = isinstance(stamp, str) and stamp[:10] < sel.cutoff_day
                    if old and (selected is None or root in selected):
                        plan.archive_metadata_records += 1
                    else:
                        kept_last.append([runtime, session, stamp])
                        if root is not None:
                            plan.kept_roots.add(root)
                updated["session_last"] = kept_last
                plan.remaining_records += len(kept_last)
            reductions[filename] = (name, updated)
            if updated != raw:
                plan.updates[filename] = (name, updated)
        elif role == "counters" and selected is None:
            raw = store.read_blob(profile, filename)
            if not isinstance(raw, dict) or not isinstance(raw.get("counted", []), list):
                raise ValueError("Original counter linkage shape is invalid.")
            counted = raw.get("counted", [])
            if any(
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or day(row[1]) is None
                for row in counted
            ):
                raise ValueError("Original counter linkage date is invalid.")
            kept = [row for row in counted if row[1] >= sel.cutoff_day]
            if atomic:
                plan.selected_inputs[filename] = raw
            if len(kept) != len(counted):
                plan.archive_metadata_records += len(counted) - len(kept)
                plan.updates[filename] = (name, {**raw, "counted": kept})
        elif role == "sealed" and selected is None:
            raw = store.read_blob(profile, filename)
            if not isinstance(raw, dict):
                raise ValueError("Original sealed input is invalid.")
            if atomic:
                plan.selected_inputs[filename] = raw
            updated = {
                key: value
                for key, value in raw.items()
                if day(key) is None or key >= sel.cutoff_day
            }
            plan.sealed_removed += tuple(key for key in raw if key not in updated)
            if updated != raw:
                plan.updates[filename] = (name, updated)
    return reductions
