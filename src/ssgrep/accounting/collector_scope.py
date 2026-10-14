"""Scope changes isolate old derived totals before any new provenance is attached.

Archived blobs remain private and purgeable, but are never active/exportable. The new
ledger rederives only permitted retained evidence; cumulative counters start a new epoch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_state import COUNTERS, decode_ledger, ledger_blob_name
from ssgrep.accounting.counters import LiveCounters, start_epoch
from ssgrep.accounting.journal_reader import JournalRead
from ssgrep.accounting.policy_rules import valid_collection_scope


def prepare_scope(
    profile: str,
    checkpoint: store.Checkpoint,
    scope: dict[str, Any],
    journal: JournalRead,
    now: datetime,
) -> bool:
    """Initialize scope or rotate existing state; callers hold the worker lease."""
    old = checkpoint.collection_scope
    changed = checkpoint.generation > 0 and (not valid_collection_scope(old) or old != scope)
    if checkpoint.generation > 0 and checkpoint.methodology:
        raw = store.read_blob(
            profile, checkpoint.blobs.get(ledger_blob_name(checkpoint.methodology))
        )
        if raw is not None:
            provenance = raw.get("collection_scope") if isinstance(raw, dict) else None
            changed = changed or provenance != scope
            if changed:
                decode_ledger(raw)
    if changed:
        previous = store.read_blob(profile, checkpoint.blobs.get(COUNTERS))
        counters = LiveCounters.from_dict(previous) if previous is not None else None
        epoch = checkpoint.scope_epoch
        for name, file_name in list(checkpoint.blobs.items()):
            if "-scope-" not in name:
                checkpoint.blobs[f"{name}-scope-{epoch}"] = file_name
                del checkpoint.blobs[name]
        reset = start_epoch(counters, journal.events, now)
        checkpoint.blobs[COUNTERS] = store.write_blob(profile, COUNTERS, reset.to_dict())
        checkpoint.scope_epoch += 1
        checkpoint.backfills = []
        checkpoint.input_digest = ""
    checkpoint.collection_scope = scope
    return changed
