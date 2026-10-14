"""Derived-file cleanup that follows a committed purge; idempotent and state-driven.

It looks only at the COMMITTED collector state, never at a plan, so a run that crashed after the
checkpoint commit is completed by simply running it again: receipt files that no longer belong to
the ledger go, the receipt/history files are rewritten from the purged ledger, every snapshot
generation older than the committed one is deleted (aggregates cannot be split per workspace or
date), acknowledged pins on deleted snapshots are released, and evidence segments or state blobs
no committed checkpoint names are swept.
"""

from __future__ import annotations

from dataclasses import dataclass

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_publish import history_document, write_history, write_receipts
from ssgrep.accounting.collector_state import load_state
from ssgrep.accounting.purge_files import (
    delete_generations,
    delete_receipts,
    generation_files,
    receipt_ids,
)
from ssgrep.accounting.purge_spool import release_pins


@dataclass(frozen=True)
class Cleaned:
    receipt_files: int = 0
    snapshot_generations: int = 0
    delivery_pins_released: int = 0
    orphans_swept: int = 0


def _committed(profile: str, checkpoint: store.Checkpoint) -> VersionedLedger | None:
    state = load_state(profile, checkpoint, checkpoint.methodology)
    if state.ledger is None:
        return None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded)


def stale_receipts(profile: str, ledger: VersionedLedger) -> set[str]:
    """Receipt files whose receipt is not in the committed ledger (read-only)."""
    return receipt_ids(profile) - {r.receipt_id for r in ledger.rows().receipts}


def older_generations(profile: str, committed: int) -> list:
    return [g for g in generation_files(profile) if g.number < committed]


def cleanup(profile: str, *, today: str, publish: bool, generations: bool) -> Cleaned:
    """Remove stale derived files; ``publish`` also rewrites receipts/history from the ledger.

    ``generations`` deletes every snapshot older than the committed one (set after a purge
    committed a fresh generation, or when an unfinished purge is being completed).
    """
    checkpoint = store.load_checkpoint(profile)
    if checkpoint is None:
        return Cleaned()
    ledger = _committed(profile, checkpoint)
    receipts = 0
    if ledger is not None:
        receipts = delete_receipts(profile, stale_receipts(profile, ledger))
        if publish:
            rows = ledger.rows()
            write_receipts(profile, rows)
            write_history(profile, history_document(rows, ledger.methodology.units[0], today))
    removed = 0
    released = 0
    if ledger is not None and generations:
        removed = delete_generations(profile, older_generations(profile, checkpoint.generation))
        surviving = frozenset(g.snapshot_id for g in generation_files(profile))
        released = release_pins(profile, surviving)
    swept = store.sweep_orphans(profile, checkpoint) if publish else 0
    return Cleaned(receipts, removed, released, swept)
