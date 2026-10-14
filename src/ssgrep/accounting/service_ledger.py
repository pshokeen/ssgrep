"""Read-only access to the collector's committed ledger for filtered savings views.

``collector/checkpoint.json`` names content-addressed state blobs; this module loads the
reduction document (retained executions, session lineage, working directories) and the
versioned ledger of the checkpoint's methodology and returns canonical ledger rows. Nothing
is written, created, repaired or recomputed: every failure becomes a stable reason string so
the facade can say "filters unavailable (reason)" instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ssgrep.accounting import collector_state, collector_store, storage
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.replay import LedgerRows
from ssgrep.accounting.valuation_inputs import Candidate, Debit, Methodology


@dataclass(frozen=True)
class LedgerData:
    generation: int
    methodology: str
    rows: LedgerRows
    doc: ReductionDoc | None
    sealed_days: tuple[str, ...]
    methodology_spec: Methodology
    candidates: tuple[Candidate, ...]
    debits: tuple[Debit, ...]
    evidence_expired_before: str | None


def _reason(error: Exception) -> str:
    if isinstance(error, CollectError):
        return str(error.condition)
    if isinstance(error, storage.UnsafeStorageError):
        return "storage_unsafe"
    return "state_corrupt"


def load_reduction(profile: str) -> tuple[ReductionDoc | None, str | None]:
    """Only the last reduction document (executions + lineage); never the ledger."""
    try:
        checkpoint = collector_store.load_checkpoint(profile)
        if checkpoint is None:
            return None, "no_checkpoint"
        raw = collector_store.read_blob(profile, checkpoint.blobs.get(collector_state.REDUCTION))
        if raw is None:
            return None, "no_reduction"
        return ReductionDoc.from_dict(raw), None
    except (
        CollectError,
        storage.UnsafeStorageError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as error:
        return None, _reason(error)


def load_ledger(profile: str) -> tuple[LedgerData | None, str | None]:
    """The committed ledger rows with the reduction document, or ``(None, reason)``."""
    try:
        checkpoint = collector_store.load_checkpoint(profile)
        if checkpoint is None or not checkpoint.methodology:
            return None, "no_checkpoint"
        state = collector_state.load_state(profile, checkpoint, checkpoint.methodology)
        if state.ledger is None:
            return None, "no_ledger"
        ledger = VersionedLedger.restore_versioned(state.ledger, state.superseded)
        if ledger.methodology.hash != checkpoint.methodology:
            return None, "state_corrupt"
        retention = checkpoint.retention if isinstance(checkpoint.retention, dict) else {}
        expired = retention.get("evidence_expired_before")
        return (
            LedgerData(
                generation=checkpoint.generation,
                methodology=checkpoint.methodology,
                rows=ledger.rows(),
                doc=state.reduction,
                sealed_days=tuple(sorted(state.sealed)),
                methodology_spec=state.ledger.methodology,
                candidates=state.ledger.candidates,
                debits=state.ledger.debits,
                evidence_expired_before=expired if isinstance(expired, str) else None,
            ),
            None,
        )
    except (
        CollectError,
        storage.UnsafeStorageError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        AssertionError,
    ) as error:
        return None, _reason(error)
