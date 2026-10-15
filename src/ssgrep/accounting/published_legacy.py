"""One-way admission of the published session-scoped ledger, never old valuation.

The original publication is opaque evidence. Its methodology digest is checked
before migration; no current dataclass decoder interprets its old work units.
"""

from __future__ import annotations

from ssgrep.accounting import collector_store as store, snapshots
from ssgrep.accounting.collector_state import ledger_blob_name
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.native_identity import IDENTITY_METHOD

_METHOD_FIELDS = {
    "label",
    "units",
    "thresholds",
    "frequency_view",
    "tokenizer",
    "pricing",
    "calibrated",
}


def validate_checkpoint(profile: str, checkpoint: store.Checkpoint) -> bool:
    """Recognize only the actual published method shape and verify its namespace.

    False delegates to the strict current decoder. True deliberately skips orphan
    collection until identity migration has committed all old authoritative refs.
    """
    if not checkpoint.methodology or checkpoint.identity_method == IDENTITY_METHOD:
        return False
    document = store.read_blob(
        profile, checkpoint.blobs.get(ledger_blob_name(checkpoint.methodology))
    )
    if not isinstance(document, dict):
        return False
    state = document.get("state")
    method = state.get("methodology") if isinstance(state, dict) else None
    if not isinstance(method, dict) or set(method) != _METHOD_FIELDS:
        return False
    if digest_of(method)[:32] != checkpoint.methodology:
        raise CollectError(
            "state_corrupt", "Published ledger methodology differs from its checkpoint."
        )
    if not isinstance(method["label"], str) or not isinstance(method["units"], list):
        raise CollectError("state_corrupt", "Published methodology fields are invalid.")
    if not all(isinstance(state.get(key), list) for key in ("candidates", "debits", "frozen")):
        raise CollectError("state_corrupt", "Published ledger rows are invalid.")
    if set(state) != {"methodology", "candidates", "debits", "frozen"}:
        raise CollectError(
            "unsupported_schema", "Published ledger state contains unsupported fields."
        )
    if set(document) - {"state", "superseded", "collection_scope"}:
        raise CollectError("state_corrupt", "Published ledger contains unsupported fields.")
    # A bad inactive pointer must not become authorization to destroy valid old
    # children. Read every committed blob before any destructive recovery action.
    for reference in checkpoint.blobs.values():
        store.read_blob(profile, reference)
    if checkpoint.generation:
        try:
            published = snapshots.read_generation(profile, checkpoint.generation)
        except (ValueError, OSError) as error:
            raise CollectError("state_corrupt", "Published snapshot cannot be verified.") from error
        if published is None or published.methodology_id != checkpoint.methodology:
            raise CollectError(
                "state_corrupt", "Published snapshot namespace differs from its checkpoint."
            )
    return True
