"""Read-only ``--preview`` view of the exact sanitized candidate for one destination/signal.

Wraps ``delivery_candidate.preview_candidate`` (which writes nothing: no key, no scope, no
registry/spool/delivery state) into plain data for the renderers. Request bytes are the
candidate's own bytes; hashes are computed over exactly those bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ssgrep.accounting import delivery_candidate, export_policy
from ssgrep.accounting.export_consent import host_display, require_target
from ssgrep.accounting.export_model import FIELDS_VERSION
from ssgrep.accounting.ledger_contracts import Signal


@dataclass(frozen=True)
class BatchView:
    batch_id: str
    bytes: int
    items: int
    sha256: str
    body: bytes

    def payload(self) -> Any:
        return json.loads(self.body)


@dataclass(frozen=True)
class PreviewView:
    profile: str
    destination: str
    signal: Signal
    state: str
    reason: str | None
    policy_version: int
    endpoint_host: str | None
    fields_version: str
    ephemeral_key: bool
    generation: int | None
    snapshot_id: str | None
    omitted: int
    notes: tuple[str, ...]
    batches: tuple[BatchView, ...]

    @property
    def total_bytes(self) -> int:
        return sum(b.bytes for b in self.batches)

    @property
    def total_items(self) -> int:
        return sum(b.items for b in self.batches)


def build_preview(profile: str, destination: str, signal: Signal) -> PreviewView:
    """The exact sanitized candidate, read-only. Raises ``PolicyError`` for refused targets."""
    require_target(profile, destination, signal)
    result = delivery_candidate.preview_candidate(profile, destination, signal)
    dest = result.destination
    built = result.candidate
    batches = tuple(
        BatchView(b.batch_id, len(b.body), b.items, hashlib.sha256(b.body).hexdigest(), b.body)
        for b in (built.batches if built is not None else ())
    )
    reason = None
    if result.state == "policy_blocked":
        reason = export_policy.gate(
            profile, destination, signal, need_enabled=False, need_unpaused=False
        ).reason
    return PreviewView(
        profile=profile,
        destination=destination,
        signal=signal,
        state=result.state,
        reason=reason,
        policy_version=result.policy.policy_version,
        endpoint_host=host_display(dest.endpoint) if dest is not None else None,
        fields_version=FIELDS_VERSION[signal.value],
        ephemeral_key=result.ephemeral_key,
        generation=built.generation if built is not None else None,
        snapshot_id=built.snapshot_id if built is not None else None,
        omitted=built.omitted if built is not None else 0,
        notes=tuple(built.notes) if built is not None else (),
        batches=batches,
    )
