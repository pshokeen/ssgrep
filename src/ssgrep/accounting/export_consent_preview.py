"""Read-only ``--preview`` view of the exact sanitized candidate for one destination/signal.

Wraps ``delivery_candidate.preview_candidate`` (which writes nothing: no key, no scope, no
registry/spool/delivery state) into plain data for the renderers. Request bytes are the
candidate's own bytes; hashes are computed over exactly those bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from ssgrep.accounting import collector_store, delivery_candidate, export_policy, snapshots, storage
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.export_authority import migration_of
from ssgrep.accounting.export_consent import host_display, require_target
from ssgrep.accounting.export_model import FIELDS_VERSION
from ssgrep.accounting.export_records_model import RecordsUnavailable
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


def _empty_snapshot_store(profile: str) -> bool:
    """Prove absence without mistaking an unreadable pointer or unsafe directory for setup."""
    try:
        directory = storage.open_profile_dir(profile, snapshots.SNAPSHOT_DIR, create=False)
    except FileNotFoundError:
        return True
    try:
        raw = storage.read_file(directory, snapshots.CURRENT_NAME, snapshots.MAX_SNAPSHOT_BYTES)
        if raw is not None:
            snapshots.decode(raw)  # unlike current_generation, corruption must not mean absence
            return False
        return not storage.list_names(directory)
    finally:
        os.close(directory)


def build_preview(profile: str, destination: str, signal: Signal) -> PreviewView:
    """The exact sanitized candidate, read-only. Raises ``PolicyError`` for refused targets."""
    authority = require_target(profile, destination, signal)
    try:
        checkpoint = collector_store.load_checkpoint(profile)
    except CollectError as error:
        raise RecordsUnavailable(error.condition) from None
    migration = migration_of(checkpoint)
    if migration.reason:
        dest = next(d for d in authority.policy.destinations if d.destination_id == destination)
        if checkpoint is None and _empty_snapshot_store(profile):
            return PreviewView(
                profile,
                destination,
                signal,
                "no_snapshot",
                None,
                authority.policy.policy_version,
                host_display(dest.endpoint),
                FIELDS_VERSION[signal.value],
                False,
                None,
                None,
                0,
                (),
                (),
            )
        return PreviewView(
            profile,
            destination,
            signal,
            migration.reason,
            migration.reason,
            authority.policy.policy_version,
            host_display(dest.endpoint),
            FIELDS_VERSION[signal.value],
            False,
            None,
            None,
            0,
            ("current_projection_unavailable_until_identity_migration",),
            (),
        )
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
