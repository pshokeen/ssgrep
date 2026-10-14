"""Build the exact sanitized candidate batches for one destination/signal (no I/O except reads).

Preview, the delivery cycle and the tombstone writer all go through here, so what ``--preview``
prints is byte-for-byte what a send would spool (for the same snapshot, policy and key):

* metrics: ``export.project_metrics`` (positive allowlist) -> ``otlp.split_metrics``;
* logs: ledger rows -> ``export_records.project_candidates`` -> ``plan_revisions`` against the
  registry (only NEW or CHANGED records, as higher revisions) -> ``otlp.split_logs``;
* tombstones: registry-derived deletion bodies -> ``otlp.split_logs`` with ``t...`` batch ids.

Nothing here creates state: the pseudonym key is only read (``ephemeral_key`` stands in when it
does not exist yet, and the candidate says so), the registry is only loaded, and ledger rows are
only read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from ssgrep.accounting import export_ids, export_records, otlp
from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.export import (
    Projection,
    installed_version,
    project_metrics,
    unix_nano,
)
from ssgrep.accounting.export_model import Batch, Resource
from ssgrep.accounting.export_policy import ScopeState, scope_document, snapshot_eligible
from ssgrep.accounting.export_records_model import Planned
from ssgrep.accounting.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
)

TOMBSTONE_GENERATION = 0


@dataclass(frozen=True)
class Candidate:
    """The sanitized candidate for one destination/signal and what it was derived from."""

    signal: Signal
    destination: str
    generation: int
    snapshot_id: str
    batches: tuple[Batch, ...]
    planned: tuple[tuple[Planned, ...], ...]  # per batch (logs only)
    scope: dict[str, object]
    omitted: int = 0
    notes: tuple[str, ...] = ()
    ephemeral_key: bool = False
    policy_version: int = 1


def key_for(profile: str, *, create: bool) -> tuple[bytes, bool]:
    """(pseudonym key, ephemeral?) — read-only unless ``create`` (only the send path creates)."""
    key = export_ids.load_key(profile, create=create)
    if key is None:
        return export_ids.ephemeral_key(), True
    return key, False


def _batch_id(generation: int, snapshot_id: str, index: int) -> str:
    return f"g{generation:010d}-{snapshot_id[:16]}-{index:04d}"


def _resource(
    profile: str,
    destination: str,
    policy: ProfilePolicy,
    key: bytes,
    envelope: SnapshotEnvelope | None,
) -> Resource:
    return Resource(
        installed_version(),
        export_ids.installation_id(key, destination, profile),
        policy.policy_version,
        counter_epoch=envelope.counter_epoch if envelope is not None else None,
    )


def _require_collection_scope(envelope: SnapshotEnvelope, scope: dict, version: int) -> None:
    state = ScopeState("", None, scope.get("collection_scope"))
    if not snapshot_eligible(envelope, state, version):
        raise ValueError("ineligible_scope: recollect under current workspace ownership")


def build_metrics(
    profile: str,
    dest: DestinationPolicy,
    policy: ProfilePolicy,
    envelope: SnapshotEnvelope,
    key: bytes,
    *,
    ephemeral: bool = False,
) -> Candidate:
    resource = _resource(profile, dest.destination_id, policy, key, envelope)
    scope = scope_document(policy, dest, Signal.METRICS, resource.installation_id)
    _require_collection_scope(envelope, scope, policy.policy_version)
    projection: Projection = project_metrics(envelope)
    chunks = otlp.split_metrics(resource, projection.points)
    batches = tuple(
        Batch(
            _batch_id(envelope.generation, envelope.snapshot_id, index),
            Signal.METRICS,
            envelope.generation,
            envelope.snapshot_id,
            index,
            len(chunks),
            "data",
            body,
            count,
        )
        for index, (body, count) in enumerate(chunks)
    )
    return Candidate(
        Signal.METRICS,
        dest.destination_id,
        envelope.generation,
        envelope.snapshot_id,
        batches,
        (),
        scope,
        projection.omitted,
        projection.notes,
        ephemeral,
        policy.policy_version,
    )


def _record_ids(body: bytes) -> list[tuple[str, int]]:
    """(record_id, revision) of every log record in a serialized logs request."""
    document = json.loads(body)
    found: list[tuple[str, int]] = []
    for resource in document.get("resourceLogs", []):
        for scope in resource.get("scopeLogs", []):
            for record in scope.get("logRecords", []):
                inner = json.loads(record["body"]["stringValue"])
                found.append((inner["record_id"], inner["revision"]))
    return found


def _logs_batches(
    resource: Resource,
    planned: Sequence[Planned],
    time_nano: int,
    ids: Sequence[str],
    *,
    generation: int,
    snapshot_id: str,
    kind: str,
) -> tuple[tuple[Batch, ...], tuple[tuple[Planned, ...], ...]]:
    records = export_records.log_records(planned, time_unix_nano=time_nano)
    chunks = otlp.split_logs(resource, records)
    by_key = {(p.record_id, p.revision): p for p in planned}
    batches: list[Batch] = []
    groups: list[tuple[Planned, ...]] = []
    for index, (body, count) in enumerate(chunks):
        members = tuple(by_key[key] for key in _record_ids(body))
        batches.append(
            Batch(
                ids[index],
                Signal.LOGS,
                generation,
                snapshot_id,
                index,
                len(chunks),
                kind,
                body,
                count,
                tuple(sorted({m.record_id for m in members})),
            )
        )
        groups.append(members)
    return tuple(batches), tuple(groups)


def _chunk_count(resource: Resource, planned: Sequence[Planned], time_nano: int) -> int:
    records = export_records.log_records(planned, time_unix_nano=time_nano)
    return len(otlp.split_logs(resource, records))


def build_logs(
    profile: str,
    dest: DestinationPolicy,
    policy: ProfilePolicy,
    envelope: SnapshotEnvelope,
    key: bytes,
    *,
    ephemeral: bool = False,
) -> Candidate:
    """Ledger-record revisions the registry does not know yet (logs; consent required by caller)."""
    resource = _resource(profile, dest.destination_id, policy, key, None)
    scope = scope_document(policy, dest, Signal.LOGS, resource.installation_id)
    _require_collection_scope(envelope, scope, policy.policy_version)
    rows = export_records.ledger_rows(profile, collection_scope=scope["collection_scope"])
    planned: list[Planned] = []
    omitted = 0
    if rows is not None:
        candidates, omitted = export_records.project_candidates(
            rows, key=key, destination=dest.destination_id
        )
        registry = export_records.load_registry(profile, dest.destination_id)
        planned = export_records.plan_revisions(registry, candidates)
    time_nano = unix_nano(envelope.collected_at)
    count = _chunk_count(resource, planned, time_nano) if planned else 0
    ids = [_batch_id(envelope.generation, envelope.snapshot_id, i) for i in range(count)]
    batches, groups = _logs_batches(
        resource,
        planned,
        time_nano,
        ids,
        generation=envelope.generation,
        snapshot_id=envelope.snapshot_id,
        kind="data",
    )
    return Candidate(
        Signal.LOGS,
        dest.destination_id,
        envelope.generation,
        envelope.snapshot_id,
        batches,
        groups,
        scope,
        omitted,
        (),
        ephemeral,
        policy.policy_version,
    )


def tombstone_ids(planned: Sequence[Planned], stamp: int, count: int) -> list[str]:
    """Deterministic ``t<unix>-<hash8>-<index>`` ids for one tombstone set."""
    digest = hashlib.sha256(
        canonical_bytes(sorted((p.record_id, p.revision) for p in planned))
    ).hexdigest()[:8]
    return [f"t{stamp:010d}-{digest}-{index:04d}" for index in range(count)]


def build_tombstones(
    profile: str,
    dest: DestinationPolicy,
    policy: ProfilePolicy,
    deletions: Sequence[Planned],
    key: bytes,
    *,
    stamp: int,
) -> Candidate:
    """Deletion-revision log batches (``kind='tombstone'``); empty input yields no batches."""
    resource = _resource(profile, dest.destination_id, policy, key, None)
    scope = scope_document(policy, dest, Signal.LOGS, resource.installation_id)
    time_nano = stamp * 1_000_000_000
    count = _chunk_count(resource, deletions, time_nano) if deletions else 0
    batches, groups = _logs_batches(
        resource,
        deletions,
        time_nano,
        tombstone_ids(deletions, stamp, count),
        generation=TOMBSTONE_GENERATION,
        snapshot_id="",
        kind="tombstone",
    )
    return Candidate(
        Signal.LOGS,
        dest.destination_id,
        TOMBSTONE_GENERATION,
        "",
        batches,
        groups,
        scope,
        0,
        (),
        False,
        policy.policy_version,
    )
