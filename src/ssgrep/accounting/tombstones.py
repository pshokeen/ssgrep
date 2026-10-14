"""Remote-deletion signalling for previously exported per-task ledger records.

Purge removes local state; it cannot erase anything a receiver already stored. What it CAN do,
when current policy still permits sending logs to a destination, is spool revisioned
``deletion`` records (tombstones) that the delivery cycle sends later. This module is the only
place that decides which records need one and writes them:

* ``plan_tombstones`` is read-only (safe for ``--dry-run``): per destination it names the
  records the registry shows were SENT (acked or unresolved; a record that only ever sat
  queued never left the machine), whether signalling is permitted right now, and the honest
  limits: the operator-erasure obligations when it is not, and whether aggregate metrics were
  ever accepted (those cannot be recalled).
* ``enqueue_tombstones`` re-gates every destination against the CURRENT policy immediately
  before writing, so a revocation between plan and enqueue (or a plan that was never
  permitted) writes nothing: no spool batch and no registry change. Pausing does not block
  spooling (delivery still gates sending); a revoked destination/signal does.
* ``forget_acknowledged_deletions`` drops the minimal linkage a registry keeps for a record
  once the collector ACCEPTED its deletion; ``pending_tombstones`` counts the ones still
  waiting to be delivered, so status never claims erasure it cannot know about.

Nothing here claims backend erasure: collector acceptance is the strongest state this product
can observe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ssgrep.accounting import acks, export_build, export_policy, export_records, export_spool
from ssgrep.accounting.acks_model import DeliveryCorrupt
from ssgrep.accounting.export_records_model import DELETION, Registry
from ssgrep.accounting.export_registry import update_registry
from ssgrep.accounting.ledger_contracts import ProfilePolicy, Signal

MAX_OBLIGATIONS = 1000
NO_RECORDS = "no_exported_records"
NO_KEY = "no_pseudonym_key"
_LIVE_STATES = ("queued", "in_flight")

__all__ = [
    "MAX_OBLIGATIONS",
    "TombstonePlan",
    "TombstoneResult",
    "enqueue_tombstones",
    "forget_acknowledged_deletions",
    "pending_tombstones",
    "plan_tombstones",
]


@dataclass(frozen=True)
class TombstonePlan:
    """What purge would ask one destination to delete, and what it cannot promise."""

    destination_id: str
    permitted: bool
    reason: str | None  # gate code when not permitted; ``no_exported_records`` when nothing sent
    record_ids: tuple[str, ...]  # sent records selected for deletion (sorted)
    record_count: int
    obligations: tuple[str, ...]  # <= MAX_OBLIGATIONS ids the OPERATOR must erase; () if permitted
    non_reversible_metrics: (
        bool  # metrics were accepted at least once: aggregates cannot be recalled
    )


@dataclass(frozen=True)
class TombstoneResult:
    destination_id: str
    queued_records: int
    batches: int
    batch_ids: tuple[str, ...]
    skipped_reason: str | None
    obligations: tuple[str, ...]


def _bounded(record_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(record_ids[:MAX_OBLIGATIONS])


def _sent(registry: Registry, record_ids: Sequence[str]) -> list[str]:
    """Ids whose registry entry shows an acked or unresolved (sent) revision.

    A record that exists only as queued-never-sent has nothing to retract at the receiver.
    Records already carrying a deletion stay listed (the operator's obligation is unchanged);
    ``build_deletions`` is what keeps a second enqueue from planning them again.
    """
    return [
        rid
        for rid in record_ids
        if (entry := registry.entries.get(rid)) is not None
        and (entry.acked is not None or entry.unknown is not None)
    ]


def _select(
    profile: str,
    destination: str,
    registry: Registry,
    before_day: str | None,
    roots: frozenset[str] | None,
) -> list[str]:
    if before_day is not None:
        ids = export_records.records_before(profile, destination, before_day, roots)
    elif roots is not None:
        ids = export_records.records_for_roots(profile, destination, roots)
    else:
        ids = sorted(registry.entries)
    in_flight = {
        rid
        for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
        if meta.kind == "data" and meta.state == "in_flight"
        for rid in meta.record_ids
    }
    return sorted(set(_sent(registry, ids)) | (set(ids) & in_flight))


def _metrics_accepted(profile: str, destination: str) -> bool:
    """Whether the metrics signal was ever acked; an unreadable document cannot rule it out."""
    try:
        state = acks.read_delivery(profile, destination, Signal.METRICS)
    except DeliveryCorrupt:
        return True
    return state is not None and state.last_acked_generation is not None


def plan_tombstones(
    profile: str,
    policy: ProfilePolicy,
    *,
    before_day: str | None,
    root_keys: frozenset[str] | None,
) -> list[TombstonePlan]:
    """One plan per policy destination that exported anything; reads only, creates nothing.

    A destination qualifies when its registry has records or its metrics were ever acked (so
    the non-reversible-metrics limitation is reported even with no per-task records).
    ``before_day`` keeps records with ``period < before_day``; ``root_keys`` keeps records of
    those conversations; both together intersect; neither selects every record. A corrupt
    registry raises ``RegistryCorrupt`` (what was exported is then unknown).
    """
    plans: list[TombstonePlan] = []
    for dest in policy.destinations:
        destination = dest.destination_id
        registry = export_records.load_registry(profile, destination)
        metrics = _metrics_accepted(profile, destination)
        if not registry.entries and not metrics:
            continue
        ids = tuple(_select(profile, destination, registry, before_day, root_keys))
        verdict = export_policy.gate(
            profile, destination, Signal.LOGS, need_enabled=True, need_unpaused=False
        )
        reason = verdict.reason if not verdict.ok else (None if ids else NO_RECORDS)
        plans.append(
            TombstonePlan(
                destination,
                verdict.ok,
                reason,
                ids,
                len(ids),
                () if verdict.ok else _bounded(ids),
                metrics,
            )
        )
    return plans


def _skipped(plan: TombstonePlan, reason: str | None) -> TombstoneResult:
    owed = plan.obligations if not plan.permitted else _bounded(plan.record_ids)
    return TombstoneResult(plan.destination_id, 0, 0, (), reason, owed)


def _refresh_counts(profile: str, destination: str) -> None:
    """Pending = queued + in-flight logs batches; unknown = those already attempted."""
    live = [
        meta
        for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
        if meta.state in _LIVE_STATES
    ]
    pending, unknown = len(live), sum(1 for meta in live if meta.attempts > 0)
    acks.update_delivery(
        profile, destination, Signal.LOGS, lambda d: acks.set_counts(d, pending, unknown)
    )


def _enqueue_one(profile: str, plan: TombstonePlan, now: datetime) -> TombstoneResult:
    if not plan.permitted:
        return _skipped(plan, plan.reason)
    if not plan.record_ids:
        return _skipped(plan, plan.reason or NO_RECORDS)
    destination = plan.destination_id
    verdict = export_policy.gate(
        profile, destination, Signal.LOGS, need_enabled=True, need_unpaused=False
    )
    if not verdict.ok or verdict.destination is None:
        return _skipped(plan, verdict.reason)
    key, ephemeral = export_build.key_for(profile, create=False)
    if ephemeral:
        return _skipped(plan, NO_KEY)
    registry = export_records.load_registry(profile, destination)
    deletions = export_records.build_deletions(
        registry, plan.record_ids, key=key, destination=destination
    )
    stamp = int(now.timestamp())
    cand = export_build.build_tombstones(
        profile, verdict.destination, verdict.policy, deletions, key, stamp=stamp
    )
    created = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    batch_ids: list[str] = []
    queued = 0
    for batch, members in zip(cand.batches, cand.planned, strict=True):
        export_spool.write_batch(
            profile, destination, Signal.LOGS, batch, created_utc=created, scope=cand.scope
        )
        export_records.mark_queued(profile, destination, members, batch.batch_id)
        batch_ids.append(batch.batch_id)
        queued += len(members)
    if batch_ids:
        _refresh_counts(profile, destination)
    return TombstoneResult(destination, queued, len(batch_ids), tuple(batch_ids), None, ())


def enqueue_tombstones(
    profile: str, policy: ProfilePolicy, plans: Sequence[TombstonePlan], *, now: datetime
) -> list[TombstoneResult]:
    """Spool deletion batches for every still-permitted plan; never bypasses revocation.

    Each destination is re-gated against the CURRENT policy right before it is written (the
    ``policy`` argument is the planning-time snapshot and is not trusted for authorization or
    for the scope stamped on the batches). Idempotent: a record whose latest known revision is
    already a deletion (queued, sent or acked) is not planned again.
    """
    del policy
    return [_enqueue_one(profile, plan, now) for plan in plans]


def forget_acknowledged_deletions(profile: str, destination: str) -> int:
    """Drop registry entries whose latest state is a deletion the collector accepted."""
    removed: list[str] = []

    def prune(registry: Registry) -> Registry:
        kept = {}
        for record_id, entry in registry.entries.items():
            acked, latest = entry.acked, entry.latest
            if (
                acked is not None
                and acked.type == DELETION
                and latest is not None
                and latest.revision <= acked.revision
            ):
                removed.append(record_id)
            else:
                kept[record_id] = entry
        return Registry(kept)

    update_registry(profile, destination, prune, create=False)
    return len(removed)


def pending_tombstones(profile: str, destination: str) -> int:
    """Tombstone batches still waiting to be delivered (queued or in flight)."""
    return sum(
        1
        for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
        if meta.kind == "tombstone" and meta.state in _LIVE_STATES
    )
