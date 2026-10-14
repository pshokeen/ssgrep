"""Spool the current snapshot's batches (durably, before any send) and record the collection.

Order, so that a crash anywhere is repaired by the next cycle (batch writes are idempotent):

1. every batch file + meta is written and fsynced (``export_spool.write_batch``);
2. for logs, ALL batches are registered as ``queued`` in ONE atomic registry write;
3. ``acks.collected`` (+ ``ack_vacuous`` when nothing is to be sent) is persisted.

A generation is built only while it is not acknowledged, has no recorded progress and its batches
are not all on disk. A half-written set (fewer batches than ``total``, or logs not registered)
whose batches were never attempted is deleted and rebuilt: the bytes are deterministic, so the
rebuild is identical unless the ledger changed meanwhile, in which case the NEW state wins. Sending
never starts for a generation whose build failed in this cycle (``Cycle.skip``).

Failures are recorded as stable codes only (``reason_of``): an untrusted snapshot can make the
projection raise, and its text must never reach a result, a document or a log.
"""

from __future__ import annotations

from ssgrep.accounting import (
    acks,
    export_build,
    export_policy,
    export_records,
    export_spool,
)
from ssgrep.accounting.delivery_candidate import current_envelope
from ssgrep.accounting.delivery_registry import body_pairs, queue_batches
from ssgrep.accounting.delivery_state import batches, sync_counts
from ssgrep.accounting.delivery_types import Cycle, CycleError, reason_of
from ssgrep.accounting.export_spool import Meta
from ssgrep.accounting.ledger_contracts import Signal, SnapshotEnvelope


def _registered(cycle: Cycle, metas: list[Meta]) -> bool:
    """Whether the registry holds every unattempted log batch as queued under its own id."""
    if cycle.signal is not Signal.LOGS:
        return True
    registry = export_records.load_registry(cycle.profile, cycle.destination)
    for meta in metas:
        if meta.state != "queued" or meta.attempts:
            continue
        body = export_spool.read_body(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)
        for record_id, revision in body_pairs(body):
            entry = registry.entries.get(record_id)
            if entry is None or not any(
                s.batch == meta.batch_id and s.revision == revision for s in entry.queued
            ):
                return False
    return True


def _complete(cycle: Cycle, metas: list[Meta]) -> bool:
    total = metas[0].total
    same = all(m.total == total and m.snapshot_id == metas[0].snapshot_id for m in metas)
    return same and {m.index for m in metas} == set(range(total)) and _registered(cycle, metas)


def _drop(cycle: Cycle, metas: list[Meta]) -> None:
    for meta in metas:
        if cycle.signal is Signal.LOGS:
            export_records.reset_queued(cycle.profile, cycle.destination, meta.batch_id)
        export_spool.delete_batch(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)


def _pending_build(cycle: Cycle, envelope: SnapshotEnvelope) -> bool:
    """True when ``envelope``'s generation still has to be (re)built."""
    delivery = acks.read_delivery(cycle.profile, cycle.destination, cycle.signal)
    generation = envelope.generation
    if delivery is not None:
        if generation <= (delivery.last_acked_generation or 0):
            return False
        if any(p.generation == generation for p in delivery.progress):
            return False
    existing = [m for m in batches(cycle) if m.kind == "data" and m.generation == generation]
    if not existing:
        return True
    if _complete(cycle, existing):
        acks.update_delivery(  # a crash after the spool write must still record the collection
            cycle.profile, cycle.destination, cycle.signal, lambda d: acks.collected(d, generation)
        )
        return False
    if all(m.state == "queued" and not m.attempts for m in existing):
        _drop(cycle, existing)  # half-built: never sent, deterministic, safe to rebuild
        return True
    return False


def _build(cycle: Cycle, envelope: SnapshotEnvelope) -> export_build.Candidate:
    """Project the snapshot; a projection that rejects it is the stable ``snapshot_invalid``.

    A metrics projection that drops EVERY point of a non-empty snapshot (e.g. an unparseable
    ``collected_at``) exported nothing: that is a failure, never a vacuous acknowledgment.
    """
    assert cycle.key is not None
    try:
        if cycle.signal is Signal.METRICS:
            built = export_build.build_metrics(
                cycle.profile, cycle.dest, cycle.policy, envelope, cycle.key
            )
        else:
            built = export_build.build_logs(
                cycle.profile, cycle.dest, cycle.policy, envelope, cycle.key
            )
    except ValueError as error:
        if reason_of(error) == "invalid_state":  # untrusted snapshot text: code only
            raise CycleError("snapshot_invalid") from None
        raise
    if not built.batches and built.omitted and cycle.signal is Signal.METRICS:
        raise CycleError("snapshot_unexportable")
    return built


def _record(cycle: Cycle, built: export_build.Candidate) -> None:
    """Write, register, then record the collection (vacuous acknowledgment when empty)."""
    stamp = cycle.stamp()
    for batch in built.batches:
        export_spool.write_batch(
            cycle.profile,
            cycle.destination,
            cycle.signal,
            batch,
            created_utc=stamp,
            scope=built.scope,
        )
    if cycle.signal is Signal.LOGS and built.batches:
        queue_batches(
            cycle.profile,
            cycle.destination,
            [
                (b.batch_id, members)
                for b, members in zip(built.batches, built.planned, strict=True)
            ],
        )

    def collect(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, built.generation)
        if not built.batches:
            d = acks.ack_vacuous(d, built.generation, built.snapshot_id, stamp)
        return d

    acks.update_delivery(cycle.profile, cycle.destination, cycle.signal, collect)
    sync_counts(cycle.profile, cycle.destination, cycle.signal)
    cycle.tally.enqueued += len(built.batches)


def enqueue(cycle: Cycle) -> None:
    """Build and spool the current snapshot when it needs it; never raises (codes only)."""
    tally = cycle.tally
    envelope = current_envelope(cycle.profile)
    if envelope is None:
        tally.idle = "no_snapshot"
        return
    generation = envelope.generation
    try:
        if not _pending_build(cycle, envelope):
            return
        if cycle.scope_state is None or not export_policy.snapshot_eligible(
            envelope, cycle.scope_state, cycle.policy.policy_version
        ):
            tally.idle = "ineligible_scope"
            return
        _record(cycle, _build(cycle, envelope))
    except Exception as error:  # noqa: BLE001 - a bad snapshot/state must not stop other sends
        cycle.skip.add(generation)
        tally.enqueue_error = reason_of(error)
