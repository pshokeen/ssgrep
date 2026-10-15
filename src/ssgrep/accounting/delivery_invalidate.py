"""Pre-dispatch revalidation (design D11): queued bytes are only ever as good as TODAY's policy.

``revalidate`` compares the scope document every queued batch was created under with the scope
document today's policy would authorize (``export_policy.batch_current``; tombstones are judged
only by destination/endpoint/identity/logs permission). A stale batch is NEVER sent:

* its unsent bytes are deleted, its registry ``queued`` entries are reset (logs) and its
  generation's progress is dropped (``acks.reset_generation``);
* when the generation is the current snapshot and that snapshot may still leave the machine under
  the new scope (``snapshot_eligible``), it is REPROJECTED by the enqueue step (same batch ids,
  new bytes);
* otherwise the loss is explicit: progress ``discarded`` for each unsent index (the generation can
  never become fully acknowledged) and a ``policy_contraction`` loss summary.

``apply_denial`` is the same story when sending is not authorized at all right now: pause and a
disabled signal HOLD the bounded queue; a revoked destination/signal discards it
(``policy_contraction``). A leftover ``in_flight`` batch of a dead process is returned to the
queue as unknown and reported as ``in_flight_nonrecallable``: it may already have been delivered.
"""

from __future__ import annotations

from ssgrep.accounting import acks, export_policy, export_records, export_spool, snapshots
from ssgrep.accounting.delivery_candidate import current_envelope
from ssgrep.accounting.delivery_state import LIVE, batches, enforce_retention, recover, sync_counts
from ssgrep.accounting.delivery_types import Cycle
from ssgrep.accounting.export_authority import AUTHORITY_HOLD_REASONS
from ssgrep.accounting.export_policy import Gate
from ssgrep.accounting.export_spool import Meta
from ssgrep.accounting.ledger_contracts import Signal

LOSS = "policy_contraction"


def _remove(cycle: Cycle, meta: Meta) -> None:
    """Registry first (a crash in between is repaired by the next pass), then the files."""
    if cycle.signal is Signal.LOGS:
        export_records.reset_queued(cycle.profile, cycle.destination, meta.batch_id)
    export_spool.delete_batch(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)


def _reprojectable(cycle: Cycle, generation: int) -> bool:
    """The snapshot of ``generation`` is current and may still leave under today's scope."""
    if not cycle.enqueue or cycle.scope_state is None:
        return False
    envelope = current_envelope(cycle.profile)
    return (
        envelope is not None
        and envelope.generation == generation
        and export_policy.snapshot_eligible(
            envelope, cycle.scope_state, cycle.policy.policy_version
        )
    )


def _discard_generation(
    cycle: Cycle, generation: int, all_metas: list[Meta], *, reproject_ok: bool
):
    members = [m for m in all_metas if m.kind == "data" and m.generation == generation]
    live = [m for m in members if m.state in LIVE]
    stamp = cycle.stamp()
    again = reproject_ok and _reprojectable(cycle, generation)
    for meta in members if again else live:  # a rebuild replaces settled batches too
        _remove(cycle, meta)

    def change(d: acks.Delivery) -> acks.Delivery:
        if again:
            return acks.reset_generation(d, generation)
        for meta in live:
            d = acks.outcome(
                d,
                generation=generation,
                snapshot_id=meta.snapshot_id,
                index=meta.index,
                total=meta.total,
                kind="discarded",
                now=stamp,
            )
        return acks.record_loss(d, LOSS, len(live), sum(m.items for m in live), stamp)

    acks.update_delivery(cycle.profile, cycle.destination, cycle.signal, change)
    cycle.tally.invalidated += len(live)


def discard(cycle: Cycle, stale: list[Meta], *, reproject_ok: bool) -> None:
    """Delete ``stale`` unsent batches (and their generations' housekeeping) with explicit loss."""
    all_metas = batches(cycle)
    for generation in sorted({m.generation for m in stale if m.kind == "data"}):
        _discard_generation(cycle, generation, all_metas, reproject_ok=reproject_ok)
    tombstones = [m for m in stale if m.kind == "tombstone"]
    if tombstones:
        stamp = cycle.stamp()
        for meta in tombstones:
            _remove(cycle, meta)
        acks.update_delivery(
            cycle.profile,
            cycle.destination,
            cycle.signal,
            lambda d: acks.record_loss(
                d, LOSS, len(tombstones), sum(m.items for m in tombstones), stamp
            ),
        )
        cycle.tally.invalidated += len(tombstones)
    sync_counts(cycle.profile, cycle.destination, cycle.signal)


def is_current(cycle: Cycle, meta: Meta) -> bool:
    """May ``meta``'s immutable bytes still be sent under TODAY's scope? (the one judgement)"""
    if not export_policy.batch_current(meta.scope, cycle.scope, meta.kind):
        return False
    if meta.kind == "tombstone":
        return True
    assert cycle.scope_state is not None  # every caller refreshes the authorized scope first
    try:
        envelope = snapshots.read_generation(cycle.profile, meta.generation)
    except (OSError, ValueError):
        return False
    return (
        envelope is not None
        and envelope.snapshot_id == meta.snapshot_id
        and export_policy.snapshot_eligible(
            envelope, cycle.scope_state, cycle.policy.policy_version
        )
    )


def revalidate(cycle: Cycle) -> None:
    """Invalidate every queued batch whose creation scope is not today's scope."""
    stale = [m for m in batches(cycle) if m.state in LIVE and not is_current(cycle, m)]
    if stale:
        discard(cycle, stale, reproject_ok=True)


HOLD_ONLY = frozenset({"export_paused", "policy_unreadable"}) | AUTHORITY_HOLD_REASONS


def denial_state(gate: Gate) -> tuple[str, str]:
    """(CycleResult state, stable reason) for a gate that denies sending right now."""
    if gate.reason in HOLD_ONLY:  # sending is held (never discarded) until policy is corrected
        return "paused", gate.reason
    if gate.reason == "not_enabled":
        return "not_enabled", "not_enabled"
    if gate.reason == "endpoint_invalid":
        return "blocked", "endpoint_invalid"
    return "revoked", gate.reason or "revoked"


def apply_denial(cycle: Cycle, gate: Gate) -> None:
    """Sending is not authorized now: no network, queue held or (when revoked) discarded.

    Pause and an unreadable policy only HOLD (nothing but crash recovery is written); a disabled
    signal keeps its bounded queue under retention; a revoked destination/signal discards the
    queue with an explicit ``policy_contraction`` loss.
    """
    cycle.gate = gate
    tally = cycle.tally
    cleared = recover(cycle)
    if cleared:
        batch_id = cleared[-1]
        acks.update_delivery(
            cycle.profile,
            cycle.destination,
            cycle.signal,
            lambda d: acks.mark_nonrecallable(d, batch_id),
        )
        tally.nonrecallable = batch_id
    if gate.revoked:
        live = [m for m in batches(cycle) if m.state in LIVE]
        if live:
            discard(cycle, live, reproject_ok=False)
    tally.stop = denial_state(gate)
    if gate.reason not in HOLD_ONLY:
        enforce_retention(cycle)
    sync_counts(cycle.profile, cycle.destination, cycle.signal)
