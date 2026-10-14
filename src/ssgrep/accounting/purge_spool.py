"""Erase selected DATA payloads without erasing delivery truth or remote deletion controls.

Logs are selected by record id; metrics conservatively invalidate aggregate bodies whenever
anything is purged. Settled payloads are private data too, but erasing an acknowledged or
partially accepted body is not a new delivery loss. Minimal history lives in delivery/registry.

A durable meta marker blocks dispatch before bookkeeping/body deletion. The body is unlinked
and fsynced before the marker goes, so a restarted purge can finish even after registry cleanup.
An active send cannot be recalled: retain its marker until settlement/recovery erases the body.
Tombstone requests are never selected or erased here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ssgrep.accounting import acks, export_files, export_records, export_spool, storage
from ssgrep.accounting.delivery_registry import body_pairs, promote_unknown
from ssgrep.accounting.export_spool_meta import Meta
from ssgrep.accounting.ledger_contracts import Signal, valid_profile_id

LOSS_REASON = "purge_discarded"


@dataclass(frozen=True)
class SpoolPlan:
    doomed: tuple[Meta, ...] = ()
    not_recallable: int = 0

    @property
    def count(self) -> int:
        return sum(meta.state != "in_flight" for meta in self.doomed)


def export_destinations(profile: str) -> list[str]:
    """Destination ids that have an ``export/<id>/`` directory (read-only)."""
    try:
        fd = storage.open_profile_dir(profile, "export", create=False)
    except FileNotFoundError:
        return []
    try:
        names = storage.list_names(fd)
        return [n for n in names if valid_profile_id(n) and _is_dir(fd, n)]
    finally:
        os.close(fd)


def _is_dir(fd: int, name: str) -> bool:
    import stat

    return stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode)


def plan_spool(
    profile: str, selected: dict[str, frozenset[str]], *, purge_metrics: bool
) -> SpoolPlan:
    doomed: list[Meta] = []
    in_flight = 0
    for destination, signal in sorted(export_spool.iter_spools(profile)):
        mine = selected.get(destination, frozenset())
        for meta in export_spool.list_batches(profile, destination, signal):
            if meta.kind != "data":
                continue
            hit = meta.purge_requested or (signal is Signal.METRICS and purge_metrics)
            hit = hit or (signal is Signal.LOGS and bool(mine.intersection(meta.record_ids)))
            if not hit:
                continue
            doomed.append(meta)
            in_flight += meta.state == "in_flight"
    return SpoolPlan(tuple(doomed), in_flight)


def _outcome(d: acks.Delivery, meta: Meta) -> str:
    """Delivery commits first; honor its outcome if a crash left the meta behind."""
    if meta.state in ("acked", "partial", "lost", "discarded"):
        return meta.state
    if d.last_acked_generation == meta.generation:
        return "acked"
    for progress in d.progress:
        if (progress.generation, progress.snapshot_id) == (meta.generation, meta.snapshot_id):
            for kind in ("acked", "partial", "lost", "discarded"):
                if meta.index in getattr(progress, kind):
                    return kind
    return meta.state


def _discard(d: acks.Delivery, meta: Meta, stamp: str) -> acks.Delivery:
    state = _outcome(d, meta)
    if state in ("acked", "partial", "lost", "discarded"):
        return d
    uncertain = (
        meta.state == "in_flight"
        or meta.warning == "possibly_delivered"
        or d.in_flight_nonrecallable == meta.batch_id
    )
    # Keep per-index discard progress as an idempotence receipt. Resetting a metrics
    # generation here would forget that receipt and double-count a crash retry.
    if meta.generation >= 1 and meta.snapshot_id:
        d = acks.outcome(
            d,
            generation=meta.generation,
            snapshot_id=meta.snapshot_id,
            index=meta.index,
            total=meta.total,
            kind="discarded",
            now=stamp,
        )
    return d if uncertain else acks.record_loss(d, LOSS_REASON, 1, meta.items, stamp)


def _sync_counts(profile: str, destination: str, signal: Signal) -> None:
    live = [
        m
        for m in export_spool.list_batches(profile, destination, signal)
        if m.state in ("queued", "in_flight")
    ]
    pending = len(live)
    unknown = sum(m.state == "in_flight" or m.warning == "possibly_delivered" for m in live)
    acks.update_delivery(
        profile, destination, signal, lambda d: acks.set_counts(d, pending, unknown)
    )


def _erase(profile: str, meta: Meta, stamp: str) -> bool:
    """Finish one marked batch when no sender can still be using its body."""
    destination, signal = meta.destination_id, meta.signal
    current = export_spool.read_meta(profile, destination, signal, meta.batch_id)
    if current is None:
        return False
    meta = current
    delivery = acks.read_delivery(profile, destination, signal)
    state = _outcome(delivery, meta) if delivery is not None else meta.state
    if signal is Signal.LOGS:
        if state == "acked":
            export_records.mark_acked(profile, destination, meta.batch_id)
            try:
                body = export_spool.read_body(profile, destination, signal, meta.batch_id)
            except (export_spool.SpoolCorrupt, FileNotFoundError):
                pass  # a previous attempt already erased it after committing the registry
            else:
                promote_unknown(profile, destination, body_pairs(body))
        elif state == "partial":
            export_records.mark_partial(profile, destination, meta.batch_id)
        elif meta.state == "in_flight" or meta.warning == "possibly_delivered":
            export_records.mark_unknown(profile, destination, meta.batch_id)
        else:
            export_records.reset_queued(profile, destination, meta.batch_id)

    def record(d: acks.Delivery) -> acks.Delivery:
        if meta.state == "in_flight":
            d = acks.mark_nonrecallable(d, meta.batch_id)
        d = _discard(d, meta, stamp)
        if d.in_flight is not None and d.in_flight.batch_id == meta.batch_id:
            d = acks.finish_in_flight(d)
        return d

    acks.update_delivery(profile, destination, signal, record)
    return export_spool.erase_marked_batch(profile, destination, signal, meta.batch_id)


def finish_pending(profile: str, destination: str, signal: Signal, stamp: str) -> int:
    """Complete durable privacy requests; caller holds the destination/signal send lock."""
    removed = 0
    for meta in export_spool.list_batches(profile, destination, signal):
        if meta.kind == "data" and meta.purge_requested:
            removed += _erase(profile, meta, stamp)
    if removed:
        _sync_counts(profile, destination, signal)
    return removed


def apply_spool(profile: str, plan: SpoolPlan, stamp: str) -> tuple[int, int]:
    """Return (removed, nonrecallable); a live sender finishes its marked body at settlement."""
    touched: set[tuple[str, Signal]] = set()
    removed = 0
    not_recallable = 0
    for planned in plan.doomed:
        destination, signal = planned.destination_id, planned.signal
        try:
            meta = export_spool.update_meta(
                profile,
                destination,
                signal,
                planned.batch_id,
                lambda m: replace(m, purge_requested=True),
            )
        except FileNotFoundError:
            continue
        with export_files.locked(
            profile, ("export", destination), f"send-{signal.value}", wait=False
        ) as held:
            if meta.state == "in_flight":
                not_recallable += 1
                acks.update_delivery(
                    profile,
                    destination,
                    signal,
                    lambda d, m=meta: acks.mark_nonrecallable(d, m.batch_id),
                )
                if held is None:
                    continue
            removed += _erase(profile, meta, stamp)
        touched.add((destination, signal))
    for destination, signal in sorted(touched):
        _sync_counts(profile, destination, signal)
    return removed, not_recallable


def release_pins(profile: str, surviving: frozenset[str]) -> int:
    """Unpin acknowledged snapshots whose generation file a purge deleted."""
    released = 0
    for state in acks.list_deliveries(profile):
        pin = state.last_acked_snapshot_id
        if pin is None or pin in surviving:
            continue
        done = False

        def release(d: acks.Delivery, pinned: str = pin) -> acks.Delivery:
            nonlocal done
            if d.last_acked_snapshot_id != pinned:
                return d
            done = True
            return acks.release_acked_snapshot(d)

        acks.update_delivery(profile, state.destination_id, state.signal, release)
        released += done
    return released
