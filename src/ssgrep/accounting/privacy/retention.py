"""Export-side retention: effective numbers, bounded spool enforcement, snapshot expiry.

``effective_retention`` is the single place the export side reads the retention numbers (the
collector reads the same policy fields through ``retention_hooks``). A managed ceiling can only
SHORTEN: ``effective = min(profile value, bound)`` where the bound is the ceiling when one is given
and the shipped default otherwise. The sanitized spool is DERIVED from evidence, so it can never
outlive the evidence horizon: ``spool_days = min(SPOOL_RETENTION_DAYS, evidence_days)``.

Spool enforcement removes only spool files and updates only the delivery document of its own
destination/signal; source transcripts, search indexes, collector evidence/ledger and snapshots
are never touched. Every removal of UNSENT or UNRESOLVED bytes is an explicit, visible loss:
``acks.record_loss`` (batches/records/reason), the batch's generation progress is marked ``lost``
(so that generation can never become fully acknowledged) and ``on_discard(meta)`` lets the record
registry reset its queued/unknown entries. The loss and a single pending batch id are recorded
atomically BEFORE deletion. That batch is resumed first after interruption, including tombstones
and orphan bodies whose metadata was already removed; its loss is never counted twice.
``last_acked_*`` never changes. Already acknowledged/partial/lost/discarded batches still on
disk are housekeeping: they age out or are evicted first under size pressure, without new loss,
progress marks or ``on_discard``. Tombstone batches obey
the same bounds; their loss reason is ``tombstone_expired`` and, since they belong to no snapshot
generation, they never touch generation progress. Enforcement requires the caller's per-signal
send lock (``export/<destination>/.send-<signal>.lock``), spanning recovery, retention and sending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import chain
from typing import Any

from ssgrep.accounting.collect import store
from ssgrep.accounting.export import acks, spool as export_spool
from ssgrep.accounting.export.spool import Meta, Usage
from ssgrep.accounting.ledger.ledger_contracts import (
    EVIDENCE_RETENTION_DAYS,
    RECEIPT_RETENTION_DAYS,
    SPOOL_MAX_BYTES,
    SPOOL_RETENTION_DAYS,
    ProfilePolicy,
    Signal,
)
from ssgrep.accounting.privacy.published import day, references
from ssgrep.accounting.privacy.purge import PurgeError, Selection
from ssgrep.accounting.views import snapshots

SOURCES = ("default", "managed_ceiling", "local")
_KEYS = ("evidence_days", "receipt_days", "spool_days", "spool_bytes")
HOUSEKEEPING = ("acked", "partial", "lost", "discarded")
_UNSENT = ("queued", "in_flight", "quarantined")
_AGE_REASONS = {
    "queued": "spool_age_expired",
    "in_flight": "delivery_unknown_expired",
    "quarantined": "quarantine_expired",
}
SIZE_REASON = "spool_size_exceeded"
TOMBSTONE_REASON = "tombstone_expired"

Discard = Callable[[Meta], None]


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Effective:
    """The retention numbers in force and, per value, why (default|managed_ceiling|local)."""

    evidence_days: int
    receipt_days: int
    spool_days: int
    spool_bytes: int
    sources: dict[str, str]

    def __post_init__(self) -> None:
        for name in _KEYS:
            _positive(getattr(self, name), name)
        if set(self.sources) != set(_KEYS) or any(v not in SOURCES for v in self.sources.values()):
            raise ValueError(f"sources must map {_KEYS} to one of {SOURCES}")


def _bounded(
    profile_value: object, default: int, ceiling: object, name: str, *, managed: bool
) -> tuple[int, str]:
    """``min(profile, default, ceiling)`` and why: managed ceiling, plain default, or local."""
    value = _positive(profile_value, name)
    cap = _positive(ceiling, f"ceiling {name}") if managed else None
    bound = default if cap is None else min(default, cap)
    value = min(value, bound)
    if value < bound:
        return value, "local"
    return value, "managed_ceiling" if cap is not None and cap < default else "default"


def effective_retention(policy: ProfilePolicy, ceiling: ProfilePolicy | None = None) -> Effective:
    """Shortest-wins retention numbers for ``policy`` under an optional managed ``ceiling``."""
    evidence, evidence_source = _bounded(
        policy.evidence_retention_days,
        EVIDENCE_RETENTION_DAYS,
        getattr(ceiling, "evidence_retention_days", None),
        "evidence_retention_days",
        managed=ceiling is not None,
    )
    receipt, receipt_source = _bounded(
        policy.receipt_retention_days,
        RECEIPT_RETENTION_DAYS,
        getattr(ceiling, "receipt_retention_days", None),
        "receipt_retention_days",
        managed=ceiling is not None,
    )
    spool_days = min(SPOOL_RETENTION_DAYS, evidence)
    spool_source = "default" if spool_days == SPOOL_RETENTION_DAYS else evidence_source
    return Effective(
        evidence_days=evidence,
        receipt_days=receipt,
        spool_days=spool_days,
        spool_bytes=SPOOL_MAX_BYTES,
        sources={
            "evidence_days": evidence_source,
            "receipt_days": receipt_source,
            "spool_days": spool_source,
            "spool_bytes": "default",
        },
    )


# --- spool enforcement ------------------------------------------------------------------------


@dataclass(frozen=True)
class SpoolEnforcement:
    """What one enforcement pass removed from one destination/signal spool, and why."""

    destination: str
    signal: Signal
    expired_age: int  # unsent batches (queued/quarantined/tombstone) removed by the age bound
    evicted_size: int  # unsent batches removed by the size bound
    unknown_expired: int  # in-flight (sent-or-not unknown) batches removed by the age bound
    records_lost: int
    bytes_freed: int
    remaining: Usage
    housekeeping_removed: int = 0  # acked/partial/lost/discarded metas aged out or evicted


def _moment(stamp: str) -> datetime:
    moment = datetime.fromisoformat(stamp)
    if moment.tzinfo is None:
        raise ValueError(f"timestamp {stamp!r} is not timezone-aware")
    return moment


def _checked_now(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now


def _stamp(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reason(meta: Meta, *, by_age: bool) -> str | None:
    if meta.state not in _UNSENT:
        return None
    if meta.kind == "tombstone":
        return TOMBSTONE_REASON
    return _AGE_REASONS[meta.state] if by_age else SIZE_REASON


def _already_lost(d: acks.Delivery, meta: Meta) -> bool:
    if meta.kind != "data":
        return False
    return any(meta.index in p.lost for p in d.progress if p.generation == meta.generation)


def _mark_lost(d: acks.Delivery, meta: Meta, stamp: str) -> acks.Delivery:
    """Mark the batch's index ``lost`` so its generation can never become fully acknowledged."""
    if meta.kind != "data" or meta.generation <= (d.last_acked_generation or 0):
        return d
    known = next((p for p in d.progress if p.generation == meta.generation), None)
    if known is not None and (known.snapshot_id, known.total) != (meta.snapshot_id, meta.total):
        return d
    return acks.outcome(
        d,
        generation=meta.generation,
        snapshot_id=meta.snapshot_id,
        index=meta.index,
        total=meta.total,
        kind="lost",
        now=stamp,
    )


def _add_loss(d: acks.Delivery, reason: str, records: int, stamp: str) -> acks.Delivery:
    """``acks.record_loss`` for one batch; same reason+instant folds into one summary entry."""
    top = d.loss_summary[0] if d.loss_summary else None
    if top is not None and top.reason == reason and top.at == stamp:
        folded = replace(
            d, loss_summary=d.loss_summary[1:], lost_records=d.lost_records - top.records
        )
        return acks.record_loss(folded, reason, top.batches + 1, top.records + records, stamp)
    return acks.record_loss(d, reason, 1, records, stamp)


def _release_in_flight(d: acks.Delivery, batch_id: str) -> acks.Delivery:
    if d.in_flight is not None and d.in_flight.batch_id == batch_id:
        d = acks.finish_in_flight(d)
    if d.in_flight_nonrecallable == batch_id:
        d = replace(d, in_flight_nonrecallable=None)
    return d


def _clear_pending(profile: str, destination: str, signal: Signal, batch_id: str) -> None:
    acks.update_delivery(
        profile,
        destination,
        signal,
        lambda d: (
            replace(d, retention_pending_batch_id=None)
            if d.retention_pending_batch_id == batch_id
            else d
        ),
    )


def _remove(
    profile: str,
    destination: str,
    signal: Signal,
    meta: Meta,
    reason: str | None,
    stamp: str,
    on_discard: Discard | None,
) -> int:
    """Record the loss (when there is one), reset the registry, then delete the files.

    Returns the records newly counted as lost (0 for housekeeping or an already-counted batch).
    """
    newly_lost = 0
    if reason is not None:

        def lose(d: acks.Delivery) -> acks.Delivery:
            nonlocal newly_lost
            if d.retention_pending_batch_id != meta.batch_id and not _already_lost(d, meta):
                d = _add_loss(_mark_lost(d, meta, stamp), reason, meta.items, stamp)
                newly_lost = meta.items
            return replace(
                _release_in_flight(d, meta.batch_id), retention_pending_batch_id=meta.batch_id
            )

        acks.update_delivery(profile, destination, signal, lose)
        if on_discard is not None:
            on_discard(meta)
    export_spool.delete_batch(profile, destination, signal, meta.batch_id)
    if reason is not None:
        _clear_pending(profile, destination, signal, meta.batch_id)
    return newly_lost


def _eviction_order(metas: list[Meta], in_flight_id: str | None) -> list[Meta]:
    """Oldest first within: housekeeping, then unsent, then the in-flight batch last."""

    def rank(meta: Meta) -> int:
        if meta.state in HOUSEKEEPING:
            return 0
        return 2 if meta.state == "in_flight" or meta.batch_id == in_flight_id else 1

    return sorted(metas, key=rank)


def _sync_counts(profile: str, destination: str, signal: Signal, remaining: list[Meta]) -> None:
    pending = sum(1 for m in remaining if m.state in ("queued", "in_flight"))
    unknown = sum(1 for m in remaining if m.state == "in_flight")
    current = acks.read_delivery(profile, destination, signal)
    if current is None or (current.pending_batches, current.unknown_batches) == (pending, unknown):
        return
    acks.update_delivery(
        profile, destination, signal, lambda d: acks.set_counts(d, pending, unknown)
    )


def enforce_spool(
    profile: str,
    destination: str,
    signal: Signal,
    *,
    now: datetime,
    effective: Effective,
    on_discard: Discard | None = None,
) -> SpoolEnforcement:
    """Apply the age bound, then the size bound; each unsent removal is an explicit loss."""
    _checked_now(now)
    sig = Signal(signal)
    stamp = _stamp(now)
    horizon = timedelta(days=effective.spool_days)
    before = export_spool.usage(profile, destination, sig)
    metas = export_spool.list_batches(profile, destination, sig)
    delivery = acks.read_delivery(profile, destination, sig)
    pending = None if delivery is None else delivery.retention_pending_batch_id
    if pending is not None:
        if not any(meta.batch_id == pending for meta in metas):
            # A crash can leave only the body: delete_batch removes metadata first.
            export_spool.delete_batch(profile, destination, sig, pending)
            _clear_pending(profile, destination, sig, pending)
        else:
            metas.sort(key=lambda meta: meta.batch_id != pending)
    counts = {"age": 0, "size": 0, "unknown": 0, "housekeeping": 0, "records": 0}

    def drop(meta: Meta, *, by_age: bool) -> None:
        reason = _reason(meta, by_age=by_age)
        lost = _remove(profile, destination, sig, meta, reason, stamp, on_discard)
        if reason is None:
            counts["housekeeping"] += 1
            return
        counts["records"] += lost
        key = ("unknown" if meta.state == "in_flight" else "age") if by_age else "size"
        counts[key] += 1

    survivors: list[Meta] = []
    for meta in metas:  # boundary: exactly ``spool_days`` old is NOT yet expired
        if meta.batch_id == pending or now - _moment(meta.created_utc) > horizon:
            drop(meta, by_age=True)
        else:
            survivors.append(meta)
    usage_now = export_spool.usage(profile, destination, sig)
    if usage_now.bytes > effective.spool_bytes:
        delivery = acks.read_delivery(profile, destination, sig)
        in_flight_id = (
            None
            if delivery is None or delivery.in_flight is None
            else (delivery.in_flight.batch_id)
        )
        for meta in _eviction_order(survivors, in_flight_id):
            if usage_now.bytes <= effective.spool_bytes:
                break
            drop(meta, by_age=False)
            survivors.remove(meta)
            usage_now = export_spool.usage(profile, destination, sig)
    _sync_counts(profile, destination, sig, survivors)
    remaining = export_spool.usage(profile, destination, sig)
    return SpoolEnforcement(
        destination=destination,
        signal=sig,
        expired_age=counts["age"],
        evicted_size=counts["size"],
        unknown_expired=counts["unknown"],
        records_lost=counts["records"],
        bytes_freed=before.bytes - remaining.bytes,
        remaining=remaining,
        housekeeping_removed=counts["housekeeping"],
    )


def enforce_all(
    profile: str,
    *,
    now: datetime,
    policy: ProfilePolicy,
    ceiling: ProfilePolicy | None = None,
    on_discard: Discard | None = None,
) -> list[SpoolEnforcement]:
    """Enforce every existing (destination, signal) spool of ``profile`` under ``policy``."""
    effective = effective_retention(policy, ceiling)
    return [
        enforce_spool(
            profile, destination, signal, now=now, effective=effective, on_discard=on_discard
        )
        for destination, signal in sorted(export_spool.iter_spools(profile))
    ]


# --- acknowledged-snapshot expiry and status --------------------------------------------------


def expire_acknowledged_snapshots(
    profile: str, destination: str, signal: Signal, *, now: datetime, effective: Effective
) -> bool:
    """Release the acknowledged snapshot once it is older than ``receipt_days`` or already gone.

    After release ``acks.resolve_acknowledged`` reports ``acknowledged_snapshot_expired`` and
    never substitutes newer data; ``last_acked_generation`` is untouched. Returns whether the
    pin was released.
    """
    _checked_now(now)
    sig = Signal(signal)
    delivery = acks.read_delivery(profile, destination, sig)
    pinned = None if delivery is None else delivery.last_acked_snapshot_id
    if pinned is None:
        return False
    try:
        envelope = snapshots.read_by_snapshot_id(profile, pinned)
    except ValueError:
        envelope = None  # a damaged generation file can never serve as the acknowledged data
    if envelope is not None:
        collected = _moment(envelope.collected_at)
        if now - collected <= timedelta(days=effective.receipt_days):
            return False
    released = False

    def release(d: acks.Delivery) -> acks.Delivery:
        nonlocal released
        if d.last_acked_snapshot_id != pinned:  # a newer full ack landed meanwhile
            return d
        released = True
        return acks.release_acked_snapshot(d)

    acks.update_delivery(profile, destination, sig, release)
    return released


def _spool_view(
    profile: str, destination: str, signal: Signal, effective: Effective, now: datetime
):
    metas = export_spool.list_batches(profile, destination, signal)
    usage = export_spool.usage(profile, destination, signal)
    horizon = timedelta(days=effective.spool_days)
    states: dict[str, int] = {}
    for meta in metas:
        states[meta.state] = states.get(meta.state, 0) + 1
    return {
        "destination": destination,
        "signal": signal.value,
        "batches": usage.batches,
        "bytes": usage.bytes,
        "limit_bytes": effective.spool_bytes,
        "limit_days": effective.spool_days,
        "oldest_created_utc": min((m.created_utc for m in metas), default=None),
        "over_age_batches": sum(1 for m in metas if now - _moment(m.created_utc) > horizon),
        "states": dict(sorted(states.items())),
    }


def retention_status(
    profile: str,
    policy: ProfilePolicy,
    ceiling: ProfilePolicy | None = None,
    *,
    now: datetime,
) -> dict[str, Any]:
    """JSON-safe retention view for ``savings status`` (integers, codes and timestamps only)."""
    _checked_now(now)
    effective = effective_retention(policy, ceiling)
    losses = [
        {
            "destination": d.destination_id,
            "signal": d.signal.value,
            "lost_records": d.lost_records,
            "loss_summary": [
                {
                    "reason": loss.reason,
                    "batches": loss.batches,
                    "records": loss.records,
                    "at": loss.at,
                }
                for loss in d.loss_summary
            ],
        }
        for d in acks.list_deliveries(profile)
        if d.lost_records or d.loss_summary
    ]
    return {
        "effective": {
            "evidence_days": effective.evidence_days,
            "receipt_days": effective.receipt_days,
            "spool_days": effective.spool_days,
            "spool_bytes": effective.spool_bytes,
            "sources": dict(effective.sources),
        },
        "spools": [
            _spool_view(profile, destination, signal, effective, now)
            for destination, signal in sorted(export_spool.iter_spools(profile))
        ],
        "losses": losses,
    }


_CACHES = {
    "partitions",
    "journal-cache",
    "root-evaluations",
    "frequency-maintenance",
    "frequency-assignments",
}


@dataclass(frozen=True)
class ChildResult:
    changed: bool
    expired_receipts: tuple[str, ...]
    manifest: dict | None


def _minimum(days):
    minimum, complete = None, True
    for value in days:
        stamp = day(value)
        if stamp is None:
            complete = False
        elif minimum is None or stamp < minimum:
            minimum = stamp
    return minimum, complete


def _record_day(record):
    stamp = record.get("day") or record.get("period")
    if stamp is None and isinstance(record.get("ts_utc"), str):
        stamp = record["ts_utc"][:10]
    return stamp


def _raw_minimum(role, raw):
    if role.startswith("ledger-"):
        state = raw["state"]
        records = chain(
            state.get("candidates", []),
            state.get("debits", []),
            (
                row
                for root in state.get("frozen", [])
                for row in chain(root.get("leaves", []), root.get("receipts", []))
            ),
        )
        return _minimum(_record_day(row) for row in records)
    if role == "reduction":
        return _minimum(
            chain(
                (
                    _record_day(row)
                    for row in chain(raw.get("executions", []), raw.get("invocations", []))
                ),
                (
                    stamp[:10] if isinstance(stamp, str) else None
                    for _, _, stamp in raw.get("session_last", [])
                ),
            )
        )
    if role == "counters":
        return _minimum(row[1] for row in raw.get("counted", []))
    if role == "sealed":
        return _minimum(raw)
    return None, False


def _index(source, minimum, complete, before_day):
    return {
        "schema": 1,
        "source": source,
        "minimum": minimum,
        "complete": complete,
        "clean_before": before_day,
    }


def stage_children(
    profile,
    checkpoint,
    *,
    name,
    before_day,
    manifest=None,
    history=False,
    receipt_ids=(),
    blob_names=(),
    segment_names=(),
):
    """Process <=32 indexed receipts OR one explicitly selected atomic child.

    The caller stats/deadline-gates work. Raw input/history bodies retain their
    existing MAX_BLOB/MAX_SNAPSHOT limits; one segment may be read twice only
    when filtering writes a replacement. No unselected child body is read.
    """
    from ssgrep.accounting.privacy.purge_published import _plan_selected

    if day(before_day) is None or not name.startswith("identity-archive-"):
        raise ValueError("Invalid original publication expiry selection.")
    atoms = int(history) + len(blob_names) + len(segment_names)
    if len(receipt_ids) > 32 or atoms > 1 or receipt_ids and atoms:
        raise ValueError("Original expiry requires <=32 receipts or one other child.")
    if not receipt_ids and not atoms:
        return ChildResult(False, (), manifest)
    manifest = (
        manifest if manifest is not None else store.read_blob(profile, checkpoint.blobs[name])
    )
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise PurgeError("unsupported_schema", "Original publication manifest is unsupported.")
    original = manifest["published_documents"]
    references(original)
    documents, blobs = dict(original), dict(manifest.get("blobs", {}))
    refs = list(manifest.get("segments", []))
    updates, dropped, summaries = {}, set(), dict(original.get("retention_child_records", {}))
    indexes = dict(original.get("retention_child_dates", {}))
    raw_plan = None
    changed = False
    expired_receipts = []
    if receipt_ids:
        periods, roots = original.get("receipt_periods"), original.get("receipt_roots")
        periods = periods if isinstance(periods, dict) else {}
        roots = roots if isinstance(roots, dict) else {}
        receipts = dict(original.get("receipt_blobs", {}))
        periods = {key: value for key, value in periods.items() if key in receipts}
        roots = {key: value for key, value in roots.items() if key in receipts}
        changed = periods != original.get("receipt_periods") or roots != original.get(
            "receipt_roots"
        )
        for receipt in receipt_ids:
            if receipt not in receipts:
                continue
            if day(periods.get(receipt)) is None or not isinstance(roots.get(receipt), str):
                from ssgrep.accounting.privacy.published import read_document

                _, row = read_document(profile, receipts[receipt], "receipt")
                if row.get("receipt_id") != receipt:
                    raise ValueError("Original receipt identity does not match.")
                periods[receipt], roots[receipt] = row["period"], row["root"]
                changed = True
            if periods[receipt] < before_day:
                dropped.add(receipts.pop(receipt))
                expired_receipts.append(receipt)
                periods.pop(receipt, None)
                roots.pop(receipt, None)
                changed = True
        complete = len(periods) == len(receipts) and all(
            day(value) is not None for value in periods.values()
        )
        documents.update(
            receipt_blobs=receipts,
            receipt_periods=periods,
            receipt_roots=roots,
            receipt_count=len(receipts),
            receipt_min_period=min(periods.values(), default=None) if complete else None,
        )
    elif segment_names:
        selected = set(segment_names)
        after = []
        for raw in refs:
            ref = store.SegmentRef.from_dict(raw)
            if ref.name not in selected:
                after.append(raw)
                continue
            minimum, complete, expired = None, True, False
            for item in store.read_segment(profile, ref):
                stamp = day(item.ts_utc[:10]) if isinstance(item.ts_utc, str) else None
                if stamp is None:
                    complete = False
                elif stamp < before_day:
                    expired = True
                elif minimum is None or stamp < minimum:
                    minimum = stamp
            indexes.pop("segment:" + ref.name, None)
            if not expired:
                after.append(raw)
                indexes["segment:" + ref.name] = _index(ref.name, minimum, complete, before_day)
                continue
            kept_days = []

            def kept(ref=ref, kept_days=kept_days):
                for item in store.read_segment(profile, ref):
                    stamp = day(item.ts_utc[:10]) if isinstance(item.ts_utc, str) else None
                    if stamp is None or stamp >= before_day:
                        kept_days.append(stamp)
                        yield item

            replacements = store.write_segments(profile, kept(), origin=ref.origin)
            position = 0
            for replacement in replacements:
                count = replacement.records
                part_minimum, part_complete = _minimum(kept_days[position : position + count])
                indexes["segment:" + replacement.name] = _index(
                    replacement.name, part_minimum, part_complete, before_day
                )
                position += count
            after.extend(replacement.to_dict() for replacement in replacements)
            changed = True
        refs = after
    else:
        selected = {key: value for key, value in blobs.items() if key in blob_names}
        subset_documents = {
            **original,
            "history_blob": original.get("history_blob") if history else None,
            "receipt_blobs": {},
            "roots_by_period": original.get("roots_by_period") if history else None,
        }
        subset = {
            **manifest,
            "blobs": selected,
            "segments": [],
            "session_roots": [],
            "session_cwd": [],
            "published_documents": subset_documents,
        }
        plan = _plan_selected(
            profile, subset, Selection(profile, before_day + "T00:00:00Z", None), atomic=True
        )
        raw_plan = plan
        updates.update(plan.updates)
        dropped.update(plan.dropped)
        for key, value in selected.items():
            if key.split("-scope-", 1)[0] in _CACHES:
                dropped.add(value)  # disposable, never another workspace's canonical measurements
        if history:
            documents["roots_by_period"] = plan.documents.get("roots_by_period")
            documents["history_rows"] = plan.history_remaining
        else:
            for key in selected:
                role = key.split("-scope-", 1)[0]
                summaries[key] = (
                    plan.remaining_records
                    if role.startswith("ledger-") or role == "reduction"
                    else None
                )
        changed = bool(
            updates
            or dropped
            or plan.documents != subset_documents
            or history
            and documents.get("history_rows") != original.get("history_rows")
            or summaries != original.get("retention_child_records", {})
        )
    memo = {
        old: store.write_blob(profile, prefix, document)
        for old, (prefix, document) in updates.items()
        if old not in dropped
    }
    retained = {key: memo.get(value, value) for key, value in blobs.items() if value not in dropped}
    documents["history_blob"] = memo.get(
        documents.get("history_blob"), documents.get("history_blob")
    )
    documents["retention_child_records"] = summaries
    if raw_plan is not None:
        for key in blob_names:
            old = blobs.get(key)
            source = raw_plan.selected_inputs.get(old)
            if key in retained and source is not None:
                after = updates[old][1] if old in updates else source
                minimum, complete = _raw_minimum(key.split("-scope-", 1)[0], after)
                indexes["blob:" + key] = _index(retained[key], minimum, complete, before_day)
        if history and documents.get("history_blob"):
            periods = list((documents.get("roots_by_period") or {}).keys())
            if raw_plan.history_minimum_day is not None:
                periods.append(raw_plan.history_minimum_day)
            minimum, complete = _minimum(periods)
            indexes["history"] = _index(documents["history_blob"], minimum, complete, before_day)
    allowed = {"blob:" + key for key in retained} | {"segment:" + ref["name"] for ref in refs}
    if documents.get("history_blob"):
        allowed.add("history")
    documents["retention_child_dates"] = {
        key: value for key, value in indexes.items() if key in allowed
    }
    changed = changed or documents != original
    if not changed:
        return ChildResult(False, (), manifest)
    updated_manifest = {
        **manifest,
        "blobs": retained,
        "segments": refs,
        "published_documents": documents,
    }
    checkpoint.blobs = {**checkpoint.blobs, name: store.write_blob(profile, name, updated_manifest)}
    if checkpoint.identity_migration.get("status") == "complete":
        prior = checkpoint.identity_migration.get("superseded_blobs", {})
        pointers = {key: value for key, value in prior.items() if blobs.get(key) != value}
        for key, value in retained.items():
            pointers.setdefault(key, value)
        checkpoint.identity_migration = {
            **checkpoint.identity_migration,
            "superseded_blobs": pointers,
        }
    return ChildResult(True, tuple(expired_receipts), updated_manifest)
