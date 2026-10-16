"""Durable per-destination/per-signal delivery state with MONOTONIC generations.

``accounting/<profile>/delivery/<destination_id>/<signal>.json`` is written ONLY by
``update_delivery`` (flock-serialized read-modify-write across processes). ``accepted_means`` is
the constant ``collector_acceptance``: an OTLP 200 is receipt by the collector, never a claim of
backend durability. The acknowledged generation only ever advances; a snapshot is acknowledged
only after EVERY batch of that generation/signal was fully accepted, and metrics and logs keep
independent documents. The document model/codec lives in ``acks_model`` and response parsing in
``acks_response``; both are re-exported here as the public surface.
"""

from __future__ import annotations

import calendar
import contextlib
import fcntl
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from typing import Any, TypeGuard

from ssgrep.accounting.ledger.ledger_contracts import Signal, SnapshotEnvelope, valid_profile_id
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.serve.metrics import ExportHealth
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.snapshots import DELIVERY_DIR, MAX_DELIVERY_BYTES

LOCK_OPEN_ATTEMPTS = 16

__all__ = [
    "ACCEPTED_MEANS",
    "AckResult",
    "Acknowledged",
    "BLOCK_REASONS",
    "Delivery",
    "DeliveryCorrupt",
    "InFlight",
    "KINDS",
    "Loss",
    "Progress",
    "ack_vacuous",
    "attempted",
    "blocked",
    "collected",
    "decode",
    "describe",
    "encode",
    "export_health",
    "finish_in_flight",
    "list_deliveries",
    "mark_nonrecallable",
    "outcome",
    "parse_response",
    "read_delivery",
    "record_loss",
    "release_acked_snapshot",
    "reset_generation",
    "resolve_acknowledged",
    "retry_scheduled",
    "set_counts",
    "start_in_flight",
    "unblock_if_changed",
    "update_delivery",
]


# --- durable state ----------------------------------------------------------------------------


def _ids(profile: str, destination: str, signal: Signal | str) -> Signal:
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise ValueError("profile and destination ids must be bounded identifiers")
    return Signal(signal)


def read_delivery(profile: str, destination: str, signal: Signal) -> Delivery | None:
    """Stored state, ``None`` when absent; never creates anything."""
    sig = _ids(profile, destination, signal)
    try:
        dir_fd = storage.open_profile_dir(profile, DELIVERY_DIR, destination, create=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise DeliveryCorrupt(f"unsafe delivery path: {error}") from error
    try:
        data = storage.read_file(dir_fd, f"{sig.value}.json", MAX_DELIVERY_BYTES)
    except (OSError, ValueError) as error:
        raise DeliveryCorrupt(f"unreadable delivery document: {error}") from error
    finally:
        os.close(dir_fd)
    return None if data is None else decode(data, destination, sig)


def _open_lock(dir_fd: int, name: str) -> int:
    """Open/create the lock file; concurrent first creation can transiently report ENOENT."""
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(LOCK_OPEN_ATTEMPTS - 1):
        try:
            return os.open(name, flags, 0o600, dir_fd=dir_fd)
        except FileNotFoundError:
            continue
    return os.open(name, flags, 0o600, dir_fd=dir_fd)


def update_delivery(
    profile: str, destination: str, signal: Signal, change: Callable[[Delivery], Delivery]
) -> Delivery:
    """Lock, read (or start fresh), apply ``change``, validate, and atomically write."""
    sig = _ids(profile, destination, signal)
    name = f"{sig.value}.json"
    try:
        dir_fd = storage.open_profile_dir(profile, DELIVERY_DIR, destination)
    except OSError as error:
        raise DeliveryCorrupt(f"unsafe delivery path: {error}") from error
    try:
        lock = _open_lock(dir_fd, f".{sig.value}.lock")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = storage.read_file(dir_fd, name, MAX_DELIVERY_BYTES)
            except (OSError, ValueError) as error:
                raise DeliveryCorrupt(f"unreadable delivery document: {error}") from error
            current = Delivery(destination, sig) if data is None else decode(data, destination, sig)
            updated = change(current)
            encoded = encode(updated)
            parse_document(encoded, destination, sig)  # refuse to persist an invalid state
            storage.atomic_write(dir_fd, name, encoded)
            return updated
        finally:
            os.close(lock)  # closing releases the flock
    finally:
        os.close(dir_fd)


def list_deliveries(profile: str) -> list[Delivery]:
    """Every readable delivery document; corrupt or unsafe entries are skipped."""
    found: list[Delivery] = []
    try:
        root = storage.open_profile_dir(profile, DELIVERY_DIR, create=False)
    except (OSError, ValueError):
        return found
    try:
        destinations = storage.list_names(root)
    except OSError:
        destinations = []
    finally:
        os.close(root)
    for destination in destinations:
        if not valid_profile_id(destination):
            continue
        for signal in Signal:
            with contextlib.suppress(DeliveryCorrupt, OSError, ValueError):
                state = read_delivery(profile, destination, signal)
                if state is not None:
                    found.append(state)
    return found


# --- pure transitions -------------------------------------------------------------------------


def collected(d: Delivery, generation: int) -> Delivery:
    return replace(
        d, last_collected_generation=max(d.last_collected_generation or 0, as_gen(generation))
    )


def attempted(d: Delivery, generation: int, now: str) -> Delivery:
    return replace(
        d,
        last_attempted_generation=max(d.last_attempted_generation or 0, as_gen(generation)),
        last_attempt_at=as_time(now),
    )


def start_in_flight(d: Delivery, batch_id: str, generation: int, now: str) -> Delivery:
    return replace(d, in_flight=InFlight(as_token(batch_id), as_gen(generation), as_time(now)))


def finish_in_flight(d: Delivery) -> Delivery:
    return replace(d, in_flight=None)


def mark_nonrecallable(d: Delivery, batch_id: str) -> Delivery:
    return replace(d, in_flight_nonrecallable=as_token(batch_id))


def set_counts(d: Delivery, pending_batches: int, unknown_batches: int) -> Delivery:
    return replace(
        d, pending_batches=as_int(pending_batches), unknown_batches=as_int(unknown_batches)
    )


def retry_scheduled(d: Delivery, next_retry_utc: str, category: str) -> Delivery:
    return replace(d, next_retry_utc=as_time(next_retry_utc), last_error_category=as_code(category))


def blocked(d: Delivery, reason: str, fingerprint: str) -> Delivery:
    return replace(
        d, blocked_reason=as_block_reason(reason), blocked_fingerprint=as_token(fingerprint)
    )


def unblock_if_changed(d: Delivery, fingerprint: str) -> Delivery:
    """Clear a block only when the config/credential fingerprint differs from the blocked one."""
    if d.blocked_reason is None or fingerprint == d.blocked_fingerprint:
        return d
    return replace(d, blocked_reason=None, blocked_fingerprint=None)


def record_loss(d: Delivery, reason: str, batches: int, records: int, now: str) -> Delivery:
    """Loss is data-deletion bookkeeping only: the acknowledged generation never moves."""
    loss = Loss(reason, as_int(batches), as_int(records), as_time(now))
    return replace(
        d,
        loss_summary=((loss, *d.loss_summary))[:MAX_LOSSES],
        lost_records=min(d.lost_records + loss.records, INT64_MAX),
    )


def release_acked_snapshot(d: Delivery) -> Delivery:
    """Retention dropped the acknowledged snapshot: remember it as expired, unpin it."""
    if d.last_acked_snapshot_id is None:
        return d
    return replace(d, last_acked_snapshot_id=None, expired_snapshot_id=d.last_acked_snapshot_id)


def reset_generation(d: Delivery, generation: int) -> Delivery:
    """Drop an un-acknowledged generation's progress (its queued bytes were invalidated)."""
    as_gen(generation)
    if generation <= (d.last_acked_generation or 0):
        return d
    kept = tuple(p for p in d.progress if p.generation != generation)
    return d if len(kept) == len(d.progress) else replace(d, progress=kept)


def ack_vacuous(d: Delivery, generation: int, snapshot_id: str, now: str) -> Delivery:
    """Acknowledge a generation with nothing to send for this signal (same effect as a full ack).

    Refused (unchanged) for an older/equal generation and for a generation with any recorded
    partial/quarantined/lost/discarded or still-unacked batch.
    """
    as_gen(generation), as_snapshot(snapshot_id), as_time(now)
    if generation <= (d.last_acked_generation or 0):
        return d
    for entry in d.progress:
        if entry.generation == generation and len(entry.acked) != entry.total:
            return d
    return _advance(d, d.progress, generation, snapshot_id, now)


def _advance(
    d: Delivery, progress: tuple[Progress, ...], generation: int, snapshot_id: str, now: str
) -> Delivery:
    return replace(
        d,
        last_acked_generation=generation,
        last_acked_snapshot_id=snapshot_id,
        last_acked_at=as_time(now),
        blocked_reason=None,
        blocked_fingerprint=None,
        next_retry_utc=None,
        last_error_category=None,
        expired_snapshot_id=None,
        progress=tuple(p for p in progress if p.generation > generation),
    )


def outcome(
    d: Delivery,
    *,
    generation: int,
    snapshot_id: str,
    index: int,
    total: int,
    kind: str,
    rejected: int = 0,
    warning: str | None = None,
    identity_unavailable: bool = False,
    now: str,
) -> Delivery:
    """Record one batch outcome; full ack needs every index acked and only ever advances."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    as_gen(generation), as_snapshot(snapshot_id), as_int(rejected), optional(as_code, warning)
    as_int(total, 1, MAX_BATCHES), as_int(index, 0, total - 1)
    if generation == d.last_acked_generation:
        return d  # already acknowledged: duplicate reports change nothing
    previous = next((p for p in d.progress if p.generation == generation), None)
    if previous is None:
        previous = Progress(generation, snapshot_id, total)
    elif (previous.snapshot_id, previous.total) != (snapshot_id, total):
        raise ValueError("snapshot_id/total mismatch for an existing generation")
    sets = {name: set(getattr(previous, name)) for name in KINDS}
    if kind == "acked" and any(index in sets[name] for name in BAD_KINDS):
        return d  # a partial/quarantined/lost/discarded batch never turns into a full ack
    is_new = index not in sets[kind]
    for members in sets.values():
        members.discard(index)
    sets[kind].add(index)
    updated = Progress(
        generation, snapshot_id, total, **{n: tuple(sorted(m)) for n, m in sets.items()}
    )
    kept = tuple(p for p in d.progress if p.generation != generation)
    result = replace(
        d, progress=tuple(sorted((*kept, updated), key=lambda p: p.generation))[-MAX_PROGRESS:]
    )
    if kind == "partial" and is_new:
        result = replace(
            result,
            partial_batches=result.partial_batches + 1,
            rejected_points=min(result.rejected_points + rejected, INT64_MAX),
        )
    if kind == "quarantined" and is_new:
        result = replace(result, quarantined_batches=result.quarantined_batches + 1)
    if identity_unavailable:
        result = replace(result, identity_unavailable=True)
    if kind == "acked" and warning is not None:
        result = replace(result, warning_category=warning)
    if len(updated.acked) != total or len(previous.acked) == total:
        return result
    if generation > (d.last_acked_generation or 0):
        return _advance(result, result.progress, generation, snapshot_id, now)
    return replace(result, late_acks=result.late_acks + 1)


# --- acknowledged resolution and views --------------------------------------------------------


@dataclass(frozen=True)
class Acknowledged:
    state: str  # 'acknowledged' | 'not_acknowledged' | 'acknowledged_snapshot_expired'
    delivery: Delivery | None
    envelope: SnapshotEnvelope | None


def resolve_acknowledged(profile: str, destination: str, signal: Signal) -> Acknowledged:
    """The EXACT acknowledged snapshot by digest; never ``current.json``, never newer data."""
    try:
        state = read_delivery(profile, destination, signal)
    except DeliveryCorrupt:
        return Acknowledged("not_acknowledged", None, None)
    if state is None:
        return Acknowledged("not_acknowledged", None, None)
    if state.last_acked_snapshot_id is None:
        expired = state.expired_snapshot_id is not None
        return Acknowledged(
            "acknowledged_snapshot_expired" if expired else "not_acknowledged", state, None
        )
    try:
        envelope = snapshots.read_by_snapshot_id(profile, state.last_acked_snapshot_id)
    except (OSError, ValueError):
        envelope = None
    if envelope is None:
        return Acknowledged("acknowledged_snapshot_expired", state, None)
    return Acknowledged("acknowledged", state, envelope)


def export_health(profile: str) -> list[ExportHealth]:
    health: list[ExportHealth] = []
    for d in list_deliveries(profile):
        lost: dict[str, int] = {}
        for loss in d.loss_summary:
            lost[loss.reason] = lost.get(loss.reason, 0) + loss.records
        health.append(
            ExportHealth(
                destination=d.destination_id,
                signal=d.signal.value,
                last_full_ack_epoch=None if d.last_acked_at is None else epoch_of(d.last_acked_at),
                pending_batches=d.pending_batches,
                rejected_points=d.rejected_points,
                lost=tuple(sorted(lost.items())),
            )
        )
    return health


def _state(d: Delivery) -> str:
    if d.blocked_reason is not None:
        return "blocked"
    if any(p.partial for p in d.progress):
        return "partial"
    if d.unknown_batches or d.in_flight_nonrecallable is not None:
        return "unknown"
    behind = (
        d.last_collected_generation is not None
        and (d.last_acked_generation or 0) < d.last_collected_generation
    )
    if d.pending_batches or d.in_flight is not None or behind:
        return "pending"
    return "idle" if d.last_acked_generation is None else "acknowledged"


def describe(d: Delivery) -> dict[str, object]:
    """Sanitized JSON-safe status view (codes, counts and timestamps only)."""
    return {
        "destination_id": d.destination_id,
        "signal": d.signal.value,
        "state": _state(d),
        "accepted_means": d.accepted_means,
        "last_collected_generation": d.last_collected_generation,
        "last_attempted_generation": d.last_attempted_generation,
        "last_acked_generation": d.last_acked_generation,
        "last_acked_snapshot_id": d.last_acked_snapshot_id,
        "expired_snapshot_id": d.expired_snapshot_id,
        "last_acked_at": d.last_acked_at,
        "last_attempt_at": d.last_attempt_at,
        "pending_batches": d.pending_batches,
        "unknown_batches": d.unknown_batches,
        "partial_batches": d.partial_batches,
        "quarantined_batches": d.quarantined_batches,
        "rejected_points": d.rejected_points,
        "identity_unavailable": d.identity_unavailable,
        "late_acks": d.late_acks,
        "lost_records": d.lost_records,
        "blocked_reason": d.blocked_reason,
        "next_retry_utc": d.next_retry_utc,
        "last_error_category": d.last_error_category,
        "warning_category": d.warning_category,
        "in_flight_nonrecallable": d.in_flight_nonrecallable,
        "loss_summary": [asdict(loss) for loss in d.loss_summary],
        "progress": [
            {
                "generation": p.generation,
                "total": p.total,
                "acked": len(p.acked),
                "partial": len(p.partial),
                "quarantined": len(p.quarantined),
                "lost": len(p.lost),
                "discarded": len(p.discarded),
            }
            for p in d.progress
        ],
    }


"""Delivery document model, bounded-vocabulary validators, and the strict JSON codec.

Everything here is ``int``/``str``/``bool`` only (floats are rejected on read), unknown keys are
rejected, and every string is an enumerated code, a bounded identifier, or a UTC timestamp.
"""


SCHEMA = 1
INT64_MAX = (1 << 63) - 1
MAX_GENERATION = 10**10 - 1
MAX_BATCHES = 2000
MAX_LOSSES = 32
MAX_PROGRESS = 16
ACCEPTED_MEANS = "collector_acceptance"
BLOCK_REASONS = ("unauthorized", "forbidden", "credential_invalid", "policy_blocked")
KINDS = ("acked", "partial", "quarantined", "lost", "discarded")
BAD_KINDS = KINDS[1:]
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_SNAPSHOT_RE = re.compile(r"[0-9a-f]{16,64}")
_TIME_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{1,6})?Z")
DECIMAL_RE = re.compile(r"0|[1-9][0-9]{0,18}")
REJECTED_FIELD = {Signal.METRICS: "rejectedDataPoints", Signal.LOGS: "rejectedLogRecords"}


class DeliveryCorrupt(Exception):
    """The delivery document is unreadable, malformed, or sits on an unsafe path."""


@dataclass(frozen=True)
class Loss:
    reason: str
    batches: int
    records: int
    at: str

    def __post_init__(self) -> None:
        as_code(self.reason)


@dataclass(frozen=True)
class Progress:
    """Per-generation batch outcomes by batch INDEX; the five sets are disjoint."""

    generation: int
    snapshot_id: str
    total: int
    acked: tuple[int, ...] = ()
    partial: tuple[int, ...] = ()
    quarantined: tuple[int, ...] = ()
    lost: tuple[int, ...] = ()
    discarded: tuple[int, ...] = ()


@dataclass(frozen=True)
class InFlight:
    batch_id: str
    generation: int
    started_at: str


@dataclass(frozen=True)
class Delivery:
    destination_id: str
    signal: Signal
    last_collected_generation: int | None = None
    last_attempted_generation: int | None = None
    last_acked_generation: int | None = None
    last_acked_snapshot_id: str | None = None
    pending_batches: int = 0
    rejected_points: int = 0
    unknown_batches: int = 0
    blocked_reason: str | None = None
    next_retry_utc: str | None = None
    last_error_category: str | None = None
    last_attempt_at: str | None = None
    last_acked_at: str | None = None
    partial_batches: int = 0
    quarantined_batches: int = 0
    late_acks: int = 0
    lost_records: int = 0
    loss_summary: tuple[Loss, ...] = ()
    retention_pending_batch_id: str | None = None
    warning_category: str | None = None
    identity_unavailable: bool = False
    in_flight: InFlight | None = None
    in_flight_nonrecallable: str | None = None
    progress: tuple[Progress, ...] = ()
    expired_snapshot_id: str | None = None
    blocked_fingerprint: str | None = None
    accepted_means: str = ACCEPTED_MEANS


# --- scalar validators (ValueError on anything outside the bounded vocabulary) -------------


def is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def as_int(value: object, low: int = 0, high: int = INT64_MAX) -> int:
    if not is_int(value) or not low <= value <= high:
        raise ValueError(f"expected integer in [{low}, {high}], got {value!r}")
    return value


def as_gen(value: object) -> int:
    return as_int(value, 1, MAX_GENERATION)


def _match(pattern: re.Pattern[str], value: object, what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {what}")
    return value


def as_code(value: object) -> str:
    return _match(_CODE_RE, value, "code")


def as_time(value: object) -> str:
    stamp = _match(_TIME_RE, value, "UTC timestamp")
    datetime.fromisoformat(stamp)  # ValueError for impossible calendar dates/times
    return stamp


def as_snapshot(value: object) -> str:
    return _match(_SNAPSHOT_RE, value, "snapshot id")


def optional(check: Callable[[object], Any], value: object) -> Any:
    return None if value is None else check(value)


def as_block_reason(value: object) -> str:
    if not isinstance(value, str) or value not in BLOCK_REASONS:
        raise ValueError(f"blocked reason must be one of {BLOCK_REASONS}")
    return value


def as_token(value: object) -> str:
    return _match(_TOKEN_RE, value, "identifier")


def epoch_of(stamp: str) -> int:
    return calendar.timegm(datetime.fromisoformat(stamp).utctimetuple())


# --- strict codec ---------------------------------------------------------------------------


def _object(raw: object, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict) or not set(raw) <= allowed:
        raise ValueError("unexpected or missing object shape")
    return raw


def _no_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _indexes(raw: object, total: int) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise ValueError("batch index list expected")
    items = tuple(as_int(i, 0, total - 1) for i in raw)
    if items != tuple(sorted(set(items))):
        raise ValueError("batch indexes must be unique and ascending")
    return items


def _progress(raw: object) -> Progress:
    obj = _object(raw, {f.name for f in fields(Progress)})
    total = as_int(obj.get("total"), 1, MAX_BATCHES)
    sets = {name: _indexes(obj.get(name, []), total) for name in KINDS}
    if sum(len(s) for s in sets.values()) != len({i for s in sets.values() for i in s}):
        raise ValueError("progress sets must be disjoint")
    return Progress(
        as_gen(obj.get("generation")), as_snapshot(obj.get("snapshot_id")), total, **sets
    )


def _loss(raw: object) -> Loss:
    obj = _object(raw, {"reason", "batches", "records", "at"})
    return Loss(
        as_code(obj.get("reason")),
        as_int(obj.get("batches")),
        as_int(obj.get("records")),
        as_time(obj.get("at")),
    )


def _in_flight(raw: object) -> InFlight:
    obj = _object(raw, {"batch_id", "generation", "started_at"})
    return InFlight(
        as_token(obj.get("batch_id")), as_gen(obj.get("generation")), as_time(obj.get("started_at"))
    )


def _bounded_list(raw: object, limit: int, build: Callable[[object], Any]) -> tuple[Any, ...]:
    if not isinstance(raw, list) or len(raw) > limit:
        raise ValueError(f"list of at most {limit} entries expected")
    return tuple(build(item) for item in raw)


_GENERATIONS = ("last_collected_generation", "last_attempted_generation", "last_acked_generation")
_COUNTS = (
    "pending_batches",
    "rejected_points",
    "unknown_batches",
    "partial_batches",
    "quarantined_batches",
    "late_acks",
    "lost_records",
)
_OPTIONALS: dict[str, Callable[[object], Any]] = {
    "last_acked_snapshot_id": as_snapshot,
    "expired_snapshot_id": as_snapshot,
    "blocked_reason": as_block_reason,
    "last_error_category": as_code,
    "warning_category": as_code,
    "next_retry_utc": as_time,
    "last_attempt_at": as_time,
    "last_acked_at": as_time,
    "in_flight_nonrecallable": as_token,
    "retention_pending_batch_id": as_token,
    "blocked_fingerprint": as_token,
    "in_flight": _in_flight,
}


def parse_document(data: bytes, destination: str, signal: Signal) -> Delivery:
    try:
        raw = json.loads(data, parse_float=_no_float, parse_constant=_no_float)
    except RecursionError as error:
        raise ValueError("document nesting too deep") from error
    obj = _object(raw, {"schema", *(f.name for f in fields(Delivery))})
    if obj.get("schema") != SCHEMA or not is_int(obj.get("schema")):
        raise ValueError("unsupported delivery schema")
    if obj.get("destination_id") != destination or obj.get("signal") != signal.value:
        raise ValueError("document does not belong to this destination/signal")
    if obj.get("accepted_means", ACCEPTED_MEANS) != ACCEPTED_MEANS:
        raise ValueError("accepted_means is a constant")
    identity = obj.get("identity_unavailable", False)
    if not isinstance(identity, bool):
        raise ValueError("identity_unavailable must be a boolean")
    values: dict[str, Any] = {name: optional(as_gen, obj.get(name)) for name in _GENERATIONS}
    values.update({name: as_int(obj.get(name, 0)) for name in _COUNTS})
    values.update({name: optional(check, obj.get(name)) for name, check in _OPTIONALS.items()})
    values["loss_summary"] = _bounded_list(obj.get("loss_summary", []), MAX_LOSSES, _loss)
    values["progress"] = _bounded_list(obj.get("progress", []), MAX_PROGRESS, _progress)
    numbers = [p.generation for p in values["progress"]]
    if numbers != sorted(set(numbers)):
        raise ValueError("progress generations must be unique and ascending")
    return Delivery(destination, signal, identity_unavailable=identity, **values)


def encode(delivery: Delivery) -> bytes:
    document = {"schema": SCHEMA, **asdict(delivery)}
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def decode(data: bytes, destination: str, signal: Signal) -> Delivery:
    try:
        return parse_document(data, destination, signal)
    except (ValueError, TypeError) as error:
        raise DeliveryCorrupt(f"delivery document is invalid: {error}") from error


"""Classify an HTTP 200 OTLP/HTTP export response (design D10); never retains receiver text."""


DECIMAL_RE = re.compile(r"0|[1-9][0-9]{0,18}")
REJECTED_FIELD = {Signal.METRICS: "rejectedDataPoints", Signal.LOGS: "rejectedLogRecords"}


@dataclass(frozen=True)
class AckResult:
    outcome: str  # 'full' | 'full_with_warning' | 'partial' | 'unparseable'
    rejected: int = 0
    warning_category: str | None = None
    identity_unavailable: bool = False


class _FloatLiteral(str):
    """Marks a JSON float so it is never turned into a Python float (and never accepted)."""


_UNPARSEABLE = AckResult("unparseable")


def _rejected_count(value: object) -> int | None:
    if is_int(value) and 0 <= value <= INT64_MAX:
        return value
    if (
        isinstance(value, str)
        and not isinstance(value, _FloatLiteral)
        and DECIMAL_RE.fullmatch(value)
    ):
        return int(value) if int(value) <= INT64_MAX else None
    return None


def parse_response(signal: Signal, body: bytes | None) -> AckResult:
    """Classify an HTTP 200 OTLP response body; ``errorMessage`` text is never retained."""
    if not body:
        return AckResult("full")
    try:
        document = json.loads(
            body.decode("utf-8"), parse_float=_FloatLiteral, parse_constant=_FloatLiteral
        )
    except (ValueError, RecursionError):
        return _UNPARSEABLE
    if not isinstance(document, dict):
        return _UNPARSEABLE
    if "partialSuccess" not in document:
        return AckResult("full")
    partial = document["partialSuccess"]
    if not isinstance(partial, dict):
        return _UNPARSEABLE
    if not partial or partial == {"errorMessage": ""}:
        return AckResult("full")  # proto3 JSON default values carry no information
    rejected = _rejected_count(partial.get(REJECTED_FIELD[Signal(signal)], 0))
    if rejected is None:
        return _UNPARSEABLE
    if rejected == 0:
        return AckResult("full_with_warning", warning_category="receiver_warning")
    return AckResult("partial", rejected=rejected, identity_unavailable=True)
