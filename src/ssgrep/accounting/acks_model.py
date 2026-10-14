"""Delivery document model, bounded-vocabulary validators, and the strict JSON codec.

Everything here is ``int``/``str``/``bool`` only (floats are rejected on read), unknown keys are
rejected, and every string is an enumerated code, a bounded identifier, or a UTC timestamp.
"""

from __future__ import annotations

import calendar
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, TypeGuard

from ssgrep.accounting.ledger_contracts import Signal

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
