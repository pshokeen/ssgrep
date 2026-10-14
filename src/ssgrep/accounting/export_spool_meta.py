"""Batch meta model and strict JSON codec for the durable export spool.

``<batch_id>.meta.json`` is the only mutable record of a spooled batch; the body next to it is
immutable. A ``Meta`` is validated at construction (so ``dataclasses.replace`` inside an update
callback is validated too) and the decoder refuses anything outside schema 1: unknown or
duplicate keys, floats, bools where integers belong, impossible timestamps, and newer schemas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from ssgrep.accounting.acks_model import MAX_GENERATION, as_code, as_int, as_time, is_int
from ssgrep.accounting.export_model import MAX_BATCH_BYTES
from ssgrep.accounting.ledger_contracts import Signal, valid_profile_id

SCHEMA = 1
STATES = ("queued", "in_flight", "acked", "partial", "quarantined", "lost", "discarded")
KINDS = ("data", "tombstone")
MAX_SCOPE_BYTES = 16 * 1024
MAX_META_BYTES = 1024 * 1024
MAX_RECORD_IDS = 20_000
MAX_WARNING_CHARS = 512
_MAX_DEPTH = 8
_MAX_INT = (1 << 63) - 1
_BATCH_ID_RE = re.compile(r"[gt][0-9]{10}-[0-9a-f]{8,32}-[0-9]{4}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_SNAPSHOT_RE = re.compile(r"[A-Za-z0-9_.:-]{0,128}")
IDENTITY = (
    "batch_id",
    "destination_id",
    "signal",
    "snapshot_id",
    "generation",
    "index",
    "total",
    "kind",
    "created_utc",
    "bytes",
    "sha256",
    "items",
    "record_ids",
    "scope",
)


class SpoolCorrupt(Exception):
    """Unsafe path, bad JSON/shape, size or sha256 mismatch, or a newer schema."""


def valid_batch_id(value: object) -> bool:
    return isinstance(value, str) and _BATCH_ID_RE.fullmatch(value) is not None


def _match(pattern: re.Pattern[str], value: object, what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {what}")
    return value


def _plain(value: object, depth: int = 0) -> Any:
    """A deep, JSON-only copy: str/int/bool/None, lists and string-keyed objects, no floats."""
    if depth > _MAX_DEPTH:
        raise ValueError("scope nests too deeply")
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        if not -_MAX_INT <= value <= _MAX_INT:
            raise ValueError("scope integers must fit in 64 bits")
        return value
    if isinstance(value, list | tuple):
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, Mapping):
        copy: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("scope keys must be strings")
            copy[key] = _plain(item, depth + 1)
        return copy
    raise ValueError(f"scope holds a non-JSON value of type {type(value).__name__}")


def _text(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@dataclass(frozen=True)
class Meta:
    batch_id: str
    destination_id: str
    signal: Signal
    snapshot_id: str
    generation: int
    index: int
    total: int
    kind: str  # "data" | "tombstone"
    created_utc: str
    bytes: int
    sha256: str
    items: int
    record_ids: tuple[str, ...]
    scope: Mapping[str, object]
    state: str = "queued"
    attempts: int = 0
    last_attempt_utc: str | None = None
    next_attempt_utc: str | None = None
    last_status: int | None = None
    last_category: str | None = None
    rejected: int = 0
    warning: str | None = None
    purge_requested: bool = False  # durable privacy barrier; never authorizes a resend

    def __post_init__(self) -> None:
        _match(_BATCH_ID_RE, self.batch_id, "batch id")
        if not isinstance(self.destination_id, str) or not valid_profile_id(self.destination_id):
            raise ValueError("destination id must be a bounded identifier")
        signal = Signal(self.signal)
        object.__setattr__(self, "signal", signal)
        _match(_SNAPSHOT_RE, self.snapshot_id, "snapshot id")
        as_int(self.generation, 0, MAX_GENERATION)
        as_int(self.index, 0, 9999)
        as_int(self.total, 1, 10000)
        if self.index >= self.total:
            raise ValueError("batch index must be below the batch count")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.kind == "tombstone" and signal is not Signal.LOGS:
            raise ValueError("tombstone batches exist only for logs")
        as_time(self.created_utc)
        as_int(self.bytes, 1, MAX_BATCH_BYTES)
        _match(_SHA256_RE, self.sha256, "sha256")
        as_int(self.items, 0, 10**9)
        self._validate_record_ids()
        self._validate_scope()
        self._validate_progress()

    def _validate_record_ids(self) -> None:
        ids = self.record_ids
        if not isinstance(ids, list | tuple) or len(ids) > MAX_RECORD_IDS:
            raise ValueError(f"record_ids must be a list of at most {MAX_RECORD_IDS} ids")
        for record_id in ids:
            _match(_TOKEN_RE, record_id, "record id")
        object.__setattr__(self, "record_ids", tuple(ids))

    def _validate_scope(self) -> None:
        if not isinstance(self.scope, Mapping):
            raise ValueError("scope must be an object")
        plain = _plain(self.scope)
        if len(_text(plain)) > MAX_SCOPE_BYTES:
            raise ValueError(f"scope exceeds {MAX_SCOPE_BYTES} canonical bytes")
        object.__setattr__(self, "scope", plain)

    def _validate_progress(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        if type(self.purge_requested) is not bool:
            raise ValueError("purge_requested must be a boolean")
        if self.purge_requested and self.kind != "data":
            raise ValueError("deletion requests cannot be privacy-discarded")
        as_int(self.attempts)
        as_int(self.rejected)
        for stamp in (self.last_attempt_utc, self.next_attempt_utc):
            if stamp is not None:
                as_time(stamp)
        if self.last_status is not None:
            as_int(self.last_status, 0, 999)
        if self.last_category is not None:
            as_code(self.last_category)
        if self.warning is not None and (
            not isinstance(self.warning, str) or len(self.warning) > MAX_WARNING_CHARS
        ):
            raise ValueError(f"warning must be text of at most {MAX_WARNING_CHARS} characters")


_FIELD_NAMES = tuple(field.name for field in fields(Meta))


def encode_meta(meta: Meta) -> bytes:
    document: dict[str, Any] = {name: getattr(meta, name) for name in _FIELD_NAMES}
    document["signal"] = meta.signal.value
    document["record_ids"] = list(meta.record_ids)
    document["schema"] = SCHEMA
    data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if len(data) > MAX_META_BYTES:
        raise ValueError(f"meta exceeds {MAX_META_BYTES} bytes")
    return data


def _no_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate key {key!r}")
        document[key] = value
    return document


def _parse(data: bytes, batch_id: str, destination: str, signal: Signal) -> Meta:
    raw = json.loads(
        data, parse_float=_no_float, parse_constant=_no_float, object_pairs_hook=_unique
    )
    if not isinstance(raw, dict):
        raise ValueError("meta document must be an object")
    schema = raw.get("schema")
    if not is_int(schema):
        raise ValueError("meta schema must be an integer")
    if schema > SCHEMA:
        raise ValueError(f"meta schema {schema} is newer than supported schema {SCHEMA}")
    if schema != SCHEMA:
        raise ValueError(f"unsupported meta schema {schema}")
    values = {key: value for key, value in raw.items() if key != "schema"}
    unknown = set(values) - set(_FIELD_NAMES)
    missing = set(IDENTITY) - set(values)
    if unknown or missing:
        raise ValueError(f"unexpected keys {sorted(unknown)} or missing keys {sorted(missing)}")
    if not isinstance(values["record_ids"], list):
        raise ValueError("record_ids must be a list")
    if (values["batch_id"], values["destination_id"], values["signal"]) != (
        batch_id,
        destination,
        signal.value,
    ):
        raise ValueError("meta does not belong to this batch/destination/signal")
    return Meta(**values)


def decode_meta(data: bytes, batch_id: str, destination: str, signal: Signal) -> Meta:
    try:
        return _parse(data, batch_id, destination, signal)
    except (ValueError, TypeError, RecursionError) as error:
        raise SpoolCorrupt(f"meta for {batch_id} is invalid: {error}") from error
