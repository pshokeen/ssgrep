"""Durable, immutable outbound spool: the exact request bytes of every batch, fsynced before send.

Layout under ``accounting/<profile>/spool/<destination_id>/<signal>/``:

* ``<batch_id>.json``       the EXACT request body. Immutable: written to a unique 0600 temporary
  name, fsynced, hard-linked to its final name, then the directory is fsynced BEFORE the meta
  exists, so a publication crash leaves at worst an orphan body. Privacy erasure deliberately
  retains a marked meta until its body unlink is durable, allowing restart cleanup.
* ``<batch_id>.meta.json``  the only mutable record (``export_spool_meta`` codec), replaced
  atomically via ``storage.atomic_write``.

A single per-directory ``flock`` serializes every writer (write, update, delete, sweep, purge) so
the two files of one batch are never torn by a concurrent sweep; readers take no lock because
every publication is an atomic rename or link. Only names matching the batch-id grammar are ever
used as filenames, every path is opened component by component with ``O_NOFOLLOW``, and
``read_body`` re-verifies size AND sha256 so a flipped byte on disk is never sent.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from ssgrep.accounting.export.acks import MAX_GENERATION, as_code, as_int, as_time, is_int
from ssgrep.accounting.export.otlp import MAX_BATCH_BYTES, Batch
from ssgrep.accounting.ledger.contracts import canonical_bytes
from ssgrep.accounting.ledger.ledger_contracts import Signal, valid_profile_id
from ssgrep.accounting.privacy import storage

SPOOL_DIR = "spool"
LOCK_NAME = ".spool.lock"
LOCK_OPEN_ATTEMPTS = 16
_META_SUFFIX = ".meta.json"
_BODY_SUFFIX = ".json"

__all__ = [
    "BatchConflict",
    "Meta",
    "STATES",
    "SpoolCorrupt",
    "Usage",
    "corrupt_batches",
    "delete_batch",
    "drop_destination",
    "erase_marked_batch",
    "iter_spools",
    "list_batches",
    "read_body",
    "read_meta",
    "sweep_orphans",
    "update_meta",
    "usage",
    "write_batch",
]


class BatchConflict(Exception):
    """The same batch id already exists with different request bytes."""


@dataclass(frozen=True)
class Usage:
    batches: int
    bytes: int  # body + meta sizes actually on disk


# --- plumbing -----------------------------------------------------------------------------------


def _ids(profile: str, destination: str, signal: Signal | str) -> Signal:
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise ValueError("profile and destination ids must be bounded identifiers")
    return Signal(signal)


def _signals(profile: str, destination: str, signal: Signal | None) -> list[Signal]:
    """The one given signal, or both; ids are validated either way."""
    if signal is None:
        _ids(profile, destination, Signal.METRICS)
        return list(Signal)
    return [_ids(profile, destination, signal)]


def _batch_id(batch_id: str) -> str:
    if not valid_batch_id(batch_id):
        raise ValueError(f"invalid batch id {batch_id!r}")
    return batch_id


def _body_name(batch_id: str) -> str:
    return f"{batch_id}{_BODY_SUFFIX}"


def _meta_name(batch_id: str) -> str:
    return f"{batch_id}{_META_SUFFIX}"


def _classify(name: str) -> tuple[str, bool] | None:
    """``(batch_id, is_meta)`` for a spool file name; None for anything else (never touched)."""
    if name.endswith(_META_SUFFIX):
        stem, is_meta = name[: -len(_META_SUFFIX)], True
    elif name.endswith(_BODY_SUFFIX):
        stem, is_meta = name[: -len(_BODY_SUFFIX)], False
    else:
        return None
    return (stem, is_meta) if valid_batch_id(stem) else None


def _is_temporary(name: str) -> bool:
    return name.startswith(".") and name.endswith(".tmp")


@contextlib.contextmanager
def _spool(profile: str, destination: str, signal: Signal, *, create: bool) -> Iterator[int | None]:
    """Directory descriptor of one spool; None when absent and not ``create``."""
    try:
        dir_fd = storage.open_profile_dir(
            profile, SPOOL_DIR, destination, signal.value, create=create
        )
    except FileNotFoundError:
        yield None
        return
    except OSError as error:
        raise SpoolCorrupt(f"unsafe spool path: {error}") from error
    try:
        yield dir_fd
    finally:
        os.close(dir_fd)


def _open_lock(dir_fd: int) -> int:
    """Open/create the lock file; concurrent first creation can transiently report ENOENT."""
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        for _ in range(LOCK_OPEN_ATTEMPTS - 1):
            try:
                return os.open(LOCK_NAME, flags, 0o600, dir_fd=dir_fd)
            except FileNotFoundError:
                continue
        return os.open(LOCK_NAME, flags, 0o600, dir_fd=dir_fd)
    except OSError as error:
        raise SpoolCorrupt(f"unsafe spool lock file: {error}") from error


@contextlib.contextmanager
def _locked(dir_fd: int) -> Iterator[None]:
    lock = _open_lock(dir_fd)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
    finally:
        os.close(lock)  # closing releases the flock


def _read(dir_fd: int, name: str, limit: int) -> bytes | None:
    try:
        return storage.read_file(dir_fd, name, limit)
    except (OSError, ValueError) as error:
        raise SpoolCorrupt(f"unreadable spool file {name}: {error}") from error


def _load(dir_fd: int, batch_id: str, destination: str, signal: Signal) -> Meta | None:
    data = _read(dir_fd, _meta_name(batch_id), MAX_META_BYTES)
    return None if data is None else decode_meta(data, batch_id, destination, signal)


def _remove(dir_fd: int, name: str) -> bool:
    try:
        return storage.delete_file(dir_fd, name)
    except OSError as error:
        raise SpoolCorrupt(f"cannot remove spool file {name}: {error}") from error


# --- writing ------------------------------------------------------------------------------------


def _new_meta(
    destination: str, signal: Signal, batch: Batch, created_utc: str, scope: Mapping[str, object]
) -> Meta:
    if batch.signal is not signal:
        raise ValueError("batch signal does not match the spool signal")
    body = batch.body
    if not isinstance(body, bytes) or not 0 < len(body) <= MAX_BATCH_BYTES:
        raise ValueError(f"batch body must be 1..{MAX_BATCH_BYTES} bytes")
    try:
        document = json.loads(body)
    except (ValueError, RecursionError) as error:
        raise ValueError("batch body must be a JSON document") from error
    if not isinstance(document, dict):
        raise ValueError("batch body must be a JSON object")
    return Meta(
        batch_id=batch.batch_id,
        destination_id=destination,
        signal=signal,
        snapshot_id=batch.snapshot_id,
        generation=batch.generation,
        index=batch.index,
        total=batch.total,
        kind=batch.kind,
        created_utc=created_utc,
        bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        items=batch.items,
        record_ids=tuple(batch.record_ids),
        scope=scope,
    )


def write_batch(
    profile: str,
    destination: str,
    signal: Signal,
    batch: Batch,
    *,
    created_utc: str,
    scope: Mapping[str, object],
) -> Meta:
    """Persist body then meta durably; an identical re-write returns the EXISTING meta."""
    sig = _ids(profile, destination, signal)
    meta = _new_meta(destination, sig, batch, created_utc, scope)
    with _spool(profile, destination, sig, create=True) as dir_fd:
        assert dir_fd is not None
        with _locked(dir_fd):
            existing = _load(dir_fd, meta.batch_id, destination, sig)
            if existing is not None:
                if (existing.bytes, existing.sha256) != (meta.bytes, meta.sha256):
                    raise BatchConflict(f"batch {meta.batch_id} exists with different bytes")
                return existing
            name = _body_name(meta.batch_id)
            if not storage.write_once(dir_fd, name, batch.body):  # orphan of a crashed write?
                if _read(dir_fd, name, MAX_BATCH_BYTES) != batch.body:
                    raise BatchConflict(f"batch {meta.batch_id} body exists with different bytes")
            storage.atomic_write(dir_fd, _meta_name(meta.batch_id), encode_meta(meta))
    return meta


def update_meta(
    profile: str,
    destination: str,
    signal: Signal,
    batch_id: str,
    change: Callable[[Meta], Meta],
) -> Meta:
    """Flock-serialized read-modify-write of the meta; identity fields are immutable."""
    sig = _ids(profile, destination, signal)
    _batch_id(batch_id)
    with _spool(profile, destination, sig, create=False) as dir_fd:
        if dir_fd is None:
            raise FileNotFoundError(errno.ENOENT, f"no spooled batch {batch_id}")
        with _locked(dir_fd):
            current = _load(dir_fd, batch_id, destination, sig)
            if current is None:
                raise FileNotFoundError(errno.ENOENT, f"no spooled batch {batch_id}")
            proposed = change(current)
            if not isinstance(proposed, Meta):
                raise ValueError("update callback must return a Meta")
            updated = replace(proposed)  # re-validates every field, including the state
            for name in IDENTITY:
                if getattr(updated, name) != getattr(current, name):
                    raise ValueError(f"{name} is immutable")
            if current.purge_requested and not updated.purge_requested:
                raise ValueError("a privacy purge cannot be revoked")
            if updated != current:
                storage.atomic_write(dir_fd, _meta_name(batch_id), encode_meta(updated))
            return updated


# --- reading ------------------------------------------------------------------------------------


def read_meta(profile: str, destination: str, signal: Signal, batch_id: str) -> Meta | None:
    """The stored meta, None when absent; ``SpoolCorrupt`` when unreadable/invalid/unsafe."""
    sig = _ids(profile, destination, signal)
    _batch_id(batch_id)
    with _spool(profile, destination, sig, create=False) as dir_fd:
        return None if dir_fd is None else _load(dir_fd, batch_id, destination, sig)


def read_body(profile: str, destination: str, signal: Signal, batch_id: str) -> bytes:
    """The exact request bytes, verified against the meta's size AND sha256."""
    sig = _ids(profile, destination, signal)
    _batch_id(batch_id)
    with _spool(profile, destination, sig, create=False) as dir_fd:
        meta = None if dir_fd is None else _load(dir_fd, batch_id, destination, sig)
        if dir_fd is None or meta is None:
            raise FileNotFoundError(errno.ENOENT, f"no spooled batch {batch_id}")
        body = _read(dir_fd, _body_name(batch_id), MAX_BATCH_BYTES)
    if body is None:
        raise SpoolCorrupt(f"body of batch {batch_id} is missing")
    if len(body) != meta.bytes or hashlib.sha256(body).hexdigest() != meta.sha256:
        raise SpoolCorrupt(f"body of batch {batch_id} does not match its recorded size/sha256")
    return body


def _meta_ids(dir_fd: int) -> list[str]:
    found = (_classify(name) for name in storage.list_names(dir_fd))
    return [item[0] for item in found if item is not None and item[1]]


def list_batches(profile: str, destination: str, signal: Signal) -> list[Meta]:
    """Readable metas, oldest first; corrupt metas are skipped; never creates directories."""
    sig = _ids(profile, destination, signal)
    metas: list[Meta] = []
    with _spool(profile, destination, sig, create=False) as dir_fd:
        if dir_fd is None:
            return metas
        for batch_id in _meta_ids(dir_fd):
            with contextlib.suppress(SpoolCorrupt):
                meta = _load(dir_fd, batch_id, destination, sig)
                if meta is not None:
                    metas.append(meta)
    return sorted(metas, key=lambda m: (m.generation, m.kind == "tombstone", m.index, m.batch_id))


def corrupt_batches(profile: str, destination: str, signal: Signal) -> list[str]:
    """Ids whose meta is unreadable/invalid or whose body is missing or the wrong size."""
    sig = _ids(profile, destination, signal)
    bad: list[str] = []
    with _spool(profile, destination, sig, create=False) as dir_fd:
        if dir_fd is None:
            return bad
        for batch_id in _meta_ids(dir_fd):
            try:
                meta = _load(dir_fd, batch_id, destination, sig)
                if meta is None:
                    continue
                info = os.stat(_body_name(batch_id), dir_fd=dir_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_size != meta.bytes:
                    bad.append(batch_id)
            except (SpoolCorrupt, FileNotFoundError):
                bad.append(batch_id)
    return bad


# --- accounting, deletion and recovery ----------------------------------------------------------


def _usage_of(dir_fd: int) -> tuple[set[str], int]:
    ids: set[str] = set()
    size = 0
    for name in storage.list_names(dir_fd):
        classified = _classify(name)
        if classified is None:
            continue
        try:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:  # deleted since the listing
            continue
        if stat.S_ISREG(info.st_mode):
            ids.add(classified[0])
            size += info.st_size
    return ids, size


def usage(profile: str, destination: str, signal: Signal | None = None) -> Usage:
    """Batches (distinct ids with a body or meta) and the bytes both files occupy on disk."""
    batches = size = 0
    for sig in _signals(profile, destination, signal):
        with _spool(profile, destination, sig, create=False) as dir_fd:
            if dir_fd is not None:
                ids, used = _usage_of(dir_fd)
                batches, size = batches + len(ids), size + used
    return Usage(batches, size)


def _delete_batch_files(dir_fd: int, batch_id: str) -> bool:
    removed_meta = _remove(dir_fd, _meta_name(batch_id))  # meta first: never a meta w/o its body
    removed_body = _remove(dir_fd, _body_name(batch_id))
    return removed_meta or removed_body


def delete_batch(profile: str, destination: str, signal: Signal, batch_id: str) -> bool:
    """Remove meta, then body, then fsync the directory; False when nothing was there."""
    sig = _ids(profile, destination, signal)
    _batch_id(batch_id)
    with _spool(profile, destination, sig, create=False) as dir_fd:
        if dir_fd is None:
            return False
        with _locked(dir_fd):
            removed = _delete_batch_files(dir_fd, batch_id)
            if removed:
                storage.fsync_dir(dir_fd)
            return removed


def erase_marked_batch(profile: str, destination: str, signal: Signal, batch_id: str) -> bool:
    """Erase a privacy-marked DATA body before its meta, keeping crash recovery selectable.

    Delivery/registry bookkeeping must be committed first. The marker survives any failure
    before the body unlink is durable; unlike ordinary eviction no unselectable body remains.
    """
    sig = _ids(profile, destination, signal)
    _batch_id(batch_id)
    with _spool(profile, destination, sig, create=False) as dir_fd:
        if dir_fd is None:
            return False
        with _locked(dir_fd):
            meta = _load(dir_fd, batch_id, destination, sig)
            if meta is None:
                return False
            if meta.kind != "data" or not meta.purge_requested:
                raise ValueError("only privacy-marked data may be erased")
            _remove(dir_fd, _body_name(batch_id))
            storage.fsync_dir(dir_fd)
            _remove(dir_fd, _meta_name(batch_id))
            storage.fsync_dir(dir_fd)
            return True


def iter_spools(profile: str) -> list[tuple[str, Signal]]:
    """Existing ``(destination, signal)`` spool directories; tolerant, read-only."""
    found: list[tuple[str, Signal]] = []
    if not valid_profile_id(profile):
        return found
    try:
        root = storage.open_profile_dir(profile, SPOOL_DIR, create=False)
    except OSError:
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
            try:
                os.close(
                    storage.open_profile_dir(
                        profile, SPOOL_DIR, destination, signal.value, create=False
                    )
                )
            except OSError:
                continue
            found.append((destination, signal))
    return found


def sweep_orphans(profile: str, destination: str, signal: Signal) -> int:
    """Delete bodies without a meta, leaked ``.tmp`` files and metas without a body."""
    sig = _ids(profile, destination, signal)
    with _spool(profile, destination, sig, create=False) as dir_fd:
        if dir_fd is None:
            return 0
        with _locked(dir_fd):
            names = storage.list_names(dir_fd)
            classified = {name: _classify(name) for name in names}
            bodies = {c[0] for c in classified.values() if c is not None and not c[1]}
            metas = {c[0] for c in classified.values() if c is not None and c[1]}
            doomed = [_body_name(i) for i in sorted(bodies - metas)]
            doomed += [_meta_name(i) for i in sorted(metas - bodies)]
            doomed += [name for name in names if _is_temporary(name)]
            removed = sum(_remove(dir_fd, name) for name in doomed)
            if removed:
                storage.fsync_dir(dir_fd)
            return removed


def drop_destination(profile: str, destination: str, signal: Signal | None = None) -> int:
    """Purge every batch (and leaked temporary) of one destination; returns the batch count."""
    dropped = 0
    for sig in _signals(profile, destination, signal):
        with _spool(profile, destination, sig, create=False) as dir_fd:
            if dir_fd is None:
                continue
            with _locked(dir_fd):
                names = storage.list_names(dir_fd)
                ids = {c[0] for c in map(_classify, names) if c is not None}
                for batch_id in sorted(ids):
                    _delete_batch_files(dir_fd, batch_id)
                for name in names:
                    if _is_temporary(name):
                        _remove(dir_fd, name)
                storage.fsync_dir(dir_fd)
                dropped += len(ids)
    return dropped


"""Batch meta model and strict JSON codec for the durable export spool.

``<batch_id>.meta.json`` is the only mutable record of a spooled batch; the body next to it is
immutable. A ``Meta`` is validated at construction (so ``dataclasses.replace`` inside an update
callback is validated too) and the decoder refuses anything outside schema 1: unknown or
duplicate keys, floats, bools where integers belong, impossible timestamps, and newer schemas.
"""


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


MAX_STATE_BYTES = 1024 * 1024


class StateCorrupt(Exception):
    """A state file is unreadable, unsafe, or not strict float-free JSON (never rewritten)."""


def _check(profile: str, parts: tuple[str, ...]) -> None:
    if not valid_profile_id(profile) or not all(
        valid_profile_id(p) or p == "export" for p in parts
    ):
        raise ValueError("profile and path components must be bounded identifiers")


def _state_open_lock(dir_fd: int, name: str) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(LOCK_OPEN_ATTEMPTS - 1):
        with contextlib.suppress(FileNotFoundError):  # first-creation race reports ENOENT
            return os.open(name, flags, 0o600, dir_fd=dir_fd)
    return os.open(name, flags, 0o600, dir_fd=dir_fd)


@contextlib.contextmanager
def locked(
    profile: str, dirs: tuple[str, ...], name: str, *, wait: bool = True
) -> Iterator[int | None]:
    """Hold the exclusive lock for ``name`` (yields the directory fd; ``None`` if ``wait=False``
    and another process holds it)."""
    _check(profile, dirs)
    try:
        dir_fd = storage.open_profile_dir(profile, *dirs)
    except OSError as error:
        raise StateCorrupt(f"unsafe state path: {error}") from error
    try:
        lock = _state_open_lock(dir_fd, f".{name}.lock")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield None
                return
            yield dir_fd
        finally:
            os.close(lock)
    finally:
        os.close(dir_fd)


def write_json(dir_fd: int, name: str, document: Any) -> None:
    """Atomically replace ``name`` with the canonical bytes of ``document`` (no floats)."""
    storage.atomic_write(dir_fd, name, canonical_bytes(document) + b"\n")
