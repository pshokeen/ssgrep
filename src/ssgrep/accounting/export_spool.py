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
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace

from ssgrep.accounting import storage
from ssgrep.accounting.export_model import MAX_BATCH_BYTES, Batch
from ssgrep.accounting.export_spool_meta import (
    IDENTITY,
    MAX_META_BYTES,
    STATES,
    Meta,
    SpoolCorrupt,
    decode_meta,
    encode_meta,
    valid_batch_id,
)
from ssgrep.accounting.ledger_contracts import Signal, valid_profile_id

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
