"""``accounting/<profile>/export/<destination>/records.json``: the local ledger-log registry.

Written only through ``update_registry`` (flock-serialised read-modify-write, atomic
replacement, like ``acks.update_delivery``); read through ``load_registry``, which is
read-only and never creates anything. The document is strict, float-free, and bounded
(``MAX_RECORDS`` entries, ``MAX_REGISTRY_BYTES`` bytes): anything else raises
``RegistryCorrupt`` and NOTHING is rewritten, so a damaged registry is never "repaired" into
an empty one that would re-send history as new contributions.

Document::

    {"schema": 1, "records": {record_id: {"task", "root_key", "period", "methodology",
        "unit": {"unit", "method", "tokenizer", "currency"},
        "acked": S | null, "unknown": S | null, "queued": [S + {"batch"}, ...]}}}
    S = {"revision": int, "digest": body digest, "content": content digest, "type": ...}

The registry holds only pseudonyms, hashes, the period day and the unit label: enough to plan
revisions and to delete records without the ledger, and nothing a conversation could be
recovered from (``root_key`` is an unkeyed 16-hex hash used only locally by purge).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.export_ids import EXPORT_DIR
from ssgrep.accounting.export_records_model import (
    BATCH_ID,
    CURRENCY,
    HEX16,
    HEX32,
    HEX64,
    INT64_MAX,
    MAX_QUEUED,
    MAX_RECORDS,
    MAX_REGISTRY_BYTES,
    METHOD_NAME,
    METHODOLOGY,
    PERIOD,
    RECORD_TYPES,
    SCHEMA,
    TOKENIZER,
    UNIT_NAME,
    Entry,
    Registry,
    RegistryCorrupt,
    State,
)
from ssgrep.accounting.ledger_contracts import valid_profile_id

NAME = "records.json"
LOCK = ".records.lock"
LOCK_OPEN_ATTEMPTS = 16
_STATE_KEYS = frozenset(("revision", "digest", "content", "type"))
_ENTRY_KEYS = frozenset(
    ("task", "root_key", "period", "methodology", "unit", "acked", "unknown", "queued")
)
_UNIT_KEYS = frozenset(("unit", "method", "tokenizer", "currency"))


def _bad(condition: str) -> RegistryCorrupt:
    return RegistryCorrupt(condition)


def _no_float(_: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _no_constant(_: str) -> Any:
    raise ValueError("non-finite numbers are not allowed")


def _text(value: Any, pattern: Any) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise _bad("registry_invalid")
    return value


def _opt_text(value: Any, pattern: Any) -> str | None:
    return None if value is None else _text(value, pattern)


def _mapping(value: Any, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _bad("registry_invalid")
    return value


def _state(raw: Any, *, queued: bool) -> State:
    data = _mapping(raw, _STATE_KEYS | {"batch"} if queued else _STATE_KEYS)
    revision = data["revision"]
    if type(revision) is not int or not 1 <= revision <= INT64_MAX:
        raise _bad("registry_invalid")
    if data["type"] not in RECORD_TYPES:
        raise _bad("registry_invalid")
    batch = _text(data["batch"], BATCH_ID) if queued else None
    return State(
        revision, _text(data["digest"], HEX64), _text(data["content"], HEX64), data["type"], batch
    )


def _entry(raw: Any) -> Entry:
    data = _mapping(raw, _ENTRY_KEYS)
    if not isinstance(data["queued"], list) or len(data["queued"]) > MAX_QUEUED:
        raise _bad("registry_invalid")
    period = _text(data["period"], PERIOD)
    try:
        date.fromisoformat(period)
    except ValueError:
        raise _bad("registry_invalid") from None
    unit = _mapping(data["unit"], _UNIT_KEYS)
    queued = tuple(_state(item, queued=True) for item in data["queued"])
    if [s.revision for s in queued] != sorted({s.revision for s in queued}):
        raise _bad("registry_invalid")
    return Entry(
        _text(data["task"], HEX32),
        _text(data["root_key"], HEX16),
        period,
        (
            _text(unit["unit"], UNIT_NAME),
            _text(unit["method"], METHOD_NAME),
            _opt_text(unit["tokenizer"], TOKENIZER),
            _opt_text(unit["currency"], CURRENCY),
        ),
        _text(data["methodology"], METHODOLOGY),
        None if data["acked"] is None else _state(data["acked"], queued=False),
        None if data["unknown"] is None else _state(data["unknown"], queued=False),
        queued,
    )


def decode(data: bytes) -> Registry:
    """Strictly parse a registry document (``RegistryCorrupt`` on anything unexpected)."""
    if len(data) > MAX_REGISTRY_BYTES:
        raise _bad("registry_oversize")
    try:
        raw = json.loads(data, parse_float=_no_float, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise _bad("registry_invalid") from None
    if not isinstance(raw, dict) or set(raw) != {"schema", "records"}:
        raise _bad("registry_invalid")
    if type(raw["schema"]) is not int or raw["schema"] != SCHEMA:
        raise _bad("registry_schema")
    records = raw["records"]
    if not isinstance(records, dict) or len(records) > MAX_RECORDS:
        raise _bad("registry_oversize" if isinstance(records, dict) else "registry_invalid")
    return Registry({_text(rid, HEX32): _entry(value) for rid, value in records.items()})


def _state_dict(state: State | None, *, queued: bool = False) -> dict[str, Any] | None:
    if state is None:
        return None
    out: dict[str, Any] = {
        "revision": state.revision,
        "digest": state.digest,
        "content": state.content,
        "type": state.type,
    }
    if queued:
        out["batch"] = state.batch
    return out


def encode(registry: Registry) -> bytes:
    """Canonical bytes of ``registry``; refuses what ``decode`` would refuse."""
    if len(registry.entries) > MAX_RECORDS:
        raise _bad("registry_oversize")
    records = {}
    for rid, e in sorted(registry.entries.items()):
        if len(e.queued) > MAX_QUEUED:
            raise _bad("registry_oversize")
        unit, method, tokenizer, currency = e.unit
        records[rid] = {
            "task": e.task,
            "root_key": e.root_key,
            "period": e.period,
            "methodology": e.methodology,
            "unit": {"unit": unit, "method": method, "tokenizer": tokenizer, "currency": currency},
            "acked": _state_dict(e.acked),
            "unknown": _state_dict(e.unknown),
            "queued": [_state_dict(s, queued=True) for s in e.queued],
        }
    data = json.dumps(
        {"schema": SCHEMA, "records": records}, sort_keys=True, separators=(",", ":")
    ).encode()
    if len(data) > MAX_REGISTRY_BYTES:
        raise _bad("registry_oversize")
    return data


def _open_dir(profile: str, destination: str, *, create: bool) -> int | None:
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise _bad("registry_unsafe")
    try:
        return storage.open_profile_dir(profile, EXPORT_DIR, destination, create=create)
    except FileNotFoundError:
        return None
    except OSError:
        raise _bad("registry_unsafe") from None


def _read(dir_fd: int) -> Registry:
    try:
        data = storage.read_file(dir_fd, NAME, MAX_REGISTRY_BYTES)
    except ValueError:
        raise _bad("registry_oversize") from None
    except OSError:
        raise _bad("registry_unsafe") from None
    return Registry() if data is None else decode(data)


def load_registry(profile: str, destination: str) -> Registry:
    """The stored registry; empty when absent. Read-only: never creates a file or directory."""
    dir_fd = _open_dir(profile, destination, create=False)
    if dir_fd is None:
        return Registry()
    try:
        return _read(dir_fd)
    finally:
        os.close(dir_fd)


def _open_lock(dir_fd: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(LOCK_OPEN_ATTEMPTS - 1):
        with contextlib.suppress(FileNotFoundError):  # a concurrent first creation can race
            return os.open(LOCK, flags, 0o600, dir_fd=dir_fd)
    return os.open(LOCK, flags, 0o600, dir_fd=dir_fd)


def update_registry(
    profile: str,
    destination: str,
    change: Callable[[Registry], Registry],
    *,
    create: bool = True,
) -> Registry:
    """Lock, read, apply ``change``, validate by encoding, and atomically write.

    With ``create`` false an absent registry is left absent (a transition on nothing is a
    no-op that creates no file). The file is rewritten only when the content changed.
    """
    dir_fd = _open_dir(profile, destination, create=create)
    if dir_fd is None:
        return Registry()
    try:
        try:
            lock = _open_lock(dir_fd)
        except OSError:
            raise _bad("registry_unsafe") from None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                before = storage.read_file(dir_fd, NAME, MAX_REGISTRY_BYTES)
            except ValueError:
                raise _bad("registry_oversize") from None
            except OSError:
                raise _bad("registry_unsafe") from None
            current = Registry() if before is None else decode(before)
            updated = change(current)
            encoded = encode(updated)
            if encoded != before and not (before is None and not updated.entries):
                storage.atomic_write(dir_fd, NAME, encoded)
            return updated
        finally:
            os.close(lock)  # closing releases the flock
    finally:
        os.close(dir_fd)
