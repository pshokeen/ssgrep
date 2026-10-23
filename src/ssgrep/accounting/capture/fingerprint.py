"""Keyed request fingerprints: reproducible locally, never reveal arguments.

A fingerprint is the HMAC-SHA256 of a canonical, normalized request under a
per-profile local key. The same function runs at capture time (from the live
arguments) and in the reducer (from arguments parsed out of a local
transcript), so equal private arguments reconcile without any plaintext query
or ``where`` predicate ever being written to the journal.

The key lives in ``<data-root>/accounting/<profile>/fingerprint.key`` (0600,
created exactly once, atomically). It is deliberately a different file from
every endpoint token or remote ingest credential: those secrets are never an
HMAC key and the HMAC key is never a credential.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import unicodedata
from typing import Literal, overload

from ssgrep.accounting.ledger.contracts import Operation, canonical_bytes
from ssgrep.accounting.privacy import storage

KEY_FILE = "fingerprint.key"
KEY_BYTES = 32
NORMALIZATION = "fp1"
_cache: dict[tuple[storage.SafeRootSnapshot, str], bytes] = {}
_signatures: dict[tuple[storage.SafeRootSnapshot, str], tuple[int, ...]] = {}
_lock = threading.Lock()


def _after_fork() -> None:
    """Only the forking thread survives; never inherit another thread's mutex."""
    global _lock, _cache, _signatures
    _lock = threading.Lock()
    _cache = {}
    _signatures = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _text(value: str | None) -> str | None:
    """NFC + surrounding-whitespace strip; blank means absent (as search treats it)."""
    if value is None:
        return None
    stripped = unicodedata.normalize("NFC", value).strip()
    return stripped or None


def normalize(operation: Operation, *, query: str | None = None, where: str | None = None) -> bytes:
    """Canonical bytes of the normalized request (the HMAC message).

    Search is keyed on its stripped query and predicate; show on its ref; note
    on its title. Budgets and limits are separate event fields, not identity.
    """
    document = {"v": NORMALIZATION, "op": str(operation), "q": _text(query), "w": _text(where)}
    return canonical_bytes(document)


def request_fingerprint(
    key: bytes, operation: Operation, *, query: str | None = None, where: str | None = None
) -> str:
    """Hex HMAC-SHA256 of the normalized request."""
    message = normalize(operation, query=query, where=where)
    return hmac.digest(key, message, hashlib.sha256).hex()


def key_id(key: bytes | None) -> str | None:
    """Non-secret local key version; not a tenant identity or an export credential."""
    return hmac.digest(key, b"ssgrep-fingerprint-key-id-v1", hashlib.sha256).hex() if key else None


def _signature(descriptor: int) -> tuple[int, ...]:
    info = os.fstat(descriptor)
    return (
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_size,
        info.st_mode,
        info.st_uid,
    )


@overload
def load_key(
    profile: str,
    *,
    create: Literal[True],
    root: storage.SafeRootSnapshot | None = None,
) -> bytes: ...


@overload
def load_key(
    profile: str,
    *,
    create: Literal[False],
    root: storage.SafeRootSnapshot | None = None,
) -> bytes | None: ...


def load_key(
    profile: str,
    *,
    create: bool,
    root: storage.SafeRootSnapshot | None = None,
) -> bytes | None:
    """The profile's HMAC key; created atomically on first use when ``create``.

    Every lookup, including cache hits and creating calls, validates existing
    private parents without repairing permissions. The leaf signature and bytes
    come from the same pinned descriptor, never a reopened complete pathname.
    An explicit root fixes the configured path and inode across every open and
    creation; changing the global setting cannot redirect a creating call.
    Raises ``storage.UnsafeStorageError`` for unsafe storage. Returns None
    when the key is absent and ``create`` is false (a reader never mints one).
    """
    root = storage.snapshot(create=create) if root is None else root
    if root is None:
        if create:
            raise storage.RootChanged(0, "fingerprint root disappeared during creation")
        return None
    cache_key = (root, profile)
    with _lock:
        try:
            directory = storage.open_profile_dir(profile, create=False, root=root)
        except FileNotFoundError:
            if not create:
                return None
            directory = storage.open_profile_dir(profile, create=True, root=root)
        try:
            try:
                descriptor = storage.open_regular(directory, KEY_FILE)
            except FileNotFoundError:
                if not create:
                    return None
                storage.write_once(directory, KEY_FILE, os.urandom(KEY_BYTES))
                try:
                    descriptor = storage.open_regular(directory, KEY_FILE)
                except FileNotFoundError as error:
                    raise storage.UnsafeStorageError(
                        0, "fingerprint key is missing or damaged"
                    ) from error
            try:
                signature = _signature(descriptor)
                cached = _cache.get(cache_key)
                if cached is not None and _signatures.get(cache_key) == signature:
                    return cached
                key = storage.read_descriptor(descriptor, max_bytes=KEY_BYTES * 4)
                if len(key) != KEY_BYTES:
                    raise storage.UnsafeStorageError(0, "fingerprint key is missing or damaged")
                if _signature(descriptor) == signature:
                    _cache[cache_key] = key
                    _signatures[cache_key] = signature
                else:
                    _cache.pop(cache_key, None)
                    _signatures.pop(cache_key, None)
                return key
            finally:
                os.close(descriptor)
        finally:
            os.close(directory)


def forget_keys() -> None:
    """Drop cached keys (tests and key rotation)."""
    with _lock:
        _cache.clear()
        _signatures.clear()
