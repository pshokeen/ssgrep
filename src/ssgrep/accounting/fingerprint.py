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

from ssgrep.accounting import storage
from ssgrep.accounting.contracts import Operation, canonical_bytes

KEY_FILE = "fingerprint.key"
KEY_BYTES = 32
NORMALIZATION = "fp1"
_cache: dict[tuple[str, str], bytes] = {}
_lock = threading.Lock()


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


@overload
def load_key(profile: str, *, create: Literal[True]) -> bytes: ...


@overload
def load_key(profile: str, *, create: Literal[False]) -> bytes | None: ...


def load_key(profile: str, *, create: bool) -> bytes | None:
    """The profile's HMAC key; created atomically on first use when ``create``.

    Raises ``storage.UnsafeStorageError`` for unsafe storage. Returns None
    when the key is absent and ``create`` is false (a reader never mints one).
    """
    cache_key = (str(storage.data_root()), profile)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    with _lock:
        try:
            directory = storage.open_profile_dir(profile, create=create)
        except FileNotFoundError:
            return None
        try:
            key = storage.read_file(directory, KEY_FILE, KEY_BYTES * 4)
            if key is None and create:
                candidate = os.urandom(KEY_BYTES)
                storage.write_once(directory, KEY_FILE, candidate)
                key = storage.read_file(directory, KEY_FILE, KEY_BYTES * 4)
        finally:
            os.close(directory)
        if key is None and not create:
            return None
        if key is None or len(key) != KEY_BYTES:
            raise storage.UnsafeStorageError(0, "fingerprint key is missing or damaged")
        _cache[cache_key] = key
        return key


def forget_keys() -> None:
    """Drop cached keys (tests and key rotation)."""
    with _lock:
        _cache.clear()
