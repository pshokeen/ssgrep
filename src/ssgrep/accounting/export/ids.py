"""Export pseudonyms: keyed, destination-scoped, never a hash of a user name or a path.

``<data>/accounting/<profile>/export/pseudonym.key`` holds 32 random bytes (0600) that exist
for exactly one purpose: deriving the installation, task and record pseudonyms an
organization sees. It is a different file from the request-fingerprint key, the local
endpoint token, and every remote ingest credential, so none of those secrets can double as
an identity key (privacy spec). Every pseudonym mixes in the destination id, so the same
device exporting to two organizations presents two unrelated identities, and neither payload
can assert authority over the other's tenant.

A read-only caller (``--dry-run``) never creates the key: it gets an ephemeral in-memory key
and must say so, because the identifiers it shows will differ once the real key exists.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from ssgrep.accounting.ledger.contracts import canonical_bytes
from ssgrep.accounting.privacy import storage

EXPORT_DIR = "export"
KEY_FILE = "pseudonym.key"
KEY_BYTES = 32
PSEUDONYM_HEX = 32


def load_key(profile: str, *, create: bool) -> bytes | None:
    """The profile's pseudonym key; minted once when ``create``; None when absent otherwise."""
    try:
        directory = storage.open_profile_dir(profile, EXPORT_DIR, create=create)
    except FileNotFoundError:
        return None
    try:
        key = storage.read_file(directory, KEY_FILE, KEY_BYTES * 4)
        if key is None and create:
            storage.write_once(directory, KEY_FILE, os.urandom(KEY_BYTES))
            key = storage.read_file(directory, KEY_FILE, KEY_BYTES * 4)
    finally:
        os.close(directory)
    if key is None:
        return None
    if len(key) != KEY_BYTES:
        raise storage.UnsafeStorageError(0, "pseudonym key is damaged")
    return key


def ephemeral_key() -> bytes:
    """A throwaway key for previews that must not create state."""
    return os.urandom(KEY_BYTES)


def pseudonym(key: bytes, kind: str, destination: str, *parts: str) -> str:
    """128-bit keyed pseudonym of ``parts`` scoped to ``kind`` and ``destination``."""
    message = canonical_bytes([kind, destination, *parts])
    return hmac.digest(key, message, hashlib.sha256).hex()[:PSEUDONYM_HEX]


def installation_id(key: bytes, destination: str, profile: str) -> str:
    return pseudonym(key, "installation", destination, profile)
