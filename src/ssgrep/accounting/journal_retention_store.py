"""Bounded checkpoint references to horizon-scoped journal identity provenance.

Identity payloads use the existing immutable collector blobs. At most 256
HMAC-prefix handles are checkpoint roots, not one field per accepted phase.
Parsed immutable payloads are reused while their physical signatures agree.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.collector_types import CollectError

STORE_VERSION = 2
ROOT_PREFIX = "journal-retention:"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_REF = re.compile(r"^journal-retention-([0-9a-f]{2})-([0-9a-f]{24})\.json$")


def journal_retention_refs(record: dict[str, Any]) -> set[str]:
    """Blob roots for current checkpoints, archives, scoped retention and purge."""
    return set(record.get("identity_shards", {}).values())


def _signatures(profile: str, refs: dict[str, str]) -> tuple:
    try:
        directory = storage.open_profile_dir(profile, "collector", "state", create=False)
    except FileNotFoundError as error:
        raise CollectError(
            "state_corrupt", "Journal identity blob directory is missing."
        ) from error
    try:
        signatures = []
        for bucket, name in sorted(refs.items()):
            match = _REF.fullmatch(name)
            if match is None or match[1] != bucket:
                raise ValueError("invalid journal identity reference")
            try:
                fd = storage.open_regular(directory, name)
            except FileNotFoundError as error:
                raise CollectError("state_corrupt", "Journal identity blob is missing.") from error
            try:
                info = os.fstat(fd)
                signatures.append(
                    (
                        bucket,
                        name,
                        info.st_dev,
                        info.st_ino,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                    )
                )
            finally:
                os.close(fd)
        return tuple(signatures)
    finally:
        os.close(directory)


@lru_cache(maxsize=4)
def _load_identities(root: str, profile: str, signatures: tuple) -> tuple[dict, list[str]]:
    from ssgrep.accounting import collector_store as store

    retired = {}
    bridges = []
    for bucket, name, *_ in signatures:
        payload = store.read_blob(profile, name)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise CollectError("state_corrupt", "Unsupported journal identity blob.")
        digest = hashlib.sha256(store.dumps(payload)).hexdigest()[:24]
        match = _REF.fullmatch(name)
        if match is None or match[2] != digest:
            raise CollectError("state_corrupt", "Journal identity blob digest changed.")
        for key, day in payload["retired"].items():
            if not _HEX.fullmatch(key) or not key.startswith(bucket):
                raise CollectError("state_corrupt", "Journal identity key is invalid.")
            retired[key] = day
        for key in payload["bridges"]:
            if not _HEX.fullmatch(key) or not key.startswith(bucket):
                raise CollectError("state_corrupt", "Journal bridge identity is invalid.")
            bridges.append(key)
    return retired, sorted(bridges)


def expand_journal_retention(profile: str, record: dict[str, Any]) -> dict[str, Any]:
    """Resolve the supplied checkpoint snapshot; never reload a newer checkpoint."""
    version = record.get("version")
    if version == 1:
        return dict(record)  # old inline checkpoint migrates on its next acceptance commit
    if version != STORE_VERSION:
        raise ValueError("unsupported journal retention checkpoint")
    refs = record.get("identity_shards", {})
    if not isinstance(refs, dict) or len(refs) > 256:
        raise ValueError("invalid journal identity references")
    # Validate before opening any blob directory/name.
    for bucket, name in refs.items():
        if not isinstance(name, str) or not isinstance(bucket, str):
            raise ValueError("invalid journal identity reference")
        match = _REF.fullmatch(name)
        if match is None or match[1] != bucket:
            raise ValueError("invalid journal identity reference")
    retired, bridges = (
        ({}, [])
        if not refs
        else _load_identities(str(storage.data_root()), profile, _signatures(profile, refs))
    )
    expanded = {key: value for key, value in record.items() if key != "identity_shards"}
    expanded.update(
        version=1,
        retired=retired,
        bridges=bridges,
        _identity_shards=refs,
        _identities_changed=False,
    )
    return expanded


def persist_journal_retention(
    profile: str, record: dict[str, Any], *, blobs: dict[str, str]
) -> dict[str, Any]:
    """Stage immutable payloads BEFORE the same checkpoint accepts counters/totals.

    Register every handle in checkpoint.blobs so ordinary orphan sweeping keeps
    it. Reuse unchanged identities without copying/serializing the all-ID map.
    Legacy inline records are cut over only after the new blobs are durable.
    """
    from ssgrep.accounting import collector_store as store

    if record.get("version") != 1:
        raise ValueError("persist requires expanded journal retention record")
    refs = dict(record.get("_identity_shards", {}))
    if record.get("_identities_changed", True) or "_identity_shards" not in record:
        shards: dict[str, dict[str, Any]] = {}
        for key, day in record["retired"].items():
            if not _HEX.fullmatch(key):
                raise ValueError("invalid journal identity key")
            payload = shards.setdefault(key[:2], {"version": 1, "retired": {}, "bridges": []})
            payload["retired"][key] = day
        for key in record["bridges"]:
            if not _HEX.fullmatch(key):
                raise ValueError("invalid journal bridge identity")
            payload = shards.setdefault(key[:2], {"version": 1, "retired": {}, "bridges": []})
            payload["bridges"].append(key)
        refs = {
            bucket: store.write_blob(profile, f"journal-retention-{bucket}", payload)
            for bucket, payload in sorted(shards.items())
        }
    for key in tuple(blobs):
        if key.startswith(ROOT_PREFIX):
            del blobs[key]
    blobs.update({f"{ROOT_PREFIX}{bucket}": name for bucket, name in refs.items()})
    persisted = {
        key: value
        for key, value in record.items()
        if key not in {"retired", "bridges"} and not key.startswith("_")
    }
    persisted.update(version=STORE_VERSION, identity_shards=refs)
    return persisted
