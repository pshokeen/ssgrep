"""Checkpoint-authorized journal maintenance and horizon-scoped replay protection.

The worker stages the record with accepted totals, commits it, then calls finish.
Capture never imports this module. Floors only advance; private references last
30 UTC days by default, retired phase hashes last the aggregate horizon. A
started phase spanning the floor retains measurements, not sensitive references.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.capture.journal_codec import minimal_started
from ssgrep.accounting.collect.collector import CollectError
from ssgrep.accounting.ledger.contracts import ExecutionEvent, Phase
from ssgrep.accounting.privacy import storage

if TYPE_CHECKING:
    from ssgrep.accounting.capture.journal_compact import JournalMaintenance
    from ssgrep.accounting.capture.journal_reader import JournalRead

VERSION = 1
STORE_VERSION = 2
ROOT_PREFIX = "journal-retention:"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_REF = re.compile(r"^journal-retention-([0-9a-f]{2})-([0-9a-f]{24})\.json$")
__all__ = [
    "apply_journal_retention",
    "expand_journal_retention",
    "finish_journal_retention",
    "journal_retention_refs",
    "load_journal_retention",
    "persist_journal_retention",
    "prepare_journal_retention",
    "retained_event",
    "phase_key",
]


def phase_key(event: ExecutionEvent, record: Mapping[str, Any]) -> str:
    key = bytes.fromhex(record["key"])
    return hmac.new(
        key, f"{event.execution_id}|{event.phase.value}".encode(), hashlib.sha256
    ).hexdigest()


def load_journal_retention(profile: str) -> dict[str, Any] | None:
    # The existing checkpoint is the sole authority; no second accounting database.
    from ssgrep.accounting.collect.store import load_checkpoint

    checkpoint = load_checkpoint(profile)
    if checkpoint is None:
        return None
    record = checkpoint.retention.get("journal")
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("unsupported journal retention checkpoint")
    return expand_journal_retention(profile, record)


def prepare_journal_retention(
    journal: JournalRead,
    *,
    previous: Mapping[str, Any] | None,
    horizon_day: str,
    aggregate_horizon_day: str,
) -> dict[str, Any]:
    """Stage exact read proofs and minimal phase identities alongside accepted totals.

    Does not touch any file. The caller must persist this under checkpoint.retention
    ['journal'] in the SAME durable commit as the reduction/counters. Lost precommit
    work therefore cannot authorize erasure. The old checkpoint needs no migration.
    """
    from ssgrep.accounting.collect.collector import _check_day

    _check_day(horizon_day)
    _check_day(aggregate_horizon_day)
    old = previous or {}
    if old and old.get("version") != VERSION:
        raise ValueError("unsupported journal retention checkpoint")
    horizon = max(horizon_day, old.get("horizon_day", ""))
    aggregate = max(aggregate_horizon_day, old.get("aggregate_horizon_day", ""))
    if (
        old.get("_identity_shards") is not None
        and getattr(journal, "cache_blobs", {})
        and not journal.cache_dirty
        and horizon == old.get("horizon_day")
        and aggregate == old.get("aggregate_horizon_day")
    ):
        return dict(old)
    old_retired = old.get("retired", {})
    retired = old_retired
    if aggregate > old.get("aggregate_horizon_day", ""):
        expired = {key for key, day in retired.items() if day < aggregate}
        if expired:
            retired = {key: day for key, day in retired.items() if key not in expired}
    additions = {
        key: day for key, day in journal.retired.items() if day >= aggregate and key not in retired
    }
    record: dict[str, Any] = {
        "version": VERSION,
        "key": old.get("key") or secrets.token_hex(32),
        "horizon_day": horizon,
        "aggregate_horizon_day": aggregate,
        "retired": retired,
        "proofs": [],
        "bridges": [],
    }
    if "_identity_shards" in old:
        record["_identity_shards"] = old["_identity_shards"]
    live = {
        event.execution_id
        for event in journal.events
        if event.ts_utc[:10] >= horizon
        and phase_key(event, record) not in retired
        and phase_key(event, record) not in additions
    }
    for event in journal.events:
        day, key = event.ts_utc[:10], phase_key(event, record)
        if day >= horizon:
            continue
        if (
            event.phase is Phase.STARTED
            and event.execution_id in live
            and key not in retired
            and key not in additions
        ):
            record["bridges"].append(key)
        elif day >= aggregate and key not in retired:
            additions.setdefault(key, day)
    record["bridges"] = sorted(set(record["bridges"]))
    if additions:
        retired = {**retired, **additions}
    record["retired"] = retired
    record["_identities_changed"] = (
        "_identity_shards" not in old
        or retired is not old_retired
        or record["bridges"] != old.get("bridges", [])
    )
    damaged = {damage.segment for damage in journal.damage}
    record["proofs"] = [
        proof.to_dict()
        for proof in journal.proofs
        if proof.name in damaged
        or proof.name in journal.retention_pending
        or proof.size == 0
        or any(
            retained_event(event, record) != event
            for event in journal.segment_events.get(proof.name, ())
        )
    ]
    record["queued_segments"] = max(0, len(record["proofs"]) - 32)
    record["proofs"] = record["proofs"][:32]
    return record


def apply_journal_retention(journal: JournalRead, record: Mapping[str, Any] | None) -> None:
    """Apply only the durable floor; reject copies even with mutated timestamps.

    Newly expired keys are carried on the read result until a future checkpoint
    accepts them. They cannot authorize physical deletion on their own.
    """
    if record is None:
        return
    horizon = record["horizon_day"]
    retired = record["retired"]
    live = {
        event.execution_id
        for event in journal.events
        if event.ts_utc[:10] >= horizon and phase_key(event, record) not in retired
    }
    kept = []
    for event in journal.events:
        key, day = phase_key(event, record), event.ts_utc[:10]
        if key in retired:
            journal.retired_events += 1
        elif day >= horizon:
            kept.append(event)
        elif event.phase is Phase.STARTED and event.execution_id in live:
            kept.append(minimal_started(event))
        else:
            journal.retired_events += 1
            if day >= record["aggregate_horizon_day"]:
                journal.retired[key] = day
    journal.events = kept


def retained_event(event: ExecutionEvent, record: Mapping[str, Any]) -> ExecutionEvent | None:
    """The exact post-checkpoint event representation authorized for compaction."""
    key = phase_key(event, record)
    if key in record["retired"]:
        return None
    if event.ts_utc[:10] >= record["horizon_day"]:
        return event
    if event.phase is Phase.STARTED and key in record["bridges"]:
        return minimal_started(event)
    return None


def finish_journal_retention(
    profile: str,
    *,
    generation: int,
    byte_budget: int = 32 * 1024 * 1024,
    segment_budget: int = 32,
    legacy_writers_stopped: bool = False,
) -> JournalMaintenance:
    """Resume bounded maintenance AFTER commit, under the existing worker lease.

    Reloads the durable checkpoint and refuses a generation mismatch. Repeat at
    recovery before reading the next cycle. Legacy segments predate cooperative
    locking: explicitly stop old binaries before authorizing their migration.
    """
    from ssgrep.accounting.capture.journal_compact import compact_committed_journal

    return compact_committed_journal(
        profile,
        generation=generation,
        byte_budget=byte_budget,
        segment_budget=segment_budget,
        legacy_writers_stopped=legacy_writers_stopped,
    )


# Checkpoint store: bounded references to horizon-scoped journal identity
# provenance. Identity payloads use the existing immutable collector blobs. At
# most 256 HMAC-prefix handles are checkpoint roots, not one field per accepted
# phase. Parsed immutable payloads are reused while their physical signatures
# agree.
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
    from ssgrep.accounting.collect import store

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
    from ssgrep.accounting.collect import store

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
