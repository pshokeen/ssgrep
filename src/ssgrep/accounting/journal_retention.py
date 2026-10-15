"""Checkpoint-authorized journal maintenance and horizon-scoped replay protection.

The worker stages the record with accepted totals, commits it, then calls finish.
Capture never imports this module. Floors only advance; private references last
30 UTC days by default, retired phase hashes last the aggregate horizon. A
started phase spanning the floor retains measurements, not sensitive references.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.contracts import ExecutionEvent, Phase
from ssgrep.accounting.journal_codec import minimal_started
from ssgrep.accounting.journal_retention_store import (
    expand_journal_retention,
    journal_retention_refs,
    persist_journal_retention,
)

if TYPE_CHECKING:
    from ssgrep.accounting.journal_compact import JournalMaintenance
    from ssgrep.accounting.journal_reader import JournalRead

VERSION = 1
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
    from ssgrep.accounting.collector_store import load_checkpoint

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
    from ssgrep.accounting.retention_hooks import _check_day

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
    from ssgrep.accounting.journal_compact import compact_committed_journal

    return compact_committed_journal(
        profile,
        generation=generation,
        byte_budget=byte_budget,
        segment_budget=segment_budget,
        legacy_writers_stopped=legacy_writers_stopped,
    )
