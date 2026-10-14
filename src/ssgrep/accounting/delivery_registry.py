"""Ledger-log registry transitions the delivery cycle needs beyond ``export_records``.

``export_records`` settles one queued batch at a time; the cycle also needs

* ``queue_batches``: register EVERY batch of one enqueue in a single atomic registry write, so a
  crash leaves either none or all of them registered (never a half-registered snapshot that a
  rebuild would plan differently and split into conflicting batch ids);
* ``promote_unknown``: a batch that was sent (timeout, restart in flight) and later accepted turns
  its ``unknown`` record states into ``acked``;
* ``require_queued_spool``: refuse lost/corrupt registered bytes rather than rebuild a smaller
  generation from surviving metas and incorrectly acknowledge the missing records.

A batch whose bytes are later discarded keeps its ``unknown`` states on purpose: the receiver may
hold that revision, so the planner must supersede it (revision + 1) rather than reuse the number.

Records are matched by (record id, revision) read from the spooled body, so no ledger access and
no extra state is needed. Only pseudonymous ids and integers are handled here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace

from ssgrep.accounting import export_spool
from ssgrep.accounting.export_records_model import Planned, Registry
from ssgrep.accounting.export_records_state import queue
from ssgrep.accounting.export_registry import load_registry, update_registry
from ssgrep.accounting.ledger_contracts import Signal

Pair = tuple[str, int]


def body_pairs(body: bytes) -> list[Pair]:
    """(record_id, revision) of every log record in a serialized OTLP logs request."""
    found: list[Pair] = []
    try:
        document = json.loads(body)
        for resource in document.get("resourceLogs", []):
            for scope in resource.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    inner = json.loads(record["body"]["stringValue"])
                    found.append((inner["record_id"], int(inner["revision"])))
    except (ValueError, KeyError, TypeError, AttributeError):
        return []
    return found


def queue_batches(
    profile: str, destination: str, groups: Sequence[tuple[str, Sequence[Planned]]]
) -> None:
    """Register ``(batch_id, planned...)`` groups as queued, atomically, in order."""

    def change(registry: Registry) -> Registry:
        for batch_id, planned in groups:
            registry = queue(registry, planned, batch_id)
        return registry

    update_registry(profile, destination, change)


def require_queued_spool(profile: str, destination: str) -> None:
    """Every queued registry reference must still have its original readable body and meta.

    A missing meta is not proof that the batch was never attempted. Preserve the registry and
    acknowledgment truth until the pair is restored or explicit retention/purge settles it.
    """
    registry = load_registry(profile, destination)
    queued = {state.batch for entry in registry.entries.values() for state in entry.queued}
    for batch_id in queued:
        assert batch_id is not None  # strict registry decoding requires a batch for queued state
        try:
            export_spool.read_body(profile, destination, Signal.LOGS, batch_id)
        except FileNotFoundError:
            raise export_spool.SpoolCorrupt("registry references a missing spooled batch") from None


def promote_unknown(profile: str, destination: str, pairs: Iterable[Pair]) -> None:
    """``unknown`` states at exactly these revisions were accepted: they become ``acked``."""
    wanted = list(pairs)

    def change(registry: Registry) -> Registry:
        entries = dict(registry.entries)
        for record_id, revision in wanted:
            entry = entries.get(record_id)
            if entry is None or entry.unknown is None or entry.unknown.revision != revision:
                continue
            held = entry.acked
            acked = held if held is not None and held.revision > revision else entry.unknown
            entries[record_id] = replace(entry, acked=acked, unknown=None)
        return Registry(entries)

    update_registry(profile, destination, change, create=False)
