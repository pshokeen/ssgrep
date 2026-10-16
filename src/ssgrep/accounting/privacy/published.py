"""Closed admission and one-way reading of actual published-session-v1 documents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, cast

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.collector import CollectError
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy, Signal
from ssgrep.accounting.privacy import storage

if TYPE_CHECKING:
    from ssgrep.accounting.export.registry import Registry

FORMAT = "published-session-v1"
HISTORY_UNAVAILABLE = "published_history_unavailable"
ORIGINAL_REASON = "original_published_untrusted_model"
ARCHIVE_PREFIX = "identity-archive-"
MAX_OBLIGATIONS = 1000
NO_RECORDS = "no_records_to_delete"
NO_KEY = "no_export_key"
_LIVE_STATES = frozenset({"queued", "in_flight"})

HISTORY_FIELDS = frozenset(
    {"schema", "period", "revision", "status", "supersedes", "net", "kind", "recomputable"}
)
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "receipt_id",
        "revision",
        "root",
        "task_turn",
        "candidate_type",
        "atoms",
        "allocations",
        "exclusions",
        "costs",
        "methodology",
        "evidence_available",
        "execution_ids",
        "unit",
        "period",
        "missing_costs",
        "atom_digests",
        "credit_raw",
        "identities",
    }
)
QUANTITY_FIELDS = frozenset({"value", "unit", "basis", "method", "reason"})
IDENTITY_FIELDS = frozenset({"frequency", "tokenizer", "pricing"})
REUSE_FIELDS = frozenset({"class", "relay", "report", "action_ok", "novel", "user_supplied"})
BASES = frozenset({"observed", "reported", "estimated", "unavailable"})
REUSE_CLASSES = frozenset(
    {"none_observed", "inspected", "action_reference", "validated", "contradicted"}
)
SCENARIOS = frozenset({"conservative", "central", "optimistic"})


def _fail() -> None:
    # Never echo rejected keys/values, which may themselves contain private data.
    raise ValueError("Original published document does not match its supported closed schema.")


def _object(value: Any, required: frozenset[str], optional: frozenset[str] = frozenset()) -> dict:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        _fail()
    return value


def _string(value: Any) -> None:
    if not isinstance(value, str):
        _fail()


def _integer(value: Any, *, positive: bool = False) -> None:
    if type(value) is not int or (positive and value < 1):
        _fail()


def _boolean(value: Any) -> None:
    if type(value) is not bool:
        _fail()


def _day(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 10:
        _fail()
    try:
        if date.fromisoformat(value).isoformat() != value:
            _fail()
    except ValueError:
        _fail()


def _schema(value: Any) -> None:
    if type(value) is not int or value != 1:
        _fail()


def _strings(value: Any) -> None:
    if not isinstance(value, list):
        _fail()
    for item in value:
        _string(item)


def _pairs(value: Any, first, second) -> None:
    if not isinstance(value, list):
        _fail()
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            _fail()
        first(pair[0])
        second(pair[1])


def _scenario(value: Any) -> None:
    if not isinstance(value, str) or value not in SCENARIOS:
        _fail()


def _nonnegative(value: Any) -> None:
    _integer(value)
    if value < 0:
        _fail()


def _quantity(value: Any) -> None:
    quantity = _object(value, QUANTITY_FIELDS)
    for field in ("unit", "basis", "method", "reason"):
        _string(quantity[field])
    if quantity["basis"] not in BASES:
        _fail()
    amount = quantity["value"]
    if amount is not None:
        _integer(amount)
    if (amount is None) != (quantity["basis"] == "unavailable"):
        _fail()
    if amount is None and not quantity["reason"]:
        _fail()


def _history(document: Any) -> None:
    if not isinstance(document, list):
        _fail()
    for value in document:
        row = _object(value, HISTORY_FIELDS)
        _schema(row["schema"])
        _day(row["period"])
        _integer(row["revision"], positive=True)
        if row["supersedes"] is not None:
            _integer(row["supersedes"], positive=True)
        _string(row["status"])
        _string(row["kind"])
        _boolean(row["recomputable"])
        _quantity(row["net"])


def _reuse(value: Any) -> None:
    block = _object(value, REUSE_FIELDS)
    if not isinstance(block["class"], str) or block["class"] not in REUSE_CLASSES:
        _fail()
    for field in REUSE_FIELDS - {"class"}:
        _boolean(block[field])


def _receipt(document: Any) -> None:
    row = _object(document, RECEIPT_FIELDS, frozenset({"reuse"}))
    _schema(row["schema"])
    _day(row["period"])
    _integer(row["revision"], positive=True)
    for field in ("receipt_id", "root", "candidate_type", "methodology", "unit"):
        _string(row[field])
    if row["task_turn"] is not None:
        _string(row["task_turn"])
    _boolean(row["evidence_available"])
    for field in ("atoms", "exclusions", "execution_ids"):
        _strings(row[field])
    _pairs(row["allocations"], _scenario, _strings)
    _pairs(row["costs"], _string, _quantity)
    _pairs(row["missing_costs"], _string, _string)
    _pairs(row["atom_digests"], _string, _string)
    _pairs(row["credit_raw"], _scenario, _nonnegative)
    identities = _object(row["identities"], IDENTITY_FIELDS)
    for value in identities.values():
        _string(value)
    if "reuse" in row:
        _reuse(row["reuse"])


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate original publication JSON object key.")
        result[key] = value
    return result


def _no_float(_text):
    raise ValueError("Original publication JSON must be finite integer-valued data.")


def decode_published(raw: bytes, kind: str) -> Any:
    """Decode UTF8 with duplicate-key/float rejection, then validate every nested shape."""
    if not isinstance(raw, bytes):
        _fail()
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_no_float,
            parse_constant=_no_float,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Original publication UTF8 or nesting is invalid.") from error
    if kind == "history":
        _history(document)
    elif kind == "receipt":
        _receipt(document)
    else:
        raise ValueError("Unknown original publication kind.")
    return document


def day(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return value if date.fromisoformat(value).isoformat() == value else None
    except ValueError:
        return None


def references(document: object) -> set[str]:
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise CollectError("state_corrupt", "Original publication reference map is invalid.")
    history, receipts = document.get("history_blob"), document.get("receipt_blobs", {})
    if history is not None and not isinstance(history, str) or not isinstance(receipts, dict):
        raise CollectError("state_corrupt", "Original publication references are invalid.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in receipts.items()):
        raise CollectError("state_corrupt", "Original receipt references are invalid.")
    return ({history} if history else set()) | set(receipts.values())


def decode_original(raw: bytes, kind: str) -> Any:
    """Admit only closed actual407 publications, before any original-byte envelope write."""
    return decode_published(raw, kind)


def read_document(profile: str, filename: str, kind: str) -> tuple[dict[str, Any], Any]:
    envelope = store.read_blob(profile, filename)
    if not isinstance(envelope, dict) or envelope.get("kind") != kind:
        raise ValueError("Original publication envelope is invalid.")
    text, expected = envelope.get("original_utf8"), envelope.get("sha256")
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError("Original publication digest does not match.")
    return envelope, decode_original(text.encode(), kind)


def manifests(profile: str, blobs: dict[str, str]):
    for name, filename in sorted(blobs.items()):
        if not name.startswith(ARCHIVE_PREFIX):
            continue
        manifest = store.read_blob(profile, filename)
        documents = manifest.get("published_documents") if isinstance(manifest, dict) else None
        if documents is not None:
            references(documents)
            yield name, manifest, documents


def _original_rows(profile: str, namespace: str, documents: dict[str, Any]):
    filename = documents.get("history_blob")
    if filename is None:
        return [], (HISTORY_UNAVAILABLE,)
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Original publication namespace is unavailable.")
    envelope, originals = read_document(profile, filename, "history")
    return [
        {
            **original,
            "namespace": namespace,
            "current": False,
            "status": "superseded",
            "published_status": original["status"],
            "published_format": FORMAT,
            "revision_name": "published-original:" + namespace,
            "published_sha256": envelope["sha256"],
        }
        for original in originals
    ], ()


def archived_history(
    profile: str, blobs: dict[str, str]
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows, warnings = [], set()
    for _, manifest, documents in manifests(profile, blobs):
        original, missing = _original_rows(profile, manifest.get("methodology"), documents)
        rows.extend(original)
        warnings.update(missing)
    return rows, tuple(sorted(warnings))


def _merge_rows(current, originals, warnings):
    # Bare genuine-v1 rows are replaced only by their verified archive copy, never
    # promoted into a current scenario. Modern current rows retain their identity.
    keys = ("schema", "period", "revision", "supersedes", "net", "kind", "recomputable")

    def key(row):
        return store.dumps({name: row.get(name) for name in keys})

    original_keys = {key(row) for row in originals}
    remaining = [
        row
        for row in current
        if row.get("published_format") != FORMAT
        and not ("format" not in row and key(row) in original_keys)
    ]
    if warnings:
        remaining = [{**row, "published_warnings": list(warnings)} for row in remaining]
    return [*remaining, *originals], warnings


def merge_history(profile: str, current: list[dict[str, Any]], blobs: dict[str, str]):
    return _merge_rows(current, *archived_history(profile, blobs))


def merge_stored_history(profile: str, raw: Any):
    rows = [row for row in raw if isinstance(row, dict)] if isinstance(raw, list) else []
    # Current-only publications need no legacy archive lookup. Original wrappers,
    # however, are a derived cache: only the committed surviving archive authorizes
    # them, including during recovery after a purge committed without an active ledger.
    if rows and all(
        row.get("format") == "history_v2"
        and row.get("published_format") != FORMAT
        and not row.get("published_warnings")
        for row in rows
    ):
        return rows, ()
    try:
        checkpoint = store.load_checkpoint(profile)
        if checkpoint is None:
            remaining = [row for row in rows if row.get("published_format") != FORMAT]
            return remaining, (HISTORY_UNAVAILABLE,) if len(remaining) != len(rows) else ()
        merged, warnings = merge_history(profile, rows, checkpoint.blobs)
        capture = checkpoint.identity_migration.get("published_capture")
        if isinstance(capture, str) and not any(
            row.get("published_format") == FORMAT for row in merged
        ):
            document = store.read_blob(profile, capture)
            references(document)
            document = cast(dict[str, Any], document)  # references validated the document mapping.
            namespace = document.get("source_namespace")
            if namespace:
                originals, missing = _original_rows(profile, namespace, document)
                return _merge_rows(rows, originals, tuple(sorted(set(warnings) | set(missing))))
            return merged, (HISTORY_UNAVAILABLE,)
        return merged, warnings
    except CollectError as error:
        raise ValueError("Original publication state cannot be read.") from error


def archived_receipt_ids(profile: str, blobs: dict[str, str]) -> set[str]:
    from ssgrep.accounting.views.records import RECEIPT_ID_RE

    ids = {
        receipt
        for _, _, documents in manifests(profile, blobs)
        for receipt in documents["receipt_blobs"]
    }
    if any(RECEIPT_ID_RE.fullmatch(receipt) is None for receipt in ids):
        raise ValueError("Original receipt identity is invalid.")
    return ids


def restore_receipts(profile: str, blobs: dict[str, str], active: set[str]) -> int:
    written = 0
    allowed = archived_receipt_ids(profile, blobs)
    if not allowed:
        return 0
    directory = storage.open_profile_dir(profile, "receipts")
    try:
        for _, _, documents in manifests(profile, blobs):
            for receipt_id, filename in documents["receipt_blobs"].items():
                if receipt_id not in allowed:
                    raise ValueError("Original receipt identity is invalid.")
                envelope, receipt = read_document(profile, filename, "receipt")
                if receipt.get("receipt_id") != receipt_id:
                    raise ValueError("Original receipt identity does not match.")
                body = envelope["original_utf8"].encode()
                existing = storage.read_file(directory, receipt_id + ".json", len(body) + 1)
                if receipt_id in active and existing != body:
                    raise ValueError("Original and current receipt identities conflict.")
                if existing != body:
                    storage.atomic_write(directory, receipt_id + ".json", body)
                    written += 1
    finally:
        import os

        os.close(directory)
    return written


__all__ = [
    "ARCHIVE_PREFIX",
    "FORMAT",
    "HISTORY_UNAVAILABLE",
    "MAX_OBLIGATIONS",
    "ORIGINAL_REASON",
    "TombstonePlan",
    "TombstoneResult",
    "archived_history",
    "archived_receipt_ids",
    "decode_original",
    "day",
    "decode_published",
    "enqueue_tombstones",
    "forget_acknowledged_deletions",
    "manifests",
    "merge_history",
    "merge_stored_history",
    "pending_tombstones",
    "plan_tombstones",
    "read_document",
    "references",
    "restore_receipts",
]


@dataclass(frozen=True)
class TombstonePlan:
    """What purge would ask one destination to delete, and what it cannot promise."""

    destination_id: str
    permitted: bool
    reason: str | None  # gate code when not permitted; ``no_exported_records`` when nothing sent
    record_ids: tuple[str, ...]  # sent records selected for deletion (sorted)
    record_count: int
    obligations: tuple[str, ...]  # <= MAX_OBLIGATIONS ids the OPERATOR must erase; () if permitted
    non_reversible_metrics: (
        bool  # metrics were accepted at least once: aggregates cannot be recalled
    )


@dataclass(frozen=True)
class TombstoneResult:
    destination_id: str
    queued_records: int
    batches: int
    batch_ids: tuple[str, ...]
    skipped_reason: str | None
    obligations: tuple[str, ...]


def _bounded(record_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(record_ids[:MAX_OBLIGATIONS])


def _sent(registry: Registry, record_ids: Sequence[str]) -> list[str]:
    """Ids whose registry entry shows an acked or unresolved (sent) revision.

    A record that exists only as queued-never-sent has nothing to retract at the receiver.
    Records already carrying a deletion stay listed (the operator's obligation is unchanged);
    ``build_deletions`` is what keeps a second enqueue from planning them again.
    """
    return [
        rid
        for rid in record_ids
        if (entry := registry.entries.get(rid)) is not None
        and (entry.acked is not None or entry.unknown is not None)
    ]


def _select(
    profile: str,
    destination: str,
    registry: Registry,
    before_day: str | None,
    roots: frozenset[str] | None,
) -> list[str]:
    from ssgrep.accounting.export import registry as export_registry, spool as export_spool

    if before_day is not None:
        ids = export_registry.records_before(profile, destination, before_day, roots)
    elif roots is not None:
        ids = export_registry.records_for_roots(profile, destination, roots)
    else:
        ids = sorted(registry.entries)
    in_flight = {
        rid
        for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
        if meta.kind == "data" and meta.state == "in_flight"
        for rid in meta.record_ids
    }
    return sorted(set(_sent(registry, ids)) | (set(ids) & in_flight))


def _metrics_accepted(profile: str, destination: str) -> bool:
    """Whether the metrics signal was ever acked; an unreadable document cannot rule it out."""
    from ssgrep.accounting.export import acks
    from ssgrep.accounting.export.acks import DeliveryCorrupt

    try:
        state = acks.read_delivery(profile, destination, Signal.METRICS)
    except DeliveryCorrupt:
        return True
    return state is not None and state.last_acked_generation is not None


def plan_tombstones(
    profile: str,
    policy: ProfilePolicy,
    *,
    before_day: str | None,
    root_keys: frozenset[str] | None,
) -> list[TombstonePlan]:
    """One plan per policy destination that exported anything; reads only, creates nothing.

    A destination qualifies when its registry has records or its metrics were ever acked (so
    the non-reversible-metrics limitation is reported even with no per-task records).
    ``before_day`` keeps records with ``period < before_day``; ``root_keys`` keeps records of
    those conversations; both together intersect; neither selects every record. A corrupt
    registry raises ``RegistryCorrupt`` (what was exported is then unknown).
    """
    from ssgrep.accounting.export import consent as export_policy, registry as export_registry

    plans: list[TombstonePlan] = []
    for dest in policy.destinations:
        destination = dest.destination_id
        registry = export_registry.load_registry(profile, destination)
        metrics = _metrics_accepted(profile, destination)
        if not registry.entries and not metrics:
            continue
        ids = tuple(_select(profile, destination, registry, before_day, root_keys))
        verdict = export_policy.gate(
            profile, destination, Signal.LOGS, need_enabled=True, need_unpaused=False
        )
        reason = verdict.reason if not verdict.ok else (None if ids else NO_RECORDS)
        plans.append(
            TombstonePlan(
                destination,
                verdict.ok,
                reason,
                ids,
                len(ids),
                () if verdict.ok else _bounded(ids),
                metrics,
            )
        )
    return plans


def _skipped(plan: TombstonePlan, reason: str | None) -> TombstoneResult:
    owed = plan.obligations if not plan.permitted else _bounded(plan.record_ids)
    return TombstoneResult(plan.destination_id, 0, 0, (), reason, owed)


def _refresh_counts(profile: str, destination: str) -> None:
    """Pending = queued + in-flight logs batches; unknown = those already attempted."""
    from ssgrep.accounting.export import acks, spool as export_spool

    live = [
        meta
        for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
        if meta.state in _LIVE_STATES
    ]
    pending, unknown = len(live), sum(1 for meta in live if meta.attempts > 0)
    acks.update_delivery(
        profile, destination, Signal.LOGS, lambda d: acks.set_counts(d, pending, unknown)
    )


def _enqueue_one(profile: str, plan: TombstonePlan, now: datetime) -> TombstoneResult:
    from ssgrep.accounting.export import (
        consent as export_policy,
        projection as export_build,
        registry as export_registry,
        spool as export_spool,
    )

    if not plan.permitted:
        return _skipped(plan, plan.reason)
    if not plan.record_ids:
        return _skipped(plan, plan.reason or NO_RECORDS)
    destination = plan.destination_id
    verdict = export_policy.gate(
        profile, destination, Signal.LOGS, need_enabled=True, need_unpaused=False
    )
    if not verdict.ok or verdict.destination is None:
        return _skipped(plan, verdict.reason)
    key, ephemeral = export_build.key_for(profile, create=False)
    if ephemeral:
        return _skipped(plan, NO_KEY)
    registry = export_registry.load_registry(profile, destination)
    deletions = export_registry.build_deletions(
        registry, plan.record_ids, key=key, destination=destination
    )
    stamp = int(now.timestamp())
    cand = export_build.build_tombstones(
        profile, verdict.destination, verdict.policy, deletions, key, stamp=stamp
    )
    created = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    batch_ids: list[str] = []
    queued = 0
    for batch, members in zip(cand.batches, cand.planned, strict=True):
        export_spool.write_batch(
            profile, destination, Signal.LOGS, batch, created_utc=created, scope=cand.scope
        )
        export_registry.mark_queued(profile, destination, members, batch.batch_id)
        batch_ids.append(batch.batch_id)
        queued += len(members)
    if batch_ids:
        _refresh_counts(profile, destination)
    return TombstoneResult(destination, queued, len(batch_ids), tuple(batch_ids), None, ())


def enqueue_tombstones(
    profile: str, policy: ProfilePolicy, plans: Sequence[TombstonePlan], *, now: datetime
) -> list[TombstoneResult]:
    """Spool deletion batches for every still-permitted plan; never bypasses revocation.

    Each destination is re-gated against the CURRENT policy right before it is written (the
    ``policy`` argument is the planning-time snapshot and is not trusted for authorization or
    for the scope stamped on the batches). Idempotent: a record whose latest known revision is
    already a deletion (queued, sent or acked) is not planned again.
    """
    del policy
    return [_enqueue_one(profile, plan, now) for plan in plans]


def forget_acknowledged_deletions(profile: str, destination: str) -> int:
    """Drop registry entries whose latest state is a deletion the collector accepted."""
    from ssgrep.accounting.export.registry import DELETION, Registry, update_registry

    removed: list[str] = []

    def prune(registry: Registry) -> Registry:
        kept = {}
        for record_id, entry in registry.entries.items():
            acked, latest = entry.acked, entry.latest
            if (
                acked is not None
                and acked.type == DELETION
                and latest is not None
                and latest.revision <= acked.revision
            ):
                removed.append(record_id)
            else:
                kept[record_id] = entry
        return Registry(kept)

    update_registry(profile, destination, prune, create=False)
    return len(removed)


def pending_tombstones(profile: str, destination: str) -> int:
    """Tombstone batches still waiting to be delivered (queued or in flight)."""
    from ssgrep.accounting.export import spool as export_spool

    return sum(
        1
        for meta in export_spool.list_batches(profile, destination, Signal.LOGS)
        if meta.kind == "tombstone" and meta.state in _LIVE_STATES
    )
