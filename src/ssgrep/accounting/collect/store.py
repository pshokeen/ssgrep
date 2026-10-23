"""Durable collector state: one atomic checkpoint, immutable evidence segments, state blobs.

Layout under ``<data>/accounting/<profile>/collector/`` (all 0600 in 0700 directories)::

    checkpoint.json          the single commit point (write + fsync + rename + directory fsync)
    evidence/seg-<id>.jsonl  immutable newline-complete retained RawEvidence segments
    state/<name>-<sha>.json  content-addressed blobs (ledger, counters, executions, ...)

Exactly-once recovery: a cycle writes new segments/blobs FIRST (each unreferenced until the
checkpoint names it), then commits the checkpoint. A crash before the rename leaves only
orphans, which the next cycle sweeps; the checkpoint still names the previous complete state,
so accepted evidence is neither lost nor reduced twice. A checkpoint from a NEWER schema is
refused untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ssgrep.accounting.collect import store
from ssgrep.accounting.ledger.contracts import (
    SCHEMA_VERSION,
    RawEvidence,
)
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.storage import from_plain, to_plain


class CollectError(Exception):
    """A collection failure with a stable condition (never raw exception text)."""

    def __init__(
        self, condition: str, message: str, command: str = "ssgrep savings status"
    ) -> None:
        super().__init__(message)
        self.condition = condition
        self.command = command


CHECKPOINT = "checkpoint.json"
EVIDENCE_DIR = "evidence"
STATE_DIR = "state"
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_BLOB_BYTES = 512 * 1024 * 1024
SEGMENT_TARGET_BYTES = 32 * 1024 * 1024
MAX_SEGMENTS = 32
CONSUMED_KINDS = frozenset(
    {
        "tool_call",
        "tool_result",
        "user_turn",
        "session_header",
        "lineage",
        "pending_tool",
        "pending_message",
        "abandoned_pending",
        "ownership_claim",
    }
)


def _reject_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate keys are not allowed in accounting data")
        result[key] = value
    return result


def loads(data: bytes) -> Any:
    return json.loads(data, parse_float=_reject_float, object_pairs_hook=_unique_object)


def dumps(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def source_key(source_id: str) -> str:
    """Opaque 16-hex source handle: safe to show; the path never leaves the store."""
    return hashlib.sha256(source_id.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SegmentRef:
    name: str
    records: int
    size: int
    origin: str = "live"  # live | backfill
    index: str | None = None  # immutable session/native-key index manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "records": self.records,
            "size": self.size,
            "origin": self.origin,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SegmentRef:
        return cls(
            str(raw["name"]),
            int(raw["records"]),
            int(raw["size"]),
            str(raw["origin"]),
            None if raw.get("index") is None else str(raw["index"]),
        )


@dataclass
class SourceRecord:
    """Everything remembered about one native source between cycles."""

    runtime: str
    source_id: str  # private: ``runtime:/path``; shown only as ``source_key``
    state: dict[str, Any] = field(default_factory=dict)  # readers.base.SourceState.to_dict()
    size: int = 0
    status: str = "ok"  # ok | missing | unreadable
    baseline: str = "live"  # skipped = pre-existing history not imported | live
    skipped_bytes: int = 0
    pending_tail: int = 0
    more: bool = False
    complete: bool = False
    mtime_ns: int = 0
    last_ts: str | None = None
    resets: int = 0
    gaps: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return source_key(self.source_id)

    @property
    def offset(self) -> int:
        return int(self.state.get("offset", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "source_id": self.source_id,
            "state": self.state,
            "size": self.size,
            "status": self.status,
            "baseline": self.baseline,
            "skipped_bytes": self.skipped_bytes,
            "pending_tail": self.pending_tail,
            "more": self.more,
            "complete": self.complete,
            "mtime_ns": self.mtime_ns,
            "last_ts": self.last_ts,
            "resets": self.resets,
            "gaps": dict(sorted(self.gaps.items())),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SourceRecord:
        last = raw.get("last_ts")
        return cls(
            runtime=str(raw["runtime"]),
            source_id=str(raw["source_id"]),
            state=dict(raw.get("state", {})),
            size=int(raw.get("size", 0)),
            status=str(raw.get("status", "ok")),
            baseline=str(raw.get("baseline", "live")),
            skipped_bytes=int(raw.get("skipped_bytes", 0)),
            pending_tail=int(raw.get("pending_tail", 0)),
            more=bool(raw.get("more", False)),
            complete=bool(raw.get("complete", False)),
            mtime_ns=int(raw.get("mtime_ns", 0)),
            last_ts=None if last is None else str(last),
            resets=int(raw.get("resets", 0)),
            gaps={str(k): int(v) for k, v in dict(raw.get("gaps", {})).items()},
        )


@dataclass
class Checkpoint:
    """The commit record of the collector; every field is plain JSON data."""

    profile: str
    generation: int = 0
    cycles: int = 0
    baseline_ts: str | None = None
    since_day: str | None = None
    methodology: str = ""
    input_digest: str = ""
    journal_sig: str = ""
    compacted_day: str | None = None
    sources: dict[str, SourceRecord] = field(default_factory=dict)
    segments: list[SegmentRef] = field(default_factory=list)
    blobs: dict[str, str] = field(default_factory=dict)
    journal: dict[str, int] = field(default_factory=dict)
    retention: dict[str, Any] = field(default_factory=dict)
    backfills: list[dict[str, Any]] = field(default_factory=list)
    backfill_job: dict[str, Any] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    frequency: dict[str, int] = field(default_factory=dict)  # last unpinned corpus count
    schema: int = SCHEMA_VERSION
    collection_scope: dict[str, Any] = field(default_factory=dict)
    scope_epoch: int = 1
    identity_method: str = ""
    identity_migration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile,
            "generation": self.generation,
            "cycles": self.cycles,
            "baseline_ts": self.baseline_ts,
            "since_day": self.since_day,
            "methodology": self.methodology,
            "input_digest": self.input_digest,
            "journal_sig": self.journal_sig,
            "compacted_day": self.compacted_day,
            "sources": {k: v.to_dict() for k, v in sorted(self.sources.items())},
            "segments": [s.to_dict() for s in self.segments],
            "blobs": dict(sorted(self.blobs.items())),
            "journal": dict(sorted(self.journal.items())),
            "retention": self.retention,
            "backfills": self.backfills,
            "backfill_job": self.backfill_job,
            "dropped": dict(sorted(self.dropped.items())),
            "frequency": dict(sorted(self.frequency.items())),
            "collection_scope": self.collection_scope,
            "scope_epoch": self.scope_epoch,
            "identity_method": self.identity_method,
            "identity_migration": self.identity_migration,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Checkpoint:
        return cls(
            profile=str(raw["profile"]),
            generation=int(raw["generation"]),
            cycles=int(raw.get("cycles", 0)),
            baseline_ts=raw.get("baseline_ts"),
            since_day=raw.get("since_day"),
            methodology=str(raw.get("methodology", "")),
            input_digest=str(raw.get("input_digest", "")),
            journal_sig=str(raw.get("journal_sig", "")),
            compacted_day=raw.get("compacted_day"),
            sources={str(k): SourceRecord.from_dict(v) for k, v in raw["sources"].items()},
            segments=[SegmentRef.from_dict(s) for s in raw["segments"]],
            blobs={str(k): str(v) for k, v in raw["blobs"].items()},
            journal={str(k): int(v) for k, v in raw.get("journal", {}).items()},
            retention=dict(raw.get("retention", {})),
            backfills=list(raw.get("backfills", [])),
            backfill_job=dict(raw.get("backfill_job", {})),
            dropped={str(k): int(v) for k, v in raw.get("dropped", {}).items()},
            frequency={str(k): int(v) for k, v in raw.get("frequency", {}).items()},
            collection_scope=raw.get("collection_scope", {}),
            scope_epoch=max(1, int(raw.get("scope_epoch", 1))),
            identity_method=str(raw.get("identity_method", "")),
            identity_migration=dict(raw.get("identity_migration", {})),
        )


PARTITIONS_BLOB = "partitions"


@dataclass
class PartitionState:
    metas: dict[str, str] = field(default_factory=dict)
    parts: dict[str, dict] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    journal_digests: dict[str, str] = field(default_factory=dict)
    metadata_pending: list[str] = field(default_factory=list)
    pending_groups: dict[str, list[str]] = field(default_factory=dict)
    operation: str | None = None
    blocked_reason: str | None = None
    unavailable: dict[str, str] = field(default_factory=dict)
    maintenance: str | None = None
    methodology: str = ""
    frequency_refs: list[str] = field(default_factory=list)
    reference_index_complete: bool = True

    @property
    def pending_projections(self) -> tuple[str, ...]:
        """Current cached components not yet accepted under lookup-level identity."""
        from ssgrep.accounting.collect.state import CANDIDATE_IDENTITY_METHOD

        return tuple(
            ident
            for ident, part in sorted(self.parts.items())
            if part.get("candidate_identity_method") != CANDIDATE_IDENTITY_METHOD
        )

    def to_dict(self):
        return {
            "schema": 1,
            "metas": self.metas,
            "parts": self.parts,
            "pending": self.pending,
            "journal_digests": self.journal_digests,
            "metadata_pending": self.metadata_pending,
            "pending_groups": self.pending_groups,
            "operation": self.operation,
            "blocked_reason": self.blocked_reason,
            "unavailable": self.unavailable,
            "maintenance": self.maintenance,
            "methodology": self.methodology,
            "frequency_refs": self.frequency_refs,
            "reference_index_complete": self.reference_index_complete,
        }


def load_partitions(profile: str, checkpoint: store.Checkpoint) -> PartitionState:
    raw = store.read_blob(profile, checkpoint.blobs.get(PARTITIONS_BLOB))
    if raw is None:
        return PartitionState()
    try:
        if not isinstance(raw, dict) or type(raw.get("schema")) is not int or raw["schema"] != 1:
            raise ValueError("unsupported partition schema")
        raw.pop("schema")
        state = PartitionState(**raw)
        if not isinstance(state.pending, list) or not isinstance(state.parts, dict):
            raise ValueError("invalid partition queue")
        if (
            not isinstance(state.metadata_pending, list)
            or any(not isinstance(name, str) for name in (*state.pending, *state.metadata_pending))
            or state.operation is not None
            and not isinstance(state.operation, str)
            or state.blocked_reason not in (None, "reduction_blocked")
        ):
            raise ValueError("invalid partition progress")
        if not isinstance(state.pending_groups, dict) or any(
            not isinstance(ident, str)
            or not isinstance(members, list)
            or any(not isinstance(name, str) for name in members)
            for ident, members in state.pending_groups.items()
        ):
            raise ValueError("invalid pending component plan")
        return state
    except (ValueError, KeyError, TypeError) as error:
        raise CollectError("state_corrupt", "Collector partition state is invalid.") from error


def save_partitions(profile: str, state: PartitionState, blobs: dict[str, str]) -> None:
    blobs[PARTITIONS_BLOB] = store.write_blob(profile, PARTITIONS_BLOB, state.to_dict())


def partition_references(profile: str, filename: str) -> set[str]:
    raw = store.read_blob(profile, filename)
    if not isinstance(raw, dict) or type(raw.get("schema")) is not int or raw["schema"] != 1:
        raise CollectError("state_corrupt", "Collector partition manifest is invalid.")
    return set(raw["metas"].values()) | {part["blob"] for part in raw["parts"].values()}


# ---- checkpoint ------------------------------------------------------------------------


def load_checkpoint(profile: str) -> Checkpoint | None:
    """The committed checkpoint, or None before the first cycle."""
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return None
    except storage.UnsafeStorageError as error:
        raise CollectError("storage_unsafe", f"Collector storage is unsafe: {error}") from error
    try:
        data = storage.read_file(directory, CHECKPOINT, MAX_CHECKPOINT_BYTES)
    except storage.UnsafeStorageError as error:
        raise CollectError("storage_unsafe", "Collector storage is unsafe.") from error
    except (ValueError, OSError) as error:
        raise CollectError("state_corrupt", "Collector checkpoint is unreadable.") from error
    finally:
        os.close(directory)
    if data is None:
        return None
    try:
        raw = loads(data)
        version = raw["schema"]
    except (ValueError, KeyError, TypeError) as error:
        raise CollectError("state_corrupt", "Collector checkpoint is not valid JSON.") from error
    if isinstance(version, bool) or not isinstance(version, int) or version > SCHEMA_VERSION:
        raise CollectError(
            "unsupported_schema",
            f"Collector state uses schema {version!r}; this ssgrep reads schema {SCHEMA_VERSION}. "
            "Nothing was changed; use a newer ssgrep.",
        )
    try:
        return Checkpoint.from_dict(raw)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise CollectError("state_corrupt", "Collector checkpoint fields are invalid.") from error


def commit_checkpoint(profile: str, checkpoint: Checkpoint) -> None:
    """Atomically publish ``checkpoint`` (the exactly-once commit point)."""
    directory = storage.open_profile_dir(profile, "collector")
    try:
        storage.atomic_write(directory, CHECKPOINT, dumps(checkpoint.to_dict()) + b"\n")
    finally:
        os.close(directory)


# ---- evidence segments -----------------------------------------------------------------


def write_segments(
    profile: str, items: Iterable[RawEvidence], *, origin: str = "live"
) -> list[SegmentRef]:
    """Serialize immutable compact segments; rewriting never admits plaintext cache entries."""
    from ssgrep.accounting.collect.cycle import IndexWriter
    from ssgrep.accounting.collect.replay import evidence_line

    refs: list[SegmentRef] = []
    directory = storage.open_profile_dir(profile, "collector", EVIDENCE_DIR)
    try:
        buffer = bytearray()
        count = 0
        index = IndexWriter()

        def flush() -> None:
            nonlocal buffer, count, index
            if not count:
                return
            body = bytes(buffer)
            name = f"seg-{hashlib.sha256(body).hexdigest()[:24]}.jsonl"
            storage.write_once(directory, name, body)  # content addressed: idempotent
            refs.append(SegmentRef(name, count, len(body), origin, index.write(profile)))
            buffer, count, index = bytearray(), 0, IndexWriter()

        for item in items:
            line = evidence_line(item) + b"\n"
            index.add(item, len(buffer), line)
            buffer += line
            count += 1
            if len(buffer) >= SEGMENT_TARGET_BYTES:
                flush()
        flush()
    finally:
        os.close(directory)
    return refs


def read_segment(profile: str, ref: SegmentRef) -> Iterator[RawEvidence]:
    """Stream one committed segment; a damaged or missing committed segment is state_corrupt."""
    from ssgrep.accounting.collect.replay import evidence_from_line

    try:
        directory = storage.open_profile_dir(profile, "collector", EVIDENCE_DIR, create=False)
    except FileNotFoundError as error:
        raise CollectError("state_corrupt", "Retained evidence directory is missing.") from error
    try:
        data = storage.read_file(directory, ref.name, ref.size + 1)
    except ValueError as error:
        raise CollectError("state_corrupt", "A retained evidence segment is oversized.") from error
    finally:
        os.close(directory)
    if data is None or len(data) != ref.size:
        raise CollectError("state_corrupt", "A retained evidence segment is missing or damaged.")
    for line in data.split(b"\n"):
        if line:
            try:
                yield evidence_from_line(line)
            except (ValueError, KeyError, TypeError) as error:
                raise CollectError(
                    "state_corrupt", "A retained evidence record is damaged."
                ) from error


def iter_evidence(
    profile: str,
    refs: Iterable[SegmentRef],
    *,
    sessions: set[tuple[str, str]] | None = None,
    native_keys: set[tuple[str, str, str]] | None = None,
    index_cache: dict | None = None,
) -> Iterator[RawEvidence]:
    from ssgrep.accounting.collect.cycle import iter_selected

    for ref in refs:
        if sessions is None and native_keys is None:
            yield from read_segment(profile, ref)
        else:
            yield from iter_selected(
                profile, ref, sessions=sessions, native_keys=native_keys, index_cache=index_cache
            )


# ---- state blobs -----------------------------------------------------------------------


def write_blob(profile: str, name: str, value: Any) -> str:
    """Content-addressed JSON blob; returns the file name to put in ``Checkpoint.blobs``."""
    body = dumps(value)
    file_name = f"{name}-{hashlib.sha256(body).hexdigest()[:24]}.json"
    directory = storage.open_profile_dir(profile, "collector", STATE_DIR)
    try:
        storage.write_once(directory, file_name, body)
    finally:
        os.close(directory)
    return file_name


def read_blob(profile: str, file_name: str | None) -> Any | None:
    """A blob's JSON value; None when no blob is referenced. A missing one is state_corrupt."""
    if file_name is None:
        return None
    try:
        directory = storage.open_profile_dir(profile, "collector", STATE_DIR, create=False)
    except FileNotFoundError as error:
        raise CollectError("state_corrupt", "Collector state directory is missing.") from error
    try:
        data = storage.read_file(directory, file_name, MAX_BLOB_BYTES)
    except ValueError as error:
        raise CollectError("state_corrupt", f"State blob {file_name} is too large.") from error
    finally:
        os.close(directory)
    if data is None:
        raise CollectError("state_corrupt", f"State blob {file_name} is missing.")
    try:
        value = loads(data)
        if value is None:
            raise ValueError("referenced state cannot be null")
        return value
    except ValueError as error:
        raise CollectError("state_corrupt", f"State blob {file_name} is damaged.") from error


# ---- recovery --------------------------------------------------------------------------


def sweep_orphans(profile: str, checkpoint: Checkpoint | None) -> int:
    """Delete segments/blobs/temporaries no committed checkpoint references; returns the count.

    Safe only for the lease holder: an uncommitted segment of a crashed cycle is exactly what
    must disappear so its evidence is read again and reduced once.
    """

    keep_segments, keep_blobs = (
        referenced_files(profile, checkpoint) if checkpoint else (set(), set())
    )
    removed = 0
    for sub, keep in ((EVIDENCE_DIR, keep_segments), (STATE_DIR, keep_blobs)):
        try:
            directory = storage.open_profile_dir(profile, "collector", sub, create=False)
        except FileNotFoundError:
            continue
        try:
            for name in storage.list_names(directory):
                temporary = name.startswith(".") and name.endswith(".tmp")
                if temporary or (not name.startswith(".") and name not in keep):
                    removed += storage.delete_file(directory, name)
        finally:
            os.close(directory)
    return removed


"""Immutable compact-evidence indexes: session ranges and native-identity shards.

Only blob handles enter the checkpoint. Identity tables live in 256 independently
addressed shards, so admission never copies all retained IDs into a new checkpoint.
Each selected line is checked against its indexed SHA256 before decoding.
"""


"""Freeze per-lookup inputs without reserializing every ID on one new lookup."""


"""Per-root persistence for the existing LedgerState, with resumable legacy conversion.

The checkpoint references one immutable manifest. A legacy base remains readable
while at most 32 unchanged roots per cycle are materialized; changed roots always
get their own canonical blob. Explicit purge writes have no legacy base, so erased
metadata cannot remain reachable through a conversion overlay.
"""


"""Root/session native-row manifests in the existing immutable collector store."""


"""Reuse existing RootLedger evaluations through immutable per-root blob handles."""


ROOT_EVALUATIONS_BLOB = "root-evaluations"


def restore_cache(profile, ledger, blobs):
    from ssgrep.accounting.ledger.valuation import Leaf
    from ssgrep.accounting.views.records import ReceiptRecord

    raw = store.read_blob(profile, blobs.get(ROOT_EVALUATIONS_BLOB))
    if raw is None:
        return
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise CollectError("state_corrupt", "Root evaluation cache is invalid.")
    if raw.get("methodology") != ledger.methodology.hash:
        return  # A method namespace can never borrow another method's canonical rows.
    try:
        ledger._root_cache_files = dict(raw["roots"])
        ledger._root_cache_keys = {}
        for root, filename in raw["roots"].items():
            rows = store.read_blob(profile, filename)
            if not isinstance(rows, list):
                raise ValueError("invalid root evaluation rows")
            for row in rows:
                keys = tuple(row["keys"])
                leaves = tuple(from_plain(Leaf, leaf) for leaf in row["leaves"])
                receipts = {
                    name: from_plain(ReceiptRecord, value)
                    for name, value in row["receipts"].items()
                }
                ledger._eval_cache[(root, keys)] = (leaves, receipts)
                ledger._root_cache_keys.setdefault(root, []).append(keys)
    except (KeyError, ValueError, TypeError) as error:
        raise CollectError("state_corrupt", "Root evaluation cache rows are invalid.") from error


def save_cache(profile, ledger, blobs):
    grouped = defaultdict(list)
    for root, keys in sorted(ledger._eval_cache):
        grouped[root].append(keys)
    roots = {}
    old_files = ledger._root_cache_files
    old_keys = ledger._root_cache_keys
    for root, keys in sorted(grouped.items()):
        if keys == old_keys.get(root) and root in old_files:
            roots[root] = old_files[root]
            continue
        rows = []
        for key in keys:
            leaves, receipts = ledger._eval_cache[(root, key)]
            rows.append(
                {"keys": list(key), "leaves": to_plain(leaves), "receipts": to_plain(receipts)}
            )
        roots[root] = store.write_blob(profile, "root-evaluation", rows)
    blobs[ROOT_EVALUATIONS_BLOB] = store.write_blob(
        profile,
        ROOT_EVALUATIONS_BLOB,
        {
            "schema": 1,
            "methodology": ledger.methodology.hash,
            "roots": roots,
        },
    )


def root_cache_references(profile, filename):
    raw = store.read_blob(profile, filename)
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != 1
        or not isinstance(raw.get("roots"), dict)
    ):
        raise CollectError("state_corrupt", "Root evaluation cache manifest is invalid.")
    return set(raw["roots"].values())


def referenced_files(profile: str, checkpoint: Checkpoint) -> tuple[set[str], set[str]]:
    segments = {ref.name for ref in checkpoint.segments}
    refs = list(checkpoint.segments)
    blobs = set(checkpoint.blobs.values())
    pending = dict(checkpoint.identity_migration)
    from ssgrep.accounting.privacy.published import references as publication_references

    capture = pending.get("published_capture")
    if capture is not None:
        if not isinstance(capture, str):
            raise CollectError(
                "state_corrupt", "Original publication capture reference is invalid."
            )
        blobs.add(capture)
        blobs.update(publication_references(read_blob(profile, capture)))
    blobs.update(str(name) for name in pending.get("superseded_blobs", {}).values())
    nested = list(checkpoint.blobs.items())
    nested.extend(pending.get("superseded_blobs", {}).items())
    for name, filename in list(nested):
        if name.split("-scope-", 1)[0] == "frequency-maintenance":
            job = read_blob(profile, filename)
            if not isinstance(job, dict) or job.get("schema") != 1:
                raise CollectError("state_corrupt", "Frequency maintenance manifest is invalid.")
            for kind, child in job.get("sources", ()):
                blobs.add(child)
                nested.append((f"frequency-maintenance-ref:{kind}", child))
    for field_name in ("staged_segments", "compact_legacy_segments"):
        for raw in pending.get(field_name, []):
            ref = SegmentRef.from_dict(raw)
            segments.add(ref.name)
            refs.append(ref)
    manifests = [
        filename
        for name, filename in nested
        if name.rsplit(":", 1)[-1].startswith("identity-archive-")
    ]
    seen = set()
    while manifests:
        filename = manifests.pop()
        if filename in seen:
            continue
        seen.add(filename)
        if len(seen) > 32:
            raise CollectError("state_corrupt", "Too many identity archive manifests.")
        raw = read_blob(profile, filename)
        if not isinstance(raw, dict) or raw.get("schema") != 1:
            raise CollectError("state_corrupt", "Identity archive manifest is invalid.")
        documents = raw.get("published_documents")
        if documents is not None:
            blobs.update(publication_references(documents))
        for segment in raw.get("segments", []):
            ref = SegmentRef.from_dict(segment)
            segments.add(ref.name)
            refs.append(ref)
        for name, child in raw.get("blobs", {}).items():
            blobs.add(str(child))
            nested.append((f"{filename}:{name}", str(child)))
            if name.startswith("identity-archive-"):
                manifests.append(str(child))
    from ssgrep.accounting.collect.cycle import index_references
    from ssgrep.accounting.collect.partitions import reduction_references
    from ssgrep.accounting.collect.read import journal_references
    from ssgrep.accounting.collect.state import assignment_references, ledger_references

    blobs.update(index_references(profile, refs))
    for name, filename in nested:
        kind = name.rsplit(":", 1)[-1].split("-scope-", 1)[0]
        if kind == "partitions":
            blobs.update(partition_references(profile, filename))
        elif kind == "journal-cache":
            blobs.update(journal_references(profile, filename))
        elif kind == "root-evaluations":
            blobs.update(root_cache_references(profile, filename))
        elif kind.startswith("ledger-"):
            blobs.update(ledger_references(read_blob(profile, filename)))
        elif kind == "reduction":
            blobs.update(reduction_references(read_blob(profile, filename)))
        elif kind == "frequency-assignments":
            blobs.update(assignment_references(read_blob(profile, filename)))
    return segments, blobs


MAX_ATOMIC_SEGMENT_BYTES = MAX_BLOB_BYTES


def migrate_evidence(
    profile: str, checkpoint: Checkpoint, *, byte_budget: int = MAX_ATOMIC_SEGMENT_BYTES
) -> Checkpoint:
    """Advance checkpoint-first conversion of retained evidence to compact format."""
    from dataclasses import replace

    from ssgrep.accounting.collect.collector import CollectError as MigrationError
    from ssgrep.accounting.collect.replay import COMPACT
    from ssgrep.accounting.collect.replay_evidence import maintain_cache

    maintain_cache(profile)
    retention = dict(checkpoint.retention)
    ready = (
        retention.get("evidence_format") == COMPACT
        and retention.get("archive_evidence_format") == COMPACT
    )
    if ready:
        if not retention.get("evidence_cleanup_pending"):
            return checkpoint
        sweep_orphans(profile, checkpoint)
        retention.pop("evidence_cleanup_pending", None)
        cleaned = replace(checkpoint, retention=retention)
        commit_checkpoint(profile, cleaned)
        return cleaned
    job = dict(retention.get("evidence_compaction", {}))
    job.setdefault("phase", "archives" if retention.get("evidence_format") == COMPACT else "active")
    refs, blobs = list(checkpoint.segments), dict(checkpoint.blobs)
    worked = False
    last_bytes = 0

    def blocked(required):
        job["blocked_child_bytes"] = required
        retention.update(
            evidence_compaction=job,
            evidence_migration_pending=True,
            evidence_migration_last_bytes=last_bytes,
        )
        result = replace(checkpoint, retention=retention)
        commit_checkpoint(profile, result)
        return result

    def compact_one(ref):
        from ssgrep.accounting.collect.replay import compact_payload

        if not 0 <= ref.size <= MAX_ATOMIC_SEGMENT_BYTES:
            raise MigrationError(
                "evidence_migration_child_bound",
                "Legacy evidence segment exceeds the atomic migration limit.",
            )
        return write_segments(
            profile,
            (
                replace(item, payload=compact_payload(item.payload))
                for item in read_segment(profile, ref)
            ),
            origin=ref.origin,
        )

    if int(job.get("blocked_child_bytes", 0)) > byte_budget:
        return checkpoint
    job.pop("blocked_child_bytes", None)
    if job["phase"] == "active":
        cursor = int(job.get("index", 0))
        if cursor < len(refs):
            old = refs[cursor]
            if old.size > byte_budget:
                return blocked(old.size)
            compact = compact_one(old)
            refs[cursor : cursor + 1] = compact
            job["index"] = cursor + len(compact)
            job["cleanup"] = bool(job.get("cleanup")) or compact != [old]
            last_bytes, worked = old.size, True
        if int(job.get("index", 0)) >= len(refs):
            retention["evidence_format"] = COMPACT
            job.update(phase="archives", index=0)
    archives = sorted(name for name in blobs if name.startswith("identity-archive-"))
    after = str(job.get("after", ""))
    pending = [name for name in archives if name > after]
    if job["phase"] == "archives" and not worked and pending:
        name = pending[0]
        directory = storage.open_profile_dir(profile, "collector", "state", create=False)
        try:
            descriptor = storage.open_regular(directory, blobs[name])
            try:
                size = os.fstat(descriptor).st_size
            finally:
                os.close(descriptor)
        finally:
            os.close(directory)
        if size > byte_budget:
            return blocked(size)
        manifest = read_blob(profile, blobs[name])
        last_bytes = size
        if not isinstance(manifest, dict) or manifest.get("schema") != 1:
            raise MigrationError("state_corrupt", "Identity archive cannot be compacted safely.")
        old_refs = [SegmentRef.from_dict(raw) for raw in manifest.get("segments", [])]
        cursor = int(job.get("index", 0))
        if cursor < len(old_refs):
            old = old_refs[cursor]
            if old.size + size > byte_budget:
                return blocked(old.size + size)
            compact = compact_one(old)
            old_refs[cursor : cursor + 1] = compact
            job["index"] = cursor + len(compact)
            if compact != [old]:
                blobs[name] = write_blob(
                    profile, name, {**manifest, "segments": [ref.to_dict() for ref in old_refs]}
                )
                job["cleanup"] = True
            last_bytes += old.size
        if int(job.get("index", 0)) >= len(old_refs):
            job.update(after=name, index=0)
        worked = True
    done = job["phase"] == "archives" and not any(
        name > str(job.get("after", "")) for name in archives
    )
    retention["evidence_migration_last_bytes"] = last_bytes
    if done:
        retention.update(evidence_format=COMPACT, archive_evidence_format=COMPACT)
        retention.pop("evidence_compaction", None)
        retention.pop("evidence_migration_pending", None)
        if job.get("cleanup") or retention.get("evidence_cleanup_pending"):
            retention["evidence_cleanup_pending"] = True
    else:
        retention["evidence_compaction"] = job
        retention["evidence_migration_pending"] = True
    migrated = replace(checkpoint, segments=refs, blobs=blobs, retention=retention)
    commit_checkpoint(profile, migrated)
    return (
        migrate_evidence(profile, migrated, byte_budget=byte_budget)
        if done and retention.get("evidence_cleanup_pending")
        else migrated
    )
