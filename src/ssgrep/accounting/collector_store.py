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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import SCHEMA_VERSION, RawEvidence
from ssgrep.accounting.evidence_codec import evidence_from_line, evidence_line

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
STRUCTURAL_KINDS = frozenset({"user_turn", "session_header", "lineage"})


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
    except (ValueError, OSError) as error:
        raise CollectError(
            "state_corrupt", f"Collector checkpoint is unreadable: {error}"
        ) from error
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
    from ssgrep.accounting.collector_segment_index import IndexWriter

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
    from ssgrep.accounting.collector_segment_index import iter_selected

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
    from ssgrep.accounting.evidence_references import referenced_files

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
