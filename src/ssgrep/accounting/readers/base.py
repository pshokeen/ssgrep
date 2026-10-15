"""Shared reader machinery: checkpoints, reset detection, bounded reads, dedupe.

A reader turns one native source (a JSONL file or an OpenCode database) into
``RawEvidence``. Everything here is read-only and treats transcript content as
untrusted data: nothing decoded is ever executed, fetched, or used as a path
outside the configured runtime roots.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.identity_claims import OwnershipClaims
from ssgrep.accounting.readers.source_io import SourceFile, open_source
from ssgrep.accounting.readers.stream import JsonlScanner
from ssgrep.accounting.readers.streaming import (
    DOWN,
    EQUAL,
    UP,
    StreamState,
    compare,
    is_stream,
    same_context,
)
from ssgrep.accounting.readers.values import loads

WINDOW = 4096


@dataclass(frozen=True)
class Source:
    """One native source; identity is its path (never followed through symlinks)."""

    runtime: str
    path: Path
    role: str = "main"

    @property
    def source_id(self) -> str:
        return f"{self.runtime}:{self.path}"


@dataclass(frozen=True)
class Gap:
    """Evidence deliberately not read; the denominator of coverage claims."""

    source_id: str
    reason: str
    size: int
    locator: str
    offset: int | None = None


@dataclass
class SourceState:
    """Checkpoint after a read; ``offset`` always sits just past a newline."""

    offset: int = 0
    dev: int = 0
    inode: int = 0
    head_digest: str = ""
    tail_digest: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "dev": self.dev,
            "inode": self.inode,
            "head_digest": self.head_digest,
            "tail_digest": self.tail_digest,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceState:
        return cls(
            offset=int(data.get("offset", 0)),
            dev=int(data.get("dev", 0)),
            inode=int(data.get("inode", 0)),
            head_digest=str(data.get("head_digest", "")),
            tail_digest=str(data.get("tail_digest", "")),
            extra=dict(data.get("extra", {})),
        )


@dataclass
class ReadResult:
    evidence: list[RawEvidence] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    state: SourceState = field(default_factory=SourceState)
    pending_bytes: int = 0
    pending_records: int = 0
    reset: str | None = None
    more: bool = False
    unrecognized: dict[str, int] = field(default_factory=dict)
    bytes_read: int = 0

    def evidence_ids(self) -> list[str]:
        return [item.record_id for item in self.evidence]


def _digest_window(handle: SourceFile, start: int, end: int) -> str:
    if end <= start:
        return ""
    return hashlib.sha256(handle.read(start, end - start)).hexdigest()


def _discarded(extra: Mapping[str, object]) -> int:
    value = extra.get("jsonl_discarded", 0)
    return value if isinstance(value, int) and value > 0 else 0


def detect_reset(handle: SourceFile, state: SourceState) -> str | None:
    """Verify the committed prefix and any in-progress oversized discard."""
    observed = state.offset + _discarded(state.extra)
    if observed <= 0:
        return None
    info = handle.stat()
    if info.st_size < observed:
        return "truncated"
    if (info.st_dev, info.st_ino) != (state.dev, state.inode):
        return "rotated"
    head = _digest_window(handle, 0, min(observed, WINDOW))
    tail = _digest_window(handle, max(0, observed - WINDOW), observed)
    if head != state.head_digest or tail != state.tail_digest:
        return "mutated"
    return None


def checkpoint(handle: SourceFile, offset: int, extra: dict[str, Any]) -> SourceState:
    info = handle.stat()
    observed = offset + _discarded(extra)
    return SourceState(
        offset=offset,
        dev=info.st_dev,
        inode=info.st_ino,
        head_digest=_digest_window(handle, 0, min(observed, WINDOW)),
        tail_digest=_digest_window(handle, max(0, observed - WINDOW), observed),
        extra=extra,
    )


class JsonlReader(ABC):
    """Base for newline-delimited runtimes; subclasses decode one parsed record."""

    runtime: ClassVar[str]
    skip_dirs: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.env = env

    # ---- discovery -------------------------------------------------------
    @abstractmethod
    def roots(self) -> list[Path]: ...

    def role_for(self, path: Path) -> str:
        return "main"

    def source_for(self, path: Path) -> Source:
        return Source(self.runtime, path, self.role_for(path))

    def discover(self) -> list[Source]:
        """Every ``*.jsonl`` beneath the roots; symlinks are listed, never followed."""
        found: dict[Path, Source] = {}
        for root in self.roots():
            if not root.is_dir():
                continue
            for base, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in self.skip_dirs)
                for name in sorted(files):
                    path = Path(base, name)
                    if name.endswith(".jsonl") and not path.is_symlink() and self.accepts(path):
                        found.setdefault(path, self.source_for(path))
        return list(found.values())

    def accepts(self, path: Path) -> bool:
        return True

    def finish_cycle(self, evidence: list[RawEvidence]) -> list[RawEvidence]:
        """Hook: resolve records that only make sense together (streamed updates)."""
        return evidence

    # ---- decoding --------------------------------------------------------
    @abstractmethod
    def decode(
        self, source: Source, obj: Any, ctx: dict[str, Any], locator: str
    ) -> list[RawEvidence] | None:
        """Evidence for one record; None when the record format is not recognized."""
        ...

    @staticmethod
    def type_of(obj: Any) -> str:
        if isinstance(obj, dict) and isinstance(obj.get("type"), str):
            return obj["type"][:64]
        return "<non-object>" if not isinstance(obj, dict) else "<untyped>"

    def baseline_context(self, source: Source, offset: int, handle: SourceFile) -> dict[str, Any]:
        """Bounded metadata reconstruction; never returns skipped usage evidence."""
        return {}

    # ---- reading ---------------------------------------------------------
    def read(
        self, source: Source, state: SourceState | None = None, budget: int | None = None
    ) -> ReadResult:
        with open_source(source.path) as handle:
            return self._read(source, state or SourceState(), budget, handle)

    def _read(
        self, source: Source, state: SourceState, budget: int | None, handle: SourceFile
    ) -> ReadResult:
        reset = detect_reset(handle, state)
        if reset:
            state = SourceState()
        ctx = dict(state.extra)
        if state.offset and not ctx:
            ctx = self.baseline_context(source, state.offset, handle)
        result = ReadResult(reset=reset)
        scanner = JsonlScanner(
            source.path, state.offset, budget=budget, source=handle, discarded=_discarded(ctx)
        )
        for line in scanner:
            locator = f"offset:{line.offset}"
            if line.data is None:
                result.gaps.append(
                    Gap(source.source_id, "oversized_record", line.size, locator, line.offset)
                )
                continue
            try:
                obj = loads(line.data)
            except ValueError:
                result.gaps.append(
                    Gap(source.source_id, "malformed_json", line.size, locator, line.offset)
                )
                continue
            decoded = self.decode(source, obj, ctx, locator)
            if decoded is None:
                kind = self.type_of(obj)
                result.unrecognized[kind] = result.unrecognized.get(kind, 0) + 1
            else:
                result.evidence.extend(decoded)
        result.evidence = self.finish_cycle(result.evidence)
        result.more = scanner.more
        result.pending_bytes = 0 if scanner.more else scanner.pending
        if scanner.discarded:
            ctx["jsonl_discarded"] = scanner.discarded
        else:
            ctx.pop("jsonl_discarded", None)
        result.state = checkpoint(handle, scanner.committed, ctx)
        # Charge newly examined transcript bytes, not verification windows or
        # prefetched records left for the next quantum. Discard progress spends
        # budget even though its verified newline checkpoint has not advanced.
        examined_end = scanner.committed + max(scanner.discarded, scanner.pending)
        result.bytes_read = examined_end - state.offset - _discarded(state.extra)
        return result


@dataclass(frozen=True)
class Conflict:
    """Same native identity, different content: reported, never counted twice."""

    runtime: str
    session_id: str
    record_id: str
    kind: str
    digests: tuple[str, ...]
    kept_digest: str
    source_ids: tuple[str, ...]


class EvidenceLedger:
    """Counts each native event once across copied/rotated sources.

    Keys are 128-bit hashes of the native identity, so a multi-gigabyte census keeps
    ~100 bytes per event; full identity and variants are retained only for conflicts.

    Streamed usage (``streaming_updates`` in the payload) resolves differently: a
    greater later update *supersedes*, an older prefix copy is *stale*, and only a
    counter decrease in one source's stream order or mixed movement is a conflict.
    Read the resolved value from ``final_usage()``, never by summing admitted rows.
    """

    def __init__(self) -> None:
        self._first: dict[bytes, str] = {}
        self._conflicting: dict[bytes, tuple[str, str, str, str, dict[str, set[str]]]] = {}
        self._origin: dict[bytes, str] = {}
        self._streams: dict[bytes, StreamState] = {}
        self._usage_revisions: dict[bytes, RawEvidence] = {}
        self._ownership = OwnershipClaims()
        self.duplicates = 0
        self.duplicates_by_runtime: dict[str, int] = {}
        self.superseded = 0
        self.stale_prefixes = 0

    @staticmethod
    def _key(evidence: RawEvidence) -> bytes:
        identity = "\0".join(evidence.native_key)
        return hashlib.blake2b(identity.encode(), digest_size=16).digest()

    def _duplicate(self, evidence: RawEvidence) -> bool:
        self.duplicates += 1
        runtime = evidence.runtime
        self.duplicates_by_runtime[runtime] = self.duplicates_by_runtime.get(runtime, 0) + 1
        return False

    def _conflict(self, key: bytes, evidence: RawEvidence, *known: RawEvidence) -> bool:
        entry = self._conflicting.get(key)
        if entry is None:
            names = (*evidence.native_key, evidence.kind)
            entry = (*names, {self._first[key]: {self._origin[key]}})
            self._conflicting[key] = entry
        for item in (*known, evidence):
            entry[4].setdefault(item.digest, set()).add(item.source_id)
        return False

    def admit(self, evidence: RawEvidence) -> bool:
        """True exactly once per identity; a changed variant is a conflict, not usage."""
        key = self._key(evidence)
        self._usage_revisions.pop(key, None)
        self._ownership.note(evidence, key)
        conflict = self._ownership.conflict_evidence(evidence)
        if conflict is not None:
            target = self._key(conflict)
            if target in self._first and self._first[target] != conflict.digest:
                self._conflict(target, conflict)
        first = self._first.get(key)
        if first is None:
            self._first[key] = evidence.digest
            self._origin[key] = evidence.source_id
            if is_stream(evidence):
                self._streams[key] = StreamState(evidence)
            for conflict in self._ownership.conflicts.get(key, {}).values():
                if conflict.digest != evidence.digest:
                    self._conflict(key, conflict)
            return True
        if is_stream(evidence) and key in self._streams:
            return self._admit_stream(key, self._streams[key], evidence)
        if evidence.digest == first:
            return self._duplicate(evidence)
        return self._conflict(key, evidence)

    def ownership_claim(self, evidence: RawEvidence) -> RawEvidence | None:
        """One compact newly observed copy owner, separate from usage admission."""
        key = self._key(evidence)
        return self._ownership.take(evidence, key, conflict=key in self._conflicting)

    def usage_revision(self, evidence: RawEvidence) -> RawEvidence | None:
        """Greater provider counters replace the prior sample, never add an execution."""
        key = self._key(evidence)
        revision = self._usage_revisions.pop(key, None)
        return revision if key not in self._conflicting else None

    def _admit_stream(self, key: bytes, state: StreamState, evidence: RawEvidence) -> bool:
        best = state.best
        skip = {"_evidence_format"}
        if evidence.origin_status == "provider_message_id":
            skip.update({"parent_session", "cwd"})
        before = {k: v for k, v in best.payload.items() if k not in skip}
        after = {k: v for k, v in evidence.payload.items() if k not in skip}
        if not same_context(before, after):
            return self._conflict(key, evidence, best)
        tokens = evidence.payload["tokens"]
        last = state.last.get(evidence.source_id)
        # within one source's stream order a counter may only grow
        reference = last if last is not None else best.payload["tokens"]
        relation = compare(reference, tokens)
        if last is not None and relation not in {EQUAL, UP}:
            return self._conflict(key, evidence, best)
        if last is None and relation not in {EQUAL, UP, DOWN}:
            return self._conflict(key, evidence, best)
        vs_best = compare(best.payload["tokens"], tokens)
        state.note(evidence)
        if vs_best == UP:
            state.best = evidence
            self._usage_revisions[key] = evidence
            self.superseded += 1
            return False
        if vs_best == DOWN:
            self.stale_prefixes += 1
            return False
        return self._duplicate(evidence)

    def final_usage(self) -> Iterator[tuple[RawEvidence, int]]:
        """Resolved streamed usage: the greatest update per message and its update count."""
        for state in self._streams.values():
            yield state.best, state.total_updates

    @property
    def conflict_count(self) -> int:
        return len(self._conflicting)

    def conflicts(self) -> list[Conflict]:
        found = []
        for key, (runtime, session, record, kind, variants) in sorted(
            self._conflicting.items(), key=lambda item: item[1][:3]
        ):
            sources = sorted({source for group in variants.values() for source in group})
            found.append(
                Conflict(
                    runtime,
                    session,
                    record,
                    kind,
                    tuple(sorted(variants)),
                    self._first[key],
                    tuple(sources),
                )
            )
        return found

    def drain(self, items: list[RawEvidence]) -> Iterator[RawEvidence]:
        return (item for item in items if self.admit(item))
