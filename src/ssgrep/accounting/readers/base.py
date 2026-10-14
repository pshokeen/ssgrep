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


def _digest_window(path: Path, start: int, end: int) -> str:
    if end <= start:
        return ""
    with path.open("rb") as handle:
        handle.seek(start)
        return hashlib.sha256(handle.read(end - start)).hexdigest()


def detect_reset(path: Path, state: SourceState) -> str | None:
    """Why the committed prefix can no longer be trusted, or None when it is intact."""
    if state.offset <= 0:
        return None
    info = path.stat()
    if info.st_size < state.offset:
        return "truncated"
    if (info.st_dev, info.st_ino) != (state.dev, state.inode):
        return "rotated"
    head = _digest_window(path, 0, min(state.offset, WINDOW))
    tail = _digest_window(path, max(0, state.offset - WINDOW), state.offset)
    if head != state.head_digest or tail != state.tail_digest:
        return "mutated"
    return None


def checkpoint(path: Path, offset: int, extra: dict[str, Any]) -> SourceState:
    info = path.stat()
    return SourceState(
        offset=offset,
        dev=info.st_dev,
        inode=info.st_ino,
        head_digest=_digest_window(path, 0, min(offset, WINDOW)),
        tail_digest=_digest_window(path, max(0, offset - WINDOW), offset),
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

    # ---- reading ---------------------------------------------------------
    def read(
        self, source: Source, state: SourceState | None = None, budget: int | None = None
    ) -> ReadResult:
        state = state or SourceState()
        path = source.path
        reset = detect_reset(path, state)
        if reset:
            state = SourceState()
        ctx = dict(state.extra)
        result = ReadResult(reset=reset)
        scanner = JsonlScanner(path, state.offset, budget=budget)
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
        result.bytes_read = scanner.committed - state.offset
        result.state = checkpoint(path, scanner.committed, ctx)
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
        self.duplicates = 0
        self.duplicates_by_runtime: dict[str, int] = {}
        self.superseded = 0
        self.stale_prefixes = 0

    @staticmethod
    def _key(evidence: RawEvidence) -> bytes:
        identity = f"{evidence.runtime}\0{evidence.session_id}\0{evidence.record_id}"
        return hashlib.blake2b(identity.encode(), digest_size=16).digest()

    def _duplicate(self, evidence: RawEvidence) -> bool:
        self.duplicates += 1
        runtime = evidence.runtime
        self.duplicates_by_runtime[runtime] = self.duplicates_by_runtime.get(runtime, 0) + 1
        return False

    def _conflict(self, key: bytes, evidence: RawEvidence, *known: RawEvidence) -> bool:
        entry = self._conflicting.get(key)
        if entry is None:
            names = (evidence.runtime, evidence.session_id, evidence.record_id, evidence.kind)
            entry = (*names, {self._first[key]: {self._origin[key]}})
            self._conflicting[key] = entry
        for item in (*known, evidence):
            entry[4].setdefault(item.digest, set()).add(item.source_id)
        return False

    def admit(self, evidence: RawEvidence) -> bool:
        """True exactly once per identity; a changed variant is a conflict, not usage."""
        key = self._key(evidence)
        first = self._first.get(key)
        if first is None:
            self._first[key] = evidence.digest
            self._origin[key] = evidence.source_id
            if is_stream(evidence):
                self._streams[key] = StreamState(evidence)
            return True
        if is_stream(evidence) and key in self._streams:
            return self._admit_stream(key, self._streams[key], evidence)
        if evidence.digest == first:
            return self._duplicate(evidence)
        return self._conflict(key, evidence)

    def _admit_stream(self, key: bytes, state: StreamState, evidence: RawEvidence) -> bool:
        best = state.best
        if not same_context(best.payload, evidence.payload):
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
