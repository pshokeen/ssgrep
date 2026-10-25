"""Recovery reader: admits only complete, valid, unique events.

Segments are immutable append-only files. A line counts only when it is
newline-terminated, at most ``MAX_EVENT_BYTES``, and strictly valid. Anything
else is *damage*:

* an unterminated final fragment is an **incomplete tail** (a short write or a
  write still in flight): excluded from completed counts and reported;
* a terminated line that fails validation, or exceeds the bound, is
  **quarantined**: a metadata record (segment, offset, length, digest, reason
  -- never the raw bytes) is written once under ``journal-quarantine/`` so the
  damage is reported identically after every restart. Checkpoint-authorized
  retention may later replace a segment with retained-only immutable evidence.

Events dedupe by ``(execution_id, phase)``. Identical replays collapse to one;
differing variants of the same identity are reported as conflicts and the
variant with the smallest content digest is accepted, so the result does not
depend on file order.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from ssgrep.accounting.capture.journal import SEGMENT_PREFIX, SEGMENT_SUFFIX
from ssgrep.accounting.capture.journal_codec import MalformedEvent, parse_event
from ssgrep.accounting.ledger.contracts import MAX_EVENT_BYTES, ExecutionEvent, canonical_bytes
from ssgrep.accounting.privacy import storage

QUARANTINE_DIR = "journal-quarantine"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Damage:
    segment: str
    offset: int
    length: int
    digest: str
    reason: str
    kind: str  # "quarantined" | "incomplete_tail"


@dataclass(frozen=True)
class SegmentProof:
    """Exact immutable prefix observed by the accepted collection cycle."""

    name: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int = 0

    @classmethod
    def capture(cls, name: str, fd: int) -> SegmentProof:
        info = os.fstat(fd)
        return cls(name, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JournalRead:
    events: list[ExecutionEvent] = field(default_factory=list)
    damage: list[Damage] = field(default_factory=list)
    duplicates: int = 0
    conflicts: list[tuple[str, str]] = field(default_factory=list)
    segments: int = 0
    proofs: list[SegmentProof] = field(default_factory=list)
    retired: dict[str, str] = field(default_factory=dict)
    retired_events: int = 0
    segment_events: dict[str, list[ExecutionEvent]] = field(default_factory=dict)
    retention_pending: set[str] = field(default_factory=set)
    signature: str | None = None
    cache_blobs: dict[str, str] = field(default_factory=dict)
    cache_retention_tag: str | None = None
    cache_dirty: set[str] = field(default_factory=set)
    bytes_read: int = 0

    @property
    def quarantined(self) -> list[Damage]:
        return [d for d in self.damage if d.kind == "quarantined"]

    @property
    def incomplete_tails(self) -> list[Damage]:
        return [d for d in self.damage if d.kind == "incomplete_tail"]


def _lines(fd: int, size: int | None = None) -> Iterator[tuple[int, bytes | None, bool, int]]:
    """Yield ``(offset, line, terminated, length)``; ``line`` is None when oversize.

    Oversize lines are skipped incrementally: their bytes are counted, never
    retained, so a hostile or corrupted segment cannot force unbounded memory.
    """
    pending = bytearray()
    start = 0
    oversize = False
    consumed = 0
    while True:
        data = os.read(fd, _CHUNK if size is None else min(_CHUNK, size))
        if size is not None:
            size -= len(data)
        if not data:
            break
        while data:
            newline = data.find(b"\n")
            if newline < 0:
                if oversize:
                    consumed += len(data)
                else:
                    pending += data
                    if len(pending) >= MAX_EVENT_BYTES:
                        oversize, consumed = True, len(pending)
                        pending.clear()
                break
            head, data = data[:newline], data[newline + 1 :]
            if oversize:
                consumed += len(head) + 1
                yield start, None, True, consumed
                start += consumed
                oversize, consumed = False, 0
                continue
            pending += head
            length = len(pending) + 1
            yield start, (None if length > MAX_EVENT_BYTES else bytes(pending)), True, length
            start += length
            pending.clear()
    if oversize:
        yield start, None, False, consumed
    elif pending:
        yield start, bytes(pending), False, len(pending)


def _quarantine(directory: int | None, damage: Damage) -> None:
    if directory is None:
        return
    name = hashlib.sha256(f"{damage.segment}:{damage.offset}:{damage.digest}".encode()).hexdigest()
    document = {
        "segment": damage.segment,
        "offset": damage.offset,
        "length": damage.length,
        "digest": damage.digest,
        "reason": damage.reason,
    }
    storage.write_once(directory, f"{name}.json", json.dumps(document, sort_keys=True).encode())


def _scan_segment(
    name: str,
    fd: int,
    quarantine: int | None,
    result: JournalRead,
    size: int,
    *,
    start_offset: int = 0,
) -> None:
    for offset, line, terminated, length in _lines(fd, size):
        offset += start_offset
        result.bytes_read += length
        digest = "" if line is None else hashlib.sha256(line).hexdigest()
        if not terminated:
            damage = Damage(
                name,
                offset,
                length,
                digest,
                "unterminated final record (short write or write in flight)",
                "incomplete_tail",
            )
        elif line is None:
            damage = Damage(
                name, offset, length, digest, "line exceeds the event bound", "quarantined"
            )
        else:
            try:
                event = parse_event(line)
                result.events.append(event)
                result.segment_events.setdefault(name, []).append(event)
                continue
            except MalformedEvent as error:
                damage = Damage(name, offset, length, digest, str(error), "quarantined")
        result.damage.append(damage)
        if damage.kind == "quarantined":
            _quarantine(quarantine, damage)


_PHASE_ORDER = {"started": 0, "completed": 1, "rendered": 2}


def _dedupe(result: JournalRead) -> None:
    """Collapse by ``(execution_id, phase)``; the outcome never depends on file order."""
    variants: dict[tuple[str, str], dict[str, ExecutionEvent]] = {}
    for event in result.events:
        key = (event.execution_id, str(event.phase))
        digest = hashlib.sha256(canonical_bytes(event)).hexdigest()
        held = variants.setdefault(key, {})
        if digest in held:
            result.duplicates += 1
        else:
            held[digest] = event
    accepted: list[ExecutionEvent] = []
    for key in sorted(variants, key=lambda k: (k[0], _PHASE_ORDER.get(k[1], 9))):
        held = variants[key]
        if len(held) > 1:
            result.conflicts.append(key)
        accepted.append(held[min(held)])  # smallest content digest wins, deterministically
    result.events = accepted


def read_journal(
    profile: str,
    *,
    quarantine: bool = True,
    previous: dict[str, dict] | None = None,
    retention: dict | None = None,
) -> JournalRead:
    """Reduce every segment of ``profile`` to accepted events plus a damage report.

    Missing storage is an empty read. ``quarantine`` writes damage metadata
    (idempotently); pass False for a strictly read-only view.
    """
    result = JournalRead()
    try:
        directory = storage.open_profile_dir(profile, "journal", create=False)
    except FileNotFoundError:
        return result
    quarantine_fd: int | None = None
    try:
        if quarantine:
            quarantine_fd = storage.open_profile_dir(profile, QUARANTINE_DIR)
        for name in storage.list_names(directory):
            if not (name.startswith(SEGMENT_PREFIX) and name.endswith(SEGMENT_SUFFIX)):
                continue
            fd = storage.open_regular(directory, name)
            try:
                proof = SegmentProof.capture(name, fd)
                result.proofs.append(proof)
                result.segments += 1
                from ssgrep.accounting.collect.read import reuse_segment

                cached = (previous or {}).get(name)
                start = reuse_segment(result, proof, cached)
                if start is not None:
                    os.lseek(fd, start, os.SEEK_SET)
                    _scan_segment(
                        name, fd, quarantine_fd, result, proof.size - start, start_offset=start
                    )
                    result.cache_dirty.add(name)
            finally:
                os.close(fd)
    finally:
        os.close(directory)
        if quarantine_fd is not None:
            os.close(quarantine_fd)
    from ssgrep.accounting.capture.journal_retention import (
        apply_journal_retention,
        load_journal_retention,
    )

    apply_journal_retention(
        result, retention if retention is not None else load_journal_retention(profile)
    )
    _dedupe(result)
    return result
