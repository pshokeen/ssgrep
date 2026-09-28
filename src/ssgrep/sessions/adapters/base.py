"""Canonical source and record contracts shared by transcript adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ssgrep.sessions import records
from ssgrep.utilities.types import SessionFile


@dataclass(frozen=True)
class SourceFingerprint:
    """Cheap change detector persisted in the existing cursor table."""

    size: int
    mtime: float
    digest: str


@dataclass(frozen=True)
class TranscriptSource:
    """One independently indexable session exposed by a runtime adapter."""

    adapter: str
    key: str
    session: SessionFile
    fingerprint: SourceFingerprint
    cache_cwds: bool = False


@dataclass(frozen=True)
class ReadResult:
    """Canonical Claude-shaped records plus adapter parse diagnostics."""

    records: tuple[dict, ...]
    malformed_records: int = 0
    skipped_records: int = 0


class TranscriptAdapter(Protocol):
    """Discovery and normalization boundary for one coding-agent runtime."""

    name: str

    def discover(
        self, *, scope: str | None = None, no_subagents: bool = False
    ) -> list[TranscriptSource]: ...

    def read(self, source: TranscriptSource) -> ReadResult: ...

    def present(self, source: TranscriptSource) -> bool: ...


def file_fingerprint(path: Path) -> SourceFingerprint | None:
    """Fingerprint a JSONL source without parsing its full contents."""
    try:
        stat = path.stat()
        with path.open("rb") as stream:
            first_line = stream.readline()
    except OSError:
        return None
    return SourceFingerprint(
        size=stat.st_size,
        mtime=stat.st_mtime,
        digest=hashlib.sha256(first_line).hexdigest(),
    )


def jsonl_source(
    session: SessionFile,
    *,
    adapter: str,
    cache_cwds: bool = False,
) -> TranscriptSource | None:
    """Build a source descriptor for one complete-or-growing JSONL file."""
    fingerprint = file_fingerprint(session.path)
    if fingerprint is None:
        return None
    return TranscriptSource(
        adapter=adapter,
        key=str(session.path.absolute()),
        session=session,
        fingerprint=fingerprint,
        cache_cwds=cache_cwds,
    )


def read_jsonl(path: Path) -> ReadResult:
    """Read complete JSONL records, tolerating a concurrently appended tail."""
    raw: list[dict] = []
    malformed = 0
    skipped = 0
    with path.open("rb") as stream:
        for line_bytes in stream:
            if not line_bytes.endswith(b"\n"):
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > records.MAX_LINE_BYTES:
                skipped += 1
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(value, dict):
                skipped += 1
                continue
            if not records.is_known_record(value):
                skipped += 1
            raw.append(value)
    return ReadResult(tuple(raw), malformed_records=malformed, skipped_records=skipped)
