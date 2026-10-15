"""Bounded authorized native sources, separate from consumer observation retention.

Index references are locators, not cryptographic attestations. A first read records
observed-as-of digests; supplied prior digests are checked, never manufactured.
No usage is imported, no native/index files or transcript mirrors are written.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES, RawEvidence
from ssgrep.accounting.readers.base import JsonlReader
from ssgrep.accounting.readers.values import loads
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.timeline import Timeline, build_timelines
from ssgrep.accounting.valuation_inputs import AtomStatus

MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_RECORDS = 6000
SOURCE_KINDS = frozenset({"session_header", "user_turn", "tool_call", "tool_result"})


class NativeSourceStatus(StrEnum):
    OK = "ok"
    MISSING = "source_missing"
    EXPIRED = "source_expired"
    CHANGED = "source_digest_drift"
    UNAPPROVED = "source_unapproved"
    OUT_OF_SCOPE = "source_out_of_scope"
    SCOPE_UNKNOWN = "source_scope_unknown"
    IDENTITY_MISMATCH = "source_identity_mismatch"
    UNRESOLVED = "source_unresolved"
    UNSUPPORTED = "source_unsupported"
    BOUNDED = "source_read_bound"
    INCOMPLETE = "source_incomplete"
    UNREADABLE = "source_unreadable"


@dataclass(frozen=True)
class NativeSourceReference:
    ref: str
    runtime: str
    session_id: str
    path: Path
    source_digest: str = ""
    record_digests: tuple[tuple[str, str], ...] = ()
    cwd: str | None = None
    expires_at_ms: int | None = None
    observed_at_ms: int | None = None
    indexed_session_id: str | None = None  # locator ownership, not a native-origin assertion
    source_bytes: int | None = None  # immutable JSONL byte prefix, not current file size
    row_locators: tuple[str, ...] = ()  # immutable OpenCode native row set


@dataclass(frozen=True)
class NativeSourceResult:
    status: NativeSourceStatus
    line: Timeline | None = None
    bytes_read: int = 0
    source_digest: str = ""
    record_digests: tuple[tuple[str, str], ...] = ()
    observed_at_ms: int | None = None

    @property
    def reason(self) -> str:
        return "" if self.status is NativeSourceStatus.OK else self.status.value


@dataclass(frozen=True)
class NativeRecordResult:
    status: NativeSourceStatus
    evidence: RawEvidence | None = None
    bytes_read: int = 0

    @property
    def reason(self) -> str:
        return "" if self.status is NativeSourceStatus.OK else self.status.value


class SourceReadError(Exception):
    def __init__(self, status: NativeSourceStatus, bytes_read: int = 0):
        self.status = status
        self.bytes_read = bytes_read


def open_source(path: Path, approved_roots: tuple[Path, ...]) -> int:
    """No-follow descriptor traversal and nonblocking regular-file validation.

    Resolve only the explicitly approved roots (e.g. platform /tmp alias). Never
    resolve a source-provided component: linked ancestors cannot escape consent.
    """
    if not path.is_absolute() or ".." in path.parts:
        raise SourceReadError(NativeSourceStatus.UNAPPROVED)
    for root in approved_roots:
        if not root.is_absolute():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        descriptor = None
        try:
            descriptor = os.open(
                root.resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            for part in relative.parts[:-1]:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            file_fd = os.open(
                relative.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            info = os.fstat(file_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o022
            ):
                os.close(file_fd)
                raise SourceReadError(NativeSourceStatus.UNAPPROVED)
            return file_fd
        except FileNotFoundError:
            raise SourceReadError(NativeSourceStatus.MISSING) from None
        except OSError as error:
            unsafe = error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EACCES)
            raise SourceReadError(
                NativeSourceStatus.UNAPPROVED if unsafe else NativeSourceStatus.UNREADABLE
            ) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
    raise SourceReadError(NativeSourceStatus.UNAPPROVED)


@dataclass
class NativeSourceAdapter:
    references: Mapping[str, NativeSourceReference]
    readers: Mapping[str, Any]
    approved_roots: tuple[Path, ...]
    scoper: Scoper
    now_ms: int
    byte_budget: int = MAX_SOURCE_BYTES
    max_records: int = MAX_SOURCE_RECORDS
    repository: Any = None
    bytes_read: int = 0
    observed_references: dict[str, NativeSourceReference] = field(default_factory=dict)
    _snapshots: dict[tuple, tuple[Any, NativeSourceReference, NativeSourceResult]] = field(
        default_factory=dict
    )

    def _snapshot_key(self, reference):
        authority = repr((self.scoper.policy, self.scoper.ownership))
        return (
            reference.runtime,
            str(reference.path),
            reference.session_id,
            reference.cwd,
            reference.source_digest,
            reference.source_bytes,
            tuple(sorted(reference.record_digests)),
            tuple(sorted(reference.row_locators)),
            reference.indexed_session_id,
            reference.observed_at_ms is None,
            authority,
            tuple(map(str, self.approved_roots)),
        )

    @staticmethod
    def _snapshot_stamp(reference):
        paths = [reference.path]
        if reference.runtime == "opencode":
            paths.append(Path(str(reference.path) + "-wal"))
        stamps = []
        for path in paths:
            try:
                info = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                stamps.append(None)
                continue
            stamps.append(
                (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_mode,
                    info.st_uid,
                )
            )
        return tuple(stamps)

    def read_record(
        self,
        reference: NativeSourceReference,
        origin_identity: str,
        digest: str,
        *,
        locator: str = "",
    ) -> NativeRecordResult:
        """Direct known record lookup for compact replay; no full-session scan."""
        from ssgrep.accounting.native_source_record import read_record

        return read_record(self, reference, origin_identity, digest, locator)

    def resolve(self, ref: str) -> NativeSourceResult:
        reference = self.observed_references.get(ref) or self.references.get(ref)
        if reference is None:
            from ssgrep.accounting.native_source_index import index_reference

            reference, status = index_reference(ref, self.repository)
            if reference is None:
                return NativeSourceResult(status)
        if reference.expires_at_ms is not None and reference.expires_at_ms <= self.now_ms:
            return NativeSourceResult(NativeSourceStatus.EXPIRED)
        if reference.cwd is not None:
            scope, why = self.scoper.source_status(reference.cwd)
            if scope is not AtomStatus.OK:
                return NativeSourceResult(NativeSourceStatus(why))
        reader = self.readers.get(reference.runtime)
        if reader is None:
            return NativeSourceResult(NativeSourceStatus.UNSUPPORTED)
        snapshot_key = self._snapshot_key(reference)
        try:
            stamp = self._snapshot_stamp(reference)
        except OSError:
            return NativeSourceResult(NativeSourceStatus.UNREADABLE)
        cached = self._snapshots.get(snapshot_key)
        if cached is not None and cached[0] == stamp:
            self.observed_references[ref] = replace(cached[1], ref=ref)
            return replace(cached[2], bytes_read=0)
        read_stamp = stamp
        budget = min(MAX_SOURCE_BYTES, self.byte_budget) - self.bytes_read
        if budget <= 0:
            return NativeSourceResult(NativeSourceStatus.BOUNDED)
        count = 0
        started_bytes = self.bytes_read
        try:
            descriptor = open_source(reference.path, self.approved_roots)
            try:
                if isinstance(reader, JsonlReader):
                    evidence, digest, count = self._jsonl(reader, reference, descriptor, budget)
                elif reference.runtime == "opencode":
                    from ssgrep.accounting.native_source_db import read_session

                    evidence, digest, count = read_session(
                        reader, reference, descriptor, budget, self.max_records
                    )
                else:
                    raise SourceReadError(NativeSourceStatus.UNSUPPORTED)
            finally:
                os.close(descriptor)
        except SourceReadError as error:
            self.bytes_read += error.bytes_read
            return NativeSourceResult(error.status, bytes_read=self.bytes_read - started_bytes)
        except OSError:
            return NativeSourceResult(
                NativeSourceStatus.UNREADABLE, bytes_read=self.bytes_read - started_bytes
            )
        if reference.runtime == "opencode":
            self.bytes_read += count
        if reference.source_digest and reference.source_digest != digest:
            return NativeSourceResult(NativeSourceStatus.CHANGED, bytes_read=count)
        lines = build_timelines(evidence)
        key = (reference.runtime, reference.session_id)
        if (
            reference.runtime == "claude"
            and reference.indexed_session_id is not None
            and reference.observed_at_ms is None
            and len(lines.sessions) == 1
        ):
            key = next(iter(lines.sessions))
        line = lines.sessions.get(key)
        if line is None or set(lines.sessions) != {key}:
            return NativeSourceResult(NativeSourceStatus.IDENTITY_MISMATCH, bytes_read=count)
        if lines.conflicted:
            return NativeSourceResult(NativeSourceStatus.CHANGED, bytes_read=count)
        if lines.ownership_unknown:
            return NativeSourceResult(NativeSourceStatus.IDENTITY_MISMATCH, bytes_read=count)
        if any(
            r.ev.origin_status in {"missing_entry_id", "missing_session_id"} for r in line.records
        ):
            return NativeSourceResult(NativeSourceStatus.IDENTITY_MISMATCH, bytes_read=count)
        held = dict(reference.record_digests)
        actual = {r.ev.native_identity: r.ev.digest for r in line.records}
        if any(actual.get(record_id) != digest for record_id, digest in held.items()):
            return NativeSourceResult(NativeSourceStatus.CHANGED, bytes_read=count)
        for record in line.records:
            scope, why = self.scoper.source_status(line.cwd_at(record.seq))
            if scope is not AtomStatus.OK and record.kind in {"tool_call", "tool_result"}:
                return NativeSourceResult(NativeSourceStatus(why), bytes_read=count)
        self.observed_references[ref] = replace(
            reference,
            session_id=key[1],
            source_digest=digest,
            record_digests=tuple(sorted(actual.items())),
            observed_at_ms=self.now_ms,
            source_bytes=count if isinstance(reader, JsonlReader) else None,
            row_locators=tuple(sorted({r.ev.locator for r in line.records}))
            if reference.runtime == "opencode"
            else (),
        )
        physical_bytes = self.bytes_read - started_bytes
        result = NativeSourceResult(
            NativeSourceStatus.OK,
            line,
            physical_bytes,
            digest,
            tuple(sorted(actual.items())),
            self.now_ms,
        )
        observed = self.observed_references[ref]
        try:
            stamp = self._snapshot_stamp(observed)
        except OSError:
            return result
        if stamp != read_stamp:
            return result
        self._snapshots[snapshot_key] = (stamp, observed, result)
        self._snapshots[self._snapshot_key(observed)] = (stamp, observed, result)
        return result

    def _jsonl(self, reader, reference, descriptor, budget):
        from ssgrep.accounting.native_source_proof import ReadProof

        proof = ReadProof()
        started_bytes = self.bytes_read

        def charge(size):
            self.bytes_read += size
            if self.bytes_read - started_bytes > budget:
                raise SourceReadError(NativeSourceStatus.BOUNDED)

        info = os.fstat(descriptor)
        end = reference.source_bytes if reference.source_bytes is not None else info.st_size
        if end < 0 or info.st_size < end:
            raise SourceReadError(NativeSourceStatus.CHANGED)
        if end > budget:
            raise SourceReadError(NativeSourceStatus.BOUNDED)
        evidence = []
        digest = hashlib.sha256()
        context: dict[str, Any] = {}
        offset = records = 0
        source = reader.source_for(reference.path)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while offset < end:
                remaining = min(budget, end) - offset
                data = handle.readline(min(MAX_RAW_RECORD_BYTES + 1, remaining))
                if not data:
                    break
                charge(len(data))
                proof.add(offset, data)
                if len(data) > remaining or len(data) > MAX_RAW_RECORD_BYTES:
                    raise SourceReadError(NativeSourceStatus.BOUNDED)
                if not data.endswith(b"\n"):
                    raise SourceReadError(NativeSourceStatus.INCOMPLETE)
                records += 1
                if records > min(MAX_SOURCE_RECORDS, self.max_records):
                    raise SourceReadError(NativeSourceStatus.BOUNDED)
                digest.update(data)
                try:
                    decoded = reader.decode(source, loads(data), context, f"offset:{offset}")
                except (ValueError, KeyError, TypeError, RecursionError):
                    raise SourceReadError(NativeSourceStatus.INCOMPLETE) from None
                if decoded:
                    evidence.extend(item for item in decoded if item.kind in SOURCE_KINDS)
                    if len(evidence) > min(MAX_SOURCE_RECORDS, self.max_records):
                        raise SourceReadError(NativeSourceStatus.BOUNDED)
                offset += len(data)
        proof.verify(descriptor, info, charge)
        observed_digest = digest.hexdigest()
        # Charged during scanning so rejected/malformed reads also consume budget.
        return reader.finish_cycle(evidence), observed_digest, offset
