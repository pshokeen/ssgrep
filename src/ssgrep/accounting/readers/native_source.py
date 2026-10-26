"""Bounded authorized native sources, separate from consumer observation retention.

Index references are locators, not cryptographic attestations. A first read records
observed-as-of digests; supplied prior digests are checked, never manufactured.
No usage is imported, no native/index files or transcript mirrors are written.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES, RawEvidence
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus
from ssgrep.accounting.racy import racy
from ssgrep.accounting.readers.base import JsonlReader, ReadResult, loads
from ssgrep.accounting.readers.opencode_storage import (
    SnapshotChanged,
    SnapshotUnavailable,
    named_row,
)
from ssgrep.accounting.readers.timeline import Timeline, build_timelines

if TYPE_CHECKING:
    from ssgrep.accounting.views.filter import Scoper

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

    Approved roots are canonical bindings, not aliases to resolve on every read.
    No source or approved-root component may redirect consent through a symlink.
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
                path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            for part in path.parts[1:-1]:
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
    def _snapshot_paths(reference):
        if reference.runtime == "opencode":
            return [reference.path, Path(str(reference.path) + "-wal")]
        return [reference.path]

    @classmethod
    def _snapshot_stamp(cls, reference):
        stamps = []
        for path in cls._snapshot_paths(reference):
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

    def _snapshot_digest(self, reference, stamp):
        """Racy-clean: bytes that must back a stamp too fresh to prove the source unchanged."""
        digests = []
        for path, row in zip(self._snapshot_paths(reference), stamp, strict=True):
            if row is None:
                digests.append(None)
                continue
            try:
                descriptor = open_source(path, self.approved_roots)
            except SourceReadError:
                return None
            try:
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, 1 << 20):
                    digest.update(chunk)
            except OSError:
                return None
            finally:
                os.close(descriptor)
            digests.append(digest.hexdigest())
        return tuple(digests)

    def read_record(
        self,
        reference: NativeSourceReference,
        origin_identity: str,
        digest: str,
        *,
        locator: str = "",
    ) -> NativeRecordResult:
        """Direct known record lookup for compact replay; no full-session scan."""
        return read_record(self, reference, origin_identity, digest, locator)

    def resolve(self, ref: str) -> NativeSourceResult:
        reference = self.observed_references.get(ref) or self.references.get(ref)
        if reference is None:
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
        observed_ns = time.time_ns()
        try:
            stamp = self._snapshot_stamp(reference)
        except OSError:
            return NativeSourceResult(NativeSourceStatus.UNREADABLE)
        cached = self._snapshots.get(snapshot_key)
        if (
            cached is not None
            and cached[0][0] == stamp
            and (not cached[0][1] or cached[0][1] == self._snapshot_digest(reference, stamp))
        ):
            self.observed_references[ref] = replace(cached[1], ref=ref)
            return replace(cached[2], bytes_read=0)
        read_stamp = stamp
        budget = min(MAX_SOURCE_BYTES, self.byte_budget) - self.bytes_read
        if budget <= 0:
            return NativeSourceResult(NativeSourceStatus.BOUNDED)
        # A racy stamp is cached only with the bytes taken before this read (() = not racy).
        racy_stamp = racy(observed_ns, *(n for row in stamp if row for n in row[3:5]))
        guard = self._snapshot_digest(reference, stamp) if racy_stamp else ()
        count = 0
        started_bytes = self.bytes_read
        try:
            descriptor = open_source(reference.path, self.approved_roots)
            try:
                if isinstance(reader, JsonlReader):
                    evidence, digest, count = self._jsonl(reader, reference, descriptor, budget)
                elif reference.runtime == "opencode":
                    from ssgrep.accounting.readers.opencode_storage import read_session

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
        if stamp != read_stamp or guard is None:
            return result
        self._snapshots[snapshot_key] = ((stamp, guard), observed, result)
        self._snapshots[self._snapshot_key(observed)] = ((stamp, guard), observed, result)
        return result

    def _jsonl(self, reader, reference, descriptor, budget):
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


class ReadProof:
    """Hashes only bytes actually read for native identity, context and payload.

    Changed descriptor metadata requires re-reading those ranges under the same
    budget. Size growth alone is NOT evidence that the prefix remained intact.
    Nothing here retains plaintext or reads unselected source tails.
    """

    def __init__(self):
        self.ranges = []

    def add(self, offset, data):
        if not data:
            return
        if self.ranges:
            start, end, digest = self.ranges[-1]
            if start <= offset and offset + len(data) <= end:
                return
            if offset == end:
                digest.update(data)
                self.ranges[-1] = (start, offset + len(data), digest)
                return
        self.ranges.append((offset, offset + len(data), hashlib.sha256(data)))

    def verify(self, descriptor, before, charge):
        after = os.fstat(descriptor)
        self._identity(before, after)

        def fields(info):
            return info.st_size, info.st_mtime_ns, info.st_ctime_ns

        if fields(before) == fields(after):
            return
        for start, end, expected in self.ranges:
            actual = hashlib.sha256()
            position = start
            while position < end:
                data = os.pread(descriptor, min(64 * 1024, end - position), position)
                charge(len(data))
                if not data:
                    raise SourceReadError(NativeSourceStatus.CHANGED)
                actual.update(data)
                position += len(data)
            if actual.digest() != expected.digest():
                raise SourceReadError(NativeSourceStatus.CHANGED)
        self._identity(before, os.fstat(descriptor))

    @staticmethod
    def _identity(before, after):
        if (before.st_dev, before.st_ino) != (
            after.st_dev,
            after.st_ino,
        ) or after.st_size < before.st_size:
            raise SourceReadError(NativeSourceStatus.CHANGED)
        if not stat.S_ISREG(after.st_mode) or after.st_uid != os.geteuid() or after.st_mode & 0o022:
            raise SourceReadError(NativeSourceStatus.UNAPPROVED)


COLUMNS = ["episode_id", "session_id", "runtime", "source_path", "cwd", "source_status"]
MAX_LOCATOR_BYTES = 16 * 1024


def index_reference(ref: str, repository: Any = None):
    """One bounded episode lookup using today's existing v6 index fields.

    No prior native digests exist in that schema: they are observed by the native
    adapter and may subsequently be persisted as minimal source references.
    """
    if not isinstance(ref, str) or len(ref.encode()) > 512 or ":ep:" not in ref:
        return None, NativeSourceStatus.UNRESOLVED
    owned = repository is None
    try:
        if owned:
            from ssgrep.store import LanceStore

            repository = LanceStore()
        if not repository.exists():
            return None, NativeSourceStatus.UNRESOLVED
        if repository.get_meta("index_state") != "ready":
            return None, NativeSourceStatus.UNRESOLVED
        generation = repository.get_meta("last_index_time")
        literal = "'" + ref.replace("'", "''") + "'"
        rows = list(
            repository.rows("episodes", where=f"episode_id = {literal}", columns=COLUMNS, limit=2)
        )
        if len(rows) != 1:
            return None, NativeSourceStatus.UNRESOLVED
        row = rows[0]
        if sum(len(str(row.get(column, "")).encode()) for column in COLUMNS) > MAX_LOCATOR_BYTES:
            return None, NativeSourceStatus.BOUNDED
        if (
            repository.get_meta("index_state") != "ready"
            or repository.get_meta("last_index_time") != generation
        ):
            return None, NativeSourceStatus.CHANGED
        status = row.get("source_status", "available")
        if status in {"expired", "purged"}:
            return None, NativeSourceStatus.EXPIRED
        if status in {"absent", "missing"}:
            return None, NativeSourceStatus.MISSING
        if status != "available":
            return None, NativeSourceStatus.UNAPPROVED
        runtime, session, path = row.get("runtime"), row.get("session_id"), row.get("source_path")
        if not (
            isinstance(runtime, str)
            and runtime
            and isinstance(session, str)
            and session
            and isinstance(path, str)
            and path
        ):
            return None, NativeSourceStatus.IDENTITY_MISMATCH
        # Native index adapters namespace non-Claude session IDs once.
        if runtime != "claude" and session.startswith(runtime + ":"):
            session = session[len(runtime) + 1 :]
        cwd = row.get("cwd")
        indexed_session = row["session_id"] if runtime == "claude" else None
        return NativeSourceReference(
            ref,
            runtime,
            session,
            Path(path),
            cwd=cwd if isinstance(cwd, str) else None,
            indexed_session_id=indexed_session,
        ), NativeSourceStatus.OK
    except (OSError, ValueError, KeyError, RuntimeError, ImportError):
        return None, NativeSourceStatus.UNRESOLVED
    finally:
        if owned and repository is not None:
            repository.close()


CONTEXT_BYTES = 64 * 1024
OFFSET = re.compile(r"^(?:offset|byte):(\d+)$")


@dataclass(frozen=True)
class NativeRecordContext:
    """Validated native location, separate from the original digested payload."""

    cwd: str | None


class CodexReplayContext:
    """The first native session_meta owns a Codex rollout, including fork copies."""

    def __init__(self, reference, scoper):
        self.session = reference.session_id
        self.scoper = scoper
        self.values = {}
        self.header_seen = False

    def source_context(self) -> NativeRecordContext:
        value = self.values.get("cwd")
        cwd = value if isinstance(value, str) else None
        scope, why = self.scoper.source_status(cwd)
        if scope is not AtomStatus.OK:
            raise SourceReadError(NativeSourceStatus(why))
        return NativeRecordContext(cwd)

    def decode(self, reader, source, obj, locator):
        is_header = isinstance(obj, dict) and obj.get("type") == "session_meta"
        if is_header:
            if self.header_seen:
                return []  # embedded parent metadata cannot replace the file owner
            payload = obj.get("payload")
            if (
                not isinstance(payload, dict)
                or (payload.get("id") or payload.get("session_id")) != self.session
            ):
                raise SourceReadError(NativeSourceStatus.IDENTITY_MISMATCH)
            # Do not seed session: CodexReader._meta initializes cwd/parent only once.
            decoded = reader.decode(source, obj, self.values, locator) or []
            if self.values.get("session") != self.session:
                raise SourceReadError(NativeSourceStatus.IDENTITY_MISMATCH)
            self.header_seen = True
        else:
            if not self.header_seen:
                raise SourceReadError(NativeSourceStatus.IDENTITY_MISMATCH)
            decoded = reader.decode(source, obj, self.values, locator) or []
        self.source_context()
        return decoded


def read_record(adapter, reference, identity, digest, locator):
    if reference.expires_at_ms is not None and reference.expires_at_ms <= adapter.now_ms:
        return NativeRecordResult(NativeSourceStatus.EXPIRED)
    if reference.cwd is not None:
        scope, why = adapter.scoper.source_status(reference.cwd)
        if scope is not AtomStatus.OK:
            return NativeRecordResult(NativeSourceStatus(why))
    reader = adapter.readers.get(reference.runtime)
    if reader is None:
        return NativeRecordResult(NativeSourceStatus.UNSUPPORTED)
    budget = min(MAX_SOURCE_BYTES, adapter.byte_budget) - adapter.bytes_read
    if budget <= 0:
        return NativeRecordResult(NativeSourceStatus.BOUNDED)
    count = 0
    started_bytes = adapter.bytes_read
    native_context = None
    try:
        descriptor = open_source(reference.path, adapter.approved_roots)
        try:
            if isinstance(reader, JsonlReader):
                decoded, count, native_context = _jsonl(
                    adapter, reader, reference, descriptor, locator, budget
                )
            elif reference.runtime == "opencode":
                decoded, count, native_context = _database(
                    adapter, reader, reference, descriptor, locator, budget
                )
                adapter.bytes_read += count
            else:
                raise SourceReadError(NativeSourceStatus.UNSUPPORTED)
        finally:
            os.close(descriptor)
    except SourceReadError as error:
        adapter.bytes_read += error.bytes_read
        return NativeRecordResult(error.status, bytes_read=adapter.bytes_read - started_bytes)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return NativeRecordResult(
            NativeSourceStatus.UNREADABLE, bytes_read=adapter.bytes_read - started_bytes
        )
    matches = [item for item in decoded if item.native_identity == identity]
    if len(matches) != 1:
        return NativeRecordResult(NativeSourceStatus.IDENTITY_MISMATCH, bytes_read=count)
    item = matches[0]
    if (item.runtime, item.session_id) != (reference.runtime, reference.session_id):
        return NativeRecordResult(NativeSourceStatus.IDENTITY_MISMATCH, bytes_read=count)
    if item.digest != digest:
        return NativeRecordResult(NativeSourceStatus.CHANGED, bytes_read=count)
    fallback_cwd = native_context.cwd if native_context is not None else reference.cwd
    cwd = item.payload.get("cwd") or item.payload.get("directory") or fallback_cwd
    scope, why = adapter.scoper.source_status(cwd if isinstance(cwd, str) else None)
    if scope is not AtomStatus.OK:
        return NativeRecordResult(NativeSourceStatus(why), bytes_read=count)
    return NativeRecordResult(NativeSourceStatus.OK, item, count)


def decode_source_record(reader, source, obj, context, locator, codex=None):
    """Use native envelope/header cwd, never cwd hidden in tool arguments."""
    if codex is not None:
        return codex.decode(reader, source, obj, locator)
    if isinstance(obj, dict) and isinstance(obj.get("cwd"), str):
        context["cwd"] = obj["cwd"]
    return reader.decode(source, obj, context, locator) or []


def _jsonl(adapter, reader, reference, descriptor, locator, budget):
    found = OFFSET.fullmatch(locator)
    if not found:
        raise SourceReadError(NativeSourceStatus.UNRESOLVED)
    offset = int(found.group(1))
    before = os.fstat(descriptor)
    if offset >= before.st_size:
        raise SourceReadError(NativeSourceStatus.MISSING)
    # Native cwd/ordinal/link context cannot be inferred across omitted history.
    if offset > 2 * CONTEXT_BYTES:
        raise SourceReadError(NativeSourceStatus.INCOMPLETE)
    source = reader.source_for(reference.path)
    codex = CodexReplayContext(reference, adapter.scoper) if reference.runtime == "codex" else None
    context = codex.values if codex is not None else {"session": reference.session_id}
    proof = ReadProof()
    count = 0
    header_seen = reference.runtime == "claude"

    def charge(size):
        nonlocal count
        count += size
        adapter.bytes_read += size
        if count > budget:
            raise SourceReadError(NativeSourceStatus.BOUNDED)

    def decode(raw, position):
        nonlocal header_seen
        decoded = decode_source_record(
            reader, source, loads(raw), context, f"offset:{position}", codex
        )
        headers = [item for item in decoded if item.kind == "session_header"]
        if headers:
            if reference.session_id not in {item.session_id for item in headers}:
                raise SourceReadError(NativeSourceStatus.IDENTITY_MISMATCH)
            header_seen = True
        return decoded

    with os.fdopen(descriptor, "rb", closefd=False) as stream:
        prefix = stream.read(min(offset, budget))
        charge(len(prefix))
        proof.add(0, prefix)
        if len(prefix) != offset:
            raise SourceReadError(NativeSourceStatus.BOUNDED)
        position = 0
        for raw in prefix.splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                raise SourceReadError(NativeSourceStatus.INCOMPLETE)
            decode(raw, position)
            position += len(raw)
        raw = stream.readline(min(MAX_RAW_RECORD_BYTES + 1, budget - count + 1))
        charge(len(raw))
        proof.add(offset, raw)
        if len(raw) > MAX_RAW_RECORD_BYTES:
            raise SourceReadError(NativeSourceStatus.BOUNDED)
        if not raw.endswith(b"\n"):
            raise SourceReadError(NativeSourceStatus.INCOMPLETE)
        decoded = decode(raw, offset)
        header_seen = codex.header_seen if codex is not None else header_seen
        if not header_seen:
            raise SourceReadError(NativeSourceStatus.IDENTITY_MISMATCH)
        native_context = (
            codex.source_context() if codex is not None else NativeRecordContext(context.get("cwd"))
        )
        scope, why = adapter.scoper.source_status(native_context.cwd)
        if scope is not AtomStatus.OK:
            raise SourceReadError(NativeSourceStatus(why))
        proof.verify(descriptor, before, charge)
    return decoded, count, native_context


def _database(adapter, reader, reference, descriptor, locator, budget):
    try:
        import apsw
    except ImportError:
        raise SourceReadError(NativeSourceStatus.UNREADABLE) from None

    table, separator, row_id = locator.partition(":")
    if (
        not separator
        or table not in {"session", "message", "part"}
        or not row_id
        or len(row_id) > 512
    ):
        raise SourceReadError(NativeSourceStatus.UNRESOLVED)
    info = os.fstat(descriptor)
    result = ReadResult()
    source = reader.source_for(reference.path)
    count = 0

    try:
        with reader.snapshot(reference.path, descriptor=descriptor) as connection:
            connection.set_row_trace(named_row)
            connection.set_progress_handler(lambda: 1, 100_000)
            from ssgrep.accounting.readers.opencode_storage import (
                USER_ROLE_SQL,
                session_row,
                user_part,
            )

            owner = session_row(connection, reference.session_id, budget)
            count += owner["n"]
            scope, why = adapter.scoper.source_status(owner["directory"])
            if scope is not AtomStatus.OK:
                raise SourceReadError(NativeSourceStatus(why), count)
            if table == "session":
                if row_id != reference.session_id:
                    raise SourceReadError(NativeSourceStatus.IDENTITY_MISMATCH, count)
                row = owner
            else:
                message_id = (
                    "CASE WHEN length(message_id) > 512 THEN NULL ELSE message_id END AS message_id"
                    if table == "part"
                    else "NULL AS message_id"
                )
                role = USER_ROLE_SQL if table == "part" else "NULL AS message_role"
                row = connection.execute(
                    f"SELECT id, session_id, {message_id}, {role}, time_updated, "
                    "? AS cwd, "
                    "CASE WHEN length(CAST(data AS BLOB)) > ? THEN NULL ELSE data END AS data, "
                    f"length(CAST(data AS BLOB)) AS n FROM {table} "
                    "WHERE id = ? AND session_id = ? LIMIT 1",
                    (
                        owner["directory"],
                        min(MAX_RAW_RECORD_BYTES, budget - count),
                        row_id,
                        reference.session_id,
                    ),
                ).fetchone()
            if row is None:
                raise SourceReadError(NativeSourceStatus.MISSING)
            if table != "session":
                if row["data"] is None:
                    raise SourceReadError(NativeSourceStatus.BOUNDED)
                count += int(row["n"] or 0)
            held = []
            if table != "part" or not user_part(reader, source, row, result):
                reader._row(source, table, row, held, result)
            if held or result.gaps:
                raise SourceReadError(NativeSourceStatus.INCOMPLETE, count)
    except SourceReadError as error:
        error.bytes_read = count
        raise
    except SnapshotChanged:
        raise SourceReadError(NativeSourceStatus.CHANGED, count) from None
    except SnapshotUnavailable:
        raise SourceReadError(NativeSourceStatus.UNREADABLE, count) from None
    except apsw.Error as error:
        status = (
            NativeSourceStatus.BOUNDED
            if isinstance(error, apsw.InterruptError)
            else NativeSourceStatus.UNREADABLE
        )
        raise SourceReadError(status, count) from None
    try:
        after = os.stat(reference.path, follow_symlinks=False)
    except FileNotFoundError:
        raise SourceReadError(NativeSourceStatus.MISSING, count) from None
    except OSError:
        raise SourceReadError(NativeSourceStatus.UNREADABLE, count) from None
    if (info.st_dev, info.st_ino) != (after.st_dev, after.st_ino):
        raise SourceReadError(NativeSourceStatus.UNAPPROVED, count)
    return result.evidence, count, NativeRecordContext(owner["directory"])
