"""Bounded direct native record hydration for compact references, not history scans."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.native_source import (
    MAX_SOURCE_BYTES,
    NativeRecordResult,
    NativeSourceStatus,
    SourceReadError,
    open_source,
)
from ssgrep.accounting.native_source_proof import ReadProof
from ssgrep.accounting.readers.base import JsonlReader, ReadResult
from ssgrep.accounting.readers.values import loads
from ssgrep.accounting.valuation_inputs import AtomStatus

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
    from ssgrep.accounting.readers.opencode_snapshot import (
        SnapshotChanged,
        SnapshotUnavailable,
        named_row,
    )

    try:
        with reader.snapshot(reference.path, descriptor=descriptor) as connection:
            connection.set_row_trace(named_row)
            connection.set_progress_handler(lambda: 1, 100_000)
            from ssgrep.accounting.native_source_db import USER_ROLE_SQL, session_row, user_part

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
