"""Bounded selected-session replay with independently verified prefix results.

Native context is decoded once through selected offsets; no skipped middle can
silently inherit an old cwd. Consumed-byte proofs distinguish safe concurrent
appends from rewrites. Plaintext is transient and never persisted here. Like
collection, one started bounded record may finish past the cooperative byte target.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES
from ssgrep.accounting.native_source import (
    MAX_SOURCE_BYTES,
    NativeRecordResult,
    NativeSourceStatus as Status,
    SourceReadError,
    open_source,
)
from ssgrep.accounting.native_source_proof import ReadProof
from ssgrep.accounting.native_source_record import CodexReplayContext, decode_source_record
from ssgrep.accounting.readers.base import JsonlReader
from ssgrep.accounting.readers.values import loads
from ssgrep.accounting.valuation_inputs import AtomStatus

OFFSET = re.compile(r"^(?:offset|byte):(\d+)$")


def read_batch(adapter, reference, items):
    """One source/session, exact selected outputs, complete intervening native context."""
    reader = adapter.readers.get(reference.runtime)
    if not isinstance(reader, JsonlReader):
        return [
            adapter.read_record(reference, item.native_identity, item.digest, locator=item.locator)
            for item in items
        ]
    if not items:
        return []
    if reference.expires_at_ms is not None and reference.expires_at_ms <= adapter.now_ms:
        return [NativeRecordResult(Status.EXPIRED) for _ in items]
    if reference.cwd is not None:
        scope, why = adapter.scoper.source_status(reference.cwd)
        if scope is not AtomStatus.OK:
            return [NativeRecordResult(Status(why)) for _ in items]
    budget = min(MAX_SOURCE_BYTES, adapter.byte_budget) - adapter.bytes_read
    if budget <= 0:
        return [NativeRecordResult(Status.BOUNDED) for _ in items]
    grouped = {}
    for index, item in enumerate(items):
        found = OFFSET.fullmatch(item.locator)
        if found is None:
            return [NativeRecordResult(Status.UNRESOLVED) for _ in items]
        grouped.setdefault(int(found.group(1)), []).append((index, item))
    results = [NativeRecordResult(Status.INCOMPLETE) for _ in items]
    count = 0
    descriptor = None
    before = None
    proof = ReadProof()
    codex = CodexReplayContext(reference, adapter.scoper) if reference.runtime == "codex" else None
    context = codex.values if codex is not None else {"session": reference.session_id}
    source = reader.source_for(reference.path)
    header_seen = reference.runtime == "claude"

    def charge(size, *, started_record=False):
        nonlocal count
        count += size
        adapter.bytes_read += size
        if count > budget and not started_record:
            raise SourceReadError(Status.BOUNDED)

    def decode(raw, position):
        nonlocal header_seen
        decoded = decode_source_record(
            reader, source, loads(raw), context, f"offset:{position}", codex
        )
        headers = [row for row in decoded if row.kind == "session_header"]
        if headers:
            if reference.session_id not in {row.session_id for row in headers}:
                raise SourceReadError(Status.IDENTITY_MISMATCH)
            header_seen = True
        return decoded

    def remaining(status):
        return [row if row.evidence is not None else NativeRecordResult(status) for row in results]

    try:
        descriptor = open_source(reference.path, adapter.approved_roots)
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            position = 0
            for offset, wanted in sorted(grouped.items()):
                if offset >= before.st_size:
                    for index, _ in wanted:
                        results[index] = NativeRecordResult(Status.MISSING)
                    continue
                while position < offset:
                    if count >= budget:
                        raise SourceReadError(Status.BOUNDED)
                    stream.seek(position)
                    raw = stream.readline(min(MAX_RAW_RECORD_BYTES + 1, offset - position))
                    charge(len(raw), started_record=True)
                    proof.add(position, raw)
                    if len(raw) > MAX_RAW_RECORD_BYTES:
                        raise SourceReadError(Status.BOUNDED)
                    if not raw.endswith(b"\n"):
                        raise SourceReadError(Status.INCOMPLETE)
                    decode(raw, position)
                    position += len(raw)
                stream.seek(offset)
                if count >= budget:
                    raise SourceReadError(Status.BOUNDED)
                raw = stream.readline(MAX_RAW_RECORD_BYTES + 1)
                charge(len(raw), started_record=True)
                proof.add(offset, raw)
                if len(raw) > MAX_RAW_RECORD_BYTES or not raw.endswith(b"\n"):
                    raise SourceReadError(
                        Status.BOUNDED if len(raw) > MAX_RAW_RECORD_BYTES else Status.INCOMPLETE
                    )
                decoded = decode(raw, offset)
                position = offset + len(raw)
                if not (codex.header_seen if codex is not None else header_seen):
                    raise SourceReadError(Status.IDENTITY_MISMATCH)
                native_cwd = codex.source_context().cwd if codex is not None else context.get("cwd")
                for index, expected in wanted:
                    matches = [
                        row for row in decoded if row.native_identity == expected.native_identity
                    ]
                    if len(matches) != 1:
                        results[index] = NativeRecordResult(Status.IDENTITY_MISMATCH)
                        continue
                    actual = matches[0]
                    if actual.identity != expected.identity or actual.ts_utc != expected.ts_utc:
                        results[index] = NativeRecordResult(Status.CHANGED)
                        continue
                    if (actual.runtime, actual.session_id) != (
                        expected.runtime,
                        expected.session_id,
                    ):
                        results[index] = NativeRecordResult(Status.IDENTITY_MISMATCH)
                        continue
                    cwd = actual.payload.get("cwd") or actual.payload.get("directory") or native_cwd
                    scope, why = adapter.scoper.source_status(cwd if isinstance(cwd, str) else None)
                    results[index] = (
                        NativeRecordResult(Status(why))
                        if scope is not AtomStatus.OK
                        else NativeRecordResult(Status.OK, actual)
                    )
    except SourceReadError as error:
        results = remaining(error.status)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        results = remaining(Status.UNREADABLE)
    finally:
        if descriptor is not None:
            if before is not None and any(row.evidence is not None for row in results):
                try:
                    proof.verify(descriptor, before, charge)
                except SourceReadError as error:
                    results = [NativeRecordResult(error.status) for _ in items]
                except OSError:
                    # Later unavailable I/O does not erase independently verified rows.
                    pass
            os.close(descriptor)
    if results:
        results[0] = replace(results[0], bytes_read=count)
    return results
