"""Recover legacy native IDs from approved recorded locators and verified old content."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES, RawEvidence
from ssgrep.accounting.native_identity import CONTEXT_BYTES
from ssgrep.accounting.native_source import SourceReadError, open_source
from ssgrep.accounting.readers.pi_family import HEADER_LINES, PiFamilyReader
from ssgrep.accounting.readers.values import evidence_digest, loads
from ssgrep.accounting.valuation_inputs import AtomStatus

MIGRATED_RUNTIMES = frozenset({"claude", "pi", "omp", "prime-agent"})


def _legacy_payload(item: RawEvidence, candidate: RawEvidence, source: Any, obj: dict) -> dict:
    p = candidate.payload
    if item.runtime == "claude":
        names = {
            "tool_call": {"call_id", "name", "arguments", "message_id", "cwd"},
            "tool_result": {"call_id", "output", "output_bytes", "is_error"},
            "user_turn": {"text", "cwd"},
            "usage": {"message_id", "model", "tokens", "streaming_updates"},
        }.get(item.kind, set(p))
        result = {k: v for k, v in p.items() if k in names}
        if source.role == "child":
            result["parent_session"] = p.get("parent_session")
        return result
    if item.kind == "tool_call":
        names = {"call_id", "name", "arguments", "entry_id", "cwd", "occurrence"}
    elif item.kind == "tool_result":
        names = {"occurrence", "call_id", "tool_name", "output", "output_bytes", "is_error"}
    elif item.kind == "user_turn":
        names = {"text"}
    else:
        names = set(p) - {
            "entry_id",
            "native_parent_id",
            "session_context",
            "parent_session",
            "originating_turn_record_id",
            "task_context_basis",
            "baseline_prior_context_unavailable",
        }
        if item.kind == "usage":
            names.discard("cwd")
    result = {k: v for k, v in p.items() if k in names}
    if item.kind in {"tool_call", "tool_result"}:
        suffix = item.record_id.rsplit("#", 1)
        result["occurrence"] = int(suffix[-1]) if len(suffix) == 2 and suffix[-1].isdigit() else 0
    if item.kind == "tool_call" and item.session_id == source.path.stem and not obj.get("cwd"):
        # The defective EOF baseline had no header/cwd context; verify that exact old digest.
        result["cwd"] = item.payload.get("cwd")
    return result


def _context(reader: Any, source: Any, fd: int, offset: int) -> tuple[dict, int]:
    """At most two metadata windows, using the already verified descriptor."""
    ctx: dict = {}
    first = os.pread(fd, min(offset, CONTEXT_BYTES), 0)
    consumed = len(first)
    if isinstance(reader, PiFamilyReader):
        for raw in first.splitlines(keepends=True)[:HEADER_LINES]:
            if not raw.endswith(b"\n"):
                continue
            try:
                obj = loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "session":
                reader.decode(source, obj, ctx, "offset:0")
                break
    start = max(0, offset - CONTEXT_BYTES)
    tail = os.pread(fd, offset - start, start)
    consumed += len(tail)
    if start:
        boundary = tail.find(b"\n")
        start += boundary + 1 if boundary >= 0 else len(tail)
        tail = tail[boundary + 1 :] if boundary >= 0 else b""
        ctx["call_context_incomplete"] = True
    for raw in tail.splitlines(keepends=True):
        locator = f"offset:{start}"
        start += len(raw)
        if not raw.endswith(b"\n"):
            continue
        try:
            obj = loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if isinstance(reader, PiFamilyReader) and obj.get("type") not in {"session", "message"}:
            from ssgrep.accounting.native_task_context import observe

            observe(obj, ctx, locator)
        message = obj.get("message")
        if isinstance(message, dict):
            reader.decode(
                source,
                {**obj, "message": {k: v for k, v in message.items() if k != "usage"}},
                ctx,
                locator,
            )
    return ctx, consumed


@dataclass(frozen=True)
class RecordMigration:
    evidence: RawEvidence | None
    reason: str = ""
    bytes_read: int = 0


def migrate_record(item: RawEvidence, readers: dict, scoper: Any, budget: int) -> RecordMigration:
    if item.origin_domain is not None or item.runtime not in MIGRATED_RUNTIMES:
        return RecordMigration(item)
    reader = readers.get(item.runtime)
    prefix = item.runtime + ":"
    if (
        reader is None
        or not item.source_id.startswith(prefix)
        or not item.locator.startswith("offset:")
    ):
        return RecordMigration(None, "source_identity_mismatch")
    try:
        offset = int(item.locator.split(":", 1)[1])
    except ValueError:
        return RecordMigration(None, "source_identity_mismatch")
    if offset < 0 or budget <= 2 * CONTEXT_BYTES:
        return RecordMigration(None, "source_read_bound")
    source = reader.source_for(Path(item.source_id[len(prefix) :]))
    try:
        descriptor = open_source(source.path, tuple(reader.roots()))
        with os.fdopen(descriptor, "rb") as handle:
            context, context_bytes = _context(reader, source, handle.fileno(), offset)
            handle.seek(offset)
            raw = handle.readline(min(MAX_RAW_RECORD_BYTES, budget - 2 * CONTEXT_BYTES) + 1)
        consumed = len(raw) + context_bytes
        if not raw.endswith(b"\n") or len(raw) > MAX_RAW_RECORD_BYTES:
            return RecordMigration(None, "source_read_bound", consumed)
        obj = loads(raw)
        decoded = reader.decode(source, obj, context, item.locator) or []
        candidates = [
            c
            for c in decoded
            if c.kind == item.kind
            and all(
                item.payload.get(k) in (None, c.payload.get(k)) for k in ("call_id", "message_id")
            )
        ]
        matches = [
            c
            for c in candidates
            if evidence_digest(item.kind, _legacy_payload(item, c, source, obj)) == item.digest
        ]
        if len(matches) != 1:
            return RecordMigration(
                None, "source_digest_drift" if candidates else "source_identity_mismatch", consumed
            )
        candidate = matches[0]
        cwd = candidate.payload.get("cwd") or context.get("cwd")
        scope, reason = scoper.source_status(cwd)
        if scope is not AtomStatus.OK and item.kind not in {"session_header", "lineage"}:
            return RecordMigration(None, reason, consumed)
        return RecordMigration(candidate, bytes_read=consumed)
    except SourceReadError as error:
        return RecordMigration(None, error.status.value)
    except (OSError, ValueError, TypeError):
        return RecordMigration(None, "source_unreadable")
