"""Shared decoding for Pi-format transcripts (Pi, omp, Prime).

The three runtimes write the same append-only entry format (``session`` header,
then ``message``/``model_usage``/... entries) but differ in identity, discovery
roots, and parent conventions, which live in the thin per-runtime subclasses.
This module deliberately does not import the retrieval adapters: accounting
reads raw bytes and must keep tool results, usage, and parent paths those
adapters normalize away.

Parent links are *resolved from raw evidence only*: a header's ``parentSession``
path is accepted only when it stays inside the configured runtime roots, and
only the parent's header line is read to learn its session id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.readers.base import JsonlReader, Source
from ssgrep.accounting.readers.stream import JsonlScanner
from ssgrep.accounting.readers.values import (
    as_int,
    loads,
    make_evidence,
    nano_usd,
    pick_ints,
    text_of,
    to_utc,
)

USAGE_NAMES = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cacheRead": "cache_read_tokens",
    "cacheWrite": "cache_write_tokens",
    "totalTokens": "total_tokens",
    "reasoningTokens": "reasoning_tokens",
}
COST_NAMES = ("input", "output", "cacheRead", "cacheWrite", "total")
NON_EVIDENCE = frozenset(
    {
        "title",
        "title_change",
        "session_init",
        "model_change",
        "thinking_level_change",
        "service_tier_change",
        "session_info",
        "session_state",
        "custom",
        "custom_message",
        "compaction",
        "credential_pin",
        "ttsr_injection",
        "agent_status",
        "request_started",
        "request_finished",
        "request_failed",
        "child_returned",
        "git_state",
        "session_registered",
        "label",
        "branch_summary",
    }
)
HEADER_LINES = 8


@dataclass(frozen=True)
class ParentLink:
    """Outcome of resolving a session's parent from raw header evidence."""

    state: str  # resolved | missing | outside_roots | unreadable | none
    child_session: str
    parent_session: str | None = None
    parent_path: Path | None = None


def header_session_id(path: Path) -> str | None:
    """Session id from the first lines of a transcript; reads at most a few bounded lines."""
    lines = JsonlScanner(path).__iter__()
    try:
        for count, line in enumerate(lines):
            if count >= HEADER_LINES:
                return None
            if line.data is None:
                continue
            try:
                obj = loads(line.data)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "session":
                value = obj.get("id")
                return value if isinstance(value, str) and value else None
    except OSError:
        return None
    finally:
        lines.close()
    return None


class PiFamilyReader(JsonlReader):
    runtime: ClassVar[str]

    # ---- identity / roles -------------------------------------------------
    def accepts(self, path: Path) -> bool:
        return path.name != "semantic-edges.jsonl"

    def _session(self, source: Source, ctx: dict[str, Any]) -> str:
        return str(ctx.get("session") or source.path.stem)

    # ---- decoding ----------------------------------------------------------
    def decode(
        self, source: Source, obj: Any, ctx: dict[str, Any], locator: str
    ) -> list[RawEvidence] | None:
        if not isinstance(obj, dict):
            return None
        kind = obj.get("type")
        if not isinstance(kind, str):
            return None
        if kind == "session":
            return self._header(source, obj, ctx, locator)
        session = self._session(source, ctx)
        ts = to_utc(obj.get("timestamp"))
        branch = "artifact" if source.role == "artifact" else "main"

        def evidence(record_id: str, ekind: str, body: dict[str, Any]) -> RawEvidence:
            return make_evidence(
                runtime=self.runtime,
                source_id=source.source_id,
                branch_id=branch,
                session_id=session,
                record_id=record_id,
                locator=locator,
                kind=ekind,
                ts=ts,
                payload=body,
            )

        if kind == "message":
            message = obj.get("message")
            if not isinstance(message, dict):
                return None
            return self._message(obj, message, ctx, evidence)
        if kind == "model_usage":
            return [self._usage(str(obj.get("id")), obj, evidence, purpose=obj.get("purpose"))]
        if kind == "rlm_subagent":
            return [self._lineage(obj, evidence, ctx)]
        return [] if kind in NON_EVIDENCE else None

    def _header(
        self, source: Source, obj: dict, ctx: dict[str, Any], locator: str
    ) -> list[RawEvidence]:
        raw_id = obj.get("id")
        session: str = raw_id if isinstance(raw_id, str) else source.path.stem
        ctx["session"] = session
        for table in ("calls", "results", "lineage"):
            ctx.pop(table, None)
        ctx["cwd"] = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
        body = {
            "id": session,
            "cwd": ctx["cwd"],
            "parent_session_path": _string(obj.get("parentSession")),
            "parent_session_id": _string(obj.get("parentSessionId")),
            "rlm_depth": as_int(obj.get("rlmDepth")),
            "version": as_int(obj.get("version")),
        }
        return [
            make_evidence(
                runtime=self.runtime,
                source_id=source.source_id,
                branch_id="artifact" if source.role == "artifact" else "main",
                session_id=session,
                record_id=f"session:{session}",
                locator=locator,
                kind="session_header",
                ts=to_utc(obj.get("timestamp")),
                payload=body,
            )
        ]

    def _message(self, obj: dict, message: dict, ctx: dict[str, Any], evidence):
        role = message.get("role")
        entry = str(obj.get("id") or "")
        cwd = _string(obj.get("cwd")) or ctx.get("cwd")
        if role == "assistant":
            out = []
            if isinstance(message.get("usage"), dict):
                out.append(self._usage(entry, message, evidence, model=message.get("model")))
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    call_id = block.get("id")
                    if isinstance(call_id, str):
                        n = _next(ctx, "calls", call_id)
                        body = {
                            "call_id": call_id,
                            "name": block.get("name"),
                            "arguments": block.get("arguments"),
                            "entry_id": entry,
                            "cwd": cwd,
                            "occurrence": n,
                        }
                        out.append(evidence(_rid("call", call_id, n), "tool_call", body))
            return out
        if role == "toolResult":
            call_id = message.get("toolCallId")
            if not isinstance(call_id, str):
                return None
            text = text_of(message.get("content"))
            flag = message.get("isError")
            n = _next(ctx, "results", call_id)
            body = {
                "occurrence": n,
                "call_id": call_id,
                "tool_name": message.get("toolName"),
                "output": text,
                "output_bytes": len(text.encode()),
                "is_error": flag if isinstance(flag, bool) else None,
            }
            return [evidence(_rid("result", call_id, n), "tool_result", body)]
        if role == "user":
            text = text_of(message.get("content"))
            return [evidence(f"turn:{entry}", "user_turn", {"text": text})] if text else []
        return []

    @staticmethod
    def _usage(entry: str, source: dict, evidence, **extra: Any) -> RawEvidence:
        usage = source.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        cost_nano = None
        if isinstance(cost, dict):
            cost_nano = {n: v for n in COST_NAMES if (v := nano_usd(cost.get(n))) is not None}
        body = {"tokens": pick_ints(usage, USAGE_NAMES), "cost_nano_usd": cost_nano or None}
        body.update({key: value for key, value in extra.items() if value is not None})
        return evidence(f"usage:{entry}", "usage", body)

    @staticmethod
    def _lineage(obj: dict, evidence, ctx: dict[str, Any]) -> RawEvidence:
        body = {
            "child_id": _string(obj.get("childId")),
            "session_file": _string(obj.get("sessionFile")),
            "parent_session_id": _string(obj.get("parentSessionId")),
            "parent_session_file": _string(obj.get("parentSessionFile")),
            "rlm_depth": as_int(obj.get("rlmDepth")),
            "status": _string(obj.get("status")),
        }
        # one row per status transition (running, completed, deleted), not one per child
        n = _next(ctx, "lineage", str(body["child_id"]))
        return evidence(_rid("lineage", str(body["child_id"]), n), "lineage", body)

    # ---- parent resolution ---------------------------------------------------
    def session_index(self) -> dict[str, Path]:
        """Session id to transcript path for every discovered source (header lines only)."""
        index: dict[str, Path] = {}
        for source in self.discover():
            session = header_session_id(source.path)
            if session is not None:
                index.setdefault(session, source.path)
        return index

    def parent_link(self, header: RawEvidence, index: dict[str, Path] | None = None) -> ParentLink:
        """Resolve the parent of a ``session_header`` evidence record from raw fields only."""
        child = header.session_id
        raw_path = header.payload.get("parent_session_path")
        raw_id = header.payload.get("parent_session_id")
        if raw_path:
            return self._by_path(child, raw_path)
        if raw_id:
            return self._by_id(child, raw_id, index)
        return ParentLink("none", child)

    def _resolved_roots(self) -> list[str]:
        return [os.path.realpath(root) for root in self.roots()]

    def _by_path(self, child: str, raw: str) -> ParentLink:
        candidate = os.path.realpath(raw)
        inside = any(
            candidate == root or candidate.startswith(root.rstrip(os.sep) + os.sep)
            for root in self._resolved_roots()
        )
        if not inside:
            return ParentLink("outside_roots", child)
        path = Path(candidate)
        if not path.is_file():
            return ParentLink("missing", child, parent_path=path)
        parent = header_session_id(path)
        if parent is None:
            return ParentLink("unreadable", child, parent_path=path)
        return ParentLink("resolved", child, parent, path)

    def _by_id(self, child: str, raw_id: str, index: dict[str, Path] | None) -> ParentLink:
        path = (index if index is not None else self.session_index()).get(raw_id)
        if path is None:
            return ParentLink("missing", child, raw_id)
        return ParentLink("resolved", child, raw_id, path)


def _next(ctx: dict[str, Any], table: str, key: str) -> int:
    """0-based occurrence of ``key`` in this session so far (ids like ``call_0`` are reused)."""
    counts = ctx.setdefault(table, {})
    n = counts.get(key, 0)
    counts[key] = n + 1
    return n


def _rid(kind: str, key: str, occurrence: int) -> str:
    return f"{kind}:{key}" if occurrence == 0 else f"{kind}:{key}#{occurrence}"


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
