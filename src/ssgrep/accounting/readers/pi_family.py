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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES, RawEvidence, TaskId, digest_of
from ssgrep.accounting.readers import classify as roots
from ssgrep.accounting.readers.base import (
    CHUNK_BYTES,
    JsonlReader,
    JsonlScanner,
    Source,
    SourceFile,
    as_int,
    loads,
    make_evidence,
    nano_usd,
    pick_ints,
    text_of,
    to_utc,
)
from ssgrep.accounting.readers.native_identity import (
    CONTEXT_BYTES,
    fields as task_fields,
    link_result,
    observe as observe_task,
    remember_call,
    with_origin,
)
from ssgrep.accounting.readers.timeline import (
    Record,
    SessionKey,
    Timeline,
    Timelines,
    key_str,
    record_text,
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
    lines = JsonlScanner(
        path, budget=MAX_RAW_RECORD_BYTES + CHUNK_BYTES, finish_record=False
    ).__iter__()
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

    def baseline_context(self, source: Source, offset: int, handle: SourceFile) -> dict[str, Any]:
        """Read at most 64 KiB at each end from the accepted source descriptor."""
        ctx: dict[str, Any] = {}
        first = handle.read(0, min(offset, CONTEXT_BYTES))
        start = max(0, offset - CONTEXT_BYTES)
        tail = handle.read(start, offset - start)
        for raw in first.splitlines(keepends=True)[:HEADER_LINES]:
            if not raw.endswith(b"\n"):
                continue
            try:
                obj = loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "session":
                self._header(source, obj, ctx, "offset:0")
                break
        ctx["baseline_context"] = True
        if start:
            boundary = tail.find(b"\n")
            start += boundary + 1 if boundary >= 0 else len(tail)
            tail = tail[boundary + 1 :] if boundary >= 0 else b""
            ctx["call_context_incomplete"] = True
        position = start
        for raw in tail.splitlines(keepends=True):
            locator = f"offset:{position}"
            position += len(raw)
            if not raw.endswith(b"\n"):
                continue
            try:
                obj = loads(raw)
            except ValueError:
                continue
            if not isinstance(obj, dict) or obj.get("type") == "session":
                continue
            if obj.get("type") != "message":
                observe_task(obj, ctx, locator)
                continue
            if isinstance(obj.get("cwd"), str):
                ctx["cwd"] = obj["cwd"]
            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                observe_task(obj, ctx, locator)
            if message.get("role") in {"assistant", "toolResult"}:
                self.decode(
                    source,
                    {**obj, "message": {k: v for k, v in message.items() if k != "usage"}},
                    ctx,
                    locator,
                )
        return ctx

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
        entry = _string(obj.get("id"))
        token = entry or locator
        observe_task(obj, ctx, locator)

        def evidence(record_id: str, ekind: str, body: dict[str, Any]) -> RawEvidence:
            header = ctx.get("header", {})
            body = {
                **body,
                "entry_id": entry,
                "native_parent_id": _string(obj.get("parentId")),
                "session_context": header,
                "parent_session": header.get("parent_session_id"),
                **task_fields(ctx),
            }
            item = make_evidence(
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
            domain = session if ctx.get("native_session") else source.source_id
            status = "session_scoped" if ctx.get("native_session") else "missing_session_id"
            if entry is None:
                domain, status = source.source_id, "missing_entry_id"
            return with_origin(item, domain, record_id, status)

        if kind == "message":
            message = obj.get("message")
            if not isinstance(message, dict):
                return None
            return self._message(obj, message, ctx, evidence, token)
        if kind == "model_usage":
            return [self._usage(token, obj, evidence, purpose=obj.get("purpose"))]
        if kind == "rlm_subagent":
            return [self._lineage(obj, evidence, ctx, token)]
        return [] if kind in NON_EVIDENCE else None

    def _header(
        self, source: Source, obj: dict, ctx: dict[str, Any], locator: str
    ) -> list[RawEvidence]:
        raw_id = obj.get("id")
        session = _string(raw_id) or source.path.stem
        ctx["session"] = session
        ctx["native_session"] = bool(_string(raw_id))
        ctx.pop("pending_calls", None)
        ctx.pop("call_context_incomplete", None)
        ctx["entry_turns"], ctx["turn_id"] = {}, None
        ctx.pop("baseline_context", None)
        ctx["cwd"] = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
        body = {
            "id": session,
            "cwd": ctx["cwd"],
            "parent_session_path": _string(obj.get("parentSession")),
            "parent_session_id": _string(obj.get("parentSessionId")),
            "rlm_depth": as_int(obj.get("rlmDepth")),
            "version": as_int(obj.get("version")),
        }
        ctx["header"] = body
        item = make_evidence(
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
        domain = session if ctx["native_session"] else source.source_id
        status = "session_scoped" if ctx["native_session"] else "missing_session_id"
        return [with_origin(item, domain, item.record_id, status)]

    def _message(self, obj: dict, message: dict, ctx: dict[str, Any], evidence, token: str):
        role = message.get("role")
        entry = token
        cwd = _string(obj.get("cwd")) or ctx.get("cwd")
        if _string(obj.get("cwd")):
            ctx["cwd"] = cwd
        if role == "assistant":
            out = []
            if isinstance(message.get("usage"), dict):
                out.append(self._usage(entry, message, evidence, model=message.get("model")))
            content = message.get("content")
            for index, block in enumerate(content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    call_id = block.get("id")
                    if isinstance(call_id, str):
                        body = {
                            "call_id": call_id,
                            "name": block.get("name"),
                            "arguments": block.get("arguments"),
                            "entry_id": entry,
                            "cwd": cwd,
                        }
                        rid = f"call:{call_id}@{entry}:{index}"
                        remember_call(ctx, call_id, rid, entry)
                        out.append(evidence(rid, "tool_call", body))
            return out
        if role == "toolResult":
            call_id = message.get("toolCallId")
            if not isinstance(call_id, str):
                return None
            text = text_of(message.get("content"))
            flag = message.get("isError")
            linked, status = link_result(ctx, call_id, obj.get("parentId"))
            body = {
                "call_record_id": linked,
                "call_link_status": status,
                "cwd": cwd,
                "call_id": call_id,
                "tool_name": message.get("toolName"),
                "output": text,
                "output_bytes": len(text.encode()),
                "is_error": flag if isinstance(flag, bool) else None,
            }
            return [evidence(f"result:{call_id}@{entry}", "tool_result", body)]
        if role == "user":
            text = text_of(message.get("content"))
            return (
                [evidence(f"turn:{entry}", "user_turn", {"text": text, "cwd": cwd})] if text else []
            )
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
    def _lineage(obj: dict, evidence, ctx: dict[str, Any], token: str) -> RawEvidence:
        body = {
            "child_id": _string(obj.get("childId")),
            "session_file": _string(obj.get("sessionFile")),
            "parent_session_id": _string(obj.get("parentSessionId")),
            "parent_session_file": _string(obj.get("parentSessionFile")),
            "rlm_depth": as_int(obj.get("rlmDepth")),
            "status": _string(obj.get("status")),
        }
        # one row per status transition (running, completed, deleted), not one per child
        return evidence(f"lineage:{token}", "lineage", body)

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


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{3,}")
GENERIC_NAMES = frozenset(
    {"main", "agent", "task", "child", "subagent", "explore", "general", "plan", "root", "none"}
)
LAUNCH_SKEW_MS = 2000


@dataclass(frozen=True)
class Launch:
    """The parent position that launched a child (``call_seq`` in ``parent``)."""

    parent: SessionKey
    call_seq: int
    via: str  # the child identity that the launching record names


@dataclass(frozen=True)
class Issue:
    kind: str  # missing_parent | cyclic_parent | unlinked_launch | unresolved_lineage_row
    session: str


@dataclass
class Lineage:
    parent_of: dict[SessionKey, SessionKey] = field(default_factory=dict)
    declared: dict[SessionKey, str] = field(default_factory=dict)  # declared parent ref
    root_of: dict[SessionKey, SessionKey] = field(default_factory=dict)
    launch: dict[SessionKey, Launch] = field(default_factory=dict)
    broken: set[SessionKey] = field(default_factory=set)  # missing or cyclic ancestry
    names: dict[SessionKey, set[str]] = field(default_factory=dict)  # extra launch identities
    issues: list[Issue] = field(default_factory=list)
    timelines: Timelines | None = None
    _tasks: dict[SessionKey, TaskId] = field(default_factory=dict)

    def root(self, key: SessionKey) -> str:
        return key_str(self.root_of[key])

    def identities(self, key: SessionKey) -> list[str]:
        """Names under which a parent's records refer to child ``key``."""
        assert self.timelines is not None
        return _identity_words(key, self.timelines.sessions[key], self.names.get(key, set()))

    def is_child(self, key: SessionKey) -> bool:
        return key in self.declared

    def task_of(self, key: SessionKey, seq: int) -> TaskId:
        """The task of the record at ``seq`` of ``key`` (see module docstring)."""
        assert self.timelines is not None
        record = self.timelines.sessions[key].records[seq]
        if (
            record.identity in self.timelines.ownership_unknown
            or record.identity in self.timelines.order_unknown
            or record.ev.origin_status in {"missing_session_id", "missing_entry_id"}
        ):
            return TaskId(
                f"{record.ev.runtime}:native-origin-unknown:{digest_of(record.ev.native_key)[:24]}",
                None,
                "unknown",
            )
        root = self.root(key)
        if key in self.declared:
            cached = self._tasks.get(key)
            if cached is None:
                link = self.launch.get(key)
                if key in self.broken or link is None:
                    cached = TaskId(root, None, "main")
                else:
                    cached = self.task_of(link.parent, link.call_seq)
                self._tasks[key] = cached
            return cached
        line = self.timelines.sessions[key]
        context_turn = record.ev.payload.get("originating_turn_record_id")
        if isinstance(context_turn, str):
            return TaskId(root, context_turn, "main")
        if "originating_turn_record_id" in record.ev.payload:
            return TaskId(root, None, "unknown")
        turn: Record | None = None
        for candidate in line.turns:
            if candidate.seq < seq:
                turn = candidate
            else:
                break
        return TaskId(root, turn.ev.record_id if turn else None, "main")

    def context(self, key: SessionKey, seq: int) -> tuple[list[tuple[SessionKey, int]], list[str]]:
        """Earlier available context: ``[(session, upto_seq), ...]`` plus limitation reasons."""
        assert self.timelines is not None
        record = self.timelines.sessions[key].records[seq]
        if record.ev.origin_status in {"missing_session_id", "missing_entry_id"}:
            return [], ["native_identity_unavailable"]
        if record.identity in self.timelines.ownership_unknown:
            return [], ["native_origin_ownership_unknown"]
        if record.identity in self.timelines.order_unknown:
            return [], ["native_origin_order_unknown"]
        chain: list[tuple[SessionKey, int]] = []
        reasons: list[str] = []
        turn_id = record.ev.payload.get("originating_turn_record_id")
        if record.ev.payload.get("baseline_prior_context_unavailable") and (
            not isinstance(turn_id, str) or turn_id not in self.timelines.sessions[key].by_id
        ):
            reasons.append("baseline_prior_context_unavailable")
        seen: set[SessionKey] = set()
        current: SessionKey | None = key
        upto = seq
        while current is not None and current not in seen:
            seen.add(current)
            chain.append((current, upto))
            if current not in self.declared:
                break
            link = self.launch.get(current)
            if current in self.broken:
                reasons.append("missing_or_cyclic_ancestor_context")
                break
            if link is None:
                reasons.append("launch_context_unknown")
                break
            current, upto = link.parent, link.call_seq + 1  # the launching call text counts
        return chain, reasons


def _norm(path: str) -> str:
    return os.path.normpath(path)


def _declared_parent(line: Timeline) -> tuple[str | None, str | None, str | None]:
    """``(session_id, path, opaque)`` of the declared parent from one session's evidence."""
    for record in line.records:
        if line.is_conflicted(record):
            continue
        payload = record.ev.payload
        context = payload.get("session_context")
        if isinstance(context, dict):
            pid, path = context.get("parent_session_id"), context.get("parent_session_path")
            if isinstance(pid, str) or isinstance(path, str):
                return (
                    pid if isinstance(pid, str) else None,
                    path if isinstance(path, str) else None,
                    None,
                )
        if record.kind == "session_header":
            pid = payload.get("parent_session_id") or payload.get("parent_id")
            path = payload.get("parent_session_path")
            if pid or path:
                return pid, path, None
        parent = payload.get("parent_session")
        if isinstance(parent, str) and parent:
            return parent, None, None
    return None, None, None


def _source_index(timelines: Timelines) -> dict[tuple[str, str], list[SessionKey]]:
    found: dict[tuple[str, str], list[SessionKey]] = {}
    for key, line in timelines.sessions.items():
        # A result may be logically paired elsewhere; its wrapper is not that call's origin.
        sources = {r.ev.source_id for r in line.records if r.ev.session_id == key[1]}
        for source in sources:
            found.setdefault((key[0], _norm(source.split(":", 1)[1])), []).append(key)
    return found


def _resolve_parents(timelines: Timelines, out: Lineage) -> None:
    by_source = _source_index(timelines)
    for key, line in timelines.sessions.items():
        pid, path, _ = _declared_parent(line)
        if pid is None and path is None:
            continue
        out.declared[key] = path or pid or ""
        parent: SessionKey | None = None
        if path:
            hits = by_source.get((key[0], _norm(path)), [])
            parent = hits[0] if len(hits) == 1 else None
        elif pid:
            candidate = (key[0], pid)
            parent = candidate if candidate in timelines.sessions else None
        if parent is not None and parent != key:
            out.parent_of[key] = parent
    for row in timelines.lineage:  # prime ``rlm_subagent`` rows: child file -> parent id
        if row.native_identity in timelines.conflicted:
            continue
        payload = row.payload
        child_file, parent_id = payload.get("session_file"), payload.get("parent_session_id")
        if not (isinstance(child_file, str) and isinstance(parent_id, str)):
            continue
        hits = by_source.get((row.runtime, _norm(child_file)), [])
        parent_key = (row.runtime, parent_id)
        if len(hits) == 1 and parent_key in timelines.sessions and hits[0] != parent_key:
            child = hits[0]
            out.declared.setdefault(child, parent_id)
            out.parent_of.setdefault(child, parent_key)
            child_id = payload.get("child_id")
            if isinstance(child_id, str):
                out.names.setdefault(child, set()).add(child_id)
        else:
            out.issues.append(Issue("unresolved_lineage_row", f"{row.runtime}:{parent_id}"))


def _roots(timelines: Timelines, out: Lineage) -> None:
    for key in sorted(timelines.sessions):
        trail: list[SessionKey] = []
        position: dict[SessionKey, int] = {}
        current = key
        broken = False
        while True:
            if current in out.root_of:
                root = out.root_of[current]
                broken = current in out.broken
                break
            if current in position:  # cycle: all members share the smallest member as root
                members = trail[position[current] :]
                root = min(members)
                for member in members:
                    out.issues.append(Issue("cyclic_parent", key_str(member)))
                    out.broken.add(member)
                broken = True
                break
            position[current] = len(trail)
            trail.append(current)
            parent = out.parent_of.get(current)
            if parent is None:
                if current in out.declared:
                    out.issues.append(Issue("missing_parent", key_str(current)))
                    out.broken.add(current)
                    broken = True
                root = current
                break
            current = parent
        for member in trail:
            out.root_of.setdefault(member, root)
            if broken and member in out.declared:
                out.broken.add(member)
    out.issues = sorted(set(out.issues), key=lambda i: (i.kind, i.session))


def _identity_words(key: SessionKey, line: Timeline, extra: set[str]) -> list[str]:
    names: set[str] = set(extra)
    session = key[1]
    names.add(session.rsplit("/", 1)[-1])
    names.add(session)
    for source in line.sources:
        stem = os.path.basename(source.split(":", 1)[1])
        stem = stem[: -len(".jsonl")] if stem.endswith(".jsonl") else stem
        names.add(stem[len("agent-") :] if stem.startswith("agent-") else stem)
        if "_" in stem:
            names.add(stem.rsplit("_", 1)[-1])
    return sorted(n for n in names if len(n) >= 4 and n.lower() not in GENERIC_NAMES)


def _first_words(line: Timeline) -> dict[str, int]:
    """Earliest call/result sequence at which each word appears (one pass per parent)."""
    first: dict[str, int] = {}
    for record in line.records:
        if (
            line.is_conflicted(record)
            or record.kind == "tool_call"
            and line.has_conflicted_result(record)
            or record.kind == "tool_result"
            and line.has_conflicted_call(record)
        ):
            continue
        if record.kind not in ("tool_call", "tool_result"):
            continue
        for word in {w.rstrip(".-") for w in WORD.findall(record_text(record))}:
            first.setdefault(word, record.seq)
    return first


def _link_launches(timelines: Timelines, out: Lineage) -> None:
    index: dict[SessionKey, dict[str, int]] = {}
    for child, parent in sorted(out.parent_of.items()):
        if child in out.broken:
            continue
        line, pline = timelines.sessions[child], timelines.sessions[parent]
        words = index.get(parent)
        if words is None:
            words = index[parent] = _first_words(pline)
        best: tuple[int, str] | None = None
        for name in _identity_words(child, line, out.names.get(child, set())):
            seq = words.get(name)
            if seq is not None and (best is None or seq < best[0]):
                best = (seq, name)
        if best is None:
            out.issues.append(Issue("unlinked_launch", key_str(child)))
            continue
        record = pline.records[best[0]]
        if record.kind == "tool_result":  # the launching call is the one this result answers
            call = pline.call_for(record)
            if call is None:
                out.issues.append(Issue("unlinked_launch", key_str(child)))
                continue
            record = call
        first_ts = next((r.ts for r in line.records if r.ts is not None), None)
        if record.ts is not None and first_ts is not None and first_ts + LAUNCH_SKEW_MS < record.ts:
            out.issues.append(Issue("unlinked_launch", key_str(child)))  # child predates launch
            continue
        out.launch[child] = Launch(parent, record.seq, best[1])


def build_lineage(timelines: Timelines) -> Lineage:
    """Resolve roots, parents, and launch links; surfaces every defect in ``issues``."""
    out = Lineage(timelines=timelines)
    _resolve_parents(timelines, out)
    _roots(timelines, out)
    _link_launches(timelines, out)
    for identity in sorted(timelines.ownership_unknown):
        out.issues.append(Issue("native_origin_ownership_unknown", identity))
    return out


class PiReader(PiFamilyReader):
    runtime: ClassVar[str] = "pi"

    def roots(self) -> list[Path]:
        return [roots.pi_sessions(self.env)]


class OmpReader(PiFamilyReader):
    runtime: ClassVar[str] = "omp"

    def roots(self) -> list[Path]:
        return [roots.omp_sessions(self.env)]


class PrimeReader(PiFamilyReader):
    runtime: ClassVar[str] = "prime-agent"

    def roots(self) -> list[Path]:
        return [roots.prime_sessions(self.env), roots.prime_artifacts(self.env)]

    def role_for(self, path: Path) -> str:
        artifacts = roots.prime_artifacts(self.env).resolve()
        try:
            path.relative_to(artifacts)
        except ValueError:
            return "main"
        return "lineage" if path.name == "rlm-subagents.jsonl" else "artifact"
