"""Claude native records: ownership is local; verified record UUIDs identify copies.

Provider message IDs group streaming usage only within their provider namespace.
Opaque/local tool IDs alone never prove a resumed-history origin across sessions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.ledger.contracts import RawEvidence
from ssgrep.accounting.readers import classify as roots
from ssgrep.accounting.readers.base import (
    JsonlReader,
    Source,
    as_int,
    loads,
    make_evidence,
    pick_ints,
    text_of,
    to_utc,
)
from ssgrep.accounting.readers.native_identity import (
    collapse,
    link_result,
    native_uuid,
    remember_call,
    with_origin,
)

USAGE_NAMES = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
    "cache_creation_input_tokens": "cache_write_tokens",
}
KNOWN_NON_EVIDENCE = frozenset(
    {
        "system",
        "summary",
        "file-history-snapshot",
        "queue-operation",
        "attachment",
        "progress",
        "last-prompt",
        "ai-title",
        "custom-title",
        "permission-mode",
        "agent-name",
        "pr-link",
    }
)


class ClaudeReader(JsonlReader):
    runtime: ClassVar[str] = "claude"
    skip_dirs: ClassVar[frozenset[str]] = frozenset({"tool-results", "memory"})

    def finish_cycle(self, evidence: list[RawEvidence]) -> list[RawEvidence]:
        return [
            with_origin(item, item.origin_domain, item.origin_id, item.origin_status)
            if item.origin_domain is not None and item.origin_id is not None
            else item
            for item in collapse(evidence)
        ]

    def roots(self) -> list[Path]:
        return [roots.claude_projects(self.env)]

    def role_for(self, path: Path) -> str:
        return "child" if "subagents" in path.parts else "main"

    def accepts(self, path: Path) -> bool:
        parts = path.parts
        return "workflows" not in parts or "subagents" in parts

    @staticmethod
    def agent_id(path: Path) -> str:
        stem = path.stem
        return stem[len("agent-") :] if stem.startswith("agent-") else stem

    def _identity(self, source: Source, obj: dict) -> tuple[str, str, dict[str, Any]]:
        session = obj.get("sessionId")
        session = session if isinstance(session, str) and session else source.path.stem
        if source.role == "child":
            child = self.agent_id(source.path)
            return f"{session}/{child}", f"child:{child}", {"parent_session": session}
        return session, "main", {}

    def decode(
        self, source: Source, obj: Any, ctx: dict[str, Any], locator: str
    ) -> list[RawEvidence] | None:
        if not isinstance(obj, dict):
            return None
        kind = obj.get("type")
        if not isinstance(kind, str):
            return None
        if kind in KNOWN_NON_EVIDENCE:
            return []
        message = obj.get("message")
        if kind not in ("user", "assistant") or not isinstance(message, dict):
            return None
        session, branch, extra = self._identity(source, obj)
        cwd = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
        ts = to_utc(obj.get("timestamp"))
        uuid = native_uuid(obj.get("uuid"))
        native_session = isinstance(obj.get("sessionId"), str) and bool(obj["sessionId"])
        domain = session if native_session else source.source_id
        scoped_status = "session_scoped" if native_session else "missing_session_id"

        def evidence(record_id: str, ekind: str, payload: dict[str, Any]) -> RawEvidence:
            payload = {**payload, **extra, "cwd": cwd}
            local_id = record_id
            if uuid and ekind in {"tool_call", "tool_result"}:
                local_id = f"{record_id}@{uuid}"
            if ekind == "tool_call":
                remember_call(ctx, payload["call_id"], local_id, uuid or locator)
            elif ekind == "tool_result":
                linked, status = link_result(ctx, payload["call_id"])
                payload = {**payload, "call_record_id": linked, "call_link_status": status}
            item = make_evidence(
                runtime="claude",
                source_id=source.source_id,
                branch_id=branch,
                session_id=session,
                record_id=local_id,
                locator=locator,
                kind=ekind,
                ts=ts,
                payload=payload,
            )
            if ekind == "usage":
                message_id = payload.get("message_id")
                # Anthropic's request-generated message namespace, not local fixture IDs.
                if (
                    uuid
                    and isinstance(message_id, str)
                    and re.fullmatch(r"msg_[A-Za-z0-9]{20,}", message_id)
                ):
                    return with_origin(
                        item, "claude_provider_message_id", record_id, "provider_message_id"
                    )
                return with_origin(item, domain, record_id, scoped_status)
            if uuid:
                return with_origin(
                    item, "claude_record_uuid", f"{uuid}:{record_id}", "confirmed_uuid"
                )
            return with_origin(item, domain, record_id, scoped_status)

        content = message.get("content")
        out: list[RawEvidence] = []
        if kind == "assistant":
            out.extend(self._assistant(message, cwd, evidence))
            return out
        blocks = content if isinstance(content, list) else []
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        for block in results:
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str):
                continue
            body = text_of(block.get("content"))
            flag = block.get("is_error")
            out.append(
                evidence(
                    f"result:{call_id}",
                    "tool_result",
                    {
                        "call_id": call_id,
                        "output": body,
                        "output_bytes": len(body.encode()),
                        "is_error": flag if isinstance(flag, bool) else None,
                    },
                )
            )
        if not results and obj.get("isMeta") is not True:
            prompt = text_of(content)
            if prompt:
                turn_id = uuid or str(obj.get("uuid") or locator)
                out.append(evidence(f"turn:{turn_id}", "user_turn", {"text": prompt, "cwd": cwd}))
        return out

    @staticmethod
    def _assistant(message: dict, cwd: str | None, evidence) -> list[RawEvidence]:
        out: list[RawEvidence] = []
        message_id = message.get("id") if isinstance(message.get("id"), str) else None
        usage = message.get("usage")
        if isinstance(usage, dict) and message_id:
            tokens = pick_ints(usage, USAGE_NAMES)
            reasoning = pick_ints(usage.get("output_tokens_details"), {"thinking_tokens": "r"})
            if "r" in reasoning:
                tokens["reasoning_tokens"] = reasoning["r"]
            out.append(
                evidence(
                    f"usage:{message_id}",
                    "usage",
                    {
                        "message_id": message_id,
                        "model": message.get("model"),
                        "tokens": tokens,
                        "streaming_updates": 1,
                    },
                )
            )
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_id, name = block.get("id"), block.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                continue
            out.append(
                evidence(
                    f"call:{call_id}",
                    "tool_call",
                    {
                        "call_id": call_id,
                        "name": name,
                        "arguments": block.get("input"),
                        "message_id": message_id,
                        "cwd": cwd,
                    },
                )
            )
        return out


CATEGORIES = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
IGNORED_RECORDS = frozenset(
    {"world_state", "compacted", "token_usage_record", "turn_context_delta", "event_msg"}
)
IGNORED_ITEMS = frozenset({"reasoning", "local_shell_call", "web_search_call", "ghost_snapshot"})
CALLS = {"function_call", "custom_tool_call"}
OUTPUTS = {"function_call_output", "custom_tool_call_output"}


def _categories(info: Any) -> dict[str, int] | None:
    usage = info.get("total_token_usage") if isinstance(info, dict) else None
    if not isinstance(usage, dict):
        return None
    found = {name: n for name in CATEGORIES if (n := as_int(usage.get(name))) is not None}
    return found or None


def total_usage(evidence: list[RawEvidence]) -> dict[str, int]:
    """Normalized codex usage: the sum of non-negative per-update deltas."""
    totals: dict[str, int] = {}
    for item in evidence:
        delta = item.payload.get("delta") if item.kind == "usage" else None
        for name, value in (delta or {}).items():
            totals[name] = totals.get(name, 0) + value
    return totals


class CodexReader(JsonlReader):
    runtime: ClassVar[str] = "codex"

    def roots(self) -> list[Path]:
        return [roots.codex_sessions(self.env)]

    def decode(
        self, source: Source, obj: Any, ctx: dict[str, Any], locator: str
    ) -> list[RawEvidence] | None:
        if not isinstance(obj, dict):
            return None
        kind = obj.get("type")
        if not isinstance(kind, str):
            return None
        payload = obj.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        if kind == "session_meta":
            self._meta(ctx, payload)
            return []
        if kind == "turn_context":
            ctx["turn"] = payload.get("turn_id") or ctx.get("turn")
            ctx["cwd"] = payload.get("cwd") or ctx.get("cwd")
            return []
        session = ctx.get("session") or source.path.stem
        ts = to_utc(obj.get("timestamp"))

        def evidence(record_id: str, ekind: str, body: dict[str, Any]) -> RawEvidence:
            if ctx.get("parent"):
                body = {**body, "parent_session": ctx["parent"]}
            return make_evidence(
                runtime="codex",
                source_id=source.source_id,
                branch_id="main",
                session_id=session,
                record_id=record_id,
                locator=locator,
                kind=ekind,
                ts=ts,
                payload=body,
            )

        item = payload.get("type")
        if kind == "event_msg" and item == "token_count":
            return [self._usage(ctx, payload, evidence)]
        if kind == "event_msg" and item == "task_started":
            ctx["turn"] = payload.get("turn_id") or ctx.get("turn")
            return []
        if kind != "response_item":
            return [] if kind in IGNORED_RECORDS else None
        if not isinstance(item, str):
            return None
        if item in CALLS:
            return [self._call(ctx, payload, evidence)]
        if item in OUTPUTS:
            return self._output(payload, evidence)
        if item == "message":
            return self._message(ctx, payload, evidence)
        return [] if item in IGNORED_ITEMS else None

    @staticmethod
    def _meta(ctx: dict[str, Any], payload: dict[str, Any]) -> None:
        # A forked rollout replays its parent's session_meta after its own. Only the first
        # meta owns the file: later ones are embedded parent metadata, not a new session.
        if ctx.get("session"):
            return
        session = payload.get("id") or payload.get("session_id")
        if isinstance(session, str):
            ctx["session"] = session
        ctx["cwd"] = payload.get("cwd") if isinstance(payload.get("cwd"), str) else None
        ctx.pop("prev", None)
        ctx["updates"] = 0
        spawn = (payload.get("source") or {}) if isinstance(payload.get("source"), dict) else {}
        subagent = spawn.get("subagent")
        thread = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
        parent = thread.get("parent_thread_id") if isinstance(thread, dict) else None
        ctx["parent"] = parent if isinstance(parent, str) else None

    @staticmethod
    def _call(ctx: dict[str, Any], payload: dict[str, Any], evidence) -> RawEvidence:
        call_id = str(payload.get("call_id") or payload.get("id") or "")
        raw = payload.get("arguments", payload.get("input"))
        arguments: Any = raw
        if payload.get("type") == "function_call" and isinstance(raw, str):
            try:
                arguments = loads(raw)
            except ValueError:
                arguments = raw
        body = {
            "call_id": call_id,
            "name": payload.get("name"),
            "arguments": arguments,
            "turn_id": ctx.get("turn"),
            "cwd": ctx.get("cwd"),
        }
        if isinstance(payload.get("namespace"), str):
            body["namespace"] = payload["namespace"]
        return evidence(f"call:{call_id}", "tool_call", body)

    @staticmethod
    def _output(payload: dict[str, Any], evidence) -> list[RawEvidence]:
        call_id = str(payload.get("call_id") or "")
        text = text_of(payload.get("output"))
        body = {
            "call_id": call_id,
            "output": text,
            "output_bytes": len(text.encode()),
            "is_error": None,
        }
        return [evidence(f"result:{call_id}", "tool_result", body)]

    @staticmethod
    def _message(ctx: dict[str, Any], payload: dict[str, Any], evidence) -> list[RawEvidence]:
        if payload.get("role") != "user":
            return []
        text = text_of(payload.get("content"))
        if not text:
            return []
        key = payload.get("id")
        if not isinstance(key, str):
            ctx["turns"] = int(ctx.get("turns", 0)) + 1
            key = f"n{ctx['turns']}"
        return [evidence(f"turn:{key}", "user_turn", {"text": text})]

    @staticmethod
    def _usage(ctx: dict[str, Any], payload: dict[str, Any], evidence) -> RawEvidence:
        turn = ctx.get("turn") or "?"
        ctx["updates"] = ordinal = int(ctx.get("updates", 0)) + 1
        cumulative = _categories(payload.get("info"))
        if cumulative is None:
            body = {
                "turn_id": turn,
                "counter": "total_token_usage",
                "cumulative": None,
                "delta": None,
                "reset": False,
                "unavailable": "missing_counters",
            }
            return evidence(f"usage:{ordinal}", "usage", body)
        previous = ctx.get("prev") or {}
        delta: dict[str, int] = {}
        reset = False
        for name, value in cumulative.items():
            before = previous.get(name, 0)
            if value < before:
                reset = True
                delta[name] = value
            else:
                delta[name] = value - before
        ctx["prev"] = cumulative
        body = {
            "turn_id": turn,
            "counter": "total_token_usage",
            "cumulative": cumulative,
            "delta": delta,
            "reset": reset,
        }
        return evidence(f"usage:{ordinal}", "usage", body)
