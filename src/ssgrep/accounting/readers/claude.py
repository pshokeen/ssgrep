"""Claude native records: ownership is local; verified record UUIDs identify copies.

Provider message IDs group streaming usage only within their provider namespace.
Opaque/local tool IDs alone never prove a resumed-history origin across sessions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.native_identity import link_result, native_uuid, remember_call, with_origin
from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.base import JsonlReader, Source
from ssgrep.accounting.readers.streaming import collapse
from ssgrep.accounting.readers.values import make_evidence, pick_ints, text_of, to_utc

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
