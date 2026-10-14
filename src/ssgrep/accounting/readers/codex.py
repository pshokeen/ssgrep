"""Codex rollouts: ``sessions/**/rollout-*.jsonl``.

``token_count`` events carry a *cumulative* ``total_token_usage``. Summing them
double counts (100, 150, 150 is 150, not 400), so each update is differenced
against the previous cumulative value of the same session. A counter that goes
backwards (a restarted window) is flagged ``reset`` and its new value is taken
as fresh usage, so no delta is ever negative; absent counters are explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.base import JsonlReader, Source
from ssgrep.accounting.readers.values import as_int, loads, make_evidence, text_of, to_utc

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
