"""Bounded native user-turn IDs; no skipped prompt or usage payloads are retained."""

from __future__ import annotations

from typing import Any

MAX_ENTRIES = 256


def observe(obj: dict, ctx: dict[str, Any], locator: str) -> None:
    entry = obj.get("id")
    entry = entry if isinstance(entry, str) and entry else locator
    parent = obj.get("parentId")
    parent = parent if isinstance(parent, str) and parent else None
    message = obj.get("message")
    user = isinstance(message, dict) and message.get("role") == "user"
    table = ctx.setdefault("entry_turns", {})
    if user:
        turn = f"turn:{entry}"
        basis = "native_user_turn"
    elif parent is not None:
        turn = table.get(parent)
        basis = "native_parent_chain" if turn is not None else "native_parent_context_missing"
    else:
        turn = ctx.get("turn_id")
        basis = "bounded_source_order" if turn is not None else "native_turn_context_missing"
    table[entry] = turn
    if len(table) > MAX_ENTRIES:
        table.pop(next(iter(table)))
    ctx["turn_id"], ctx["turn_basis"] = turn, basis


def fields(ctx: dict[str, Any]) -> dict:
    return {
        "originating_turn_record_id": ctx.get("turn_id"),
        "task_context_basis": ctx.get("turn_basis", "native_turn_context_missing"),
        "baseline_prior_context_unavailable": bool(ctx.get("baseline_context")),
    }
