"""native_identity_v2: explicit native domains, never output-text equivalence.

Domain                 Copy equivalence                  Non-equivalence / uncertainty
Claude record UUID     RFC UUID v4/v7 + semantic sub-ID   New UUID; local tool ID alone
Claude provider usage  API-shaped message ID + native UUID  Local/short message ID stays scoped
Pi / omp / Prime       native session + entry + sub-ID    Entry IDs are NOT globally unique
Missing session        source path + native entry         Cross-path copy unconfirmed
Missing entry          source path + byte offset          Rewrites/copies unconfirmed
Codex / OpenCode       existing session + native record   No broader domain asserted here

A same-domain same-ID changed digest is a conflict, not another event. Source,
branch, cwd and parent ownership remain on RawEvidence, not in the origin key.
UUID shape verifies a supported namespace, not authenticity or permission.
"""

from __future__ import annotations

from dataclasses import replace
from uuid import RFC_4122, UUID

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.readers.values import evidence_digest

IDENTITY_METHOD = "native_identity_v2"
CONTEXT_BYTES = 64 * 1024
MAX_PENDING_CALLS = 256


def native_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = UUID(value)
    except ValueError:
        return None
    if parsed.variant != RFC_4122 or parsed.version not in {4, 7}:
        return None
    return str(parsed)


def with_origin(item: RawEvidence, domain: str, origin: str, status: str) -> RawEvidence:
    # These are reconstructed linkage/ownership, not native content variants.
    skip = {
        "call_record_id",
        "call_link_status",
        "occurrence",
        "originating_turn_record_id",
        "task_context_basis",
        "baseline_prior_context_unavailable",
    }
    if status in {"confirmed_uuid", "provider_message_id"}:
        skip.update({"parent_session", "cwd"})
    content = {k: v for k, v in item.payload.items() if k not in skip}
    return replace(
        item,
        origin_domain=domain,
        origin_id=origin,
        origin_status=status,
        digest=evidence_digest(item.kind, content),
    )


def remember_call(ctx: dict, call_id: str, record_id: str, entry_id: str) -> None:
    table = ctx.setdefault("pending_calls", {})
    pending = table.setdefault(call_id, [])
    pair = [record_id, entry_id]
    if pair not in pending:
        pending.append(pair)
    if sum(len(v) for v in table.values()) > MAX_PENDING_CALLS:
        # Losing context must not turn a formerly ambiguous link into a unique one.
        table.clear()
        ctx["call_context_incomplete"] = True


def link_result(ctx: dict, call_id: str, parent_id: object = None) -> tuple[str | None, str]:
    table = ctx.setdefault("pending_calls", {})
    pending = table.get(call_id, [])
    linked = [p for p in pending if p[1] == parent_id] if isinstance(parent_id, str) else []
    candidates = linked or pending
    if len(candidates) == 1 and (linked or not ctx.get("call_context_incomplete")):
        chosen = candidates[0]
        pending.remove(chosen)
        if not pending:
            table.pop(call_id, None)
        return chosen[0], "native_parent" if linked else "unique_pending"
    return None, "ambiguous" if len(candidates) > 1 else "missing_context"
