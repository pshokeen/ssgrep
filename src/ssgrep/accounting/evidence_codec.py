"""Versioned minimum retained evidence; plaintext is never the default format."""

from __future__ import annotations

import json
from typing import Any

from ssgrep.accounting.contracts import SCHEMA_VERSION, RawEvidence

COMPACT = "compact-v1"
PLAINTEXT = "plaintext-cache-v1"
LEGACY = "legacy-plaintext"
# Native identifiers, ownership and interpreted outcomes, not tool arguments or content.
_IDENTIFIERS = frozenset(
    {
        "cwd",
        "directory",
        "call_id",
        "call_record_id",
        "call_link_status",
        "turn_id",
        "parent_session",
        "parent_session_id",
        "parent_id",
        "parent_session_path",
        "session_file",
        "child_id",
        "id",
        "message_id",
        "session_id",
        "status",
        "reason",
        "original_kind",
        "source_id",
        "branch_id",
        "parent",
        "claimant_source_id",
        "claimant_session_id",
        "claimant_branch_id",
        "claimant_cwd",
        "claimant_parent_session",
        "model",
        "provider",
        "counter",
        "unavailable",
        "target_digest",
        "target_record_id",
        "originating_turn_record_id",
        "task_context_basis",
    }
)
_NUMBERS = frozenset(
    {
        "exit_code",
        "is_error",
        "duration_ms",
        "duration_ns",
        "time_updated",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "updates",
        "stream_updates",
        "streaming_updates",
        "rlm_depth",
        "version",
        "reset",
        "occurrence",
        "completed_ms",
        "origin_conflict",
        "baseline_prior_context_unavailable",
    }
)


def _measurements(value: Any) -> Any:
    if value is None or isinstance(value, (int, bool)):
        return value
    names = {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "input",
        "output",
        "cacheRead",
        "cacheWrite",
        "total",
    }
    if isinstance(value, dict):
        return {
            k: v
            for k, v in value.items()
            if k in names and isinstance(v, int) and not isinstance(v, bool)
        }
    return None


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        k: v
        for k, v in payload.items()
        if k in _IDENTIFIERS and (v is None or isinstance(v, str) and len(v) <= 4096)
    }
    out.update(
        {
            k: v
            for k, v in payload.items()
            if k in _NUMBERS and (v is None or isinstance(v, (int, bool)))
        }
    )
    for key in ("tokens", "delta", "usage", "cumulative", "cost_nano_usd"):
        if key in payload:
            out[key] = _measurements(payload[key])
    if isinstance(payload.get("session_context"), dict):
        out["session_context"] = {
            k: v
            for k, v in compact_payload(payload["session_context"]).items()
            if k != "_evidence_format"
        }
    target = payload.get("target_native_key")
    if (
        isinstance(target, (list, tuple))
        and len(target) == 3
        and all(isinstance(v, str) for v in target)
    ):
        out["target_native_key"] = list(target)
    out["_evidence_format"] = COMPACT
    return out


def evidence_line(item: RawEvidence, *, plaintext: bool = False) -> bytes:
    raw = {
        name: getattr(item, name)
        for name in (
            "runtime",
            "source_id",
            "branch_id",
            "session_id",
            "record_id",
            "locator",
            "digest",
            "ts_utc",
            "kind",
            "schema",
            "origin_domain",
            "origin_id",
            "origin_status",
        )
    }
    raw["evidence_format"] = PLAINTEXT if plaintext else COMPACT
    raw["payload"] = item.payload if plaintext else compact_payload(item.payload)
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def evidence_from_line(line: bytes) -> RawEvidence:
    from ssgrep.accounting.collector_store import loads

    raw = loads(line)
    form = raw.get("evidence_format", LEGACY)
    if form not in {COMPACT, PLAINTEXT, LEGACY}:
        raise ValueError("unsupported retained evidence format")
    payload = dict(raw["payload"])
    if form == COMPACT:
        payload = compact_payload(payload)
    else:
        payload["_evidence_format"] = form
    return RawEvidence(
        runtime=str(raw["runtime"]),
        source_id=str(raw["source_id"]),
        branch_id=str(raw["branch_id"]),
        session_id=str(raw["session_id"]),
        record_id=str(raw["record_id"]),
        locator=str(raw["locator"]),
        digest=str(raw["digest"]),
        ts_utc=raw["ts_utc"],
        kind=str(raw["kind"]),
        payload=payload,
        schema=int(raw.get("schema", SCHEMA_VERSION)),
        origin_domain=raw.get("origin_domain"),
        origin_id=raw.get("origin_id"),
        origin_status=str(raw.get("origin_status", "legacy_unverified"))
        if form != LEGACY
        else "legacy_unverified",
    )
