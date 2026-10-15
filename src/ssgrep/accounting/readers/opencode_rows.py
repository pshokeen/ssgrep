"""Atomic OpenCode row decoding, including evidence normalization/construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.readers.base import Gap, ReadResult, Source
from ssgrep.accounting.readers.values import as_int, loads, nano_usd, obj, pick_ints

if TYPE_CHECKING:
    from ssgrep.accounting.readers.opencode import OpenCodeReader


TOKEN_NAMES = {
    "total": "total_tokens",
    "input": "input_tokens",
    "output": "output_tokens",
    "reasoning": "reasoning_tokens",
}


def read_row(reader, source, table, row, held, result):
    """Reject one unrepresentable row, never its following rows or earlier evidence.

    Input bytes and fresh watermarks are owned by the cursor before this call.
    Roll back partial emissions/held IDs if any later normalization stage exceeds
    Python's recursion budget; no half-call can survive as accepted evidence.
    """
    evidence_start, held_start, gap_start = len(result.evidence), len(held), len(result.gaps)
    try:
        decode_row(reader, source, table, row, held, result)
    except RecursionError:
        del result.evidence[evidence_start:]
        del held[held_start:]
        del result.gaps[gap_start:]
        result.gaps.append(
            Gap(source.source_id, "normalization_recursion", int(row["n"]), f"{table}:{row['id']}")
        )


def decode_row(reader, source, table, row, held, result):
    locator = f"{table}:{row['id']}"
    if table == "session":
        result.evidence.append(session_evidence(reader, source, row, locator))
        return
    if row["data"] is None:
        result.gaps.append(Gap(source.source_id, "oversized_record", int(row["n"]), locator))
        return
    try:
        data = loads(row["data"])
    except (ValueError, RecursionError):
        result.gaps.append(Gap(source.source_id, "malformed_json", int(row["n"]), locator))
        return
    if not isinstance(data, dict):
        result.unrecognized["<non-object>"] = result.unrecognized.get("<non-object>", 0) + 1
        result.gaps.append(Gap(source.source_id, "unsupported_record", int(row["n"]), locator))
        return
    if table == "message":
        read_message(reader, source, row, data, locator, held, result)
    else:
        reader._part(source, row, data, locator, held, result)


def session_evidence(
    reader: OpenCodeReader, source: Source, row: Mapping[str, Any], locator: str
) -> RawEvidence:
    parent = row["parent_id"]
    body = {
        "id": row["id"],
        "parent_id": parent,
        "directory": row["directory"],
        "project_id": row["project_id"],
    }
    branch = "child" if parent else "main"
    return reader._evidence(
        source, row["id"], f"session:{row['id']}", "session_header", locator, None, body, branch
    )


def read_message(
    reader: OpenCodeReader,
    source: Source,
    row: Mapping[str, Any],
    data: dict,
    locator: str,
    held: list[str],
    result: ReadResult,
) -> None:
    if data.get("role") != "assistant":
        return
    stamps = obj(data.get("time"))
    if not (stamps.get("completed") or data.get("finish") or data.get("error")):
        held.append(row["id"])
        result.evidence.append(
            reader._evidence(
                source,
                row["session_id"],
                f"pending-message:{row['id']}",
                "pending_message",
                locator,
                stamps.get("created"),
                {
                    "id": row["id"],
                    "message_id": row["id"],
                    "session_id": row["session_id"],
                    "status": "pending",
                },
            )
        )
        return
    tokens = obj(data.get("tokens"))
    counts = pick_ints(tokens, TOKEN_NAMES)
    counts.update(pick_ints(tokens.get("cache"), {"write": "cache_write_tokens"}))
    counts.update(pick_ints(tokens.get("cache"), {"read": "cache_read_tokens"}))
    body = {
        "message_id": row["id"],
        "model": data.get("modelID"),
        "provider": data.get("providerID"),
        "tokens": counts or None,
        "cost_nano_usd": nano_usd(data.get("cost")),
        "completed_ms": as_int(stamps.get("completed")),
    }
    result.evidence.append(
        reader._evidence(
            source,
            row["session_id"],
            f"usage:{row['id']}",
            "usage",
            locator,
            stamps.get("completed") or stamps.get("created"),
            body,
        )
    )
