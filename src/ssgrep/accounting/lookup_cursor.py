"""Lookup cursors retain snapshot identity; expired generations cannot become live pages."""

from __future__ import annotations

import base64
import json

from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.service_cursor import issue, redeem


def reference(cursor: str | None, command: str):
    if cursor is None:
        return None
    try:
        if not isinstance(cursor, str) or len(cursor) > 2048 or not cursor.startswith("lookups1."):
            raise ValueError("invalid cursor")
        payload = base64.urlsafe_b64decode(cursor[9:].encode() + b"=" * (-len(cursor[9:]) % 4))
        raw = json.loads(payload)
        if set(raw) != {"snapshot", "generation", "page"}:
            raise ValueError("invalid fields")
        if not isinstance(raw["snapshot"], str) or len(raw["snapshot"]) > 512:
            raise ValueError("invalid snapshot")
        if isinstance(raw["generation"], bool) or not isinstance(raw["generation"], int):
            raise ValueError("invalid generation")
        if not isinstance(raw["page"], str):
            raise ValueError("invalid page")
        return raw
    except (ValueError, TypeError, UnicodeError, KeyError) as error:
        raise PolicyError(
            "invalid_cursor", "That is not a lookup cursor. Start without --cursor.", command, 2
        ) from error


def offset(cursor, binding: str, command: str) -> int:
    return redeem("l", cursor["page"] if cursor is not None else None, binding, command)


def next_cursor(offset_: int, binding: str, snapshot: str, generation: int) -> str:
    raw = {"snapshot": snapshot, "generation": generation, "page": issue("l", offset_, binding)}
    return "lookups1." + base64.urlsafe_b64encode(canonical_bytes(raw)).decode().rstrip("=")
