"""Shared limits and record-type validation for Claude JSONL."""

from __future__ import annotations

from dataclasses import dataclass

MAX_LINE_BYTES = 500_000

_KNOWN_TYPES = {
    "user",
    "assistant",
    "system",
    "queue-operation",
    "permission-mode",
    "mode",
    "last-prompt",
    "attachment",
    "agent-name",
    "custom-title",
    "ai-title",
    "file-history-snapshot",
    "bridge-session",
    "agent-setting",
    "pr-link",
}


@dataclass
class Stats:
    skipped_oversized: int = 0
    malformed_lines: int = 0
    unknown_types: int = 0


def is_known_record(record: dict) -> bool:
    return record.get("type") in _KNOWN_TYPES
