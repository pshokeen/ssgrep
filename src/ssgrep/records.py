"""Streaming JSONL reader and record classifier."""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

MAX_LINE_BYTES = 500_000


class RecordType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    QUEUE_OPERATION = "queue-operation"
    PERMISSION_MODE = "permission-mode"
    MODE = "mode"
    LAST_PROMPT = "last-prompt"
    ATTACHMENT = "attachment"
    AGENT_NAME = "agent-name"
    CUSTOM_TITLE = "custom-title"
    AI_TITLE = "ai-title"
    FILE_HISTORY_SNAPSHOT = "file-history-snapshot"
    BRIDGE_SESSION = "bridge-session"
    AGENT_SETTING = "agent-setting"
    PR_LINK = "pr-link"
    UNKNOWN = "unknown"


@dataclass
class Stats:
    total_lines: int = 0
    parsed_records: int = 0
    skipped_oversized: int = 0
    malformed_lines: int = 0
    unknown_types: int = 0


@dataclass
class ContentBlock:
    type: str
    text: str = ""
    tool_name: str | None = None
    tool_input: dict | None = None
    signature: str = ""


def classify_record(record: dict) -> RecordType:
    rec_type = record.get("type", "")
    try:
        return RecordType(rec_type)
    except ValueError:
        return RecordType.UNKNOWN


def extract_content_blocks(record: dict) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    message = record.get("message", {})
    content = message.get("content", [])
    if not isinstance(content, list):
        return blocks
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "unknown")
        if block_type == "thinking":
            blocks.append(
                ContentBlock(
                    type="thinking",
                    text=block.get("thinking", ""),
                    signature=block.get("signature", ""),
                )
            )
        elif block_type == "text":
            blocks.append(ContentBlock(type="text", text=block.get("text", "")))
        elif block_type == "tool_use":
            blocks.append(
                ContentBlock(
                    type="tool_use", tool_name=block.get("name"), tool_input=block.get("input")
                )
            )
        elif block_type == "tool_result":
            blocks.append(ContentBlock(type="tool_result", text=str(block.get("content", ""))))
        elif block_type == "image":
            blocks.append(ContentBlock(type="image"))
        else:
            blocks.append(ContentBlock(type="unknown"))
    return blocks


def read_records(path: Path, max_line_bytes: int = MAX_LINE_BYTES) -> Generator[dict, None, Stats]:
    stats = Stats()
    with open(path, errors="replace") as f:
        for line in f:
            stats.total_lines += 1
            line = line.strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > max_line_bytes:
                stats.skipped_oversized += 1
                continue
            try:
                record = json.loads(line)
                stats.parsed_records += 1
                if classify_record(record) == RecordType.UNKNOWN:
                    stats.unknown_types += 1
                    record["_unknown_type"] = True
                yield record
            except json.JSONDecodeError:
                stats.malformed_lines += 1
                continue
    return stats
