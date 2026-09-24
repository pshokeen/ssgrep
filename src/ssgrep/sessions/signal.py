"""Signal/noise classification for transcript records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ssgrep.utilities.types import ContentType


@dataclass
class SignalResult:
    is_signal: bool
    text: str = ""
    content_type: ContentType | None = None


def classify_signal(record: dict, block: dict | None = None) -> SignalResult:
    # Check for isMeta records - exclude them from signal extraction
    if record.get("isMeta"):
        return SignalResult(False)

    rec_type = record.get("type", "")
    subtype = record.get("subtype", "")

    if rec_type == "user":
        if block and block.get("type") == "text":
            text = strip_markup(block.get("text", ""))
            return SignalResult(True, text, ContentType.PROMPT)
        if block and block.get("type") == "tool_result":
            return SignalResult(False)
        if block and block.get("type") == "image":
            return SignalResult(False)

    elif rec_type == "assistant":
        if block and block.get("type") == "thinking":
            return SignalResult(False)
        if block and block.get("type") == "text":
            text = strip_markup(block.get("text", ""))
            return SignalResult(True, text, ContentType.RESPONSE)
        if block and block.get("type") == "tool_use":
            return SignalResult(False)

    elif rec_type == "system":
        if subtype == "away_summary":
            # away_summary records carry content as a top-level string field
            content = record.get("content", "")
            text = strip_markup(content) if isinstance(content, str) else ""
            return SignalResult(True, text, ContentType.RESPONSE)
        if subtype == "local_command":
            return SignalResult(False)

    return SignalResult(False)


def strip_markup(text: str) -> str:
    text = re.sub(r"<command-name>.*?</command-name>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-message>.*?</command-message>", "", text, flags=re.DOTALL)
    return text.strip()
