"""Signal/noise classification for transcript records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ssgrep.types import ContentType


@dataclass
class SignalResult:
    is_signal: bool
    text: str = ""
    content_type: ContentType | None = None
    source_type: str = ""


def is_signal(record: dict, block: dict | None = None) -> bool:
    return classify_signal(record, block).is_signal


def classify_signal(record: dict, block: dict | None = None) -> SignalResult:
    # Check for isMeta records - exclude them from signal extraction
    if record.get("isMeta"):
        return SignalResult(False, source_type="isMeta")

    rec_type = record.get("type", "")
    subtype = record.get("subtype", "")

    if rec_type == "user":
        if block and block.get("type") == "text":
            text = strip_markup(block.get("text", ""))
            return SignalResult(True, text, ContentType.PROMPT, "user/text")
        if block and block.get("type") == "tool_result":
            return SignalResult(False, source_type="user/tool_result")
        if block and block.get("type") == "image":
            return SignalResult(False, source_type="user/image")

    elif rec_type == "assistant":
        if block and block.get("type") == "thinking":
            return SignalResult(False, source_type="assistant/thinking")
        if block and block.get("type") == "text":
            text = strip_markup(block.get("text", ""))
            return SignalResult(True, text, ContentType.RESPONSE, "assistant/text")
        if block and block.get("type") == "tool_use":
            return SignalResult(False, source_type="assistant/tool_use")

    elif rec_type == "system":
        if subtype == "away_summary":
            # away_summary records carry content as a top-level string field
            content = record.get("content", "")
            text = strip_markup(content) if isinstance(content, str) else ""
            return SignalResult(True, text, ContentType.RESPONSE, "system/away_summary")
        if subtype == "local_command":
            return SignalResult(False, source_type="system/local_command")

    return SignalResult(False, source_type=f"{rec_type}/{subtype or 'unknown'}")


def strip_markup(text: str) -> str:
    text = re.sub(r"<command-name>.*?</command-name>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-message>.*?</command-message>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_signal_text(records: list[dict]) -> str:
    parts = []
    for record in records:
        rec_type = record.get("type", "")
        if rec_type in ("user", "assistant"):
            message = record.get("message", {})
            content = message.get("content", [])
            # content can be a string (user prompts) or a list of blocks (assistant responses)
            if isinstance(content, str):
                # For string content, treat it as a single text block
                result = classify_signal(record, {"type": "text", "text": content})
                if result.is_signal and result.text:
                    parts.append(result.text)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        result = classify_signal(record, block)
                        if result.is_signal and result.text:
                            parts.append(result.text)
        elif rec_type == "system":
            result = classify_signal(record)
            if result.is_signal and result.text:
                parts.append(result.text)
    return "\n".join(parts)
