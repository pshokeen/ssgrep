"""Episode metadata harvesting without LLM calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def normalize_title(raw_title: str) -> str:
    """Normalize a title by stripping markup, removing control chars, and collapsing whitespace.

    Handles harness-injected wrapper tags generically: any <tagname>content</tagname>
    pattern (including tags with hyphens, underscores, and backslashes) is removed.
    This handles known tags (<command-name>, <local-command-stdout>, <system-reminder>,
    <\teammate-message>, etc.) and automatically handles any future wrapper tags.

    Returns empty string if the title is empty or only whitespace after normalization.
    This allows the precedence chain to skip harness-injected content and fall through
    to the next title source.
    """
    if not isinstance(raw_title, str):
        return ""

    # Cap the input before the backtracking tag regexes below: titles render
    # at a fraction of this anyway, and an adversarial repeated-tag string
    # (e.g. '<a>' * 40000) otherwise costs seconds of quadratic backtracking
    # on every call site.
    cleaned = raw_title[:2000]

    # Strip all XML-like wrapper tags: <tagname>content</tagname>
    # Also strip orphaned opening tags like <system-reminder> without closing tags
    # Pattern handles:
    # - Tags with word chars and hyphens: command-name, local-command-stdout, etc.
    # - Tags with optional backslashes: \teammate-message
    # - Optional attributes/whitespace after tag name
    # - Non-greedy content matching to handle nested tags
    # - Backreference ensures opening and closing tags match
    # Strip complete wrapper tags first
    cleaned = re.sub(
        r"<\\?([\w-]+)(?:\s[^>]*)?>.*?</\\?\1>",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    # Then strip any remaining orphaned XML-like tags (opening or closing)
    cleaned = re.sub(r"</?\\?[\w-]+(?:\s[^>]*)?>", "", cleaned)

    if not cleaned.strip():
        return ""

    # Remove ANSI escape sequences (e.g., \x1b[1m for bold)
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", cleaned)

    # Remove other control characters (keep only printable, space, tab, newline)
    cleaned = "".join(c for c in cleaned if c.isprintable() or c in ("\n", "\t", " "))

    # Collapse newlines and runs of whitespace to single space
    cleaned = re.sub(r"[\n\t]+", " ", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned


@dataclass
class AgentMeta:
    agent_type: str | None = None
    name: str | None = None
    model: str | None = None
    description: str | None = None


@dataclass
class EpisodeMetadata:
    title: str
    timestamp: datetime | None = None
    git_branch: str | None = None
    session_id: str = ""
    cwd: str | None = None
    files_touched: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    agent_meta: AgentMeta | None = None


def load_agent_meta(meta_path: Path) -> AgentMeta | None:
    try:
        with open(meta_path) as f:
            data = json.load(f)
        # Clean description through the same pipeline as other indexable fields
        desc_raw = data.get("description", "")
        description = normalize_title(desc_raw) if desc_raw else None
        return AgentMeta(
            agent_type=data.get("agentType"),
            name=data.get("name"),
            model=data.get("model"),
            description=description,
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def derive_files_touched(records: list[dict]) -> tuple[str, ...]:
    files = set()
    for record in records:
        # Skip isMeta records
        if record.get("isMeta"):
            continue
        message = record.get("message", {})
        content = message.get("content", [])
        # Only process if content is a list of blocks
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") in (
                "Read",
                "Edit",
                "Write",
            ):
                input_data = block.get("input", {})
                if isinstance(input_data, dict):
                    file_path = input_data.get("file_path")
                    if file_path:
                        files.add(file_path)
    return tuple(sorted(files))


def harvest_metadata(
    records: list[dict],
    session_id: str,
    agent_meta: AgentMeta | None = None,
    episode_index: int | None = None,
) -> EpisodeMetadata:
    timestamp = None
    git_branch = None
    cwd = None
    tool_names = []
    # Track all possible titles and select by precedence
    custom_title = ""
    ai_title = ""
    last_prompt_title = ""
    user_title = ""

    for record in records:
        # Skip isMeta records
        if record.get("isMeta"):
            continue

        rec_type = record.get("type", "")

        if rec_type == "custom-title":
            custom_title = normalize_title(record.get("custom-title", ""))
        elif rec_type == "ai-title":
            ai_title = normalize_title(record.get("ai-title", ""))
        elif rec_type == "last-prompt":
            last_prompt_title = normalize_title(record.get("last-prompt", ""))

        if rec_type == "user" and not user_title:
            message = record.get("message", {})
            content = message.get("content", [])
            # content can be a string or a list of blocks
            if isinstance(content, str):
                text = content
                if text:
                    # Normalize first (to handle markup tags) then truncate to 100 chars
                    user_title = normalize_title(text)[:100]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            # Normalize first (to handle markup tags) then truncate to 100 chars
                            user_title = normalize_title(text)[:100]
                            break

        if timestamp is None and record.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
            except Exception:
                pass

        if git_branch is None:
            git_branch = record.get("gitBranch")
        if cwd is None:
            cwd = record.get("cwd")

        if rec_type == "assistant":
            message = record.get("message", {})
            content = message.get("content", [])
            # Only process list-form content for tool_use extraction
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name")
                        if name:
                            tool_names.append(name)

    files_touched = derive_files_touched(records)

    # Apply title precedence: custom-title > ai-title > last-prompt > user > ordinal
    # episode_index from indexer makes harvest_metadata sole title authority
    if episode_index is not None:
        ordinal_fallback = f"Episode {episode_index}"
    else:
        ordinal_fallback = "Untitled Episode"
    title = custom_title or ai_title or last_prompt_title or user_title or ordinal_fallback

    # Final pass: ensure title is single-line after all fallback logic
    # Collapse newlines, carriage returns, tabs to single space
    title = re.sub(r"[\n\r\t]+", " ", title)
    title = re.sub(r" +", " ", title)
    title = title.strip()

    return EpisodeMetadata(
        title=title,
        timestamp=timestamp,
        git_branch=git_branch,
        session_id=session_id,
        cwd=cwd,
        files_touched=files_touched,
        tool_names=tuple(tool_names),
        agent_meta=agent_meta,
    )
