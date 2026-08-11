"""Result formatting for search responses."""

from __future__ import annotations

import json
from dataclasses import asdict

from ssgrep import textsafe
from ssgrep.types import EpisodeDetail, ResultCard, SearchResponse

TOKEN_BUDGET_DEFAULT = 1500


def render_result_card(card: ResultCard, format: str = "text") -> str:
    if format == "json":
        return json.dumps(asdict(card), default=str, indent=2)
    # Text mode escapes every transcript-derived field (see textsafe.printable);
    # the JSON branch above returns original bytes untouched.
    marker = " [source deleted]" if card.source_absent else ""
    lines = [f"[{card.ref}] {textsafe.printable(card.title)}{marker}"]
    if card.score:
        lines.append(f"  Score: {card.score:.3f}")
    if card.files_touched:
        lines.append(f"  Files: {textsafe.printable(', '.join(card.files_touched))}")
    if card.is_subagent:
        lines.append(f"  Agent: {textsafe.printable(card.agent_name or 'unknown')}")
        if card.agent_description:
            lines.append(f"  Task: {textsafe.printable(card.agent_description[:80])}")
    if card.excerpt:
        lines.append(f"  {textsafe.printable(card.excerpt[:200])}")
    return "\n".join(lines)


def render_search_response(
    response: SearchResponse, format: str = "text", token_budget: int = TOKEN_BUDGET_DEFAULT
) -> str:
    if format == "json":
        return json.dumps(asdict(response), default=str, indent=2)
    lines = []
    if response.stale:
        lines.append(
            f"Index is stale ({response.stale_count} transcripts changed since last index). "
            "Results below may miss recent content -- run `ssgrep index` to refresh."
        )
        lines.append("")
    if not response.results:
        if not response.index_exists:
            lines.append("No index found. Run `ssgrep index` first.")
        elif response.index_empty:
            lines.append("Index is empty.")
        else:
            lines.append("No matches found.")
        return "\n".join(lines)

    for card in response.results:
        lines.append(render_result_card(card))
        lines.append("")

    if response.omitted_count > 0:
        lines.append(f"--- {response.omitted_count} results omitted (token budget) ---")
    if response.clamped:
        lines.append("--- Results clamped to maximum ---")
    return "\n".join(lines)


def render_episode_detail(detail: EpisodeDetail, format: str = "text") -> str:
    if format == "json":
        return json.dumps(asdict(detail), default=str, indent=2)
    # Text mode escapes every transcript-derived field (see textsafe.printable);
    # the JSON branch above returns original bytes untouched.
    lines = [
        f"Episode: {detail.episode_id}",
        f"Session: {detail.session_id}",
        f"Title: {textsafe.printable(detail.title)}",
    ]
    if detail.stale:
        lines.append(
            f"Index is stale ({detail.stale_count} transcripts changed since last index). "
            "This episode may not reflect recent changes -- run `ssgrep index` to refresh."
        )
    if detail.is_subagent:
        lines.append(f"Agent: {textsafe.printable(detail.agent_name or 'unknown')}")
        if detail.agent_description:
            lines.append(f"Task: {textsafe.printable(detail.agent_description)}")
    if detail.files_touched:
        lines.append(f"Files: {textsafe.printable(', '.join(detail.files_touched))}")
    lines.append("")
    lines.append("--- Prompt ---")
    # Body text keeps genuine newlines/tabs; only control bytes are escaped.
    # Truncation markers are already included in the text by detail.py.
    lines.append(textsafe.printable(detail.prompt_text, keep="\n\t"))
    lines.append("")
    lines.append("--- Response ---")
    lines.append(textsafe.printable(detail.response_text, keep="\n\t"))
    return "\n".join(lines)
