"""Building and shaping search responses."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from ssgrep.search.rows import _ChunkHit, _EpisodeRow
from ssgrep.utilities.types import ContentType, ResultCard, SearchResponse


def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def _sortable_epoch(value: datetime) -> float:
    if value == datetime.min:
        return float("-inf")
    normalized = _normalize_dt(value)
    assert normalized is not None
    return normalized.replace(tzinfo=UTC).timestamp()


TOKEN_BUDGET_OVERHEAD_PER_CARD = 50  # ref/timestamp/score/files, not the excerpt
CHARS_PER_TOKEN = 4
EXCERPT_MAX_CHARS = 400


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 3:
        return text[:max_chars], True
    return text[: max_chars - 3] + "...", True


def _excerpt_window(text: str, query: str, max_chars: int = EXCERPT_MAX_CHARS) -> tuple[str, bool]:
    """Return a deterministic query-centered window over raw chunk text."""
    if len(text) <= max_chars:
        return text, False
    folded = text.casefold()
    normalized_query = query.strip().casefold()
    phrase_at = folded.find(normalized_query) if normalized_query else -1
    if phrase_at >= 0:
        anchor = phrase_at + len(normalized_query) // 2
        starts = [max(0, anchor - max_chars // 2)]
    else:
        tokens = tuple(dict.fromkeys(re.findall(r"[\w./:-]{3,}", normalized_query)))
        positions = [folded.find(token) for token in tokens if folded.find(token) >= 0]
        starts = [max(0, position - max_chars // 3) for position in positions] or [0]

    def score(start: int) -> tuple[int, int]:
        window = folded[start : start + max_chars]
        token_hits = sum(window.count(token) for token in tokens) if phrase_at < 0 else 0
        return token_hits, -start

    start = min(max(starts, key=score), len(text) - max_chars)
    end = start + max_chars
    if start > 0:
        boundary = text.rfind(" ", max(0, start - 40), start + 1)
        if boundary >= 0:
            start = boundary + 1
    if end < len(text):
        boundary = text.find(" ", end, min(len(text), end + 40))
        if boundary >= 0:
            end = boundary
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix, True


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _apply_token_budget(cards: list[ResultCard], budget: int) -> tuple[list[ResultCard], int, bool]:
    """Fit already-ranked cards within budget: truncate excerpts, then drop.

    `cards` must already be in final rank order. Per the Token-Budgeted
    Response Shaping requirement, excerpt text is shortened before any
    result is dropped, and only the ranked tail is ever dropped -- never an
    arbitrary subset -- so rank order integrity is preserved.

    This is the ONLY token-budget implementation in production use. render.py
    once carried a lookalike (enforce_budget), which had no production
    callers at all -- search built its responses here and never routed
    through it -- so it was deleted rather than kept as a second,
    diverging copy of this policy.
    """
    kept: list[ResultCard] = []
    truncated_any = False
    remaining = budget

    for card in cards:
        fixed_cost = _estimate_tokens(card.title) + TOKEN_BUDGET_OVERHEAD_PER_CARD
        if fixed_cost >= remaining:
            break  # not even an empty excerpt fits; drop this and every lower-ranked card

        excerpt_budget = remaining - fixed_cost
        excerpt = card.excerpt
        if _estimate_tokens(excerpt) > excerpt_budget:
            max_chars = excerpt_budget * CHARS_PER_TOKEN
            excerpt, was_truncated = _truncate_text(excerpt, max_chars)
            if was_truncated:
                truncated_any = True
                card = replace(card, excerpt=excerpt)

        kept.append(card)
        remaining -= fixed_cost + _estimate_tokens(excerpt)

    return kept, len(cards) - len(kept), truncated_any


def _rank_episodes(
    episode_scores: dict[str, float], episode_rows: dict[str, _EpisodeRow]
) -> list[str]:
    """Deterministic order: score desc, timestamp desc, episode_id asc.

    The final key (episode_id, always unique) leaves no tie unresolved, so
    output order cannot depend on upstream dict/set iteration order -- only
    on the index's actual content, which is what the Deterministic
    Ordering requirement asks for.
    """

    def sort_key(episode_id: str) -> tuple[float, float, str]:
        score = episode_scores[episode_id]
        episode = episode_rows.get(episode_id)
        ts = _normalize_dt(episode.timestamp) if episode else None
        epoch = _sortable_epoch(ts if ts is not None else datetime.min)
        return (-score, -epoch, episode_id)

    return sorted(episode_scores.keys(), key=sort_key)


def _build_result_card(
    episode_id: str,
    score: float,
    hit: _ChunkHit,
    episode: _EpisodeRow,
) -> ResultCard:
    is_subagent = episode.is_subagent
    return ResultCard(
        ref=episode_id,
        title=episode.title,
        timestamp=episode.timestamp,
        score=score,
        excerpt=hit.text,
        files_touched=episode.files_touched,
        is_subagent=is_subagent,
        # Gated on is_subagent regardless of what the row happens to carry,
        # per "Main-session match carries no subagent attribution SHALL NOT
        # fabricate agent attribution fields".
        agent_name=episode.agent_name if is_subagent else None,
        agent_description=episode.agent_description if is_subagent else None,
        parent_session_id=episode.parent_session_id if is_subagent else None,
        content_type=ContentType(hit.content_type),
        project=episode.project,
        source_path=episode.source_path,
        source_project=episode.source_project,
        git_branch=episode.git_branch,
        agent_model=episode.agent_model,
        source_absent=hit.source_status == "absent",
        runtime=episode.runtime,
    )


def _build_response(
    rolled_up: dict[str, tuple[float, _ChunkHit]],
    episode_rows: dict[str, _EpisodeRow],
    *,
    limit: int,
    token_budget: int,
    clamped: bool,
    query: str = "",
    window: Callable[[list[str]], list[tuple[str, bool]]] | None = None,
) -> SearchResponse:
    """Shape already-fused, already-boosted episode scores into a response.

    Deliberately separate from retrieval mechanics: this function only sees
    final per-episode scores, so it can be exercised directly (including
    with an empty `rolled_up`) without needing a real index on disk.

    ``window`` injects the excerpt strategy: a batch callable that maps the
    kept cards' full chunk texts to ``(excerpt, truncated)`` pairs. When
    ``None`` (the default) each card's excerpt is the lexical query-centered
    window (:func:`_excerpt_window`); ``search`` supplies the semantic snippet
    window instead so shaping itself never needs the embedding model.
    """
    total_matches = len(rolled_up)
    scores = {episode_id: score for episode_id, (score, _hit) in rolled_up.items()}
    ordered_ids = _rank_episodes(scores, episode_rows)[:limit]

    cards = []
    for episode_id in ordered_ids:
        score, hit = rolled_up[episode_id]
        episode = episode_rows.get(episode_id)
        if episode is None:
            continue
        cards.append(_build_result_card(episode_id, score, hit, episode))

    windowed_any = False
    if window is not None:
        windows = window([card.excerpt for card in cards])
        for i, (excerpt, windowed) in enumerate(windows):
            cards[i] = replace(cards[i], excerpt=excerpt)
            windowed_any = windowed_any or windowed
    else:
        for i, card in enumerate(cards):
            excerpt, windowed = _excerpt_window(card.excerpt, query)
            cards[i] = replace(card, excerpt=excerpt)
            windowed_any = windowed_any or windowed

    kept, omitted_count, budget_truncated = _apply_token_budget(cards, token_budget)

    return SearchResponse(
        results=kept,
        omitted_count=omitted_count,
        index_empty=False,
        total_matches=total_matches,
        excerpts_truncated=windowed_any or budget_truncated,
        clamped=clamped,
    )
