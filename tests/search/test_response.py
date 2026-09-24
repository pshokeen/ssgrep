"""Tests for ranking and token-budgeted response shaping."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

from ssgrep.search.response import (
    CHARS_PER_TOKEN,
    TOKEN_BUDGET_OVERHEAD_PER_CARD,
    _apply_token_budget,
    _build_response,
    _build_result_card,
    _estimate_tokens,
    _excerpt_window,
    _normalize_dt,
    _rank_episodes,
    _sortable_epoch,
    _truncate_text,
)
from ssgrep.search.rows import _ChunkHit, _EpisodeRow
from ssgrep.utilities.types import ContentType


def episode_row(**overrides: object) -> _EpisodeRow:
    values: dict[str, object] = {
        "title": "Title",
        "timestamp": None,
        "git_branch": "main",
        "files_touched": ("a.py",),
        "is_subagent": False,
        "agent_name": "should be hidden",
        "agent_description": "also hidden",
        "parent_session_id": "hidden parent",
        "project": "/project",
        "source_path": "/source.jsonl",
        "source_project": "source-project",
        "agent_model": "model",
    }
    values.update(overrides)
    return _EpisodeRow(**values)  # type: ignore


def test_datetime_normalization_and_sortable_epoch() -> None:
    naive = datetime(2025, 1, 2, 3, 4)
    aware = datetime(2025, 1, 2, 8, 34, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    assert _normalize_dt(None) is None
    assert _normalize_dt(naive) is naive
    assert _normalize_dt(aware) == naive
    assert _sortable_epoch(datetime.min) == float("-inf")
    assert _sortable_epoch(aware) == naive.replace(tzinfo=UTC).timestamp()


def test_truncate_text_boundaries() -> None:
    assert _truncate_text("abc", 3) == ("abc", False)
    assert _truncate_text("abcdef", 3) == ("abc", True)
    assert _truncate_text("abcdef", 5) == ("ab...", True)


def test_token_estimate_has_a_one_token_floor() -> None:
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("x" * (CHARS_PER_TOKEN * 3)) == 3


def test_apply_token_budget_keeps_cards_without_modification(sample_card) -> None:
    kept, omitted, truncated = _apply_token_budget([sample_card], budget=100)

    assert kept == [sample_card]
    assert omitted == 0
    assert truncated is False


def test_apply_token_budget_truncates_before_dropping_ranked_tail(sample_card) -> None:
    first = replace(sample_card, ref="first", title="T", excerpt="abcdefghijklmnopqrstuvwxyz")
    second = replace(sample_card, ref="second", title="T", excerpt="tail")
    # title costs one token, fixed overhead costs 50, leaving two excerpt tokens.
    budget = TOKEN_BUDGET_OVERHEAD_PER_CARD + 3

    kept, omitted, truncated = _apply_token_budget([first, second], budget)

    assert [card.ref for card in kept] == ["first"]
    assert kept[0].excerpt == "abcde..."
    assert omitted == 1
    assert truncated is True


def test_apply_token_budget_drops_everything_when_fixed_cost_does_not_fit(sample_card) -> None:
    kept, omitted, truncated = _apply_token_budget([sample_card], TOKEN_BUDGET_OVERHEAD_PER_CARD)

    assert kept == []
    assert omitted == 1
    assert truncated is False


def test_rank_episodes_orders_by_score_timestamp_then_identifier() -> None:
    rows = {
        "newer": episode_row(timestamp=datetime(2025, 1, 2, tzinfo=UTC)),
        "older": episode_row(timestamp=datetime(2025, 1, 1)),
        "minimum": episode_row(timestamp=datetime.min),
        "same-b": episode_row(timestamp=None),
        "same-a": episode_row(timestamp=None),
    }
    scores = {
        "lower-score": 0.2,  # deliberately has no episode row
        "minimum": 1.0,
        "same-b": 1.0,
        "same-a": 1.0,
        "older": 1.0,
        "newer": 1.0,
    }

    assert _rank_episodes(scores, rows) == [
        "newer",
        "older",
        "minimum",
        "same-a",
        "same-b",
        "lower-score",
    ]


def test_build_result_card_hides_subagent_fields_for_main_episode() -> None:
    card = _build_result_card(
        "session:ep:0",
        0.75,
        _ChunkHit("prompt", "question", "available"),
        episode_row(),
    )

    assert card.ref == "session:ep:0"
    assert card.content_type is ContentType.PROMPT
    assert card.agent_name is None
    assert card.agent_description is None
    assert card.parent_session_id is None
    assert card.source_absent is False
    assert card.project == "/project"
    assert card.git_branch == "main"


def test_build_result_card_includes_subagent_fields_and_absent_source() -> None:
    episode = episode_row(
        is_subagent=True,
        agent_name="worker",
        agent_description="delegated task",
        parent_session_id="parent",
    )
    card = _build_result_card(
        "worker:ep:1",
        0.9,
        _ChunkHit("response", "answer", "absent"),
        episode,
    )

    assert card.content_type is ContentType.RESPONSE
    assert card.agent_name == "worker"
    assert card.agent_description == "delegated task"
    assert card.parent_session_id == "parent"
    assert card.source_absent is True


def test_build_response_handles_empty_input() -> None:
    response = _build_response({}, {}, limit=10, token_budget=100, clamped=True)

    assert response.results == []
    assert response.total_matches == 0
    assert response.omitted_count == 0
    assert response.index_empty is False
    assert response.excerpts_truncated is False
    assert response.clamped is True


def test_build_response_skips_missing_rows_applies_limit_and_budget() -> None:
    rolled = {
        "missing": (1.0, _ChunkHit("response", "not returned", "available")),
        "first": (0.9, _ChunkHit("response", "x" * 40, "available")),
        "outside-limit": (0.1, _ChunkHit("prompt", "tail", "available")),
    }
    rows = {
        "first": episode_row(title="T", timestamp=datetime(2025, 1, 1)),
        "outside-limit": episode_row(title="tail"),
    }

    response = _build_response(rolled, rows, limit=2, token_budget=53, clamped=False)

    assert response.total_matches == 3
    assert [card.ref for card in response.results] == ["first"]
    assert response.results[0].excerpt == "xxxxx..."
    assert response.excerpts_truncated is True
    # Missing metadata is skipped rather than counted as a budget omission.
    assert response.omitted_count == 0
    assert response.clamped is False


def test_excerpt_window_short_text_is_unchanged() -> None:
    assert _excerpt_window("short", "short") == ("short", False)
    assert _excerpt_window("no query", "") == ("no query", False)


def test_excerpt_window_prefers_exact_phrase_anchor() -> None:
    text = "alpha " * 40 + "needle phrase here" + " beta " * 40
    window, truncated = _excerpt_window(text, "needle phrase", max_chars=200)
    assert truncated is True
    assert "needle phrase here" in window
    assert len(window) <= 280  # ellipses/boundary slack
    assert window.startswith("...") or not window.startswith("alpha")


def test_excerpt_window_dense_token_hit_beats_scattered_ones() -> None:
    text = ("noise " * 30) + ("target token. " * 5) + ("noise " * 30)
    window, truncated = _excerpt_window(text, "target token", max_chars=120)
    assert truncated is True
    assert window.count("target") >= 3


def test_excerpt_window_semantic_fallback_to_leading_window() -> None:
    text = "x" * 50 + "y" * 400
    window, truncated = _excerpt_window(text, "zzzabsent", max_chars=200)
    assert truncated is True
    assert len(window) <= 200 + 3
    assert window.endswith("...")
    assert window.startswith("xxx")


def test_excerpt_window_long_token_and_position_clamp() -> None:
    text = "word " * 300
    window, truncated = _excerpt_window(text, "word", max_chars=300)
    assert truncated is True
    assert "word" in window


def test_build_response_windows_excerpts_and_marks_flag() -> None:
    long_excerpt = "filler " * 120  # ~720 chars, beyond the 400-char window
    rolled = {"session:ep:0": (0.9, _ChunkHit("response", long_excerpt, "available"))}
    rows = {"session:ep:0": episode_row()}
    response = _build_response(
        rolled,
        rows,
        limit=10,
        token_budget=10_000,
        clamped=False,
        query="filler",
    )
    assert response.excerpts_truncated is True
    assert len(response.results[0].excerpt) <= 450
    assert "filler" in response.results[0].excerpt


def test_build_response_applies_injected_window_batch() -> None:
    """A supplied ``window`` fn replaces the lexical per-card window."""

    rolled = {
        "session:ep:0": (0.9, _ChunkHit("response", "first chunk", "available")),
        "session:ep:1": (0.8, _ChunkHit("response", "second chunk", "available")),
    }
    rows = {
        "session:ep:0": episode_row(title="T1"),
        "session:ep:1": episode_row(title="T2"),
    }
    seen: list[list[str]] = []

    def window(texts):
        seen.append(list(texts))
        return [(text.upper(), True) for text in texts]

    response = _build_response(
        rolled,
        rows,
        limit=10,
        token_budget=10_000,
        clamped=False,
        window=window,
    )

    assert seen == [["first chunk", "second chunk"]]
    assert [card.excerpt for card in response.results] == ["FIRST CHUNK", "SECOND CHUNK"]
    assert response.excerpts_truncated is True
