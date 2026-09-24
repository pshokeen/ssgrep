"""Hermetic tests for semantic snippet windowing (:mod:`ssgrep.search.snippet`)."""

from __future__ import annotations

import numpy as np
import pytest

from ssgrep.search import snippet
from ssgrep.search.response import EXCERPT_MAX_CHARS


class _FakeTokenizer:
    """A whitespace tokenizer whose ids are ``0..n-1`` and offsets are real char spans."""

    cls_token_id = 100
    sep_token_id = 101

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, object]:
        words = text.split()
        offsets: list[tuple[int, int]] = []
        pos = 0
        for word in words:
            start = text.index(word, pos)
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        return {"input_ids": list(range(len(words))), "offset_mapping": offsets}


class _FakeEmbedder:
    """An embedder stub returning a fixed matrix, plus the tokenizer above."""

    def __init__(self, matrix: np.ndarray, *, skiplist: set[int] | None = None) -> None:
        self.tokenizer = _FakeTokenizer()
        self.document_prefix_id = 200
        self.skiplist = set(skiplist or ())
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.encode_texts: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        normalize_embeddings: bool = True,
        pool_factor: int = 1,
    ) -> list[np.ndarray]:
        self.encode_texts.append(list(texts))
        return [self.matrix] * len(texts)


def test_snippet_env_defaults_enabled_and_blank_stays_enabled(monkeypatch) -> None:
    monkeypatch.delenv(snippet.SEMANTIC_SNIPPET_ENV, raising=False)
    assert snippet.enabled() is True
    monkeypatch.setenv(snippet.SEMANTIC_SNIPPET_ENV, "   ")
    assert snippet.enabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", "Off"])
def test_snippet_env_falsy_tokens_disable(monkeypatch, raw: str) -> None:
    monkeypatch.setenv(snippet.SEMANTIC_SNIPPET_ENV, raw)
    assert snippet.enabled() is False


def test_token_spans_maps_rows_through_cls_prefix_and_skiplist() -> None:
    """Row order follows pylate: CLS, doc-prefix, surviving text tokens, SEP."""
    text = "aa bb cc"
    # ids: aa=0, bb=1, cc=2; skiplist drops id 1, so "bb" leaves no span.
    embedder = _FakeEmbedder(np.zeros((5, 4)), skiplist={1})

    spans = snippet._token_spans(embedder, text)

    assert spans == [None, None, (0, 2), (6, 8), None]


def test_per_token_sims_is_max_query_similarity_per_doc_row() -> None:
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    sims = snippet._per_token_sims(query, docs)

    np.testing.assert_allclose(sims, [1.0, 0.0], rtol=1e-6)


def test_window_short_text_is_unchanged() -> None:
    assert snippet._window("short", [(0, 5)], [0.5]) == ("short", False)


def test_window_selects_the_relevance_dense_span() -> None:
    text = "aaaa bbbb cccc dddd"
    spans = [(0, 4), (5, 9), (10, 14), (15, 19)]
    sims = [0.1, 0.1, 1.0, 0.1]

    excerpt, truncated = snippet._window(text, spans, sims, max_chars=8)

    assert truncated is True
    assert "cccc" in excerpt
    assert excerpt.startswith("...")  # the dense span is not at char 0
    assert len(excerpt) <= 8 + 40 + 3  # boundary slack + ellipses


def test_window_raises_when_no_row_lands_on_text() -> None:
    text = "aaaa bbbb cccc"
    with pytest.raises(ValueError, match="no relevant character span"):
        snippet._window(text, [None, None], [0.0, 0.0], max_chars=5)


def test_semantic_windows_short_text_is_unchanged_and_skips_encode() -> None:
    embedder = _FakeEmbedder(np.zeros((4, 8)))
    text = "short text"
    assert len(text) <= EXCERPT_MAX_CHARS

    results = snippet.semantic_windows(
        embedder,
        np.zeros((2, 8), dtype=np.float32),
        [text],
        fallback=lambda t: (t, False),
    )

    assert results == [(text, False)]
    assert embedder.encode_texts == []


def test_semantic_windows_uses_fallback_when_encode_raises() -> None:
    class _RaisesOnEncode:
        def encode(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    long_text = "word " * 200  # well over EXCERPT_MAX_CHARS
    embedder = _RaisesOnEncode()

    def fallback(text: str) -> tuple[str, bool]:
        return ("FALLBACK", True)

    results = snippet.semantic_windows(
        embedder,
        np.zeros((2, 8), dtype=np.float32),
        [long_text],
        fallback=fallback,
    )

    assert results == [("FALLBACK", True)]


def test_semantic_windows_falls_back_per_chunk_after_shared_encode_failure() -> None:
    embedder = _FakeEmbedder(np.zeros((0, 8)))  # empty matrix -> no rows

    long_text = "word " * 200
    results = snippet.semantic_windows(
        embedder,
        np.zeros((2, 8), dtype=np.float32),
        [long_text],
        fallback=lambda t: ("lex", True),
    )

    assert results == [("lex", True)]


def test_semantic_windows_windows_long_text_around_dense_relevance() -> None:
    text = "alpha beta gamma delta"
    # full_ids = [CLS(100), prefix(200), alpha(0), beta(1), gamma(2), delta(3), SEP(101)]
    # -> 7 rows; row 4 is "gamma" at chars 11-16.
    matrix = np.zeros((7, 4), dtype=np.float32)
    matrix[4] = [1.0, 0.0, 0.0, 0.0]
    embedder = _FakeEmbedder(matrix)
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    results = snippet.semantic_windows(
        embedder,
        query,
        [text],
        fallback=lambda t: ("lex", True),
        max_chars=6,
    )

    excerpt, truncated = results[0]
    assert truncated is True
    assert "gamma" in excerpt
    assert excerpt.startswith("...")


def test_window_skips_span_whose_end_precedes_its_start() -> None:
    text = "aaaa bbbb cccc"
    spans = [(0, 4), (6, 3)]  # second span ends before it starts
    sims = [1.0, 0.5]

    excerpt, truncated = snippet._window(text, spans, sims, max_chars=8)

    assert truncated is True
    assert "aaaa" in excerpt


def test_window_skips_span_that_clamps_past_text_end() -> None:
    text = "aaaa bbbb cccc"
    spans = [(0, 4), (30, 40)]  # span begins past the text end -> empty after clamp
    sims = [1.0, 0.5]

    excerpt, truncated = snippet._window(text, spans, sims, max_chars=8)

    assert truncated is True
    assert "aaaa" in excerpt


def test_window_one_raises_on_row_span_mismatch() -> None:
    embedder = _FakeEmbedder(np.zeros((2, 4)))
    text = "aa bb cc"  # tokenizes to 3 words -> 6 rows after CLS/prefix/SEP

    with pytest.raises(ValueError, match="rows and spans disagree"):
        snippet._window_one(
            text,
            np.zeros((2, 4), dtype=np.float32),
            embedder,
            np.zeros((2, 4), dtype=np.float32),
        )


def test_semantic_windows_falls_back_when_windowing_raises() -> None:
    """Encode succeeds but alignment fails: the per-chunk fallback still fires."""
    embedder = _FakeEmbedder(np.zeros((2, 8)))  # 2 rows vs ~200 text tokens
    long_text = "word " * 200

    def fallback(text: str) -> tuple[str, bool]:
        return ("lex", True)

    results = snippet.semantic_windows(
        embedder,
        np.zeros((2, 8), dtype=np.float32),
        [long_text],
        fallback=fallback,
    )

    assert results == [("lex", True)]
