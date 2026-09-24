"""Semantic excerpt windowing for search result cards.

The lexical excerpt in :func:`ssgrep.search.response._excerpt_window` anchors a
card on where the query's *words* literally occur. That is often not where the
user asked: a transcript can name the exact terms early and then answer the
query later in a paraphrase. This module instead picks the excerpt from
token-level late-interaction relevance, the same signal LanceDB uses to rank,
so a card surfaces the section most similar to the query rather than the
section that merely echoes its vocabulary.

A chunk's stored ``vector`` column is token-pooled by default
(``SSGREP_POOL_FACTOR``), which destroys token-to-character alignment, so the
snippet path re-encodes each chunk unpooled (``pool_factor=1``) to recover
per-token vectors aligned with the raw text. It then scores every token
against the query matrix and slides a character window to the densest run of
high-relevance tokens.

One subtlety: PyLate's document encoding does not emit one row per input
token. It drops punctuation rows through an internal ``skiplist`` and prepends
a document-prefix token after ``[CLS]``, so output row ``k`` is *not* the
``k``-th input token. ``_token_spans`` reproduces that exact row order from the
tokenizer's ``offset_mapping`` so each encoded row maps back to its character
span in the raw text.

This path is strictly additive: every failure is caught by the caller, which
falls back to the lexical excerpt, so snippet selection can never raise or
break a search.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ssgrep.search.response import EXCERPT_MAX_CHARS

#: Env var toggling semantic snippet windowing; unset defaults to ON, any
#: falsy token (``0``/``no``/``off``/``false``) disables it. When disabled the
#: lexical excerpt (windowed around the literal query text) is used instead.
SEMANTIC_SNIPPET_ENV = "SSGREP_SEMANTIC_SNIPPET"
_FALSY = {"", "0", "false", "no", "off"}

#: A per-card excerpt result: ``(excerpt_text, was_truncated)``. The second
#: element feeds ``SearchResponse.excerpts_truncated`` exactly like the lexical
#: window does.
Excerpt = tuple[str, bool]

#: A batch window function: given the kept cards' full chunk texts, return an
#: excerpt for each. Used to inject the semantic path into response shaping
#: without coupling shaping to the embedding model.
WindowFn = Callable[[list[str]], list[Excerpt]]


def enabled() -> bool:
    """Whether semantic snippet windowing is active for this process.

    Unset (default) means ON; an explicit falsy token (``0``/``no``/``off``/
    ``false``) disables it. An empty value is treated as unset so operators can
    neither accidentally disable the feature nor trip over a blank assignment.
    """
    raw = os.environ.get(SEMANTIC_SNIPPET_ENV, "").strip().lower()
    return not (raw and raw in _FALSY)


def _token_spans(embedder: Any, text: str) -> list[tuple[int, int] | None]:
    """Map each encoded document row to its ``(start, end)`` span in ``text``.

    Mirrors PyLate's document-encoding row order exactly. The tokenizer with
    ``add_special_tokens=False`` yields the text tokens and their character
    offsets; the prefix (``document_prefix_id``) is inserted after ``[CLS]``,
    and the trailing ``[SEP]`` closes the sequence. Rows that survive the
    punctuation ``skiplist`` are in that order, so row ``k`` is either a
    special/prefix row (returned as ``None``) or the ``(k - 2)``-th text token
    when ``k - 2`` falls inside the text-token range.
    """
    tok = embedder.tokenizer
    skiplist = set(embedder.skiplist or ())
    prefix_id = int(embedder.document_prefix_id or 0)
    encoded = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    full_ids = [tok.cls_token_id, prefix_id, *encoded["input_ids"], tok.sep_token_id]
    spans: list[tuple[int, int] | None] = []
    for pos in range(len(full_ids)):
        if full_ids[pos] in skiplist:
            continue
        token_index = pos - 2
        if 0 <= token_index < len(offsets):
            spans.append((int(offsets[token_index][0]), int(offsets[token_index][1])))
        else:
            spans.append(None)
    return spans


def _per_token_sims(query_matrix: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Per-doc-token relevance: the best query token's cosine for each doc row.

    ``query_matrix`` and ``doc_matrix`` are the normalized ``(num_tokens, D)``
    late-interaction outputs, so the dot product is a cosine. Taking the max
    over query rows gives each document row its strongest query affinity,
    which is exactly the per-token contribution behind the MaxSim episode
    score.
    """
    query = np.asarray(query_matrix, dtype=np.float32)
    docs = np.asarray(doc_matrix, dtype=np.float32)
    return (query @ docs.T).max(axis=0)


def _window(
    text: str,
    spans: Sequence[tuple[int, int] | None],
    sims: Any,
    *,
    max_chars: int = EXCERPT_MAX_CHARS,
) -> Excerpt:
    """Select a ``max_chars`` excerpt around the densest run of relevance.

    Each encoded row's relevance is spread over its character span, then a
    sliding character window of ``max_chars`` is scored by summed relevance and
    the densest start wins (ties resolve to the earliest start, keeping the
    selection deterministic). The final window is snapped to whitespace
    boundaries within a small slack and ellipsized, mirroring the lexical
    window's output shape.
    """
    if len(text) <= max_chars:
        return text, False

    relevance = np.zeros(len(text), dtype=np.float32)
    for span, weight in zip(spans, sims, strict=True):
        if span is None:
            continue
        start, end = span
        if end <= start:
            continue
        start = max(0, min(start, len(text)))
        end = min(end, len(text))
        if end <= start:
            continue
        relevance[start:end] += float(weight)
    if float(relevance.max()) <= 0.0:
        raise ValueError("no relevant character span")

    current = float(relevance[:max_chars].sum())
    best, best_start = current, 0
    for start in range(1, len(text) - max_chars + 1):
        current += float(relevance[start + max_chars - 1]) - float(relevance[start - 1])
        if current > best + 1e-9:
            best, best_start = current, start

    start, end = best_start, best_start + max_chars
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


def _window_one(
    text: str,
    query_matrix: np.ndarray,
    embedder: Any,
    doc_matrix: np.ndarray,
    *,
    max_chars: int = EXCERPT_MAX_CHARS,
) -> Excerpt:
    """Semantic excerpt for one pre-encoded chunk text (raises on failure)."""
    spans = _token_spans(embedder, text)
    sims = _per_token_sims(query_matrix, doc_matrix)
    if len(sims) != len(spans):
        raise ValueError("doc rows and spans disagree")
    return _window(text, spans, sims, max_chars=max_chars)


def semantic_windows(
    embedder: Any,
    query_matrix: np.ndarray,
    texts: list[str],
    *,
    fallback: Callable[[str], Excerpt],
    max_chars: int = EXCERPT_MAX_CHARS,
) -> list[Excerpt]:
    """Semantic excerpt for a batch of chunk texts.

    Chunks already at or under ``max_chars`` are returned unchanged. Longer
    chunks are re-encoded in one batched call (``pool_factor=1`` so token rows
    stay aligned). Any failure -- model error, empty alignment, a row/span
    mismatch -- routes that one chunk to ``fallback`` (the lexical window), so
    the semantic path degrades per-chunk and never raises.
    """
    long_texts = [text for text in texts if len(text) > max_chars]
    matrices: list[np.ndarray | None] = [None] * len(long_texts)
    if long_texts:
        try:
            encoded = embedder.encode(
                long_texts,
                is_query=False,
                normalize_embeddings=True,
                pool_factor=1,
            )
            for i, matrix in enumerate(encoded):
                matrices[i] = np.asarray(matrix, dtype=np.float32)
        except Exception:
            matrices = [None] * len(long_texts)

    results: list[Excerpt] = []
    next_long = 0
    for text in texts:
        if len(text) <= max_chars:
            results.append((text, False))
            continue
        matrix = matrices[next_long] if next_long < len(matrices) else None
        next_long += 1
        if matrix is None or matrix.shape[0] == 0:
            results.append(fallback(text))
            continue
        try:
            results.append(_window_one(text, query_matrix, embedder, matrix, max_chars=max_chars))
        except Exception:
            results.append(fallback(text))
    return results
