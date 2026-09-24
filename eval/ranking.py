"""Ranking instrumentation over the prefetch, final, and brute-force seams.

Pure data functions: every entry point returns ``(ranking, elapsed_ms)`` and
never prints. Each arm observes the production pipeline at one sanctioned seam
(see ``.omo/plans/retrieval-eval-overhaul.md`` T2):

- ``prefetch_chunk_ranking`` — the UNCAPPED engine seam
  (``LanceStore.multivector_search``, ``store/__init__.py:423``): the raw
  chunk pool the production rollup consumes, at any depth.
- ``prefetch_episode_ranking`` — the same deep pool rolled up to episodes with
  the production rollup/ordering logic (``_rows_to_episodes`` +
  ``_rank_episodes`` UNSLICED): the episode-level prefetch headline.
- ``final_ranking`` — the production ``ssgrep.search.search`` path (what users
  see), capped at ``MAX_RESULT_COUNT`` (50) internally.
- ``brute_force_ranking`` — a full chunks-table scan scored with exact MaxSim
  (``ssgrep.store._maxsim``): the reference arm. ORDER is the only meaningful
  signal — the flat-scan MaxSim scale differs from the refined production
  scale (see ``.omo/notepads/late-interaction-retrieval-optimization/``).

All functions sandbox ``SSGREP_DATA_DIR`` through ``eval.harness``'s
``_private_data_dir`` context manager, so they never touch the live index.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from eval.harness import _private_data_dir, rank_episodes_for_query
from ssgrep import search as search_module
from ssgrep.search import response as response_module
from ssgrep.search.lexical import corpus_stats_from_repository, hybrid_scores
from ssgrep.store import CHUNKS_TABLE, LanceStore, _maxsim

__all__ = [
    "brute_force_ranking",
    "combined_final_and_prefetch",
    "combined_prefetch_ranking",
    "final_ranking",
    "final_ranking_with_matrix",
    "prefetch_chunk_ranking",
    "prefetch_episode_ranking",
]


def combined_prefetch_ranking(
    index_dir: Path,
    query: str,
    *,
    pool_depth: int = 800,
    chunk_depth: int = 100,
    query_matrix: np.ndarray | None = None,
) -> tuple[list[tuple[str, float]], list[tuple[str, str, float]], float]:
    """Combined prefetch: one embedding, one search, both episode and chunk rankings.

    Returns ``(episode_ranking, chunk_ranking, elapsed_ms)`` where:
    - ``episode_ranking`` is ``[(episode_id, score), ...]`` from rolling up the full pool
    - ``chunk_ranking`` is ``[(chunk_id, episode_id, score), ...]`` from the top chunks
    - ``elapsed_ms`` is the wall-clock time in milliseconds

    This avoids duplicate query embedding and LanceDB search that
    separate ``prefetch_episode_ranking`` and ``prefetch_chunk_ranking`` calls would incur.
    If ``query_matrix`` is provided, it is used directly instead of embedding the query.
    """
    started = time.perf_counter()
    with _private_data_dir(index_dir):
        matrix = query_matrix if query_matrix is not None else search_module._query_matrix(query)
        rows = LanceStore().multivector_search(matrix, limit=pool_depth)
        rolled, episode_rows = search_module._rows_to_episodes(
            rows,
            num_query_tokens=matrix.shape[0],
        )
    # Episode ranking from full pool
    scores = {episode_id: score for episode_id, (score, _hit) in rolled.items()}
    ordered = response_module._rank_episodes(scores, episode_rows)
    episode_ranking = [(episode_id, scores[episode_id]) for episode_id in ordered]
    # Chunk ranking from top chunk_depth chunks
    chunk_ranking = [
        (
            str(row["chunk_id"]),
            str(row["episode_id"]),
            search_module.DISTANCE_TO_MAXSIM_OFFSET - float(row["_distance"]),
        )
        for row in rows[:chunk_depth]
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return episode_ranking, chunk_ranking, elapsed_ms


def prefetch_chunk_ranking(
    index_dir: Path,
    query: str,
    *,
    depth: int = 100,
) -> tuple[list[tuple[str, str, float]], float]:
    """UNCAPPED chunk pool: engine MaxSim rows at ``depth``, score-converted.

    Returns ``(ranking, elapsed_ms)`` where ``ranking`` is
    ``[(chunk_id, episode_id, score), ...]`` ordered by descending score.
    ``score`` is the production MaxSim conversion
    (``maxsim = DISTANCE_TO_MAXSIM_OFFSET - _distance``), so chunk scores sit
    on the same scale the episode rollup consumes.
    """
    started = time.perf_counter()
    with _private_data_dir(index_dir):
        matrix = search_module._query_matrix(query)
        rows = LanceStore().multivector_search(matrix, limit=depth)
    ranking = [
        (
            str(row["chunk_id"]),
            str(row["episode_id"]),
            search_module.DISTANCE_TO_MAXSIM_OFFSET - float(row["_distance"]),
        )
        for row in rows
    ]
    return ranking, (time.perf_counter() - started) * 1000.0


def prefetch_episode_ranking(
    index_dir: Path,
    query: str,
    *,
    pool_depth: int = 800,
) -> tuple[list[tuple[str, float]], float]:
    """Episode-level prefetch headline: deep pool rolled up, UNSLICED.

    Feeds ``pool_depth`` chunk rows through the production rollup
    (``_rows_to_episodes``, ``search/__init__.py:204``) and orders every
    rolled-up episode with ``_rank_episodes`` (``response.py:117``) without
    slicing, so the returned ranking is the full episode pool the final page
    is drawn from. ``num_query_tokens`` is vestigial but passed per the seam.
    """
    started = time.perf_counter()
    with _private_data_dir(index_dir):
        matrix = search_module._query_matrix(query)
        rows = LanceStore().multivector_search(matrix, limit=pool_depth)
        rolled, episode_rows = search_module._rows_to_episodes(
            rows,
            num_query_tokens=matrix.shape[0],
        )
    scores = {episode_id: score for episode_id, (score, _hit) in rolled.items()}
    ordered = response_module._rank_episodes(scores, episode_rows)
    ranking = [(episode_id, scores[episode_id]) for episode_id in ordered]
    return ranking, (time.perf_counter() - started) * 1000.0


def final_ranking(
    index_dir: Path,
    query: str,
    *,
    limit: int = 10,
) -> tuple[list[tuple[str, float]], float]:
    """Production ranking: ``ssgrep.search.search`` via the harness path.

    The engine caps results at ``MAX_RESULT_COUNT`` (50) internally; a
    ``limit`` above that is clamped by production, which is exactly what this
    arm measures.
    """
    started = time.perf_counter()
    ranked, scores = rank_episodes_for_query(index_dir, query, limit=limit)
    ranking = [(episode_id, scores[episode_id]) for episode_id in ranked]
    return ranking, (time.perf_counter() - started) * 1000.0


def final_ranking_with_matrix(
    index_dir: Path,
    query: str,
    query_matrix: np.ndarray,
    *,
    limit: int = 10,
) -> tuple[list[tuple[str, float]], float]:
    """Production ranking using a pre-computed query matrix.

    Same as ``final_ranking`` but avoids re-embedding the query.
    The matrix must be a ``(num_tokens, DIMENSION)`` float32 array
    produced by ``search_module._query_matrix``.
    """
    started = time.perf_counter()
    with _private_data_dir(index_dir):
        repository = LanceStore()
        pool_limit = limit * search_module._resolved_oversample_factor()
        rows = repository.multivector_search(query_matrix, limit=pool_limit)
        rolled, episode_rows = search_module._rows_to_episodes(
            rows,
            num_query_tokens=query_matrix.shape[0],
        )
    scores = {episode_id: score for episode_id, (score, _hit) in rolled.items()}
    ordered = response_module._rank_episodes(scores, episode_rows)
    ranking = [(episode_id, scores[episode_id]) for episode_id in ordered[:limit]]
    return ranking, (time.perf_counter() - started) * 1000.0


def combined_final_and_prefetch(
    index_dir: Path,
    query: str,
    query_matrix: np.ndarray,
    *,
    final_limit: int = 10,
    pool_depth: int = 800,
    chunk_depth: int = 100,
) -> tuple[
    list[tuple[str, float]],  # final_episodes
    list[tuple[str, float]],  # prefetch_episodes
    list[tuple[str, str, float]],  # chunk_ranks
    float,  # final_ms
    float,  # prefetch_ms
]:
    """Final + prefetch rankings with separate pools, shared context.

    Returns ``(final_episodes, prefetch_episodes, chunk_ranks, final_ms, prefetch_ms)``.
    Uses a single ``_private_data_dir`` context and a single ``LanceStore`` instance,
    but performs two separate searches with appropriate pool sizes:
    - final_ranking: oversampled pool (limit * oversample_factor)
    - prefetch_ranking: full pool (pool_depth)
    """
    started = time.perf_counter()
    with _private_data_dir(index_dir):
        store = LanceStore()
        # Final ranking: oversampled pool
        final_pool_limit = final_limit * search_module._resolved_oversample_factor()
        final_rows = store.multivector_search(query_matrix, limit=final_pool_limit)
        final_rolled, final_episode_rows = search_module._rows_to_episodes(
            final_rows,
            num_query_tokens=query_matrix.shape[0],
        )
        # Prefetch ranking: full pool
        prefetch_rows = store.multivector_search(query_matrix, limit=pool_depth)
        prefetch_rolled, prefetch_episode_rows = search_module._rows_to_episodes(
            prefetch_rows,
            num_query_tokens=query_matrix.shape[0],
        )
    # Final ranking
    final_scores = {eid: score for eid, (score, _hit) in final_rolled.items()}
    # Hybrid fusion mirrors the production search path: BM25 lexical signal
    # over the pool is RRF-fused with the MaxSim episode scores whenever the
    # index carries persisted lexical stats.
    lexical_stats = corpus_stats_from_repository(store)
    if lexical_stats is not None and final_rows:
        final_scores = hybrid_scores(final_rows, final_scores, lexical_stats, query)
    final_ordered = response_module._rank_episodes(final_scores, final_episode_rows)
    final_episodes = [(eid, final_scores[eid]) for eid in final_ordered[:final_limit]]
    final_ms = (time.perf_counter() - started) * 1000.0

    # Prefetch ranking
    prefetch_started = time.perf_counter()
    prefetch_scores = {eid: score for eid, (score, _hit) in prefetch_rolled.items()}
    prefetch_ordered = response_module._rank_episodes(prefetch_scores, prefetch_episode_rows)
    prefetch_episodes = [(eid, prefetch_scores[eid]) for eid in prefetch_ordered]
    # Chunk ranking from top chunk_depth chunks
    chunk_ranks = [
        (
            str(row["chunk_id"]),
            str(row["episode_id"]),
            search_module.DISTANCE_TO_MAXSIM_OFFSET - float(row["_distance"]),
        )
        for row in prefetch_rows[:chunk_depth]
    ]
    prefetch_ms = (time.perf_counter() - prefetch_started) * 1000.0
    return final_episodes, prefetch_episodes, chunk_ranks, final_ms, prefetch_ms


def brute_force_ranking(
    index_dir: Path,
    query: str,
    *,
    limit: int | None = None,
) -> tuple[list[tuple[str, str, float]], float]:
    """Reference arm: full chunks-table scan scored with exact MaxSim.

    Every chunk's stored token matrix is scored with ``ssgrep.store._maxsim``
    (the engine's own exact kernel, renormalizing like the refined cosine
    path) and sorted descending. ``limit`` truncates the returned ranking;
    ``None`` returns every chunk. ORDER is the only meaningful signal — the
    flat-scan MaxSim scale differs from the refined production scale.
    """
    started = time.perf_counter()
    with _private_data_dir(index_dir):
        matrix = search_module._query_matrix(query)
        store = LanceStore()
        rows = store.rows(CHUNKS_TABLE)
        scored = [
            (
                str(row["chunk_id"]),
                str(row["episode_id"]),
                _maxsim(matrix, row["vector"]),
            )
            for row in rows
        ]
    scored.sort(key=lambda item: (-item[2], item[0]))
    if limit is not None:
        scored = scored[:limit]
    return scored, (time.perf_counter() - started) * 1000.0
