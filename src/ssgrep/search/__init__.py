"""Global LanceDB multivector (late-interaction) search.

LanceDB owns vector retrieval: a query is embedded into a ``(num_tokens, DIMENSION)``
matrix and searched natively against the multivector ``vector`` column, where
LanceDB computes MaxSim per chunk and returns the engine's ``_distance``
column. ssgrep adds metadata prefilters, episode roll-up, deterministic
response shaping, and its public error contracts.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime

import numpy as np

from ssgrep.indexing.embed import (
    DIMENSION,
    MODEL_ID,
    MODEL_REVISION,
    ModelDownloadError,
    configure_model_loading,
    load_embedder,
    unit_mean_vector,
)
from ssgrep.search.lexical import corpus_stats_from_repository, hybrid_scores
from ssgrep.search.render import TOKEN_BUDGET_DEFAULT
from ssgrep.search.response import _build_response, _excerpt_window
from ssgrep.search.rows import _ChunkHit, _EpisodeRow
from ssgrep.search.snippet import (
    enabled as _semantic_enabled,
    semantic_windows as _semantic_windows,
)
from ssgrep.store import CHUNKS_TABLE, LanceStore, _env_int, quote
from ssgrep.utilities.types import (
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    InvalidPredicateError,
    SearchFilters,
    SearchResponse,
)

DEFAULT_RESULT_COUNT = 10
MAX_RESULT_COUNT = 50

#: The per-episode multi-chunk evidence nudge. Scores on the refined scale
#: are ~O(num_query_tokens) (30+); an earlier 0.05 * sum(next 2) inflated
#: multi-chunk episodes by ~10% of their best-chunk score, letting episodes
#: with many ANN hits dominate the top-10 over higher-scoring single-chunk
#: episodes (measured against brute-force MaxSim ordering). The weight is 0:
#: episode score is the best chunk's MaxSim, which matches the exact-scan
#: reference ordering (r@10/rr@10/ndcg@10/p@5 at the full-scan ceiling).
MULTI_CHUNK_EVIDENCE_WEIGHT = 0.0
MULTI_CHUNK_EVIDENCE_COUNT = 2

# The per-episode rollup wants several chunks per episode, but fetching exactly
# ``limit`` chunks lets one dominant episode fill the whole engine result list
# and starve the rest. The engine call therefore fetches
# ``limit * OVERSAMPLE_FACTOR`` candidates; rollup consumes that pool and
# response shaping reduces it back to the requested limit.
OVERSAMPLE_FACTOR = 8
OVERSAMPLE_MIN = 1
OVERSAMPLE_MAX = 20

# Two-stage prefiltered search (SSGREP_TWO_STAGE, default OFF): stage 1 picks
# max(200, limit x OVERSAMPLE x WIDEN) candidates from the proxy_vector ANN —
# deeper than the single-stage pool because a mean-vector proxy is only an
# approximation of MaxSim and must not drop true hits; stage 2 rescores those
# candidates exactly. SSGREP_TWO_STAGE_CANDIDATES raises the floor when a
# corpus's proxy recall needs it (measured: 200 -> 18/20 queries matching
# brute force on a near-tie 2k-chunk corpus; 1000 closed the rest).
TWO_STAGE_ENV = "SSGREP_TWO_STAGE"
TWO_STAGE_WIDEN = 4
TWO_STAGE_CANDIDATES_ENV = "SSGREP_TWO_STAGE_CANDIDATES"
TWO_STAGE_CANDIDATES_MIN = 200
TWO_STAGE_CANDIDATES_MAX = 10000
_TWO_STAGE_TRUTHY = {"1", "true", "yes", "on"}

# Measured on lancedb 0.37.1: refine_factor(1) rescores the multivector
# _distance onto the exact scale _distance = 1 - MaxSim.
DISTANCE_TO_MAXSIM_OFFSET = 1.0


def _resolved_oversample_factor() -> int:
    """SSGREP_OVERSAMPLE clamped to [1, 20]; unparsable values keep the default."""
    return _env_int("SSGREP_OVERSAMPLE", OVERSAMPLE_FACTOR, OVERSAMPLE_MIN, OVERSAMPLE_MAX)


def _two_stage_enabled() -> bool:
    """SSGREP_TWO_STAGE gate; anything but explicit truthy tokens keeps it off."""
    return os.environ.get(TWO_STAGE_ENV, "").strip().lower() in _TWO_STAGE_TRUTHY


def _two_stage_candidate_floor() -> int:
    """SSGREP_TWO_STAGE_CANDIDATES clamped to [200, 10000]; unset keeps the floor."""
    return _env_int(
        TWO_STAGE_CANDIDATES_ENV,
        TWO_STAGE_CANDIDATES_MIN,
        TWO_STAGE_CANDIDATES_MIN,
        TWO_STAGE_CANDIDATES_MAX,
    )


def _unpadded_query_rows(query: str) -> int | None:
    """Real (non-pad) row count of the query matrix, or None when unknown.

    PyLate right-pads every query to ``query_length`` (32) mask-token rows and
    keeps them in the encoded output, so a mean over all rows would be
    dominated by that constant padding and carry almost no query signal. The
    tokenizer's attention mask marks exactly the real prefix and costs one
    tokenization — never a forward pass.
    """
    try:
        features = load_embedder().tokenize([query], is_query=True)
    except Exception:  # noqa: BLE001 - fake embedders (tests) have no tokenizer
        return None
    return int(features["attention_mask"].sum())


def _query_proxy_vector(query: str, query_matrix: np.ndarray) -> np.ndarray:
    """Stage-1 probe vector: unit-normalized mean of the query's real rows."""
    end = _unpadded_query_rows(query)
    rows = query_matrix[:end] if end is not None else query_matrix
    return unit_mean_vector(rows)


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _as_items(value: object) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if item)
    if isinstance(value, str) and value:
        return tuple(value.split("\n"))
    return ()


def _utc_literal(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value.isoformat()


def _contains(column: str, value: str) -> str:
    """Build a literal substring predicate without LIKE wildcard semantics."""
    return f"strpos({column}, {quote(value)}) > 0"


def build_where(filters: SearchFilters | None = None, where: str | None = None) -> str | None:
    """Compile public filters and an optional raw Lance predicate into one prefilter."""
    clauses: list[str] = []
    if filters:
        if filters.date_from:
            value = _utc_literal(filters.date_from)
            clauses.append(f"timestamp >= timestamp {quote(value)}")
        if filters.date_to:
            value = _utc_literal(filters.date_to)
            clauses.append(f"timestamp <= timestamp {quote(value)}")
        if filters.file_path:
            clauses.append(_contains("files_touched", filters.file_path))
        if filters.content_type:
            clauses.append(f"content_type = {quote(filters.content_type.value)}")
        if filters.branch:
            clauses.append(f"git_branch = {quote(filters.branch)}")
        if filters.project:
            clauses.append(f"project = {quote(filters.project)}")
        if filters.source_path:
            clauses.append(f"source_path = {quote(filters.source_path)}")
        if filters.session_id:
            clauses.append(f"session_id = {quote(filters.session_id)}")
        if filters.is_subagent is not None:
            clauses.append(f"is_subagent = {'true' if filters.is_subagent else 'false'}")
        if filters.agent_type:
            clauses.append(f"agent_type = {quote(filters.agent_type)}")
        if filters.agent_model:
            clauses.append(f"agent_model = {quote(filters.agent_model)}")
        if filters.tool_name:
            clauses.append(_contains("tool_names", filters.tool_name))
        if filters.runtime:
            clauses.append(f"runtime = {quote(filters.runtime)}")
    if where:
        clauses.append(f"({where})")
    return " AND ".join(clauses) or None


def _query_matrix(query: str) -> np.ndarray:
    """Embed a query into a ``(num_tokens, DIMENSION)`` float32 matrix.

    Uses the query-time embedder (``is_query=True``) so the ColBERT query
    prefix / padding semantics match the MaxSim ``Q`` used in the score scale.
    ``pool_factor=1`` is explicit: document-side token pooling
    (``SSGREP_POOL_FACTOR``) must never shrink the query matrix.
    """
    embedder = load_embedder()
    return np.asarray(
        embedder.encode(
            [query],
            is_query=True,
            normalize_embeddings=True,
            pool_factor=1,
        )[0],
        dtype=np.float32,
    )


def _rows_to_episodes(
    rows: list[dict],
    *,
    num_query_tokens: int,
) -> tuple[dict[str, tuple[float, _ChunkHit]], dict[str, _EpisodeRow]]:
    """Deterministically roll up the best chunk plus two bounded evidence votes.

    ``num_query_tokens`` is the query matrix's row count (first dimension).
    LanceDB's ``_distance`` column maps to a real MaxSim score via
    ``maxsim = DISTANCE_TO_MAXSIM_OFFSET - _distance``: production queries
    always carry ``refine_factor(1)``, which rescores onto that exact scale.
    Rows without ``_distance`` fall back to the legacy
    ``_relevance_score`` / rank-based score.
    """
    grouped: dict[str, list[tuple[float, str, _ChunkHit, _EpisodeRow]]] = {}
    for rank, row in enumerate(rows, start=1):
        episode_id = str(row["episode_id"])
        raw_distance = row.get("_distance")
        if raw_distance is not None:
            score = DISTANCE_TO_MAXSIM_OFFSET - float(raw_distance)
        else:
            raw_score = row.get("_relevance_score", 1.0 / rank)
            score = float(raw_score) if raw_score is not None else 1.0 / rank
        if not math.isfinite(score):
            score = 0.0
        score = max(score, 0.0)
        chunk_id = str(row.get("chunk_id") or f"{episode_id}:{rank}")
        hit = _ChunkHit(
            content_type=str(row["content_type"]),
            text=str(row["text"]),
            source_status=str(row.get("source_status") or "available"),
            chunk_id=chunk_id,
        )
        episode = _EpisodeRow(
            title=str(row.get("title") or episode_id),
            timestamp=_as_datetime(row.get("timestamp")),
            git_branch=row.get("git_branch"),
            files_touched=_as_items(row.get("files_touched")),
            is_subagent=bool(row.get("is_subagent", False)),
            agent_name=row.get("agent_name"),
            agent_description=row.get("agent_description"),
            parent_session_id=row.get("parent_session_id"),
            project=row.get("project"),
            source_path=row.get("source_path"),
            source_project=row.get("source_project"),
            agent_model=row.get("agent_model"),
            runtime=str(row.get("runtime") or "claude"),
        )
        grouped.setdefault(episode_id, []).append((score, chunk_id, hit, episode))

    rolled: dict[str, tuple[float, _ChunkHit]] = {}
    episodes: dict[str, _EpisodeRow] = {}
    for episode_id, candidates in grouped.items():
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best_score, _chunk_id, hit, episode = candidates[0]
        tail = candidates[1 : 1 + MULTI_CHUNK_EVIDENCE_COUNT]
        score = best_score + MULTI_CHUNK_EVIDENCE_WEIGHT * sum(item[0] for item in tail)
        rolled[episode_id] = (score, hit)
        episodes[episode_id] = episode
    return rolled, episodes


def search(
    query: str,
    *,
    limit: int | None = None,
    token_budget: int | None = None,
    filters: SearchFilters | None = None,
    where: str | None = None,
) -> SearchResponse:
    """Search every indexed project, optionally prefiltered with Lance SQL.

    Use ``where="project = '/path'"`` or
    ``SearchFilters(project=...)`` to select one project.
    """
    if not query or not query.strip():
        raise EmptyQueryError("Query must not be empty.")

    requested = DEFAULT_RESULT_COUNT if limit is None else max(limit, 0)
    effective_limit = min(requested, MAX_RESULT_COUNT)
    clamped = requested > MAX_RESULT_COUNT
    budget = TOKEN_BUDGET_DEFAULT if token_budget is None else max(token_budget, 0)
    repository = LanceStore()
    if not repository.exists():
        raise IndexNotFoundError("No global index found. Run `ssgrep index` first.")
    if repository.get_meta("index_state") != "ready":
        raise IndexNotReadyError(
            "The global index is incomplete; run `ssgrep index` to resume it.",
            condition="index_incomplete",
            command="ssgrep index",
        )
    expected_meta = {
        "schema_version": str(repository.schema_version),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vector_dimension": str(DIMENSION),
    }
    if not repository.schema_matches() or any(
        repository.get_meta(key) != value for key, value in expected_meta.items()
    ):
        raise IndexNotReadyError(
            "The global index schema or embedding model is incompatible; "
            "run `ssgrep index --rebuild`."
        )
    if repository.count(CHUNKS_TABLE) == 0:
        return SearchResponse(
            results=[],
            omitted_count=0,
            index_empty=True,
            total_matches=0,
            excerpts_truncated=False,
            clamped=clamped,
        )

    predicate = build_where(filters, where)
    try:
        configure_model_loading(MODEL_ID)
        query_matrix = _query_matrix(query.strip())
        num_query_tokens = int(query_matrix.shape[0])
        pool_limit = effective_limit * _resolved_oversample_factor()
        if _two_stage_enabled():
            rows = repository.two_stage_search(
                query_matrix,
                query_proxy=_query_proxy_vector(query.strip(), query_matrix),
                limit=max(_two_stage_candidate_floor(), pool_limit * TWO_STAGE_WIDEN),
                where=predicate,
                pool_size=pool_limit,
            )
        else:
            rows = repository.multivector_search(
                query_matrix,
                limit=pool_limit,
                where=predicate,
            )
    except ModelDownloadError as exc:
        raise IndexNotReadyError(str(exc), condition="model_unavailable", command=None) from exc
    except ValueError as exc:
        if predicate is not None:
            raise InvalidPredicateError(f"Invalid metadata predicate: {exc}") from exc
        raise IndexNotReadyError(f"LanceDB multivector search failed: {exc}") from exc
    except Exception as exc:
        raise IndexNotReadyError(f"LanceDB multivector search failed: {exc}") from exc
    rolled, episode_rows = _rows_to_episodes(rows, num_query_tokens=num_query_tokens)
    # Hybrid retrieval: BM25 lexical fusion over the MaxSim episode ranking.
    # Only when the index carries persisted lexical stats (indexes written
    # before this feature degrade to pure MaxSim ordering).
    lexical_stats = corpus_stats_from_repository(repository)
    if lexical_stats is not None and rows:
        maxsim = {eid: score for eid, (score, _hit) in rolled.items()}
        fused = hybrid_scores(rows, maxsim, lexical_stats, query.strip())
        rolled = {eid: (fused[eid], hit) for eid, (score, hit) in rolled.items()}
    semantic = None
    if _semantic_enabled():
        embedder = load_embedder()

        def semantic(texts: list[str]) -> list[tuple[str, bool]]:
            return _semantic_windows(
                embedder,
                query_matrix,
                texts,
                fallback=lambda text: _excerpt_window(text, query.strip()),
            )

    return _build_response(
        rolled,
        episode_rows,
        limit=effective_limit,
        token_budget=budget,
        clamped=clamped,
        query=query.strip(),
        window=semantic,
    )


__all__ = [
    "DEFAULT_RESULT_COUNT",
    "MAX_RESULT_COUNT",
    "build_where",
    "search",
]
