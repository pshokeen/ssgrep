"""Hybrid retrieval: BM25-style lexical fusion over the MaxSim episode ranking.

Dense late-interaction MaxSim embeddings capture semantics but compress away
rare tokens (exact identifiers, error strings, tool names) that users search
for verbatim. This module adds a BM25-style lexical signal over chunk text
and fuses it with the MaxSim episode ranking via Reciprocal Rank Fusion
(RRF), a standard hybrid-retrieval technique.

Corpus statistics (document frequencies, average chunk length) are computed
once at index time and persisted as JSON in the metadata table; search-time
BM25 scores the returned pool rows against those statistics, so a query costs
one tokenization pass over the pool (no corpus rescan).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence

#: RRF constant: rank positions are worth 1/(k+rank). 60 is the standard
#: value from the original RRF paper and gives a gentle lexical nudge.
#: Overridable via SSGREP_RRF_K for operator tuning without a rebuild.
RRF_K = 60
#: Tokens shorter than this are stop-word noise for the lexical signal.
MIN_TOKEN_LEN = 2
#: BM25 saturation (k1) and length-normalization (b) defaults (Lucene-style).
BM25_K1 = 1.2
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric/underscore tokens (identifiers survive intact)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= MIN_TOKEN_LEN]


def corpus_stats(chunk_rows: Sequence[Mapping]) -> dict:
    """df / avgdl / n_docs statistics over chunk rows (chunk_id + text).

    Deterministic: df counts each chunk once per token; avgdl is the mean
    token count. ``chunk_rows`` must be every chunk in the corpus (the full
    chunks-table scan), because df/avgdl must describe the collection.
    """
    df: Counter[str] = Counter()
    total_tokens = 0
    n_docs = 0
    for row in chunk_rows:
        toks = tokenize(row.get("text") or "")
        n_docs += 1
        total_tokens += len(toks)
        df.update(set(toks))
    return {
        "df": dict(df),
        "avgdl": total_tokens / max(1, n_docs),
        "n_docs": n_docs,
    }


def bm25_chunk(
    chunk_text: str,
    query_tokens: Sequence[str],
    stats: Mapping,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    """BM25 score of one chunk against the query tokens, given corpus stats.

    ``stats`` is the persisted ``corpus_stats`` dict. Tokens absent from the
    corpus df are treated as df=0 (maximum idf) so verbatim matches of rare
    identifiers get the strongest lexical signal.
    """
    df = stats.get("df") or {}
    n_docs = int(stats.get("n_docs") or 0)
    avgdl = float(stats.get("avgdl") or 1.0)
    tf = Counter(tokenize(chunk_text))
    if not tf:
        return 0.0
    dl = sum(tf.values())
    score = 0.0
    for token in set(query_tokens):
        freq = tf.get(token, 0)
        if freq == 0:
            continue
        doc_freq = df.get(token, 0)
        idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        score += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / avgdl))
    return score


def _best_chunk_scores(
    rows: Sequence[Mapping],
    query_tokens: Sequence[str],
    stats: Mapping,
) -> dict[str, float]:
    """Per-episode best-chunk BM25 over the pool rows."""
    best: dict[str, float] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        score = bm25_chunk(str(row.get("text") or ""), query_tokens, stats)
        if score > best.get(episode_id, 0.0):
            best[episode_id] = score
    return best


def _resolved_rrf_k() -> int:
    """Resolve SSGREP_RRF_K, clamped to [1, 500]; unparsable keeps the default."""
    try:
        value = int(os.environ.get("SSGREP_RRF_K", "").strip())
    except ValueError:
        return RRF_K
    return max(1, min(500, value))


def _resolved_dense_weight() -> float:
    """Resolve SSGREP_DENSE_WEIGHT, clamped to [0.25, 4.0]; unparsable keeps 1.0."""
    try:
        value = float(os.environ.get("SSGREP_DENSE_WEIGHT", "").strip())
    except ValueError:
        return 1.0
    return max(0.25, min(4.0, value))


def fuse_rrf(
    maxsim_scores: Mapping[str, float],
    lexical_scores: Mapping[str, float],
    *,
    k: int | None = None,
    dense_weight: float | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion of the MaxSim and lexical episode scores.

    Both signals are ranked descending; each episode accumulates
    ``weight/(k + rank)`` per signal it appears in. ``dense_weight``
    defaults to 1.0 (equal weighting, the standard RRF); >1.0 gives the
    MaxSim signal more influence over the final order (lexical stays as a
    rescue signal for exact identifiers/error strings). Episodes present in
    only one signal still participate (RRF does not require intersection).
    The result is a fused score dict sorted by descending fused value.
    """
    if dense_weight is None:
        dense_weight = _resolved_dense_weight()
    fused: dict[str, float] = {}
    if k is None:
        k = _resolved_rrf_k()
    maxsim_ranked = sorted(maxsim_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    for rank, (episode_id, _score) in enumerate(maxsim_ranked):
        fused[episode_id] = fused.get(episode_id, 0.0) + dense_weight / (k + rank + 1)
    lexical_ranked = sorted(lexical_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    for rank, (episode_id, _score) in enumerate(lexical_ranked):
        fused[episode_id] = fused.get(episode_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_scores(
    rows: Sequence[Mapping],
    maxsim_scores: Mapping[str, float],
    stats: Mapping | None,
    query: str,
) -> dict[str, float]:
    """Fused per-episode scores for a pool of rows.

    ``maxsim_scores`` maps episode_id -> rolled MaxSim score (best chunk).
    ``stats`` is the persisted corpus statistics, or ``None`` when the index
    predates lexical stats: fusion degrades to pure MaxSim (scores unchanged,
    so ordering is preserved).
    """
    if stats is None:
        return dict(maxsim_scores)
    query_tokens = tokenize(query)
    if not query_tokens:
        return dict(maxsim_scores)
    lexical = _best_chunk_scores(rows, query_tokens, stats)
    if not lexical:
        return dict(maxsim_scores)
    return fuse_rrf(maxsim_scores, lexical, k=_resolved_rrf_k())


def dump_stats(stats: Mapping) -> str:
    """Serialize corpus statistics for the metadata table."""
    return json.dumps(stats, sort_keys=True, separators=(",", ":"))


def load_stats(raw: str | None) -> dict | None:
    """Deserialize corpus statistics; malformed or absent -> None (no fusion)."""
    if not raw:
        return None
    try:
        stats = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(stats, dict) or "df" not in stats or "avgdl" not in stats:
        return None
    return stats


def corpus_stats_from_repository(repository) -> dict | None:
    """Read + parse persisted corpus stats from the metadata table.

    Returns None when the index predates lexical stats (any older index):
    hybrid fusion degrades to pure MaxSim, so old indexes stay fully
    functional until the next ``ssgrep index`` rewrite.
    """
    return load_stats(repository.get_meta("lexical_stats"))


__all__ = [
    "BM25_B",
    "BM25_K1",
    "MIN_TOKEN_LEN",
    "RRF_K",
    "bm25_chunk",
    "corpus_stats",
    "dump_stats",
    "fuse_rrf",
    "hybrid_scores",
    "load_stats",
    "tokenize",
]
