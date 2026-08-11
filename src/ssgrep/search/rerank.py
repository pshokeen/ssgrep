"""Late-interaction (ColBERT-style MaxSim) reranker leg.

Every other leg in search/__init__.py scores chunks with a single vector
(mean-pooled cosine) or a bag-of-words signal (BM25 variants) -- both throw
away word-order and per-token detail. This leg recovers some of that detail
cheaply, on a *shortlist* only, following the "prefetch, then rerank" pattern
described in Qdrant's "universal query API" writeup: a fast first stage
(the existing five legs' weighted RRF fusion) retrieves an oversampled
candidate pool, then a precise second stage reorders just that pool using
richer per-token scoring before everything is fused again.

The "precise second stage" here is not a new heavyweight cross-encoder
model -- it reuses the *same* model2vec model already loaded for the vector
leg (embed.encode_sequence(), model2vec's real per-token static
embeddings), scored with ColBERT's MaxSim: for each query token, take its
best cosine similarity to any token in the candidate document, then sum
across query tokens. This captures a coarse form of the word-order and
per-token-salience signal that a single mean-pooled vector discards
(concentrated, distinctive tokens dominate their own best match; a rare
identifier "buried" in a mostly-generic chunk doesn't get diluted the way
mean-pooling would dilute it), without adding a new dependency, without
re-embedding the corpus, and staying inside the existing latency budget
(measured: ~5ms for 50 chunk-sized texts, negligible next to the ~184ms
model load already paid for the vector leg -- see embed.py).

Tuned with eval/harness.py against the frozen corpus snapshot (see
eval/README.md, "Update 2026-08-07"): swept RERANK_CANDIDATE_POOL (the
oversampling factor --
how many of the prior fused ranking's top chunks are handed to this leg)
against RERANK_LEG_WEIGHT (this leg's RRF fusion weight). 150/0.5 was the
first working configuration (0.587 -> 0.612 mrr, zero recall@10 cost).
After CHUNK_OVERLAP widened to 300 (see chunker.py), a re-sweep found
250/1.0 -- weight at parity with the two precision legs (AND-BM25, vector
cosine), not above them -- as the largest principled step: mrr climbs
smoothly through weight 0.5->1.0 at top_n=250 with recall@10 unchanged.

Deliberately NOT pushed further: weight values above ~1.2 produce a
non-monotonic, jagged mrr-vs-weight curve (a spike at weight=3.5, then
decline through 4.0-6.0) with the exact same recall@10 and the exact same
6 missed queries at every weight tested. That shape means the extra mrr
comes from a specific rank rearrangement that happens to suit this 48-query
labelled set at one arbitrary weight, not a generalizing improvement -- and
letting this leg run at 3-6x the precision legs' weight stops being
"hybrid fusion" and starts being "MaxSim with cosmetic voting from the
other five legs."
"""

from __future__ import annotations

import numpy as np

from ssgrep import embed
from ssgrep.search.rows import _ChunkHit

# Oversampling factor: how many of the prior fused ranking's top chunks get
# handed to this leg. See module docstring for the tuning provenance.
RERANK_CANDIDATE_POOL = 250

# This leg's weight in the RRF sum alongside the two precision legs' 1.0.
# See module docstring for the tuning provenance.
RERANK_LEG_WEIGHT = 1.0

# Below this token count, MaxSim degenerates: with one query token, "max
# over document tokens" is just that single token's best match, not a real
# late-interaction signal, and it can outvote the precision legs on noise.
# Mirrors the existing len(query.split()) > 1 gate on the OR-BM25 leg
# (_fts_query_any's caller in search/__init__.py).
MIN_QUERY_TOKENS_FOR_RERANK = 2


def _cosine_maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """ColBERT-style MaxSim, normalized by query length.

    sum, over every query token, of its highest cosine similarity to any
    token in doc_tokens -- divided by the number of query tokens so scores
    are comparable across queries of different lengths (an unnormalized sum
    would systematically favor longer queries, which accumulate more terms
    to sum regardless of match quality).
    """
    if doc_tokens.shape[0] == 0 or query_tokens.shape[0] == 0:
        return 0.0
    q_norm = np.linalg.norm(query_tokens, axis=1, keepdims=True)
    d_norm = np.linalg.norm(doc_tokens, axis=1, keepdims=True)
    q_unit = query_tokens / np.clip(q_norm, 1e-8, None)
    d_unit = doc_tokens / np.clip(d_norm, 1e-8, None)
    sim = q_unit @ d_unit.T
    return float(sim.max(axis=1).sum() / q_unit.shape[0])


def rerank_leg(
    query: str,
    prior_fused: dict[str, float],
    chunk_hits: dict[str, _ChunkHit],
    *,
    top_n: int = RERANK_CANDIDATE_POOL,
) -> list[str]:
    """Rank the top-`top_n` candidates from `prior_fused` by MaxSim score.

    Returns a chunk-id list ordered best-first, meant to be fused as one
    more leg into reciprocal_rank_fusion() alongside the five existing legs
    -- never used standalone, since it only ever sees a shortlist the prior
    legs already surfaced (this is a *rerank* of a prefetch, not an
    independent retrieval pass over the whole corpus).

    Returns [] (a no-op leg) for short queries (see
    MIN_QUERY_TOKENS_FOR_RERANK) or when no candidate has fetched text.
    """
    if len(query.split()) < MIN_QUERY_TOKENS_FOR_RERANK:
        return []
    candidates = sorted(prior_fused.items(), key=lambda item: -item[1])[:top_n]
    candidate_ids = [chunk_id for chunk_id, _score in candidates if chunk_id in chunk_hits]
    if not candidate_ids:
        return []

    query_tokens = embed.encode_sequence([query])[0]
    doc_token_seqs = embed.encode_sequence([chunk_hits[c].text for c in candidate_ids])

    scored = [
        (chunk_id, _cosine_maxsim(query_tokens, doc_tokens))
        for chunk_id, doc_tokens in zip(candidate_ids, doc_token_seqs, strict=True)
    ]
    scored.sort(key=lambda item: -item[1])
    return [chunk_id for chunk_id, _score in scored]
