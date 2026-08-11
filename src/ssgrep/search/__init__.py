"""Hybrid BM25 + vector retrieval, fused by weighted Reciprocal Rank Fusion.

Two precision legs run on every query: a BM25 keyword leg over the SQLite
FTS5 index (store.search_fts, implicit AND) and a brute-force cosine
similarity leg over the memory-mapped vector store (vectors.cosine_top_k).
Neither is ever skipped based on the query's apparent shape. Three cheap
support legs fill the signals those two cannot see (each measured in with
eval/harness.py -- see the weight constants below): an OR-BM25 leg for
partial token overlap, a trigram-BM25 leg (chunks_fts_tri) for subword and
morphological overlap, and a phrase leg for word-order precision. A sixth
leg (search/rerank.py) reranks the shortlist those five legs surface using
per-token late-interaction (MaxSim) scoring -- a prefetch-then-rerank stage,
not an independent retrieval pass. Results fuse by weighted Reciprocal Rank
Fusion (D4), roll up to their parent episode -- the retrieval unit (D5) --
with a bounded non-best-chunk tail, receive a bounded main-session rank
preference (D5a), and are shaped into a token-budgeted SearchResponse (D8).

This module never indexes: it only opens what index() already built, and
raises rather than repairs when the index is absent or unsafe to query. That
rule protects the latency budget -- everything here lives in the ~66ms left
after the embedding model's ~184ms load.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ssgrep import (
    discovery,
    discovery_roots,
    embed,
    indexer_support,
    notes,
    repair,
    staleness,
    store,
    vectors,
)
from ssgrep.render import TOKEN_BUDGET_DEFAULT
from ssgrep.search.fetch import (
    _fetch_chunk_hits,
    _fetch_chunk_ids_by_vec_row,
    _fetch_episode_rows,
)
from ssgrep.search.filters import (
    _chunk_passes_content_type,
    _episode_passes_filters,
    _filter_leg,
    _normalize_dt,
    _sortable_epoch,
)
from ssgrep.search.rerank import RERANK_LEG_WEIGHT, rerank_leg
from ssgrep.search.response import (
    _apply_token_budget,
    _build_response,
    _estimate_tokens,
    _rank_episodes,
    _truncate_text,
)
from ssgrep.search.rows import _ChunkHit, _EpisodeRow
from ssgrep.types import (
    EmptyQueryError,
    FileCursor,
    IndexNotFoundError,
    IndexNotReadyError,
    SearchFilters,
    SearchResponse,
)

INDEX_DIRNAME = ".ssgrep"

RRF_K = 60

# Bounded, tunable per D5a / the Main-Session Rank Preference requirement.
# apply_main_session_boost() adds this flat constant to every main-session
# episode's fused score. Pass main_session_boost=<value> to search() to
# override per call; the mechanism stays available for a future corpus
# with a different main/subagent balance even though the measured default
# below is 0.0.
#
# Value is measured, not guessed: tuned with the eval harness
# (eval/harness.py) rather than estimated.
# Swept 0.0-1.0 (17 values) on 2026-07-27 against the real corpus, post
# the index-duplication fix (5,957 chunks, 3,360 episodes, 72.4% of
# chunks from subagent episodes), over the 36-query labelled set in
# eval/queries.jsonl:
#
#   boost   overall recall@10   overall MRR   subagent-only recall@10
#   0.0     77.8%               0.558         69.2%   <- shipped
#   0.005   66.7%               0.476         46.2%
#   0.01    58.3%               0.463         30.8%   <- previous default
#   0.02    50.0%               0.375          7.7%
#   >=0.03  44.4%               0.359          0.0%
#
# recall@10 and MRR fall monotonically as the boost rises from 0 -- no
# positive value, however small (0.001 tested), beats disabling the
# preference. The original 0.01 estimate assumed a per-leg RRF
# contribution of ~0.0164 as its reference scale ("~30-60% of one rank-1
# leg contribution"), but real end-to-end episode scores after RRF +
# roll-up sit in a much narrower band, so a fixed additive constant that
# size acts as a near-override, not a tie-breaker -- and it lands hardest
# on subagent-only queries, which hold the majority of this corpus's
# indexable prose (82% by bytes). Full sweep, per-query
# detail, and reproduction steps: eval/results/boost_sweep_2026-07-27.json,
# eval/results/baseline_boost_0.0_2026-07-27.json and
# baseline_boost_0.01_2026-07-27.json (same directory). A change to this
# constant should come with a fresh harness run showing the new value
# actually wins on that same curve, not a guess -- a regression test in
# tests/test_search.py pins the value below and will fail loudly otherwise.
MAIN_SESSION_BOOST = 0.0

DEFAULT_RESULT_COUNT = 10
MAX_RESULT_COUNT = 50

# Per-leg candidate pool fetched before filtering/rollup. Wide enough that
# filters and within-episode chunk collisions don't starve the final ranked
# list. Brute-force cosine cost is dominated by the O(corpus) score pass,
# not by k (D2/D12: 0.1ms at 20k, 6.6ms at 500k), so widening this does not
# touch the latency budget.
LEG_POOL_SIZE = 500

# Fusion weight of the recall-oriented OR-BM25 leg (_fts_query_any) relative
# to the two precision legs' 1.0. Tuned with eval/harness.py on the labelled
# query set; see reciprocal_rank_fusion's docstring for the rationale.
OR_LEG_WEIGHT = 0.9

# Fusion weight of the trigram-BM25 leg (store.search_fts_trigram), the
# subword companion to the OR leg: it matches on shared 3-character
# substrings, bridging morphology and compounding ("undercounted" vs
# "under-reported") that both word-boundary FTS legs miss. Same tuning
# provenance as OR_LEG_WEIGHT; recall on the labelled set plateaus for
# weights 0.9-1.1 and degrades by 1.3. Re-verified on the enlarged 48-query
# set (2026-08-06): 1.0-1.1 measure one query better than 0.9 (a
# blind-composed multi-hop query crossing the top-10 boundary), the 6-query
# holdout is rank-identical across 0.9-1.1, and ablation shows this leg is
# the largest recall contributor (-4 queries when dropped) -- so it ships
# at the plateau's center, level with the precision legs.
TRIGRAM_LEG_WEIGHT = 1.0

# Fusion weight of the phrase-proximity leg (_phrase_query). Measured on the
# labelled set: recall is flat across 0.5-0.9 and degrades by 1.2 (loose
# common-word phrases start outvoting the precision legs on paraphrase
# queries), so it ships at the conservative end of the plateau.
PHRASE_LEG_WEIGHT = 0.5

TOKEN_BUDGET_OVERHEAD_PER_CARD = 50  # ref/timestamp/score/files, not the excerpt
CHARS_PER_TOKEN = 4


def _fts_query_any(query: str) -> str:
    """Build a recall-oriented FTS5 MATCH expression: tokens joined by OR.

    The companion to _fts_query()'s implicit-AND precision leg. A paraphrase
    or multi-hop query rarely contains every literal token of the episode it
    is looking for, so the AND leg correctly returns nothing for it -- but a
    *subset* of its tokens (a library name, one identifier, one error word)
    is often present verbatim. OR semantics let BM25 rank on partial token
    overlap, where rare shared tokens score high and ubiquitous ones score
    near zero, giving the fusion a keyword recall signal the AND leg cannot
    provide. Tokens are quoted exactly as in _fts_query(), and deduplicated
    because repeating a token in an OR expression cannot change which rows
    match it.
    """
    seen: set[str] = set()
    parts = []
    for t in query.split():
        escaped = t.replace('"', '""')
        if escaped not in seen:
            seen.add(escaped)
            parts.append(f'"{escaped}"')
    return " OR ".join(parts)


def _phrase_query(query: str) -> str:
    """Build the MATCH expression for the phrase-proximity leg.

    Adjacent bigrams and trigrams of the query's informative tokens (every
    token in the phrase >= 4 chars, per the _trigram_query rationale), each
    quoted as an FTS5 phrase, OR'd together. A pasted literal like
    `MTEB Retrieval 35.06` names its source episode almost uniquely as a
    *phrase*, while the single-token legs dilute it into terms shared by
    every episode that discusses the same topic -- word order is the signal
    the bag-of-words legs cannot see. Queries with no two adjacent
    informative tokens produce an empty expression and the leg is skipped.
    """
    raw = query.split()
    escaped = [t.replace('"', '""') for t in raw]
    informative = [len(t) >= 4 and any(c.isalnum() for c in t) for t in raw]
    seen: set[str] = set()
    phrases = []
    for n in (2, 3):
        for i in range(len(escaped) - n + 1):
            if all(informative[i : i + n]):
                phrase = '"' + " ".join(escaped[i : i + n]) + '"'
                if phrase.lower() not in seen:
                    seen.add(phrase.lower())
                    phrases.append(phrase)
    return " OR ".join(phrases)


def _trigram_query(query: str) -> str:
    """Build the MATCH expression for the trigram leg.

    Same OR-of-quoted-tokens shape as _fts_query_any, with one extra rule:
    tokens shorter than four characters are dropped. Under the trigram
    tokenizer a three-character function word like "the" or "was" is a
    single trigram shared with a large fraction of every English chunk, so
    it contributes pure noise; and one- or two-character tokens produce no
    trigram at all. Word-boundary matching for short exact tokens ("RRF",
    "f32") is already the AND/OR legs' job -- this leg exists for substring
    overlap inside longer words, where three-plus shared trigrams start to
    mean something.
    """
    seen: set[str] = set()
    parts = []
    for t in query.split():
        if len(t) < 4:
            continue
        escaped = t.replace('"', '""')
        if escaped not in seen:
            seen.add(escaped)
            parts.append(f'"{escaped}"')
    return " OR ".join(parts)


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free-text user input.

    Every whitespace-split token is quoted as an FTS5 string literal, with
    embedded double quotes doubled per FTS5 escaping rules. Unescaped, a
    literal query such as `TypeError: cannot read property 'x' of
    undefined` raises sqlite3.OperationalError -- ':' and stray '"' are
    FTS5 syntax characters -- which is exactly the shape of error strings
    and identifiers this leg exists to serve. Quoted, adjacent tokens
    combine with FTS5's default implicit AND, which is what gives the BM25
    leg high precision on literal queries and correctly zero matches on
    paraphrases that share no literal tokens with the indexed text.
    """
    tokens = query.split()
    escaped = (t.replace('"', '""') for t in tokens)
    return " ".join(f'"{t}"' for t in escaped)


def _index_paths(project_dir: Path) -> tuple[Path, Path]:
    """Resolve the current generation's db and vector paths (read-only).

    Constructing GenerationalStore only reads an existing `.manifest` if
    present; it never creates the index directory or any file.
    """
    index_dir = project_dir / INDEX_DIRNAME
    generation = store.GenerationalStore(index_dir)
    return generation.get_index_path(), generation.get_vector_path()


def _open_ready_index(db_path: Path, vec_path: Path) -> sqlite3.Connection:
    """Open the index, raising if it's absent or unsafe to query.

    This function only validates the index, never repairs. Repair of bounded
    tails happens later, on the search path, after staleness detection (D13).

    Raises:
        IndexNotFoundError: no index.db at the resolved path.
        IndexNotReadyError: index.db exists but isn't a valid ssgrep index,
            its schema_version doesn't match what this build expects, or
            chunks and vectors.f32 have drifted out of alignment (e.g. a
            crash between the two writes -- see local-index spec's
            Vector/Row Alignment requirement).
    """
    if not db_path.exists():
        raise IndexNotFoundError(f"No index found at {db_path}. Run `ssgrep index` first.")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        schema_version = store.get_meta(conn, "schema_version")
    except sqlite3.Error as exc:
        conn.close()
        raise IndexNotReadyError(
            f"Index at {db_path} is corrupt or unreadable. "
            f"Run `ssgrep index --rebuild` to recreate it."
        ) from exc

    if schema_version != str(store.SCHEMA_VERSION):
        conn.close()
        raise IndexNotReadyError(
            f"Index schema version {schema_version!r} does not match the "
            f"expected {store.SCHEMA_VERSION!r}; run `ssgrep index --rebuild`."
        )

    is_aligned, reason = vectors.validate_alignment(db_path, vec_path)
    if not is_aligned:
        conn.close()
        # The remedy belongs in the message text, not only in .command. Its
        # two sibling raises above spell it out; this one did not, and
        # cli/commands/search.py prints str(error) alone -- so a human at the
        # terminal was told their index was corrupt and given nothing to type.
        # `ssgrep index` is named first because it repairs this in place via
        # recover(), with no re-embedding of the corpus.
        raise IndexNotReadyError(
            f"Index at {db_path} is inconsistent: {reason}. Run `ssgrep index` to "
            "repair it in place; `ssgrep index --rebuild` if that does not help.",
            command="ssgrep index",
        )

    return conn


def reciprocal_rank_fusion(
    *ranked_lists: list[str],
    k: int = RRF_K,
    weights: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Fuse ranked document-id lists with Reciprocal Rank Fusion (D4).

    score(d) = sum, over every leg in which d appears, of w_leg/(k + rank),
    where rank is the 1-indexed position of d in that leg's list. A
    document need not appear in every leg; fusion operates only on rank
    position, never on the legs' raw, incomparable scores (BM25 weight vs.
    cosine similarity).

    `weights` scales each leg's vote (default: 1.0 for every leg). The two
    precision legs (AND-BM25, vector) carry full weight; a recall-oriented
    leg like the OR-BM25 leg is fused at reduced weight (OR_LEG_WEIGHT) so
    it can surface episodes the precision legs missed without letting loose
    partial-token matches outvote a precision-leg consensus at the top of
    the ranking.
    """
    if weights is None:
        weights = tuple(1.0 for _ in ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"{len(ranked_lists)} legs but {len(weights)} weights")
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for position, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position)
    return scores


# Bounded contribution of an episode's non-best matching chunks to its
# rolled-up score: score = best + ROLLUP_TAIL_WEIGHT * sum(next best chunks,
# up to ROLLUP_TAIL_CHUNKS - 1 of them). Exists because the fused legs vote
# per *chunk*: an episode whose evidence is split across two chunks (the
# vector leg's best excerpt in one, the keyword legs' in another) used to
# have one of the two discarded by a pure-MAX roll-up, letting an episode
# with a single middling chunk that happened to collect every leg's vote
# outrank it. The tail is small and hard-capped so the roll-up requirement's
# intent stands: an episode is still not rewarded merely for having MANY
# matching chunks -- one clearly stronger chunk keeps beating any pile of
# weak ones (a chunk 5x stronger than each of two weak chunks wins by >3x).
# Tuned with eval/harness.py, retuned 2026-08-07 on the n=66 label set
# under the final six-leg config (0.2 was tuned pre-rerank-leg,
# pre-chunk-window, on n=48 -- the surface moved under it). Mechanism
# diagnosis first, not a blind sweep: three labelled queries' target
# episodes carried a top-12 fused chunk yet ranked 12-16 at the episode
# level, beaten by episodes drawing 10-27% of their rolled-up score from
# tail votes across dozens of pooled chunks -- exactly the vote-
# concentration failure this cap exists to prevent. Swept 0.0-0.3 at n=66:
# recall@10 is maximal (+2 queries over 0.2, zero new misses) on the
# 0.0-0.10 plateau and degrades monotonically above 0.15; 0.05 is the
# plateau's best-MRR interior point. Validated out-of-sample before
# shipping: live-corpus n=66 rerun fixes THREE queries (+4.5 recall pts,
# zero new misses), and a 10-query blind holdout composed after all other
# tuning froze -- evaluated exactly once, for this decision -- holds
# recall at 10/10 with MRR improving under 0.05. The benchmark MRR dip
# (-2.6%) is rank-churn of already-hit queries, the class of movement the
# 2026-08-07 session measured as non-generalizing (eval/README.md, Update
# 2026-08-07); the recall gain is a step-function move confirmed on all
# three evidence tiers.
ROLLUP_TAIL_WEIGHT = 0.05
ROLLUP_TAIL_CHUNKS = 3

# The ranking's theoretical score ceiling: an episode whose best chunk ranks
# first in every fusion leg AND whose roll-up tail is maximal (tail chunks
# each also at the per-chunk ceiling). Derived from the live constants --
# import-not-copy, the same anti-drift principle the provenance system uses
# -- so a weight or tail retune moves it automatically. Iteration-21
# calibration measured the best real top score at 0.0964 against this
# ceiling's ~0.0974: real results genuinely approach it, which is what makes
# score/SCORE_CEILING (ResultCard.score_normalized) a meaningful 0-1 scale
# where the raw RRF-sum magnitude is not.
SCORE_CEILING = (
    (1.0 + 1.0 + OR_LEG_WEIGHT + TRIGRAM_LEG_WEIGHT + PHRASE_LEG_WEIGHT + RERANK_LEG_WEIGHT)
    / (RRF_K + 1)
    * (1.0 + ROLLUP_TAIL_WEIGHT * (ROLLUP_TAIL_CHUNKS - 1))
)


def roll_up_to_episodes(
    chunk_scores: dict[str, float], chunk_hits: dict[str, _ChunkHit]
) -> dict[str, tuple[float, _ChunkHit]]:
    """Collapse chunk-level fused scores to one entry per parent episode.

    An episode's score is the max fused score among its own matching chunks
    plus a bounded tail (see ROLLUP_TAIL_WEIGHT above) -- never an open
    sum, so an episode is not rewarded merely for having many matching
    chunks (Episode-Level Result Roll-up requirement). The chunk retained
    is the one that produced the max, since its text becomes the result's
    excerpt.
    """
    per_episode: dict[str, list[float]] = {}
    best: dict[str, tuple[float, _ChunkHit]] = {}
    for chunk_id, score in chunk_scores.items():
        hit = chunk_hits.get(chunk_id)
        if hit is None:
            continue
        per_episode.setdefault(hit.episode_id, []).append(score)
        current = best.get(hit.episode_id)
        if current is None or score > current[0]:
            best[hit.episode_id] = (score, hit)
    rolled: dict[str, tuple[float, _ChunkHit]] = {}
    for episode_id, (top_score, hit) in best.items():
        scores = sorted(per_episode[episode_id], reverse=True)
        tail = sum(scores[1:ROLLUP_TAIL_CHUNKS])
        rolled[episode_id] = (top_score + ROLLUP_TAIL_WEIGHT * tail, hit)
    return rolled


def apply_main_session_boost(
    episode_scores: dict[str, float],
    is_subagent: dict[str, bool],
    boost: float = MAIN_SESSION_BOOST,
) -> dict[str, float]:
    """Apply the bounded, additive Main-Session Rank Preference (D5a).

    A small constant added to main-session episodes lets an exact tie, or a
    near tie, resolve toward the parent narrative, while a subagent match
    whose score gap exceeds the boost still wins outright -- a filter would
    violate "a clearly stronger subagent match SHALL still be able to
    outrank a weak main-session match"; an unbounded multiplier would too.
    boost=0.0 disables the preference: episodes rank purely by fused score.
    """
    return {
        episode_id: score + (0.0 if is_subagent.get(episode_id, False) else boost)
        for episode_id, score in episode_scores.items()
    }


def search(
    project_dir: Path,
    query: str,
    *,
    limit: int | None = None,
    token_budget: int | None = None,
    filters: SearchFilters | None = None,
    main_session_boost: float = MAIN_SESSION_BOOST,
) -> SearchResponse:
    """Search project_dir's index. Drop-in body for api.search()'s contract.

    Raises:
        EmptyQueryError: query is empty or whitespace-only. Checked before
            any filesystem or index access, so no leg ever runs.
        IndexNotFoundError: no index exists for the project.
        IndexNotReadyError: an index exists but is not safe to query.

    A query that simply matches nothing is not an error: it returns an
    empty SearchResponse with index_exists=True.
    """
    if not query or not query.strip():
        raise EmptyQueryError("Query must not be empty or whitespace-only.")

    filters = filters or SearchFilters()
    requested = DEFAULT_RESULT_COUNT if limit is None else limit
    clamped = requested > MAX_RESULT_COUNT
    effective_limit = min(max(requested, 0), MAX_RESULT_COUNT)
    budget = TOKEN_BUDGET_DEFAULT if token_budget is None else token_budget

    db_path, vec_path = _index_paths(project_dir)
    conn = _open_ready_index(db_path, vec_path)

    # Detect staleness early (stat-only, no parsing) using discovered files
    staleness_report = staleness_summary(project_dir)
    is_stale = staleness.is_index_stale(staleness_report)
    stale_count_value = staleness.stale_count(staleness_report)

    try:
        # Attempt bounded tail repair on appended sessions, within strict byte
        # and time caps, before retrieval. Non-blocking: if the write lock is
        # unavailable, returns immediately without blocking the search.
        if is_stale:
            for stale_info in staleness_report.stale_files:
                if stale_info.status == staleness.FileStatus.APPENDED:
                    session_id_row = conn.execute(
                        "SELECT session_id FROM sessions WHERE path = ?",
                        (str(stale_info.path),),
                    ).fetchone()
                    if session_id_row:
                        session_id = session_id_row[0]
                        repair.repair_current_session_tail(
                            project_dir,
                            stale_info.path,
                            session_id,
                            index_dir=db_path.parent,
                        )

        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total_chunks == 0:
            return SearchResponse(
                results=[],
                index_exists=True,
                index_empty=True,
                total_matches=0,
                clamped=clamped,
                stale=is_stale,
                stale_count=stale_count_value,
            )

        bm25_hits = store.search_fts(conn, _fts_query(query), limit=LEG_POOL_SIZE)
        bm25_chunk_ids = [chunk_id for chunk_id, _rank in bm25_hits]

        # Recall-oriented OR leg (see _fts_query_any). Skipped for single-
        # token queries, where it would be byte-identical to the AND leg and
        # only double-count its evidence in the fusion.
        bm25_any_chunk_ids: list[str] = []
        if len(set(query.split())) > 1:
            bm25_any_hits = store.search_fts(conn, _fts_query_any(query), limit=LEG_POOL_SIZE)
            bm25_any_chunk_ids = [chunk_id for chunk_id, _rank in bm25_any_hits]

        # Phrase-proximity leg (see _phrase_query): word order as a
        # precision signal for pasted literals and named concepts.
        phrase_chunk_ids: list[str] = []
        phrase_match = _phrase_query(query)
        if phrase_match:
            phrase_hits = store.search_fts(conn, phrase_match, limit=LEG_POOL_SIZE)
            phrase_chunk_ids = [chunk_id for chunk_id, _rank in phrase_hits]

        # Subword recall leg over the trigram-tokenized mirror (see
        # store.search_fts_trigram). Runs for every query shape: even a
        # single-token query benefits from substring matching.
        tri_chunk_ids: list[str] = []
        tri_match = _trigram_query(query)
        if tri_match:
            try:
                tri_hits = store.search_fts_trigram(conn, tri_match, limit=LEG_POOL_SIZE)
                tri_chunk_ids = [chunk_id for chunk_id, _rank in tri_hits]
            except sqlite3.OperationalError:
                # An index built before chunks_fts_tri existed cannot serve
                # this leg; the schema_version gate in _open_ready_index
                # normally rules this out, so this is pure belt-and-braces.
                tri_chunk_ids = []

        # Embed the query exactly once; both legs need it at most this once.
        query_vec = embed.encode([query])[0]
        vstore = vectors.open_vectors(vec_path, dimension=embed.DIMENSION)
        # cosine_top_k() returns plain (int, float) tuples, not an ndarray or
        # a view into vstore's mmap -- verified directly: its return line is
        # `[(int(i), float(scores[i])) for i in top_indices]`, and `scores`
        # is itself freshly allocated (matrix_norm @ query_norm), never
        # store.array itself. So no array crosses this boundary at all, and
        # there is nothing here that could later observe vectors.f32 being
        # remapped by an append -- moot anyway, since search() is a single
        # synchronous, read-only call that never appends within its own
        # lifetime. vectors.read() (which hands back store.array[rows], a
        # genuine copy under fancy indexing, not a view) is never called by
        # this module at all.
        vec_hits = vectors.cosine_top_k(vstore, query_vec, k=LEG_POOL_SIZE)
        vec_rows_ranked = [row for row, _score in vec_hits]

        row_to_chunk = _fetch_chunk_ids_by_vec_row(conn, vec_rows_ranked)
        vec_chunk_ids = [row_to_chunk[row] for row in vec_rows_ranked if row in row_to_chunk]

        all_chunk_ids = list(
            set(bm25_chunk_ids)
            | set(vec_chunk_ids)
            | set(bm25_any_chunk_ids)
            | set(tri_chunk_ids)
            | set(phrase_chunk_ids)
        )
        chunk_hits = _fetch_chunk_hits(conn, all_chunk_ids)

        all_episode_ids = list({hit.episode_id for hit in chunk_hits.values()})
        episode_rows = _fetch_episode_rows(conn, all_episode_ids)
    finally:
        conn.close()

    bm25_filtered = _filter_leg(bm25_chunk_ids, chunk_hits, episode_rows, filters)
    vec_filtered = _filter_leg(vec_chunk_ids, chunk_hits, episode_rows, filters)
    bm25_any_filtered = _filter_leg(bm25_any_chunk_ids, chunk_hits, episode_rows, filters)
    tri_filtered = _filter_leg(tri_chunk_ids, chunk_hits, episode_rows, filters)
    phrase_filtered = _filter_leg(phrase_chunk_ids, chunk_hits, episode_rows, filters)

    # Preliminary five-leg fusion, used only to build the reranker's
    # candidate shortlist (the "prefetch" stage) -- see search/rerank.py's
    # module docstring for why a second, precise pass over just this
    # shortlist recovers signal a single mean-pooled vector discards.
    prelim_fused = reciprocal_rank_fusion(
        bm25_filtered,
        vec_filtered,
        bm25_any_filtered,
        tri_filtered,
        phrase_filtered,
        k=RRF_K,
        weights=(1.0, 1.0, OR_LEG_WEIGHT, TRIGRAM_LEG_WEIGHT, PHRASE_LEG_WEIGHT),
    )
    rerank_chunk_ids = rerank_leg(query, prelim_fused, chunk_hits)

    fused = reciprocal_rank_fusion(
        bm25_filtered,
        vec_filtered,
        bm25_any_filtered,
        tri_filtered,
        phrase_filtered,
        rerank_chunk_ids,
        k=RRF_K,
        weights=(1.0, 1.0, OR_LEG_WEIGHT, TRIGRAM_LEG_WEIGHT, PHRASE_LEG_WEIGHT, RERANK_LEG_WEIGHT),
    )
    rolled_up = roll_up_to_episodes(fused, chunk_hits)

    is_subagent_map = {
        episode_id: episode.is_subagent for episode_id, episode in episode_rows.items()
    }
    boosted = apply_main_session_boost(
        {episode_id: score for episode_id, (score, _hit) in rolled_up.items()},
        is_subagent_map,
        boost=main_session_boost,
    )
    rolled_up = {
        episode_id: (boosted[episode_id], hit) for episode_id, (_score, hit) in rolled_up.items()
    }

    return _build_response(
        rolled_up,
        episode_rows,
        limit=effective_limit,
        token_budget=budget,
        clamped=clamped,
        is_stale=is_stale,
        stale_count=stale_count_value,
        score_ceiling=SCORE_CEILING,
    )


def _read_file_cursors(db_path: Path) -> dict[Path, FileCursor]:
    """Load stored file cursors for staleness comparison.

    Read-only and best-effort: returns {} if the index doesn't exist or
    predates the session_files table, rather than raising -- staleness
    reporting is an adjunct, never a reason to fail.
    """
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT path, size, mtime, byte_offset, first_line_hash FROM session_files"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {
        Path(row[0]): FileCursor(
            path=Path(row[0]),
            size=row[1],
            mtime=row[2],
            byte_offset=row[3],
            first_line_hash=row[4],
        )
        for row in rows
    }


def staleness_summary(
    project_dir: Path,
    discovered_files: list[tuple[Path, int, float]] | None = None,
) -> staleness.StalenessReport:
    """Cheap stat-and-cursor staleness check across this project's transcripts.

    Detects staleness (appended, rewritten, or vanished files) for reporting
    and to gate bounded tail repair on the search path. Search performs
    bounded repair on appended sessions within strict byte and time caps,
    refuses unbounded backlogs with an actionable message, and never performs
    unbounded indexing (D13). This is a thin, independently-testable adapter
    over staleness.detect_staleness for status() and other callers that need
    to report or act on staleness.

    `discovered_files` defaults to a real discovery.discover_sessions()
    scan at the scope the index was BUILT with (the persisted `scope` meta
    key, falling back to project_dir for legacy indexes); pass it explicitly
    (as tests do) to avoid depending on the real ~/.claude/projects tree.
    Reading the persisted scope is load-bearing, not cosmetic: an index built
    with `--scope /old/path` compared against a project_dir-scoped discovery
    would see every cursor file as vanished, report total staleness, and
    steer the user toward the destructive rebuild the shrink guard exists to
    prevent.
    """
    db_path, _vec_path = _index_paths(project_dir)
    cursors = _read_file_cursors(db_path)
    if discovered_files is None:
        effective_scope = indexer_support.read_persisted_scope(db_path) or str(project_dir)
        discovered_files = [
            (session.path, session.size, session.mtime)
            for session in discovery.discover_sessions(project_dir, scope=effective_scope)
        ]
        # Index-owned notes (notes.py) are indexed alongside transcripts, so
        # staleness must see them too -- omitted, every indexed note would
        # read as VANISHED (the same trap the persisted scope closes above).
        discovered_files += [
            (n.path, n.size, n.mtime) for n in notes.discover_notes(db_path.parent)
        ]
        # External roots (SSGREP_TRANSCRIPT_DIRS) are indexed too, so
        # staleness must see them for the same reason -- omitted, every
        # external session would read as VANISHED.
        discovered_files += [(s.path, s.size, s.mtime) for s in discovery_roots.discover_external()]
    return staleness.detect_staleness(discovered_files, cursors)


# Re-exports for backward compatibility with existing code that accesses these via search._NAME
__all__ = [
    "INDEX_DIRNAME",
    "RRF_K",
    "MAIN_SESSION_BOOST",
    "DEFAULT_RESULT_COUNT",
    "MAX_RESULT_COUNT",
    "LEG_POOL_SIZE",
    "TOKEN_BUDGET_OVERHEAD_PER_CARD",
    "CHARS_PER_TOKEN",
    "RERANK_LEG_WEIGHT",
    "SCORE_CEILING",
    "rerank_leg",
    "reciprocal_rank_fusion",
    "roll_up_to_episodes",
    "apply_main_session_boost",
    "search",
    "staleness_summary",
    "_read_file_cursors",
    "_ChunkHit",
    "_EpisodeRow",
    "_fts_query",
    "_index_paths",
    "_fetch_chunk_hits",
    "_apply_token_budget",
    "_rank_episodes",
    "_estimate_tokens",
    "_truncate_text",
    "_build_response",
    "_normalize_dt",
    "_sortable_epoch",
    "_chunk_passes_content_type",
    "_episode_passes_filters",
    "_filter_leg",
]
