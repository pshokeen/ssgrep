"""Retrieval evaluation harness for ssgrep.

Measures recall@10 and MRR of the real search pipeline against the labelled
query set in eval/queries.jsonl (see eval/build_labels.py for how targets
were derived, and eval/README.md for the full methodology writeup).

Index isolation: this harness NEVER builds into <project_dir>/.ssgrep — that
directory is the shared, live index other agents and tests in this repo
depend on concurrently. Every index this module builds goes into a caller-
supplied private `index_dir` via indexer.index()'s index_dir override, which
exists precisely "to keep index state separate from the scanned project,
mainly for tests" (its own docstring). Discovery scope (which real sessions
get indexed) is still the real project_dir — only storage location moves.

Querying reuses search.py's own scoring primitives directly (reciprocal_rank_
fusion, roll_up_to_episodes, apply_main_session_boost, _rank_episodes) against
an explicit (db_path, vec_path) pair, rather than going through search.search()
(which hardcodes project_dir/".ssgrep" and has no index_dir parameter). This
is the same ranking algorithm production code runs -- imported, not
reimplemented -- with only the staleness/tail-repair/token-budget shaping
skipped, none of which affects ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ssgrep import embed, indexer, store, vectors
from ssgrep import search as search_module
from ssgrep.types import IndexStats

PROJECT_DIR = Path(__file__).resolve().parent.parent
QUERIES_PATH = Path(__file__).resolve().parent / "queries.jsonl"
DEFAULT_LIMIT = 10


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Load the labelled query set. Each row's target_episode_ids is a set of
    acceptable answers (see module docstring in build_labels.py for why more
    than one can be correct on a real, messy corpus)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_private_index(index_dir: Path, *, rebuild: bool = True) -> IndexStats:
    """Build (or rebuild) an index scoped to the real project but stored at
    `index_dir`, never at PROJECT_DIR/.ssgrep."""
    index_dir.mkdir(parents=True, exist_ok=True)
    return indexer.index(PROJECT_DIR, rebuild=rebuild, quiet=True, index_dir=index_dir)


def _index_paths_for(index_dir: Path) -> tuple[Path, Path]:
    gen_store = store.GenerationalStore(index_dir)
    return gen_store.get_index_path(), gen_store.get_vector_path()


def rank_episodes_for_query(
    db_path: Path,
    vec_path: Path,
    query: str,
    *,
    main_session_boost: float,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[str], dict[str, float]]:
    """Return (ranked episode_ids[:limit], full episode_id -> boosted score map).

    Reuses search.py's own BM25 leg, vector leg, RRF fusion, roll-up, and
    main-session boost -- the exact functions search.search() calls -- against
    an explicit db/vector path pair instead of a project_dir.
    """
    conn = search_module._open_ready_index(db_path, vec_path)
    try:
        bm25_hits = store.search_fts(
            conn, search_module._fts_query(query), limit=search_module.LEG_POOL_SIZE
        )
        bm25_chunk_ids = [chunk_id for chunk_id, _rank in bm25_hits]

        bm25_any_chunk_ids: list[str] = []
        if len(set(query.split())) > 1:
            bm25_any_hits = store.search_fts(
                conn, search_module._fts_query_any(query), limit=search_module.LEG_POOL_SIZE
            )
            bm25_any_chunk_ids = [chunk_id for chunk_id, _rank in bm25_any_hits]

        tri_chunk_ids: list[str] = []
        tri_match = search_module._trigram_query(query)
        if tri_match:
            tri_hits = store.search_fts_trigram(conn, tri_match, limit=search_module.LEG_POOL_SIZE)
            tri_chunk_ids = [chunk_id for chunk_id, _rank in tri_hits]

        phrase_chunk_ids: list[str] = []
        phrase_match = search_module._phrase_query(query)
        if phrase_match:
            phrase_hits = store.search_fts(conn, phrase_match, limit=search_module.LEG_POOL_SIZE)
            phrase_chunk_ids = [chunk_id for chunk_id, _rank in phrase_hits]

        query_vec = embed.encode([query])[0]
        vstore = vectors.open_vectors(vec_path, dimension=embed.DIMENSION)
        vec_hits = vectors.cosine_top_k(vstore, query_vec, k=search_module.LEG_POOL_SIZE)
        vec_rows_ranked = [row for row, _score in vec_hits]

        row_to_chunk = search_module._fetch_chunk_ids_by_vec_row(conn, vec_rows_ranked)
        vec_chunk_ids = [row_to_chunk[row] for row in vec_rows_ranked if row in row_to_chunk]

        all_chunk_ids = list(
            set(bm25_chunk_ids)
            | set(vec_chunk_ids)
            | set(bm25_any_chunk_ids)
            | set(tri_chunk_ids)
            | set(phrase_chunk_ids)
        )
        chunk_hits = search_module._fetch_chunk_hits(conn, all_chunk_ids)

        all_episode_ids = list({hit.episode_id for hit in chunk_hits.values()})
        episode_rows = search_module._fetch_episode_rows(conn, all_episode_ids)
    finally:
        conn.close()

    prelim_fused = search_module.reciprocal_rank_fusion(
        bm25_chunk_ids,
        vec_chunk_ids,
        bm25_any_chunk_ids,
        tri_chunk_ids,
        phrase_chunk_ids,
        k=search_module.RRF_K,
        weights=(
            1.0,
            1.0,
            search_module.OR_LEG_WEIGHT,
            search_module.TRIGRAM_LEG_WEIGHT,
            search_module.PHRASE_LEG_WEIGHT,
        ),
    )
    rerank_chunk_ids = search_module.rerank_leg(query, prelim_fused, chunk_hits)

    fused = search_module.reciprocal_rank_fusion(
        bm25_chunk_ids,
        vec_chunk_ids,
        bm25_any_chunk_ids,
        tri_chunk_ids,
        phrase_chunk_ids,
        rerank_chunk_ids,
        k=search_module.RRF_K,
        weights=(
            1.0,
            1.0,
            search_module.OR_LEG_WEIGHT,
            search_module.TRIGRAM_LEG_WEIGHT,
            search_module.PHRASE_LEG_WEIGHT,
            search_module.RERANK_LEG_WEIGHT,
        ),
    )
    rolled_up = search_module.roll_up_to_episodes(fused, chunk_hits)

    is_subagent_map = {
        episode_id: episode.is_subagent for episode_id, episode in episode_rows.items()
    }
    boosted = search_module.apply_main_session_boost(
        {episode_id: score for episode_id, (score, _hit) in rolled_up.items()},
        is_subagent_map,
        boost=main_session_boost,
    )
    ranked = search_module._rank_episodes(boosted, episode_rows)
    return ranked[:limit], boosted


@dataclass(frozen=True)
class QueryResult:
    id: str
    query: str
    query_class: str
    subagent_only: bool
    rank: int | None
    reciprocal_rank: float
    hit_at_10: bool
    top_result_episode_id: str | None
    top_result_score: float | None


def evaluate(
    index_dir: Path,
    queries: list[dict],
    *,
    main_session_boost: float,
    limit: int = DEFAULT_LIMIT,
) -> list[QueryResult]:
    db_path, vec_path = _index_paths_for(index_dir)
    results = []
    for q in queries:
        ranked, scores = rank_episodes_for_query(
            db_path, vec_path, q["query"], main_session_boost=main_session_boost, limit=limit
        )
        targets = set(q["target_episode_ids"])
        rank = None
        for position, episode_id in enumerate(ranked, start=1):
            if episode_id in targets:
                rank = position
                break
        results.append(
            QueryResult(
                id=q["id"],
                query=q["query"],
                query_class=q["class"],
                subagent_only=q["subagent_only"],
                rank=rank,
                reciprocal_rank=(1.0 / rank) if rank else 0.0,
                hit_at_10=rank is not None,
                top_result_episode_id=ranked[0] if ranked else None,
                top_result_score=scores.get(ranked[0]) if ranked else None,
            )
        )
    return results


def _agg(rows: list[QueryResult]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "recall_at_10": None, "mrr": None}
    return {
        "n": n,
        "recall_at_10": sum(1 for r in rows if r.hit_at_10) / n,
        "mrr": sum(r.reciprocal_rank for r in rows) / n,
    }


def summarize(results: list[QueryResult]) -> dict:
    out = {"overall": _agg(results)}
    for cls in sorted({r.query_class for r in results}):
        out[f"class:{cls}"] = _agg([r for r in results if r.query_class == cls])
    out["subagent_only"] = _agg([r for r in results if r.subagent_only])
    return out


def run(
    index_dir: Path,
    *,
    main_session_boost: float,
    rebuild: bool = True,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[QueryResult], dict, IndexStats]:
    """End-to-end: build a private index, run every labelled query, summarize."""
    stats = build_private_index(index_dir, rebuild=rebuild)
    queries = load_queries()
    results = evaluate(index_dir, queries, main_session_boost=main_session_boost, limit=limit)
    summary = summarize(results)
    return results, summary, stats


def main() -> None:
    import argparse
    import tempfile
    from datetime import date

    from eval import provenance

    parser = argparse.ArgumentParser(description="Run the ssgrep retrieval eval harness")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument(
        "--main-session-boost", type=float, default=search_module.MAIN_SESSION_BOOST
    )
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="eval/harness.py baseline run",
        help="Short description recorded in the output file's provenance block.",
    )
    args = parser.parse_args()

    index_dir = args.index_dir or Path(tempfile.mkdtemp(prefix="ssgrep-eval-"))
    results, summary, stats = run(
        index_dir, main_session_boost=args.main_session_boost, rebuild=not args.no_rebuild
    )

    print(f"Index: {stats.episode_count} episodes, {stats.chunk_count} chunks")
    print(f"main_session_boost = {args.main_session_boost}")
    for key, agg in summary.items():
        if agg["n"] == 0:
            continue
        print(
            f"  {key:20s} n={agg['n']:3d}  "
            f"recall@10={agg['recall_at_10']:.3f}  mrr={agg['mrr']:.3f}"
        )

    if args.json_out:
        db_path, _vec_path = _index_paths_for(index_dir)
        payload = {
            "main_session_boost": args.main_session_boost,
            "summary": summary,
            "per_query": [r.__dict__ for r in results],
            "provenance": provenance.build_provenance(
                date=date.today().isoformat(),
                task=args.task,
                label=f"baseline, main_session_boost={args.main_session_boost}",
                stats=stats,
                db_path=db_path,
            ),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
