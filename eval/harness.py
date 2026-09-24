"""DEPRECATED: legacy evaluation harness, superseded by ``eval.run_eval``.

This module is kept only as a thin shim for stragglers that still import
``_private_data_dir`` (and the other symbols ``tests/eval/*`` exercise). Do
not build new evaluations on it.

The standardized benchmark lives in ``eval.run_eval``::

    python -m eval.run_eval --dataset v1 --arm default [--json-out PATH]

It measures the pinned metric suite (nDCG exp-log2, RR, P, R at the plan's
cutoffs) against the frozen ``eval/dataset/v1`` artifact with full provenance,
named arms, and derived gates. The legacy 66-query label set was deleted;
``load_queries`` now requires an explicit path and ``run`` fails loudly
rather than silently measuring nothing.

The functions below are unchanged from the pre-overhaul harness so existing
callers keep working: ``_private_data_dir``, ``_dir_size``, ``_percentile``,
``_token_vector_count``, ``build_private_index``, ``validate_targets``,
``rank_episodes_for_query``, ``QueryResult``, ``evaluate``, ``run``,
``derived_gates``, ``assemble_payload``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.store import EPISODES_TABLE, LanceStore
from ssgrep.store.paths import database_dir
from ssgrep.utilities.types import IndexStats

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LIMIT = 10
EVAL_TOKEN_BUDGET = 1_000_000


def _dir_size(path: Path) -> int:
    """Recursive byte total of every file beneath ``path``.

    Mirrors ``src/ssgrep/services/observability.py:_dir_size`` so the harness
    reports the same number ``ssgrep status`` would for the same directory,
    without importing a private symbol across the package boundary.
    """
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile (numpy's default method)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _token_vector_count(db_path: Path) -> dict[str, int | str]:
    """Total stored token vectors, summed from the multivector column.

    One columnar scan of the stored ``vector`` column -- no model calls, no
    re-embedding -- so the count describes exactly what is on disk. When the
    scan is not possible the count is omitted and a provenance note explains
    why instead of failing the run.
    """
    try:
        import lancedb
        import pyarrow.compute as pc

        chunks = lancedb.connect(str(db_path)).open_table("chunks")
        column = chunks.search().select(["vector"]).to_arrow().column("vector")
        # pyarrow.compute exposes these kernels dynamically; ty cannot see them.
        total = pc.sum(  # ty: ignore[unresolved-attribute]
            pc.list_value_length(column)  # ty: ignore[unresolved-attribute]
        ).as_py()
    except Exception as error:
        return {"token_vector_count_note": f"token_vector_count not derivable: {error}"}
    return {"token_vector_count": int(total or 0)}


@contextmanager
def _private_data_dir(index_dir: Path) -> Iterator[None]:
    previous = os.environ.get("SSGREP_DATA_DIR")
    os.environ["SSGREP_DATA_DIR"] = str(index_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SSGREP_DATA_DIR", None)
        else:
            os.environ["SSGREP_DATA_DIR"] = previous


def load_queries(path: Path | None = None) -> list[dict]:
    """Load one JSON object per line from ``path``.

    The legacy default label file was removed with the pre-overhaul label
    set; pass an explicit path or use ``eval.run_eval`` against a frozen
    dataset instead.
    """
    if path is None:
        raise ValueError(
            "eval.harness is deprecated: the legacy label set was removed. "
            "Use `python -m eval.run_eval --dataset v1 --arm default` "
            "against a frozen dataset instead."
        )
    rows: list[dict] = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_private_index(index_dir: Path, *, rebuild: bool = True) -> IndexStats:
    """Build the real multi-runtime pipeline without touching the live index."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with _private_data_dir(index_dir):
        return indexer.index(
            rebuild=rebuild,
            allow_shrink=True,
            scope=str(PROJECT_DIR),
        )


def validate_targets(index_dir: Path, queries: list[dict]) -> None:
    """Refuse to score labels whose targets are absent from this corpus."""
    with _private_data_dir(index_dir):
        repository = LanceStore()
        available = {
            str(row["episode_id"])
            for row in repository.rows(
                EPISODES_TABLE,
                columns=["episode_id"],
                limit=10_000_000,
            )
        }
    missing = [
        row["id"] for row in queries if not available.intersection(row["target_episode_ids"])
    ]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(queries)} labelled queries have no target in this corpus; "
            "regenerate or restore labels before measuring retrieval: " + ", ".join(missing[:10])
        )


def rank_episodes_for_query(
    index_dir: Path,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[str], dict[str, float]]:
    """Return production-ranked refs and their displayed scores."""
    with _private_data_dir(index_dir):
        response = search_module.search(
            query,
            limit=limit,
            token_budget=EVAL_TOKEN_BUDGET,
        )
    ranked = [card.ref for card in response.results]
    return ranked, {card.ref: card.score for card in response.results}


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
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[QueryResult], list[float]]:
    """Rank every query and return per-query results plus per-query latency (ms).

    The ranking computation is unchanged by timing; each sample is the wall
    time around one production ``ssgrep.search.search`` call.
    """
    results: list[QueryResult] = []
    latencies_ms: list[float] = []
    for query_row in queries:
        started = time.perf_counter()
        ranked, scores = rank_episodes_for_query(index_dir, query_row["query"], limit=limit)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        targets = set(query_row["target_episode_ids"])
        rank = next(
            (
                position
                for position, episode_id in enumerate(ranked, start=1)
                if episode_id in targets
            ),
            None,
        )
        results.append(
            QueryResult(
                id=query_row["id"],
                query=query_row["query"],
                query_class=query_row["class"],
                subagent_only=query_row["subagent_only"],
                rank=rank,
                reciprocal_rank=(1.0 / rank) if rank else 0.0,
                hit_at_10=rank is not None,
                top_result_episode_id=ranked[0] if ranked else None,
                top_result_score=scores.get(ranked[0]) if ranked else None,
            )
        )
    return results, latencies_ms


def _aggregate(rows: list[QueryResult]) -> dict[str, int | float | None]:
    count = len(rows)
    if count == 0:
        return {"n": 0, "recall_at_10": None, "mrr": None}
    return {
        "n": count,
        "recall_at_10": sum(result.hit_at_10 for result in rows) / count,
        "mrr": sum(result.reciprocal_rank for result in rows) / count,
    }


def summarize(results: list[QueryResult]) -> dict[str, dict[str, int | float | None]]:
    summary = {"overall": _aggregate(results)}
    for query_class in sorted({result.query_class for result in results}):
        summary[f"class:{query_class}"] = _aggregate(
            [result for result in results if result.query_class == query_class]
        )
    summary["subagent_only"] = _aggregate([result for result in results if result.subagent_only])
    return summary


def run(
    index_dir: Path,
    *,
    rebuild: bool = True,
    limit: int = DEFAULT_LIMIT,
) -> tuple[
    list[QueryResult],
    dict[str, dict[str, int | float | None]],
    IndexStats,
    dict[str, int | float | str],
]:
    """Build, rank, and measure one full evaluation against ``index_dir``.

    Deprecated: this entry point measured the legacy 66-query label set,
    which was removed. It now fails loudly via ``load_queries``; use
    ``eval.run_eval`` against a frozen dataset instead.

    Latency is measured warm: one labelled query is ranked and discarded
    first (model load, index open, and first-touch costs land there), then
    every labelled query is wall-timed through the production search path.
    """
    build_started = time.perf_counter()
    stats = build_private_index(index_dir, rebuild=rebuild)
    build_seconds = time.perf_counter() - build_started

    queries = load_queries()
    validate_targets(index_dir, queries)
    if queries:
        rank_episodes_for_query(index_dir, queries[0]["query"], limit=limit)  # discarded warmup
    results, latencies_ms = evaluate(index_dir, queries, limit=limit)
    summary = summarize(results)

    with _private_data_dir(index_dir):
        db_path = database_dir()
    metrics: dict[str, int | float | str] = {
        "index_size_bytes": _dir_size(db_path),
        "latency_p50_ms": _percentile(latencies_ms, 0.50),
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "build_seconds": build_seconds,
    }
    metrics.update(_token_vector_count(db_path))
    return results, summary, stats, metrics


def derived_gates(
    summary: Mapping[str, Mapping[str, int | float | None]],
    metrics: Mapping[str, int | float | str],
) -> dict[str, float | None]:
    """Gates later optimization tasks must pass, derived from this run.

    recall_floor/mrr_floor allow the plan's 0.01 absolute quality drop;
    size_ceiling/p95_ceiling are the plan's 50%-of-baseline and
    1.2x-baseline ceilings.
    """
    recall = summary["overall"]["recall_at_10"]
    mrr = summary["overall"]["mrr"]
    return {
        "recall_floor": None if recall is None else recall - 0.01,
        "mrr_floor": None if mrr is None else mrr - 0.01,
        "size_ceiling": 0.5 * float(metrics["index_size_bytes"]),
        "p95_ceiling": 1.2 * float(metrics["latency_p95_ms"]),
    }


def assemble_payload(
    results: list[QueryResult],
    summary: Mapping[str, Mapping[str, int | float | None]],
    stats: IndexStats,
    metrics: Mapping[str, int | float | str],
    *,
    date: str,
    task: str,
    label: str,
    db_path: Path,
) -> dict:
    """One result file's full content: metrics, gates, ranking detail, provenance."""
    from eval import provenance

    payload: dict = {
        "summary": dict(summary),
        "per_query": [asdict(result) for result in results],
        **metrics,
        **derived_gates(summary, metrics),
    }
    payload["provenance"] = provenance.build_provenance(
        date=date,
        task=task,
        label=label,
        stats=stats,
        db_path=db_path,
    )
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the production ssgrep retrieval evaluation")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--task", default="current multi-runtime retrieval baseline")
    args = parser.parse_args()

    index_dir = args.index_dir or Path(tempfile.mkdtemp(prefix="ssgrep-eval-"))
    results, summary, stats, metrics = run(index_dir, rebuild=not args.no_rebuild)

    print(f"Index: {stats.episode_count} episodes, {stats.chunk_count} chunks")
    for key, aggregate in summary.items():
        if aggregate["n"] == 0:
            continue
        print(
            f"  {key:20s} n={aggregate['n']:3d}  "
            f"recall@10={aggregate['recall_at_10']:.3f}  mrr={aggregate['mrr']:.3f}"
        )
    print(
        f"  index_size_bytes={metrics['index_size_bytes']}  "
        f"latency_p50_ms={metrics['latency_p50_ms']:.1f}  "
        f"latency_p95_ms={metrics['latency_p95_ms']:.1f}  "
        f"build_seconds={metrics['build_seconds']:.1f}"
    )

    if args.json_out:
        with _private_data_dir(index_dir):
            db_path = database_dir()
        payload = assemble_payload(
            results,
            summary,
            stats,
            metrics,
            date=date.today().isoformat(),
            task=args.task,
            label="production multi-runtime pipeline",
            db_path=db_path,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
