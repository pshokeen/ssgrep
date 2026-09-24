"""Judge / qrel refinement stage (Task 14 of the retrieval-eval overhaul).

Consumes the T12 ``queries.jsonl`` and a T13-built draft index, retrieves a
candidate set per query (union of the production final top-k and the deep
prefetch pool top-k), and has an LLM-as-judge grade every (query,
episode-excerpt) pair 0-3 against a fixed rubric. The judge uses the same
provider as generation (OpenRouter when ``OPENROUTER_API_KEY`` is set,
OpenAI otherwise), a stronger tier is permitted, temperature is pinned to 0,
and a token-accounting budget cap (default $50) is enforced BEFORE any API
batch is submitted.

Merge rules (label-quality logic, all disagreements logged to
``judge_disagreements.jsonl``, never silently merged):

- Grounding seeds (grade 3) override the judge only where the judge graded
  the episode < 3 AND the grounding text actually supports the query. When
  the grounding text does not support the query the judge grade is kept and
  the disagreement is logged as ``grounding_unsupported``.
- Confusable hard negatives must grade 0. When the judge grades one > 0 the
  judgment is regenerated (bounded at ``MAX_HARD_NEGATIVE_REGENERATIONS``);
  if no regeneration yields 0 the hard-negative designation is authoritative
  and the grade is forced to 0, logged as ``hard_negative_forced_zero``.

The judge NEVER sees grounding labels: the rubric prompt contains only the
query and the episode excerpt.

Outputs (written to ``--out``, default ``eval/datasetgen/``):

- ``qrels.tsv`` — BEIR interchange format, header ``query-id\\tcorpus-id\\tscore``.
- ``judge_stats.json`` — model id, per-class agreement vs grounding on the
  judged sample, cost (token-accounting estimate), counts per grade.
- ``judge_disagreements.jsonl`` — one JSON object per merge disagreement.

Usage:

    uv run python -m eval.datasetgen.judge_qrels \
        --parquet eval/datasetgen/sessions.parquet \
        --queries eval/datasetgen/queries.jsonl \
        --out eval/datasetgen \
        --index-dir /tmp/judge-index

Tests inject a mocked ``judge_fn`` (no network); the budget test proves the
cap aborts before any judge call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from eval import ranking
from eval.datasetgen.emitters import (
    claude as claude_emitter,
    codex as codex_emitter,
    opencode as opencode_emitter,
    pi as pi_emitter,
    prime_agent as prime_agent_emitter,
)
from eval.datasetgen.ingest import BenchmarkIndex, build_benchmark_index
from ssgrep.store import EPISODES_TABLE, LanceStore

#: Judge model (same provider as generation; a stronger tier is permitted).
DEFAULT_MODEL: str = "xiaomi/mimo-v2.5-pro"
#: Judge temperature is pinned to 0 (deterministic grading).
DEFAULT_TEMPERATURE: float = 0.0
#: One-time budget cap for the whole judge run (Metis resolution #5).
DEFAULT_COST_CAP_USD: float = 50.0
#: Candidate depth per query: union of final top-k and prefetch pool top-k.
TOP_K: int = 20
#: Deep prefetch pool depth (the T2 ``prefetch_episode_ranking`` default).
PREFETCH_POOL_DEPTH: int = 800
#: Bounded regeneration loop for confusable hard negatives.
MAX_HARD_NEGATIVE_REGENERATIONS: int = 2
#: Excerpt cap per side (prompt / response) shown to the judge.
EXCERPT_CHARS: int = 600
#: The graded relevance scale (unjudged episodes are grade 0 by convention).
GRADES: tuple[int, ...] = (0, 1, 2, 3)

#: Estimated USD per 1M tokens (input, output); used ONLY for the projected
#: cost guardrail, never for billing.
PRICE_PER_MTOKENS: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-4o": (2.5, 10.0),
    "anthropic/claude-sonnet-4": (3.0, 15.0),
    "xiaomi/mimo-v2.5-pro": (0.435, 0.87),
}
_FALLBACK_PRICE: tuple[float, float] = (0.5, 1.5)

#: Fixed grading rubric. The judge sees ONLY this plus the query and excerpt
#: (never grounding labels -- bias guard).
RUBRIC_PROMPT: str = (
    "You are grading how relevant an episode from a coding-session transcript "
    "is to a search query. Grade on a 0-3 scale:\n\n"
    "3 = directly and fully answers the query; the episode is exactly what the "
    "query is looking for.\n"
    "2 = relevant and partially answers the query; addresses the topic but "
    "misses part of the ask.\n"
    "1 = marginally related; tangentially touches the topic but does not "
    "answer the query.\n"
    "0 = not relevant; does not address the query at all.\n\n"
    "Respond with ONLY a single integer grade (0, 1, 2, or 3). No explanation, "
    "no punctuation."
)

_STOPWORDS: frozenset[str] = frozenset(
    "a an and are as at be been being but by for from has have had how i in is it its of on "
    "or that the this to was we were what when where which who will with you your our my "
    "there here then now also can could would should may might must do does did not no yes "
    "so if then than very just more most some any all each every both few such only own same "
    "too".split()
)


class BudgetError(RuntimeError):
    """Raised when the projected judge cost exceeds the configured cap."""


@dataclass(frozen=True)
class JudgeResult:
    """Outcome of one judge run: final qrels, stats, and disagreements."""

    qrels: list[tuple[str, str, int]]
    stats: dict[str, Any]
    disagreements: list[dict[str, Any]]
    projected_cost_usd: float


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    """Load ``queries.jsonl`` (one JSON object per line)."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_candidates(
    index_dir: Path,
    query: str,
    *,
    top_k: int = TOP_K,
    pool_depth: int = PREFETCH_POOL_DEPTH,
) -> list[str]:
    """Union of the final top-k and the prefetch pool top-k episode ids.

    Order: final first, then prefetch episodes not already present. The union
    keeps both the production page and the deep prefetch pool in scope, which
    is the candidate set the judge grades.
    """
    final, _elapsed = ranking.final_ranking(index_dir, query, limit=top_k)
    prefetch, _elapsed = ranking.prefetch_episode_ranking(index_dir, query, pool_depth=pool_depth)
    ordered: list[str] = []
    seen: set[str] = set()
    for episode_id, _score in final + prefetch[:top_k]:
        if episode_id not in seen:
            seen.add(episode_id)
            ordered.append(episode_id)
    return ordered


def _episode_excerpt(store: LanceStore, episode_id: str, *, max_chars: int = EXCERPT_CHARS) -> str:
    """Bounded prompt + response excerpt for one episode."""
    safe = episode_id.replace("'", "''")
    rows = store.rows(
        EPISODES_TABLE,
        where=f"episode_id = '{safe}'",
        columns=["prompt_text", "response_text"],
    )
    if not rows:
        return ""
    prompt = str(rows[0].get("prompt_text") or "")
    response = str(rows[0].get("response_text") or "")
    return f"PROMPT:\n{prompt[:max_chars]}\n\nRESPONSE:\n{response[:max_chars]}"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) for budget accounting."""
    return max(1, len(text) // 4)


def _pair_cost_usd(query: str, excerpt: str, model: str) -> float:
    """Projected USD for one (query, excerpt) judge call."""
    input_tokens = (
        _estimate_tokens(RUBRIC_PROMPT) + _estimate_tokens(query) + _estimate_tokens(excerpt)
    )
    output_tokens = 8  # a single grade digit plus whitespace
    rate = PRICE_PER_MTOKENS.get(model, _FALLBACK_PRICE)
    return (input_tokens * rate[0] + output_tokens * rate[1]) / 1_000_000.0


def _projected_cost_usd(
    pairs: list[dict[str, Any]],
    model: str,
    *,
    regeneration_pairs: list[dict[str, Any]] | None = None,
    max_regenerations: int = MAX_HARD_NEGATIVE_REGENERATIONS,
) -> float:
    """Projected USD for the whole judge run.

    Includes the worst-case hard-negative regeneration calls (each regenerated
    pair costs one extra call per loop), so the accounting never understates
    the run's cost.
    """
    total = sum(_pair_cost_usd(pair["query"], pair["excerpt"], model) for pair in pairs)
    for pair in regeneration_pairs or []:
        total += max_regenerations * _pair_cost_usd(pair["query"], pair["excerpt"], model)
    return total


def _resolve_api() -> tuple[str, str]:
    """(endpoint, api_key) for the judge provider (same provider as generation)."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return "https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        return "https://api.openai.com/v1", os.environ["OPENAI_API_KEY"]
    raise ValueError(
        "neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set; the judge stage requires one"
    )


def _chat_completion(
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    *,
    endpoint: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 8,
) -> str:
    """One chat-completions call; returns the assistant text."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def _parse_grade(text: str) -> int:
    """Extract the 0-3 grade from a judge response; fail loudly otherwise."""
    match = re.search(r"\b([0-3])\b", text)
    if match is None:
        raise ValueError(f"judge response did not contain a 0-3 grade: {text!r}")
    return int(match.group(1))


def _judge_grade(
    model: str,
    query: str,
    excerpt: str,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> int:
    """Grade one (query, excerpt) pair through the judge model (temperature 0)."""
    if api_key is None or endpoint is None:
        endpoint, api_key = _resolve_api()
    messages = [
        {"role": "system", "content": RUBRIC_PROMPT},
        {"role": "user", "content": f"QUERY: {query}\n\nEPISODE EXCERPT:\n{excerpt}"},
    ]
    text = _chat_completion(model, api_key, messages, endpoint=endpoint)
    return _parse_grade(text)


def _grounding_supports(query: str, grounding: dict[str, Any]) -> bool:
    """True when the grounding text plausibly supports the query.

    The grounding text is the target episode's own prompt/response, so a query
    authored from it shares content words by construction; this check guards
    against degenerate seeds (empty or unrelated text).
    """
    text = f"{grounding.get('prompt', '')} {grounding.get('response', '')}".lower()
    query_words = {
        word
        for word in re.split(r"[^a-zA-Z0-9]+", query.lower())
        if word and word not in _STOPWORDS
    }
    if not query_words:
        return True
    return any(word in text for word in query_words)


def _merge_grounding(
    queries: list[dict[str, Any]],
    grades: dict[tuple[str, str], int],
    *,
    known_episodes: set[str] | None = None,
) -> tuple[dict[tuple[str, str], int], list[dict[str, Any]]]:
    """Grounding seeds (grade 3) override the judge only where the judge
    graded < 3 AND the grounding text actually supports the query.

    Every override or unsupported seed is logged as a disagreement; nothing is
    merged silently. Seeds for episodes the retrieval never surfaced are added
    as grade 3 (they are targets by construction) unless the episode is absent
    from the index entirely.
    """
    final = dict(grades)
    disagreements: list[dict[str, Any]] = []
    for query in queries:
        for grounding in query.get("grounding", []):
            episode_id = grounding["episode_id"]
            key = (query["id"], episode_id)
            judge_grade = final.get(key)
            if judge_grade is None:
                if known_episodes is not None and episode_id not in known_episodes:
                    disagreements.append(
                        {
                            "query_id": query["id"],
                            "episode_id": episode_id,
                            "judge_grade": None,
                            "final_grade": None,
                            "reason": "grounding_episode_missing",
                        }
                    )
                    continue
                final[key] = 3
                continue
            if judge_grade == 3:
                continue
            if _grounding_supports(query["query"], grounding):
                final[key] = 3
                disagreements.append(
                    {
                        "query_id": query["id"],
                        "episode_id": episode_id,
                        "judge_grade": judge_grade,
                        "final_grade": 3,
                        "reason": "grounding_override",
                    }
                )
            else:
                disagreements.append(
                    {
                        "query_id": query["id"],
                        "episode_id": episode_id,
                        "judge_grade": judge_grade,
                        "final_grade": judge_grade,
                        "reason": "grounding_unsupported",
                    }
                )
    return final, disagreements


def _enforce_hard_negatives(
    queries: list[dict[str, Any]],
    grades: dict[tuple[str, str], int],
    judge_fn: Callable[[str, str], int],
    excerpts: dict[str, str],
    disagreements: list[dict[str, Any]],
    *,
    max_regenerations: int = MAX_HARD_NEGATIVE_REGENERATIONS,
) -> dict[tuple[str, str], int]:
    """Confusable hard negatives must grade 0.

    When the judge grades one > 0 the judgment is regenerated (bounded at
    ``max_regenerations``); if no regeneration yields 0 the hard-negative
    designation is authoritative and the grade is forced to 0. Every
    regeneration/force is logged as a disagreement.
    """
    final = dict(grades)
    for query in queries:
        for episode_id in query.get("hard_negative_episode_ids", []):
            key = (query["id"], episode_id)
            grade = final.get(key)
            if grade is None or grade == 0:
                continue
            excerpt = excerpts.get(episode_id, "")
            regenerated = 0
            while regenerated < max_regenerations:
                regenerated += 1
                new_grade = judge_fn(query["query"], excerpt)
                if new_grade == 0:
                    final[key] = 0
                    disagreements.append(
                        {
                            "query_id": query["id"],
                            "episode_id": episode_id,
                            "judge_grade": grade,
                            "final_grade": 0,
                            "reason": "hard_negative_regenerated_zero",
                            "regenerations": regenerated,
                        }
                    )
                    break
            else:
                final[key] = 0
                disagreements.append(
                    {
                        "query_id": query["id"],
                        "episode_id": episode_id,
                        "judge_grade": grade,
                        "final_grade": 0,
                        "reason": "hard_negative_forced_zero",
                        "regenerations": regenerated,
                    }
                )
    return final


def _build_stats(
    queries: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    final_grades: dict[tuple[str, str], int],
    disagreements: list[dict[str, Any]],
    model: str,
    projected_cost_usd: float,
) -> dict[str, Any]:
    """judge_stats.json payload: model, agreement vs grounding, cost, grades."""
    grade_counts = {str(grade): 0 for grade in GRADES}
    for grade in final_grades.values():
        grade_counts[str(grade)] += 1

    per_class: dict[str, list[int]] = {}
    for query in queries:
        for grounding in query.get("grounding", []):
            judge_grade = final_grades.get((query["id"], grounding["episode_id"]))
            if judge_grade is None:
                continue
            per_class.setdefault(query["class"], []).append(judge_grade)
    agreement: dict[str, float] = {}
    judged_total = agreed_total = 0
    for class_name in sorted(per_class):
        judged = len(per_class[class_name])
        agreed = sum(1 for judge_grade in per_class[class_name] if judge_grade == 3)
        agreement[f"class:{class_name}"] = agreed / judged
        judged_total += judged
        agreed_total += agreed
    if judged_total:
        agreement["overall"] = agreed_total / judged_total

    reasons: dict[str, int] = {}
    for disagreement in disagreements:
        reason = disagreement["reason"]
        reasons[reason] = reasons.get(reason, 0) + 1

    candidate_keys = {(pair["query_id"], pair["episode_id"]) for pair in pairs}
    covered = sum(1 for key in candidate_keys if key in final_grades)
    candidate_coverage = covered / len(candidate_keys) if candidate_keys else 1.0

    return {
        "model": model,
        "temperature": DEFAULT_TEMPERATURE,
        "cost_usd": round(projected_cost_usd, 4),
        "projected_cost_usd": round(projected_cost_usd, 4),
        "n_queries": len(queries),
        "n_candidate_pairs": len(pairs),
        "n_qrels_rows": len(final_grades),
        "grade_counts": grade_counts,
        "agreement_vs_grounding": agreement,
        "disagreement_count": len(disagreements),
        "disagreement_reasons": reasons,
        "candidate_coverage": candidate_coverage,
    }


def write_qrels(qrels: list[tuple[str, str, int]], out_path: str | Path) -> None:
    """Write BEIR-format qrels TSV (header ``query-id\\tcorpus-id\\tscore``)."""
    lines = ["query-id\tcorpus-id\tscore"]
    for query_id, episode_id, grade in sorted(qrels):
        lines.append(f"{query_id}\t{episode_id}\t{grade}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stats(stats: dict[str, Any], out_path: str | Path) -> None:
    """Write judge_stats.json (sorted keys for determinism)."""
    Path(out_path).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_disagreements(disagreements: list[dict[str, Any]], out_path: str | Path) -> None:
    """Write judge_disagreements.jsonl (one JSON object per line)."""
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in disagreements]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def judge_index(
    handle: BenchmarkIndex,
    queries_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    top_k: int = TOP_K,
    pool_depth: int = PREFETCH_POOL_DEPTH,
    judge_fn: Callable[[str, str], int] | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    out_dir: str | Path | None = None,
) -> JudgeResult:
    """Run the judge stage against an already-built ``BenchmarkIndex``.

    Collects the candidate set per query, enforces the budget cap BEFORE any
    judge call, grades every pair, merges grounding seeds and hard negatives,
    and (when ``out_dir`` is given) writes ``qrels.tsv``, ``judge_stats.json``
    and ``judge_disagreements.jsonl``.
    """
    queries = load_queries(queries_path)
    store = handle.store()

    candidates: dict[str, list[str]] = {}
    for query in queries:
        candidates[query["id"]] = collect_candidates(
            handle.index_dir, query["query"], top_k=top_k, pool_depth=pool_depth
        )

    all_episode_ids = sorted({episode for eps in candidates.values() for episode in eps})
    excerpts = {episode_id: _episode_excerpt(store, episode_id) for episode_id in all_episode_ids}

    pairs: list[dict[str, Any]] = []
    for query in queries:
        for episode_id in candidates[query["id"]]:
            pairs.append(
                {
                    "query_id": query["id"],
                    "query": query["query"],
                    "episode_id": episode_id,
                    "excerpt": excerpts[episode_id],
                }
            )

    hard_negative_keys = {
        (query["id"], episode_id)
        for query in queries
        for episode_id in query.get("hard_negative_episode_ids", [])
    }
    regeneration_pairs = [
        pair for pair in pairs if (pair["query_id"], pair["episode_id"]) in hard_negative_keys
    ]
    projected = _projected_cost_usd(pairs, model, regeneration_pairs=regeneration_pairs)
    if projected > cost_cap_usd:
        raise BudgetError(
            f"projected cost ${projected:.2f} exceeds cap ${cost_cap_usd:.2f}; "
            "aborting before any API batch"
        )

    judge = judge_fn
    if judge is None:

        def _default_judge(query: str, excerpt: str) -> int:
            return _judge_grade(model, query, excerpt, api_key=api_key, endpoint=endpoint)

        judge = _default_judge

    grades: dict[tuple[str, str], int] = {}
    for pair in pairs:
        grades[(pair["query_id"], pair["episode_id"])] = judge(pair["query"], pair["excerpt"])

    final_grades, disagreements = _merge_grounding(
        queries, grades, known_episodes=set(all_episode_ids)
    )
    final_grades = _enforce_hard_negatives(queries, final_grades, judge, excerpts, disagreements)

    qrels = sorted(
        (query_id, episode_id, grade) for (query_id, episode_id), grade in final_grades.items()
    )
    stats = _build_stats(queries, pairs, final_grades, disagreements, model, projected)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_qrels(qrels, out / "qrels.tsv")
        write_stats(stats, out / "judge_stats.json")
        write_disagreements(disagreements, out / "judge_disagreements.jsonl")

    return JudgeResult(
        qrels=qrels,
        stats=stats,
        disagreements=disagreements,
        projected_cost_usd=projected,
    )


def _emit_dataset(parquet_path: str | Path, dataset_dir: str | Path) -> Path:
    """Emit all five runtime transcripts from sessions.parquet (T7-T11)."""
    source = Path(parquet_path)
    dataset = Path(dataset_dir)
    transcripts = dataset / "transcripts"
    claude_emitter.emit_sessions(source, transcripts / "claude")
    codex_emitter.emit_sessions(source, transcripts / "codex")
    pi_emitter.emit_sessions(source, transcripts / "pi")
    rows = pq.read_table(source).to_pylist()
    prime_agent_emitter.emit(rows, transcripts / "prime-agent")
    opencode_emitter.emit_parquet(source, dataset / "opencode.db")
    return dataset


def judge_dataset(
    dataset_dir: str | Path,
    index_dir: str | Path,
    queries_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    top_k: int = TOP_K,
    pool_depth: int = PREFETCH_POOL_DEPTH,
    judge_fn: Callable[[str, str], int] | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    out_dir: str | Path | None = None,
) -> JudgeResult:
    """Build the draft index via T13, then run the judge stage."""
    handle = build_benchmark_index(dataset_dir, index_dir)
    return judge_index(
        handle,
        queries_path,
        model=model,
        cost_cap_usd=cost_cap_usd,
        top_k=top_k,
        pool_depth=pool_depth,
        judge_fn=judge_fn,
        api_key=api_key,
        endpoint=endpoint,
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).with_name("sessions.parquet"),
        help="T6 sessions.parquet input (emitted into a temp dataset when --dataset is absent)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).with_name("queries.jsonl"),
        help="T12 queries.jsonl input",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent,
        help="output dir for qrels.tsv / judge_stats.json / judge_disagreements.jsonl",
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="draft index dir")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="pre-emitted dataset dir (T13 layout); when absent, emitted from --parquet",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--pool-depth", type=int, default=PREFETCH_POOL_DEPTH)
    args = parser.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print(
            "FATAL: neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set; "
            "the judge stage requires one.",
            file=sys.stderr,
        )
        return 2
    if not args.queries.exists():
        print(f"FATAL: queries file not found: {args.queries}", file=sys.stderr)
        return 2

    index_dir = args.index_dir or Path(tempfile.mkdtemp(prefix="ssgrep-judge-index-"))
    if args.dataset is not None:
        dataset_dir = args.dataset
    else:
        dataset_dir = Path(tempfile.mkdtemp(prefix="ssgrep-judge-dataset-"))
        _emit_dataset(args.parquet, dataset_dir)

    try:
        result = judge_dataset(
            dataset_dir,
            index_dir,
            args.queries,
            model=args.model,
            cost_cap_usd=args.cost_cap_usd,
            top_k=args.top_k,
            pool_depth=args.pool_depth,
            out_dir=args.out,
        )
    except BudgetError as error:
        print(f"ABORT: {error}", file=sys.stderr)
        return 3

    print(
        f"judged {result.stats['n_candidate_pairs']} pairs across "
        f"{result.stats['n_queries']} queries"
    )
    print(f"  projected cost ${result.projected_cost_usd:.2f} (cap ${args.cost_cap_usd:.2f})")
    print(f"  grade counts: {result.stats['grade_counts']}")
    print(f"  disagreements: {result.stats['disagreement_count']}")
    print(
        f"  wrote {args.out / 'qrels.tsv'}, {args.out / 'judge_stats.json'}, "
        f"{args.out / 'judge_disagreements.jsonl'}"
    )
    return 0


__all__ = [
    "BudgetError",
    "DEFAULT_COST_CAP_USD",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "EXCERPT_CHARS",
    "GRADES",
    "JudgeResult",
    "MAX_HARD_NEGATIVE_REGENERATIONS",
    "PREFETCH_POOL_DEPTH",
    "PRICE_PER_MTOKENS",
    "RUBRIC_PROMPT",
    "TOP_K",
    "collect_candidates",
    "judge_dataset",
    "judge_index",
    "load_queries",
    "write_disagreements",
    "write_qrels",
    "write_stats",
]


if __name__ == "__main__":
    sys.exit(main())
