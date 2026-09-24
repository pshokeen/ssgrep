"""Pinned ir_measures metric engine for ssgrep retrieval evaluation.

This module is the single sanctioned place where retrieval quality becomes
numbers. Every measure is computed by ir_measures (``iter_calc`` /
``calc_aggregate``); nothing here re-implements metric math.

Pinned formulations (normative in the plan; mirrored in ``METRIC_DEFINITIONS``
and stamped into every report under ``metric_definitions``)::

    nDCG@5, nDCG@10   graded 0-3 relevance, exp-log2 formulation
                      gain = 2**rel - 1, discount = log2(rank + 1)
    RR@10 (primary), RR@50   reciprocal rank of the first grade>0 doc
    P@5, P@10               fraction of top-k that is grade>0 (denominator k)
    R@10, R@50, R@100       fraction of ALL grade>0 docs retrieved in top-k

Hard rules implemented here (rationale in the plan):

* Query strings are mapped onto the integer topic ids the gdeval perl
  provider requires (its script rejects non-numeric topic ids), then mapped
  back in the report. The mapping is deterministic: sorted evaluated ids
  -> 1..N (so equal query sets always map identically).
* Queries whose qrels contain no grade>0 doc are EXCLUDED from the nDCG and
  RR means -- trec_eval reports every such query as 0.0, which would
  pollute the mean. They stay in the P/R means (0.0 is correct there) and
  the count is surfaced as ``excluded_query_count``. The exclusion is applied
  by restricting the inputs passed to ``calc_aggregate``, never by
  hand-rolling mean math on the per-query output.
* A run whose scores are all zero (or that is empty) is flagged
  ``empty_run: true`` and metrics are short-circuited to 0.0 (nothing was
  retrieved; feeding an all-zero run to ir_measures would count the zero-
  scored docs as retrieved). The flag tells consumers those 0.0s do not
  describe retrieval quality.
* gdeval (the nDCG provider) rounds per-query nDCG to 5 decimal places
  (``printf %.5f`` in its perl). rr/p/r pass through pytrec-eval unrounded.
  Raw-call equality tests therefore compare nDCG exactly (both sides
  rounded identically), while hand-math literals tolerate the 5-dp cut.
* k beyond the returned depth is fine: ir_measures computes over the
  available depth, and P@k divides by k exactly as trec_eval does.
* ``dcg='exp-log2'`` is pinned EXPLICITLY on every nDCG measure (never the
  provider default) and recorded in ``METRIC_DEFINITIONS``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import ir_measures
from ir_measures import RR, P, R, nDCG

RESULT_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# The exact measure list the engine can emit. Order here defines the
# per-query record ordering consumers see. Keep in sync with
# METRIC_DEFINITIONS below.
# ---------------------------------------------------------------------------
MEASURES: tuple[Any, ...] = (
    nDCG(dcg="exp-log2") @ 5,
    nDCG(dcg="exp-log2") @ 10,
    RR @ 10,
    RR @ 50,
    P @ 5,
    P @ 10,
    R @ 10,
    R @ 50,
    R @ 100,
)


def _canonical_key(measure: Any) -> str:
    """Stable report key, e.g. ``nDCG(dcg='exp-log2')@5`` -> ``ndcg@5``."""
    return f"{measure.NAME.lower()}@{measure['cutoff']}"


MEASURE_TO_KEY: dict[Any, str] = {m: _canonical_key(m) for m in MEASURES}
KEY_TO_MEASURE: dict[str, Any] = {key: m for m, key in MEASURE_TO_KEY.items()}
ALL_KEYS = tuple(_canonical_key(m) for m in MEASURES)

# Measures whose mean excludes zero-relevant queries (module docstring).
_EXCLUDED_MEAN_KEYS = frozenset({"ndcg@5", "ndcg@10", "rr@10", "rr@50"})
_INCLUDED_MEAN_KEYS = frozenset({"p@5", "p@10", "r@10", "r@50", "r@100"})

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "ndcg": {
        "formulation": ("nDCG with exp-log2 gain: gain = 2**rel - 1, discount = log2(rank + 1)"),
        "dcg": "exp-log2",
        "relevance_scale": "graded 0-3",
        "cutoffs": [5, 10],
        "provider": "ir_measures gdeval",
        "notes": (
            "Requires numeric topic ids (internal mapping). Queries with no "
            "grade>0 doc are excluded from the mean and reported in "
            "excluded_query_count."
        ),
    },
    "rr": {
        "formulation": "reciprocal rank of the first grade>0 doc",
        "cutoffs": [10, 50],
        "primary_cutoff": 10,
        "provider": "ir_measures pytrec-eval",
        "notes": "Queries with no grade>0 doc excluded from the mean.",
    },
    "p": {
        "formulation": "grade>0 docs in top-k / k",
        "cutoffs": [5, 10],
        "provider": "ir_measures pytrec-eval",
        "notes": "k > returned depth: P@k still divides by k (trec_eval).",
    },
    "r": {
        "formulation": "fraction of all grade>0 docs retrieved within top-k",
        "cutoffs": [10, 50, 100],
        "provider": "ir_measures pytrec-eval",
        "semantics": (
            "TREC convention (not Success@k): fraction of relevant docs "
            "retrieved, not whether any was found."
        ),
    },
}


def _topic_map(query_ids: Sequence[str]) -> dict[str, str]:
    """Deterministic original-id -> numeric-topic-id mapping (gdeval wants ints)."""
    sorted_ids = sorted(query_ids)
    return {qid: str(idx) for idx, qid in enumerate(sorted_ids, start=1)}


def _has_positive_relevance(qrels: Mapping[str, Mapping[str, int]], qid: str) -> bool:
    return any(rel > 0 for rel in qrels.get(qid, {}).values())


@dataclass(frozen=True)
class MetricReport:
    """One evaluation's full results plus the metadata needed to interpret it.

    Field order mirrors the payload contract: ``aggregate``, ``per_query``,
    ``metric_definitions``, ``result_schema_version``, then the edge-case
    annotations. ``dataclasses.asdict(report)`` serializes directly.
    """

    aggregate: dict[str, float | None]
    per_query: list[dict[str, Any]]
    metric_definitions: dict[str, dict[str, Any]]
    result_schema_version: str = RESULT_SCHEMA_VERSION
    n_queries: int = 0
    excluded_query_count: int = 0
    excluded_query_ids: tuple[str, ...] = ()
    empty_run: bool = False


def evaluate_run(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    *,
    query_ids: Sequence[str] | None = None,
) -> MetricReport:
    """Compute the full pinned metric suite for one run.

    ``qrels`` maps query ids to graded relevance (0-3); ``run`` maps query
    ids to a ranking of docs with scores (higher = better), in any order.
    ir_measures is the only calculator. Pass ``query_ids`` to evaluate a
    subset (used by the runner to slice summary-by-class); the topic mapping
    is deterministic regardless of subset.
    """
    plain_qrels = {str(qid): dict(docs) for qid, docs in qrels.items()}
    plain_run = {
        str(qid): {doc: float(score) for doc, score in docs.items()} for qid, docs in run.items()
    }

    evaluated_ids = (
        [str(qid) for qid in query_ids] if query_ids is not None else sorted(plain_qrels)
    )

    # Restrict inputs to the evaluated set so run-only queries never leak in.
    evaluated_qrels = {qid: plain_qrels[qid] for qid in evaluated_ids if qid in plain_qrels}
    evaluated_run = {qid: plain_run[qid] for qid in evaluated_ids if qid in plain_run}

    empty_run = not any(score != 0.0 for docs in evaluated_run.values() for score in docs.values())

    excluded_ids = sorted(
        qid for qid in evaluated_ids if not _has_positive_relevance(plain_qrels, qid)
    )
    excluded_topics = {_topic_map(evaluated_ids)[qid] for qid in excluded_ids}

    topic_map = _topic_map(evaluated_ids)
    numeric_qrels = {
        topic_map[qid]: {doc: int(rel) for doc, rel in docs.items()}
        for qid, docs in evaluated_qrels.items()
    }
    numeric_run = {
        topic_map[qid]: {doc: float(score) for doc, score in docs.items()}
        for qid, docs in evaluated_run.items()
    }

    # ------------------------------------------------------------------
    # Per-query records: one row per (query_id, metric) so downstream
    # consumers can pivot without re-running anything.
    # ------------------------------------------------------------------
    per_query: dict[str, dict[str, float | None]] = {
        qid: {key: 0.0 for key in ALL_KEYS} for qid in evaluated_ids
    }
    if not empty_run:
        for metric in ir_measures.iter_calc(list(MEASURES), numeric_qrels, numeric_run):
            original = next(qid for qid, topic in topic_map.items() if topic == metric.query_id)
            per_query[original][_canonical_key(metric.measure)] = float(metric.value)

    per_query_rows: list[dict[str, Any]] = [
        {"query_id": qid, "metric": key, "value": per_query[qid][key]}
        for qid in evaluated_ids
        for key in ALL_KEYS
    ]

    # ------------------------------------------------------------------
    # Aggregates. One calc_aggregate call over the full sub-run covers the
    # P/R family (zero-relevant queries belong to those means); the nDCG/RR
    # family is aggregated over the sub-run with excluded queries dropped.
    # ------------------------------------------------------------------
    aggregate: dict[str, float | None] = {}
    if not evaluated_ids:
        aggregate = {key: None for key in ALL_KEYS}
    elif empty_run:
        aggregate = {key: 0.0 for key in ALL_KEYS}
    else:
        full = ir_measures.calc_aggregate(list(MEASURES), numeric_qrels, numeric_run)
        for key in _INCLUDED_MEAN_KEYS:
            measure = KEY_TO_MEASURE[key]
            aggregate[key] = float(full[measure])

        kept_qrels = {t: d for t, d in numeric_qrels.items() if t not in excluded_topics}
        kept_run = {t: d for t, d in numeric_run.items() if t not in excluded_topics}
        if kept_qrels:
            kept_measures = [KEY_TO_MEASURE[key] for key in ALL_KEYS if key in _EXCLUDED_MEAN_KEYS]
            kept_aggregates = ir_measures.calc_aggregate(kept_measures, kept_qrels, kept_run)
            for key in sorted(_EXCLUDED_MEAN_KEYS):
                aggregate[key] = float(kept_aggregates[KEY_TO_MEASURE[key]])
        else:
            for key in sorted(_EXCLUDED_MEAN_KEYS):
                aggregate[key] = None

    return MetricReport(
        aggregate=aggregate,
        per_query=per_query_rows,
        metric_definitions=dict(METRIC_DEFINITIONS),
        n_queries=len(evaluated_ids),
        excluded_query_count=len(excluded_ids),
        excluded_query_ids=tuple(excluded_ids),
        empty_run=empty_run,
    )


def group_summary(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float | None]]:
    """Aggregate per group, mirroring the roundup-by-class shape.

    ``groups`` maps a summary key (``"overall"``, ``"class:<c>"``,
    ``"subagent_only"``) to the query ids in that slice. Each slice is
    evaluated by ``evaluate_run`` so the same exclusion rules apply.
    """
    return {
        name: evaluate_run(qrels, run, query_ids=list(ids)).aggregate
        for name, ids in groups.items()
    }


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "MEASURES",
    "MEASURE_TO_KEY",
    "KEY_TO_MEASURE",
    "METRIC_DEFINITIONS",
    "MetricReport",
    "evaluate_run",
    "group_summary",
]
