"""Golden-fixture tests for the eval/metrics.py ir_measures engine.

Each fixture has a hand-computed expected value (exp-log2 gains, natural-log
discounts -- equivalent up to a constant since nDCG normalizes), plus a
stronger assertion: the engine's per-query output is EXACTLY what a raw
``ir_measures.iter_calc`` call returns on the same (numeric-mapped) inputs.
Raw-call equality is the reproducibility pin; the literals are the
human-auditable anchors.

Fixtures intentionally cover the plan's edge cases: graded qrels, score ties,
zero-relevant queries, k larger than the returned depth, and runs whose scores
are all zero (``empty_run``).
"""

from __future__ import annotations

import math
from typing import Any

import ir_measures
import pytest

from eval.metrics import (
    ALL_KEYS,
    MEASURES,
    METRIC_DEFINITIONS,
    RESULT_SCHEMA_VERSION,
    evaluate_run,
)

# ---------------------------------------------------------------------------
# Fixture: 5 queries exercising every contract.
# ---------------------------------------------------------------------------
GOLDEN_QRELS: dict[str, dict[str, int]] = {
    # Perfect graded ranking: rels 3,2,1 match run order exactly.
    "q-perfect": {"e-1": 3, "e-2": 2, "e-3": 1},
    # Tied scores between e-a and e-b (both rel 2); e-neut (rel 0) ranks
    # first. Hand math is tie-break-independent: both tied docs carry the
    # same grade, so nDCG and RR do not depend on which sits at rank 2.
    "q-tie": {"e-a": 2, "e-b": 2, "e-neut": 0},
    # Zero-relevant query: every qrel is grade 0. Excluded from nDCG/RR
    # means (reported in excluded_query_count); P/R still treat it as 0.0.
    "q-zero": {"e-n": 0},
    # Positive-relevance labels but the query never appears in the run:
    # trec_eval convention counts it as retrieving nothing (0.0 everywhere,
    # but it HAS relevant docs so it stays in the nDCG/RR means).
    "q-unretrieved": {"e-u": 2},
}

GOLDEN_RUN: dict[str, dict[str, float]] = {
    "q-perfect": {"e-1": 9.0, "e-2": 8.0, "e-3": 7.0},
    "q-tie": {"e-neut": 6.0, "e-b": 5.0, "e-a": 5.0},
    "q-zero": {"e-n": 4.9},
    # q-unretrieved deliberately absent.
}

# ---------------------------------------------------------------------------
# Hand-computed expected values per query.
# exp-log2: gain(g) = 2**g - 1; discount ln(rank + 1); ideal DCG capped at k.
#
# q-perfect: gains at ranks 1,2,3 are 7,3,1 == ideal order -> nDCG@k == 1.0
#   for every cutoff. First relevant at rank 1 -> RR == 1.0. P@k == 3/k.
#
# q-tie: ranking is [e-neut(rel0), e-a/b(rel2, gain 3), e-b/a(rel2)].
#   DCG@3   = 0/ln2 + 3/ln3 + 3/ln4         = 3*(1/ln3 + 1/ln4)
#   IDCG@3  = 3/ln2 + 3/ln3 + 0 (rank<=3)   = 3*(1/ln2 + 1/ln3)
#   nDCG@3  = (1/ln3 + 1/ln4)/(1/ln2 + 1/ln3)  ~ 0.693415
#   nDCG@5 and @10 agree (ranking is depth 3; DCG and IDCG frozen after 3).
#   RR == 1/2 = 0.5 (first relevant at rank 2; tie-break irrelevant).
#   P@3 = 2/3, P@5 = 2/5, P@10 = 2/10.  R@10 = R@50 = R@100 = 1.0 (both found).
# ---------------------------------------------------------------------------

_INV_LN = {k: 1.0 / math.log(k) for k in (2, 3, 4)}

_TIE_NDCG = (_INV_LN[3] + _INV_LN[4]) / (_INV_LN[2] + _INV_LN[3])

GOLDEN_PER_QUERY: dict[str, dict[str, float]] = {
    "q-perfect": {
        "ndcg@5": 1.0,
        "ndcg@10": 1.0,
        "rr@10": 1.0,
        "rr@50": 1.0,
        "p@5": 3 / 5,
        "p@10": 3 / 10,
        "r@10": 1.0,
        "r@50": 1.0,
        "r@100": 1.0,
    },
    "q-tie": {
        "ndcg@5": _TIE_NDCG,
        "ndcg@10": _TIE_NDCG,
        "rr@10": 0.5,
        "rr@50": 0.5,
        "p@5": 2 / 5,
        "p@10": 2 / 10,
        "r@10": 1.0,
        "r@50": 1.0,
        "r@100": 1.0,
    },
    "q-zero": {
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
        "rr@10": 0.0,
        "rr@50": 0.0,
        "p@5": 0.0,
        "p@10": 0.0,
        "r@10": 0.0,
        "r@50": 0.0,
        "r@100": 0.0,
    },
    "q-unretrieved": {
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
        "rr@10": 0.0,
        "rr@50": 0.0,
        "p@5": 0.0,
        "p@10": 0.0,
        "r@10": 0.0,
        "r@50": 0.0,
        "r@100": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Raw ir_measures helpers: the exact calls the engine is required to match.
# ---------------------------------------------------------------------------


def _id_map(query_ids: list[str]) -> dict[str, str]:
    """The documented deterministic mapping (sorted ids -> 1..N)."""
    return {qid: str(i) for i, qid in enumerate(sorted(query_ids), start=1)}


def _key(measure: Any) -> str:
    return f"{measure.NAME.lower()}@{measure['cutoff']}"


def _raw_per_query(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    ids: list[str],
) -> dict[str, dict[str, float]]:
    """A raw ``iter_calc`` over the engine's measure list, ids mapped.

    Mirrors exactly what the engine does internally so the comparison
    validates the wrapper (mapping, row shape) rather than the math.
    """
    id_map = _id_map(ids)
    numeric_qrels = {
        id_map[qid]: {doc: int(rel) for doc, rel in docs.items()} for qid, docs in qrels.items()
    }
    numeric_run = {
        id_map[qid]: {doc: float(score) for doc, score in docs.items()}
        for qid, docs in run.items()
        if qid in id_map
    }
    inverse = {value: key for key, value in id_map.items()}
    out: dict[str, dict[str, float]] = {qid: {key: 0.0 for key in ALL_KEYS} for qid in ids}
    for metric in ir_measures.iter_calc(list(MEASURES), numeric_qrels, numeric_run):
        out[inverse[metric.query_id]][_key(metric.measure)] = float(metric.value)
    return out


def _raw_aggregate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    ids: list[str],
) -> dict[str, float]:
    id_map = _id_map(ids)
    numeric_qrels = {
        id_map[qid]: {doc: int(rel) for doc, rel in docs.items()} for qid, docs in qrels.items()
    }
    numeric_run = {
        id_map[qid]: {doc: float(score) for doc, score in docs.items()}
        for qid, docs in run.items()
        if qid in id_map
    }
    out = ir_measures.calc_aggregate(list(MEASURES), numeric_qrels, numeric_run)
    return {_key(measure): float(out[measure]) for measure in MEASURES}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pinned_definition_is_exp_log2():
    """The exp-log2 pin is normative; never rely on the provider default."""
    assert METRIC_DEFINITIONS["ndcg"]["dcg"] == "exp-log2"
    for measure in MEASURES:
        if measure.NAME.lower() == "ndcg":
            assert measure["dcg"] == "exp-log2"


def test_golden_fixtures():
    report = evaluate_run(GOLDEN_QRELS, GOLDEN_RUN)
    for qid, expected in GOLDEN_PER_QUERY.items():
        row = {m["metric"]: m["value"] for m in report.per_query if m["query_id"] == qid}
        for metric, value in expected.items():
            # gdeval (nDCG) rounds per-query output to 5 decimals; rr/p/r
            # pass through pytrec-eval unrounded.
            tolerance = 1e-4 if metric.startswith("ndcg") else 1e-9
            assert row[metric] == pytest.approx(value, abs=tolerance), (
                f"{qid} {metric}: engine={row[metric]!r} hand={value!r}"
            )


def test_golden_fixtures_match_raw_ir_calc():
    """Engine output is EXACTLY a raw ir_measures call (no hand math)."""
    report = evaluate_run(GOLDEN_QRELS, GOLDEN_RUN)
    raw = _raw_per_query(GOLDEN_QRELS, GOLDEN_RUN, list(GOLDEN_QRELS))
    for row in report.per_query:
        assert row["value"] == pytest.approx(raw[row["query_id"]][row["metric"]], abs=1e-12), row


def test_zero_relevant_query():
    """Queries with no grade>0 doc leave the nDCG/RR means but not P/R."""
    report = evaluate_run(GOLDEN_QRELS, GOLDEN_RUN)
    assert report.excluded_query_count == 1
    assert report.excluded_query_ids == ("q-zero",)
    # Mean over the three queries with relevant docs: perfect (1.0) + tie +
    # unretrieved (0.0 -- it has rel docs so it stays in the nDCG/RR means).
    tie_ndcg = GOLDEN_PER_QUERY["q-tie"]["ndcg@5"]
    # gdeval rounds each per-query nDCG to 5 decimals before averaging.
    tie_ndcg_rounded = round(tie_ndcg, 5)
    assert report.aggregate["ndcg@5"] == pytest.approx((1.0 + tie_ndcg_rounded + 0.0) / 3)
    assert report.aggregate["rr@10"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    # P@10 mean includes the zero-relevant query (0/10) and the unretrieved
    # query (0/10) -- they belong to the P means per trec_eval.
    assert report.aggregate["p@10"] == pytest.approx((0.3 + 0.2 + 0.0 + 0.0) / 4)
    assert report.aggregate["r@10"] == pytest.approx((1.0 + 1.0 + 0.0 + 0.0) / 4)
    assert report.aggregate["r@50"] == pytest.approx((1.0 + 1.0 + 0.0 + 0.0) / 4)


def test_aggregate_matches_raw_aggregate():
    """Aggregate equals calc_aggregate over the same excluded/kept spaces."""
    report = evaluate_run(GOLDEN_QRELS, GOLDEN_RUN)
    # P/R family: full space (zero-relevant query included).
    for key in ("p@5", "p@10", "r@10", "r@50", "r@100"):
        assert report.aggregate[key] == pytest.approx(
            _raw_aggregate(GOLDEN_QRELS, GOLDEN_RUN, list(GOLDEN_QRELS))[key]
        )
    # nDCG/RR family: space with the excluded query dropped.
    kept_qrels = {q: d for q, d in GOLDEN_QRELS.items() if q != "q-zero"}
    kept_ids = sorted(kept_qrels)
    for key in ("ndcg@5", "ndcg@10", "rr@10", "rr@50"):
        assert report.aggregate[key] == pytest.approx(
            _raw_aggregate(kept_qrels, GOLDEN_RUN, kept_ids)[key]
        )


def test_ties_nonzero_recall_and_rr_non_trivial():
    """The tie fixture must actually exercise ranking, not all-perfect."""
    row = {
        m["metric"]: m["value"]
        for m in evaluate_run(GOLDEN_QRELS, GOLDEN_RUN).per_query
        if m["query_id"] == "q-tie"
    }
    assert row["rr@10"] == pytest.approx(0.5)
    assert row["ndcg@5"] < 1.0
    assert row["r@10"] == pytest.approx(1.0)
    assert row["r@50"] == 1.0  # both relevant docs retrieved


def test_metric_definitions_present_and_canonical():
    report = evaluate_run(GOLDEN_QRELS, GOLDEN_RUN)
    assert report.result_schema_version == RESULT_SCHEMA_VERSION
    assert report.metric_definitions == METRIC_DEFINITIONS
    assert {"ndcg", "rr", "p", "r"} == set(METRIC_DEFINITIONS)
    defined_keys = {
        f"{family}@{cutoff}"
        for family, block in METRIC_DEFINITIONS.items()
        for cutoff in block["cutoffs"]
    }
    assert set(ALL_KEYS) == defined_keys


def test_empty_run_scores_zero_flagged():
    """An all-zero run is flagged; metrics remain well-defined 0.0s."""
    qrels = {"q1": {"d1": 2}}
    run = {"q1": {"d1": 0.0}}  # zero score: nothing was retrieved
    report = evaluate_run(qrels, run)
    assert report.empty_run is True
    assert report.aggregate["p@5"] == 0.0
    assert report.aggregate["ndcg@5"] == 0.0
    assert report.per_query[0]["value"] == 0.0


def test_empty_inputs():
    report = evaluate_run({}, {})
    assert report.n_queries == 0
    assert report.aggregate == {key: None for key in ALL_KEYS}
    assert report.per_query == []
    assert report.empty_run is True


def test_k_greater_than_depth():
    """k beyond returned depth: computed over available depth; P@k divides by k."""
    qrels = {"q-shallow": {"d-s1": 3, "d-s2": 0}}
    run = {"q-shallow": {"d-s1": 4.0, "d-s2": 2.0}}  # depth 2
    report = evaluate_run(qrels, run)
    row = {m["metric"]: m["value"] for m in report.per_query if m["query_id"] == "q-shallow"}
    assert row["ndcg@10"] == pytest.approx(1.0)  # gain 7 at rank 1, ideal capped
    assert row["p@10"] == pytest.approx(1 / 10)  # 1 relevant / 10, not /2
    assert row["r@10"] == pytest.approx(1.0)
    assert row["r@100"] == pytest.approx(1.0)
