"""Unit tests for the eval harness's size/latency capture and gate derivation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lancedb
import pyarrow as pa
import pytest

from eval import harness
from ssgrep.utilities.types import IndexStats

LABELLED_QUERIES = [
    {
        "id": "q1",
        "query": "alpha query",
        "class": "paraphrase",
        "subagent_only": False,
        "target_episode_ids": ["ep-1"],
    },
    {
        "id": "q2",
        "query": "beta query",
        "class": "exact-identifier",
        "subagent_only": True,
        "target_episode_ids": ["ep-missing"],
    },
]


def _fake_ranked(index_dir: Path, query: str, *, limit: int = 10):
    ranked = ["ep-1"] if query == "alpha query" else ["ep-9"]
    return ranked, {ref: 1.0 for ref in ranked}


def test_dir_size_sums_nested_files_and_handles_missing(tmp_path):
    missing = tmp_path / "missing"
    assert harness._dir_size(missing) == 0

    index_path = tmp_path / "index"
    nested = index_path / "nested"
    nested.mkdir(parents=True)
    (index_path / "one.bin").write_bytes(b"123")
    (nested / "two.bin").write_bytes(b"45678")

    assert harness._dir_size(index_path) == 8


def test_percentile_interpolates_and_rejects_empty():
    sample = [float(value) for value in range(1, 101)]

    assert harness._percentile(sample, 0.50) == 50.5
    assert harness._percentile(sample, 0.95) == pytest.approx(95.05)
    assert harness._percentile([7.0], 0.95) == 7.0

    with pytest.raises(ValueError, match="empty sample"):
        harness._percentile([], 0.5)


def test_evaluate_times_every_query_without_changing_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "rank_episodes_for_query", _fake_ranked)

    results, latencies_ms = harness.evaluate(tmp_path, LABELLED_QUERIES)

    assert [result.id for result in results] == ["q1", "q2"]
    assert results[0].rank == 1 and results[0].hit_at_10
    assert results[1].rank is None and not results[1].hit_at_10
    assert len(latencies_ms) == 2
    assert all(latency > 0 for latency in latencies_ms)


def test_run_warms_up_once_then_times_all_queries(tmp_path, monkeypatch):
    events: list[str] = []

    def fake_build(index_dir: Path, *, rebuild: bool = True):
        events.append("build")
        return SimpleNamespace(
            session_count=0,
            episode_count=0,
            chunk_count=0,
            model_id="m",
            vector_dimension=96,
        )

    def fake_validate(index_dir: Path, queries: list[dict]):
        events.append("validate")

    def fake_ranked(index_dir: Path, query: str, *, limit: int = 10):
        events.append(f"search:{query}")
        return _fake_ranked(index_dir, query, limit=limit)

    monkeypatch.setattr(harness, "build_private_index", fake_build)
    monkeypatch.setattr(harness, "validate_targets", fake_validate)
    monkeypatch.setattr(harness, "load_queries", lambda: LABELLED_QUERIES)
    monkeypatch.setattr(harness, "rank_episodes_for_query", fake_ranked)

    results, summary, stats, metrics = harness.run(tmp_path)

    # One discarded warmup on the first labelled query, then the timed pass over all of them.
    assert events == [
        "build",
        "validate",
        "search:alpha query",
        "search:alpha query",
        "search:beta query",
    ]
    assert summary["overall"]["n"] == 2
    assert metrics["index_size_bytes"] == 0  # no lancedb directory was built by the fake
    assert float(metrics["latency_p50_ms"]) >= 0
    assert float(metrics["latency_p95_ms"]) >= 0
    assert float(metrics["build_seconds"]) >= 0
    assert "token_vector_count_note" in metrics  # no chunks table to scan
    assert isinstance(stats.chunk_count, int)


def test_derived_gates_match_plan_formulas():
    summary = {"overall": {"n": 66, "recall_at_10": 0.879, "mrr": 0.632}}
    metrics = {"index_size_bytes": 1000, "latency_p95_ms": 100.0}

    gates = harness.derived_gates(summary, metrics)

    assert gates["recall_floor"] == pytest.approx(0.869)
    assert gates["mrr_floor"] == pytest.approx(0.622)
    assert gates["size_ceiling"] == pytest.approx(500.0)
    assert gates["p95_ceiling"] == pytest.approx(120.0)


def test_derived_gates_tolerate_empty_summary():
    summary = {"overall": {"n": 0, "recall_at_10": None, "mrr": None}}
    metrics = {"index_size_bytes": 0, "latency_p95_ms": 0.0}

    gates = harness.derived_gates(summary, metrics)

    assert gates["recall_floor"] is None
    assert gates["mrr_floor"] is None


def test_token_vector_count_sums_stored_token_vectors(tmp_path):
    db_path = tmp_path / "lancedb"
    vectors = pa.array(
        [[[1.0] * 4, [2.0] * 4, [3.0] * 4], [[4.0] * 4]],
        type=pa.list_(pa.list_(pa.float32(), 4)),
    )
    table = pa.table({"chunk_id": pa.array(["a", "b"]), "vector": vectors})
    lancedb.connect(str(db_path)).create_table("chunks", table)

    result = harness._token_vector_count(db_path)

    assert result == {"token_vector_count": 4}


def test_token_vector_count_notes_when_not_derivable(tmp_path):
    result = harness._token_vector_count(tmp_path / "lancedb")

    assert "token_vector_count" not in result
    assert "token_vector_count_note" in result


def test_assemble_payload_places_metrics_and_gates_top_level(monkeypatch, tmp_path):
    monkeypatch.setattr("eval.provenance.build_provenance", lambda **kwargs: {"provenance": True})
    result = harness.QueryResult(
        id="q1",
        query="alpha query",
        query_class="paraphrase",
        subagent_only=False,
        rank=1,
        reciprocal_rank=1.0,
        hit_at_10=True,
        top_result_episode_id="ep-1",
        top_result_score=1.0,
    )
    summary = {"overall": {"n": 1, "recall_at_10": 0.5, "mrr": 0.5}}
    stats = IndexStats(
        session_count=0,
        episode_count=0,
        chunk_count=0,
        index_size_bytes=0,
        last_index_time=None,
        model_id="m",
        vector_dimension=96,
        skipped_records=0,
        malformed_records=0,
        schema_version=5,
        data_dir=str(tmp_path),
    )
    metrics = {
        "index_size_bytes": 1000,
        "latency_p50_ms": 50.0,
        "latency_p95_ms": 100.0,
        "build_seconds": 12.5,
        "token_vector_count": 42,
    }

    payload = harness.assemble_payload(
        [result],
        summary,
        stats,
        metrics,
        date="2026-08-21",
        task="test task",
        label="test label",
        db_path=tmp_path,
    )

    assert payload["summary"] == summary
    assert payload["per_query"][0]["id"] == "q1"
    for key, value in metrics.items():
        assert payload[key] == value
    assert payload["recall_floor"] == pytest.approx(0.49)
    assert payload["mrr_floor"] == pytest.approx(0.49)
    assert payload["size_ceiling"] == pytest.approx(500.0)
    assert payload["p95_ceiling"] == pytest.approx(120.0)
    assert payload["provenance"] == {"provenance": True}
