"""Tests for the T12 query generation stage.

``generate_queries`` is pure computation over ``sessions.parquet`` (no LLM,
no network), so the tests generate in-memory from the committed parquet and
assert the plan's acceptance criteria: >= 600 queries, >= 60 per class, both
splits populated, every class x runtime cell in train >= 5, zero anchor
leakage in paraphrase/multi-hop queries, an episode-safe train/holdout split,
answerable + pairwise-distinct cross-runtime queries, and byte-identical
determinism (including against the committed ``queries.jsonl`` artifact).
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from eval.datasetgen.generate_queries import (
    RUNTIMES,
    SCENARIO_CLASSES,
    _build_index,
    _shared_content_words,
    generate_queries,
    write_queries,
)

PARQUET_PATH = Path(__file__).parents[2] / "eval" / "datasetgen" / "sessions.parquet"
COMMITTED_QUERIES = Path(__file__).parents[2] / "eval" / "datasetgen" / "queries.jsonl"


def test_counts_and_strata() -> None:
    queries = generate_queries(PARQUET_PATH)

    assert len(queries) >= 600
    per_class: dict[str, int] = {}
    for query in queries:
        per_class[query["class"]] = per_class.get(query["class"], 0) + 1
    for class_name in SCENARIO_CLASSES:
        assert per_class[class_name] >= 60, (class_name, per_class[class_name])

    splits = {query["split"] for query in queries}
    assert splits == {"train", "holdout"}

    train_cells: dict[tuple[str, str], int] = {}
    for query in queries:
        if query["split"] == "train":
            key = (query["class"], query["runtime"])
            train_cells[key] = train_cells.get(key, 0) + 1
    for class_name in SCENARIO_CLASSES:
        for runtime in RUNTIMES:
            count = train_cells.get((class_name, runtime), 0)
            assert count >= 5, (class_name, runtime, count)


def test_no_episode_belongs_to_both_splits() -> None:
    queries = generate_queries(PARQUET_PATH)

    train_episodes: set[str] = set()
    holdout_episodes: set[str] = set()
    for query in queries:
        if query["split"] == "train":
            train_episodes.update(query["target_episode_ids"])
        else:
            holdout_episodes.update(query["target_episode_ids"])
    overlap = train_episodes & holdout_episodes
    assert not overlap, f"{len(overlap)} episodes targeted by both splits"


def test_cross_runtime_queries_answerable() -> None:
    queries = generate_queries(PARQUET_PATH)
    episodes = _build_index(pq.read_table(PARQUET_PATH).to_pylist()).episodes

    cross = [q for q in queries if q["class"] == "cross-runtime/project-scoped"]
    assert len(cross) == 86
    for query in cross:
        project = query["anchors"][0] if query["anchors"] else ""
        query_without_project = query["query"].replace(project, "")
        for episode_id in query["target_episode_ids"]:
            episode = episodes[episode_id]
            target_text = episode["prompt"] + " " + episode["response"]
            assert _shared_content_words(query_without_project, target_text) >= 1, (
                query["id"],
                episode_id,
                query["query"],
            )


def test_cross_runtime_query_texts_unique() -> None:
    queries = generate_queries(PARQUET_PATH)

    cross_texts = [q["query"] for q in queries if q["class"] == "cross-runtime/project-scoped"]
    assert len(cross_texts) == 86
    assert len(set(cross_texts)) == 86


def test_no_anchor_leakage() -> None:
    queries = generate_queries(PARQUET_PATH)
    for query in queries:
        if query["class"] not in ("paraphrase", "multi-hop"):
            continue
        lowered = query["query"].lower()
        for anchor in query["anchors"]:
            assert anchor.lower() not in lowered, (
                query["id"],
                query["class"],
                anchor,
                query["query"],
            )


def test_determinism(tmp_path: Path) -> None:
    first = generate_queries(PARQUET_PATH)
    second = generate_queries(PARQUET_PATH)
    assert first == second

    out = tmp_path / "queries.jsonl"
    write_queries(first, out)
    assert COMMITTED_QUERIES.exists(), "committed queries.jsonl artifact missing"
    assert out.read_bytes() == COMMITTED_QUERIES.read_bytes()

    # Every line parses as JSON with the documented schema fields.
    for line in out.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for field in (
            "id",
            "query",
            "class",
            "anchors",
            "target_episode_ids",
            "target_session_ids",
            "subagent_only",
            "split",
            "runtime",
            "hard_negative_episode_ids",
            "grounding",
        ):
            assert field in record, (record["id"], field)
        assert record["split"] in ("train", "holdout")
        assert record["class"] in SCENARIO_CLASSES
        assert record["runtime"] in RUNTIMES
        assert record["target_episode_ids"]
