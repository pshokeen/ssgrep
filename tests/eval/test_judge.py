"""Tests for ``eval.datasetgen.judge_qrels`` (Task 14).

The fixture index is built through the REAL indexer on a five-runtime
emitter-output dataset (the T13 pattern) with the fake-embedder boundary.
Judge responses are MOCKED (no network): ``judge_fn`` is injected, and the
budget test proves the cap aborts before any judge call.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.datasetgen.emitters import (
    claude as claude_emitter,
    codex as codex_emitter,
    opencode as opencode_emitter,
    pi as pi_emitter,
    prime_agent as prime_agent_emitter,
)
from eval.datasetgen.ingest import BenchmarkIndex, build_benchmark_index
from eval.datasetgen.judge_qrels import BudgetError, judge_index
from ssgrep import search as search_module
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import EPISODES_TABLE

pytestmark = pytest.mark.slow


def fake_vector(text: str) -> np.ndarray:
    """Deterministic ``(num_tokens, DIMENSION)`` float32 matrix for one text."""
    s = str(text)
    grams = [s[i : i + 3] for i in range(max(1, len(s) - 2))]
    if not grams:
        grams = [s]
    rows = []
    for gram in grams:
        seed = int.from_bytes(gram.encode("utf-8"), "little") % (2**32)
        rng = np.random.default_rng(seed)
        row = rng.standard_normal(DIMENSION).astype(np.float32)
        norm = np.linalg.norm(row)
        rows.append((row / np.maximum(norm, 1e-8)).astype(np.float32))
    return np.stack(rows).astype(np.float32)


class FakePyLateEmbedder:
    """CocoIndex provider stand-in producing ``(num_tokens, DIMENSION)`` matrices."""

    def __init__(self, model_name_or_path: str = "", *, device: str | None = None) -> None:
        self._model = model_name_or_path
        self._device = device

    def encode_many(self, texts: list[str], *, is_query: bool = False) -> list[np.ndarray]:
        return [fake_vector(text) for text in texts]

    async def encode_many_async(
        self, texts: list[str], *, is_query: bool = False
    ) -> list[np.ndarray]:
        return self.encode_many(texts, is_query=is_query)

    def __coco_memo_key__(self) -> object:
        return (self._model, self._device)


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the model boundary for deterministic offline stand-ins."""
    monkeypatch.setattr("ssgrep.indexing.embed.ensure_model_downloaded", lambda *a, **kw: None)
    monkeypatch.setattr(app_mod, "ensure_model_downloaded", lambda *a, **kw: None)
    monkeypatch.setattr(app_mod, "ColBERTEmbedder", FakePyLateEmbedder)
    monkeypatch.setattr(search_module, "_query_matrix", lambda query: fake_vector(str(query)))


def _session(
    runtime: str,
    index: int,
    title: str,
    project: str,
    episodes: list[tuple[str, str]],
) -> dict:
    return {
        "session_id": f"{runtime}-{index:02d}",
        "runtime": runtime,
        "scenario_class": "error-string",
        "difficulty": "easy",
        "episode_length_bucket": "short",
        "title": title,
        "project": project,
        "files_touched": [],
        "tool_names": [],
        "episodes": [{"prompt": prompt, "response": response} for prompt, response in episodes],
    }


ROWS: list[dict] = [
    _session(
        "claude",
        0,
        "Cache invalidation",
        "/fictional/checkout/catalog",
        [
            ("Why is the cache stale?", "The TTL was reset on every read."),
            ("How do I fix it?", "Bump the version key on writes."),
        ],
    ),
    _session(
        "codex",
        0,
        "Retry backoff hang",
        "/fictional/checkout/payments",
        [
            ("Where is the backoff policy?", "It lives in retry.py, capped at 60 seconds."),
            ("How do I test it?", "Run the retry harness with a three-attempt cap."),
        ],
    ),
    _session(
        "pi",
        0,
        "Invoice PDFs arrive empty",
        "/fictional/checkout/billing",
        [("Generated invoice PDFs are blank.", "The template engine resolves the wrong key.")],
    ),
    _session(
        "prime-agent",
        0,
        "Pool ceiling tuning",
        "/fictional/checkout/payments",
        [("Raise the pool ceiling?", "Bump MAX_POOL in the config.")],
    ),
    _session(
        "opencode",
        0,
        "Golden page-count fixtures",
        "/fictional/checkout/billing",
        [
            (
                "How should we verify PDF page counts in CI?",
                "Compare against a golden fixture per template.",
            )
        ],
    ),
]


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Run all five emitters on a hand-built parquet; return (dataset, parquet)."""
    parquet_path = tmp_path / "sessions.parquet"
    names = {key for row in ROWS for key in row}
    columns = {name: [row.get(name) for row in ROWS] for name in names}
    pq.write_table(pa.Table.from_pydict(columns), parquet_path)

    dataset = tmp_path / "dataset"
    transcripts = dataset / "transcripts"
    claude_emitter.emit_sessions(parquet_path, transcripts / "claude")
    codex_emitter.emit_sessions(parquet_path, transcripts / "codex")
    pi_emitter.emit_sessions(parquet_path, transcripts / "pi")
    prime_agent_emitter.emit(ROWS, transcripts / "prime-agent")
    opencode_emitter.emit_parquet(parquet_path, dataset / "opencode.db")
    return dataset, parquet_path


def _build_index(tmp_path: Path) -> BenchmarkIndex:
    """Build the five-runtime draft index (fake-embedder boundary applied)."""
    dataset, _parquet = _build_dataset(tmp_path)
    return build_benchmark_index(dataset, tmp_path / "index")


def _episode_ids(handle: BenchmarkIndex) -> list[str]:
    """Read the actual indexed episode ids (claude ids are path-hashed)."""
    store = handle.store()
    rows = store.rows(EPISODES_TABLE, columns=["episode_id"])
    return sorted(str(row["episode_id"]) for row in rows)


def _write_queries(tmp_path: Path, queries: list[dict]) -> Path:
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(query) for query in queries) + "\n", encoding="utf-8")
    return path


def _make_queries(episode_ids: list[str]) -> list[dict]:
    """Three queries with grounding seeds and one confusable hard negative."""
    target_a, target_b, hard_neg = episode_ids[0], episode_ids[1], episode_ids[2]
    return [
        {
            "id": "para-0001",
            "query": "how did we handle the cache invalidation",
            "class": "paraphrase",
            "anchors": ["cache_invalidation"],
            "anchor_mode": "all",
            "target_episode_ids": [target_a],
            "target_session_ids": [],
            "subagent_only": False,
            "split": "train",
            "runtime": "claude",
            "hard_negative_episode_ids": [hard_neg],
            "grounding": [
                {
                    "episode_id": target_a,
                    "project": "/fictional/checkout/catalog",
                    "runtime": "claude",
                    "prompt": "Why is the cache stale?",
                    "response": "The TTL was reset on every read.",
                }
            ],
            "notes": "",
        },
        {
            "id": "error-0001",
            "query": "retry backoff capped at 60 seconds",
            "class": "error-string",
            "anchors": ["retry.py"],
            "anchor_mode": "all",
            "target_episode_ids": [target_b],
            "target_session_ids": [],
            "subagent_only": False,
            "split": "train",
            "runtime": "codex",
            "hard_negative_episode_ids": [],
            "grounding": [
                {
                    "episode_id": target_b,
                    "project": "/fictional/checkout/payments",
                    "runtime": "codex",
                    "prompt": "Where is the backoff policy?",
                    "response": "It lives in retry.py, capped at 60 seconds.",
                }
            ],
            "notes": "",
        },
        {
            "id": "tool-0001",
            "query": "how was the invoice pdf failure resolved",
            "class": "tool-failure-recovery",
            "anchors": ["invoice"],
            "anchor_mode": "all",
            "target_episode_ids": [episode_ids[3]],
            "target_session_ids": [],
            "subagent_only": False,
            "split": "holdout",
            "runtime": "pi",
            "hard_negative_episode_ids": [],
            "grounding": [
                {
                    "episode_id": episode_ids[3],
                    "project": "/fictional/checkout/billing",
                    "runtime": "pi",
                    "prompt": "Generated invoice PDFs are blank.",
                    "response": "The template engine resolves the wrong key.",
                }
            ],
            "notes": "",
        },
    ]


def _load_beir_qrels(path: Path) -> dict[tuple[str, str], int]:
    """Parse a BEIR-format qrels TSV (header ``query-id\\tcorpus-id\\tscore``)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "query-id\tcorpus-id\tscore"
    qrels: dict[tuple[str, str], int] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        query_id, corpus_id, score = line.split("\t")
        qrels[(query_id, corpus_id)] = int(score)
    return qrels


def _load_disagreements(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_qrel_emission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None) -> None:
    """Final qrels TSV parses under a BEIR loader; seeds preserved; disagreements logged."""
    handle = _build_index(tmp_path)
    episode_ids = _episode_ids(handle)
    queries = _make_queries(episode_ids)
    queries_path = _write_queries(tmp_path, queries)

    calls: list[tuple[str, str]] = []

    def judge_fn(query: str, excerpt: str) -> int:
        calls.append((query, excerpt))
        return 1  # judge disagrees with grounding seeds and hard negatives

    out_dir = tmp_path / "out"
    result = judge_index(handle, queries_path, judge_fn=judge_fn, out_dir=out_dir)

    # qrels TSV parses under a BEIR-style loader.
    qrels = _load_beir_qrels(out_dir / "qrels.tsv")
    target_a, target_b, hard_neg = episode_ids[0], episode_ids[1], episode_ids[2]
    # Grounding seeds are preserved as grade 3 (override path).
    assert qrels[("para-0001", target_a)] == 3
    assert qrels[("error-0001", target_b)] == 3
    assert qrels[("tool-0001", episode_ids[3])] == 3
    # Confusable hard negative is forced to grade 0.
    assert qrels[("para-0001", hard_neg)] == 0
    # Every grade is within {0, 1, 2, 3}.
    assert all(grade in (0, 1, 2, 3) for grade in qrels.values())

    # Disagreements are logged, not silently merged.
    disagreements = _load_disagreements(out_dir / "judge_disagreements.jsonl")
    reasons = {item["reason"] for item in disagreements}
    assert "grounding_override" in reasons
    assert "hard_negative_forced_zero" in reasons
    assert result.disagreements == disagreements

    # judge_stats.json carries model, per-class agreement, cost, grade counts.
    stats = json.loads((out_dir / "judge_stats.json").read_text(encoding="utf-8"))
    assert stats["model"]
    assert stats["grade_counts"]["3"] >= 1
    assert stats["grade_counts"]["0"] >= 1
    assert stats["agreement_vs_grounding"]["overall"] is not None
    assert stats["cost_usd"] >= 0.0
    assert stats["candidate_coverage"] == 1.0

    # The judge never saw grounding labels: every call is just query + excerpt.
    assert calls


def test_budget_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None) -> None:
    """A cap forced to $0.01 aborts with a projected-cost message before any API batch."""
    handle = _build_index(tmp_path)
    episode_ids = _episode_ids(handle)
    queries = _make_queries(episode_ids)
    queries_path = _write_queries(tmp_path, queries)

    calls: list[tuple[str, str]] = []

    def judge_fn(query: str, excerpt: str) -> int:
        calls.append((query, excerpt))
        return 3

    with pytest.raises(BudgetError) as excinfo:
        judge_index(
            handle,
            queries_path,
            model="anthropic/claude-sonnet-4",
            cost_cap_usd=0.01,
            judge_fn=judge_fn,
        )
    assert "projected cost" in str(excinfo.value)
    assert calls == []  # aborted before any judge call
