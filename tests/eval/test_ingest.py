"""Tests for ``eval.datasetgen.ingest`` (Task 13).

The five-runtime test builds a small dataset by running all five emitters
(T7-T11) on a hand-built parquet, ingests it through the REAL indexer via
``build_benchmark_index``, and validates the result against a manifest derived
from the source records. The missing-dir test proves the helper refuses loudly
when any emitter output is absent.
"""

from __future__ import annotations

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
from eval.datasetgen.ingest import (
    RUNTIME_LABELS,
    build_benchmark_index,
    derive_manifest,
)
from ssgrep import search as search_module
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import CHUNKS_TABLE

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
    # app.py binds ensure_model_downloaded via `from ... import`, so the module
    # patch above does not reach the app's local reference; patch it directly.
    monkeypatch.setattr(app_mod, "ensure_model_downloaded", lambda *a, **kw: None)
    monkeypatch.setattr(app_mod, "ColBERTEmbedder", FakePyLateEmbedder)
    monkeypatch.setattr(search_module, "_query_matrix", lambda query: fake_vector(str(query)))


def _session(
    runtime: str,
    index: int,
    title: str,
    project: str,
    episodes: list[tuple[str, str]],
    *,
    files: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
) -> dict:
    return {
        "session_id": f"{runtime}-{index:02d}",
        "runtime": runtime,
        "scenario_class": "error-string",
        "difficulty": "easy",
        "episode_length_bucket": "short",
        "title": title,
        "project": project,
        "files_touched": list(files),
        "tool_names": list(tools),
        "episodes": [{"prompt": prompt, "response": response} for prompt, response in episodes],
    }


ROWS: list[dict] = [
    _session(
        "claude",
        0,
        "Retry backoff hang",
        "/fictional/checkout/payments",
        [
            ("Where is the backoff policy?", "It lives in retry.py, capped at 60 seconds."),
            ("How do I test it?", "Run the retry harness with a three-attempt cap."),
        ],
        files=("retry.py",),
        tools=("Read",),
    ),
    _session(
        "claude",
        1,
        "Async migration failures",
        "/fictional/checkout/inventory",
        [("Why did the async migration fail?", "The batch writer raced the checkpoint flush.")],
    ),
    _session(
        "codex",
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
        1,
        "Rate limiter exhaustion",
        "/fictional/checkout/gateway",
        [("Why does the route 429 under load?", "The shared limiter is exhausted.")],
    ),
    _session(
        "pi",
        0,
        "Invoice PDFs arrive empty",
        "/fictional/checkout/billing",
        [("Generated invoice PDFs are blank.", "The template engine resolves the wrong key.")],
    ),
    _session(
        "pi",
        1,
        "Template registration",
        "/fictional/checkout/billing",
        [("Where is the invoice template registered?", "Under billing/templates/invoice.jinja2")],
    ),
    _session(
        "prime-agent",
        0,
        "Pool ceiling tuning",
        "/fictional/checkout/payments",
        [("Raise the pool ceiling?", "Bump MAX_POOL in the config.")],
    ),
    _session(
        "prime-agent",
        1,
        "Checkpoint flush race",
        "/fictional/checkout/inventory",
        [("How do we serialize the flush?", "Take the checkpoint lock before writing.")],
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
    _session(
        "opencode",
        1,
        "Version key on writes",
        "/fictional/checkout/catalog",
        [("How do we invalidate the cache?", "Bump the version key on every write.")],
    ),
]

EXPECTED_SESSIONS = 10
EXPECTED_EPISODES = sum(len(row["episodes"]) for row in ROWS)


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


def test_five_runtime_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None
) -> None:
    """Full emitter-output dataset ingests with counts matching the source records."""
    dataset, parquet_path = _build_dataset(tmp_path)
    manifest = derive_manifest(parquet_path)

    index_dir = tmp_path / "index"
    handle = build_benchmark_index(dataset, index_dir, manifest=manifest)

    assert handle.stats.malformed_records == 0
    assert handle.stats.session_count == EXPECTED_SESSIONS
    assert handle.stats.episode_count == EXPECTED_EPISODES
    assert handle.stats.chunk_count >= EXPECTED_EPISODES
    assert handle.db_path == index_dir / "lancedb"

    store = handle.store()
    # Every runtime label is present in the chunks table.
    for label in RUNTIME_LABELS:
        assert store.count(CHUNKS_TABLE, where=f"runtime = '{label}'") > 0, label
    # Per-runtime session/episode counts match the manifest exactly.
    for table, expected in manifest.items():
        for label, want in expected.items():
            assert store.count(table, where=f"runtime = '{label}'") == want, (table, label)


def test_missing_dataset_dir_fails_loudly(tmp_path: Path) -> None:
    """A missing emitter dir is refused with an error naming it."""
    dataset, _ = _build_dataset(tmp_path)
    (dataset / "transcripts" / "codex").rename(dataset / "transcripts" / "codex-gone")

    with pytest.raises(FileNotFoundError) as excinfo:
        build_benchmark_index(dataset, tmp_path / "index")
    message = str(excinfo.value)
    assert "codex" in message
    assert "missing emitter output" in message


def test_derive_manifest_maps_claude_to_native(tmp_path: Path) -> None:
    """``derive_manifest`` uses post-ingestion labels (claude -> native)."""
    parquet_path = tmp_path / "sessions.parquet"
    names = {key for row in ROWS for key in row}
    columns = {name: [row.get(name) for row in ROWS] for name in names}
    pq.write_table(pa.Table.from_pydict(columns), parquet_path)

    manifest = derive_manifest(parquet_path)
    assert manifest["sessions"]["native"] == 2
    assert manifest["sessions"]["codex"] == 2
    assert manifest["sessions"]["pi"] == 2
    assert manifest["sessions"]["prime-agent"] == 2
    assert manifest["sessions"]["opencode"] == 2
    assert "claude" not in manifest["sessions"]
    assert manifest["episodes"]["native"] == 3
    assert sum(manifest["episodes"].values()) == EXPECTED_EPISODES
