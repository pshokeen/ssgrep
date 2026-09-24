"""Round-trip and concurrent-reader tests for the opencode SQLite emitter.

The round-trip test builds a real index through the REAL indexer with
``SSGREP_OPENCODE_DB`` pointed at the emitted database (no live opencode app
involved), using the same fake-embedder boundary as ``tests/pipeline/test_app.py``
and ``tests/eval/test_ranking.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from eval import harness
from eval.datasetgen.emitters.opencode import emit_parquet, emit_sessions
from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, LanceStore

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
    index: int,
    title: str,
    project: str,
    episodes: list[tuple[str, str]],
    *,
    files: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
) -> dict:
    return {
        "session_id": f"sess-{index:04d}",
        "title": title,
        "project": project,
        "episodes": episodes,
        "files_touched": list(files),
        "tool_names": list(tools),
    }


SAMPLES: list[dict] = [
    _session(
        0,
        "Retry backoff hang",
        "payments",
        [
            ("Where is the backoff policy?", "It lives in retry.py, capped at 60 seconds."),
            ("How do I test it?", "Run the retry harness with a three-attempt cap."),
        ],
        files=("retry.py",),
        tools=("read",),
    ),
    _session(
        1,
        "Async migration failures",
        "inventory",
        [("Why did the async migration fail?", "The batch writer raced the checkpoint flush.")],
    ),
    _session(
        2,
        "Cache invalidation",
        "catalog",
        [
            ("Why is the cache stale?", "The TTL was reset on every read."),
            ("How do I fix it?", "Bump the version key on writes."),
            ("How do I verify it?", "Run the cache probe."),
        ],
    ),
]


def test_emit_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    emit_sessions(SAMPLES, first)
    emit_sessions(SAMPLES, second)
    assert first.read_bytes() == second.read_bytes()


def test_integrity_check(tmp_path: Path) -> None:
    db = tmp_path / "opencode.db"
    emit_sessions(SAMPLES, db)
    with sqlite3.connect(db) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    assert result == ("ok",)


def test_concurrent_readers(tmp_path: Path) -> None:
    """Two simultaneous read-only snapshot connections see identical counts."""
    db = tmp_path / "opencode.db"
    emit_sessions(SAMPLES, db)
    uri = f"{db.absolute().as_uri()}?mode=ro"
    expected = _expected_counts(db)
    with (
        sqlite3.connect(uri, uri=True, timeout=1.0) as one,
        sqlite3.connect(uri, uri=True, timeout=1.0) as two,
    ):
        one.execute("PRAGMA query_only = ON")
        two.execute("PRAGMA query_only = ON")
        one.execute("BEGIN")
        two.execute("BEGIN")
        for table, expected_count in expected.items():
            first = one.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            second = two.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert first == second == expected_count
        one.execute("ROLLBACK")
        two.execute("ROLLBACK")


def _expected_counts(db: Path) -> dict[str, int]:
    with sqlite3.connect(db) as connection:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("session", "message", "part")
        }


def test_round_trip_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_models: None
) -> None:
    """Emit -> ingest through the real indexer via SSGREP_OPENCODE_DB."""
    db = tmp_path / "opencode.db"
    emit_sessions(SAMPLES, db)
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(db))

    index_dir = tmp_path / "index"
    with harness._private_data_dir(index_dir):
        indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        # Second pass: the first pass initializes before rows land, so the
        # vector index only builds once live rows exist (T3 learning).
        indexer.index(rebuild=False, quiet=True)
        store = LanceStore()
        episodes = store.rows(EPISODES_TABLE)
        chunks = store.rows(CHUNKS_TABLE)

    expected_episodes = sum(len(s["episodes"]) for s in SAMPLES)
    assert len({row["session_id"] for row in episodes}) == 3
    assert len(episodes) == expected_episodes
    assert chunks, "empty chunks table"
    assert all(row["runtime"] == "opencode" for row in chunks)

    session_episodes = [row for row in episodes if row["session_id"] == "opencode:sess-0000"]
    target = next(
        row for row in session_episodes if row["prompt_text"] == "Where is the backoff policy?"
    )
    assert target["response_text"] == "It lives in retry.py, capped at 60 seconds."
    assert target["title"] == "Retry backoff hang"
    assert target["files_touched"] == "retry.py"
    assert "Read" in target["tool_names"]


def test_emit_parquet_filters_runtime(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "runtime": ["opencode", "codex", "opencode"],
            "episodes": [
                [{"prompt": "alpha?", "response": "alpha!"}],
                [{"prompt": "beta?", "response": "beta!"}],
                [{"prompt": "gamma?", "response": "gamma!"}],
            ],
            "title": ["one", "two", "three"],
            "project": ["alpha-app", "beta-app", "gamma-app"],
        }
    )
    parquet = tmp_path / "sessions.parquet"
    pq.write_table(table, parquet)
    db = tmp_path / "opencode.db"
    assert emit_parquet(parquet, db) == 2
    with sqlite3.connect(db) as connection:
        sessions = connection.execute("SELECT id, title FROM session ORDER BY id").fetchall()
        assert [row[1] for row in sessions] == ["one", "three"]
