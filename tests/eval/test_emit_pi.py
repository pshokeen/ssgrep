"""Round-trip tests for the Pi session emitter through the REAL indexer.

The emitter converts ``sessions.parquet`` rows (``runtime == 'pi'``) into Pi
JSONL under ``SSGREP_PI_SESSIONS_DIR``; the fixture index is built by the real
indexer from that directory (the same fake-embedder boundary as
``tests/pipeline/test_app.py``), then asserted through LanceStore: session
count, runtime label on every chunk, episode counts matching source records,
and noise markers absent from chunk text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval import harness
from eval.datasetgen.emitters import pi as pi_emitter
from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore

pytestmark = pytest.mark.slow

SESSION_COUNT = 3
EPISODES_PER_SESSION = 2


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


def _write_parquet(path: Path) -> list[dict]:
    """Write a small ``sessions.parquet`` with pi rows plus noise columns."""
    rows: list[dict] = []
    for session in range(SESSION_COUNT):
        episodes = [
            {
                "prompt": f"pi problem {session} round {episode}: how to retry?",
                "response": f"pi answer {session} round {episode}: backoff then retry.",
            }
            for episode in range(EPISODES_PER_SESSION)
        ]
        rows.append(
            {
                "runtime": "pi",
                "scenario_class": "error-string",
                "episodes": episodes,
                "title": f"pi session {session}",
                "project": "/fictional/payments",
                "files_touched": [f"/fictional/payments/retry{session}.py"],
                "tool_names": ["Read"],
                "difficulty": "easy",
                "episode_length_bucket": "short",
                # Never-indexed noise markers (thinking / toolResult stand-ins).
                "noise_thinking": f"do-not-index-thinking-{session}",
                "noise_toolresult": f"do-not-index-toolresult-{session}",
            }
        )
    rows.append({"runtime": "codex", "title": "ignored"})
    table = pa.table(
        {
            "runtime": pa.array([row["runtime"] for row in rows], pa.string()),
            "scenario_class": pa.array(
                [row.get("scenario_class", "") for row in rows], pa.string()
            ),
            "episodes": pa.array(
                [row.get("episodes", []) for row in rows],
                pa.list_(
                    pa.struct([pa.field("prompt", pa.string()), pa.field("response", pa.string())])
                ),
            ),
            "title": pa.array([row["title"] for row in rows], pa.string()),
            "project": pa.array([row.get("project", "/fictional") for row in rows], pa.string()),
            "files_touched": pa.array(
                [row.get("files_touched") for row in rows], pa.list_(pa.string())
            ),
            "tool_names": pa.array([row.get("tool_names") for row in rows], pa.list_(pa.string())),
            "difficulty": pa.array([row.get("difficulty", "") for row in rows], pa.string()),
            "episode_length_bucket": pa.array(
                [row.get("episode_length_bucket", "") for row in rows], pa.string()
            ),
            "noise_thinking": pa.array(
                [row.get("noise_thinking", "") for row in rows], pa.string()
            ),
            "noise_toolresult": pa.array(
                [row.get("noise_toolresult", "") for row in rows], pa.string()
            ),
        }
    )
    pq.write_table(table, path)
    return rows


def test_emission_is_deterministic(tmp_path: Path) -> None:
    parquet = tmp_path / "sessions.parquet"
    _write_parquet(parquet)
    first = pi_emitter.emit_sessions(parquet, tmp_path / "out-a")
    second = pi_emitter.emit_sessions(parquet, tmp_path / "out-b")
    assert len(first) == SESSION_COUNT
    assert [path.name for path in first] == [path.name for path in second]
    for path_a, path_b in zip(first, second, strict=True):
        assert path_a.read_text() == path_b.read_text()


def test_round_trip(fake_models: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parquet = tmp_path / "sessions.parquet"
    rows = _write_parquet(parquet)
    out_dir = tmp_path / "pi-sessions"
    written = pi_emitter.emit_sessions(parquet, out_dir)
    assert len(written) == SESSION_COUNT
    for path in written:
        text = path.read_text()
        assert text.endswith("\n")
        assert json.loads(text.splitlines()[0])["type"] == "session"

    monkeypatch.setenv("SSGREP_PI_SESSIONS_DIR", str(out_dir))
    index_dir = tmp_path / "index"
    with harness._private_data_dir(index_dir):
        stats = indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        # Second pass: the first pass initializes before rows land, so the
        # vector index only builds once live rows exist.
        indexer.index(rebuild=False, quiet=True)
        repository = LanceStore()
        sessions = repository.rows(SESSIONS_TABLE, columns=["session_id", "runtime"])
        episodes = repository.rows(EPISODES_TABLE, columns=["episode_id", "runtime"])
        chunks = repository.rows(CHUNKS_TABLE, columns=["chunk_id", "text", "runtime"])

    assert stats.session_count == SESSION_COUNT
    assert len(sessions) == SESSION_COUNT
    assert {row["runtime"] for row in sessions} == {"pi"}

    expected_episodes = sum(len(row["episodes"]) for row in rows if row["runtime"] == "pi")
    assert stats.episode_count == expected_episodes
    assert len(episodes) == expected_episodes
    assert {row["runtime"] for row in episodes} == {"pi"}

    assert stats.chunk_count >= expected_episodes
    assert chunks
    assert {row["runtime"] for row in chunks} == {"pi"}

    all_text = "\n".join(str(row["text"]) for row in chunks)
    for session in range(SESSION_COUNT):
        assert f"do-not-index-thinking-{session}" not in all_text
        assert f"do-not-index-toolresult-{session}" not in all_text
