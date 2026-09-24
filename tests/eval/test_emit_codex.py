"""Round-trip tests for eval.datasetgen.emitters.codex.

The emitter output is consumed through the REAL codex adapter and the REAL
indexer: ``SSGREP_CODEX_SESSIONS_DIR`` points at the emitted directory and a
private index is built under ``SSGREP_DATA_DIR`` with the embedding-model
boundary swapped for a deterministic offline fake (the same pattern as
``tests/pipeline/test_app.py`` / ``tests/eval/test_ranking.py``). Session,
episode, and chunk counts must match the parquet source records; the runtime
column must be ``codex`` on every chunk; and the excluded record types the
adapter drops (reasoning, tool-output echoes, developer messages) must never
surface in indexed chunk text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval import harness
from eval.datasetgen.emitters import codex as codex_emitter
from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, LanceStore

pytestmark = pytest.mark.slow


class _Session(TypedDict, total=False):
    """One synthetic parquet row; columns may be absent (``total=False``)."""

    runtime: str
    scenario_class: str
    title: str
    project: str
    difficulty: str
    episode_length_bucket: str
    files_touched: list[str]
    tool_names: list[str]
    episodes: list[dict[str, str]]


SESSIONS: list[_Session] = [
    {
        "runtime": "codex",
        "scenario_class": "error-string",
        "title": "Retry loop hangs after attempt three",
        "project": "/fictional/checkout/payments",
        "difficulty": "easy",
        "episode_length_bucket": "short",
        "files_touched": ["/fictional/checkout/payments/retry.py"],
        "tool_names": ["read", "shell"],
        "episodes": [
            {
                "prompt": "The retry loop hangs after attempt three. Where is the backoff policy?",
                "response": "Backoff lives in retry.py, capped at 60 seconds.",
            },
            {
                "prompt": "Why does the payment route return 429 under load?",
                "response": "The shared rate limiter is exhausted; raise the pool ceiling.",
            },
        ],
    },
    {
        "runtime": "codex",
        "scenario_class": "paraphrase",
        "title": "Invoice PDFs arrive empty",
        "project": "/fictional/checkout/billing",
        "difficulty": "hard",
        "episode_length_bucket": "long",
        "files_touched": ["/fictional/checkout/billing/invoice.py"],
        "tool_names": ["read", "execute"],
        "episodes": [
            {
                "prompt": (
                    "Generated invoice PDFs come back blank. Which template renderer is wrong?"
                ),
                "response": "The template engine resolves the wrong data key",
            },
            {
                "prompt": "Where is the invoice template registered?",
                "response": "See billing/templates/invoice.jinja2 as configured",
            },
            {
                "prompt": "How should we validate generated PDF page counts in CI?",
                "response": "Compare against a golden page-count fixture per template.",
            },
        ],
    },
    {
        # Not a codex row: the emitter must skip it entirely.
        "runtime": "pi",
        "scenario_class": "tool-failure-recovery",
        "title": "Untouched row",
        "project": "/fictional/checkout/other",
        "difficulty": "easy",
        "episode_length_bucket": "short",
        "episodes": [
            {
                "prompt": "This is a pi row that must not be emitted",
                "response": "It belongs to another emitter.",
            }
        ],
    },
]


def fake_vector(text: str) -> np.ndarray:
    """Deterministic ``(num_tokens, DIMENSION)`` float32 matrix for one text.

    Each token row is derived from a character trigram; rows are L2-normalized
    like the real ColBERT output.
    """
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


def _write_sessions_parquet(path: Path) -> None:
    names = {key for row in SESSIONS for key in row}
    columns = {name: [row.get(name) for row in SESSIONS] for name in names}
    pq.write_table(pa.Table.from_pydict(columns), path)


def _emit(parquet_path: Path, out_dir: Path, *, include_excluded: bool = False) -> list[Path]:
    return codex_emitter.emit_sessions(parquet_path, out_dir, include_excluded=include_excluded)


@pytest.fixture
def emitted_sessions(tmp_path: Path) -> Path:
    """Emit the codex rows (with exclusion canaries) and return the directory."""
    parquet_path = tmp_path / "sessions.parquet"
    _write_sessions_parquet(parquet_path)
    out_dir = tmp_path / "codex-sessions"
    written = _emit(parquet_path, out_dir, include_excluded=True)
    assert len(written) == 2  # only the two codex rows
    return out_dir


def test_emitted_files_are_well_formed(emitted_sessions: Path) -> None:
    """Every file starts with session_meta and every line ends with a newline."""
    files = sorted(emitted_sessions.iterdir())
    assert [path.name for path in files] == ["codex-0000.jsonl", "codex-0001.jsonl"]
    for path in files:
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        lines = [line for line in raw.decode("utf-8").split("\n") if line]
        first = json.loads(lines[0])
        assert first["type"] == "session_meta"
        assert first["payload"]["id"] == path.stem
        assert all(
            json.loads(line)["type"] in {"session_meta", "turn_context", "response_item"}
            for line in lines
        )


def test_emitted_files_are_byte_stable(tmp_path: Path) -> None:
    """Two emission runs over the same parquet produce byte-identical trees."""
    parquet_path = tmp_path / "sessions.parquet"
    _write_sessions_parquet(parquet_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = _emit(parquet_path, first_dir)
    second = _emit(parquet_path, second_dir)
    assert [path.name for path in first] == [path.name for path in second]
    for one, two in zip(first, second, strict=True):
        assert one.read_bytes() == two.read_bytes()


def test_round_trip(emitted_sessions: Path, tmp_path: Path, monkeypatch, fake_models) -> None:
    """The real indexer must ingest emitted rollouts to a codex-only index."""
    monkeypatch.setenv("SSGREP_CODEX_SESSIONS_DIR", str(emitted_sessions))
    index_dir = tmp_path / "index"
    with harness._private_data_dir(index_dir):
        stats = indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        # Second pass builds the vector index once live rows exist (T3/T2).
        stats = indexer.index(rebuild=False, quiet=True)

    expected_episodes = sum(len(row["episodes"]) for row in SESSIONS if row["runtime"] == "codex")
    assert stats.session_count == 2
    assert stats.episode_count == expected_episodes
    assert stats.chunk_count >= expected_episodes
    assert stats.skipped_records == 0
    assert stats.malformed_records == 0

    with harness._private_data_dir(index_dir):
        chunks = LanceStore().rows(CHUNKS_TABLE, columns=["runtime", "session_id", "text"])
        episodes = LanceStore().rows(
            EPISODES_TABLE, columns=["episode_id", "prompt_text", "response_text", "runtime"]
        )

    assert stats.chunk_count == len(chunks)
    assert set(row["runtime"] for row in chunks) == {"codex"}
    assert set(row["runtime"] for row in episodes) == {"codex"}

    # Episode ids follow the <session_id>:ep:<n> scheme under the codex prefix.
    session_ids = {row["session_id"] for row in chunks}
    assert session_ids == {"codex:codex-0000", "codex:codex-0001"}

    # Raw ids are assigned in content-sorted row order (the emitter's _row_key:
    # project, title, episodes), mirroring emit_sessions.
    codex_rows = sorted(
        (row for row in SESSIONS if row["runtime"] == "codex"),
        key=lambda row: (
            row["project"],
            row["title"],
            json.dumps(row["episodes"], sort_keys=True),
        ),
    )
    expected_episode_ids = {
        f"codex:codex-{index:04d}:ep:{ep}"
        for index, row in enumerate(codex_rows)
        for ep in range(len(row["episodes"]))
    }
    assert {row["episode_id"] for row in episodes} == expected_episode_ids

    # Excluded record shapes must never surface in indexed chunk text.
    chunk_text = "\n".join(str(row["text"]) for row in chunks)
    for marker in ("CANARY_REASONING_", "CANARY_TOOL_OUTPUT_", "CANARY_DEVELOPER_"):
        assert marker not in chunk_text

    # Spot-check first-episode text fidelity through the indexed episode rows.
    by_id = {row["episode_id"]: row for row in episodes}
    first_prompt = codex_rows[0]["episodes"][0]["prompt"]
    first_response = codex_rows[0]["episodes"][0]["response"]
    assert by_id["codex:codex-0000:ep:0"]["prompt_text"] == first_prompt
    assert by_id["codex:codex-0000:ep:0"]["response_text"] == first_response
