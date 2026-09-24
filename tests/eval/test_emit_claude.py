"""Round-trip tests for eval.datasetgen.emitters.claude.

The emitter output is consumed through the REAL native adapter and the REAL
indexer: ``SSGREP_TRANSCRIPT_DIRS`` points at the emitted directory (tagged
``native=<dir>``) and a private index is built under ``SSGREP_DATA_DIR`` with
the embedding-model boundary swapped for a deterministic offline fake (the same
pattern as ``tests/pipeline/test_app.py`` / ``tests/eval/test_ranking.py``).
Session, episode, and chunk counts must match the parquet source records.

Runtime-label note (documented deviation, T4 section 9): ingesting external
roots stamps ``runtime == 'native'``, NOT ``'claude'`` — only a synthetic
``CLAUDE_CONFIG_DIR/projects`` tree yields the ``claude`` label. This test
asserts ``native`` because that is the real ingestion path the T13 benchmark
harness uses; the ``claude`` census bucket is satisfied by the file format.
Session ids in the index take the external-root ``<stem>~<8-hex>`` shape
(discovery_roots.py:113-117), so episode ids are matched through the sessions'
source paths rather than raw filenames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval import harness
from eval.datasetgen.emitters import claude as claude_emitter
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
    version: str
    git_branch: str
    episodes: list[dict[str, str]]


SESSIONS: list[_Session] = [
    {
        "runtime": "claude",
        "scenario_class": "error-string",
        "title": "Retry loop never sleeps past attempt three",
        "project": "/fictional/checkout/payments",
        "difficulty": "easy",
        "episode_length_bucket": "short",
        "files_touched": ["/fictional/checkout/payments/retry.py"],
        "tool_names": ["Read", "Edit"],
        "version": "2.1.0",
        "git_branch": "fix/retry-backoff",
        "episodes": [
            {
                "prompt": "The retry loop hangs after attempt three. Where is the backoff policy?",
                "response": "Backoff is bounded to 60 seconds in retry.py.",
            },
            {
                "prompt": "Why does the payment route 429 under load?",
                "response": "The shared rate limiter is exhausted; raise the pool ceiling.",
            },
        ],
    },
    {
        "runtime": "claude",
        "scenario_class": "paraphrase",
        "title": "Invoice PDFs arrive empty",
        "project": "/fictional/checkout/billing",
        "difficulty": "hard",
        "episode_length_bucket": "long",
        "files_touched": ["/fictional/checkout/billing/invoice.py"],
        "tool_names": ["Read"],
        "episodes": [
            {
                "prompt": "Generated invoice PDFs are blank. Which renderer is wrong?",
                "response": "The template engine resolves the wrong data key",
            },
            {
                "prompt": "Where is the invoice template registered?",
                "response": "Under billing/templates/invoice.jinja2",
            },
            {
                "prompt": "How should we verify PDF page counts in CI?",
                "response": "Compare against a golden page-count fixture per template.",
            },
        ],
    },
    {
        # Not a claude row: the emitter must skip it entirely.
        "runtime": "codex",
        "scenario_class": "tool-failure-recovery",
        "title": "Untouched row",
        "project": "/fictional/checkout/other",
        "difficulty": "easy",
        "episode_length_bucket": "short",
        "episodes": [
            {
                "prompt": "This is a codex row that must not be emitted",
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


def _emit(parquet_path: Path, out_dir: Path) -> list[Path]:
    return claude_emitter.emit_sessions(parquet_path, out_dir)


@pytest.fixture
def emitted_sessions(tmp_path: Path) -> Path:
    """Emit the claude rows and return the emitted directory."""
    parquet_path = tmp_path / "sessions.parquet"
    _write_sessions_parquet(parquet_path)
    out_dir = tmp_path / "claude-sessions"
    written = _emit(parquet_path, out_dir)
    assert len(written) == 2  # only the two claude rows
    return out_dir


def test_emitted_files_are_well_formed(emitted_sessions: Path) -> None:
    """Every file leads with custom-title, pairs stay aligned, lines newline-terminated."""
    files = sorted(emitted_sessions.iterdir())
    assert [path.name for path in files] == ["claude-0000.jsonl", "claude-0001.jsonl"]
    for path in files:
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        records = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]
        assert records[0]["type"] == "custom-title"
        assert records[0]["custom-title"]
        turns = [record for record in records[1:] if record["type"] in {"user", "assistant"}]
        assert [record["type"] for record in turns] == [
            record_type for pair in range(len(turns) // 2) for record_type in ("user", "assistant")
        ]
        for record in turns:
            assert record["cwd"] and record["sessionId"] and record["timestamp"]
            assert record["uuid"] and record["version"]


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
    """The real indexer must ingest emitted transcripts to a native-runtime index."""
    monkeypatch.setenv("SSGREP_TRANSCRIPT_DIRS", f"native={emitted_sessions}")
    index_dir = tmp_path / "index"
    with harness._private_data_dir(index_dir):
        stats = indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        # Second pass builds the vector index once live rows exist (T3/T2).
        stats = indexer.index(rebuild=False, quiet=True)

    claude_rows = [row for row in SESSIONS if row["runtime"] == "claude"]
    expected_episodes = sum(len(row["episodes"]) for row in claude_rows)
    assert stats.session_count == 2
    assert stats.episode_count == expected_episodes
    assert stats.chunk_count >= expected_episodes
    assert stats.skipped_records == 0
    assert stats.malformed_records == 0

    with harness._private_data_dir(index_dir):
        chunks = LanceStore().rows(CHUNKS_TABLE, columns=["runtime", "session_id", "text"])
        episodes = LanceStore().rows(
            EPISODES_TABLE,
            columns=["episode_id", "session_id", "prompt_text", "response_text", "runtime"],
        )

    # External roots stamp runtime='native' (T4 §9 documented deviation).
    assert stats.chunk_count == len(chunks)
    assert set(row["runtime"] for row in chunks) == {"native"}
    assert set(row["runtime"] for row in episodes) == {"native"}

    # Two sessions, both indexed with the external-root session-id shape
    # <stem>~<8-hex-sha1-of-absolute-path>.
    session_ids = {row["session_id"] for row in episodes}
    assert len(session_ids) == 2
    emitted_paths = sorted(emitted_sessions.iterdir())
    for path in emitted_paths:
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        assert f"{path.stem}~{digest}" in session_ids

    # Episode ids follow the <session_id>:ep:<n> scheme.
    episode_ids = {row["episode_id"] for row in episodes}
    assert all(episode_id.count(":ep:") == 1 for episode_id in episode_ids)

    # Spot-check first-episode text fidelity through the indexed episode rows.
    # Raw ids are assigned in content-sorted row order (the emitter's _row_key:
    # project, title, episodes), so the expected session matches source rows
    # sorted the same way.
    expected_rows = sorted(
        (row for row in SESSIONS if row["runtime"] == "claude"),
        key=lambda row: (
            row["project"],
            row["title"],
            json.dumps(row["episodes"], sort_keys=True),
        ),
    )
    first_session = expected_rows[0]
    first_path = emitted_paths[0]
    digest = hashlib.sha1(str(first_path.resolve()).encode("utf-8")).hexdigest()[:8]
    first_session_id = f"{first_path.stem}~{digest}"
    # Each session contributes one row per episode; pick the row whose
    # episode_id ends with the session-level 0 index.
    ep0 = next(
        episode
        for episode in episodes
        if episode["session_id"] == first_session_id and episode["episode_id"].endswith(":ep:0")
    )
    assert ep0["prompt_text"]
    assert ep0["response_text"]
    expected_prompt = first_session["episodes"][0]["prompt"]
    expected_response = first_session["episodes"][0]["response"]
    assert ep0["prompt_text"] == expected_prompt
    assert ep0["response_text"] == expected_response
