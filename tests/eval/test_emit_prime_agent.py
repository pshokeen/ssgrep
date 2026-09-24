"""Round-trip test for the prime-agent emitter through the real indexer.

Emits synthetic prime-agent sessions plus ``session-artifacts/`` children,
ingests them via ``SSGREP_PRIME_AGENT_SESSIONS_DIR`` with the REAL indexer
(only the embedding-model boundary is faked, the ``tests/pipeline/test_app.py``
pattern), and asserts the LanceDB census matches the source records:
session/episode counts, ``runtime == 'prime-agent'`` on every chunk, and
artifact children indexed as subagents (``is_main=False`` by location,
``pi.py:384-390``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval import harness
from eval.datasetgen.emitters import prime_agent
from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore

pytestmark = pytest.mark.slow

MAIN_SESSIONS = [
    {
        "session_id": "prime-0001",
        "project": "/fictional/checkout/payments",
        "title": "Retry loop never sleeps",
        "episodes": [
            {
                "prompt": "payments prompt 0: where is the backoff policy?",
                "response": "payments response 0: bounded backoff caps at 60s in retry.py.",
            },
            {
                "prompt": "payments prompt 1: why does attempt three hang?",
                "response": "payments response 1: the cap is hit after the third attempt.",
            },
        ],
        "model": "claude-sonnet-4-5",
        "git_branch": "fix/retry",
        "tool_names": ["Read", "Write"],
        "files_touched": ["/fictional/checkout/payments/retry.py"],
    },
    {
        "session_id": "prime-0002",
        "project": "/fictional/checkout/payments",
        "title": "Backoff tuning",
        "episodes": [
            {
                "prompt": "backoff prompt 0: tune the sleep schedule.",
                "response": "backoff response 0: exponential with jitter.",
            },
            {
                "prompt": "backoff prompt 1: cap the total wait.",
                "response": "backoff response 1: sixty seconds is the ceiling.",
            },
            {
                "prompt": "backoff prompt 2: log the retries.",
                "response": "backoff response 2: emit a warning per attempt.",
            },
        ],
        "model": "claude-sonnet-4-5",
    },
]

ARTIFACT_SESSIONS = [
    {
        "session_id": "art-0001",
        "project": "/fictional/checkout/payments",
        "title": "Retry findings summary",
        "episodes": [
            {
                "prompt": "summary prompt 0: summarize the retry findings.",
                "response": "summary response 0: backoff caps at 60s in retry.py.",
            },
            {
                "prompt": "summary prompt 1: list the touched files.",
                "response": "summary response 1: retry.py and tests.",
            },
        ],
        "parent_session_id": "prime-0001",
        "rlmDepth": 1,
        "is_artifact": True,
    },
    {
        "session_id": "art-0002",
        "project": "/fictional/checkout/payments",
        "title": "Backoff notes",
        "episodes": [
            {
                "prompt": "notes prompt 0: capture the tuning notes.",
                "response": "notes response 0: jitter plus a hard ceiling.",
            },
        ],
        "parent_session_id": "prime-0002",
        "rlmDepth": 1,
        "is_artifact": True,
    },
]

EXPECTED_MAIN = 2
EXPECTED_ARTIFACTS = 2
EXPECTED_SESSIONS = EXPECTED_MAIN + EXPECTED_ARTIFACTS
EXPECTED_EPISODES = 2 + 3 + 2 + 1
EXPECTED_SUBAGENT_EPISODES = 2 + 1


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


def _all_rows() -> pd.DataFrame:
    return pd.DataFrame(MAIN_SESSIONS + ARTIFACT_SESSIONS)


def _ingest(out_dir: Path, index_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the prime-agent override at the emitted sessions root and index."""
    monkeypatch.setenv("SSGREP_PRIME_AGENT_SESSIONS_DIR", str(out_dir / prime_agent.MAIN_DIR))
    with harness._private_data_dir(index_dir):
        indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        # Second pass: the first pass initializes before rows land, so the
        # vector index only builds once live rows exist (T3 learning).
        indexer.index(rebuild=False, quiet=True)


def test_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_models: None,
) -> None:
    out_dir = tmp_path / "emitted"
    report = prime_agent.emit(_all_rows(), out_dir)
    assert report == {"main_sessions": EXPECTED_MAIN, "artifact_sessions": EXPECTED_ARTIFACTS}

    # Layout: sessions/ plus its sibling session-artifacts/ (pi.py:381-382).
    assert (out_dir / prime_agent.MAIN_DIR / "prime-0001.jsonl").is_file()
    assert (out_dir / prime_agent.ARTIFACTS_DIR / "art-0001.jsonl").is_file()
    assert not (out_dir / prime_agent.ARTIFACTS_DIR / "prime-0001.jsonl").exists()

    index_dir = tmp_path / "index"
    _ingest(out_dir, index_dir, monkeypatch)

    with harness._private_data_dir(index_dir):
        store = LanceStore()
        # Session/episode counts match the source records.
        assert store.count(SESSIONS_TABLE) == EXPECTED_SESSIONS
        assert store.count(SESSIONS_TABLE, where="runtime = 'prime-agent'") == EXPECTED_SESSIONS
        assert store.count(EPISODES_TABLE) == EXPECTED_EPISODES
        # Every chunk carries runtime 'prime-agent' (short texts chunk 1:1).
        assert store.count(CHUNKS_TABLE) == EXPECTED_EPISODES * 2
        assert store.count(CHUNKS_TABLE, where="runtime = 'prime-agent'") == EXPECTED_EPISODES * 2
        # Artifact children are ingested as subagents (is_main=False by location).
        assert store.count(EPISODES_TABLE, where="is_subagent = true") == EXPECTED_SUBAGENT_EPISODES
        assert store.count(EPISODES_TABLE, where="is_subagent = false") == EXPECTED_MAIN + 3

        episodes = store.rows(
            EPISODES_TABLE,
            columns=[
                "episode_id",
                "session_id",
                "prompt_text",
                "response_text",
                "files_touched",
                "tool_names",
                "parent_session_id",
                "is_subagent",
            ],
            limit=100,
        )
    by_id = {row["episode_id"]: row for row in episodes}

    # Spot-check first episode text equality.
    first = by_id["prime-agent:prime-0001:ep:0"]
    assert first["prompt_text"] == "payments prompt 0: where is the backoff policy?"
    assert first["response_text"] == "payments response 0: bounded backoff caps at 60s in retry.py."
    # toolCall blocks survive as files_touched/tool_names metadata.
    assert "retry.py" in first["files_touched"]
    assert "Read" in first["tool_names"]

    # Artifact children carry the namespaced parent linkage.
    subagent = [row for row in episodes if row["is_subagent"]]
    parents = {row["parent_session_id"] for row in subagent}
    assert parents == {"prime-agent:prime-0001", "prime-agent:prime-0002"}
    assert {row["session_id"] for row in subagent} == {
        "prime-agent:art-0001",
        "prime-agent:art-0002",
    }


def test_deterministic_output(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    prime_agent.emit(_all_rows(), out_a)
    prime_agent.emit(_all_rows(), out_b)
    for rel in sorted((out_a / prime_agent.MAIN_DIR).rglob("*.jsonl")):
        assert rel.read_bytes() == (out_b / prime_agent.MAIN_DIR / rel.name).read_bytes()
    for rel in sorted((out_a / prime_agent.ARTIFACTS_DIR).rglob("*.jsonl")):
        assert rel.read_bytes() == (out_b / prime_agent.ARTIFACTS_DIR / rel.name).read_bytes()


def test_filters_non_prime_agent_rows(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "session_id": "other-0001",
                "runtime": "pi",
                "project": "/fictional/other",
                "title": "Not prime agent",
                "episodes": [{"prompt": "p", "response": "r"}],
            }
        ]
    )
    report = prime_agent.emit(rows, tmp_path)
    assert report == {"main_sessions": 0, "artifact_sessions": 0}
    assert not (tmp_path / prime_agent.MAIN_DIR / "other-0001.jsonl").exists()


def test_emitted_files_are_newline_terminated(tmp_path: Path) -> None:
    prime_agent.emit(_all_rows(), tmp_path)
    for path in sorted((tmp_path / prime_agent.MAIN_DIR).rglob("*.jsonl")):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        for line in raw.splitlines():
            json.loads(line)
