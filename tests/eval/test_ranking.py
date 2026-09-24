"""Tests for eval.ranking: prefetch/final/brute-force ranking instrumentation.

The fixture index is built through the REAL indexer on native-format
transcripts (one JSONL file per session via ``SSGREP_TRANSCRIPT_DIRS``), with
the embedding-model boundary swapped for a deterministic offline fake (the
same pattern as ``tests/pipeline/test_app.py``). The corpus is tiny, so the
engine's ANN is effectively exhaustive and its chunk scores agree with exact
MaxSim — which is exactly what the brute-force superset test verifies.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from eval import harness, ranking
from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.search import response as response_module
from ssgrep.store import CHUNKS_TABLE, LanceStore

pytestmark = pytest.mark.slow

EXACT_QUERY = "purple monkey dishwasher"

#: Filler vocabulary sharing no character trigram with EXACT_QUERY, so only
#: the seeded exact-match episode scores meaningfully above zero.
_FILLER = [
    "zebra",
    "quartz",
    "jigsaw",
    "kayak",
    "xylophone",
    "vortex",
    "nebula",
    "cactus",
    "falcon",
    "harbor",
    "mosaic",
    "tundra",
    "wombat",
    "yonder",
    "lizard",
    "octopus",
    "penguin",
    "rhino",
    "sphinx",
    "tiger",
    "umbrella",
    "viper",
    "walrus",
    "zephyr",
    "bamboo",
    "cobalt",
    "dune",
    "ember",
    "fjord",
    "gizmo",
    "hazel",
    "indigo",
    "jasmine",
    "kettle",
    "lagoon",
    "mango",
    "nectar",
    "onyx",
    "pebble",
    "quiver",
    "raven",
    "saffron",
    "tulip",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zinnia",
]

SESSION_COUNT = 6
EPISODES_PER_SESSION = 10
MATCH_SESSION = 0
MATCH_EPISODE = 0


def fake_vector(text: str) -> np.ndarray:
    """Deterministic ``(num_tokens, DIMENSION)`` float32 matrix for one text.

    Each token row is derived from a character trigram, so texts sharing
    trigrams (a query and the chunk that answers it) get identical rows and
    rank higher under native MaxSim. Rows are L2-normalized like the real
    ColBERT output.
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


def _write_session(root: Path, session_id: str, episodes: list[tuple[str, str]]) -> Path:
    path = root / f"{session_id}.jsonl"
    records: list[dict] = []
    for index, (prompt, response) in enumerate(episodes):
        stamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
        records.append(
            {
                "type": "user",
                "cwd": "/work/app",
                "message": {"content": prompt},
                "timestamp": stamp.isoformat(),
                "sessionId": session_id,
            }
        )
        records.append(
            {
                "type": "assistant",
                "message": {"content": response},
                "timestamp": (stamp + timedelta(seconds=1)).isoformat(),
                "sessionId": session_id,
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


@pytest.fixture
def fixture_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_models: None,
) -> Path:
    """A 60-episode index built by the real indexer from native transcripts."""
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    for session in range(SESSION_COUNT):
        episodes: list[tuple[str, str]] = []
        for episode in range(EPISODES_PER_SESSION):
            if session == MATCH_SESSION and episode == MATCH_EPISODE:
                response = "The purple monkey dishwasher protocol is the answer."
            else:
                offset = (session * EPISODES_PER_SESSION + episode) % len(_FILLER)
                response = " ".join((_FILLER * 2)[offset : offset + 6]) + "."
            prompt = f"how do I fix the {_FILLER[(session + episode) % len(_FILLER)]} error"
            episodes.append((prompt, response))
        _write_session(transcripts, f"session-{session}", episodes)

    monkeypatch.setenv("SSGREP_TRANSCRIPT_DIRS", f"native={transcripts}")
    index_dir = tmp_path / "index"
    with harness._private_data_dir(index_dir):
        indexer.index(rebuild=True, allow_shrink=True, quiet=True)
        # Second pass: the first pass initializes before rows land, so the
        # vector index only builds once live rows exist (T3 learning).
        indexer.index(rebuild=False, quiet=True)
    return index_dir


def _brute_force_top_episodes(index_dir: Path, query: str, *, top: int) -> list[str]:
    """Top episodes by exact MaxSim, rolled up with the production logic.

    Feeds the brute-force chunk scores back through ``_rows_to_episodes`` +
    ``_rank_episodes`` (as ``_distance = 1 - maxsim`` rows) so the episode
    ordering uses the exact same rollup and tie-breaks as the engine.
    """
    chunks, _elapsed = ranking.brute_force_ranking(index_dir, query)
    by_chunk = {chunk_id: score for chunk_id, _episode_id, score in chunks}
    with harness._private_data_dir(index_dir):
        rows = LanceStore().rows(CHUNKS_TABLE)
    synthetic = []
    for row in rows:
        synthetic_row = dict(row)
        synthetic_row["_distance"] = (
            search_module.DISTANCE_TO_MAXSIM_OFFSET - by_chunk[str(row["chunk_id"])]
        )
        synthetic.append(synthetic_row)
    rolled, episode_rows = search_module._rows_to_episodes(synthetic, num_query_tokens=32)
    scores = {episode_id: score for episode_id, (score, _hit) in rolled.items()}
    ordered = response_module._rank_episodes(scores, episode_rows)
    return ordered[:top]


def test_deterministic_orderings(fixture_index: Path) -> None:
    first, _elapsed = ranking.prefetch_episode_ranking(fixture_index, EXACT_QUERY)
    second, _elapsed = ranking.prefetch_episode_ranking(fixture_index, EXACT_QUERY)
    assert [episode_id for episode_id, _score in first] == [
        episode_id for episode_id, _score in second
    ]
    assert first == second

    final_first, _elapsed = ranking.final_ranking(fixture_index, EXACT_QUERY)
    final_second, _elapsed = ranking.final_ranking(fixture_index, EXACT_QUERY)
    assert final_first == final_second


def test_brute_force_superset(fixture_index: Path) -> None:
    engine, _elapsed = ranking.prefetch_episode_ranking(fixture_index, EXACT_QUERY, pool_depth=800)
    engine_top5 = [episode_id for episode_id, _score in engine[:5]]
    brute_top5 = _brute_force_top_episodes(fixture_index, EXACT_QUERY, top=5)
    assert set(engine_top5) <= set(brute_top5)
    # The exact-match episode ranks first on both arms.
    assert engine_top5[0] == brute_top5[0]


def test_final_ranking_respects_max_result_count_cap(fixture_index: Path) -> None:
    result, _elapsed = ranking.final_ranking(fixture_index, EXACT_QUERY, limit=100)
    assert len(result) <= search_module.MAX_RESULT_COUNT
    # The corpus holds 60 episodes; the cap is what binds, not the corpus.
    assert len(result) > 10
