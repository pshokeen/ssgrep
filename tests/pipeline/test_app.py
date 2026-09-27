"""Slow integration tests: full CocoIndex round trips over isolated corpora.

The embedding model is faked so the whole index -> search round trip runs
offline in CI: the CocoIndex provider is swapped for a deterministic
``FakePyLateEmbedder`` (producing ``(num_tokens, 128)`` multivectors) and the
LanceDB query-time ``ssgrep`` embedding function for a registry fake. The
real-model path is covered separately by the ``network``-marked smoke tests.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest

from ssgrep.indexing import embed as embed_mod
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import app as app_mod
from ssgrep.services import api
from ssgrep.store import (
    CHUNKS_TABLE,
    CURSORS_TABLE,
    CWD_CACHE_TABLE,
    EPISODES_TABLE,
    META_TABLE,
    SCHEMA_VERSION,
    SESSIONS_TABLE,
    SOURCES_TABLE,
    LanceStore,
)
from ssgrep.utilities.types import IndexNotReadyError, IndexStats, RebuildWouldShrinkError

pytestmark = pytest.mark.slow

QUARANTINED_MESSAGE = (
    "The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`."
)


def test_quarantined_upstream_warning_is_filtered():
    """cocoindex's deprecated ST call must not spam the progress bar."""
    app_mod._QUARANTINED_UPSTREAM_WARNINGS.index(QUARANTINED_MESSAGE)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        app_mod._quarantine_upstream_future_warnings()
        warnings.warn(QUARANTINED_MESSAGE, FutureWarning, stacklevel=2)
        warnings.warn("some unrelated future tightening", FutureWarning, stacklevel=2)
    shown = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert [str(w.message) for w in shown] == ["some unrelated future tightening"]


PROMPT = "How do I fix the auth bug?"
RESPONSE = "Use a retry policy with exponential backoff."


def fake_vector(text: str) -> np.ndarray:
    """Deterministic ``(num_tokens, 128)`` float32 matrix for one text (test-only).

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
    """CocoIndex provider stand-in producing ``(num_tokens, 128)`` matrices."""

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


def install_fake_query_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route LanceDB's query-time ``ssgrep`` embedding function to a fake."""
    from lancedb.embeddings.base import TextEmbeddingFunction
    from lancedb.embeddings.registry import EmbeddingFunctionRegistry

    class FakeLanceEmbedding(TextEmbeddingFunction):
        def ndims(self) -> int:
            return DIMENSION

        def generate_embeddings(self, texts, *_args, **_kwargs):
            return [fake_vector(text) for text in (texts if isinstance(texts, list) else [texts])]

        def compute_query_embeddings(self, query, *_args, **_kwargs):
            return [fake_vector(str(query))]

        def compute_source_embeddings(self, texts, *_args, **_kwargs):
            return self.generate_embeddings(texts)

    registry = EmbeddingFunctionRegistry.get_instance()
    monkeypatch.setitem(registry._functions, "ssgrep", FakeLanceEmbedding)


def write_claude_session(session_id: str = "abc123") -> Path:
    root = Path(os.environ["CLAUDE_CONFIG_DIR"])
    project = root / "projects" / "proj-1"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{session_id}.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "type": "user",
                    "cwd": "/work/app",
                    "message": {"content": PROMPT},
                    "timestamp": "2025-01-01T00:00:00Z",
                    "sessionId": session_id,
                },
                {
                    "type": "assistant",
                    "message": {"content": RESPONSE},
                    "timestamp": "2025-01-01T00:00:01Z",
                    "sessionId": session_id,
                },
            )
        )
        + "\n"
    )
    return path


def session_path(session_id: str = "abc123") -> Path:
    return Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "proj-1" / f"{session_id}.jsonl"


def write_long_session(session_id: str = "long") -> Path:
    """Write a session whose response yields >=256 token vectors (IVF-PQ minimum).

    ``fake_vector`` emits one ``(128,)`` row per character trigram of the
    embedding text, so a long response guarantees the corpus has enough token
    vectors for ``ensure_vector_index`` to build the cosine IVF-PQ index
    instead of deferring it.
    """
    root = Path(os.environ["CLAUDE_CONFIG_DIR"])
    project = root / "projects" / "proj-1"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{session_id}.jsonl"
    response = " ".join(
        f"the quick brown fox jumps over the lazy dog number {i} with plenty of words"
        for i in range(12)
    )
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "type": "user",
                    "cwd": "/work/app",
                    "message": {"content": PROMPT},
                    "timestamp": "2025-01-01T00:00:00Z",
                    "sessionId": session_id,
                },
                {
                    "type": "assistant",
                    "message": {"content": response},
                    "timestamp": "2025-01-01T00:00:01Z",
                    "sessionId": session_id,
                },
            )
        )
        + "\n"
    )
    return path


@pytest.fixture(autouse=True)
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the model boundary for deterministic offline stand-ins."""
    monkeypatch.setattr("ssgrep.indexing.embed.ensure_model_downloaded", lambda *a, **kw: None)
    # app.py binds ensure_model_downloaded via `from ... import`, so the module
    # patch above does not reach the app's local reference; patch it directly.
    monkeypatch.setattr(app_mod, "ensure_model_downloaded", lambda *a, **kw: None)
    install_fake_query_embedding(monkeypatch)
    monkeypatch.setattr(app_mod, "ColBERTEmbedder", FakePyLateEmbedder)
    # Search is native LanceDB MaxSim over the multivector column (no re-ranker,
    # no model download), so the query-time embedding function is the only
    # model boundary the round trip needs.
    install_fake_query_embedding(monkeypatch)
    import ssgrep.search as search_mod

    monkeypatch.setattr(search_mod, "_query_matrix", lambda query: fake_vector(str(query)))


def test_fresh_index_search_roundtrip(fake_models) -> None:
    write_claude_session()
    stats = api.index()
    assert stats.session_count == 1
    assert stats.episode_count == 1
    assert stats.chunk_count >= 1
    assert stats.model_id and stats.vector_dimension == DIMENSION

    response = api.search("retry policy", limit=5)
    assert response.total_matches >= 1
    assert any("exponential backoff" in result.excerpt for result in response.results)

    # Stored vectors match the encoder used at index time within float16
    # storage quantization (the column dtype is halffloat since T9).
    repo = LanceStore()
    rows = repo.rows(CHUNKS_TABLE, columns=["chunk_id", "search_text", "vector"])
    assert len(rows) == stats.chunk_count
    for row in rows:
        np.testing.assert_allclose(row["vector"], fake_vector(row["search_text"]), atol=1e-3)

    # The immutable registry recorded the source snapshot.
    registry = repo.rows(SOURCES_TABLE, columns=["key", "adapter", "first_line_hash"])
    assert len(registry) == 1
    assert registry[0]["key"].endswith("abc123.jsonl")
    assert repo.get_meta("index_state") == "ready"


def test_incremental_no_change_is_idempotent(fake_models) -> None:
    write_claude_session()
    api.index()
    repo = LanceStore()
    first = sorted(repo.rows(CHUNKS_TABLE), key=lambda row: row["chunk_id"])
    stats = api.index()
    assert stats.session_count == 1
    second = sorted(repo.rows(CHUNKS_TABLE), key=lambda row: row["chunk_id"])
    assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]
    for before, after in zip(first, second, strict=True):
        assert before["text"] == after["text"]
        np.testing.assert_array_equal(before["vector"], after["vector"])


def test_changed_source_reprocesses_content(fake_models) -> None:
    path = write_claude_session()
    api.index()
    path.write_text(path.read_text().replace("retry policy", "circuit breaker"))
    api.index()
    response = api.search("circuit breaker", limit=5)
    assert any("circuit breaker" in result.excerpt for result in response.results)


def test_disappeared_source_retains_rows_as_tombstone(fake_models) -> None:
    write_claude_session()
    api.index()
    session_path().unlink()
    stats = api.index()
    assert stats.session_count == 1  # rows retained
    assert stats.tombstoned_source_count == 1
    assert stats.tombstoned_chunk_count == stats.chunk_count
    repo = LanceStore()
    assert (
        repo.rows(SESSIONS_TABLE, columns=["source_status"], limit=1)[0]["source_status"]
        == "absent"
    )

    # Restoring the identical byte content memo-hits and flips back to available.
    write_claude_session()
    stats = api.index()
    assert stats.tombstoned_source_count == 0
    assert (
        repo.rows(SESSIONS_TABLE, columns=["source_status"], limit=1)[0]["source_status"]
        == "available"
    )


def test_rebuild_guard_and_allow_shrink(fake_models) -> None:
    write_claude_session()
    api.index()
    session_path().unlink()
    with pytest.raises(RebuildWouldShrinkError):
        api.index(rebuild=True)
    stats = api.index(rebuild=True, allow_shrink=True)
    assert stats.session_count == 0
    assert stats.chunk_count == 0
    assert stats.tombstoned_source_count == 0


def test_missing_source_full_reprocess_preserves_archive(fake_models, monkeypatch) -> None:
    """A memo miss must preserve archived rows, even beyond Lance's default limit."""
    import asyncio

    original_drive = app_mod._drive_update

    async def bounded_drive(*args, **kwargs):
        # The engine retries processor errors indefinitely; bound the regression.
        await asyncio.wait_for(original_drive(*args, **kwargs), timeout=15)

    monkeypatch.setattr(app_mod, "_drive_update", bounded_drive)
    archived = write_claude_session("archived")
    pair = [json.loads(line) for line in archived.read_text().splitlines()]
    archived.write_text("".join(json.dumps(row) + "\n" for _ in range(12) for row in pair))
    live = write_claude_session("live")
    api.index()
    repo = LanceStore()
    tables = (SESSIONS_TABLE, EPISODES_TABLE, CHUNKS_TABLE)

    def archive_rows():
        return {
            name: repo.rows(name, where="session_id = 'archived'", limit=repo.count(name))
            for name in tables
        }

    before = archive_rows()
    assert len(before[SESSIONS_TABLE]) == 1
    assert len(before[EPISODES_TABLE]) == 12
    assert len(before[CHUNKS_TABLE]) > 10
    archived.unlink()
    live.write_text(live.read_text().replace("retry policy", "circuit breaker"))
    monkeypatch.setenv(
        "SSGREP_COCOINDEX_DB", str(Path(os.environ["SSGREP_DATA_DIR"]) / "fresh-journal.db")
    )
    import subprocess
    import sys

    worker = """
import asyncio
import sys
sys.path.insert(0, sys.argv[1])
import pytest, test_app
test_app.fake_models.__wrapped__(pytest.MonkeyPatch())
async def strict_drive(app, *, total, full_reprocess, quiet):
    await asyncio.wait_for(original_drive(
        app, total=total, full_reprocess=full_reprocess, quiet=quiet
    ), timeout=10)
original_drive = test_app.app_mod._drive_update
test_app.app_mod._drive_update = strict_drive
test_app.api.index(full_reprocess=True)
"""
    recovery = subprocess.run(
        [sys.executable, "-c", worker, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert recovery.returncode == 0, recovery.stdout + recovery.stderr
    # The bug this preserves against: cocoindex logging a full traceback per
    # tombstoned source ("component build failed") instead of quietly
    # reconciling from the archived snapshot.
    assert "Traceback" not in recovery.stderr
    assert "component build failed" not in recovery.stderr
    stats = api.status()
    assert stats.tombstoned_source_count == 1
    assert stats.tombstoned_chunk_count == len(before[CHUNKS_TABLE])
    # Non-vacuous: proves the archive branch actually ran for this source,
    # not just that a memo-hit happened to leave its rows untouched.
    assert stats.archived_source_count == 1
    after = archive_rows()
    for name in tables:
        expected = [{**row, "source_status": "absent"} for row in before[name]]
        if name == SESSIONS_TABLE:
            assert after[name][0]["absent_since"] is not None
            expected[0]["absent_since"] = after[name][0]["absent_since"]
        key = {
            SESSIONS_TABLE: "session_id",
            EPISODES_TABLE: "episode_id",
            CHUNKS_TABLE: "chunk_id",
        }[name]
        assert sorted(after[name], key=lambda row: row[key]) == sorted(
            expected, key=lambda row: row[key]
        )
    archived_hits = api.search("retry policy", where="session_id = 'archived'", limit=20)
    assert archived_hits.results
    assert all("exponential backoff" in result.excerpt for result in archived_hits.results)
    live_rows = repo.rows(EPISODES_TABLE, where="session_id = 'live'", limit=100)
    assert live_rows and all("circuit breaker" in row["response_text"] for row in live_rows)
    api.index(full_reprocess=True)
    assert archive_rows() == after


@pytest.mark.parametrize("broken_table", [SESSIONS_TABLE, EPISODES_TABLE, CHUNKS_TABLE])
def test_missing_source_rejects_incomplete_archive(fake_models, broken_table) -> None:

    from ssgrep.pipeline.archive import capture_archives
    from ssgrep.pipeline.sources import read_registry

    path = write_claude_session()
    api.index()
    repo = LanceStore()
    descriptor = read_registry(repo)[str(path.absolute())]
    path.unlink()
    if broken_table == CHUNKS_TABLE:
        # Remove only one chunk: nonempty tables alone do not prove completeness.
        chunk_id = repo.rows(CHUNKS_TABLE, limit=1)[0]["chunk_id"]
        repo.delete(CHUNKS_TABLE, f"chunk_id = '{chunk_id}'")
    else:
        repo.delete(broken_table, "session_id = 'abc123'")
    with pytest.raises(ValueError, match="incomplete archive"):
        capture_archives({descriptor.key: descriptor})


def test_genuine_read_error_on_existing_file_surfaces(fake_models, monkeypatch) -> None:
    """A real adapter failure (file present, unreadable) must not be swallowed as archived."""
    from ssgrep.sessions import adapters as transcript_adapters

    ok = write_claude_session("ok")
    broken = write_claude_session("broken")
    api.index()
    ok.write_text(ok.read_text().replace("retry policy", "circuit breaker"))
    broken_key = str(broken.absolute())
    real_read_source = transcript_adapters.read_source

    def flaky_read_source(source):
        if source.key == broken_key:
            raise PermissionError(13, "Permission denied", broken_key)
        return real_read_source(source)

    monkeypatch.setattr(transcript_adapters, "read_source", flaky_read_source)
    with pytest.raises(RuntimeError, match="component errors"):
        api.index(full_reprocess=True)

    # The failing source must not be archived (its file exists; this is a real
    # error), and the unrelated healthy source must still have been processed.
    repo = LanceStore()
    assert repo.count(SESSIONS_TABLE, "session_id = 'broken'") == 1
    ok_rows = repo.rows(EPISODES_TABLE, where="session_id = 'ok'", limit=100)
    assert ok_rows and all("circuit breaker" in row["response_text"] for row in ok_rows)


def test_rebuild_reindexes_full_corpus(fake_models) -> None:
    write_claude_session("one")
    api.index()
    write_claude_session("two")
    stats = api.index(rebuild=True)
    assert stats.session_count == 2
    assert stats.chunk_count >= 1
    repo = LanceStore()
    assert len(repo.rows(SOURCES_TABLE, columns=["key"])) == 2


def test_rebuild_writes_lateon_metadata(fake_models) -> None:
    """A fresh rebuild writes the ColBERT/v5 metadata contract to the store."""
    write_claude_session("one")
    write_claude_session("two")
    api.index(rebuild=True)
    repo = LanceStore()
    assert repo.get_meta("model_id") == embed_mod.MODEL_ID
    assert repo.get_meta("model_revision") == embed_mod.MODEL_REVISION
    assert repo.get_meta("vector_dimension") == str(embed_mod.DIMENSION)
    assert repo.get_meta("schema_version") == str(SCHEMA_VERSION)


def test_rebuild_multivector_schema_and_norms(fake_models) -> None:
    """A fresh rebuild writes the ColBERT/v5 multivector contract end to end.

    Uses long sessions so the corpus has >=256 token vectors and the cosine
    IVF-PQ index is actually built (not deferred) by a later ``initialize()``.
    """
    write_long_session("one")
    write_long_session("two")
    api.index(rebuild=True)
    repo = LanceStore()

    # Table metadata contract.
    assert repo.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert repo.get_meta("model_id") == embed_mod.MODEL_ID
    assert repo.get_meta("vector_dimension") == str(embed_mod.DIMENSION)

    # chunks.vector column schema is a nested list of 96-d float16 vectors.
    table = repo.table(CHUNKS_TABLE)
    assert table is not None
    field = table.schema.field("vector")
    assert field.type == pa.list_(pa.list_(pa.float16(), DIMENSION))

    # Per-token L2 norms are unit (ColBERT normalize_embeddings=True); the
    # float16 storage dtype quantizes, so the tolerance is f16-scale.
    rows = repo.rows(CHUNKS_TABLE, columns=["vector"])
    assert rows
    for row in rows:
        vectors = np.asarray(row["vector"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    # A later initialize() builds the cosine IVF-PQ index once rows exist.
    # (The proxy_vector index defers here: this corpus has <256 chunks and the
    # proxy column carries one vector per chunk — see the store round-trip
    # test for a corpus large enough to build both indexes.)
    repo.initialize()
    indices = table.list_indices()
    assert any(index.index_type == "IvfPq" and "vector" in index.columns for index in indices)


def test_rebuild_fills_unit_norm_proxy_vectors(fake_models) -> None:
    """The pipeline fills proxy_vector at encode time: unit-mean of the chunk matrix."""
    write_long_session("one")
    api.index(rebuild=True)
    repo = LanceStore()
    rows = repo.rows(CHUNKS_TABLE, columns=["vector", "proxy_vector"])
    assert rows
    for row in rows:
        proxy = row["proxy_vector"]
        assert proxy is not None
        matrix = np.asarray(row["vector"], dtype=np.float32)
        mean = matrix.mean(axis=0)
        np.testing.assert_allclose(proxy, mean / np.linalg.norm(mean), atol=1e-3)
        assert abs(np.linalg.norm(np.asarray(proxy, dtype=np.float32)) - 1.0) < 1e-3


def test_multivector_index_size_ratio_logged(fake_models, caplog) -> None:
    """Log the on-disk size ratio of the multivector index vs the old 768-d single-vector.

    The old backend stored one 768-d float32 vector per chunk (3072 bytes);
    the multivector backend stores a ``(num_tokens, 128)`` float32 matrix per
    chunk and produces roughly 4x more chunks (280-token budget vs 1300-char
    chunks). This is a SOFT assertion: we log the measured ratio and the
    theoretical expectation rather than hard-failing on a range. Building a
    real old-format database is impractical (that machinery is removed), so we
    derive the old footprint from the actual persisted token counts.
    """
    write_long_session("one")
    write_long_session("two")
    api.index(rebuild=True)
    repo = LanceStore()

    def _dir_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    actual_bytes = _dir_size(repo.path)

    # Per-chunk token counts from the persisted multivector column.
    rows = repo.rows(CHUNKS_TABLE, columns=["vector"])
    token_counts = [len(np.asarray(r["vector"], dtype=np.float32)) for r in rows]
    total_tokens = sum(token_counts)
    chunk_count = len(rows)

    # Theoretical old single-vector footprint: one 768-d float32 per chunk,
    # with ~4x fewer chunks at the old 1300-char budget.
    old_bytes_per_chunk = 768 * 4
    old_bytes_est = (chunk_count / 4) * old_bytes_per_chunk

    # Theoretical multivector footprint from the actual token counts.
    new_bytes_est = total_tokens * 128 * 4

    measured_ratio = actual_bytes / old_bytes_est if old_bytes_est else 0.0
    theoretical_ratio = new_bytes_est / old_bytes_est if old_bytes_est else 0.0

    caplog.set_level("INFO")
    print(
        f"[index-size-ratio] chunks={chunk_count} total_tokens={total_tokens} "
        f"actual_bytes={actual_bytes} old_bytes_est={old_bytes_est:.0f} "
        f"measured_ratio={measured_ratio:.1f}x theoretical_ratio={theoretical_ratio:.1f}x"
    )

    # Soft assertion: the multivector index is strictly larger than the old
    # single-vector estimate (tens-of-fold growth expected, but we do not
    # hard-fail on an exact range).
    assert measured_ratio > 1.0


def test_search_rejects_stale_v4_index(fake_models) -> None:
    """A v4 (mpnet, 768-d) index fails the search compatibility gate."""
    write_claude_session()
    api.index()
    repo = LanceStore()
    repo.set_meta("schema_version", "4")
    repo.set_meta("model_id", "sentence-transformers/all-mpnet-base-v2")
    repo.set_meta("vector_dimension", "768")
    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        api.search("retry policy")


def test_scope_run_leaves_out_of_scope_sources_available(fake_models, tmp_path) -> None:
    write_claude_session()
    api.index()
    session_path().unlink()
    stats = api.index(scope=str(tmp_path / "elsewhere"))
    assert stats.session_count == 1
    assert stats.tombstoned_source_count == 0
    repo = LanceStore()
    assert (
        repo.rows(SESSIONS_TABLE, columns=["source_status"], limit=1)[0]["source_status"]
        == "available"
    )


def test_index_not_ready_after_meta_tamper(fake_models) -> None:
    write_claude_session()
    api.index()
    LanceStore().set_meta("model_id", "tampered")
    with pytest.raises(IndexNotReadyError):
        api.index()


def test_live_mode_polls_until_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    sample = IndexStats(
        session_count=1,
        episode_count=1,
        chunk_count=2,
        index_size_bytes=0,
        last_index_time=None,
        model_id="m",
        vector_dimension=DIMENSION,
        skipped_records=0,
        malformed_records=0,
        schema_version=1,
        data_dir=str(tmp_path),
    )

    def fake_once(repository=None, **kwargs) -> IndexStats:
        calls.append(kwargs)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return sample

    monkeypatch.setattr(app_mod, "_reconcile_once", fake_once)
    intervals: list[float] = []
    monkeypatch.setattr(app_mod.time, "sleep", lambda seconds: intervals.append(seconds))
    monkeypatch.setenv(app_mod.POLL_INTERVAL_ENV, "0.25")
    result = app_mod.run(live=True)
    assert len(calls) == 3  # poll cycle, interrupt cycle, final catch-up cycle
    assert intervals == [0.25]
    assert result is sample


def test_guard_shrink_noop_when_index_missing() -> None:
    app_mod._guard_shrink(LanceStore(), 0, allow_shrink=False)


def test_write_cwd_cache_skips_non_cwd_sources() -> None:
    repository = Mock()
    source = Mock()
    source.cache_cwds = False
    app_mod._write_cwd_cache(repository, [source])
    repository.upsert.assert_not_called()


def test_live_mode_invalid_interval_falls_back_to_default(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_once(repository=None, **kwargs) -> object:
        calls.append(kwargs)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return object()

    monkeypatch.setattr(app_mod, "_reconcile_once", fake_once)
    intervals: list[float] = []
    monkeypatch.setattr(app_mod.time, "sleep", lambda seconds: intervals.append(seconds))
    monkeypatch.setenv(app_mod.POLL_INTERVAL_ENV, "not-a-number")
    app_mod.run(live=True)
    assert intervals == [app_mod._DEFAULT_POLL_SECONDS]


def test_full_run_compacts_tables_once_after_reconcile(fake_models, monkeypatch) -> None:
    """One catch-up run optimizes the data tables exactly once, then stamps meta."""
    write_claude_session()
    calls: list[int] = []
    original = LanceStore.optimize_tables

    def spy(self):
        calls.append(1)
        original(self)

    monkeypatch.setattr(LanceStore, "optimize_tables", spy)
    api.index()
    assert len(calls) == 1
    stamp = LanceStore().get_meta("last_optimize_time")
    assert stamp is not None
    datetime.fromisoformat(stamp)

    # A second full run compacts again — once per run, not per source.
    write_claude_session("second")
    api.index()
    assert len(calls) == 2


def test_live_poll_loop_never_compacts(monkeypatch, tmp_path) -> None:
    """Poll cycles (and their interrupt catch-up) reconcile without compacting."""
    calls: list[dict] = []
    sample = IndexStats(
        session_count=1,
        episode_count=1,
        chunk_count=2,
        index_size_bytes=0,
        last_index_time=None,
        model_id="m",
        vector_dimension=DIMENSION,
        skipped_records=0,
        malformed_records=0,
        schema_version=1,
        data_dir=str(tmp_path),
    )

    def fake_once(repository=None, **kwargs) -> IndexStats:
        calls.append(kwargs)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return sample

    monkeypatch.setattr(app_mod, "_reconcile_once", fake_once)
    optimized: list[str] = []
    monkeypatch.setattr(LanceStore, "optimize_tables", lambda self: optimized.append("run"))
    monkeypatch.setattr(app_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setenv(app_mod.POLL_INTERVAL_ENV, "0.01")
    app_mod.run(live=True)
    assert optimized == []


class _FakeTable:
    def __init__(self, *, boom: bool = False) -> None:
        self.optimized = False
        self._boom = boom

    def optimize(self, **kwargs: object) -> None:
        if self._boom:
            raise RuntimeError("compaction exploded")
        self.optimized = True


def test_optimize_failure_warns_and_continues_remaining_tables(monkeypatch, caplog) -> None:
    """A raising table.optimize() logs a warning; the other tables still optimize."""
    store = LanceStore()
    tables = {
        CHUNKS_TABLE: _FakeTable(boom=True),
        EPISODES_TABLE: _FakeTable(),
        SESSIONS_TABLE: _FakeTable(),
        META_TABLE: _FakeTable(),
        CURSORS_TABLE: _FakeTable(),
        CWD_CACHE_TABLE: _FakeTable(),
        SOURCES_TABLE: _FakeTable(),
    }
    monkeypatch.setattr(store, "table", lambda name, create=False: tables[name])
    with caplog.at_level(logging.WARNING):
        store.optimize_tables()
    assert tables[CHUNKS_TABLE].optimized is False
    for name in (
        EPISODES_TABLE,
        SESSIONS_TABLE,
        META_TABLE,
        CURSORS_TABLE,
        CWD_CACHE_TABLE,
        SOURCES_TABLE,
    ):
        table = tables[name]
        assert table is not None
        assert table.optimized is True
    messages = [record.getMessage() for record in caplog.records]
    assert any("chunks" in message and "compaction exploded" in message for message in messages)


def test_optimize_skips_tables_that_do_not_exist(monkeypatch) -> None:
    """A missing table (``table() -> None``) is skipped without error."""
    store = LanceStore()
    tables = {
        CHUNKS_TABLE: _FakeTable(),
        EPISODES_TABLE: None,
        SESSIONS_TABLE: _FakeTable(),
        META_TABLE: None,
        CURSORS_TABLE: _FakeTable(),
        CWD_CACHE_TABLE: None,
        SOURCES_TABLE: _FakeTable(),
    }
    monkeypatch.setattr(store, "table", lambda name, create=False: tables[name])
    store.optimize_tables()
    chunks_table = tables[CHUNKS_TABLE]
    sessions_table = tables[SESSIONS_TABLE]
    assert chunks_table is not None
    assert sessions_table is not None
    assert chunks_table.optimized is True
    assert sessions_table.optimized is True


def test_reconcile_succeeds_when_optimize_raises(fake_models, monkeypatch, caplog) -> None:
    """A wholesale compaction failure never fails the index run."""
    write_claude_session()

    def broken(self):
        raise RuntimeError("compaction exploded")

    monkeypatch.setattr(LanceStore, "optimize_tables", broken)
    with caplog.at_level(logging.WARNING):
        stats = api.index()
    assert stats.session_count == 1
    assert stats.chunk_count >= 1
    assert LanceStore().get_meta("last_optimize_time") is None
    assert any(
        "Post-reconcile compaction failed" in record.getMessage() for record in caplog.records
    )


def test_run_app_wraps_model_load_failures(monkeypatch) -> None:
    """A gated/download failure inside the app run becomes ModelDownloadError."""
    import httpx
    from huggingface_hub.errors import GatedRepoError

    from ssgrep.indexing.embed import ModelDownloadError
    from ssgrep.pipeline.sources import SourceDescriptor

    def gated(*_args, **_kwargs):
        response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x"))
        raise GatedRepoError("401 Client Error. Cannot access gated repo", response=response)

    monkeypatch.setattr(app_mod, "_build_app", gated)
    descriptor = SourceDescriptor(
        adapter="native",
        key="/tmp/x.jsonl",
        path="/tmp/x.jsonl",
        size=0,
        mtime=0.0,
        digest="",
    )
    with pytest.raises(ModelDownloadError, match="gated"):
        app_mod._run_app(
            {"/tmp/x.jsonl": descriptor}, environment=None, full_reprocess=False, quiet=False
        )

    # A non-model failure propagates unchanged.
    def boom(*_args, **_kwargs):
        raise RuntimeError("engine died")

    monkeypatch.setattr(app_mod, "_build_app", boom)
    with pytest.raises(RuntimeError, match="engine died"):
        app_mod._run_app(
            {"/tmp/x.jsonl": descriptor}, environment=None, full_reprocess=False, quiet=False
        )


def test_build_environment_wraps_download_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A download error in _provide() is translated to ModelDownloadError.

    The ``try/except`` in ``_provide()`` wraps ``ensure_model_downloaded`` so
    a network failure during download becomes an actionable
    ``ModelDownloadError`` with recovery instructions.
    """
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        app_mod,
        "ensure_model_downloaded",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("Connection refused")),
    )
    with pytest.raises(embed_mod.ModelDownloadError, match="Connection refused"):
        app_mod._build_environment(quiet=True)

    # A non-model failure propagates unchanged (the ``raise`` after
    # ``raise_model_error`` re-raises the original exception).
    monkeypatch.setattr(
        app_mod,
        "ensure_model_downloaded",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad thing")),
    )
    with pytest.raises(ValueError, match="bad thing"):
        app_mod._build_environment(quiet=True)


def test_note_is_searchable_without_a_separate_index_run(fake_models) -> None:
    """The guidance promises `note` needs no index run and no rebuild.

    The recipe some projects hand-rolled before this (append a record, then
    `index --rebuild`, or the content is silently unretrievable) is the exact
    trap this asserts is gone:
    the note command reindexes as it writes, so an agent that follows rule 5 and
    then verifies with rule 6 gets its own content back on the first try.
    """
    import typer

    from ssgrep.cli.commands.note_command import NoteCommand

    NoteCommand(typer.Typer()).handle(
        title="why does the zzqqxx blorptastic deploy step flake",
        body="Because the retry clock uses wall time; freeze it in the fixture.",
    )

    # No api.index() call here on purpose: `note` must have done it.
    response = api.search("zzqqxx blorptastic", limit=10)

    assert response.total_matches >= 1
    assert any("zzqqxx blorptastic" in result.title for result in response.results), (
        f"the authored note did not return; got {[r.title for r in response.results]}"
    )


def test_note_retrieval_discriminates_against_other_content(fake_models) -> None:
    """Guards the test above: with one document, ANY query "finds" it.

    The note-is-searchable assertion only means something if ranking can tell
    the note apart from unrelated indexed content, so this indexes a second
    session and pins that each query puts the RIGHT document first -- the
    "something returned is not my thing returned" distinction the guidance
    makes in rule 6.
    """
    import typer

    from ssgrep.cli.commands.note_command import NoteCommand

    write_claude_session()
    NoteCommand(typer.Typer()).handle(
        title="why does the zzqqxx blorptastic deploy step flake",
        body="Because the retry clock uses wall time; freeze it in the fixture.",
    )

    note_first = api.search("zzqqxx blorptastic", limit=10)
    session_first = api.search("retry policy exponential backoff", limit=10)

    assert "zzqqxx blorptastic" in note_first.results[0].title
    assert "zzqqxx blorptastic" not in session_first.results[0].title
