"""Tests for the serializing index shim over the CocoIndex pipeline."""

from __future__ import annotations

import threading
import time

from ssgrep.indexing import indexer
from ssgrep.indexing.embed import DIMENSION, MODEL_ID, MODEL_REVISION
from ssgrep.utilities.types import IndexStats


def sample_stats(data_dir: str) -> IndexStats:
    return IndexStats(
        session_count=1,
        episode_count=1,
        chunk_count=2,
        index_size_bytes=1024,
        last_index_time=None,
        model_id=MODEL_ID,
        vector_dimension=DIMENSION,
        skipped_records=0,
        malformed_records=0,
        schema_version=1,
        data_dir=data_dir,
    )


def test_index_delegates_all_options_to_pipeline(tmp_path, monkeypatch) -> None:
    seen: list[dict] = []

    def fake_run(**kwargs) -> IndexStats:
        seen.append(kwargs)
        return sample_stats(str(tmp_path))

    monkeypatch.setattr(indexer, "_pipeline_run", fake_run)
    result = indexer.index(
        rebuild=True,
        no_subagents=True,
        allow_shrink=True,
        scope=str(tmp_path / "relative" / "repo"),
        quiet=True,
        live=False,
        full_reprocess=True,
    )
    assert result.session_count == 1
    assert seen == [
        {
            "rebuild": True,
            "no_subagents": True,
            "allow_shrink": True,
            "scope": str((tmp_path / "relative" / "repo").absolute()),
            "quiet": True,
            "live": False,
            "full_reprocess": True,
        }
    ]


def test_index_passes_none_scope_through(tmp_path, monkeypatch) -> None:
    seen: list[dict] = []

    def fake_run(**kwargs) -> IndexStats:
        seen.append(kwargs)
        return sample_stats(str(tmp_path))

    monkeypatch.setattr(indexer, "_pipeline_run", fake_run)
    indexer.index(scope=None)
    assert seen[0]["scope"] is None


def test_index_creates_the_lock_file(tmp_path, monkeypatch) -> None:
    calls: list[IndexStats] = []

    def fake_run(**kwargs) -> IndexStats:
        calls.append(sample_stats(str(tmp_path)))
        return calls[-1]

    from ssgrep.store.paths import data_dir

    monkeypatch.setattr(indexer, "_pipeline_run", fake_run)
    indexer.index()
    lock = data_dir() / "index.lock"
    assert lock.exists()
    assert len(calls) == 1


def test_lock_serializes_concurrent_runs(tmp_path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    runs: list[int] = []
    run_counter = 0

    def fake_run(**kwargs) -> IndexStats:
        nonlocal run_counter
        run_counter += 1
        runs.append(run_counter)  # recorded on entry, while the lock is held
        if run_counter == 1:
            entered.set()
            assert release.wait(10), "lock holder timed out"
        return sample_stats(str(tmp_path))

    monkeypatch.setattr(indexer, "_pipeline_run", fake_run)

    first = threading.Thread(target=indexer.index)
    first.start()
    assert entered.wait(10), "first run never acquired the lock"

    second_started = threading.Event()
    second_calls: list[IndexStats] = []

    def second() -> None:
        second_started.set()
        second_calls.append(indexer.index())

    second_thread = threading.Thread(target=second)
    second_thread.start()
    assert second_started.wait(10)
    time.sleep(0.3)
    assert runs == [1], "second run proceeded while the lock was held"
    release.set()
    first.join(10)
    second_thread.join(10)
    assert runs == [1, 2]
    assert len(second_calls) == 1


def test_model_identity_reexports() -> None:
    assert indexer.MODEL_ID == MODEL_ID
    assert indexer.MODEL_REVISION == MODEL_REVISION
    assert indexer.DIMENSION == DIMENSION
