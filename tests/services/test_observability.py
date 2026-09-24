"""Unit tests for index observability without opening a real Lance database."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ssgrep.indexing import embed
from ssgrep.services import observability
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE
from ssgrep.utilities.types import IndexNotReadyError, IndexStats


def test_parse_time_accepts_iso_and_degrades_invalid_values():
    expected = datetime(2025, 4, 5, 6, 7, 8, tzinfo=UTC)

    assert observability._parse_time(expected.isoformat()) == expected
    assert observability._parse_time(None) is None
    assert observability._parse_time("") is None
    assert observability._parse_time("definitely not ISO-8601") is None


def test_dir_size_handles_missing_and_nested_files(tmp_path):
    missing = tmp_path / "missing"
    assert observability._dir_size(missing) == 0

    index_path = tmp_path / "index"
    nested = index_path / "nested"
    nested.mkdir(parents=True)
    (index_path / "one.bin").write_bytes(b"123")
    (nested / "two.bin").write_bytes(b"45678")
    (nested / "empty-dir").mkdir()

    assert observability._dir_size(index_path) == 8


def test_status_for_missing_index_returns_defaults_without_querying_store(
    monkeypatch, fake_store_factory
):
    repository = fake_store_factory(exists=False, schema_version=13)
    monkeypatch.setattr(observability, "LanceStore", lambda: repository)

    result = observability.status()

    assert isinstance(result, IndexStats)
    assert result == IndexStats(
        session_count=0,
        episode_count=0,
        chunk_count=0,
        index_size_bytes=0,
        last_index_time=None,
        model_id=embed.MODEL_ID,
        vector_dimension=embed.DIMENSION,
        skipped_records=0,
        malformed_records=0,
        schema_version=13,
        index_exists=False,
        data_dir=str(repository.root),
    )
    assert repository.count_calls == []
    assert repository.meta_calls == []
    assert not repository.path.exists(), "status must not create storage"


def test_status_for_existing_index_reports_counts_metadata_and_disk_usage(
    monkeypatch, fake_store_factory
):
    timestamp = datetime(2025, 5, 6, 7, 8, 9, tzinfo=UTC)
    optimize_time = datetime(2025, 5, 6, 8, 0, 0, tzinfo=UTC)
    metadata = {
        "last_index_time": timestamp.isoformat(),
        "last_optimize_time": optimize_time.isoformat(),
        "model_id": "lightonai/answerai-colbert-small-v1",
        "vector_dimension": "96",
        "skipped_records": "11",
        "malformed_records": "3",
        "schema_version": "5",
    }
    counts = {
        (SESSIONS_TABLE, None): 4,
        (EPISODES_TABLE, None): 9,
        (CHUNKS_TABLE, None): 17,
        (SESSIONS_TABLE, "source_status = 'absent'"): 2,
        (CHUNKS_TABLE, "source_status = 'absent'"): 6,
    }
    repository = fake_store_factory(exists=True, metadata=metadata, counts=counts, schema_version=5)
    repository.seed_rows(
        SESSIONS_TABLE,
        [
            {"runtime": "claude"},
            {"runtime": "pi"},
            {"runtime": "claude"},
            {"runtime": "opencode"},
        ],
    )
    repository.path.mkdir(parents=True)
    (repository.path / "manifest").write_bytes(b"1234")
    nested = repository.path / "table.lance"
    nested.mkdir()
    (nested / "data").write_bytes(b"abcdef")
    monkeypatch.setattr(observability, "LanceStore", lambda: repository)

    result = observability.status()

    assert result == IndexStats(
        session_count=4,
        episode_count=9,
        chunk_count=17,
        index_size_bytes=10,
        last_index_time=timestamp,
        last_optimize_time=optimize_time,
        model_id="lightonai/answerai-colbert-small-v1",
        vector_dimension=96,
        skipped_records=11,
        malformed_records=3,
        schema_version=5,
        tombstoned_source_count=2,
        tombstoned_chunk_count=6,
        index_exists=True,
        data_dir=str(repository.root),
        runtime_counts=(("claude", 2), ("opencode", 1), ("pi", 1)),
    )
    assert repository.rows_calls == [
        (SESSIONS_TABLE, {"columns": ["runtime"], "limit": 1_000_000, "where": None})
    ]
    assert repository.count_calls == [
        (SESSIONS_TABLE, None),
        (EPISODES_TABLE, None),
        (CHUNKS_TABLE, None),
        (SESSIONS_TABLE, "source_status = 'absent'"),
        (CHUNKS_TABLE, "source_status = 'absent'"),
    ]
    assert repository.meta_calls == [
        # Compatibility gate reads the three contract keys first.
        "schema_version",
        "model_id",
        "vector_dimension",
        # Then the stats construction reads every metadata key.
        "last_index_time",
        "last_optimize_time",
        "model_id",
        "vector_dimension",
        "skipped_records",
        "malformed_records",
        "schema_version",
    ]


def test_status_rejects_stale_v4_index(monkeypatch, fake_store_factory):
    """A v4 index (768-d mpnet, schema 4) must be flagged, not silently read."""
    metadata = {
        "schema_version": "4",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "vector_dimension": "768",
    }
    repository = fake_store_factory(exists=True, metadata=metadata, schema_version=5)
    monkeypatch.setattr(observability, "LanceStore", lambda: repository)

    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        observability.status()


def test_status_rejects_table_schema_mismatch(monkeypatch, fake_store_factory):
    repository = fake_store_factory(exists=True, metadata={}, schema_version=5)
    repository.schema_matches = lambda: False
    monkeypatch.setattr(observability, "LanceStore", lambda: repository)

    with pytest.raises(IndexNotReadyError, match="--rebuild"):
        observability.status()


def test_status_uses_defaults_for_absent_metadata(monkeypatch, fake_store_factory):
    counts = {
        (SESSIONS_TABLE, None): 0,
        (EPISODES_TABLE, None): 0,
        (CHUNKS_TABLE, None): 0,
        (SESSIONS_TABLE, "source_status = 'absent'"): 0,
        (CHUNKS_TABLE, "source_status = 'absent'"): 0,
    }
    repository = fake_store_factory(exists=True, metadata={}, counts=counts, schema_version=5)
    repository.seed_rows(SESSIONS_TABLE, [{"runtime": "prime-agent"}])
    repository.path.mkdir(parents=True)
    monkeypatch.setattr(observability, "LanceStore", lambda: repository)

    result = observability.status()

    assert result.last_index_time is None
    assert result.last_optimize_time is None
    assert result.model_id == embed.MODEL_ID
    assert result.vector_dimension == embed.DIMENSION
    assert result.skipped_records == 0
    assert result.malformed_records == 0
    assert result.schema_version == 5
    assert result.index_exists is True
