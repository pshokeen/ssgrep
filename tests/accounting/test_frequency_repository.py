"""Real Lance checkout/direct-take semantics, isolated from the application database."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import lancedb
import pytest

from ssgrep.accounting import fingerprint, index_frequency_repository, storage
from ssgrep.accounting.index_frequency import _generation, read_index_view
from ssgrep.store import LanceStore
from tests.accounting.test_frequency_generations import PROFILE, RARE


def test_checkout_keeps_metadata_and_direct_row_offsets_after_live_append(tmp_path, monkeypatch):
    path = tmp_path / "isolated.lance"
    database = lancedb.connect(str(path), read_consistency_interval=timedelta(seconds=0))
    metadata = database.create_table(
        "metadata",
        [
            {"key": "last_index_time", "value": "epoch-1"},
            {"key": "index_state", "value": "ready"},
        ],
    )
    chunks = database.create_table(
        "chunks", [{"chunk_id": f"chunk-{i}", "cwd": "/approved"} for i in range(20)]
    )

    def private_store():
        store = LanceStore()
        store.path = path
        return store

    monkeypatch.setattr(index_frequency_repository, "LanceStore", private_store)
    live = index_frequency_repository.IndexRepository()
    generation = _generation(live)
    assert generation is not None
    frozen = live.snapshot(generation)
    try:
        before = frozen.rows_page(start=7, limit=5)
        assert [row["chunk_id"] for row in before] == [f"chunk-{i}" for i in range(7, 12)]
        chunks.add([{"chunk_id": "new-chunk", "cwd": "/foreign"}])
        metadata.update(where="key = 'last_index_time'", values={"value": "epoch-2"})
        assert _generation(live) != generation and live.count("chunks") == 21
        assert _generation(frozen) == generation and frozen.count("chunks") == 20
        assert frozen.rows_page(start=7, limit=5) == before
        assert frozen.rows_page(start=19, limit=1) == [{"chunk_id": "chunk-19", "cwd": "/approved"}]
        assert frozen.get_meta("last_index_time") == "epoch-1"
    finally:
        frozen.close()
        live.close()


@pytest.mark.parametrize("duplicate", [False, True], ids=["distinct", "exact-duplicate"])
def test_public_frequency_requires_unique_real_lance_chunk_identities(
    tmp_path, monkeypatch, duplicate
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "identity.lance"
    database = lancedb.connect(str(path), read_consistency_interval=timedelta(seconds=0))
    database.create_table(
        "metadata",
        [
            {"key": "last_index_time", "value": "identity-epoch"},
            {"key": "index_state", "value": "ready"},
            {
                "key": "lexical_stats",
                "value": json.dumps({"n_docs": 511, "avgdl": 1, "df": {RARE.text: 1}}),
            },
        ],
    )
    rows = [{"chunk_id": f"chunk-{i}", "cwd": "/approved"} for i in range(511)]
    if duplicate:
        rows[1]["chunk_id"] = rows[0]["chunk_id"]
    database.create_table("chunks", rows)

    def private_store():
        store = LanceStore()
        store.path = path
        return store

    monkeypatch.setattr(index_frequency_repository, "LanceStore", private_store)
    repository = index_frequency_repository.IndexRepository()
    try:
        result = read_index_view(
            PROFILE,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
        )
        if duplicate:
            assert not result.known and result.unavailable_reason == "frequency_stats_incompatible"
            assert result.rarity_milli(RARE) is None
            descriptor = storage.open_profile_dir(PROFILE, "collector", create=False)
            try:
                encoded = storage.read_file(descriptor, "frequency-build-head.json", 1024 * 1024)
                assert encoded is not None
                head = json.loads(encoded)
            finally:
                os.close(descriptor)
            assert head["job"]["chunk_cursor"] == 1
            assert head["job"]["seen"][0]["records"] == 1
        else:
            assert result.known and result.doc_count == 511
            assert len(result.df) == 1 and result.rarity_milli(RARE) == 8000
    finally:
        repository.close()
