"""Unit tests for the lazy LanceDB repository."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import ssgrep.store as store


class FakeQuery:
    def __init__(self, result: list[dict[str, Any]] | None = None) -> None:
        self.result = [] if result is None else result
        self.calls: list[tuple[Any, ...]] = []

    def where(self, predicate: str, *, prefilter: bool) -> FakeQuery:
        self.calls.append(("where", predicate, prefilter))
        return self

    def refine_factor(self, factor: int) -> FakeQuery:
        self.calls.append(("refine_factor", factor))
        return self

    def nprobes(self, probes: int) -> FakeQuery:
        self.calls.append(("nprobes", probes))
        return self

    def select(self, columns: list[str]) -> FakeQuery:
        self.calls.append(("select", columns))
        return self

    def limit(self, limit: int) -> FakeQuery:
        self.calls.append(("limit", limit))
        return self

    def to_list(self) -> list[dict[str, Any]]:
        self.calls.append(("to_list",))
        return self.result


class FakeMerge:
    def __init__(self, table: FakeTable, key: str) -> None:
        self.table = table
        self.key = key
        self.calls: list[str] = []

    def when_matched_update_all(self) -> FakeMerge:
        self.calls.append("update")
        return self

    def when_not_matched_insert_all(self) -> FakeMerge:
        self.calls.append("insert")
        return self

    def execute(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append("execute")
        self.table.executed = records
        return {"key": self.key, "records": records}


class FakeTable:
    def __init__(
        self,
        *,
        names: list[str] | None = None,
        rows: list[dict[str, Any]] | None = None,
        count: int = 0,
    ) -> None:
        self.schema = SimpleNamespace(names=[] if names is None else names)
        self.result_rows = [] if rows is None else rows
        self.row_count = count
        #: Optional mapping of exact ``where`` strings to canned counts, used by
        #: tests that need ``count_rows(where)`` to differ from ``row_count``
        #: (e.g. the proxy-index "IS NOT NULL" fill guard).
        self.where_counts: dict[str, int] = {}
        self.queries: list[FakeQuery] = []
        self.deleted: list[str] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.executed: list[dict[str, Any]] | None = None
        self.indices: list[Any] = []
        self.created_indices: list[
            tuple[str | None, Any, bool, str | None, str | None, int | None, int | None, int | None]
        ] = []
        self.merge: FakeMerge | None = None
        self.search_args: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def count_rows(self, where: str | None = None) -> int:
        self.last_count_where = where
        if where is not None and where in self.where_counts:
            return self.where_counts[where]
        return self.row_count

    def search(self, *args: Any, **kwargs: Any) -> FakeQuery:
        self.search_args.append((args, kwargs))
        query = FakeQuery(self.result_rows)
        self.queries.append(query)
        return query

    def merge_insert(self, key: str) -> FakeMerge:
        self.merge = FakeMerge(self, key)
        return self.merge

    def delete(self, where: str) -> str:
        self.deleted.append(where)
        return "deleted"

    def update(self, *, where: str, values: dict[str, Any]) -> str:
        self.updated.append((where, values))
        return "updated"

    def list_indices(self) -> list[Any]:
        return self.indices

    def create_index(
        self,
        column: str | None = None,
        *,
        config: Any = None,
        replace: bool = False,
        metric: str | None = None,
        vector_column_name: str | None = None,
        num_sub_vectors: int | None = None,
        num_partitions: int | None = None,
        num_bits: int | None = None,
    ) -> None:
        self.created_indices.append(
            (
                column,
                config,
                replace,
                metric,
                vector_column_name,
                num_sub_vectors,
                num_partitions,
                num_bits,
            )
        )


class FakeDB:
    def __init__(self, tables: dict[str, FakeTable] | None = None) -> None:
        self.tables = {} if tables is None else tables
        self.created: list[tuple[str, Any]] = []
        self.dropped = False
        self.closed = False

    def list_tables(self) -> Any:
        return SimpleNamespace(tables=list(self.tables))

    def open_table(self, name: str) -> FakeTable:
        return self.tables[name]

    def create_table(self, name: str, *, schema: Any, **kwargs: Any) -> FakeTable:
        table = FakeTable(names=list(schema.model_fields))
        self.tables[name] = table
        self.created.append((name, schema))
        return table

    def drop_all_tables(self) -> None:
        self.dropped = True

    def close(self) -> None:
        self.closed = True


def test_quote_constants_and_properties() -> None:
    repository = store.LanceStore()

    assert store.quote("plain") == "'plain'"
    assert store.quote("Ada's") == "'Ada''s'"
    assert repository.root == repository.path.parent
    assert repository.schema_version == store.SCHEMA_VERSION
    assert {
        store.CHUNKS_TABLE,
        store.EPISODES_TABLE,
        store.SESSIONS_TABLE,
        store.CURSORS_TABLE,
        store.META_TABLE,
        store.CWD_CACHE_TABLE,
        store.SOURCES_TABLE,
    } == set(store.TABLE_SCHEMAS)
    assert set(store.__all__) == {
        "CHUNKS_TABLE",
        "CURSORS_TABLE",
        "CWD_CACHE_TABLE",
        "SOURCES_TABLE",
        "EPISODES_TABLE",
        "META_TABLE",
        "SESSIONS_TABLE",
        "LanceStore",
        "ensure_data_dir",
        "quote",
    }


def test_connect_is_lazy_creates_privately_and_reuses_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    fake_db = FakeDB()
    connect_calls: list[tuple[str, Any]] = []
    ensured: list[bool] = []
    chmod_calls: list[tuple[Path, int]] = []

    monkeypatch.setattr(store, "ensure_data_dir", lambda: ensured.append(True))
    monkeypatch.setattr(
        store.lancedb,
        "connect",
        lambda path, read_consistency_interval: (
            connect_calls.append((path, read_consistency_interval)) or fake_db
        ),
    )
    monkeypatch.setattr(store.os, "chmod", lambda path, mode: chmod_calls.append((path, mode)))

    assert repository._connect(create=False) is None
    assert connect_calls == []
    assert repository._connect(create=True) is fake_db
    assert ensured == [True]
    assert connect_calls[0][0] == str(repository.path)
    assert connect_calls[0][1].total_seconds() == 0
    assert chmod_calls == [(repository.path, 0o700)]
    assert repository._connect(create=False) is fake_db
    assert len(connect_calls) == 1


def test_connect_existing_database_without_create(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    repository.path.mkdir(parents=True)
    fake_db = FakeDB()
    monkeypatch.setattr(store.lancedb, "connect", lambda *_args, **_kwargs: fake_db)

    assert repository.exists()
    assert repository._connect(create=False) is fake_db


def test_connect_raises_nofile_limit_only_when_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fd raise fires exactly once, on the real connection path only."""
    repository = store.LanceStore()
    fake_db = FakeDB()
    raises: list[bool] = []
    monkeypatch.setattr(store, "raise_nofile_limit", lambda: raises.append(True))
    monkeypatch.setattr(store.lancedb, "connect", lambda *_args, **_kwargs: fake_db)

    assert repository._connect(create=False) is None
    assert raises == []

    repository.path.mkdir(parents=True)
    assert repository._connect(create=False) is fake_db
    assert repository._connect(create=False) is fake_db
    assert raises == [True]


def test_table_open_create_missing_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    chunks = FakeTable()
    db = FakeDB({store.CHUNKS_TABLE: chunks})
    repository._db = db

    assert repository.table(store.CHUNKS_TABLE) is chunks
    assert repository.table(store.META_TABLE) is None
    metadata = repository.table(store.META_TABLE, create=True)
    assert metadata is db.tables[store.META_TABLE]
    assert db.created == [(store.META_TABLE, store.TABLE_SCHEMAS[store.META_TABLE])]
    with pytest.raises(KeyError, match="unknown ssgrep table: mystery") as exc_info:
        repository.table("mystery", create=True)
    assert isinstance(exc_info.value.__cause__, KeyError)

    empty = store.LanceStore()
    monkeypatch.setattr(empty, "_connect", lambda *, create: None)
    assert empty.table(store.CHUNKS_TABLE) is None


def test_schema_matches_exact_columns_and_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    tables = {
        name: FakeTable(names=list(schema.model_fields))
        for name, schema in store.TABLE_SCHEMAS.items()
    }
    repository = store.LanceStore()
    repository._db = FakeDB(tables)
    assert repository.schema_matches() is True

    tables[store.CHUNKS_TABLE].schema.names.append("future_column")
    assert repository.schema_matches() is False

    monkeypatch.setattr(repository, "table", lambda _name: (_ for _ in ()).throw(RuntimeError()))
    assert repository.schema_matches() is False


def test_initialize_creates_every_table_and_vector_index(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    created: list[tuple[str, bool]] = []
    vector_calls: list[bool] = []
    monkeypatch.setattr(
        repository, "table", lambda name, *, create=False: created.append((name, create))
    )
    monkeypatch.setattr(repository, "ensure_vector_index", lambda: vector_calls.append(True))

    assert repository.initialize() is repository
    assert created == [(name, True) for name in store.TABLE_SCHEMAS]
    assert vector_calls == [True]


def test_reset_drops_closes_removes_and_reinitializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = store.LanceStore()
    repository.path = tmp_path / "data" / "lancedb"
    repository.path.mkdir(parents=True)
    (repository.path / "old").write_text("old")
    db = FakeDB()
    repository._db = db
    initialized: list[bool] = []
    monkeypatch.setattr(
        repository,
        "initialize",
        lambda: initialized.append(True) or repository,
    )

    assert repository.reset() is repository
    assert db.dropped and db.closed
    assert repository._db is None
    assert not repository.path.exists()
    assert initialized == [True]

    initialized.clear()
    assert repository.reset() is repository
    assert initialized == [True]


def test_count_rows_delete_and_update_for_present_or_missing_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=7)
    monkeypatch.setattr(repository, "table", lambda name: table if name == "present" else None)

    assert repository.count("present", "source_status = 'available'") == 7
    assert table.last_count_where == "source_status = 'available'"
    assert repository.count("missing") == 0
    assert repository.delete("present", "id = 'one'") == "deleted"
    assert table.deleted == ["id = 'one'"]
    assert repository.delete("missing", "anything") is None
    values = {"source_status": "absent"}
    assert repository.update("present", "id = 'one'", values) == "updated"
    assert table.updated == [("id = 'one'", values)]
    assert repository.update("missing", "anything", values) is None


def test_rows_builds_query_and_missing_table_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    table = FakeTable(rows=[{"value": "answer"}])
    monkeypatch.setattr(repository, "table", lambda name: table if name == "present" else None)

    assert repository.rows("missing") == []
    assert repository.rows("present", where="key = 'question'", columns=("value",), limit=1) == [
        {"value": "answer"}
    ]
    assert table.queries[-1].calls == [
        ("where", "key = 'question'", True),
        ("select", ["value"]),
        ("limit", 1),
        ("to_list",),
    ]
    assert repository.rows("present") == [{"value": "answer"}]
    assert table.queries[-1].calls == [("to_list",)]


def test_records_copy_inputs_and_fill_only_optional_defaults() -> None:
    source = {"path": "/one", "size": 2, "mtime": 3.0, "cwds": "[]"}
    records = store.LanceStore._records(store.CWD_CACHE_TABLE, source)
    assert records == [source]
    assert records[0] is not source

    chunks = store.LanceStore._records(
        store.CHUNKS_TABLE,
        [
            {
                "chunk_id": "c",
                "episode_id": "e",
                "session_id": "s",
                "project": "/project",
                "source_path": "/session",
                "text": "text",
                "vector": [0.0] * 256,
                "content_type": "prompt",
            }
        ],
    )
    assert chunks[0]["source_project"] is None
    assert chunks[0]["source_status"] == "available"
    assert chunks[0]["is_subagent"] is False
    assert chunks[0]["title"] == ""
    assert "chunk_id" in chunks[0]


def test_upsert_empty_and_merge_insert_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    table = FakeTable()
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    assert repository.upsert(store.META_TABLE, []) is None
    result = repository.upsert(store.META_TABLE, {"key": "model", "value": "local"})
    assert result == {
        "key": store.PRIMARY_KEYS[store.META_TABLE],
        "records": [{"key": "model", "value": "local"}],
    }
    assert table.merge is not None
    assert table.merge.calls == ["update", "insert", "execute"]


def test_metadata_convenience_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    row_calls: list[tuple[Any, ...]] = []
    results = [[{"value": 123}], []]

    def fake_rows(name: str, **kwargs: Any) -> list[dict[str, Any]]:
        row_calls.append((name, kwargs))
        return results.pop(0)

    monkeypatch.setattr(repository, "rows", fake_rows)
    assert repository.get_meta("owner's key", "fallback") == "123"
    assert row_calls[-1] == (
        store.META_TABLE,
        {"where": "key = 'owner''s key'", "columns": ["value"], "limit": 1},
    )
    assert repository.get_meta("missing", "fallback") == "fallback"

    upserts: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        repository, "upsert", lambda name, rows: upserts.append((name, rows)) or "written"
    )
    assert repository.set_meta("model", "test") == "written"
    assert upserts == [(store.META_TABLE, {"key": "model", "value": "test"})]


def test_index_tuning_constants() -> None:
    """Executable spec of the shipped IVF-PQ tuning defaults (Task 3)."""
    assert store.NUM_SUB_VECTORS == 12
    assert 96 % store.NUM_SUB_VECTORS == 0
    assert store.REFINE_FACTOR == 2
    assert store.MAX_NUM_PARTITIONS == 64
    assert store.NPROBES == max(1, store.MAX_NUM_PARTITIONS // 4)
    assert store.NPROBES == 16


def test_pq_bits_constants() -> None:
    """Executable spec of the PQ code-width decision (Task 7)."""
    assert store.DEFAULT_PQ_BITS == 8
    assert store.ALLOWED_PQ_BITS == (4, 8)


@pytest.mark.parametrize("bits", [4, 8])
def test_ensure_vector_index_passes_explicit_num_bits(
    monkeypatch: pytest.MonkeyPatch, bits: int
) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=10)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    repository.ensure_vector_index(num_bits=bits)
    assert table.created_indices == [
        (None, None, False, "cosine", "vector", 12, 1, bits),
        (None, None, False, "cosine", "proxy_vector", 12, 1, bits),
    ]


def test_ensure_vector_index_resolves_num_bits_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=10)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    monkeypatch.delenv("SSGREP_PQ_BITS", raising=False)
    repository.ensure_vector_index(replace=True)
    assert table.created_indices == [
        (None, None, True, "cosine", "vector", 12, 1, 8),
        (None, None, True, "cosine", "proxy_vector", 12, 1, 8),
    ]

    monkeypatch.setenv("SSGREP_PQ_BITS", "4")
    repository.ensure_vector_index(replace=True)
    assert table.created_indices[-2:] == [
        (None, None, True, "cosine", "vector", 12, 1, 4),
        (None, None, True, "cosine", "proxy_vector", 12, 1, 4),
    ]


@pytest.mark.parametrize("raw", ["3", "16", "0", "-4", "not-a-number"])
def test_ensure_vector_index_rejects_invalid_env_bits_explicitly(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """Invalid SSGREP_PQ_BITS must fail loudly, never fall back silently."""
    repository = store.LanceStore()
    table = FakeTable(count=10)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)
    monkeypatch.setenv("SSGREP_PQ_BITS", raw)

    with pytest.raises(ValueError, match="SSGREP_PQ_BITS"):
        repository.ensure_vector_index()
    assert table.created_indices == []


@pytest.mark.parametrize("bits", [3, 16, 0])
def test_ensure_vector_index_rejects_invalid_num_bits_argument(
    monkeypatch: pytest.MonkeyPatch, bits: int
) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=10)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    with pytest.raises(ValueError, match="num_bits"):
        repository.ensure_vector_index(num_bits=bits)
    assert table.created_indices == []


def test_invalid_env_bits_raise_even_when_index_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation precedes every early return: no silent no-op on bad config."""
    repository = store.LanceStore()
    empty = FakeTable(count=0)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: empty)
    monkeypatch.setenv("SSGREP_PQ_BITS", "3")

    with pytest.raises(ValueError, match="SSGREP_PQ_BITS"):
        repository.ensure_vector_index()

    existing = FakeTable(count=10)
    existing.indices = [SimpleNamespace(index_type="IvfPq", columns=["vector"])]
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: existing)
    with pytest.raises(ValueError, match="SSGREP_PQ_BITS"):
        repository.ensure_vector_index()


@pytest.mark.parametrize(
    ("rows", "partitions"),
    [(0, 1), (1, 1), (4095, 1), (4096, 1), (8192, 2), (262143, 63), (262144, 64), (10**8, 64)],
)
def test_partition_formula_follows_rows_over_4096_capped(rows: int, partitions: int) -> None:
    assert store._index_num_partitions(rows) == partitions


def test_ensure_vector_index_creates_cosine_index_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=10)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    repository.ensure_vector_index()
    assert table.created_indices == [
        (None, None, False, "cosine", "vector", 12, 1, 8),
        (None, None, False, "cosine", "proxy_vector", 12, 1, 8),
    ]


def test_ensure_vector_index_skips_empty_and_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    empty = FakeTable(count=0)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: empty)
    repository.ensure_vector_index()
    assert empty.created_indices == []

    table = FakeTable(count=10)
    table.indices = [SimpleNamespace(index_type="IvfPq", columns=["vector"])]
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)
    repository.ensure_vector_index()
    # The vector index is skipped, but a pre-T10 corpus has no proxy index
    # yet, so the second index is still built (upgrade path).
    assert table.created_indices == [(None, None, False, "cosine", "proxy_vector", 12, 1, 8)]

    repository.ensure_vector_index(replace=True)
    assert table.created_indices[-2:] == [
        (None, None, True, "cosine", "vector", 12, 1, 8),
        (None, None, True, "cosine", "proxy_vector", 12, 1, 8),
    ]


def test_ensure_vector_index_skips_proxy_index_when_column_unfilled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corpus written before the proxy fill carries all-null proxy values.

    Training IVF-PQ on an all-null column is pointless; the proxy index is
    deferred until rows actually carry proxy vectors (next reconcile/rebuild).
    """
    repository = store.LanceStore()
    table = FakeTable(count=30)
    table.where_counts["proxy_vector IS NOT NULL"] = 0
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    repository.ensure_vector_index()
    assert table.created_indices == [(None, None, False, "cosine", "vector", 12, 1, 8)]


def test_ensure_vector_index_defers_when_corpus_too_small_for_pq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=5)

    def too_small(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("lance error: Not enough rows to train PQ. Requires 256 rows")

    monkeypatch.setattr(table, "create_index", too_small)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    repository.ensure_vector_index()
    assert table.created_indices == []


def test_index_noise_predicate_drops_kmeans_advisories() -> None:
    is_noise = store._is_index_noise
    assert is_noise(
        b"[2026-08-17T13:14:14Z WARN  lance_index::vector::kmeans] "
        b"KMeans: more than 10% of clusters are empty: 1 of 2."
    )
    assert is_noise(b"    Help: this could mean your dataset has many duplicate vectors.")
    assert not is_noise(b"index built")
    assert not is_noise(b"")


def test_filtered_stderr_suppresses_kmeans_lines_and_replays_rest() -> None:
    read_fd, write_fd = os.pipe()
    saved = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        with store._filtered_stderr(store._is_index_noise):
            os.write(2, b"KMeans: more than 10% of clusters are empty: 1 of 2.\n")
            os.write(2, b"    Help: this could mean your dataset has many duplicate vectors.\n")
            os.write(2, b"real message\n")
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    data = b""
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        data += chunk
    os.close(read_fd)
    assert b"KMeans" not in data
    assert b"Help" not in data
    assert b"real message" in data


def test_initialize_surfaces_vector_index_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = store.LanceStore()
    table = FakeTable(count=10)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("index build failed")

    monkeypatch.setattr(table, "create_index", boom)
    monkeypatch.setattr(repository, "table", lambda _name, *, create=False: table)

    with pytest.raises(RuntimeError, match="index build failed"):
        repository.initialize()


def test_initialize_creates_cosine_vector_index_on_populated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    repository = store.LanceStore()
    rows = [
        {
            "chunk_id": f"c{i}",
            "episode_id": "e",
            "session_id": "s",
            "project": "/p",
            "source_path": "/s",
            "text": "text",
            "search_text": "text",
            "vector": [[0.1] * 96 for _ in range(20)],
            "content_type": "prompt",
        }
        for i in range(30)
    ]
    repository.upsert(store.CHUNKS_TABLE, rows)

    repository.initialize()

    table = repository.table(store.CHUNKS_TABLE)
    assert table is not None
    indices = table.list_indices()
    assert any(index.index_type == "IvfPq" and "vector" in index.columns for index in indices)


def test_multivector_search_handles_absent_empty_and_filtered_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    monkeypatch.setattr(repository, "table", lambda _name: None)
    assert repository.multivector_search(np.zeros((3, 96)), limit=5) == []

    empty = FakeTable(count=0)
    monkeypatch.setattr(repository, "table", lambda _name: empty)
    assert repository.multivector_search(np.zeros((3, 96)), limit=5) == []
    assert empty.search_args == []

    populated = FakeTable(rows=[{"chunk_id": "c"}], count=1)
    monkeypatch.setattr(repository, "table", lambda _name: populated)
    matrix = np.zeros((3, 96), dtype=np.float32)
    assert repository.multivector_search(matrix, limit=3, where="project = '/p'") == [
        {"chunk_id": "c"}
    ]
    assert len(populated.search_args) == 1
    args, kwargs = populated.search_args[0]
    assert args[0] is matrix
    assert kwargs == {"vector_column_name": "vector"}
    assert populated.queries[-1].calls == [
        ("refine_factor", 2),
        ("nprobes", 16),
        ("where", "project = '/p'", True),
        ("limit", 3),
        ("to_list",),
    ]

    assert repository.multivector_search(np.zeros((2, 96)), limit=1) == [{"chunk_id": "c"}]
    assert len(populated.search_args) == 2
    args2, kwargs2 = populated.search_args[1]
    assert args2[0].shape == (2, 96)
    assert kwargs2 == {"vector_column_name": "vector"}
    assert populated.queries[-1].calls == [
        ("refine_factor", 2),
        ("nprobes", 16),
        ("limit", 1),
        ("to_list",),
    ]


def test_multivector_search_env_overrides_are_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    table = FakeTable(rows=[{"chunk_id": "c"}], count=1)
    monkeypatch.setattr(repository, "table", lambda _name: table)
    matrix = np.zeros((2, 96), dtype=np.float32)

    monkeypatch.setenv("SSGREP_REFINE_FACTOR", "100")
    monkeypatch.setenv("SSGREP_NPROBES", "0")
    repository.multivector_search(matrix, limit=1)
    assert table.queries[-1].calls[:2] == [("refine_factor", 20), ("nprobes", 1)]

    monkeypatch.setenv("SSGREP_REFINE_FACTOR", "not-a-number")
    monkeypatch.setenv("SSGREP_NPROBES", "-5")
    repository.multivector_search(matrix, limit=1)
    assert table.queries[-1].calls[:2] == [("refine_factor", 2), ("nprobes", 1)]


def test_two_stage_search_handles_absent_and_empty_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    monkeypatch.setattr(repository, "table", lambda _name: None)
    assert repository.two_stage_search(np.zeros((3, 96)), query_proxy=np.zeros(96), limit=5) == []

    empty = FakeTable(count=0)
    monkeypatch.setattr(repository, "table", lambda _name: empty)
    assert repository.two_stage_search(np.zeros((3, 96)), query_proxy=np.zeros(96), limit=5) == []
    assert empty.search_args == []


def test_two_stage_search_stage_one_chain_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 1: proxy ANN with refine/nprobes/where/select(chunk_id)/limit."""
    repository = store.LanceStore()
    table = FakeTable(rows=[{"chunk_id": "c"}], count=1)
    monkeypatch.setattr(repository, "table", lambda _name: table)
    fetched: list[str | None] = []

    def fake_rows(name: str, *, where: str | None = None, **_kwargs: Any) -> list[dict[str, Any]]:
        fetched.append(where)
        return []

    monkeypatch.setattr(repository, "rows", fake_rows)

    proxy = np.ones(96, dtype=np.float32) / 8
    assert (
        repository.two_stage_search(
            np.zeros((2, 96)), query_proxy=proxy, limit=7, where="runtime = 'pi'"
        )
        == []
    )
    args, kwargs = table.search_args[0]
    assert args[0] is proxy
    assert kwargs == {"vector_column_name": "proxy_vector"}
    assert table.queries[-1].calls == [
        ("refine_factor", 2),
        ("nprobes", 16),
        ("where", "runtime = 'pi'", True),
        ("select", ["chunk_id", "_distance"]),
        ("limit", 7),
        ("to_list",),
    ]
    assert fetched == ["chunk_id IN ('c')"]


def test_two_stage_search_pool_size_slices_after_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pool_size mirrors multivector_search's limit: rollup pool, not candidates."""
    repository = store.LanceStore()
    table = FakeTable(rows=[{"chunk_id": "c"}], count=1)
    monkeypatch.setattr(repository, "table", lambda _name: table)

    e0 = np.zeros(96, dtype=np.float32)
    e0[0] = 1.0
    e1 = np.zeros(96, dtype=np.float32)
    e1[1] = 1.0
    candidates = [
        {
            "chunk_id": f"c{i}",
            "episode_id": "e",
            "vector": [e0.tolist(), e1.tolist()] if i < 3 else [[0.0] * 96],
        }
        for i in range(5)
    ]

    def fake_rows(name: str, *, where: str | None = None, **_kwargs: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in candidates]

    monkeypatch.setattr(repository, "rows", fake_rows)

    scored = repository.two_stage_search(np.stack([e0, e1]), query_proxy=e0, limit=10, pool_size=3)
    assert len(scored) == 3
    # Best exact scores survive the slice; orthogonal chunks are dropped.
    assert all(row["chunk_id"] in {"c0", "c1", "c2"} for row in scored)


def test_two_stage_search_scores_candidates_with_exact_client_maxsim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 2 scores ONLY candidate chunks and reports 1 - MaxSim distances.

    Stage-1 proxy cosine scores must never leak into the output: final
    ranking uses exclusively the stage-2 MaxSim-derived ``_distance``.
    """
    repository = store.LanceStore()
    table = FakeTable(rows=[{"chunk_id": "c"}], count=1)
    monkeypatch.setattr(repository, "table", lambda _name: table)

    e0 = np.zeros(96, dtype=np.float32)
    e0[0] = 1.0
    e1 = np.zeros(96, dtype=np.float32)
    e1[1] = 1.0
    candidates = [
        {"chunk_id": "b-partial", "episode_id": "e", "vector": [e1.tolist()]},
        {"chunk_id": "a-exact", "episode_id": "e", "vector": [e0.tolist(), e1.tolist()]},
        {"chunk_id": "c-none", "episode_id": "e", "vector": [[0.0] * 96]},
    ]
    captured_where: list[str | None] = []

    def fake_rows(name: str, *, where: str | None = None, **_kwargs: Any) -> list[dict[str, Any]]:
        captured_where.append(where)
        return [dict(row) for row in candidates]

    monkeypatch.setattr(repository, "rows", fake_rows)
    query_matrix = np.stack([e0, e1])

    scored = repository.two_stage_search(query_matrix, query_proxy=e0, limit=3)

    assert len(captured_where) == 1
    assert captured_where[0] is not None
    assert captured_where[0] == "chunk_id IN ('x', 'y', 'z')" or captured_where[0].startswith(
        "chunk_id IN ("
    )
    by_id = {row["chunk_id"]: row["_distance"] for row in scored}
    assert abs(by_id["a-exact"] - (1 - 2.0)) < 1e-6
    assert abs(by_id["b-partial"] - (1 - 1.0)) < 1e-6
    assert abs(by_id["c-none"] - (1 - 0.0)) < 1e-6
    # Deterministic engine-like order: best score first, chunk_id tiebreak.
    assert [row["chunk_id"] for row in scored] == ["a-exact", "b-partial", "c-none"]


def test_two_stage_search_returns_empty_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = store.LanceStore()
    table = FakeTable(rows=[], count=1)
    monkeypatch.setattr(repository, "table", lambda _name: table)
    monkeypatch.setattr(
        repository, "rows", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unreachable"))
    )

    assert repository.two_stage_search(np.zeros((2, 96)), query_proxy=np.zeros(96), limit=5) == []


def _unit_token(i: int) -> list[list[float]]:
    """One-token 96-dim matrix whose single vector is basis vector e_i."""
    row = [0.0] * 96
    row[i] = 1.0
    return [row]


def _round_trip_chunk(chunk_id: str, vector: list[list[float]]) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "episode_id": "e",
        "session_id": "s",
        "project": "/p",
        "source_path": "/s",
        "text": chunk_id,
        "search_text": chunk_id,
        "vector": vector,
        "content_type": "prompt",
    }


def test_multivector_search_flat_round_trip_absolute_maxsim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine pin (flat/bypass path): _distance = T + 1 - 2*MaxSim exactly.

    Hand-computable 96-dim unit vectors; query [e0, e1] (T=2). The corpus sits
    below the PQ training floor so no IVF-PQ index exists and LanceDB
    brute-forces on its own scale: exact match -> -1.0, half overlap -> +1.0,
    orthogonal -> +3.0. refine_factor is a no-op here, so this scale only
    coincides with the refined 1 - MaxSim scale on the identity case (M=T);
    ordering is preserved either way and real corpora are always indexed.
    """
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    repository = store.LanceStore()
    e = [_unit_token(i)[0] for i in range(4)]
    repository.upsert(
        store.CHUNKS_TABLE,
        [
            _round_trip_chunk("exact", [e[0], e[1]]),
            _round_trip_chunk("partial", [e[1], e[2]]),
            _round_trip_chunk("none", [e[2], e[3]]),
        ],
    )
    query = np.stack([e[0], e[1]]).astype(np.float32)

    results = repository.multivector_search(query, limit=10)

    distances = {row["chunk_id"]: row["_distance"] for row in results}
    assert abs(distances["exact"] - (2 + 1 - 2 * 2.0)) < 1e-4
    assert abs(distances["partial"] - (2 + 1 - 2 * 1.0)) < 1e-4
    assert abs(distances["none"] - (2 + 1 - 2 * 0.0)) < 1e-4


def test_multivector_search_refined_round_trip_absolute_maxsim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine pin (indexed path): refine_factor(1) rescores onto 1 - MaxSim.

    With an IVF-PQ index an unrefined ANN search reports the lossy
    sum-of-min-cosine-distance scale (in [0, 2T]; ~0 for a perfect match), so
    the production method's refine_factor(1) rescore is what puts every score
    on the exact _distance = 1 - MaxSim scale (exact match, T=2 -> -1.0).
    """
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    repository = store.LanceStore()
    e = [_unit_token(i)[0] for i in range(4)]
    rng = np.random.default_rng(7)
    filler = [
        _round_trip_chunk(
            f"filler{i}",
            (lambda m: (m / np.linalg.norm(m, axis=1, keepdims=True)).tolist())(
                rng.normal(size=(16, 96))
            ),
        )
        for i in range(30)
    ]
    repository.upsert(
        store.CHUNKS_TABLE,
        [
            _round_trip_chunk("exact", [e[0], e[1]]),
            _round_trip_chunk("partial", [e[1], e[2]]),
            _round_trip_chunk("none", [e[2], e[3]]),
            *filler,
        ],
    )
    repository.initialize()
    query = np.stack([e[0], e[1]]).astype(np.float32)

    results = repository.multivector_search(query, limit=50)

    distances = {row["chunk_id"]: row["_distance"] for row in results}
    assert abs(distances["exact"] - (1 - 2.0)) < 1e-4
    assert abs(distances["partial"] - (1 - 1.0)) < 1e-4
    assert abs(distances["none"] - (1 - 0.0)) < 1e-4

    table = repository.table(store.CHUNKS_TABLE)
    assert table is not None
    raw = table.search(query, vector_column_name="vector").limit(50).to_list()
    raw_distances = {row["chunk_id"]: row["_distance"] for row in raw}
    assert abs(raw_distances["exact"]) < 0.5
    assert abs((2 - raw_distances["exact"]) - 2.0) < 0.5


def _unit_mean(matrix: list[list[float]]) -> list[float]:
    arr = np.asarray(matrix, dtype=np.float32)
    return (arr.mean(axis=0) / np.linalg.norm(arr.mean(axis=0))).tolist()


def test_two_stage_search_round_trip_absolute_maxsim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine pin (two-stage path): stage 2 reproduces the refined MaxSim scale.

    Hand-computable 96-dim unit vectors; query [e0, e1] (T=2). Stage 1 selects
    candidates from the proxy_vector IVF-PQ index, stage 2 rescores exactly
    those chunks client-side: exact match -> 1 - 2.0, partial -> 1 - 1.0,
    orthogonal -> 1 - 0.0. This is the round-trip proof that client-side
    MaxSim rescoring lands on the same absolute scale as the production
    single-stage refined path.
    """
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    repository = store.LanceStore()
    e = [_unit_token(i)[0] for i in range(4)]
    rng = np.random.default_rng(11)
    # 260 filler chunks: the proxy column carries one vector per chunk, so the
    # IVF-PQ 256-row training floor needs >=256 chunks for BOTH indexes to build.
    filler = []
    for i in range(260):
        matrix = rng.normal(size=(16, 96))
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        row = _round_trip_chunk(f"filler{i}", matrix.tolist())
        row["proxy_vector"] = _unit_mean(matrix.tolist())
        filler.append(row)
    hand_rows = []
    for chunk_id, matrix in (
        ("exact", [e[0], e[1]]),
        ("partial", [e[1], e[2]]),
        ("none", [e[2], e[3]]),
    ):
        row = _round_trip_chunk(chunk_id, [list(token) for token in matrix])
        row["proxy_vector"] = _unit_mean([list(token) for token in matrix])
        hand_rows.append(row)
    repository.upsert(store.CHUNKS_TABLE, [*hand_rows, *filler])
    repository.initialize()

    table = repository.table(store.CHUNKS_TABLE)
    assert table is not None
    indices = table.list_indices()
    assert any(index.index_type == "IvfPq" and "vector" in index.columns for index in indices)
    assert any(index.index_type == "IvfPq" and "proxy_vector" in index.columns for index in indices)

    query = np.stack([e[0], e[1]]).astype(np.float32)
    query_proxy = np.asarray(_unit_mean(query.tolist()), dtype=np.float32)

    # Candidate limit covers the whole corpus, so even the orthogonal chunk
    # (which a real 50-candidate prefilter would rightly drop) is rescored.
    results = repository.two_stage_search(query, query_proxy=query_proxy, limit=300)

    distances = {row["chunk_id"]: row["_distance"] for row in results}
    assert abs(distances["exact"] - (1 - 2.0)) < 1e-4
    assert abs(distances["partial"] - (1 - 1.0)) < 1e-4
    assert abs(distances["none"] - (1 - 0.0)) < 1e-4

    # A tight candidate budget really does prefilter: only 5 survivors, and
    # the exact match ranks first on its stage-2 MaxSim score.
    tight = repository.two_stage_search(query, query_proxy=query_proxy, limit=5)
    assert 0 < len(tight) <= 5
    assert tight[0]["chunk_id"] == "exact"
    scores = [row["_distance"] for row in tight]
    assert scores == sorted(scores)


def test_close_closes_when_supported_and_always_clears_handle() -> None:
    repository = store.LanceStore()
    db = FakeDB()
    repository._db = db
    repository.close()
    assert db.closed
    assert repository._db is None

    repository._db = object()
    repository.close()
    assert repository._db is None
    repository.close()
