"""Tests for Lance table schemas and the local embedding adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from ssgrep.indexing.embed import DIMENSION, MODEL_ID
from ssgrep.store import schema


def test_embedding_adapter_dimension_and_generation(monkeypatch) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_encode(texts, *, is_query, normalize_embeddings):
        calls.append((list(texts), is_query))
        return [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]

    monkeypatch.setattr(schema, "load_embedder", lambda: SimpleNamespace(encode=fake_encode))
    adapter = schema.SsgrepEmbedding.create()

    assert adapter.ndims() == DIMENSION == 96
    result = adapter.generate_embeddings(iter(("first", "second")), "ignored", option=True)
    assert len(result) == 2
    assert all(v.shape == (2, 2) for v in result)  # (num_tokens, dim) per text
    assert calls == [(["first"], False), (["second"], False)]

    query = adapter.compute_query_embeddings("query")
    assert query.shape == (2, 2)
    assert calls[-1] == (["query"], True)
    assert MODEL_ID == "lightonai/answerai-colbert-small-v1"


def test_table_schema_and_primary_key_maps_are_complete() -> None:
    expected = {"chunks", "episodes", "sessions", "cursors", "metadata", "cwd_cache", "sources"}
    assert schema.SCHEMA_VERSION == 6
    assert set(schema.TABLE_SCHEMAS) == expected
    assert set(schema.PRIMARY_KEYS) == expected
    assert schema.PRIMARY_KEYS == {
        "chunks": "chunk_id",
        "episodes": "episode_id",
        "sessions": "session_id",
        "cursors": "path",
        "metadata": "key",
        "cwd_cache": "path",
        "sources": "key",
    }
    assert schema.TABLE_SCHEMAS == {
        "chunks": schema.ChunkModel,
        "episodes": schema.EpisodeModel,
        "sessions": schema.SessionModel,
        "cursors": schema.CursorModel,
        "metadata": schema.MetadataModel,
        "cwd_cache": schema.CwdCacheModel,
        "sources": schema.SourceModel,
    }
    assert schema.COMPAT_TABLES == ("chunks", "episodes", "sessions")


def test_chunk_model_fields_and_defaults() -> None:
    fields = schema.ChunkModel.model_fields
    assert fields["source_project"].default is None
    assert fields["source_status"].default == "available"
    assert fields["title"].default == ""
    assert fields["timestamp"].default is None
    assert fields["files_touched"].default == ""
    assert fields["tool_names"].default == ""
    assert fields["is_subagent"].default is False
    assert fields["parent_session_id"].default is None
    assert fields["agent_type"].default is None
    assert fields["agent_name"].default is None
    assert fields["agent_description"].default is None
    assert fields["agent_model"].default is None
    assert fields["claude_version"].default is None
    assert fields["entrypoint"].default is None
    assert fields["permission_mode"].default is None
    assert fields["user_type"].default is None
    assert fields["chunk_id"].is_required()
    assert fields["vector"].is_required()
    assert fields["runtime"].default == "claude"
    assert "search_text" in fields
    assert fields["text"].is_required()
    assert schema.EMBEDDING.ndims() == DIMENSION == 96


def test_chunk_vector_is_multivector_96_f16() -> None:
    """The chunk vector column is a list of 96-d float16 vectors (T9 storage
    dtype), not a 1-D vector."""
    field = schema.ChunkModel.to_arrow_schema().field("vector")
    assert field.type == pa.list_(pa.list_(pa.float16(), 96))
    assert field.type.value_type == pa.list_(pa.float16(), 96)
    assert field.type.value_type.value_type == pa.float16()


def test_f16_storage_round_trip_preserves_maxsim_ranking(tmp_path) -> None:
    """Known float32 matrices written into the float16 column read back as
    f16-upcast values whose MaxSim ORDER matches the float32 control, with
    max absolute score deviation < 0.05 (T9 eval gate tolerance).

    The toy corpus is built from an orthonormal basis so consecutive docs'
    MaxSim scores differ by a designed gap (>= 0.02) far above f16
    quantization noise: random matrices produce sub-micro near-ties that
    flip under ANY dtype change and would make the ordering claim vacuous.
    """
    import lancedb

    rng = np.random.default_rng(9)
    basis = np.linalg.qr(rng.normal(size=(96, 20)))[0].T.astype(np.float32)  # (20, 96)

    def normalize(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)

    queries = [
        normalize(
            np.stack(
                [
                    basis[2 * t] + basis[2 * t + 1] + (0.1 if k == t else 0.0) * basis[16 + k]
                    for t in range(4)
                ]
            )
        )
        for k in range(3)
    ]
    alphas = [1.0 - 0.04 * i for i in range(12)]
    docs = [
        normalize(np.stack([alpha * basis[2 * t] + basis[8 + t] for t in range(4)]))
        for alpha in alphas
    ]

    db = lancedb.connect(str(tmp_path / "lancedb"))
    tbl = db.create_table("chunks", schema=schema.ChunkModel)
    records = [
        schema.ChunkModel.model_validate(
            {
                "chunk_id": f"c{i}",
                "episode_id": "e",
                "session_id": "s",
                "project": "/project",
                "source_path": "/transcript.jsonl",
                "text": f"chunk {i}",
                "search_text": f"chunk {i}",
                "content_type": "response",
                "vector": m.tolist(),
            }
        ).model_dump()
        for i, m in enumerate(docs)
    ]
    tbl.add(records)

    stored = {row["chunk_id"]: row["vector"] for row in tbl.to_arrow().to_pylist()}
    assert len(stored) == 12
    upcast = {cid: np.asarray(v, dtype=np.float32) for cid, v in stored.items()}
    assert upcast["c0"].dtype == np.float32

    def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
        return float(np.max(query @ doc.T).sum())

    def ranking(matrix_by_id: dict[str, np.ndarray], query: np.ndarray) -> list[str]:
        scored = sorted(((maxsim(query, m), cid) for cid, m in matrix_by_id.items()), reverse=True)
        return [cid for _, cid in scored]

    control_by_id = {f"c{i}": m for i, m in enumerate(docs)}
    max_deviation = 0.0
    for query in queries:
        assert ranking(control_by_id, query)[:10] == ranking(upcast, query)[:10]
        for cid, doc in control_by_id.items():
            deviation = abs(maxsim(query, doc) - maxsim(query, upcast[cid]))
            max_deviation = max(max_deviation, deviation)

    assert max_deviation < 0.05, f"f16 MaxSim deviation {max_deviation} exceeded 0.05"


def test_proxy_vector_column_is_nullable_fixed_96() -> None:
    """v6 declares the mean-vector prefilter column (unfilled; Task 10 owns it).

    Declaring it inside the v6 bump avoids a post-v6 schema change: stale v5
    tables fail ``schema_matches()`` and hit the ``--rebuild`` gate cleanly.
    """
    fields = schema.ChunkModel.model_fields
    assert fields["proxy_vector"].default is None
    field = schema.ChunkModel.to_arrow_schema().field("proxy_vector")
    assert field.nullable
    assert field.type == pa.list_(pa.float32(), 96)
    assert field.type.value_type == pa.float32()


def test_search_text_is_plain_string_column() -> None:
    """search_text stays a plain string column: no embedding source binding."""
    field = schema.ChunkModel.to_arrow_schema().field("search_text")
    assert field.type == pa.string()
    assert schema.ChunkModel.parse_embedding_functions() == []


def test_episode_and_scalar_models_validate_values_and_defaults() -> None:
    stamp = datetime(2025, 2, 3, tzinfo=UTC)
    episode = schema.EpisodeModel(
        episode_id="e",
        session_id="s",
        project="/project",
        title="Title",
        timestamp=stamp,
        source_path="/transcript.jsonl",
    )
    assert episode.timestamp == stamp
    assert episode.source_status == "available"
    assert episode.prompt_text == episode.response_text == ""
    assert episode.files_touched == episode.tool_names == ""
    assert episode.is_subagent is False
    assert episode.source_project is None
    assert episode.agent_model is None
    assert episode.runtime == "claude"
    assert episode.claude_version is None
    assert episode.entrypoint is None
    assert episode.permission_mode is None
    assert episode.user_type is None

    session = schema.SessionModel(session_id="s", path="/transcript.jsonl")
    assert session.source_status == "available"
    assert session.runtime == "claude"
    assert session.absent_since is None
    assert schema.CursorModel(path="/p", size=1, mtime=2.5, first_line_hash="h").size == 1
    assert schema.MetadataModel(key="schema", value="2").value == "2"
    assert schema.CwdCacheModel(path="/p", size=1, mtime=2.5, cwds="[]").cwds == "[]"


def test_multivector_table_reopen_search_and_reconstruction(tmp_path, monkeypatch) -> None:
    """A multivector chunk table round-trips through disk: the embedding
    function is reconstructable from table metadata and a raw ``(T, 96)``
    query matrix searches correctly after re-open (spike: no reshaping)."""
    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry

    rng = np.random.default_rng(7)
    doc1 = rng.normal(size=(4, 96)).astype(np.float32)
    doc2 = rng.normal(size=(3, 96)).astype(np.float32)
    doc1 /= np.linalg.norm(doc1, axis=-1, keepdims=True)
    doc2 /= np.linalg.norm(doc2, axis=-1, keepdims=True)
    query = doc1.copy()  # perfect MaxSim match for doc1

    def fake_encode(texts, *, is_query, normalize_embeddings):
        return [doc1 for _ in texts]

    monkeypatch.setattr(schema, "load_embedder", lambda: SimpleNamespace(encode=fake_encode))

    db = lancedb.connect(str(tmp_path / "lancedb"))
    tbl = db.create_table(
        "chunks", schema=schema.ChunkModel, embedding_functions=[schema.EMBEDDING_CONFIG]
    )
    records = schema.ChunkModel.model_validate(
        {
            "chunk_id": "c1",
            "episode_id": "e1",
            "session_id": "s1",
            "project": "/project",
            "source_path": "/transcript.jsonl",
            "text": "How do I handle async migration failures?",
            "search_text": "async migration failures",
            "content_type": "response",
            "vector": doc1.tolist(),
        }
    ).model_dump()
    records = [
        records,
        schema.ChunkModel.model_validate(
            {
                "chunk_id": "c2",
                "episode_id": "e1",
                "session_id": "s1",
                "project": "/project",
                "source_path": "/transcript.jsonl",
                "text": "The cache invalidation strategy keeps stale data.",
                "search_text": "cache invalidation strategy",
                "content_type": "response",
                "vector": doc2.tolist(),
            }
        ).model_dump(),
    ]
    tbl.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(
        records
    )

    # Re-open from disk: a fresh connection reconstructs the table.
    db2 = lancedb.connect(str(tmp_path / "lancedb"))
    tbl2 = db2.open_table("chunks")
    assert tbl2.count_rows() == 2

    # Embedding-function reconstruction from table metadata.
    funcs = EmbeddingFunctionRegistry.get_instance().parse_functions(tbl2.schema.metadata)
    assert "vector" in funcs
    assert funcs["vector"].function.ndims() == 96
    query_matrix = funcs["vector"].function.compute_query_embeddings("some query")
    assert query_matrix.shape == (4, 96)

    # Raw query matrix searches without reshaping; MaxSim ranks doc1 first.
    results = tbl2.search(query, vector_column_name="vector").limit(2).to_list()
    assert [r["chunk_id"] for r in results] == ["c1", "c2"]


def test_inserting_wrong_dimension_vector_fails(tmp_path) -> None:
    """A 64-dim vector into the 96-dim multivector column raises a clear
    error instead of silently truncating."""
    import lancedb

    import ssgrep.store as store

    db = lancedb.connect(str(tmp_path / "lancedb"))
    tbl = db.create_table("chunks", schema=schema.ChunkModel)
    records = store.LanceStore._records(
        "chunks",
        [
            {
                "chunk_id": "c1",
                "episode_id": "e1",
                "session_id": "s1",
                "project": "/project",
                "source_path": "/transcript.jsonl",
                "text": "x",
                "search_text": "x",
                "content_type": "response",
                "vector": [[1.0] * 64],
            }
        ],
    )
    with pytest.raises(Exception, match="expected size"):
        tbl.merge_insert(
            "chunk_id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(records)

    with pytest.raises(Exception, match="96"):
        schema.ChunkModel(
            chunk_id="c1",
            episode_id="e1",
            session_id="s1",
            project="/project",
            source_path="/transcript.jsonl",
            text="x",
            search_text="x",
            content_type="response",
            vector=[[1.0] * 64],
        )
