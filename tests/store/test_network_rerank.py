"""Network smoke tests for the real-model embedding path.

Validates the built-in LanceDB integration ssgrep now relies on — the
``sentence-transformers`` embedding function — against a small public Hugging
Face model instead of the pinned ``sentence-transformers/all-mpnet-base-v2``.
These tests download ~150 MB on first run; they are excluded from the default
``pytest`` run (``not network``) and executed by the scheduled network-tests
workflow.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.network

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _make_index(tmp_path_factory: pytest.TempPathFactory):
    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from lancedb.pydantic import LanceModel, Vector

    registry = EmbeddingFunctionRegistry.get_instance()
    embedder = registry.get("sentence-transformers").create(name=EMBED_MODEL, device="cpu")

    class Chunk(LanceModel):
        chunk_id: str
        search_text: str = embedder.SourceField()
        vector: Vector(384) = embedder.VectorField()  # type: ignore[valid-type]  # ty: ignore[invalid-type-form]
        text: str

    db = lancedb.connect(str(tmp_path_factory.mktemp("lancedb")))
    table = db.create_table("chunks", schema=Chunk)
    table.add(
        [
            {
                "chunk_id": "c1",
                "search_text": "the quick brown fox jumps over the lazy dog",
                "text": "the quick brown fox",
            },
            {
                "chunk_id": "c2",
                "search_text": "quantum entanglement is a strange phenomenon",
                "text": "quantum entanglement",
            },
            {
                "chunk_id": "c3",
                "search_text": "machine learning models need vectors to search",
                "text": "machine learning",
            },
            {
                "chunk_id": "c4",
                "search_text": "the dog and the fox are animals",
                "text": "the dog and the fox",
            },
            {
                "chunk_id": "c5",
                "search_text": "fast green turtles swim in the warm ocean",
                "text": "turtles and ocean",
            },
        ]
    )
    return table


def test_embedding_function_reconstruction_from_table_metadata(tmp_path_factory) -> None:
    """Query-time metadata reconstruction resolves the built-in function."""
    from lancedb.embeddings import EmbeddingFunctionRegistry

    table = _make_index(tmp_path_factory)
    metadata = table.schema.metadata
    parsed = EmbeddingFunctionRegistry.get_instance().parse_functions(metadata)
    assert "vector" in parsed
    vector = parsed["vector"].function.compute_query_embeddings("fox dog")[0]
    assert len(vector) == 384
