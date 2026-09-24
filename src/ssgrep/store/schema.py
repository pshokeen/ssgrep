"""Lance models for the global ssgrep database."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pyarrow as pa
from lancedb.embeddings import EmbeddingFunctionConfig, EmbeddingFunctionRegistry
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.pydantic import LanceModel, MultiVector, Vector

from ssgrep.indexing.embed import DIMENSION, load_embedder

# v6 (2026-08-22): document-side token pooling (SSGREP_POOL_FACTOR) changed the
# stored token-vector population, and chunks gained the nullable proxy_vector
# mean-vector column. No in-place migration: stale databases hit the
# "--rebuild" gate.
SCHEMA_VERSION = 6

_registry = EmbeddingFunctionRegistry.get_instance()


@_registry.register("ssgrep")
class SsgrepEmbedding(TextEmbeddingFunction):
    """Lance adapter: per-token late-interaction embeddings from the pinned
    PyLate ColBERT model (``ssgrep.indexing.embed.MODEL_ID``), loaded lazily
    through ``ssgrep.indexing.embed.load_embedder`` so the model is cached once
    per process. ``generate_embeddings`` returns one ``(num_tokens,
    DIMENSION)`` matrix per text (``is_query=False``);
    ``compute_query_embeddings`` returns the query's ``(num_tokens,
    DIMENSION)`` matrix (``is_query=True``). The vector dimension comes from
    the pinned model constant so creating the table never loads the model."""

    def ndims(self) -> int:
        return DIMENSION

    def generate_embeddings(self, texts: Any, *_args: Any, **_kwargs: Any) -> list[Any | None]:
        embedder = load_embedder()
        return [
            np.asarray(
                embedder.encode([text], is_query=False, normalize_embeddings=True)[0],
                dtype=np.float32,
            )
            for text in self.sanitize_input(texts)
        ]

    def compute_query_embeddings(self, query: Any, *_args: Any, **_kwargs: Any) -> Any:
        embedder = load_embedder()
        return np.asarray(
            embedder.encode([query], is_query=True, normalize_embeddings=True)[0],
            dtype=np.float32,
        )


EMBEDDING = SsgrepEmbedding.create()

#: The embedding-function config attached to the chunks table at creation
#: time. ``search_text`` is a plain string column (no ``SourceField`` binding
#: on the model), so the metadata is attached explicitly here: LanceDB stores
#: it in the table schema and reconstructs ``SsgrepEmbedding`` on every query.
EMBEDDING_CONFIG = EmbeddingFunctionConfig(
    source_column="search_text",
    vector_column="vector",
    function=EMBEDDING,
)


class ChunkModel(LanceModel):
    """Search text with every scalar needed for a pre-ranking filter."""

    chunk_id: str
    episode_id: str
    session_id: str
    project: str
    source_path: str
    source_project: str | None = None
    source_status: str = "available"
    runtime: str = "claude"
    text: str
    search_text: str
    #: float16 storage halves raw column bytes; pylate computes float32 and
    #: the pipeline write boundary casts. Deliberately still v6: rides the
    #: same forced-rebuild contract, no version bump.
    vector: MultiVector(DIMENSION, value_type=pa.float16()) = EMBEDDING.VectorField()  # type: ignore[valid-type]  # ty: ignore[invalid-type-form]
    #: Mean-vector prefilter column: L2-normalized mean of the chunk's pooled
    #: token vectors, filled at encode time by the pipeline write boundary.
    #: Stays float32 — v6 databases already carry this declared f32 column,
    #: and changing its type would break writes without a version gate. The
    #: IVF-PQ index on it is built alongside the ``vector`` index; queries
    #: against it are gated behind ``SSGREP_TWO_STAGE`` (default off).
    proxy_vector: Vector(DIMENSION) | None = None  # ty: ignore[invalid-type-form]
    content_type: str
    title: str = ""
    timestamp: datetime | None = None
    git_branch: str | None = None
    cwd: str | None = None
    files_touched: str = ""
    tool_names: str = ""
    is_subagent: bool = False
    parent_session_id: str | None = None
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_model: str | None = None
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None


class EpisodeModel(LanceModel):
    episode_id: str
    session_id: str
    project: str
    title: str
    timestamp: datetime | None = None
    git_branch: str | None = None
    cwd: str | None = None
    files_touched: str = ""
    tool_names: str = ""
    prompt_text: str = ""
    response_text: str = ""
    is_subagent: bool = False
    parent_session_id: str | None = None
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_model: str | None = None
    source_path: str
    source_project: str | None = None
    source_status: str = "available"
    runtime: str = "claude"
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None


class SessionModel(LanceModel):
    session_id: str
    path: str
    runtime: str = "claude"
    source_status: str = "available"
    absent_since: datetime | None = None


class CursorModel(LanceModel):
    path: str
    size: int
    mtime: float
    first_line_hash: str


class MetadataModel(LanceModel):
    key: str
    value: str


class CwdCacheModel(LanceModel):
    path: str
    size: int
    mtime: float
    cwds: str


class SourceModel(LanceModel):
    """Persistent discovery snapshot for one transcript source.

    The registry that Option A's pipeline rebuilds its LiveMap from: every
    source key ever indexed, with the full frozen discovery snapshot needed
    to reconstruct byte-identical ``TranscriptSource`` descriptors for
    sources that have since disappeared (so the engine memo-hits and keeps
    their tombstoned rows). Rows are removed only by ``ssgrep prune``.
    """

    key: str
    adapter: str
    path: str
    size: int
    mtime: float
    first_line_hash: str
    cache_cwds: bool = False
    session_id: str
    is_main: bool = True
    parent_session_id: str | None = None
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_model: str | None = None
    project_paths: str = ""
    source_project: str | None = None
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None
    runtime: str = "claude"


TABLE_SCHEMAS: dict[str, type[LanceModel]] = {
    "chunks": ChunkModel,
    "episodes": EpisodeModel,
    "sessions": SessionModel,
    "cursors": CursorModel,
    "metadata": MetadataModel,
    "cwd_cache": CwdCacheModel,
    "sources": SourceModel,
}

#: Data tables whose exact columns define index compatibility for searches.
#: The ``sources`` registry is internal to the indexer and excluded: adding it
#: must not force a rebuild of pre-existing indexes.
COMPAT_TABLES = ("chunks", "episodes", "sessions")

PRIMARY_KEYS = {
    "chunks": "chunk_id",
    "episodes": "episode_id",
    "sessions": "session_id",
    "cursors": "path",
    "metadata": "key",
    "cwd_cache": "path",
    "sources": "key",
}
