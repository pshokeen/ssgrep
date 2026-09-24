"""Declared-row Pydantic models and builders for the CocoIndex pipeline.

The models mirror ``ssgrep.store.schema``'s LanceModels (themselves Pydantic)
field for field, so a CocoIndex USER-managed write into ssgrep-created tables
is type-safe end to end and every non-nullable column is covered. The vector
field is a per-token multivector ``(num_tokens, DIMENSION)`` matrix declared
via an explicit ``LanceType`` (``VECTOR_LANCE``) instead of a provider-style
annotation: CocoIndex's ``VectorSchemaProvider`` path only supports a single
1-D vector, so the explicit nested-list column spec is what lets the whole
engine ingest/persist/index/search multivectors. Timestamps carry an explicit
``LanceType`` too: by default CocoIndex maps ``datetime`` to a JSON string
column, which would conflict with the real ``timestamp[us]`` columns in the
existing tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from cocoindex.connectors.lancedb import LanceType
from pydantic import BaseModel, ConfigDict

from ssgrep.indexing import chunker
from ssgrep.indexing.embed import DIMENSION, unit_mean_vector
from ssgrep.utilities.types import Chunk, Episode, SessionFile

CONTEXT_TITLE_CHARS = 160

#: Hard token budget for the embedded ``search_text``.  The default model's
#: window is 299 model tokens, but the embedder's own tokenizer can count a
#: few tokens more than ``chunker._tokenizer()`` for wide/box-drawing
#: characters, so the cap sits below 299 to guarantee the model never sees an
#: over-length sequence.
SEARCH_MAX_TOKENS = 290

#: Combined token budget reserved for the title/project context prefix.  The
#: context is secondary to the chunk content, so the title and (when present)
#: the project identity share one fixed slice and the chunk keeps the rest.
#: The title alone gets the whole slice; adding project context splits it
#: (title keeps the remainder after the project's slice) so the chunk content
#: budget is essentially unchanged.
SEARCH_TITLE_TOKENS = 40

#: Fixed token slice of ``SEARCH_TITLE_TOKENS`` reserved for the project
#: identity when it is known.  The project name is the one token that makes
#: project-scoped queries retrievable; it must be smaller than
#: ``SEARCH_TITLE_TOKENS`` so the title always keeps a guaranteed slice.
SEARCH_PROJECT_TOKENS = 16


def _epoch_micros(value: datetime) -> int:
    """Unsigned-instant microseconds so the value survives engine serialization."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return int(value.timestamp() * 1_000_000)
    return value  # pragma: no cover - encoder is only applied to datetime values


#: Explicit column spec for timestamp fields. Passed via ``column_specs`` to
#: ``TableSchema.from_class``: cocoindex's default ``datetime`` mapping is a
#: JSON-string column (and its annotation-metadata unwrapping drops ``Annotated``
#: extras on Python 3.13), which would collide with the real ``timestamp[us]``
#: columns in the ssgrep-created tables. The epoch encoder keeps values
#: msgspec-safe through engine serialization.
TIMESTAMP_SPEC = LanceType(pa.timestamp("us"), encoder=_epoch_micros)

#: Multivector column spec: a per-token ``(num_tokens, DIMENSION)`` matrix
#: stored as a nested arrow list of float16 (halves raw column bytes; pylate
#: computes float32 and the encoder casts at this write boundary). The explicit
#: ``LanceType`` (with an encoder that tolist's the numpy matrix) is what lets
#: the whole CocoIndex engine ingest, persist, index, and search multivector
#: columns — the built-in ``VectorSchemaProvider`` path only emits a single
#: 1-D ``fixed_size_list``.
MV_TYPE = pa.list_(pa.list_(pa.float16(), DIMENSION))
VECTOR_LANCE = LanceType(MV_TYPE, encoder=lambda v: np.asarray(v, np.float16).tolist())

#: Single-vector prefilter column spec: the chunk's L2-normalized mean token
#: vector, matched at two-stage-query time against the query's own mean
#: vector. The arrow type mirrors the nullable ``fixed_size_list<float>[96]``
#: column declared on ``ChunkModel.proxy_vector`` in v6 — float32, because
#: existing v6 databases already carry that declared type and changing it
#: would break writes without a schema-version gate.
PROXY_TYPE = pa.list_(pa.float32(), DIMENSION)


def _proxy_encoder(value: object) -> object:
    """ndarray -> plain float list; None passes through as a null cell."""
    if value is None:
        return None
    return np.asarray(value, np.float32).tolist()


PROXY_LANCE = LanceType(PROXY_TYPE, encoder=_proxy_encoder)


class _RowModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChunkRow(_RowModel):
    """One searchable chunk with denormalized episode metadata."""

    chunk_id: str
    episode_id: str
    session_id: str
    project: str
    source_path: str
    vector: Annotated[npt.NDArray[np.float32], VECTOR_LANCE]
    proxy_vector: npt.NDArray[np.float32] | None = None
    source_project: str | None = None
    source_status: str = "available"
    runtime: str = "claude"
    text: str = ""
    search_text: str = ""
    content_type: str = "response"
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


class EpisodeRow(_RowModel):
    """Complete prompt/response plus the metadata ``show`` needs."""

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
    source_path: str = ""
    source_project: str | None = None
    source_status: str = "available"
    runtime: str = "claude"
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None


class SessionRow(_RowModel):
    """Source identity and availability status."""

    session_id: str
    path: str
    runtime: str = "claude"
    source_status: str = "available"
    absent_since: datetime | None = None


def project_path(episode: Episode | None, session: SessionFile) -> str:
    """The canonical absolute project directory for an episode."""
    candidates = ([episode.cwd] if episode and episode.cwd else []) + list(session.project_paths)
    return next(
        (str(Path(value).expanduser().absolute()) for value in candidates if value),
        "",
    )


def _search_metadata(episode: Episode, session: SessionFile) -> dict:
    """The shared metadata dict that every row type denormalizes."""
    project = episode.project or project_path(episode, session)
    return {
        "episode_id": episode.episode_id,
        "session_id": episode.session_id,
        "project": project,
        "title": episode.title,
        "timestamp": episode.timestamp,
        "git_branch": episode.git_branch,
        "cwd": episode.cwd,
        "files_touched": "\n".join(episode.files_touched),
        "tool_names": "\n".join(episode.tool_names),
        "is_subagent": episode.is_subagent,
        "parent_session_id": episode.parent_session_id,
        "agent_type": episode.agent_type,
        "agent_name": episode.agent_name,
        "agent_description": episode.agent_description,
        "agent_model": episode.agent_model,
        "source_path": episode.source_path or str(session.path.absolute()),
        "source_project": episode.source_project,
        "source_status": "available",
        "runtime": episode.runtime,
        "claude_version": episode.claude_version,
        "entrypoint": episode.entrypoint,
        "permission_mode": episode.permission_mode,
        "user_type": episode.user_type,
    }


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate ``text`` so it re-encodes to at most ``max_tokens`` tokens."""
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    # ``decode`` of a token slice can re-encode to a slightly larger count
    # (e.g. merged subword boundaries), so step the prefix window down until the
    # decoded text re-encodes to at most ``max_tokens``.  The window only ever
    # shrinks, so this is guaranteed to terminate within ``max_tokens``
    # iterations (the empty prefix decodes to zero tokens), unlike a re-clamping
    # loop that could oscillate forever at the token boundary.
    hi = max_tokens
    while hi > 0:
        text = tokenizer.decode(ids[:hi], skip_special_tokens=True)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
            return text
        hi -= 1
    return ""


def _project_tail(project: str) -> str:
    """The last two path components of ``project``, enough to identify it."""
    return "/".join(project.rstrip("/").split("/")[-2:])


def _contextual_search_text(part: Chunk, episode: Episode, project: str = "") -> str:
    """Add short context for embedding/FTS without polluting excerpts.

    The result is the string the embedder receives, so it must never exceed
    the model's 299-token window even when the title or chunk is full of
    wide/box-drawing characters.  The title and project identity share one
    context budget (``SEARCH_TITLE_TOKENS``): the project (when known) gets
    its fixed slice, the title the remainder, and the chunk gets the rest of
    ``SEARCH_MAX_TOKENS``; the final string is hard-clamped so the combined
    sequence is deterministically capped.
    """
    tokenizer = chunker._tokenizer()
    if project.strip():
        project_tail = _truncate_to_tokens(tokenizer, _project_tail(project), SEARCH_PROJECT_TOKENS)
        title = _truncate_to_tokens(
            tokenizer, episode.title.strip(), SEARCH_TITLE_TOKENS - SEARCH_PROJECT_TOKENS
        )
        prefix = f"Project: {project_tail}\nTitle: {title}\nContent: "
    else:
        title = _truncate_to_tokens(tokenizer, episode.title.strip(), SEARCH_TITLE_TOKENS)
        prefix = f"Title: {title}\nContent: "
    content_budget = SEARCH_MAX_TOKENS - len(tokenizer.encode(prefix, add_special_tokens=False))
    content = _truncate_to_tokens(tokenizer, part.text, content_budget)
    return _truncate_to_tokens(tokenizer, prefix + content, SEARCH_MAX_TOKENS)


def chunk_payloads(episode: Episode, session: SessionFile) -> list[tuple[Chunk, str]]:
    """Chunk one episode, pairing every chunk with its embedding text."""
    project = _search_metadata(episode, session)["project"]
    return [
        (part, _contextual_search_text(part, episode, project=project))
        for part in chunker.chunk_episode(episode)
    ]


def build_session_row(
    *,
    session_id: str,
    path: str,
    runtime: str,
    source_status: str = "available",
) -> SessionRow:
    return SessionRow(
        session_id=session_id,
        path=path,
        runtime=runtime,
        source_status=source_status,
    )


def build_episode_row(episode: Episode, session: SessionFile) -> EpisodeRow:
    """One episode row with the complete transcript texts."""
    metadata = _search_metadata(episode, session)
    return EpisodeRow(
        episode_id=episode.episode_id,
        session_id=episode.session_id,
        project=metadata["project"],
        title=metadata["title"],
        timestamp=metadata["timestamp"],
        git_branch=metadata["git_branch"],
        cwd=metadata["cwd"],
        files_touched=metadata["files_touched"],
        tool_names=metadata["tool_names"],
        prompt_text=episode.prompt_text,
        response_text=episode.response_text,
        is_subagent=episode.is_subagent,
        parent_session_id=episode.parent_session_id,
        agent_type=episode.agent_type,
        agent_name=episode.agent_name,
        agent_description=episode.agent_description,
        agent_model=episode.agent_model,
        source_path=metadata["source_path"],
        source_project=episode.source_project,
        source_status="available",
        runtime=episode.runtime,
        claude_version=episode.claude_version,
        entrypoint=episode.entrypoint,
        permission_mode=episode.permission_mode,
        user_type=episode.user_type,
    )


def build_chunk_row(
    part: Chunk,
    search_text: str,
    vector: np.ndarray,
    episode: Episode,
    session: SessionFile,
) -> ChunkRow:
    """One chunk row: denormalized metadata plus the precomputed vector."""
    metadata = _search_metadata(episode, session)
    return ChunkRow(
        chunk_id=part.chunk_id,
        episode_id=episode.episode_id,
        session_id=episode.session_id,
        project=metadata["project"],
        source_path=metadata["source_path"],
        vector=np.asarray(vector, dtype=np.float32),
        proxy_vector=unit_mean_vector(vector),
        source_project=episode.source_project,
        source_status="available",
        runtime=episode.runtime,
        text=part.text,
        search_text=search_text,
        content_type=part.content_type.value,
        title=metadata["title"],
        timestamp=metadata["timestamp"],
        git_branch=metadata["git_branch"],
        cwd=metadata["cwd"],
        files_touched=metadata["files_touched"],
        tool_names=metadata["tool_names"],
        is_subagent=episode.is_subagent,
        parent_session_id=episode.parent_session_id,
        agent_type=episode.agent_type,
        agent_name=episode.agent_name,
        agent_description=episode.agent_description,
        agent_model=episode.agent_model,
        claude_version=episode.claude_version,
        entrypoint=episode.entrypoint,
        permission_mode=episode.permission_mode,
        user_type=episode.user_type,
    )


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


__all__ = [
    "CONTEXT_TITLE_CHARS",
    "SEARCH_MAX_TOKENS",
    "SEARCH_TITLE_TOKENS",
    "SEARCH_PROJECT_TOKENS",
    "MV_TYPE",
    "VECTOR_LANCE",
    "PROXY_LANCE",
    "ChunkRow",
    "EpisodeRow",
    "SessionRow",
    "build_chunk_row",
    "build_episode_row",
    "build_session_row",
    "chunk_payloads",
    "directory_size",
    "project_path",
]
