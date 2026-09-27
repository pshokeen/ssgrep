"""Source descriptors and the persistent registry that drives the LiveMap.

Option A's source model: every source key ever indexed lives in the immutable
``sources`` registry table, so a source that disappears from discovery keeps a
byte-identical frozen descriptor forever (until ``ssgrep prune`` removes it via
``delete_registry``). The engine usually memo-hits on that descriptor, its rows
are retained, and the post-step marks them tombstoned. A change to the
pipeline code itself invalidates every memo, including theirs; when that
forces a deleted source to re-run, ``pipeline/archive.py`` re-declares its
rows from a pre-run snapshot instead of reading the gone file, so the memo
miss still does not delete them (a validation failure there fails the whole
run closed rather than reconciling to empty). OpenCode sessions carry their
own per-session fingerprints (message/part counts and updated timestamps,
computed by the OpenCode adapter), so ``ssgrep index`` re-embeds only sessions
that actually changed; an unchanged session memo-hits like any other source.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ssgrep.sessions.adapters.base import SourceFingerprint, TranscriptSource
from ssgrep.store import SOURCES_TABLE, LanceStore, quote as _quote
from ssgrep.utilities.types import SessionFile

_PROJECT_SEPARATOR = "\n"


@dataclasses.dataclass(frozen=True)
class SourceDescriptor:
    """The stable, all-strings value the pipeline memoizes per source key.

    Mirrors the ``sources`` registry columns exactly so descriptors can round
    trip through the registry without loss. Plain strings only: CocoIndex
    hashes argument values for memoization, and Path/datetime objects have no
    business in that key space.
    """

    adapter: str
    key: str
    path: str
    size: int
    mtime: float
    digest: str
    cache_cwds: bool = False
    session_id: str = ""
    is_main: bool = True
    parent_session_id: str | None = None
    agent_type: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_model: str | None = None
    project_paths: tuple[str, ...] = ()
    source_project: str | None = None
    claude_version: str | None = None
    entrypoint: str | None = None
    permission_mode: str | None = None
    user_type: str | None = None
    runtime: str = "claude"


def to_descriptor(source: TranscriptSource) -> SourceDescriptor:
    """Freeze one freshly discovered source into descriptor form."""
    session = source.session
    return SourceDescriptor(
        adapter=source.adapter,
        key=source.key,
        path=str(session.path),
        size=source.fingerprint.size,
        mtime=source.fingerprint.mtime,
        digest=source.fingerprint.digest,
        cache_cwds=source.cache_cwds,
        session_id=session.session_id,
        is_main=session.is_main,
        parent_session_id=session.parent_session_id,
        agent_type=session.agent_type,
        agent_name=session.agent_name,
        agent_description=session.agent_description,
        agent_model=session.agent_model,
        project_paths=session.project_paths,
        source_project=session.source_project,
        claude_version=session.claude_version,
        entrypoint=session.entrypoint,
        permission_mode=session.permission_mode,
        user_type=session.user_type,
        runtime=session.runtime,
    )


def to_transcript_source(descriptor: SourceDescriptor) -> TranscriptSource:
    """Rebuild the adapter contract a component needs to read one source."""
    session = SessionFile(
        path=Path(descriptor.path),
        session_id=descriptor.session_id,
        is_main=descriptor.is_main,
        parent_session_id=descriptor.parent_session_id,
        agent_type=descriptor.agent_type,
        agent_name=descriptor.agent_name,
        agent_description=descriptor.agent_description,
        agent_model=descriptor.agent_model,
        project_paths=descriptor.project_paths,
        source_project=descriptor.source_project,
        claude_version=descriptor.claude_version,
        entrypoint=descriptor.entrypoint,
        permission_mode=descriptor.permission_mode,
        user_type=descriptor.user_type,
        runtime=descriptor.runtime,
    )
    return TranscriptSource(
        adapter=descriptor.adapter,
        key=descriptor.key,
        session=session,
        fingerprint=SourceFingerprint(
            size=descriptor.size,
            mtime=descriptor.mtime,
            digest=descriptor.digest,
        ),
        cache_cwds=descriptor.cache_cwds,
    )


def to_row(descriptor: SourceDescriptor) -> dict:
    """One ``sources`` registry row for a descriptor."""
    return {
        "key": descriptor.key,
        "adapter": descriptor.adapter,
        "path": descriptor.path,
        "size": descriptor.size,
        "mtime": descriptor.mtime,
        "first_line_hash": descriptor.digest,
        "cache_cwds": descriptor.cache_cwds,
        "session_id": descriptor.session_id,
        "is_main": descriptor.is_main,
        "parent_session_id": descriptor.parent_session_id,
        "agent_type": descriptor.agent_type,
        "agent_name": descriptor.agent_name,
        "agent_description": descriptor.agent_description,
        "agent_model": descriptor.agent_model,
        "project_paths": _PROJECT_SEPARATOR.join(descriptor.project_paths),
        "source_project": descriptor.source_project,
        "claude_version": descriptor.claude_version,
        "entrypoint": descriptor.entrypoint,
        "permission_mode": descriptor.permission_mode,
        "user_type": descriptor.user_type,
        "runtime": descriptor.runtime,
    }


def from_row(row: dict) -> SourceDescriptor:
    """Rebuild a frozen descriptor from one registry row (round trip)."""
    paths = str(row.get("project_paths") or "")
    return SourceDescriptor(
        adapter=str(row["adapter"]),
        key=str(row["key"]),
        path=str(row["path"]),
        size=int(row["size"]),
        mtime=float(row["mtime"]),
        digest=str(row["first_line_hash"]),
        cache_cwds=bool(row.get("cache_cwds", False)),
        session_id=str(row["session_id"]),
        is_main=bool(row.get("is_main", True)),
        parent_session_id=row.get("parent_session_id"),
        agent_type=row.get("agent_type"),
        agent_name=row.get("agent_name"),
        agent_description=row.get("agent_description"),
        agent_model=row.get("agent_model"),
        project_paths=tuple(p for p in paths.split(_PROJECT_SEPARATOR) if p),
        source_project=row.get("source_project"),
        claude_version=row.get("claude_version"),
        entrypoint=row.get("entrypoint"),
        permission_mode=row.get("permission_mode"),
        user_type=row.get("user_type"),
        runtime=str(row.get("runtime") or "claude"),
    )


def union_descriptors(
    fresh: list[TranscriptSource], registry: dict[str, SourceDescriptor]
) -> dict[str, SourceDescriptor]:
    """The LiveMap entry set: fresh discovery wins, registry fills the rest.

    Registry-only keys (disappeared sources) keep their frozen descriptors so
    the engine memo-hits and retains their rows; fresh keys that change the
    registry are refreshed in the same value space.
    """
    merged = dict(registry)
    for source in fresh:
        descriptor = to_descriptor(source)
        merged[descriptor.key] = descriptor
    return merged


def read_registry(repository: LanceStore) -> dict[str, SourceDescriptor]:
    """Every persisted source snapshot, keyed by source key."""
    rows = repository.rows(SOURCES_TABLE, limit=1_000_000)
    return {str(row["key"]): from_row(row) for row in rows}


def write_registry(repository: LanceStore, descriptors: list[SourceDescriptor]) -> None:
    """Persist the fresh side of the registry after a successful run.

    Absent keys are intentionally not rewritten: their frozen rows stay until
    ``ssgrep prune`` removes them.
    """
    if not descriptors:
        return
    repository.upsert(SOURCES_TABLE, [to_row(descriptor) for descriptor in descriptors])


def delete_registry(repository: LanceStore, keys: list[str]) -> None:
    """Remove registry rows for pruned sources (``ssgrep prune``)."""
    for key in keys:
        repository.delete(SOURCES_TABLE, f"key = {_quote(key)}")


__all__ = [
    "SourceDescriptor",
    "delete_registry",
    "from_row",
    "read_registry",
    "to_descriptor",
    "to_row",
    "to_transcript_source",
    "union_descriptors",
    "write_registry",
]
