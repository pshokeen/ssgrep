"""Redeclare a missing source's retained rows without rewriting history."""

from __future__ import annotations

import numpy as np

from ssgrep.indexing.chunker import chunk_episode
from ssgrep.indexing.embed import DIMENSION
from ssgrep.pipeline import rows
from ssgrep.pipeline.sources import SourceDescriptor, to_transcript_source
from ssgrep.sessions.adapters.base import TranscriptSource
from ssgrep.store import (
    CHUNKS_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    SOURCES_TABLE,
    LanceStore,
    quote,
)
from ssgrep.utilities.types import Episode


def capture_archives(entries: dict[str, SourceDescriptor]) -> dict:
    """Materialize absent-source declarations before reconciliation can mutate rows.

    One connection is shared across every missing source: a real corpus can
    carry thousands of tombstoned sources, and this scan runs on every
    reconcile (most of them will memo-hit and never touch the snapshot), so
    opening a fresh LanceDB connection per source would make every ``ssgrep
    note`` / ``index`` call slower as the archive grows.
    """
    missing = {}
    for key, descriptor in entries.items():
        source = to_transcript_source(descriptor)
        if not source.session.path.exists():
            missing[key] = source
    if not missing:
        return {}
    repo = LanceStore()
    try:
        return {key: retained_rows(source, repo=repo) for key, source in missing.items()}
    finally:
        repo.close()


def retained_rows(source: TranscriptSource, *, repo: LanceStore | None = None):
    """Preserve the indexed snapshot, not claim completeness of a lost file.

    An empty source legitimately owns only a session row. Historical runtimes
    also reused session IDs across paths. Validate their combined snapshot, then
    declare only rows attributed to THIS source, avoiding stale alias writes.
    Extra historical chunks are retained only for registered shared-ID sources;
    current episode chunks must still exist and match, and links/vectors must be
    valid. Nothing is regenerated, and unknown source identities fail closed.

    ``repo`` lets ``capture_archives`` share one connection across a whole
    batch; callers that pass none (standalone use, tests) get one opened and
    closed here as before.
    """
    owns_repo = repo is None
    repo = repo or LanceStore()
    predicate = f"session_id = {quote(source.session.session_id)}"
    try:
        stored = []
        for table in (SESSIONS_TABLE, EPISODES_TABLE, CHUNKS_TABLE, SOURCES_TABLE):
            count = repo.count(table, predicate)
            items = repo.rows(table, where=predicate, limit=count) if count else []
            if len(items) != count:
                raise ValueError("truncated rows")
            stored.append(items)
        sessions, episodes, chunks, registry = stored
        aliases = {item["key"]: item["path"] for item in registry}
        if (
            len(sessions) != 1
            or sessions[0]["path"] not in aliases
            or aliases.get(source.key) != str(source.session.path)
        ):
            raise ValueError("session identity mismatch")
        episode_ids = {row["episode_id"] for row in episodes}
        expected_ids = {f"{source.session.session_id}:ep:{i}" for i in range(len(episodes))}
        if episode_ids != expected_ids:
            raise ValueError("episode sequence is incomplete")
        expected_chunks = {}
        for row in episodes:
            episode = Episode(
                **{
                    key: row[key]
                    for key in ("episode_id", "session_id", "prompt_text", "response_text", "title")
                }
            )
            expected_chunks.update({part.chunk_id: part for part in chunk_episode(episode)})
        actual_ids = {row["chunk_id"] for row in chunks}
        if not set(expected_chunks) <= actual_ids:
            raise ValueError("missing chunks for retained episode text")
        if len(aliases) == 1 and actual_ids != set(expected_chunks):
            raise ValueError("unexpected chunks without registered historical sources")
        for row in episodes + chunks:
            if row["source_path"] not in set(aliases.values()):
                raise ValueError("source identity mismatch")
        for row in chunks:
            part = expected_chunks.get(row["chunk_id"])
            if row["episode_id"] not in episode_ids or row["content_type"] not in (
                "prompt",
                "response",
            ):
                raise ValueError("chunk content or episode link mismatch")
            if part is not None and (
                row["text"] != part.text or row["content_type"] != part.content_type.value
            ):
                raise ValueError("chunk content or episode link mismatch")
            vector = np.asarray(row["vector"], dtype=np.float32)
            if (
                vector.ndim != 2
                or vector.shape[0] == 0
                or vector.shape[1] != DIMENSION
                or not np.isfinite(vector).all()
            ):
                raise ValueError("invalid retained vector")
            row["vector"] = vector
            if row["proxy_vector"] is not None:
                proxy = np.asarray(row["proxy_vector"], dtype=np.float32)
                if proxy.shape != (DIMENSION,) or not np.isfinite(proxy).all():
                    raise ValueError("invalid retained proxy vector")
                row["proxy_vector"] = proxy
        owned = [
            [row for row in sessions if row["path"] == source.key],
            [row for row in episodes if row["source_path"] == str(source.session.path.absolute())],
            [row for row in chunks if row["source_path"] == str(source.session.path.absolute())],
        ]
        result = []
        for model, items in zip(
            (rows.SessionRow, rows.EpisodeRow, rows.ChunkRow), owned, strict=True
        ):
            if any(set(item) != set(model.model_fields) for item in items):
                raise ValueError("stored row schema mismatch")
            result.append([model.model_validate(item) for item in items])
        return result
    except (KeyError, TypeError, ValueError) as error:
        # Actionable, not just diagnostic: this fails the WHOLE run closed (see
        # module docstring), so the source key/path and a real recovery command
        # must be in the message itself -- `note` and MCP-startup print this
        # text verbatim with no other hint (unlike `ssgrep index`'s CLI wrapper).
        # `ssgrep prune` cannot target this source: an archive fails validation
        # either before it was ever tombstoned (nothing for `prune` to select)
        # or, once it is a registry-only remnant, after `prune` already deleted
        # its session row (also nothing for `prune` to select). Only a rebuild
        # re-derives the registry from scratch, dropping the bad entry with it.
        raise ValueError(
            f"Cannot preserve source {source.key!r} (recorded path: {source.session.path}): "
            f"incomplete archive ({error}). Run `ssgrep index --rebuild` to recover (add "
            "--allow-shrink if it reports the rebuild would shrink the corpus); this "
            "discards every tombstoned source's history, not only this one."
        ) from error
    finally:
        if owns_repo:
            repo.close()
