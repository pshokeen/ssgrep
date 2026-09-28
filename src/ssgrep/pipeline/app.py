"""Build and drive the single CocoIndex App that owns transcript ingestion.

Architecture (Option A, approved):

- The ``sources`` Lance table is the persistent registry. A run merges fresh
  discovery with every registry key; disappeared sources keep byte-identical
  frozen descriptors, so the engine usually memo-hits and retains their rows,
  and the post-step tombstones them (``source_status='absent'``). A pipeline
  code change invalidates every memo, including theirs; when that forces a
  deleted source to re-run, ``_run_app`` freezes ``capture_archives``'s
  snapshot of its rows *before* the engine starts, so ``process_source`` can
  redeclare them from the archive instead of reading a file that is gone.
- All three data tables are USER-managed: ssgrep still owns their lifecycle
  and schema (``LanceStore``), CocoIndex only declares rows. Writing through
  ``LanceStore`` also guarantees ``ssgrep.store.schema`` is imported, which
  registers the ``ssgrep`` embedding function Lance's hybrid search needs at
  query time.
- Catch-up mode is the default (scan, sync delta, exit). ``live=True`` runs a
  foreground polling loop: no daemon is ever installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import warnings
from collections.abc import Collection
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectorkits.target import ManagedBy
from cocoindex.connectors import lancedb
from cocoindex.connectors.lancedb import LanceType
from cocoindex.resources.live_map import LiveMap
from cocoindex.resources.schema import VectorSchemaProvider
from usecli import Spinner

from ssgrep.indexing.embed import (
    MODEL_ID,
    MODEL_REVISION,
    configure_model_loading,
    ensure_model_downloaded,
    raise_model_error,
)
from ssgrep.indexing.lateon import ColBERTEmbedder
from ssgrep.pipeline import rows as rows_mod
from ssgrep.pipeline.archive import capture_archives
from ssgrep.pipeline.components import process_source, produce_entries
from ssgrep.pipeline.diagnostics import current as diagnostics
from ssgrep.pipeline.sources import (
    SourceDescriptor,
    read_registry,
    to_descriptor,
    union_descriptors,
    write_registry,
)
from ssgrep.pipeline.state import APP_NAME, ARCHIVED_ROWS, EMBEDDER, LANCE_DB, lmdb_path
from ssgrep.pipeline.update import (
    _drive_update as _drive_update,
    _source_progress as _source_progress,
)
from ssgrep.search.lexical import corpus_stats, dump_stats
from ssgrep.sessions import adapters as transcript_adapters
from ssgrep.store import (
    CHUNKS_TABLE,
    CWD_CACHE_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    LanceStore,
    ensure_data_dir,
    quote,
)
from ssgrep.utilities.limits import raise_nofile_limit
from ssgrep.utilities.types import (
    IndexNotReadyError,
    IndexStats,
    RebuildWouldShrinkError,
)

#: Foreground poll interval for ``--live`` (seconds).
POLL_INTERVAL_ENV = "SSGREP_INDEX_POLL_SECONDS"
_DEFAULT_POLL_SECONDS = 5.0

logger = logging.getLogger(__name__)

# --- Upstream FutureWarning quarantine -------------------------------------
#: cocoindex 1.0.20 (newest as of ssgrep 0.2.0) calls
#: ``SentenceTransformer.get_sentence_embedding_dimension()``, deprecated in
#: sentence-transformers >= 5.7.0. No newer cocoindex exists and the pinned
#: site-packages can't be patched, so silence only that exact upstream message
#: (no other FutureWarning is hidden).
_QUARANTINED_UPSTREAM_WARNINGS = (
    "The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.",
)


def _quarantine_upstream_future_warnings() -> None:
    for _cocoindex_future_warning in _QUARANTINED_UPSTREAM_WARNINGS:
        warnings.filterwarnings(
            "ignore",
            message=_cocoindex_future_warning,
            category=FutureWarning,
        )


_quarantine_upstream_future_warnings()


def _build_environment(*, quiet: bool = False) -> coco.Environment:
    """One self-contained runtime; no process-global cocoindex state.

    The library pins a process-wide default environment whose lifespan runs
    exactly once, so repeated in-process runs (MCP-triggered indexes, ``--live``
    polling) would silently reuse the first run's journal path. Each app owns a
    dedicated ``coco.Environment`` instead, with its own journal, providers, and
    event loop (the loop thread is daemonized; one-shot runs exit cleanly).
    """

    async def _provide() -> None:
        configure_model_loading(MODEL_ID)
        try:
            ensure_model_downloaded(
                MODEL_ID, MODEL_REVISION, quiet=quiet, description="Downloading embedding model"
            )
        except Exception as exc:  # noqa: BLE001 - model failures are user-facing
            raise_model_error(exc, MODEL_ID)
            raise
        env.context_provider.provide(EMBEDDER, ColBERTEmbedder())
        env.context_provider.provide(LANCE_DB, await lancedb.connect_async(str(_database_dir())))

    settings = coco.Settings(db_path=lmdb_path())
    env = coco.Environment(settings, name=APP_NAME)
    raise_nofile_limit()
    asyncio.run_coroutine_threadsafe(_provide(), env.event_loop).result()
    return env


@coco.fn
async def _app_main(source_map: dict[str, SourceDescriptor]):
    timestamp_specs: dict[str, LanceType | VectorSchemaProvider] = {
        "timestamp": rows_mod.TIMESTAMP_SPEC,
    }
    chunk_specs: dict[str, LanceType | VectorSchemaProvider] = {
        "timestamp": rows_mod.TIMESTAMP_SPEC,
        "vector": rows_mod.VECTOR_LANCE,
        "proxy_vector": rows_mod.PROXY_LANCE,
    }
    chunk_table = await lancedb.mount_table_target(
        LANCE_DB,
        CHUNKS_TABLE,
        await lancedb.TableSchema[rows_mod.ChunkRow].from_class(
            rows_mod.ChunkRow,
            primary_key=["chunk_id"],
            column_specs=chunk_specs,
        ),
        managed_by=ManagedBy.USER,
    )
    episode_table = await lancedb.mount_table_target(
        LANCE_DB,
        EPISODES_TABLE,
        await lancedb.TableSchema[rows_mod.EpisodeRow].from_class(
            rows_mod.EpisodeRow,
            primary_key=["episode_id"],
            column_specs=timestamp_specs,
        ),
        managed_by=ManagedBy.USER,
    )
    session_table = await lancedb.mount_table_target(
        LANCE_DB,
        SESSIONS_TABLE,
        await lancedb.TableSchema[rows_mod.SessionRow].from_class(
            rows_mod.SessionRow,
            primary_key=["session_id"],
            column_specs={"absent_since": rows_mod.TIMESTAMP_SPEC},
        ),
        managed_by=ManagedBy.USER,
    )
    lm: LiveMap[str, SourceDescriptor] = await LiveMap.create()
    handle = await coco.mount(produce_entries, lm, source_map)
    await handle.ready()
    await coco.mount_each(
        process_source,
        lm,
        chunk_table,
        episode_table,
        session_table,
    )


def _build_app(
    entries: dict[str, SourceDescriptor],
    environment: coco.Environment | None = None,
    *,
    quiet: bool = False,
) -> coco.App:
    """Wire the app: runtime environment, targets, and the registry LiveMap.

    ``app_main`` is module-level so its code identity is stable across runs
    (memoization fingerprints function code). Live mode passes one shared
    environment so polling cycles reuse providers and the journal.
    """
    environment = environment or _build_environment(quiet=quiet)
    return coco.App(coco.AppConfig(name=APP_NAME, environment=environment), _app_main, entries)


def _database_dir() -> Path:
    from ssgrep.store.paths import database_dir

    return database_dir()


def _expected_meta(repository: LanceStore) -> dict[str, str]:
    from ssgrep.indexing.embed import DIMENSION, MODEL_ID, MODEL_REVISION

    return {
        "schema_version": str(repository.schema_version),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vector_dimension": str(DIMENSION),
    }


def _assert_compatible(repository: LanceStore) -> None:
    expected = _expected_meta(repository)
    compatible = (
        repository.schema_matches()
        and repository.get_meta("index_state") in {"ready", "building"}
        and all(repository.get_meta(key) == value for key, value in expected.items())
    )
    if compatible:
        return
    raise IndexNotReadyError(
        "The global index schema or embedding model changed; run `ssgrep index --rebuild`."
    )


def _guard_shrink(repository: LanceStore, count: int, *, allow_shrink: bool) -> None:
    if not repository.exists() or not repository.count(SESSIONS_TABLE):
        return
    if count < repository.count(SESSIONS_TABLE) / 2 and not allow_shrink:
        raise RebuildWouldShrinkError(
            f"Rebuild would shrink the global corpus from {repository.count(SESSIONS_TABLE)} "
            f"to {count} sessions. Re-run with --allow-shrink if intentional.",
            old_counts=(
                repository.count(SESSIONS_TABLE),
                repository.count(EPISODES_TABLE),
                repository.count(CHUNKS_TABLE),
            ),
            new_counts=(count, 0, 0),
        )


def _mark_available(repository: LanceStore, session_id: str) -> None:
    """Flip one session back to available when its source reappeared."""
    predicate = f"session_id = {quote(session_id)}"
    rows = repository.rows(SESSIONS_TABLE, where=predicate, columns=["source_status"], limit=1)
    if not rows or rows[0].get("source_status") != "absent":
        return
    repository.update(
        SESSIONS_TABLE,
        predicate,
        {"source_status": "available", "absent_since": None},
    )
    repository.update(EPISODES_TABLE, predicate, {"source_status": "available"})
    repository.update(CHUNKS_TABLE, predicate, {"source_status": "available"})


def _reconcile_absent(repository: LanceStore, discovered: set[str]) -> None:
    """Tombstone gone sources and restore reappeared ones (full runs only)."""
    for row in repository.rows(SESSIONS_TABLE, columns=["session_id", "path", "source_status"]):
        path = str(row["path"])
        status = "available" if path in discovered else "absent"
        if status == row.get("source_status"):
            continue
        predicate = f"session_id = {quote(str(row['session_id']))}"
        session_values: dict[str, Any] = {
            "source_status": status,
            "absent_since": datetime.now(UTC) if status == "absent" else None,
        }
        repository.update(SESSIONS_TABLE, predicate, session_values)
        repository.update(EPISODES_TABLE, predicate, {"source_status": status})
        repository.update(CHUNKS_TABLE, predicate, {"source_status": status})


def _compact_after_reconcile(repository: LanceStore) -> str | None:
    """Compact tables once after a successful full reconciliation.

    Reclaims the row versions left behind by merge_insert and tombstone
    flips, pruning superseded fragments immediately (see
    ``POST_COMPACT_CLEANUP_RETENTION``). Best-effort by contract: a
    compaction failure is logged and the run still succeeds. Returns the ISO
    timestamp of the completed optimize, or ``None`` when it failed — the
    caller reports the shorthand byte count *after* compaction so that
    ``index`` and a subsequent ``status`` agree on the settled on-disk size.
    Called from the one-shot catch-up path only — never inside the ``--live``
    polling loop.
    """
    try:
        repository.optimize_tables()
    except Exception as exc:  # noqa: BLE001 - compaction must not fail indexing
        logger.warning("Post-reconcile compaction failed: %s", exc)
        return None
    timestamp = datetime.now(UTC).isoformat()
    repository.set_meta("last_optimize_time", timestamp)
    return timestamp


def _write_cwd_cache(repository: LanceStore, fresh: list) -> None:
    """Keep the discovery cwd projection fresh for cache_cwds sources."""
    for source in fresh:
        if not source.cache_cwds:
            continue
        repository.upsert(
            CWD_CACHE_TABLE,
            {
                "path": str(source.session.path),
                "size": source.fingerprint.size,
                "mtime": source.fingerprint.mtime,
                "cwds": "\n".join(source.session.project_paths),
            },
        )


def _run_app(
    entries: dict[str, SourceDescriptor],
    *,
    environment: coco.Environment | None,
    full_reprocess: bool,
    quiet: bool,
    fresh_keys: Collection[str] = (),
) -> None:
    """Build and drive the app, translating model-load failures.

    A rebuild drops the Lance tables, so the memo journal must not be allowed
    to skip unchanged sources: force full reprocessing or the freshly reset
    tables would stay empty for memo-hit keys. Embedding-model load failures
    (gated repo, missing auth, network, corrupt cache) become the actionable
    ``ModelDownloadError`` instead of a raw huggingface_hub stack trace.

    ``fresh_keys`` -- this run's own discovery, not just the registry -- lets
    ``capture_archives`` skip its presence check for sources it just saw.
    """
    environment = environment or _build_environment(quiet=quiet)
    try:
        app = _build_app(entries, environment=environment, quiet=quiet)
        # Missing-source declarations must come from a stable pre-update snapshot,
        # not tables being modified concurrently by sibling source components.
        archives = capture_archives(entries, fresh_keys=fresh_keys)
        environment.context_provider.provide(ARCHIVED_ROWS, archives)
        asyncio.run_coroutine_threadsafe(
            _drive_update(
                app,
                total=len(entries),
                full_reprocess=full_reprocess,
                quiet=quiet,
            ),
            environment.event_loop,
        ).result()
    except Exception as exc:  # noqa: BLE001 - model failures are user-facing
        raise_model_error(exc, MODEL_ID)
        raise

    finally:
        environment.context_provider.provide(ARCHIVED_ROWS, {})


def _reconcile_once(
    repository: LanceStore,
    *,
    rebuild: bool,
    no_subagents: bool,
    allow_shrink: bool,
    scope: str | None,
    full_reprocess: bool,
    quiet: bool = False,
    environment: coco.Environment | None = None,
) -> IndexStats:
    """One full catch-up cycle: discover, sync the delta, reconcile targets."""
    with Spinner("Discovering transcripts", quiet=quiet):
        fresh = transcript_adapters.discover_sources(scope=scope, no_subagents=no_subagents)
    if rebuild:
        _guard_shrink(repository, len(fresh), allow_shrink=allow_shrink)
        repository.reset()
    else:
        repository.initialize()
    repository.set_meta("index_state", "building")

    diagnostics.reset()
    registry = read_registry(repository)
    entries = union_descriptors(fresh, registry)
    _run_app(
        entries,
        environment=environment,
        full_reprocess=full_reprocess or rebuild,
        quiet=quiet,
        fresh_keys={source.key for source in fresh},
    )

    # --- post-step: registry, availability, full-text, metadata, stats ---
    with Spinner("Reconciling index state", quiet=quiet):
        write_registry(repository, [to_descriptor(source) for source in fresh])
        _write_cwd_cache(repository, fresh)
        for source in fresh:
            _mark_available(repository, source.session.session_id)
        if scope is None and not no_subagents:
            _reconcile_absent(repository, {source.key for source in fresh})

    malformed, skipped, archived = diagnostics.snapshot()
    now = datetime.now(UTC)
    for key, value in {**_expected_meta(repository), "last_index_time": now.isoformat()}.items():
        repository.set_meta(key, value)
    repository.set_meta("malformed_records", str(malformed))
    repository.set_meta("skipped_records", str(skipped))
    repository.set_meta("archived_source_count", str(archived))
    # Lexical statistics for hybrid retrieval: one columnar read of chunk
    # text (id + text only), persisted as JSON. Search-time BM25 fusion
    # reads these stats instead of rescanning the corpus.
    stats = corpus_stats(
        repository.rows(CHUNKS_TABLE, columns=["chunk_id", "text"], limit=10_000_000)
    )
    repository.set_meta("lexical_stats", dump_stats(stats))
    repository.set_meta("index_state", "ready")
    return IndexStats(
        session_count=repository.count(SESSIONS_TABLE),
        episode_count=repository.count(EPISODES_TABLE),
        chunk_count=repository.count(CHUNKS_TABLE),
        index_size_bytes=rows_mod.directory_size(repository.path),
        last_index_time=now,
        model_id=repository.get_meta("model_id") or "",
        vector_dimension=int(repository.get_meta("vector_dimension") or 0),
        skipped_records=skipped,
        malformed_records=malformed,
        archived_source_count=archived,
        schema_version=repository.schema_version,
        tombstoned_source_count=repository.count(SESSIONS_TABLE, "source_status = 'absent'"),
        tombstoned_chunk_count=repository.count(CHUNKS_TABLE, "source_status = 'absent'"),
        data_dir=str(repository.root),
        runtime_counts=transcript_adapters.source_counts(fresh),
    )


def run(
    *,
    rebuild: bool = False,
    no_subagents: bool = False,
    allow_shrink: bool = False,
    scope: str | None = None,
    quiet: bool = False,
    live: bool = False,
    full_reprocess: bool = False,
) -> IndexStats:
    """Reconcile the global transcript corpus through the CocoIndex engine.

    Catch-up mode runs one cycle and exits. ``live=True`` keeps the process in
    the foreground, re-running catch-up cycles forever (a poll, not a daemon);
    each cycle re-discovers sources, so new transcripts are picked up. Table
    compaction runs once after a successful one-shot cycle only — polling
    cycles never compact.
    """
    repository = LanceStore()
    ensure_data_dir()
    if repository.exists() and not rebuild:
        _assert_compatible(repository)
    if live:
        try:
            interval = float(os.environ.get(POLL_INTERVAL_ENV, str(_DEFAULT_POLL_SECONDS)))
        except ValueError:
            interval = _DEFAULT_POLL_SECONDS
        environment = _build_environment(quiet=quiet)
        try:
            while True:
                _reconcile_once(
                    repository,
                    rebuild=rebuild,
                    no_subagents=no_subagents,
                    allow_shrink=allow_shrink,
                    scope=scope,
                    full_reprocess=full_reprocess,
                    quiet=quiet,
                    environment=environment,
                )
                time.sleep(interval)
        except KeyboardInterrupt:
            return _reconcile_once(
                repository,
                rebuild=False,
                no_subagents=no_subagents,
                allow_shrink=allow_shrink,
                scope=scope,
                full_reprocess=False,
                quiet=quiet,
                environment=environment,
            )
    stats = _reconcile_once(
        repository,
        rebuild=rebuild,
        no_subagents=no_subagents,
        allow_shrink=allow_shrink,
        scope=scope,
        full_reprocess=full_reprocess,
        quiet=quiet,
    )
    optimize_time = _compact_after_reconcile(repository)
    # _reconcile_once measured the directory before compaction rewrote the
    # fragments; re-measure so the printed size equals what status reports.
    if optimize_time is not None:
        stats = replace(
            stats,
            index_size_bytes=rows_mod.directory_size(repository.path),
            last_optimize_time=optimize_time,
        )
    return stats


__all__ = ["run"]
