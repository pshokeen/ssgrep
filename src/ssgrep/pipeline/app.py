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
import concurrent.futures
import logging
import os
import time
import warnings
from collections.abc import Collection, Coroutine
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
from ssgrep.pipeline.reconcile import (
    assert_compatible,
    compact_after_reconcile,
    expected_meta,
    guard_shrink,
    mark_available,
    reconcile_absent,
    write_cwd_cache,
)
from ssgrep.pipeline.sources import (
    SourceDescriptor,
    read_registry,
    to_descriptor,
    union_descriptors,
    write_registry,
)
from ssgrep.pipeline.state import (
    APP_NAME,
    ARCHIVED_ROWS,
    EMBEDDER,
    LANCE_DB,
    SOURCE_MAP,
    lmdb_path,
)
from ssgrep.pipeline.update import (
    ComponentUpdateError,
    _drive_update as _drive_update,
    _source_progress as _source_progress,
)
from ssgrep.search.lexical import corpus_stats, dump_stats
from ssgrep.sessions import adapters as transcript_adapters
from ssgrep.store import (
    CHUNKS_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    LanceStore,
    ensure_data_dir,
)
from ssgrep.utilities.limits import raise_nofile_limit
from ssgrep.utilities.types import IndexStats

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


def _build_environment(path: Path, *, quiet: bool = False) -> coco.Environment:
    """One self-contained runtime; no process-global cocoindex state.

    The library pins a process-wide default environment whose lifespan runs
    exactly once, so repeated in-process runs (MCP-triggered indexes, ``--live``
    polling) would silently reuse the first run's journal path. Each app owns a
    dedicated ``coco.Environment`` instead, with its own journal, providers, and
    event loop, kept for the process by ``_runtime`` (the loop thread is
    daemonized; one-shot runs exit cleanly).
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

    env = coco.Environment(coco.Settings(db_path=path), name=APP_NAME)
    raise_nofile_limit()
    asyncio.run_coroutine_threadsafe(_provide(), env.event_loop).result()
    return env


#: Each LMDB journal's environment and App, kept for the life of the process.
_RUNTIMES: dict[Path, tuple[coco.Environment, coco.App]] = {}


def _runtime(*, quiet: bool = False) -> tuple[coco.Environment, coco.App]:
    """The environment and App for the current journal, built on first use.

    cocoindex cannot close a non-default environment: its LMDB journal is
    released only when the object is garbage-collected, and reopening the
    journal before then raises "environment already open". Every run in the
    process -- one-shot, ``--live`` cycle, MCP reconcile -- therefore reuses
    the journal's one environment, keyed by the current ``lmdb_path()`` so a
    switched data root gets its own. The engine also keeps an App's name
    registered once it has processed sources, so each run updates this one
    App; ``_app_main`` is module-level so its code identity (which memoization
    fingerprints) is stable across runs.
    """
    path = lmdb_path().resolve()
    if path not in _RUNTIMES:
        environment = _build_environment(path, quiet=quiet)
        app = coco.App(coco.AppConfig(name=APP_NAME, environment=environment), _app_main)
        _RUNTIMES[path] = (environment, app)
    return _RUNTIMES[path]


@coco.fn
async def _app_main():
    source_map: dict[str, SourceDescriptor] = coco.use_context(SOURCE_MAP)
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


async def _update(environment: coco.Environment, app: coco.App, **options: Any) -> None:
    """Drive one update against the current data root's database.

    A journal pinned by ``SSGREP_COCOINDEX_DB`` keeps its environment while the
    data root changes, so the Lance connection is provided per run.
    """
    lance = await lancedb.connect_async(str(_database_dir()))
    environment.context_provider.provide(LANCE_DB, lance)
    await _drive_update(app, **options)


def _database_dir() -> Path:
    from ssgrep.store.paths import database_dir

    return database_dir()


class ForcedInterrupt(KeyboardInterrupt):
    """A second Ctrl-C while a live cycle was being allowed to finish."""


def _drive_to_completion(
    loop: asyncio.AbstractEventLoop,
    coroutine: Coroutine[Any, Any, None],
    *,
    finish_on_interrupt: bool,
) -> None:
    """Run ``coroutine`` on the engine loop and block until it finishes.

    An interrupt (Ctrl-C) lands here, in the thread blocked on the engine, but
    the engine cannot be aborted: its update keeps running on its own threads
    and still owns the App. A live cycle is therefore allowed to finish before
    the interrupt propagates, so the final catch-up never overlaps it (a
    second Ctrl-C stops waiting with ``ForcedInterrupt``). A one-shot run has
    nothing to come back for and propagates the interrupt at once.
    """
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    try:
        future.result()
    except KeyboardInterrupt:
        if not finish_on_interrupt:
            raise
        try:
            logger.warning("Interrupted; finishing the current cycle (Ctrl-C again to stop now)")
            concurrent.futures.wait([future])
        except KeyboardInterrupt as forced:
            raise ForcedInterrupt from forced
        raise


def _run_app(
    entries: dict[str, SourceDescriptor],
    *,
    live: bool,
    full_reprocess: bool,
    quiet: bool,
    fresh_keys: Collection[str] = (),
) -> None:
    """Drive this journal's app, translating model-load failures.

    A rebuild drops the Lance tables, so the memo journal must not be allowed
    to skip unchanged sources: force full reprocessing or the freshly reset
    tables would stay empty for memo-hit keys. Embedding-model load failures
    (gated repo, missing auth, network, corrupt cache) become the actionable
    ``ModelDownloadError`` instead of a raw huggingface_hub stack trace.

    ``fresh_keys`` -- this run's own discovery, not just the registry -- lets
    ``capture_archives`` skip its presence check for sources it just saw.
    A ``live`` cycle must outlive an interrupt (see ``_drive_to_completion``).
    """
    environment, app = _runtime(quiet=quiet)
    try:
        # Missing-source declarations must come from a stable pre-update snapshot,
        # not tables being modified concurrently by sibling source components.
        archives = capture_archives(entries, fresh_keys=fresh_keys)
        environment.context_provider.provide(ARCHIVED_ROWS, archives)
        environment.context_provider.provide(SOURCE_MAP, entries)
        _drive_to_completion(
            environment.event_loop,
            _update(
                environment,
                app,
                total=len(entries),
                full_reprocess=full_reprocess,
                quiet=quiet,
            ),
            finish_on_interrupt=live,
        )
    except Exception as exc:  # noqa: BLE001 - model failures are user-facing
        raise_model_error(exc, MODEL_ID)
        raise

    finally:
        environment.context_provider.provide(ARCHIVED_ROWS, {})
        environment.context_provider.provide(SOURCE_MAP, {})


def _reconcile_once(
    repository: LanceStore,
    *,
    rebuild: bool,
    no_subagents: bool,
    allow_shrink: bool,
    scope: str | None,
    full_reprocess: bool,
    quiet: bool = False,
    live: bool = False,
) -> IndexStats:
    """One full catch-up cycle: discover, sync the delta, reconcile targets."""
    with Spinner("Discovering transcripts", quiet=quiet):
        fresh = transcript_adapters.discover_sources(scope=scope, no_subagents=no_subagents)
    if rebuild:
        guard_shrink(repository, len(fresh), allow_shrink=allow_shrink)
        repository.reset()
    else:
        repository.initialize()
    repository.set_meta("index_state", "building")

    diagnostics.reset()
    registry = read_registry(repository)
    entries = union_descriptors(fresh, registry)
    _run_app(
        entries,
        live=live,
        full_reprocess=full_reprocess or rebuild,
        quiet=quiet,
        fresh_keys={source.key for source in fresh},
    )

    # --- post-step: registry, availability, full-text, metadata, stats ---
    with Spinner("Reconciling index state", quiet=quiet):
        write_registry(repository, [to_descriptor(source) for source in fresh])
        write_cwd_cache(repository, fresh)
        for source in fresh:
            mark_available(repository, source.session.session_id)
        if scope is None and not no_subagents:
            reconcile_absent(repository, {source.key for source in fresh})

    malformed, skipped, archived = diagnostics.snapshot()
    now = datetime.now(UTC)
    for key, value in {**expected_meta(repository), "last_index_time": now.isoformat()}.items():
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

    ``rebuild`` and ``full_reprocess`` are one-time directives: live mode
    applies them to the first cycle only (a rebuild repeated every poll would
    reset the tables each cycle), and an interrupted first cycle leaves them
    pending for the final catch-up. An unfinished earlier build (or a missing
    database whose memo journal survived) forces the first cycle to ignore the
    journal for the same reason a rebuild does: the rows it vouches for are gone.
    """
    repository = LanceStore()
    ensure_data_dir()
    unfinished = False
    if not rebuild:
        unfinished = not repository.exists() or assert_compatible(repository)
    if live:
        try:
            interval = float(os.environ.get(POLL_INTERVAL_ENV, str(_DEFAULT_POLL_SECONDS)))
        except ValueError:
            interval = _DEFAULT_POLL_SECONDS
        # Every cycle updates this journal's one App (see ``_runtime``).
        pending_rebuild = rebuild
        pending_full = full_reprocess or unfinished
        try:
            while True:
                try:
                    _reconcile_once(
                        repository,
                        rebuild=pending_rebuild,
                        no_subagents=no_subagents,
                        allow_shrink=allow_shrink,
                        scope=scope,
                        full_reprocess=pending_full,
                        quiet=quiet,
                        live=True,
                    )
                except ComponentUpdateError as exc:
                    # A poller must outlive a per-source failure (locked db, unreadable
                    # file): report it and retry. The cycle itself finished, so a pending
                    # reset is spent; keep only forced reprocessing so the reset tables
                    # are refilled behind memo hits instead of being reset every poll.
                    logger.error("%s; retrying in %ss", exc, interval)
                    pending_full = pending_full or pending_rebuild
                    pending_rebuild = False
                else:
                    pending_rebuild = pending_full = False
                time.sleep(interval)
        except ForcedInterrupt:
            raise
        except KeyboardInterrupt:
            # The engine driver let the interrupted cycle finish; catch up once
            # more so the index is left ready (and any pending rebuild done).
            return _reconcile_once(
                repository,
                rebuild=pending_rebuild,
                no_subagents=no_subagents,
                allow_shrink=allow_shrink,
                scope=scope,
                full_reprocess=pending_full,
                quiet=quiet,
                live=True,
            )
    stats = _reconcile_once(
        repository,
        rebuild=rebuild,
        no_subagents=no_subagents,
        allow_shrink=allow_shrink,
        scope=scope,
        full_reprocess=full_reprocess or unfinished,
        quiet=quiet,
    )
    optimize_time = compact_after_reconcile(repository)
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
