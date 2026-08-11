"""Index observability: status reporting without model loading.

Exposes index metadata, corpus counts, freshness, model binding, parsing
degradation, and drift detection — all without loading the embedding model,
which costs 184ms and is never used by status.

All queries are against already-stored metadata and counts, or disk I/O
(file sizes), never against vectors or embeddings.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from ssgrep import discovery, embed, search, staleness, store
from ssgrep.types import IndexStats

INDEX_DIRNAME = ".ssgrep"


def status(project_dir: Path) -> IndexStats:
    """Get index observability data.

    Succeeds even when no index exists, reporting that absence as a normal
    state rather than an error. The index_exists field on the returned
    IndexStats distinguishes a missing or uninitialized index from one that
    exists but contains no data.

    Does not load the embedding model, even to validate vector dimensions.
    Model info is read from stored metadata instead.

    Args:
        project_dir: Path to the project directory.

    Returns:
        IndexStats with counts, freshness, model binding, degradation data,
        and index_exists flag to signal whether an index is present.
    """
    index_dir = project_dir / INDEX_DIRNAME
    gen_store = store.GenerationalStore(index_dir)
    db_path = gen_store.get_index_path()

    # No index exists — return a minimal IndexStats with index_exists=False
    if not db_path.exists():
        return IndexStats(
            session_count=0,
            episode_count=0,
            chunk_count=0,
            index_size_bytes=0,
            last_index_time=None,
            model_id=embed.MODEL_ID,
            vector_dimension=embed.DIMENSION,
            skipped_records=0,
            malformed_records=0,
            schema_version=store.SCHEMA_VERSION,
            tombstoned_source_count=0,
            tombstoned_chunk_count=0,
            index_exists=False,
        )

    # Index exists — query database for all counts and metadata
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Core counts
        session_count = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM sessions WHERE source_status = 'available'"
        ).fetchone()[0]

        episode_count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE source_status = 'available'"
        ).fetchone()[0]

        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_status = 'available'"
        ).fetchone()[0]

        # Tombstone counts
        tombstoned_source_count, tombstoned_chunk_count = store.get_tombstone_stats(conn)

        # Metadata: last index time, model id, vector dimension, parsing stats
        last_index_time_str = store.get_meta(conn, "last_index_time")
        last_index_time = None
        if last_index_time_str:
            try:
                last_index_time = datetime.fromisoformat(last_index_time_str)
            except ValueError:
                pass

        model_id = store.get_meta(conn, "model_id") or embed.MODEL_ID
        vector_dimension_str = store.get_meta(conn, "vector_dimension")
        vector_dimension = int(vector_dimension_str) if vector_dimension_str else embed.DIMENSION

        skipped_records_str = store.get_meta(conn, "skipped_records")
        skipped_records = int(skipped_records_str) if skipped_records_str else 0

        malformed_records_str = store.get_meta(conn, "malformed_records")
        malformed_records = int(malformed_records_str) if malformed_records_str else 0

        # Persisted by index() alongside skipped/malformed: how many work-
        # queue hints the drain's fail-closed scope filter dropped on the
        # LAST index run (blind-review nit: counted but surfaced nowhere).
        oos_str = store.get_meta(conn, "queue_items_out_of_scope")
        queue_items_out_of_scope = int(oos_str) if oos_str else 0

        # Corpus-only session count (excludes notes + external roots); persisted
        # by index() so status/MCP callers report the real value, not always 0.
        csc_str = store.get_meta(conn, "corpus_session_count")
        corpus_session_count_val = int(csc_str) if csc_str else 0

        schema_version_str = store.get_meta(conn, "schema_version")
        schema_version = int(schema_version_str) if schema_version_str else store.SCHEMA_VERSION

        # Index size on disk
        index_size_bytes = 0
        if db_path.exists():
            index_size_bytes += db_path.stat().st_size
        vec_path = gen_store.get_vector_path()
        if vec_path.exists():
            index_size_bytes += vec_path.stat().st_size

        # Detect staleness (stat-only, no model loading).
        # Also snapshot the cwd-cache fallback counter to detect cache degradation.
        cwd_cache_fallback_before = discovery._cwd_index_stats["fallback_scans"]
        staleness_report = search.staleness_summary(project_dir)
        cwd_cache_fallback_after = discovery._cwd_index_stats["fallback_scans"]
        cwd_cache_fallback_delta = cwd_cache_fallback_after - cwd_cache_fallback_before

        is_stale = staleness.is_index_stale(staleness_report)
        stale_count_value = staleness.stale_count(staleness_report)

        return IndexStats(
            session_count=session_count,
            episode_count=episode_count,
            chunk_count=chunk_count,
            index_size_bytes=index_size_bytes,
            last_index_time=last_index_time,
            model_id=model_id,
            vector_dimension=vector_dimension,
            skipped_records=skipped_records,
            malformed_records=malformed_records,
            schema_version=schema_version,
            tombstoned_source_count=tombstoned_source_count,
            tombstoned_chunk_count=tombstoned_chunk_count,
            index_exists=True,
            stale=is_stale,
            stale_count=stale_count_value,
            cwd_cache_degraded=cwd_cache_fallback_delta > 0,
            queue_items_out_of_scope=queue_items_out_of_scope,
            corpus_session_count=corpus_session_count_val,
            cwd_cache_fallback_scans=cwd_cache_fallback_delta,
        )

    finally:
        conn.close()
