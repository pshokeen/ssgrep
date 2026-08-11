"""SQLite schema and data access."""

from __future__ import annotations

import sqlite3

from ssgrep.store.compaction import cleanup_orphaned_vectors
from ssgrep.store.cwd_cache import (
    CWD_CACHE_DB_NAME,
    delete_cwd_cache_rows,
    init_cwd_cache,
    load_cwd_cache_rows,
    save_cwd_cache_rows,
)
from ssgrep.store.generations import GenerationalStore
from ssgrep.store.lifecycle import (
    delete_session_chunks,
    get_tombstone_stats,
    mark_source_available,
    tombstone_session_chunks,
)
from ssgrep.store.schema import (
    SCHEMA_VERSION,
    copy_database,
    get_meta,
    get_session_file,
    init_db,
    insert_chunk,
    insert_episode,
    insert_session,
    search_fts,
    search_fts_trigram,
    set_meta,
    upsert_session_file,
)
from ssgrep.types import Chunk, Episode, FileCursor, SessionFile

__all__ = [
    "SCHEMA_VERSION",
    "CWD_CACHE_DB_NAME",
    "Chunk",
    "Episode",
    "FileCursor",
    "GenerationalStore",
    "SessionFile",
    "cleanup_orphaned_vectors",
    "copy_database",
    "delete_cwd_cache_rows",
    "delete_session_chunks",
    "get_meta",
    "get_session_file",
    "get_tombstone_stats",
    "init_cwd_cache",
    "init_db",
    "insert_chunk",
    "insert_episode",
    "insert_session",
    "load_cwd_cache_rows",
    "mark_source_available",
    "save_cwd_cache_rows",
    "search_fts",
    "search_fts_trigram",
    "set_meta",
    "sqlite3",
    "tombstone_session_chunks",
    "upsert_session_file",
]
