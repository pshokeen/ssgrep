"""SQLite schema initialization and basic data access."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from ssgrep.types import Chunk, Episode, FileCursor, SessionFile

SCHEMA_VERSION = 4


def copy_database(source: Path, destination: Path) -> None:
    """Copy a live index database, INCLUDING anything still in its WAL.

    Never use shutil.copy2() for this. init_db() sets
    ``PRAGMA journal_mode=WAL``, so a committed transaction lives in
    ``index.db-wal`` until something checkpoints it. copy2() copies the main
    file and nothing else, so every commit since the last checkpoint is
    silently absent from the copy -- and both callers then commit that copy
    as the new live generation and unlink the original, taking the WAL's
    inode with it.

    That was a real, silent, unrecoverable loss on two ungated paths.
    `ssgrep revectorize` and `ssgrep prune` (via cleanup_orphaned_vectors)
    each build the next generation from a copy of the live database, and
    neither goes through rebuild_guard. A checkpoint only happens on a clean
    connection close, so any of these leaves the newest sessions WAL-only:
    an indexing run still in flight (exactly what hook-driven async indexing
    produces by design), a writer killed after commit (hook timeout, SIGKILL,
    power loss), or a concurrent reader holding a snapshot when the writer
    exits. Measured end to end: 9 sessions became 4 through revectorize, and
    12 became 5 through prune -- both reporting success and exiting 0, with
    only a quietly smaller count to show for it. For sessions Claude Code's
    retention had already deleted, the index was the last surviving copy.

    The backup API is the correct primitive rather than a checkpoint pragma:
    ``wal_checkpoint(TRUNCATE)`` returns BUSY under a live reader and would
    have to be checked rather than fired and forgotten, whereas
    ``Connection.backup()`` reads through the WAL by construction and needs
    no cooperation from other processes.
    """
    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(db_path.parent, 0o700)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            text TEXT NOT NULL,
            content_type TEXT NOT NULL,
            vec_row INTEGER,
            source_status TEXT NOT NULL DEFAULT 'available'
        );
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            timestamp TEXT,
            git_branch TEXT,
            cwd TEXT,
            files_touched TEXT,
            tool_names TEXT,
            is_subagent INTEGER DEFAULT 0,
            agent_type TEXT,
            agent_name TEXT,
            agent_description TEXT,
            parent_session_id TEXT,
            prompt_text TEXT,
            response_text TEXT,
            source_status TEXT NOT NULL DEFAULT 'available'
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            is_main INTEGER NOT NULL,
            parent_session_id TEXT,
            agent_hash TEXT,
            agent_type TEXT,
            agent_name TEXT,
            agent_description TEXT,
            agent_model TEXT,
            source_status TEXT NOT NULL DEFAULT 'available'
        );
        CREATE TABLE IF NOT EXISTS session_files (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            byte_offset INTEGER NOT NULL,
            first_line_hash TEXT NOT NULL,
            source_status TEXT NOT NULL DEFAULT 'available'
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id, text, content_type
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_tri USING fts5(
            chunk_id, text, tokenize='trigram'
        );
    """)

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    return conn


def insert_chunk(conn: sqlite3.Connection, chunk: Chunk, vec_row: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            chunk.chunk_id,
            chunk.episode_id,
            chunk.session_id,
            chunk.text,
            chunk.content_type.value,
            vec_row,
            "available",
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunks_fts VALUES (?, ?, ?)",
        (chunk.chunk_id, chunk.text, chunk.content_type.value),
    )
    conn.execute(
        "INSERT OR REPLACE INTO chunks_fts_tri VALUES (?, ?)",
        (chunk.chunk_id, chunk.text),
    )


def insert_episode(
    conn: sqlite3.Connection, episode: Episode, prompt_text: str, response_text: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            episode.episode_id,
            episode.session_id,
            episode.title,
            episode.timestamp.isoformat() if episode.timestamp else None,
            episode.git_branch,
            episode.cwd,
            "\n".join(episode.files_touched),
            "\n".join(episode.tool_names),
            int(episode.is_subagent),
            episode.agent_type,
            episode.agent_name,
            episode.agent_description,
            episode.parent_session_id,
            prompt_text,
            response_text,
            "available",
        ),
    )


def insert_session(conn: sqlite3.Connection, session: SessionFile) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session.session_id,
            str(session.path),
            int(session.is_main),
            session.parent_session_id,
            session.agent_hash,
            session.agent_type,
            session.agent_name,
            session.agent_description,
            session.agent_model,
            "available",
        ),
    )


def upsert_session_file(conn: sqlite3.Connection, cursor: FileCursor) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO session_files VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(cursor.path),
            cursor.size,
            cursor.mtime,
            cursor.byte_offset,
            cursor.first_line_hash,
            "available",
        ),
    )


def get_session_file(conn: sqlite3.Connection, path: Path) -> FileCursor | None:
    row = conn.execute("SELECT * FROM session_files WHERE path = ?", (str(path),)).fetchone()
    if not row:
        return None
    return FileCursor(
        path=Path(row[0]), size=row[1], mtime=row[2], byte_offset=row[3], first_line_hash=row[4]
    )


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT chunk_id, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def search_fts_trigram(
    conn: sqlite3.Connection, query: str, limit: int = 20
) -> list[tuple[str, float]]:
    """BM25 over the trigram-tokenized mirror of the chunk text.

    The trigram tokenizer matches on any shared 3-character substring, so a
    query term like "undercounted" can reach a chunk that says
    "under-reported", and "gating" can reach "gate" -- morphological and
    compound-word variation that the word-boundary unicode61 tokenizer in
    chunks_fts treats as entirely different terms. Terms shorter than three
    characters produce no trigrams and simply match nothing, which is the
    tokenizer's own documented behavior, not an error.
    """
    rows = conn.execute(
        "SELECT chunk_id, rank FROM chunks_fts_tri WHERE chunks_fts_tri MATCH ? "
        "ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
