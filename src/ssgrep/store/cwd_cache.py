"""Global cwd-projection cache for discovery."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# --- D11 cwd-projection cache ------------------------------------------
#
# discovery.discover_sessions() confirms folder scope by streaming every
# record of every transcript under ~/.claude/projects for its cwd field
# (D11: the encoded project-directory name is not a reliable proxy for
# where a session actually ran). That full scan costs ~1s over the real
# corpus and, unlike session_files above, is not scoped to any single
# project's .ssgrep/ index -- it is a property of the whole transcript
# corpus, shared by every project's search. It therefore lives in its own
# small database (see discovery._cwd_cache_db_path) rather than inside any
# one project's index.db, so the first search from ANY project warms it for
# every later search, of any project. Schema mirrors session_files' (path,
# size, mtime, byte_offset, first_line_hash) cursor shape, plus the cwds
# payload that evidence was collected for.

CWD_CACHE_DB_NAME = "cwd_index.db"


def init_cwd_cache(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the global cwd-projection cache database.

    Same connection posture as init_db(): WAL journal mode and a busy
    timeout so a concurrent ssgrep process waits on the write lock rather
    than hard-failing. The containing directory is created 0o700 since cwd
    values reveal project directory names.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(db_path.parent, 0o700)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cwd_projection (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            byte_offset INTEGER NOT NULL,
            first_line_hash TEXT NOT NULL,
            cwds TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def load_cwd_cache_rows(conn: sqlite3.Connection) -> list[tuple[str, int, float, int, str, str]]:
    """Return every cached (path, size, mtime, byte_offset, first_line_hash,
    cwds) row. `cwds` is the raw newline-joined blob; callers split it
    themselves, matching the files_touched/tool_names convention used
    elsewhere in this module.
    """
    return conn.execute(
        "SELECT path, size, mtime, byte_offset, first_line_hash, cwds FROM cwd_projection"
    ).fetchall()


def save_cwd_cache_rows(
    conn: sqlite3.Connection, rows: list[tuple[str, int, float, int, str, str]]
) -> None:
    """Upsert changed/new rows. Callers pass only rows that actually needed
    rescanning -- an unchanged file contributes nothing here, which is what
    keeps the warm-cache path cheap.
    """
    if not rows:
        return
    conn.executemany("INSERT OR REPLACE INTO cwd_projection VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def delete_cwd_cache_rows(conn: sqlite3.Connection, paths: list[str]) -> None:
    """Remove cache rows for paths the last full corpus walk did not see.

    Keeps a vanished file's stale evidence from resurrecting it in some
    later run -- the corpus prunes itself (D12), and this table must not
    outlive that.
    """
    if not paths:
        return
    conn.executemany("DELETE FROM cwd_projection WHERE path = ?", [(p,) for p in paths])
    conn.commit()
