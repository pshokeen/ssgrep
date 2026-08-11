"""Shared provenance metadata for eval/results/*.json output files.

Every committed result file must carry enough metadata to answer "what,
exactly, produced this number" without re-running anything:

- which index the numbers came from (session/episode/chunk counts, embedding
  model id, vector dimension) -- `index_stats_dict`
- whether that index's chunk set was actually deduplicated, measured rather
  than assumed -- `duplication_check` (this is the same total-vs-distinct
  chunk-text measurement that first surfaced the 254x-duplicate-chunk
  indexer defect; every future result file should be checkable against
  whether that defect, or a regression of it, was present when the number
  was measured)
- which exact version of the retrieval code computed the number -- `git_info`
- every retrieval-affecting tuning constant the code held at run time, not
  just the corpus it ran against -- `tuning_constants`

`index_stats` alone pins the corpus but not the retrieval logic: a change to
fusion, roll-up, chunking, or the main-session boost would produce a
different number against a byte-identical index. `git_info` narrows that to
a commit, but a dirty tree (the normal state while a constant is being
tuned) means the commit sha alone can silently misdescribe the code that
actually ran -- `tuning_constants` closes that last gap by reading the
values directly off the modules that own them at call time, so the file
records what the code held, not what its last commit said. This is not a
hypothetical gap: a result file with `index_stats` and a commit sha but no
tuning_constants is exactly the shape of artifact that let a shipped
MAIN_SESSION_BOOST of 0.01 sit uncontradicted next to a harness recommending
0.0 for as long as it did -- nothing in the file itself could show the
mismatch.

Centralized here, in one place, so no eval script can hand-write a result
file that omits any of this.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

from ssgrep.types import IndexStats

PROJECT_DIR = Path(__file__).resolve().parent.parent


def git_info(project_dir: Path = PROJECT_DIR) -> dict:
    """Commit sha + dirty-tree flag for the retrieval code, read at runtime.

    `dirty` is True iff `git status --porcelain` reports anything -- staged,
    unstaged, or untracked. A result file measured against a dirty tree is
    not reproducible from the commit sha alone; recording the flag makes
    that fact explicit instead of silently implying reproducibility that
    isn't there.
    """

    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    sha = _run("rev-parse", "HEAD")
    dirty = bool(_run("status", "--porcelain"))
    return {"commit": sha, "dirty": dirty}


def duplication_check(db_path: Path) -> dict:
    """Measured (not assumed) chunk-text duplication and subagent share for
    one built index.

    `duplication_factor` is total_chunks / distinct_chunk_texts -- 1.0 means
    no duplication; the pre-fix corpus measured ~254x here. `subagent_chunk_
    share` is the fraction of chunks belonging to a subagent episode (join on
    episodes.is_subagent), tracked because subagent-only recall is the query
    class most sensitive to a duplicated document flooding the candidate
    pool.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        distinct = conn.execute("SELECT COUNT(DISTINCT text) FROM chunks").fetchone()[0]
        subagent = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN episodes e ON c.episode_id = e.episode_id
            WHERE e.is_subagent = 1
            """
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "total_chunks": total,
        "distinct_chunk_texts": distinct,
        "duplication_factor": (total / distinct) if distinct else None,
        "subagent_chunks": subagent,
        "subagent_chunk_share": (subagent / total) if total else None,
    }


def tuning_constants() -> dict:
    """Every retrieval-affecting constant, read live from the modules that
    own it -- never copied by hand, so this can't itself go stale the way
    the file it exists to prevent (a shipped default with no recorded
    tuning) went stale.

    LEG_POOL_SIZE is recorded once, not as two independent per-leg values:
    search.py currently uses one shared constant for both the BM25 leg
    (store.search_fts's `limit=`) and the vector leg (vectors.cosine_top_k's
    `k=`) -- there are not yet two knobs to record separately.
    """
    from ssgrep import chunker, embed
    from ssgrep import search as search_module

    return {
        "MAIN_SESSION_BOOST": search_module.MAIN_SESSION_BOOST,
        "RRF_K": search_module.RRF_K,
        "LEG_POOL_SIZE": search_module.LEG_POOL_SIZE,
        "OR_LEG_WEIGHT": search_module.OR_LEG_WEIGHT,
        "TRIGRAM_LEG_WEIGHT": search_module.TRIGRAM_LEG_WEIGHT,
        "PHRASE_LEG_WEIGHT": search_module.PHRASE_LEG_WEIGHT,
        "CHUNK_TARGET_SIZE": chunker.CHUNK_TARGET_SIZE,
        "CHUNK_OVERLAP": chunker.CHUNK_OVERLAP,
        "embed_model_id": embed.MODEL_ID,
        "embed_dimension": embed.DIMENSION,
    }


def index_stats_dict(stats: IndexStats) -> dict:
    """The subset of IndexStats that matters for provenance: corpus size and
    embedding-model binding. Deliberately excludes fields that describe the
    index artifact rather than what was measured against it (last_index_time,
    skipped/malformed record counts, schema_version, tombstone counts).
    """
    return {
        "session_count": stats.session_count,
        "episode_count": stats.episode_count,
        "chunk_count": stats.chunk_count,
        "model_id": stats.model_id,
        "vector_dimension": stats.vector_dimension,
    }


def build_provenance(
    *,
    date: str,
    task: str,
    label: str,
    stats: IndexStats,
    db_path: Path,
    project_dir: Path = PROJECT_DIR,
) -> dict:
    """Assemble one result file's (or one result file's sub-entry's) full
    provenance block: index identity, measured duplication, the exact
    retrieval-code commit, and every tuning constant that could have
    affected the number -- all read at runtime. See module docstring for
    why each piece is necessary and none is sufficient alone.
    """
    return {
        "date": date,
        "task": task,
        "label": label,
        "index_stats": index_stats_dict(stats),
        "duplication_check": duplication_check(db_path),
        "git": git_info(project_dir),
        "tuning_constants": tuning_constants(),
    }
