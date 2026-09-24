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
- which storage/encoding engine versions produced the index -- `library_versions`

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

import subprocess
from pathlib import Path

from ssgrep.utilities.types import IndexStats

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
    """Measure duplicate raw chunks and subagent share in one Lance database."""
    import lancedb

    database = lancedb.connect(str(db_path))
    chunks = database.open_table("chunks").search().select(["text", "is_subagent"]).to_list()
    total = len(chunks)
    distinct = len({str(row["text"]) for row in chunks})
    subagent = sum(bool(row.get("is_subagent")) for row in chunks)
    return {
        "total_chunks": total,
        "distinct_chunk_texts": distinct,
        "duplication_factor": (total / distinct) if distinct else None,
        "subagent_chunks": subagent,
        "subagent_chunk_share": (subagent / total) if total else None,
    }


def tuning_constants() -> dict:
    """Retrieval-affecting production values read from their owning modules.

    The env-knob entries (``SSGREP_*``) are the EFFECTIVE values the modules
    resolve at call time — clamped/truthy-decided exactly as production
    search/indexing would apply them — so a result file records what the code
    held, not what its last commit said (module docstring).
    """
    from ssgrep import search as search_module
    from ssgrep.indexing import chunker, embed
    from ssgrep.store import NPROBES, REFINE_FACTOR, _env_int, _pq_bits_from_env

    return {
        "MULTI_CHUNK_EVIDENCE_WEIGHT": search_module.MULTI_CHUNK_EVIDENCE_WEIGHT,
        "MULTI_CHUNK_EVIDENCE_COUNT": search_module.MULTI_CHUNK_EVIDENCE_COUNT,
        "CHUNK_TOKEN_BUDGET": chunker.CHUNK_TOKEN_BUDGET,
        "CHUNK_TOKEN_OVERLAP": chunker.CHUNK_TOKEN_OVERLAP,
        "embed_model_id": embed.MODEL_ID,
        "embed_model_revision": embed.MODEL_REVISION,
        "embed_dimension": embed.DIMENSION,
        "SSGREP_POOL_FACTOR": embed.resolve_pool_factor(),
        "SSGREP_CHUNK_OVERLAP": chunker._resolved_overlap(),
        "SSGREP_PQ_BITS": _pq_bits_from_env(),
        "SSGREP_REFINE_FACTOR": _env_int("SSGREP_REFINE_FACTOR", REFINE_FACTOR, 1, 20),
        "SSGREP_NPROBES": _env_int("SSGREP_NPROBES", NPROBES, 1, 512),
        "SSGREP_OVERSAMPLE": search_module._resolved_oversample_factor(),
        "SSGREP_TWO_STAGE": search_module._two_stage_enabled(),
        "SSGREP_TWO_STAGE_CANDIDATES": search_module._two_stage_candidate_floor(),
    }


def library_versions() -> dict[str, str | None]:
    """Installed versions of the storage/encoding libraries behind the index.

    Read from package metadata (not attribute sniffing) so the record works
    regardless of whether a library exposes ``__version__``. Index size and
    latency numbers are only comparable across runs with the same engine
    versions, so every result file pins them.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for distribution in (
        "lancedb",
        "pylate",
        "ir-measures",
        "pytrec-eval-terrier",
    ):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


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
        "library_versions": library_versions(),
    }
