"""Guards and post-steps around one catch-up cycle, kept off ``app.py``.

Compatibility and shrink guards run before the engine is started; availability
tombstoning, the discovery cwd cache and post-run compaction run after it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from ssgrep.store import (
    CHUNKS_TABLE,
    CWD_CACHE_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    LanceStore,
    quote,
)
from ssgrep.utilities.types import IndexNotReadyError, RebuildWouldShrinkError

logger = logging.getLogger(__name__)


def expected_meta(repository: LanceStore) -> dict[str, str]:
    from ssgrep.indexing.embed import DIMENSION, MODEL_ID, MODEL_REVISION

    return {
        "schema_version": str(repository.schema_version),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "vector_dimension": str(DIMENSION),
    }


def assert_compatible(repository: LanceStore) -> bool:
    """Gate an incremental run; return whether it resumes an unfinished build.

    The schema/model metadata is only written when a run completes. A first
    build or ``--rebuild`` that was interrupted or failed, and a data dir with
    an empty ``lancedb``, therefore carry none of it: that is precisely the
    state ``search`` tells users to leave with a plain ``ssgrep index``, and
    there is nothing to be incompatible with while the tables that exist match
    the schema. Everything else must match the schema and model this build
    writes, or it needs ``--rebuild``.
    """
    expected = expected_meta(repository)
    state = repository.get_meta("index_state")
    unfinished = state in {None, "building"} and all(
        repository.get_meta(key) is None for key in expected
    )
    if unfinished and repository.schema_matches(allow_missing=True):
        return True
    compatible = (
        repository.schema_matches()
        and state in {"ready", "building"}
        and all(repository.get_meta(key) == value for key, value in expected.items())
    )
    if compatible:
        return False
    raise IndexNotReadyError(
        "The global index schema or embedding model changed; run `ssgrep index --rebuild`."
    )


def guard_shrink(repository: LanceStore, count: int, *, allow_shrink: bool) -> None:
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


def mark_available(repository: LanceStore, session_id: str) -> None:
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


def reconcile_absent(repository: LanceStore, discovered: set[str]) -> None:
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


def compact_after_reconcile(repository: LanceStore) -> str | None:
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


def write_cwd_cache(repository: LanceStore, fresh: list) -> None:
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
