"""Observability for the global LanceDB index."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from ssgrep.indexing import embed
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore
from ssgrep.utilities.types import IndexNotReadyError, IndexStats


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _runtime_counts(repository: LanceStore) -> tuple[tuple[str, int], ...]:
    rows = repository.rows(SESSIONS_TABLE, columns=["runtime"], limit=1_000_000)
    counts = Counter(str(row.get("runtime") or "claude") for row in rows)
    return tuple(sorted(counts.items()))


def _assert_compatible(repository: LanceStore) -> None:
    """Raise when the existing index was built by an incompatible schema/model.

    Mirrors the search compatibility gate: a stale v4 index (768-d mpnet,
    schema 4) must be reported as needing a rebuild rather than silently
    reported with its old values. Absent metadata is tolerated (a fresh but
    not-yet-indexed v5 database has none); present-but-mismatched metadata is
    a hard error.
    """
    expected_meta = {
        "schema_version": str(repository.schema_version),
        "model_id": embed.MODEL_ID,
        "vector_dimension": str(embed.DIMENSION),
    }
    mismatched = any(
        (stored := repository.get_meta(key)) is not None and stored != value
        for key, value in expected_meta.items()
    )
    if not repository.schema_matches() or mismatched:
        raise IndexNotReadyError(
            "The global index schema or embedding model is incompatible; "
            "run `ssgrep index --rebuild`."
        )


def status() -> IndexStats:
    """Report the global index without creating it."""
    repository = LanceStore()
    if not repository.exists():
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
            schema_version=repository.schema_version,
            index_exists=False,
            data_dir=str(repository.root),
        )

    _assert_compatible(repository)

    return IndexStats(
        session_count=repository.count(SESSIONS_TABLE),
        episode_count=repository.count(EPISODES_TABLE),
        chunk_count=repository.count(CHUNKS_TABLE),
        index_size_bytes=_dir_size(repository.path),
        last_index_time=_parse_time(repository.get_meta("last_index_time")),
        last_optimize_time=_parse_time(repository.get_meta("last_optimize_time")),
        model_id=repository.get_meta("model_id") or embed.MODEL_ID,
        vector_dimension=int(repository.get_meta("vector_dimension") or embed.DIMENSION),
        skipped_records=int(repository.get_meta("skipped_records") or 0),
        malformed_records=int(repository.get_meta("malformed_records") or 0),
        archived_source_count=int(repository.get_meta("archived_source_count") or 0),
        schema_version=int(repository.get_meta("schema_version") or repository.schema_version),
        tombstoned_source_count=repository.count(SESSIONS_TABLE, "source_status = 'absent'"),
        tombstoned_chunk_count=repository.count(CHUNKS_TABLE, "source_status = 'absent'"),
        index_exists=True,
        data_dir=str(repository.root),
        runtime_counts=_runtime_counts(repository),
    )
