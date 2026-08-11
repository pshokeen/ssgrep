"""Database fetch layer for chunks and episodes."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TypeVar

from ssgrep.search.rows import _ChunkHit, _EpisodeRow

_SQL_IN_BATCH_SIZE = 500

_T = TypeVar("_T")


def _batched(items: list[_T], size: int) -> list[list[_T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _split_or_empty(value: str | None) -> tuple[str, ...]:
    return tuple(value.split("\n")) if value else ()


def _fetch_chunk_ids_by_vec_row(conn: sqlite3.Connection, vec_rows: list[int]) -> dict[int, str]:
    result: dict[int, str] = {}
    for batch in _batched(vec_rows, _SQL_IN_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT vec_row, chunk_id FROM chunks WHERE vec_row IN ({placeholders})", batch
        ).fetchall()
        result.update({row[0]: row[1] for row in rows})
    return result


def _fetch_chunk_hits(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, _ChunkHit]:
    result: dict[str, _ChunkHit] = {}
    for batch in _batched(chunk_ids, _SQL_IN_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            "SELECT chunk_id, episode_id, content_type, text, source_status "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            result[row[0]] = _ChunkHit(
                chunk_id=row[0],
                episode_id=row[1],
                content_type=row[2],
                text=row[3],
                source_status=row[4],
            )
    return result


def _fetch_episode_rows(conn: sqlite3.Connection, episode_ids: list[str]) -> dict[str, _EpisodeRow]:
    result: dict[str, _EpisodeRow] = {}
    for batch in _batched(episode_ids, _SQL_IN_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            "SELECT episode_id, session_id, title, timestamp, git_branch, files_touched, "
            "is_subagent, agent_name, agent_description, parent_session_id, source_status "
            f"FROM episodes WHERE episode_id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            result[row[0]] = _EpisodeRow(
                episode_id=row[0],
                session_id=row[1],
                title=row[2],
                timestamp=_parse_timestamp(row[3]),
                git_branch=row[4],
                files_touched=_split_or_empty(row[5]),
                is_subagent=bool(row[6]),
                agent_name=row[7],
                agent_description=row[8],
                parent_session_id=row[9],
                source_status=row[10],
            )
    return result
