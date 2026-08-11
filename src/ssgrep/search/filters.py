"""Filtering and sorting for search results."""

from __future__ import annotations

from datetime import UTC, datetime

from ssgrep.search.rows import _ChunkHit, _EpisodeRow
from ssgrep.types import SearchFilters


def _normalize_dt(dt: datetime | None) -> datetime | None:
    """Convert to naive UTC so aware and naive datetimes compare safely."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _sortable_epoch(dt: datetime) -> float:
    """A monotonic numeric sort key that never invokes datetime.timestamp().

    .timestamp() on a naive datetime calls the platform's local mktime,
    which can raise OverflowError for very old dates (datetime.min is used
    below as the no-timestamp fallback). This is pure calendar/clock
    arithmetic instead, safe for any in-range datetime.
    """
    return (
        dt.toordinal() * 86400.0
        + dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + dt.microsecond / 1_000_000
    )


def _chunk_passes_content_type(hit: _ChunkHit, filters: SearchFilters) -> bool:
    if filters.content_type is None:
        return True
    return hit.content_type == filters.content_type.value


def _episode_passes_filters(episode: _EpisodeRow | None, filters: SearchFilters) -> bool:
    """Composable episode-level filters: date range, file path, branch.

    Filters compose to their intersection: every non-None filter must pass.
    """
    if (
        filters.date_from is None
        and filters.date_to is None
        and filters.file_path is None
        and filters.branch is None
    ):
        return True
    if episode is None:
        return False

    if filters.date_from is not None or filters.date_to is not None:
        ts = _normalize_dt(episode.timestamp)
        if ts is None:
            return False
        date_from = _normalize_dt(filters.date_from)
        date_to = _normalize_dt(filters.date_to)
        if date_from is not None and ts < date_from:
            return False
        if date_to is not None and ts > date_to:
            return False

    if filters.file_path is not None and not any(
        filters.file_path in f for f in episode.files_touched
    ):
        return False

    if filters.branch is not None and episode.git_branch != filters.branch:
        return False

    return True


def _filter_leg(
    ordered_chunk_ids: list[str],
    chunk_hits: dict[str, _ChunkHit],
    episode_rows: dict[str, _EpisodeRow],
    filters: SearchFilters,
) -> list[str]:
    """Apply all filters to one leg's raw ranked list, preserving order.

    Filtering before rank position is assigned (by the caller, via
    enumerate() over this function's output) keeps ranks contiguous within
    the eligible set, rather than leaving gaps where filtered-out chunks
    used to be.
    """
    kept = []
    for chunk_id in ordered_chunk_ids:
        hit = chunk_hits.get(chunk_id)
        if hit is None:
            continue
        if not _chunk_passes_content_type(hit, filters):
            continue
        if not _episode_passes_filters(episode_rows.get(hit.episode_id), filters):
            continue
        kept.append(chunk_id)
    return kept
