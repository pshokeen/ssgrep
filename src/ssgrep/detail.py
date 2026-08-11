"""Episode drill-down with full context retrieval.

The show() function retrieves full transcript content for a single episode
identified by ref, with bounded output and explicit truncation. A ref is
stable across runs and identifies exactly one episode: the retrieval unit
of one user prompt plus every assistant message it triggered.

Refs are episode_ids encoded as {session_id}:ep:{index}. Unknown or
malformed refs return None, never raising.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ssgrep import search, staleness, store
from ssgrep.types import EpisodeDetail, IndexNotFoundError, IndexNotReadyError

INDEX_DIRNAME = ".ssgrep"

# Hard bounds on episode content to prevent unbounded output.
# Episodes are stored in full; these caps apply only at retrieval.
MAX_PROMPT_CHARS = 50_000
MAX_RESPONSE_CHARS = 150_000

# Truncation marker to signal when content has been elided
TRUNCATION_MARKER = "\n[... truncated, showing first {max_chars} characters ...]"


def _normalize_dt(dt: datetime | None) -> datetime | None:
    """Convert to naive UTC so aware and naive datetimes compare safely."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _split_or_empty(value: str | None) -> tuple[str, ...]:
    return tuple(value.split("\n")) if value else ()


def _validate_episode_id(ref: str) -> str | None:
    """Validate ref format and extract session_id if valid.

    A valid ref is {session_id}:ep:{index} where index is a non-negative
    integer. Returns the session_id if valid, None if malformed.
    """
    # Pattern: anything up to :ep:, then digits
    pattern = r"^(.+):ep:(\d+)$"
    match = re.match(pattern, ref)
    if match:
        return match.group(1)
    return None


def _apply_bound(text: str, max_chars: int) -> str:
    """Truncate text if needed and append truncation marker when elided."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_MARKER.format(max_chars=max_chars)


def _index_paths(project_dir: Path) -> tuple[Path, Path]:
    """Resolve the current generation's db and vector paths (read-only).

    Constructing GenerationalStore only reads an existing `.manifest` if
    present; it never creates the index directory or any file.
    """
    index_dir = project_dir / INDEX_DIRNAME
    generation = store.GenerationalStore(index_dir)
    return generation.get_index_path(), generation.get_vector_path()


def _open_ready_index(db_path: Path, vec_path: Path) -> sqlite3.Connection:
    """Open the index, raising if it's absent or unsafe to query.

    Raises:
        IndexNotFoundError: no index.db at the resolved path.
        IndexNotReadyError: index.db exists but isn't a valid ssgrep index
            or its schema_version doesn't match.
    """
    if not db_path.exists():
        raise IndexNotFoundError(f"No index found at {db_path}. Run `ssgrep index` first.")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        schema_version = store.get_meta(conn, "schema_version")
    except sqlite3.Error as exc:
        conn.close()
        raise IndexNotReadyError(
            f"Index at {db_path} is corrupt or unreadable. "
            f"Run `ssgrep index --rebuild` to recreate it."
        ) from exc

    if schema_version != str(store.SCHEMA_VERSION):
        conn.close()
        raise IndexNotReadyError(
            f"Index schema version {schema_version!r} does not match the "
            f"expected {store.SCHEMA_VERSION!r}; run `ssgrep index --rebuild`."
        )

    return conn


def show(
    project_dir: Path,
    ref: str,
) -> EpisodeDetail | None:
    """Retrieve full context for a single episode identified by ref.

    The ref parameter identifies a single episode, which is the fundamental
    retrieval unit in the system (one user prompt plus every assistant message
    it triggered). Refs are stable across runs and encode the episode identity
    as {session_id}:ep:{index}. A single episode always maps to exactly one
    session_id and one episode_id.

    Args:
        project_dir: Path to the project directory.
        ref: Reference to an episode from search results. Refs are stable
            across runs and encode the episode identity.

    Returns:
        EpisodeDetail with full context for the identified episode, or None
        if the episode is not found or ref is invalid.

    Raises:
        IndexNotFoundError: If no index exists at the resolved path.
        IndexNotReadyError: If index exists but is not ready (corrupt or
            schema version mismatch).
    """
    # Check index first: missing/corrupt index should raise before ref validation.
    # This ensures "no index" errors (exit 4) take precedence over malformed ref (exit 3).
    db_path, vec_path = _index_paths(project_dir)
    conn = _open_ready_index(db_path, vec_path)

    try:
        # Validate ref format after confirming index exists; return None for malformed refs.
        session_id = _validate_episode_id(ref)
        if session_id is None:
            return None

        # Query the episodes table for the exact episode_id, including stored canonical text.
        ep_row = conn.execute(
            "SELECT episode_id, session_id, title, timestamp, git_branch, cwd, "
            "files_touched, tool_names, is_subagent, agent_type, agent_name, "
            "agent_description, parent_session_id, prompt_text, response_text "
            "FROM episodes WHERE episode_id = ?",
            (ref,),
        ).fetchone()

        if ep_row is None:
            return None

        # Read prompt and response text directly from the episodes table.
        prompt_text = ep_row[13] or ""
        response_text = ep_row[14] or ""

        # Apply bounds to both fields with explicit truncation markers,
        # and track whether truncation occurred.
        original_prompt_len = len(prompt_text)
        prompt_text = _apply_bound(prompt_text, MAX_PROMPT_CHARS)
        prompt_truncated = len(prompt_text) < original_prompt_len

        original_response_len = len(response_text)
        response_text = _apply_bound(response_text, MAX_RESPONSE_CHARS)
        response_truncated = len(response_text) < original_response_len

        # Detect staleness (stat-only, no parsing), mirroring search()/status(): a
        # user drilling into full episode content is just as liable to be misled
        # by a stale index as one reading search results.
        staleness_report = search.staleness_summary(project_dir)
        is_stale = staleness.is_index_stale(staleness_report)
        stale_count_value = staleness.stale_count(staleness_report)

        return EpisodeDetail(
            episode_id=ep_row[0],
            session_id=ep_row[1],
            title=ep_row[2],
            timestamp=_parse_timestamp(ep_row[3]),
            git_branch=ep_row[4],
            cwd=ep_row[5],
            prompt_text=prompt_text,
            response_text=response_text,
            files_touched=_split_or_empty(ep_row[6]),
            tool_names=_split_or_empty(ep_row[7]),
            is_subagent=bool(ep_row[8]),
            agent_type=ep_row[9],
            agent_name=ep_row[10],
            agent_description=ep_row[11],
            parent_session_id=ep_row[12],
            stale=is_stale,
            stale_count=stale_count_value,
            prompt_truncated=prompt_truncated,
            response_truncated=response_truncated,
        )
    finally:
        conn.close()
