"""Frozen API contract for ssgrep.

Signatures only — bodies raise NotImplementedError. Every module in the
system codes against these functions. No CLI or framework types may appear
here; functions accept only primitive or plain-data arguments and return
only plain data structures.
"""

from __future__ import annotations

from pathlib import Path

from ssgrep import detail, indexer, observability
from ssgrep import search as search_module
from ssgrep.types import (
    EpisodeDetail,
    IndexStats,
    SearchFilters,
    SearchResponse,
)


def index(
    project_dir: Path,
    *,
    rebuild: bool = False,
    no_subagents: bool = False,
    quiet: bool = False,
    allow_shrink: bool = False,
    scope: str | None = None,
) -> IndexStats:
    """Build or update the local index for the current project's session transcripts.

    Args:
        project_dir: Path to the project directory to index.
        rebuild: If True, force a full rebuild instead of incremental update.
        no_subagents: If True, restrict indexing to main sessions only.
        quiet: If True, suppress progress output.
        allow_shrink: If True, permit a rebuild to replace the existing index
            with a drastically smaller one. Without it, such a rebuild raises
            RebuildWouldShrinkError and leaves the existing index untouched.
        scope: Discovery scope override (a path whose recorded cwds select
            which transcripts are indexed), decoupled from where the index
            lives. Defaults to project_dir. Persisted in the index; changing
            it forces a full rebuild. The moved-repo/org-rename remedy.

    Returns:
        IndexStats with counts, freshness, model binding, and degradation data.

    Raises:
        RebuildWouldShrinkError: If a rebuild would drop the indexed session
            count to zero or below half its current value and allow_shrink is
            not set. Nothing has been committed when this is raised.
    """
    return indexer.index(
        project_dir,
        rebuild=rebuild,
        no_subagents=no_subagents,
        quiet=quiet,
        allow_shrink=allow_shrink,
        scope=scope,
    )


def search(
    project_dir: Path,
    query: str,
    *,
    limit: int | None = None,
    token_budget: int | None = None,
    filters: SearchFilters | None = None,
) -> SearchResponse:
    """Search for query against the current project's index.

    Handles degenerate outcomes explicitly:
    - Raises EmptyQueryError if query is empty or whitespace-only.
    - Raises IndexNotFoundError if no index exists for the project.
    - Raises IndexNotReadyError if index exists but is not yet ready.
    - Returns SearchResponse with empty results if query matches nothing.

    Args:
        project_dir: Path to the project directory to search.
        query: The search query string.
        limit: Maximum number of results to return.
        token_budget: Token budget for response shaping; results exceeding
            this budget are omitted and counted in response.omitted_count.
        filters: Composable filters for narrowing results.

    Returns:
        SearchResponse with results list, omitted_count for token budgeting,
        and index_exists flag to distinguish no-index from no-matches.

    Raises:
        EmptyQueryError: If query is empty or whitespace-only.
        IndexNotFoundError: If no index exists.
        IndexNotReadyError: If index is not yet ready for queries.
    """
    return search_module.search(
        project_dir,
        query,
        limit=limit,
        token_budget=token_budget,
        filters=filters,
    )


def show(
    project_dir: Path,
    ref: str,
) -> EpisodeDetail | None:
    """Retrieve full context for a single episode identified by ref.

    The ref parameter identifies a single episode, which is the fundamental
    retrieval unit in the system (one user prompt plus every assistant message
    it triggered). For subagent-transcript sessions, a single episode may
    span the entire subagent's session; the "session" terminology in CLI/MCP
    specs refers to this case but the underlying retrieval unit is still the
    episode.

    Args:
        project_dir: Path to the project directory.
        ref: Reference to an episode from search results. Refs are stable
            across runs and encode the episode identity. A single episode
            always maps to exactly one session_id and one episode_id.

    Returns:
        EpisodeDetail with full context for the identified episode, or None
        if the episode is not found or ref is invalid.

    Raises:
        IndexNotFoundError: If no index exists for the project.
        IndexNotReadyError: If index exists but is not ready (corrupt or
            schema version mismatch).
    """
    return detail.show(project_dir, ref)


def status(project_dir: Path) -> IndexStats:
    """Get index observability data.

    Succeeds even when no index exists, reporting that absence as a
    normal state rather than an error. The index_exists field on the returned
    IndexStats distinguishes a missing or uninitialized index from one that
    exists but contains no data.

    Args:
        project_dir: Path to the project directory.

    Returns:
        IndexStats with counts, freshness, model binding, degradation data,
        and index_exists flag to signal whether an index is present.
    """
    return observability.status(project_dir)
