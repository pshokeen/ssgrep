"""Framework-free service API shared by the CLI and MCP server."""

from __future__ import annotations

from ssgrep import search as search_module
from ssgrep.indexing import indexer
from ssgrep.search import detail
from ssgrep.services import observability
from ssgrep.sessions import adapters as transcript_adapters
from ssgrep.utilities.types import EpisodeDetail, IndexStats, SearchFilters, SearchResponse


def index(
    *,
    rebuild: bool = False,
    no_subagents: bool = False,
    allow_shrink: bool = False,
    scope: str | None = None,
    quiet: bool = False,
    live: bool = False,
    full_reprocess: bool = False,
) -> IndexStats:
    """Reconcile the global transcript corpus."""
    return indexer.index(
        rebuild=rebuild,
        no_subagents=no_subagents,
        allow_shrink=allow_shrink,
        scope=scope,
        quiet=quiet,
        live=live,
        full_reprocess=full_reprocess,
    )


def sources(*, scope: str | None = None, no_subagents: bool = False) -> tuple[tuple[str, int], ...]:
    """Return an automatic-discovery census without creating the index."""
    discovered = transcript_adapters.discover_sources(
        scope=scope,
        no_subagents=no_subagents,
    )
    return transcript_adapters.source_counts(discovered)


def search(
    query: str,
    *,
    limit: int | None = None,
    token_budget: int | None = None,
    filters: SearchFilters | None = None,
    where: str | None = None,
) -> SearchResponse:
    """Search every project, optionally using a Lance metadata prefilter."""
    return search_module.search(
        query,
        limit=limit,
        token_budget=token_budget,
        filters=filters,
        where=where,
    )


def show(ref: str) -> EpisodeDetail | None:
    return detail.show(ref)


def status() -> IndexStats:
    """Return global index state without creating storage."""
    return observability.status()
