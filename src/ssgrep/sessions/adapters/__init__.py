"""Runtime adapters that normalize coding-agent transcripts for indexing."""

from ssgrep.sessions.adapters.registry import (
    discover_sources,
    read_source,
    source_counts,
    source_present,
)

__all__ = ["discover_sources", "read_source", "source_counts", "source_present"]
