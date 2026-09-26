"""Built-in runtime adapter registry."""

from __future__ import annotations

from collections import Counter

from ssgrep.sessions.adapters.base import ReadResult, TranscriptAdapter, TranscriptSource
from ssgrep.sessions.adapters.codex import CodexAdapter
from ssgrep.sessions.adapters.native import NativeAdapter
from ssgrep.sessions.adapters.omp import OmpAdapter
from ssgrep.sessions.adapters.opencode import OpenCodeAdapter
from ssgrep.sessions.adapters.pi import PiAdapter
from ssgrep.sessions.adapters.prime_agent import PrimeAgentAdapter


def _adapters() -> tuple[TranscriptAdapter, ...]:
    return (
        NativeAdapter(),
        OpenCodeAdapter(),
        CodexAdapter(),
        PiAdapter(),
        PrimeAgentAdapter(),
        OmpAdapter(),
    )


def adapter_names() -> tuple[str, ...]:
    """Names of every built-in runtime adapter, in registry order.

    Exposed so documentation and guidance tests can assert against the real
    registry instead of a hand-maintained copy of it.
    """
    return tuple(adapter.name for adapter in _adapters())


def discover_sources(
    *, scope: str | None = None, no_subagents: bool = False
) -> list[TranscriptSource]:
    """Discover sessions from every locally installed supported runtime."""
    sources = [
        source
        for adapter in _adapters()
        for source in adapter.discover(scope=scope, no_subagents=no_subagents)
    ]
    return sorted(sources, key=lambda source: (source.session.runtime, source.key))


def read_source(source: TranscriptSource) -> ReadResult:
    """Dispatch normalization to the adapter that discovered ``source``."""
    for adapter in _adapters():
        if adapter.name == source.adapter:
            return adapter.read(source)
    raise KeyError(f"unknown transcript adapter: {source.adapter}")


def source_counts(sources: list[TranscriptSource]) -> tuple[tuple[str, int], ...]:
    """Stable runtime census for status and setup output."""
    counts = Counter(source.session.runtime for source in sources)
    return tuple(sorted(counts.items()))
