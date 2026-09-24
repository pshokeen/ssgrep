"""Tests for the runtime adapter registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.sessions.adapters import registry
from ssgrep.sessions.adapters.base import (
    ReadResult,
    SourceFingerprint,
    TranscriptSource,
)
from ssgrep.sessions.adapters.codex import CodexAdapter
from ssgrep.sessions.adapters.native import NativeAdapter
from ssgrep.sessions.adapters.opencode import OpenCodeAdapter
from ssgrep.sessions.adapters.pi import PiAdapter, PrimeAgentAdapter
from ssgrep.utilities.types import SessionFile


def test_adapters_include_all_runtimes() -> None:
    names = [adapter.name for adapter in registry._adapters()]
    assert names == ["native", "opencode", "codex", "pi", "prime-agent"]


def test_discover_sorts_and_federates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str | None, bool]] = []

    def fake_discover(self, *, scope=None, no_subagents=False):
        seen.append((self.name, scope, no_subagents))
        return [source(self.name if self.name != "native" else "claude", adapter=self.name)]

    monkeypatch.setattr(NativeAdapter, "discover", fake_discover)
    monkeypatch.setattr(OpenCodeAdapter, "discover", fake_discover)
    monkeypatch.setattr(CodexAdapter, "discover", fake_discover)
    monkeypatch.setattr(PiAdapter, "discover", fake_discover)
    monkeypatch.setattr(PrimeAgentAdapter, "discover", fake_discover)

    sources = registry.discover_sources(scope="/scope", no_subagents=True)

    assert [source.session.runtime for source in sources] == [
        "claude",
        "codex",
        "opencode",
        "pi",
        "prime-agent",
    ]
    assert seen == [
        ("native", "/scope", True),
        ("opencode", "/scope", True),
        ("codex", "/scope", True),
        ("pi", "/scope", True),
        ("prime-agent", "/scope", True),
    ]


def test_read_source_dispatches_by_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_read(self, source):
        called.append(self.name)
        return ReadResult(({"type": "user"},))

    monkeypatch.setattr(NativeAdapter, "read", fake_read)
    monkeypatch.setattr(OpenCodeAdapter, "read", fake_read)
    monkeypatch.setattr(CodexAdapter, "read", fake_read)
    monkeypatch.setattr(PiAdapter, "read", fake_read)
    monkeypatch.setattr(PrimeAgentAdapter, "read", fake_read)
    for name in ("native", "opencode", "codex", "pi", "prime-agent"):
        assert registry.read_source(source(name, adapter=name)).records == ({"type": "user"},)
    assert called == ["native", "opencode", "codex", "pi", "prime-agent"]


def test_read_source_unknown_adapter_raises() -> None:
    with pytest.raises(KeyError, match="unknown transcript adapter"):
        registry.read_source(source("claude", adapter="alien"))


def test_source_counts_is_stable_and_sorted() -> None:
    items = [
        source("pi"),
        source("claude"),
        source("pi"),
        source("opencode"),
        source("claude"),
    ]
    assert registry.source_counts(items) == (
        ("claude", 2),
        ("opencode", 1),
        ("pi", 2),
    )


def source(runtime: str, *, adapter: str | None = None) -> TranscriptSource:
    path = Path(f"/tmp/{runtime}-session.jsonl")
    return TranscriptSource(
        adapter=adapter or runtime,
        key=f"{runtime}:key",
        session=SessionFile(
            path=path,
            session_id=f"{runtime}-session",
            is_main=True,
            runtime=runtime,
        ),
        fingerprint=SourceFingerprint(size=1, mtime=2.0, digest="abc"),
    )
