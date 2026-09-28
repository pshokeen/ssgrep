"""Claude Code, authored-note, and configured native JSONL discovery."""

from __future__ import annotations

import dataclasses

from ssgrep.sessions import discovery, discovery_roots, notes
from ssgrep.sessions.adapters.base import ReadResult, TranscriptSource, jsonl_source, read_jsonl


class NativeAdapter:
    """Normalize the native Claude record-pair JSONL format."""

    name = "native"

    def discover(
        self, *, scope: str | None = None, no_subagents: bool = False
    ) -> list[TranscriptSource]:
        sessions = [
            *(
                dataclasses.replace(item, runtime="claude")
                for item in discovery.discover_sessions(scope=scope, no_subagents=no_subagents)
            ),
            *(dataclasses.replace(item, runtime="ssgrep") for item in notes.discover_notes()),
            *(
                dataclasses.replace(item, runtime="native")
                for item in discovery_roots.discover_external()
            ),
        ]
        sources: list[TranscriptSource] = []
        for session in sessions:
            source = jsonl_source(
                session,
                adapter=self.name,
                cache_cwds=session.runtime == "claude",
            )
            if source is not None:
                sources.append(source)
        return sources

    def read(self, source: TranscriptSource) -> ReadResult:
        return read_jsonl(source.session.path)

    def present(self, source: TranscriptSource) -> bool:
        return source.session.path.exists()
