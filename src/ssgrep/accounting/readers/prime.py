"""Prime agent: ``.../agent/sessions`` plus the sibling ``session-artifacts`` root.

Child transcripts and ``rlm-subagents.jsonl`` lineage live under session-artifacts;
excluding them would silently drop most subagent tool results from coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.pi_family import PiFamilyReader


class PrimeReader(PiFamilyReader):
    runtime: ClassVar[str] = "prime-agent"

    def roots(self) -> list[Path]:
        return [roots.prime_sessions(self.env), roots.prime_artifacts(self.env)]

    def role_for(self, path: Path) -> str:
        artifacts = roots.prime_artifacts(self.env)
        try:
            path.relative_to(artifacts)
        except ValueError:
            return "main"
        return "lineage" if path.name == "rlm-subagents.jsonl" else "artifact"
