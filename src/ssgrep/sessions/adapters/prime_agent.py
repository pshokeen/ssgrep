"""Prime Agent JSONL transcript discovery.

Prime Agent persists a variant of Pi's append-only session format, so this
adapter reuses the shared Pi parsing and only changes discovery identity
(name, runtime, root, and environment overrides) plus the extra
``session-artifacts`` root and its exclusion from main-session detection.
"""

from __future__ import annotations

from pathlib import Path

from ssgrep.sessions.adapters.pi import _PiRuntimeAdapter, _SourceInfo


class PrimeAgentAdapter(_PiRuntimeAdapter):
    """Discover Prime Agent sessions."""

    name = "prime-agent"
    runtime = "prime-agent"
    default_relative_root = (".prime", "agent", "sessions")
    session_env_vars = (
        "SSGREP_PRIME_AGENT_SESSIONS_DIR",
        "PRIME_AGENT_SESSION_DIR",
        "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
    )
    agent_env_vars = ("PRIME_AGENT_CODING_AGENT_DIR",)

    def _discovery_roots(self) -> tuple[Path, ...]:
        return (self.root, self.root.parent / "session-artifacts")

    def _is_main(self, path: Path, info: _SourceInfo) -> bool:
        artifact_root = self.root.parent / "session-artifacts"
        try:
            path.relative_to(artifact_root)
        except ValueError:
            return info.is_main
        return False
