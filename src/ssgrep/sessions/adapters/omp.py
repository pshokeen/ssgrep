"""omp session JSONL discovery and normalization.

omp shares the Pi session format: a `session` header with id/cwd/model, then
append-only entry records (`message`, `model_change`, `title_change`, `custom`,
`custom_message`, ...).  The transcript signal and normalization rules are
identical, so this module reuses the Pi adapter's parsing and only changes
discovery identity (name, runtime, root, and environment overrides).

One difference: omp writes a `title` record *before* the `session` header
(often with an empty auto-title), while Pi always starts with the `session`
header.  Discovery therefore skips leading `title` records when locating the
header; reads already tolerate them via the shared title normalization.
"""

from __future__ import annotations

from ssgrep.sessions.adapters.pi import (
    PiAdapter,
    _inspect as _pi_inspect,
    _SourceInfo,
)


def _inspect(records_: tuple[dict, ...], runtime: str) -> _SourceInfo | None:
    """Collect discovery metadata, tolerating omp's leading title record."""
    info = _pi_inspect(records_, runtime)
    if info is not None or not records_:
        return info
    rest = 0
    while rest < len(records_) and records_[rest].get("type") == "title":
        rest += 1
    if rest == 0:
        return None
    return _pi_inspect(records_[rest:], runtime)


class OmpAdapter(PiAdapter):
    """Discover omp sessions (Pi-format JSONL under ``~/.omp/agent/sessions``)."""

    name = "omp"
    runtime = "omp"
    default_relative_root = (".omp", "agent", "sessions")
    session_env_vars = ("SSGREP_OMP_SESSIONS_DIR", "OMP_SESSIONS_DIR")
    agent_env_vars = ("OMP_AGENT_DIR",)

    def _inspect_source(self, records_: tuple[dict, ...], *, runtime: str) -> _SourceInfo | None:
        return _inspect(records_, runtime)
