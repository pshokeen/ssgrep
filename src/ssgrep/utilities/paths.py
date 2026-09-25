"""Path normalization shared by discovery and explicit metadata scopes."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

_probe_lock = threading.Lock()
_probe_result: bool | None = None


def _run_probe(directory: str) -> bool:
    """Return whether differently-cased names address the same file."""
    fd, created = tempfile.mkstemp(prefix="ssgrep-CASEPROBE-", dir=directory)
    os.close(fd)
    try:
        head, name = os.path.split(created)
        other = os.path.join(head, name.replace("CASEPROBE", "caseprobe"))
        if other == created:  # pragma: no cover
            raise RuntimeError("case probe produced an identical name")
        return os.path.exists(other)
    finally:
        os.unlink(created)


def filesystem_is_case_insensitive() -> bool:
    """Probe and cache the local filesystem's case behavior."""
    global _probe_result
    with _probe_lock:
        if _probe_result is None:
            try:
                _probe_result = _run_probe(tempfile.gettempdir())
            except (OSError, RuntimeError):
                _probe_result = sys.platform in ("darwin", "win32")
        return _probe_result


def fold_case(text: str) -> str:
    """Apply local filesystem case rules to a path string."""
    normalized = os.path.normcase(text)
    return normalized.lower() if filesystem_is_case_insensitive() else normalized


def canonical(path: str | Path) -> Path:
    """Create a comparison identity without requiring the path to exist."""
    text = os.path.normpath(os.path.expanduser(os.fspath(path)))
    return Path(fold_case(text))


def resolve_live(path: str | Path) -> Path:
    """Expand and resolve a path that represents a live filesystem entry."""
    return Path(os.path.expanduser(os.fspath(path))).resolve()


def resolve_claude_dir() -> Path:
    """Return ``$CLAUDE_CONFIG_DIR`` or the default ``~/.claude`` directory."""
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    base = Path.home() / ".claude" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def _config_home() -> Path:
    """Return the XDG config home, or ``~/.config`` when unset."""
    raw = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path.home() / ".config" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def resolve_opencode_dir() -> Path:
    """Return OpenCode's global config directory (skills live beneath ``skills/``).

    Respects ``XDG_CONFIG_HOME`` and the ``OPENCODE_CONFIG_DIR`` override,
    defaulting to ``~/.config/opencode``.
    """
    raw = os.environ.get("OPENCODE_CONFIG_DIR", "").strip()
    return _config_home() / "opencode" if not raw else resolve_live(raw)


def resolve_pi_agent_dir() -> Path:
    """Return Pi's agent directory (skills live beneath ``skills/``).

    Respects the ``PI_CODING_AGENT_DIR`` override, defaulting to ``~/.pi/agent``.
    """
    raw = os.environ.get("PI_CODING_AGENT_DIR", "").strip()
    base = Path.home() / ".pi" / "agent" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def resolve_prime_agent_dir() -> Path:
    """Return Prime Agent's directory (skills live beneath ``skills/``).

    Respects the ``PRIME_AGENT_CODING_AGENT_DIR`` override, defaulting to
    ``~/.prime/agent``.
    """
    raw = os.environ.get("PRIME_AGENT_CODING_AGENT_DIR", "").strip()
    base = Path.home() / ".prime" / "agent" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def resolve_omp_agent_dir() -> Path:
    """Return omp's agent directory (skills live beneath ``skills/``).

    Respects the ``OMP_AGENT_DIR`` override, defaulting to ``~/.omp/agent``.
    """
    raw = os.environ.get("OMP_AGENT_DIR", "").strip()
    base = Path.home() / ".omp" / "agent" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def resolve_codex_dir() -> Path:
    """Return Codex's global config directory (sessions/skills live beneath it).

    Respects the ``CODEX_HOME`` override, defaulting to ``~/.codex``.
    """
    raw = os.environ.get("CODEX_HOME", "").strip()
    base = Path.home() / ".codex" if not raw else Path(os.path.expanduser(raw))
    return resolve_live(base)


def is_at_or_beneath(child: str | Path, ancestor: str | Path) -> bool:
    """Return whether child equals ancestor or is contained by it."""
    try:
        canonical(child).relative_to(canonical(ancestor))
    except ValueError:
        return False
    return True
