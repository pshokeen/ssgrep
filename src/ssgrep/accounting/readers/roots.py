"""Locate raw runtime stores from environment overrides (no adapter imports).

Precedence mirrors the retrieval adapters so both read the same stores, but this
module only *locates* paths; it never reads, creates, or resolves symlinks.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

PI_SESSION_VARS = ("SSGREP_PI_SESSIONS_DIR", "PI_SESSION_DIR", "PI_CODING_AGENT_SESSION_DIR")
OMP_SESSION_VARS = ("SSGREP_OMP_SESSIONS_DIR", "OMP_SESSIONS_DIR")
PRIME_SESSION_VARS = (
    "SSGREP_PRIME_AGENT_SESSIONS_DIR",
    "PRIME_AGENT_SESSION_DIR",
    "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
)
CODEX_SESSION_VARS = ("SSGREP_CODEX_SESSIONS_DIR", "CODEX_SESSIONS_DIR")


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _value(env: Mapping[str, str], name: str) -> str:
    return env.get(name, "").strip()


def home_dir(env: Mapping[str, str] | None = None) -> Path:
    raw = _value(_env(env), "HOME")
    return Path(raw) if raw else Path.home()


def _expand(raw: str, env: Mapping[str, str]) -> Path:
    if raw == "~" or raw.startswith("~/"):
        return home_dir(env) / raw[2:]
    return Path(raw)


def _pi_style(
    env: Mapping[str, str] | None, session_vars: tuple[str, ...], agent_var: str, default: str
) -> Path:
    values = _env(env)
    for name in session_vars:
        if _value(values, name):
            return _expand(_value(values, name), values)
    if _value(values, agent_var):
        return _expand(_value(values, agent_var), values) / "sessions"
    return home_dir(values) / default


def claude_projects(env: Mapping[str, str] | None = None) -> Path:
    values = _env(env)
    raw = _value(values, "CLAUDE_CONFIG_DIR")
    base = _expand(raw, values) if raw else home_dir(values) / ".claude"
    return base / "projects"


def codex_sessions(env: Mapping[str, str] | None = None) -> Path:
    values = _env(env)
    for name in CODEX_SESSION_VARS:
        if _value(values, name):
            return _expand(_value(values, name), values)
    raw = _value(values, "CODEX_HOME")
    base = _expand(raw, values) if raw else home_dir(values) / ".codex"
    return base / "sessions"


def omp_sessions(env: Mapping[str, str] | None = None) -> Path:
    return _pi_style(env, OMP_SESSION_VARS, "OMP_AGENT_DIR", ".omp/agent/sessions")


def pi_sessions(env: Mapping[str, str] | None = None) -> Path:
    return _pi_style(env, PI_SESSION_VARS, "PI_CODING_AGENT_DIR", ".pi/agent/sessions")


def prime_sessions(env: Mapping[str, str] | None = None) -> Path:
    return _pi_style(
        env, PRIME_SESSION_VARS, "PRIME_AGENT_CODING_AGENT_DIR", ".prime/agent/sessions"
    )


def prime_artifacts(env: Mapping[str, str] | None = None) -> Path:
    """Prime keeps child session transcripts and lineage beside its sessions root."""
    return prime_sessions(env).parent / "session-artifacts"


def opencode_db(env: Mapping[str, str] | None = None) -> Path:
    values = _env(env)
    override = _value(values, "SSGREP_OPENCODE_DB")
    if override:
        return _expand(override, values)
    data_home = _value(values, "XDG_DATA_HOME")
    base = _expand(data_home, values) if data_home else home_dir(values) / ".local" / "share"
    return base / "opencode" / "opencode.db"
