"""Idempotent MCP client registration for supported coding agents.

``ssgrep mcp install`` writes the stdio registration for ``ssgrep mcp`` into
every supported MCP client's own configuration file (or, for Claude Code, asks
its own CLI to register it). Re-runs are safe: an exact existing entry is
left untouched, a stale entry owned by ssgrep is replaced, and unrelated
client settings are never rewritten.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from ssgrep.utilities.paths import resolve_codex_dir, resolve_opencode_dir

#: Client names accepted by ``ssgrep mcp install``, in display order.
CLIENT_NAMES: tuple[str, ...] = ("claude", "cursor", "zed", "codex", "opencode")

_CODEX_HEADER = "[mcp_servers.ssgrep]"

#: (client, status, config location or None) — the installer's result unit.
#: The location is ``None`` for clients registered through their own CLI.
InstallResult = tuple[str, str, Path | None]


def ssgrep_path() -> str:
    """Return the absolute ssgrep executable for MCP clients to launch.

    GUI-spawned clients inherit a smaller PATH than an interactive shell, so
    the registration records an absolute path whenever one can be found.
    """
    beside_python = Path(sys.executable).with_name("ssgrep")
    if beside_python.exists():
        return str(beside_python)
    return shutil.which("ssgrep") or "ssgrep"


def _cursor_config_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def _zed_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "zed" / "settings.json"


def _codex_config_path() -> Path:
    return resolve_codex_dir() / "config.toml"


def _opencode_config_path() -> Path:
    """Return opencode's global config file, preferring one that already exists."""
    directory = resolve_opencode_dir()
    json_path = directory / "opencode.json"
    if json_path.exists():
        return json_path
    jsonc_path = directory / "opencode.jsonc"
    if jsonc_path.exists():
        return jsonc_path
    return json_path


def _merge_json(
    path: Path,
    apply: Callable[[dict[str, object]], str | None],
) -> str:
    """Merge an ssgrep entry into a JSON client config, preserving all else.

    ``apply`` mutates the parsed document and returns ``installed`` /
    ``updated``, ``None`` when the file already holds the exact entry (no
    write happens, so the user's formatting survives), or an ``error: ...``
    status (no write happens either). A file with comments is not valid
    strict JSON and is reported rather than guessed at.
    """
    try:
        data: dict[str, object] = {}
        if path.exists():
            text = path.read_text()
            if text.strip():
                loaded = json.loads(text)
                if not isinstance(loaded, dict):
                    return f"error: {path.name} does not contain a JSON object"
                data = loaded
        status = apply(data)
        if status is None or status.startswith("error:"):
            return status or "already_installed"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        return status
    except (OSError, json.JSONDecodeError) as error:
        return f"error: {error}"


def _apply_json_entry(
    data: dict[str, object],
    container_key: str,
    registration: dict[str, object],
) -> str | None:
    """Set ``data[container_key]["ssgrep"] = registration`` in a JSON document.

    Returns the write status, or ``None`` when the exact entry already exists.
    """
    container = data.get(container_key)
    if container is None:
        container = data[container_key] = {}
    if not isinstance(container, dict):
        return f"error: {container_key} is not a JSON object"
    existing = container.get("ssgrep")
    if existing == registration:
        return None
    container["ssgrep"] = registration
    return "updated" if existing is not None else "installed"


def _install_claude() -> InstallResult:
    """Register user-scoped through the claude CLI, which owns the file format."""
    binary = shutil.which("claude")
    if binary is None:
        return ("claude", "skipped: claude CLI not found", None)
    command = [binary, "mcp", "add", "--scope", "user", "ssgrep", "--", ssgrep_path(), "mcp"]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ("claude", f"error: {error}", None)
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode == 0:
        return ("claude", "installed", None)
    if "already" in output.lower():
        return ("claude", "already_installed", None)
    summary = output.splitlines()[0] if output else f"exit code {completed.returncode}"
    return ("claude", f"error: {summary}", None)


def _install_cursor() -> InstallResult:
    path = _cursor_config_path()
    registration: dict[str, object] = {"command": ssgrep_path(), "args": ["mcp"]}
    status = _merge_json(path, lambda data: _apply_json_entry(data, "mcpServers", registration))
    return ("cursor", status, path)


def _install_zed() -> InstallResult:
    path = _zed_config_path()
    registration: dict[str, object] = {"command": ssgrep_path(), "args": ["mcp"]}
    status = _merge_json(
        path, lambda data: _apply_json_entry(data, "context_servers", registration)
    )
    return ("zed", status, path)


def _codex_block(command: str) -> str:
    return f'{_CODEX_HEADER}\ncommand = "{command}"\nargs = ["mcp"]\n'


def _install_codex() -> InstallResult:
    """Merge a TOML section into ``~/.codex/config.toml`` by text surgery.

    Python ships only a TOML *reader*, and a re-serialization would discard
    the user's comments and layout. The ssgrep section is instead located by
    its exact header line and replaced in place, or appended at the end —
    both of which are valid TOML wherever they land.
    """
    path = _codex_config_path()
    block = _codex_block(ssgrep_path())
    try:
        text = path.read_text() if path.exists() else ""
        lines = text.splitlines(keepends=True)
        start = next((i for i, line in enumerate(lines) if line.strip() == _CODEX_HEADER), None)
        if start is not None:
            end = next(
                (j for j in range(start + 1, len(lines)) if lines[j].startswith("[")),
                len(lines),
            )
            if "".join(lines[start:end]) == block:
                return ("codex", "already_installed", path)
            lines[start:end] = [block]
            path.write_text("".join(lines))
            return ("codex", "updated", path)
        if lines:
            if not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append("\n")
        lines.append(block)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))
        return ("codex", "installed", path)
    except OSError as error:
        return ("codex", f"error: {error}", path)


def _install_opencode() -> InstallResult:
    path = _opencode_config_path()
    registration: dict[str, object] = {"type": "local", "command": [ssgrep_path(), "mcp"]}

    def apply(data: dict[str, object]) -> str | None:
        mcp = data.get("mcp")
        if mcp is None:
            mcp = data["mcp"] = {}
        if not isinstance(mcp, dict):
            return "error: mcp section is not a JSON object"
        # Older opencode configs listed servers directly under "mcp"; the
        # current schema nests them under "mcp.servers" with a list command.
        if "ssgrep" in mcp:
            servers = mcp.setdefault("servers", {})
            if not isinstance(servers, dict):
                return "error: mcp.servers is not a JSON object"
            if servers.get("ssgrep") == registration:
                del mcp["ssgrep"]
                return "updated"
            servers["ssgrep"] = registration
            del mcp["ssgrep"]
            return "updated"
        return _apply_json_entry(mcp, "servers", registration)

    return ("opencode", _merge_json(path, apply), path)


_INSTALLERS: dict[str, Callable[[], InstallResult]] = {
    "claude": _install_claude,
    "cursor": _install_cursor,
    "zed": _install_zed,
    "codex": _install_codex,
    "opencode": _install_opencode,
}


def install_mcp_registrations(
    clients: tuple[str, ...] | None = None,
) -> tuple[InstallResult, ...]:
    """Register ssgrep with each requested MCP client, independently.

    ``clients`` selects a subset of :data:`CLIENT_NAMES`; ``None`` (or an
    empty selection) registers every client. Each client is attempted
    independently, and failures become ``error: ...`` statuses rather than
    aborting the rest.
    """
    chosen = clients or CLIENT_NAMES
    unknown = [name for name in chosen if name not in CLIENT_NAMES]
    if unknown:
        raise ValueError(f"unknown MCP client(s): {', '.join(sorted(unknown))}")
    return tuple(_INSTALLERS[name]() for name in chosen)
