"""Idempotent MCP client registration for supported coding agents.

``ssgrep mcp install`` writes the stdio registration for ``ssgrep mcp`` into
every supported MCP client's own configuration file (or, for Claude Code, asks
its own CLI to register it). The launch command is ``uvx`` (pinned to the
installed version) only when it can actually resolve this same ssgrep — see
:func:`launch_command` — and the absolute path of the installed binary
otherwise. Re-runs are safe and always converge on the current registration:
every client leaves an exact existing entry untouched and replaces a stale one
owned by ssgrep without touching unrelated settings. For Claude Code the
current entry is read from Claude's own user config (``claude mcp get``
health-checks, i.e. launches, the server and has no machine-readable output);
a stale entry is replaced with ``remove`` + ``add`` and restored if the ``add``
fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from ssgrep.utilities.paths import (
    resolve_codex_dir,
    resolve_omp_agent_dir,
    resolve_opencode_dir,
)

#: Client names accepted by ``ssgrep mcp install``, in display order.
CLIENT_NAMES: tuple[str, ...] = ("claude", "cursor", "zed", "codex", "opencode", "omp")

#: Overrides launch_command()'s uvx-vs-path decision; unset/empty means "auto".
LAUNCHER_ENV_VAR = "SSGREP_MCP_LAUNCHER"
_LAUNCHER_MODES = ("auto", "uvx", "path")

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


def _launcher_mode() -> str:
    """Return the validated ``SSGREP_MCP_LAUNCHER`` mode (``auto`` when unset).

    Raises ``ValueError`` for anything other than ``auto``, ``uvx``, or
    ``path`` (case-insensitive, surrounding whitespace ignored).
    """
    raw = os.environ.get(LAUNCHER_ENV_VAR, "")
    mode = raw.strip().lower() or "auto"
    if mode not in _LAUNCHER_MODES:
        raise ValueError(
            f"{LAUNCHER_ENV_VAR}={raw!r} is invalid; expected one of: {', '.join(_LAUNCHER_MODES)}"
        )
    return mode


def _installed_from_index() -> bool:
    """Return whether the running ssgrep distribution came from a package index.

    PEP 610 records a ``direct_url.json`` file in a distribution's metadata
    whenever it was installed from anything other than an index — a VCS URL,
    a local path, a wheel file, or an editable install. Its absence means the
    package was resolved from *some* configured index (PyPI by default, but
    also a private or mirrored index), so a ``uvx ssgrep`` invocation that
    resolves against the same index can find the same release. This reads
    only local metadata already written at install time; it never touches
    the network.
    """
    try:
        dist = distribution("ssgrep")
    except PackageNotFoundError:
        return False
    return dist.read_text("direct_url.json") is None


def launch_command() -> tuple[str, list[str]]:
    """Return ``(command, args)`` that a client should use to start the server.

    Controlled by ``SSGREP_MCP_LAUNCHER`` (see :func:`_launcher_mode`):

    - ``path`` always uses the absolute path from :func:`ssgrep_path`.
    - ``uvx`` always uses ``uvx``, pinned to the installed version when it can
      be resolved (falling back to an unpinned ``uvx ssgrep`` otherwise) —
      it's the user's explicit choice, so no other check applies.
    - ``auto`` (the default) uses ``uvx`` only when it is on PATH *and* the
      running ssgrep was installed from a package index *and* its version can
      be resolved; otherwise it falls back to the absolute path. Without all
      three, ``uvx ssgrep`` may not resolve to the same code this process is
      running (see issue #4), so the absolute path is the only safe default.

    The uvx form pins the installed version so a client can never resolve a
    *different* ssgrep than the one that wrote the registration — but the pin
    is fixed at write time, so re-run ``ssgrep mcp install`` after upgrading
    ssgrep to keep it current.
    """
    mode = _launcher_mode()
    if mode == "path":
        return (ssgrep_path(), ["mcp"])

    try:
        installed_version: str | None = version("ssgrep")
    except PackageNotFoundError:
        installed_version = None

    if mode == "uvx":
        args = [f"ssgrep@{installed_version}", "mcp"] if installed_version else ["ssgrep", "mcp"]
        return ("uvx", args)

    if installed_version and shutil.which("uvx") and _installed_from_index():
        return ("uvx", [f"ssgrep@{installed_version}", "mcp"])
    return (ssgrep_path(), ["mcp"])


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
    status (no write happens either). A file with comments (Zed's
    ``settings.json`` commonly has them) is not strict JSON, and rewriting it
    would destroy them, so it is reported and left untouched.
    """
    try:
        data: dict[str, object] = {}
        if path.exists():
            text = path.read_text()
            if text.strip():
                try:
                    loaded = json.loads(text)
                except json.JSONDecodeError as error:
                    return (
                        f"error: {path.name} is not strict JSON ({error.msg} at line "
                        f"{error.lineno} column {error.colno}); ssgrep cannot safely edit "
                        "commented or malformed settings, so add the ssgrep entry by hand"
                    )
                if not isinstance(loaded, dict):
                    return f"error: {path.name} does not contain a JSON object"
                data = loaded
        status = apply(data)
        if status is None or status.startswith("error:"):
            return status or "already_installed"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        return status
    except OSError as error:
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


def _claude(binary: str, *argv: str) -> tuple[int, str]:
    """Run one ``claude`` subcommand; return ``(returncode, combined output)``.

    A claude that cannot be run at all reports return code ``-1`` with the
    reason as the output, so callers handle one failure shape.
    """
    try:
        completed = subprocess.run(
            [binary, *argv], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return (-1, str(error))
    return (completed.returncode, (completed.stdout + completed.stderr).strip())


def _claude_user_registration() -> dict[str, object] | None:
    """Return Claude's current user-scope ``ssgrep`` entry, or ``None`` if unknown.

    ``None`` means absent *or* unreadable. Claude keeps user-scope servers under
    ``mcpServers`` in ``$CLAUDE_CONFIG_DIR/.claude.json`` (``~/.claude.json``
    when unset); ``claude mcp get`` is no substitute — it launches the server
    as a health check and prints only human-readable text.
    """
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    path = (Path(raw).expanduser() if raw else Path.home()) / ".claude.json"
    try:
        servers = json.loads(path.read_text()).get("mcpServers")
    except (OSError, ValueError, AttributeError):
        return None
    entry = servers.get("ssgrep") if isinstance(servers, dict) else None
    return entry if isinstance(entry, dict) else None


def _claude_summary(output: str, code: int) -> str:
    """First line of a failed claude call's output, or its exit code when silent."""
    return output.splitlines()[0] if output else f"exit code {code}"


def _install_claude() -> InstallResult:
    """Register user-scoped through the claude CLI, replacing only a stale entry.

    ``claude mcp add`` has no update-in-place mode, so a stale command (an old
    absolute path, an old pinned version, or the unpinned ``uvx ssgrep``) is
    removed and re-added. An entry that already launches the right command is
    left alone, and if the re-add fails the previous entry is put back so a
    failed install never costs the user a working registration.
    """
    binary = shutil.which("claude")
    if binary is None:
        return ("claude", "skipped: claude CLI not found", None)
    launcher, args = launch_command()
    previous = _claude_user_registration()
    add = ("mcp", "add", "--scope", "user", "ssgrep", "--", launcher, *args)
    if previous is None:
        code, output = _claude(binary, *add)
        if code == 0:
            return ("claude", "installed", None)
        if "already" not in output.lower():
            return ("claude", f"error: {_claude_summary(output, code)}", None)
        # ``add`` reports an entry the config read could not see: replace it.
    elif (
        previous.get("command") == launcher
        and previous.get("args") == args
        and previous.get("type", "stdio") == "stdio"
    ):
        return ("claude", "already_installed", None)
    removed = _claude(binary, "mcp", "remove", "--scope", "user", "ssgrep")[0] == 0
    code, output = _claude(binary, *add)
    if code == 0:
        return ("claude", "updated", None)
    summary = _claude_summary(output, code)
    if not removed:
        return ("claude", f"error: {summary}", None)
    if previous is not None:
        restore = ("mcp", "add-json", "--scope", "user", "ssgrep", json.dumps(previous))
        if _claude(binary, *restore)[0] == 0:
            return ("claude", f"error: {summary} (the previous registration was restored)", None)
    return (
        "claude",
        f"error: {summary} (the previous ssgrep registration was removed; "
        "re-run `ssgrep mcp install claude`)",
        None,
    )


def _install_cursor() -> InstallResult:
    path = _cursor_config_path()
    launcher, args = launch_command()
    registration: dict[str, object] = {"command": launcher, "args": args}
    status = _merge_json(path, lambda data: _apply_json_entry(data, "mcpServers", registration))
    return ("cursor", status, path)


def _install_zed() -> InstallResult:
    path = _zed_config_path()
    launcher, args = launch_command()
    registration: dict[str, object] = {"command": launcher, "args": args}
    status = _merge_json(
        path, lambda data: _apply_json_entry(data, "context_servers", registration)
    )
    return ("zed", status, path)


def _codex_block(command: str, args: list[str]) -> str:
    rendered = ", ".join(json.dumps(item) for item in args)  # JSON strings are valid TOML
    return f'{_CODEX_HEADER}\ncommand = "{command}"\nargs = [{rendered}]\n'


def _install_codex() -> InstallResult:
    """Merge a TOML section into ``~/.codex/config.toml`` by text surgery.

    Python ships only a TOML *reader*, and a re-serialization would discard
    the user's comments and layout. The ssgrep section is instead located by
    its exact header line and replaced in place, or appended at the end —
    both of which are valid TOML wherever they land.
    """
    path = _codex_config_path()
    block = _codex_block(*launch_command())
    try:
        text = path.read_text() if path.exists() else ""
        lines = text.splitlines(keepends=True)
        start = next((i for i, line in enumerate(lines) if line.strip() == _CODEX_HEADER), None)
        if start is not None:
            end = next(
                (j for j in range(start + 1, len(lines)) if lines[j].startswith("[")),
                len(lines),
            )
            # The blank lines separating this section from the next one are not
            # part of it: a replacement must leave them in place.
            while end > start + 1 and not lines[end - 1].strip():
                end -= 1
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


def _omp_config_path() -> Path:
    """Return omp's user-scope MCP config file (``~/.omp/agent/mcp.json``)."""
    return resolve_omp_agent_dir() / "mcp.json"


def _install_opencode() -> InstallResult:
    path = _opencode_config_path()
    launcher, args = launch_command()
    registration: dict[str, object] = {"type": "local", "command": [launcher, *args]}

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


def _install_omp() -> InstallResult:
    path = _omp_config_path()
    launcher, args = launch_command()
    registration: dict[str, object] = {"type": "stdio", "command": launcher, "args": args}
    status = _merge_json(path, lambda data: _apply_json_entry(data, "mcpServers", registration))
    return ("omp", status, path)


_INSTALLERS: dict[str, Callable[[], InstallResult]] = {
    "claude": _install_claude,
    "cursor": _install_cursor,
    "zed": _install_zed,
    "codex": _install_codex,
    "opencode": _install_opencode,
    "omp": _install_omp,
}


def install_mcp_registrations(
    clients: tuple[str, ...] | None = None,
) -> tuple[InstallResult, ...]:
    """Register ssgrep with each requested MCP client, independently.

    ``clients`` selects a subset of :data:`CLIENT_NAMES`; ``None`` (or an
    empty selection) registers every client. Each client is attempted
    independently, and failures become ``error: ...`` statuses rather than
    aborting the rest.

    Raises ``ValueError`` up front — before any client is touched — for an
    unknown client name or an invalid ``SSGREP_MCP_LAUNCHER`` value.
    """
    _launcher_mode()
    chosen = clients or CLIENT_NAMES
    unknown = [name for name in chosen if name not in CLIENT_NAMES]
    if unknown:
        raise ValueError(f"unknown MCP client(s): {', '.join(sorted(unknown))}")
    return tuple(_INSTALLERS[name]() for name in chosen)


def failed_clients(results: tuple[InstallResult, ...]) -> tuple[str, ...]:
    """Names of the clients whose registration ended in an ``error: ...`` status.

    ``skipped: ...`` (client not installed) is not a failure.
    """
    return tuple(name for name, status, _ in results if status.startswith("error:"))
