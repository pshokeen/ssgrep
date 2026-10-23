"""Tests for idempotent MCP client registration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ssgrep.services import mcp_install

SSGREP_BIN = mcp_install.ssgrep_path()
CODEx_REGISTRATION = f'command = "{SSGREP_BIN}"\nargs = ["mcp"]\n'


def _configure_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every client config root and return the fake HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "xdg" / "opencode"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("OMP_AGENT_DIR", str(tmp_path / "omp"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.delenv(mcp_install.LAUNCHER_ENV_VAR, raising=False)
    return home


#: A version stand-in for tests that stub package-provenance detection.
PINNED_VERSION = "1.2.3"
PINNED_UVX_JSON = {"command": "uvx", "args": [f"ssgrep@{PINNED_VERSION}", "mcp"]}


def _stub_provenance(
    monkeypatch: pytest.MonkeyPatch, *, from_index: bool, version: str | None = PINNED_VERSION
) -> None:
    """Control ``_installed_from_index()`` and the resolved package version.

    ``launch_command()`` is the caller under test here, and
    ``_installed_from_index`` is deliberately a small, separately-testable
    helper so these tests never need to fake real distribution metadata.
    """
    monkeypatch.setattr(mcp_install, "_installed_from_index", lambda: from_index)
    if version is None:

        def raise_not_found(name: str) -> str:
            raise mcp_install.PackageNotFoundError(name)

        monkeypatch.setattr(mcp_install, "version", raise_not_found)
    else:
        monkeypatch.setattr(mcp_install, "version", lambda name: version)


def _no_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the fallback branch: no ``uvx`` (and no ``claude``) on PATH."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)


def _with_uvx(monkeypatch: pytest.MonkeyPatch, claude: str | None = None) -> None:
    """Pin the preferred branch: ``uvx`` on PATH (optionally ``claude`` too)."""
    import shutil

    def fake_which(name: str) -> str | None:
        if name == "uvx":
            return "/usr/local/bin/uvx"
        if name == "claude":
            return claude
        return None

    monkeypatch.setattr(shutil, "which", fake_which)


def _run_all(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run the installer with claude stubbed out, returning name -> status."""
    import shutil

    def no_which(name: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", no_which)
    results = mcp_install.install_mcp_registrations()
    return {name: status for name, status, _ in results}


def test_installs_every_file_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    (tmp_path / "xdg" / "zed").mkdir(parents=True)
    zed_path = tmp_path / "xdg" / "zed" / "settings.json"
    zed_path.write_text(json.dumps({"theme": "dark"}))

    statuses = _run_all(monkeypatch)

    assert set(statuses) == {"claude", "cursor", "zed", "codex", "opencode", "omp"}
    assert statuses["claude"] == "skipped: claude CLI not found"
    assert all(
        statuses[name] == "installed" for name in ("cursor", "zed", "codex", "opencode", "omp")
    )

    cursor = json.loads((home / ".cursor" / "mcp.json").read_text())
    assert cursor["mcpServers"]["ssgrep"] == {"command": SSGREP_BIN, "args": ["mcp"]}

    zed = json.loads(zed_path.read_text())
    assert zed["theme"] == "dark"
    assert zed["context_servers"]["ssgrep"] == {"command": SSGREP_BIN, "args": ["mcp"]}

    codex = (tmp_path / "codex" / "config.toml").read_text()
    assert "[mcp_servers.ssgrep]" in codex
    assert f'command = "{SSGREP_BIN}"' in codex
    assert 'args = ["mcp"]' in codex

    opencode = json.loads((tmp_path / "xdg" / "opencode" / "opencode.json").read_text())
    assert opencode["mcp"]["servers"]["ssgrep"] == {
        "type": "local",
        "command": [SSGREP_BIN, "mcp"],
    }

    omp = json.loads((tmp_path / "omp" / "mcp.json").read_text())
    assert omp["mcpServers"]["ssgrep"] == {
        "type": "stdio",
        "command": SSGREP_BIN,
        "args": ["mcp"],
    }


def test_reinstall_is_idempotent_and_preserves_formatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _no_uvx(monkeypatch)
    assert all(
        status == "installed"
        for _, status, _ in mcp_install.install_mcp_registrations(
            ("cursor", "zed", "codex", "opencode", "omp")
        )
    )
    before = {
        name: path.read_bytes()
        for name, path in {
            "cursor": Path.home() / ".cursor" / "mcp.json",
            "opencode": Path(mcp_install._opencode_config_path()),
            "omp": Path(mcp_install._omp_config_path()),
        }.items()
    }
    codex_before = Path(mcp_install._codex_config_path()).read_bytes()

    results = {
        name: status
        for name, status, _ in mcp_install.install_mcp_registrations(
            ("cursor", "zed", "codex", "opencode", "omp")
        )
    }

    assert all(status == "already_installed" for status in results.values())
    assert before["cursor"] == (Path.home() / ".cursor" / "mcp.json").read_bytes()
    assert before["opencode"] == Path(mcp_install._opencode_config_path()).read_bytes()
    assert before["omp"] == Path(mcp_install._omp_config_path()).read_bytes()
    assert codex_before == Path(mcp_install._codex_config_path()).read_bytes()


def test_omp_preserves_sibling_keys_and_replaces_stale_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _no_uvx(monkeypatch)
    config = tmp_path / "omp" / "mcp.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "$schema": "https://example.invalid/mcp-schema.json",
                "mcpServers": {
                    "filesystem": {"type": "stdio", "command": "npx", "args": ["-y", "fs"]},
                    "ssgrep": {"command": "pipx", "args": ["run", "ssgrep", "mcp"]},
                },
                "disabledServers": ["filesystem"],
            }
        )
    )

    (name, status, path) = mcp_install.install_mcp_registrations(("omp",))[0]
    assert (name, status, path) == ("omp", "updated", config)

    merged = json.loads(config.read_text())
    assert merged["$schema"] == "https://example.invalid/mcp-schema.json"
    assert merged["disabledServers"] == ["filesystem"]
    assert merged["mcpServers"]["filesystem"] == {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "fs"],
    }
    assert merged["mcpServers"]["ssgrep"] == {
        "type": "stdio",
        "command": SSGREP_BIN,
        "args": ["mcp"],
    }

    assert mcp_install.install_mcp_registrations(("omp",))[0][1] == "already_installed"


def test_codex_replaces_stale_wheel_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _no_uvx(monkeypatch)
    codex = tmp_path / "codex"
    codex.mkdir()
    config = codex / "config.toml"
    config.write_text(
        'model = "gpt-5"\n\n'
        "[mcp_servers.ssgrep]\n"
        'command = "uvx"\n'
        'args = ["--from", "/old/ssgrep-1.0-py3-none-any.whl", "ssgrep", "mcp"]\n\n'
        "[mcp_servers.ssgrep.env]\n"
        'KEEP = "yes"\n'
    )

    status = dict(
        (name, status) for name, status, _ in mcp_install.install_mcp_registrations(("codex",))
    )["codex"]

    assert status == "updated"
    text = config.read_text()
    assert f'command = "{SSGREP_BIN}"' in text
    assert "uvx" not in text
    assert 'model = "gpt-5"' in text
    assert "[mcp_servers.ssgrep.env]" in text
    assert 'KEEP = "yes"' in text


def test_cursor_reports_bad_json_without_touching_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    cursor_dir = home / ".cursor"
    cursor_dir.mkdir()
    broken = cursor_dir / "mcp.json"
    broken.write_text("{not json")

    statuses = _run_all(monkeypatch)

    assert statuses["cursor"].startswith("error:")
    assert broken.read_text() == "{not json"


def test_zed_comments_are_reported_not_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    zed_dir = tmp_path / "xdg" / "zed"
    zed_dir.mkdir(parents=True)
    settings = zed_dir / "settings.json"
    original = '// my settings\n{\n  "theme": "dark"\n}\n'
    settings.write_text(original)

    statuses = _run_all(monkeypatch)

    status = statuses["zed"]
    assert status.startswith("error: settings.json is not strict JSON")
    assert "commented" in status and "cannot safely edit" in status
    assert "(char 0)" not in status
    assert settings.read_text() == original


def test_failed_clients_counts_errors_but_not_skips() -> None:
    results = (
        ("claude", "skipped: claude CLI not found", None),
        ("cursor", "error: [Errno 13] Permission denied", None),
        ("zed", "installed", None),
        ("codex", "error: boom", None),
        ("omp", "already_installed", None),
    )

    assert mcp_install.failed_clients(results) == ("cursor", "codex")
    assert mcp_install.failed_clients(()) == ()


def test_opencode_migrates_legacy_flat_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    directory = tmp_path / "xdg" / "opencode"
    directory.mkdir(parents=True)
    config = directory / "opencode.json"
    config.write_text(
        json.dumps({"mcp": {"ssgrep": {"type": "local", "command": ["uvx", "ssgrep", "mcp"]}}})
    )

    statuses = _run_all(monkeypatch)

    assert statuses["opencode"] == "updated"
    data = json.loads(config.read_text())
    assert data["mcp"]["servers"]["ssgrep"] == {
        "type": "local",
        "command": [SSGREP_BIN, "mcp"],
    }
    assert "ssgrep" not in data["mcp"]


def test_opencode_prefers_existing_jsonc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(tmp_path, monkeypatch)
    directory = tmp_path / "xdg" / "opencode"
    directory.mkdir(parents=True)
    (directory / "opencode.jsonc").write_text("{}")

    statuses = _run_all(monkeypatch)

    assert statuses["opencode"] == "installed"
    assert not (directory / "opencode.json").exists()
    assert json.loads((directory / "opencode.jsonc").read_text())["mcp"]["servers"]["ssgrep"][
        "command"
    ] == [SSGREP_BIN, "mcp"]


def test_unknown_client_name_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nope"):
        mcp_install.install_mcp_registrations(("nope",))


def test_ssgrep_path_falls_back_to_path_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    missing = Path("/nonexistent-dir-without-ssgrep")
    monkeypatch.setattr(mcp_install.sys, "executable", str(missing / "python"))
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/local/bin/ssgrep" if name == "ssgrep" else None
    )

    assert mcp_install.ssgrep_path() == "/usr/local/bin/ssgrep"


def test_top_level_non_object_json_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    cursor_dir = home / ".cursor"
    cursor_dir.mkdir()
    config = cursor_dir / "mcp.json"
    config.write_text('["not", "an", "object"]')

    statuses = _run_all(monkeypatch)

    assert statuses["cursor"] == "error: mcp.json does not contain a JSON object"
    assert config.read_text() == '["not", "an", "object"]'


def test_non_object_server_container_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    cursor_dir = home / ".cursor"
    cursor_dir.mkdir()
    config = cursor_dir / "mcp.json"
    config.write_text('{"mcpServers": "broken"}')

    statuses = _run_all(monkeypatch)

    assert statuses["cursor"] == "error: mcpServers is not a JSON object"
    assert config.read_text() == '{"mcpServers": "broken"}'


def test_opencode_mcp_section_of_wrong_type_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    directory = tmp_path / "xdg" / "opencode"
    directory.mkdir(parents=True)
    config = directory / "opencode.json"
    original = json.dumps({"mcp": "legacy-string-value"})
    config.write_text(original)

    statuses = _run_all(monkeypatch)

    assert statuses["opencode"] == "error: mcp section is not a JSON object"
    assert config.read_text() == original


def test_opencode_legacy_flat_with_wrong_servers_type_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    directory = tmp_path / "xdg" / "opencode"
    directory.mkdir(parents=True)
    config = directory / "opencode.json"
    original = json.dumps(
        {"mcp": {"ssgrep": {"type": "local", "command": ["old"]}, "servers": "broken"}}
    )
    config.write_text(original)

    statuses = _run_all(monkeypatch)

    assert statuses["opencode"] == "error: mcp.servers is not a JSON object"
    assert json.loads(config.read_text()) == json.loads(original)


def test_opencode_legacy_flat_entry_already_current_migrates_without_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _no_uvx(monkeypatch)
    directory = tmp_path / "xdg" / "opencode"
    directory.mkdir(parents=True)
    config = directory / "opencode.json"
    registration = {"type": "local", "command": [mcp_install.ssgrep_path(), "mcp"]}
    config.write_text(
        json.dumps(
            {"mcp": {"ssgrep": dict(registration), "servers": {"ssgrep": dict(registration)}}}
        )
    )

    statuses = _run_all(monkeypatch)

    assert statuses["opencode"] == "updated"
    data = json.loads(config.read_text())
    assert data["mcp"]["servers"]["ssgrep"] == registration
    assert "ssgrep" not in data["mcp"]


def test_claude_failures_map_to_statuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(tmp_path, monkeypatch)
    import shutil

    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else None
    )

    def failing_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("claude disappeared")

    monkeypatch.setattr(subprocess, "run", failing_run)
    (name, status, _) = mcp_install.install_mcp_registrations(("claude",))[0]
    assert (name, status) == ("claude", "error: claude disappeared")

    def already_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="already registered elsewhere"
        )

    # ``add`` claims an entry exists that the config does not show: the
    # replace attempt fails too, and that is an error, not "already_installed".
    monkeypatch.setattr(subprocess, "run", already_run)
    status = mcp_install.install_mcp_registrations(("claude",))[0][1]
    assert status == "error: already registered elsewhere"

    def error_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="boom happened")

    monkeypatch.setattr(subprocess, "run", error_run)
    status = mcp_install.install_mcp_registrations(("claude",))[0][1]
    assert status == "error: boom happened"

    def silent_failure(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", silent_failure)
    status = mcp_install.install_mcp_registrations(("claude",))[0][1]
    assert status == "error: exit code 9"


def test_codex_appends_after_file_without_trailing_newline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    codex = tmp_path / "codex"
    codex.mkdir()
    config = codex / "config.toml"
    config.write_text('model = "gpt-5"')

    statuses = _run_all(monkeypatch)

    assert statuses["codex"] == "installed"
    text = config.read_text()
    assert text.startswith('model = "gpt-5"\n\n')
    assert f'command = "{SSGREP_BIN}"' in text


def test_codex_write_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(tmp_path, monkeypatch)
    codex_file = tmp_path / "codex"
    codex_file.write_text("i am a file, not a directory\n")  # parent mkdir will fail

    statuses = _run_all(monkeypatch)

    assert statuses["codex"].startswith("error:")
    assert codex_file.read_text() == "i am a file, not a directory\n"


CLAUDE_BIN = "/usr/local/bin/claude"
CURRENT_CLAUDE_ENTRY = {"type": "stdio", "command": SSGREP_BIN, "args": ["mcp"], "env": {}}
STALE_CLAUDE_ENTRY = {"type": "stdio", "command": "uvx", "args": ["ssgrep", "mcp"], "env": {}}
ADD_CURRENT = [
    CLAUDE_BIN,
    "mcp",
    "add",
    "--scope",
    "user",
    "ssgrep",
    "--",
    SSGREP_BIN,
    "mcp",
]
REMOVE = [CLAUDE_BIN, "mcp", "remove", "--scope", "user", "ssgrep"]


def _fake_claude_run(
    calls: list[list[str]],
    outcomes: dict[str, list[tuple[int, str]]] | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A fake ``subprocess.run`` answering each ``claude mcp <verb>`` in turn.

    ``outcomes`` maps a verb (``add``, ``remove``, ``add-json``) to the
    ``(returncode, stderr)`` answers to give successive calls of that verb;
    unlisted calls succeed.
    """
    pending = {verb: list(answers) for verb, answers in (outcomes or {}).items()}

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        answers = pending.get(command[2])
        code, stderr = answers.pop(0) if answers else (0, "")
        return subprocess.CompletedProcess(command, code, stdout="", stderr=stderr)

    return fake_run


def _claude_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: CLAUDE_BIN if name == "claude" else None)


def _seed_claude_config(tmp_path: Path, entry: object) -> Path:
    config = tmp_path / "claude" / ".claude.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"theme": "dark", "mcpServers": {"ssgrep": entry}}))
    return config


def _install_claude() -> tuple[str, str, Path | None]:
    return mcp_install.install_mcp_registrations(("claude",))[0]


def test_claude_uses_its_own_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls))
    _claude_on_path(monkeypatch)

    assert _install_claude() == ("claude", "installed", None)
    assert calls == [ADD_CURRENT]


def test_claude_matching_registration_is_already_installed_without_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotence: an identical entry is neither removed nor re-added."""
    _configure_env(tmp_path, monkeypatch)
    config = _seed_claude_config(tmp_path, CURRENT_CLAUDE_ENTRY)
    before = config.read_text()
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls))
    _claude_on_path(monkeypatch)

    assert _install_claude() == ("claude", "already_installed", None)
    assert calls == []
    assert config.read_text() == before


def test_claude_reads_default_config_when_claude_config_dir_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {"ssgrep": CURRENT_CLAUDE_ENTRY}}))
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls))
    _claude_on_path(monkeypatch)

    assert _install_claude()[1] == "already_installed"
    assert calls == []


@pytest.mark.parametrize(
    "entry",
    [
        STALE_CLAUDE_ENTRY,
        {**CURRENT_CLAUDE_ENTRY, "type": "http"},
        {"type": "stdio", "command": SSGREP_BIN, "args": ["mcp", "--extra"]},
    ],
)
def test_claude_stale_registration_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: dict[str, object]
) -> None:
    """The issue #4 repair path for Claude: ``claude mcp add`` never compares
    against an existing entry, so a stale registration (the old unpinned
    ``uvx ssgrep mcp`` form, or an outdated pinned version) must be
    removed and re-added rather than reported as already_installed."""
    _configure_env(tmp_path, monkeypatch)
    _seed_claude_config(tmp_path, entry)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls))
    _claude_on_path(monkeypatch)

    assert _install_claude() == ("claude", "updated", None)
    assert calls == [REMOVE, ADD_CURRENT]


def test_claude_add_failure_after_remove_restores_the_previous_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed re-add must not cost the user their working registration."""
    _configure_env(tmp_path, monkeypatch)
    _seed_claude_config(tmp_path, STALE_CLAUDE_ENTRY)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls, {"add": [(1, "add exploded")]}))
    _claude_on_path(monkeypatch)

    (name, status, _) = _install_claude()

    assert name == "claude"
    assert status == "error: add exploded (the previous registration was restored)"
    assert calls == [
        REMOVE,
        ADD_CURRENT,
        [
            CLAUDE_BIN,
            "mcp",
            "add-json",
            "--scope",
            "user",
            "ssgrep",
            json.dumps(STALE_CLAUDE_ENTRY),
        ],
    ]


def test_claude_failed_restore_reports_lost_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If neither the re-add nor the restore works, Claude has no ssgrep
    registration; the status must say so and how to recover."""
    _configure_env(tmp_path, monkeypatch)
    _seed_claude_config(tmp_path, STALE_CLAUDE_ENTRY)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_claude_run(calls, {"add": [(1, "add exploded")], "add-json": [(1, "nope")]}),
    )
    _claude_on_path(monkeypatch)

    status = _install_claude()[1]

    assert "add exploded" in status
    assert "the previous ssgrep registration was removed" in status
    assert "re-run `ssgrep mcp install claude`" in status
    assert [call[2] for call in calls] == ["remove", "add", "add-json"]


def test_claude_failed_remove_reports_the_add_failure_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _seed_claude_config(tmp_path, STALE_CLAUDE_ENTRY)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_claude_run(calls, {"remove": [(1, "locked")], "add": [(1, "already exists")]}),
    )
    _claude_on_path(monkeypatch)

    assert _install_claude()[1] == "error: already exists"
    assert [call[2] for call in calls] == ["remove", "add"]


def test_claude_entry_unreadable_but_add_reports_existing_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no readable config, ``add`` saying "already exists" triggers a replace."""
    _configure_env(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_claude_run(calls, {"add": [(1, "MCP server ssgrep already exists in user config")]}),
    )
    _claude_on_path(monkeypatch)

    assert _install_claude()[1] == "updated"
    assert calls == [ADD_CURRENT, REMOVE, ADD_CURRENT]


def test_claude_entry_unreadable_and_add_failing_after_remove_reports_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_claude_run(calls, {"add": [(1, "already exists"), (1, "add exploded")]}),
    )
    _claude_on_path(monkeypatch)

    status = _install_claude()[1]

    assert "add exploded" in status
    assert "the previous ssgrep registration was removed" in status
    assert [call[2] for call in calls] == ["add", "remove", "add"]  # nothing known to restore


@pytest.mark.parametrize(
    "content",
    ["{not json", "[]", '{"mcpServers": []}', '{"mcpServers": {"ssgrep": "text"}}', "{}"],
)
def test_claude_unknown_registration_state_falls_back_to_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    _configure_env(tmp_path, monkeypatch)
    config = tmp_path / "claude" / ".claude.json"
    config.parent.mkdir()
    config.write_text(content)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls))
    _claude_on_path(monkeypatch)

    assert _install_claude()[1] == "installed"
    assert calls == [ADD_CURRENT]
    assert config.read_text() == content


_CLAUDE_SHIM = """#!{python}
import json, os, sys

config = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], ".claude.json")
state = json.load(open(config)) if os.path.exists(config) else {{}}
servers = state.setdefault("mcpServers", {{}})
with open(os.environ["SHIM_LOG"], "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\\n")
verb = sys.argv[2]
if verb == "add":
    if os.environ.get("SHIM_FAIL_ADD"):
        print("claude exploded", file=sys.stderr)
        sys.exit(1)
    if "ssgrep" in servers:
        print("MCP server ssgrep already exists in user config", file=sys.stderr)
        sys.exit(1)
    command = sys.argv[sys.argv.index("--") + 1 :]
    servers["ssgrep"] = {{"type": "stdio", "command": command[0], "args": command[1:], "env": {{}}}}
elif verb == "add-json":
    servers["ssgrep"] = json.loads(sys.argv[-1])
elif verb == "remove":
    servers.pop("ssgrep", None)
json.dump(state, open(config, "w"))
"""


def test_claude_install_is_idempotent_against_a_fake_claude_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end with a real executable first on PATH: install, no-op rerun,
    stale repair, and an add failure that leaves the old entry in place."""
    _configure_env(tmp_path, monkeypatch)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "claude"
    shim.write_text(_CLAUDE_SHIM.format(python=sys.executable))
    shim.chmod(0o755)
    log = tmp_path / "shim.log"
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SHIM_LOG", str(log))
    config = tmp_path / "claude" / ".claude.json"
    config.parent.mkdir()

    def registered() -> dict[str, object]:
        return json.loads(config.read_text())["mcpServers"]["ssgrep"]

    assert _install_claude()[1] == "installed"
    assert registered() == {"type": "stdio", "command": SSGREP_BIN, "args": ["mcp"], "env": {}}

    log.write_text("")
    assert _install_claude()[1] == "already_installed"
    assert log.read_text() == ""  # a rerun does not even invoke claude

    state = json.loads(config.read_text())
    state["mcpServers"]["ssgrep"] = STALE_CLAUDE_ENTRY
    config.write_text(json.dumps(state))
    assert _install_claude()[1] == "updated"
    assert registered()["command"] == SSGREP_BIN

    state["mcpServers"]["ssgrep"] = STALE_CLAUDE_ENTRY
    config.write_text(json.dumps(state))
    monkeypatch.setenv("SHIM_FAIL_ADD", "1")
    status = _install_claude()[1]
    assert status == "error: claude exploded (the previous registration was restored)"
    assert registered() == STALE_CLAUDE_ENTRY


UVX_JSON = PINNED_UVX_JSON


def test_launch_command_auto_prefers_uvx_when_installed_from_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True)
    assert mcp_install.launch_command() == ("uvx", [f"ssgrep@{PINNED_VERSION}", "mcp"])


def test_launch_command_auto_without_uvx_uses_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True)
    assert mcp_install.launch_command() == (SSGREP_BIN, ["mcp"])


def test_launch_command_auto_not_from_index_uses_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact issue #4 repro: uvx on PATH, but ssgrep was not installed from an index."""
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=False)
    assert mcp_install.launch_command() == (SSGREP_BIN, ["mcp"])


def test_launch_command_auto_falls_back_when_version_undeterminable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True, version=None)
    assert mcp_install.launch_command() == (SSGREP_BIN, ["mcp"])


def test_launcher_env_path_overrides_uvx_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True)
    monkeypatch.setenv(mcp_install.LAUNCHER_ENV_VAR, "path")
    assert mcp_install.launch_command() == (SSGREP_BIN, ["mcp"])


def test_launcher_env_uvx_forces_uvx_even_without_uvx_on_path_or_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=False)
    monkeypatch.setenv(mcp_install.LAUNCHER_ENV_VAR, "uvx")
    assert mcp_install.launch_command() == ("uvx", [f"ssgrep@{PINNED_VERSION}", "mcp"])


def test_launcher_env_uvx_falls_back_to_unpinned_when_version_undeterminable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced ``uvx`` mode falls back to unpinned ``ssgrep``, never a literal ``ssgrep@None``."""
    _no_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=False, version=None)
    monkeypatch.setenv(mcp_install.LAUNCHER_ENV_VAR, "uvx")
    assert mcp_install.launch_command() == ("uvx", ["ssgrep", "mcp"])


def test_launcher_env_is_case_insensitive_and_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=False)
    monkeypatch.setenv(mcp_install.LAUNCHER_ENV_VAR, "  UVX  ")
    assert mcp_install.launch_command() == ("uvx", [f"ssgrep@{PINNED_VERSION}", "mcp"])


def test_invalid_launcher_env_value_is_rejected_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    monkeypatch.setenv(mcp_install.LAUNCHER_ENV_VAR, "bogus")

    with pytest.raises(ValueError, match="SSGREP_MCP_LAUNCHER") as excinfo:
        mcp_install.install_mcp_registrations(("cursor",))

    assert "bogus" in str(excinfo.value)
    assert not (home / ".cursor" / "mcp.json").exists()


def test_installed_from_index_true_when_no_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Dist:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return None

    monkeypatch.setattr(mcp_install, "distribution", lambda name: _Dist())
    assert mcp_install._installed_from_index() is True


def test_installed_from_index_false_when_direct_url_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Dist:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return '{"url": "git+https://example.invalid/ssgrep"}'

    monkeypatch.setattr(mcp_install, "distribution", lambda name: _Dist())
    assert mcp_install._installed_from_index() is False


def test_installed_from_index_false_when_package_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_not_found(name: str) -> None:
        raise mcp_install.PackageNotFoundError(name)

    monkeypatch.setattr(mcp_install, "distribution", raise_not_found)
    assert mcp_install._installed_from_index() is False


def test_installed_from_index_false_against_real_editable_checkout() -> None:
    """This test suite always runs against an editable install of ssgrep
    itself (``uv sync``), so the real (unstubbed) metadata API must agree
    with the stubbed unit tests above: not from an index."""
    assert mcp_install._installed_from_index() is False


def test_prefers_uvx_registration_in_every_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With uvx on PATH and an index install, every written entry is pinned uvx."""
    home = _configure_env(tmp_path, monkeypatch)
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True)

    statuses = {name: s for name, s, _ in mcp_install.install_mcp_registrations()}
    assert statuses["claude"] == "skipped: claude CLI not found"
    assert all(statuses[n] == "installed" for n in ("cursor", "zed", "codex", "opencode", "omp"))

    cursor = json.loads((home / ".cursor" / "mcp.json").read_text())
    assert cursor["mcpServers"]["ssgrep"] == UVX_JSON
    zed = json.loads((tmp_path / "xdg" / "zed" / "settings.json").read_text())
    assert zed["context_servers"]["ssgrep"] == UVX_JSON
    codex = (tmp_path / "codex" / "config.toml").read_text()
    assert 'command = "uvx"' in codex
    assert f'args = ["ssgrep@{PINNED_VERSION}", "mcp"]' in codex
    assert SSGREP_BIN not in codex
    opencode = json.loads((tmp_path / "xdg" / "opencode" / "opencode.json").read_text())
    assert opencode["mcp"]["servers"]["ssgrep"] == {
        "type": "local",
        "command": ["uvx", f"ssgrep@{PINNED_VERSION}", "mcp"],
    }
    omp = json.loads((tmp_path / "omp" / "mcp.json").read_text())
    assert omp["mcpServers"]["ssgrep"] == {"type": "stdio", **UVX_JSON}


def test_claude_registers_uvx_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _with_uvx(monkeypatch, claude="/usr/local/bin/claude")
    _stub_provenance(monkeypatch, from_index=True)
    calls: list[list[str]] = []

    monkeypatch.setattr(subprocess, "run", _fake_claude_run(calls))
    assert mcp_install.install_mcp_registrations(("claude",))[0][1] == "installed"
    assert calls[-1] == [
        "/usr/local/bin/claude",
        "mcp",
        "add",
        "--scope",
        "user",
        "ssgrep",
        "--",
        "uvx",
        f"ssgrep@{PINNED_VERSION}",
        "mcp",
    ]


def test_rerun_upgrades_absolute_path_entries_to_uvx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing absolute-path registration is rewritten to pinned uvx, then stays put."""
    home = _configure_env(tmp_path, monkeypatch)
    clients = ("cursor", "codex", "opencode", "omp")
    _no_uvx(monkeypatch)
    first = {n: s for n, s, _ in mcp_install.install_mcp_registrations(clients)}
    assert set(first.values()) == {"installed"}
    cursor_cfg = home / ".cursor" / "mcp.json"
    assert json.loads(cursor_cfg.read_text())["mcpServers"]["ssgrep"] == {
        "command": SSGREP_BIN,
        "args": ["mcp"],
    }

    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True)
    second = {n: s for n, s, _ in mcp_install.install_mcp_registrations(clients)}
    assert set(second.values()) == {"updated"}
    assert json.loads(cursor_cfg.read_text())["mcpServers"]["ssgrep"] == UVX_JSON
    codex = (tmp_path / "codex" / "config.toml").read_text()
    assert 'command = "uvx"' in codex and SSGREP_BIN not in codex
    opencode = json.loads((tmp_path / "xdg" / "opencode" / "opencode.json").read_text())
    assert opencode["mcp"]["servers"]["ssgrep"]["command"] == [
        "uvx",
        f"ssgrep@{PINNED_VERSION}",
        "mcp",
    ]
    omp = json.loads((tmp_path / "omp" / "mcp.json").read_text())
    assert omp["mcpServers"]["ssgrep"]["command"] == "uvx"

    third = {n: s for n, s, _ in mcp_install.install_mcp_registrations(clients)}
    assert set(third.values()) == {"already_installed"}


def test_uvx_entry_falls_back_to_absolute_path_when_uvx_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registration must be launchable where it is written, so a uvx entry is downgraded."""
    home = _configure_env(tmp_path, monkeypatch)
    cursor_dir = home / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": {"ssgrep": UVX_JSON, "other": {"command": "x"}}})
    )
    _no_uvx(monkeypatch)

    (name, status, _) = mcp_install.install_mcp_registrations(("cursor",))[0]
    assert (name, status) == ("cursor", "updated")
    data = json.loads((cursor_dir / "mcp.json").read_text())
    assert data["mcpServers"]["ssgrep"] == {"command": SSGREP_BIN, "args": ["mcp"]}
    assert data["mcpServers"]["other"] == {"command": "x"}


def test_legacy_unpinned_uvx_entry_is_repaired_to_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact issue #4 repair path: an old unpinned ``uvx ssgrep mcp`` entry
    (what pre-fix ssgrep wrote whenever uvx was on PATH, regardless of
    provenance) must be rewritten to the absolute path once the fix runs,
    because uvx is on PATH but the install is not from an index."""
    home = _configure_env(tmp_path, monkeypatch)
    cursor_dir = home / ".cursor"
    cursor_dir.mkdir()
    legacy_broken_entry = {"command": "uvx", "args": ["ssgrep", "mcp"]}
    (cursor_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": {"ssgrep": legacy_broken_entry}})
    )
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=False)

    (name, status, _) = mcp_install.install_mcp_registrations(("cursor",))[0]

    assert (name, status) == ("cursor", "updated")
    data = json.loads((cursor_dir / "mcp.json").read_text())
    assert data["mcpServers"]["ssgrep"] == {"command": SSGREP_BIN, "args": ["mcp"]}


def test_pinned_uvx_entry_is_repaired_on_version_bump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale version pin left by an older ssgrep must be rewritten to the
    currently-installed version — otherwise the CLI and MCP server drift
    apart the moment ssgrep is upgraded."""
    home = _configure_env(tmp_path, monkeypatch)
    cursor_dir = home / ".cursor"
    cursor_dir.mkdir()
    stale_entry = {"command": "uvx", "args": ["ssgrep@1.2.3", "mcp"]}
    (cursor_dir / "mcp.json").write_text(json.dumps({"mcpServers": {"ssgrep": stale_entry}}))
    _with_uvx(monkeypatch)
    _stub_provenance(monkeypatch, from_index=True, version="1.2.4")

    (name, status, _) = mcp_install.install_mcp_registrations(("cursor",))[0]

    assert (name, status) == ("cursor", "updated")
    data = json.loads((cursor_dir / "mcp.json").read_text())
    assert data["mcpServers"]["ssgrep"] == {"command": "uvx", "args": ["ssgrep@1.2.4", "mcp"]}

    third = mcp_install.install_mcp_registrations(("cursor",))[0]
    assert third[1] == "already_installed"


def test_codex_replacement_keeps_the_blank_line_before_the_next_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    _no_uvx(monkeypatch)
    codex = tmp_path / "codex"
    codex.mkdir()
    config = codex / "config.toml"
    config.write_text(
        '[mcp_servers.ssgrep]\ncommand = "old"\nargs = ["mcp"]\n\n[other]\nkey = "value"\n'
    )

    assert mcp_install.install_mcp_registrations(("codex",))[0][1] == "updated"
    assert config.read_text() == (
        "[mcp_servers.ssgrep]\n"
        f'command = "{SSGREP_BIN}"\n'
        'args = ["mcp"]\n'
        "\n"
        "[other]\n"
        'key = "value"\n'
    )

    before = config.read_text()
    assert mcp_install.install_mcp_registrations(("codex",))[0][1] == "already_installed"
    assert config.read_text() == before


def test_json_client_write_failure_is_reported_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _configure_env(tmp_path, monkeypatch)
    (home / ".cursor").write_text("a file where the config directory should be\n")

    (name, status, _) = mcp_install.install_mcp_registrations(("cursor",))[0]

    assert name == "cursor"
    assert status.startswith("error:")
    assert (home / ".cursor").read_text() == "a file where the config directory should be\n"
