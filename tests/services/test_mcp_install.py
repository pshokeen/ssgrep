"""Tests for idempotent MCP client registration."""

from __future__ import annotations

import json
import subprocess
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
    return home


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

    assert set(statuses) == {"claude", "cursor", "zed", "codex", "opencode"}
    assert statuses["claude"] == "skipped: claude CLI not found"
    assert all(statuses[name] == "installed" for name in ("cursor", "zed", "codex", "opencode"))

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


def test_reinstall_is_idempotent_and_preserves_formatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
    assert all(
        status == "installed"
        for _, status, _ in mcp_install.install_mcp_registrations(
            ("cursor", "zed", "codex", "opencode")
        )
    )
    before = {
        name: path.read_bytes()
        for name, path in {
            "cursor": Path.home() / ".cursor" / "mcp.json",
            "opencode": Path(mcp_install._opencode_config_path()),
        }.items()
    }
    codex_before = Path(mcp_install._codex_config_path()).read_bytes()

    results = {
        name: status
        for name, status, _ in mcp_install.install_mcp_registrations(
            ("cursor", "zed", "codex", "opencode")
        )
    }

    assert all(status == "already_installed" for status in results.values())
    assert before["cursor"] == (Path.home() / ".cursor" / "mcp.json").read_bytes()
    assert before["opencode"] == Path(mcp_install._opencode_config_path()).read_bytes()
    assert codex_before == Path(mcp_install._codex_config_path()).read_bytes()


def test_codex_replaces_stale_wheel_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_env(tmp_path, monkeypatch)
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

    assert statuses["zed"].startswith("error:")
    assert settings.read_text() == original


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

    monkeypatch.setattr(subprocess, "run", already_run)
    status = mcp_install.install_mcp_registrations(("claude",))[0][1]
    assert status == "already_installed"

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


def test_claude_uses_its_own_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    import shutil

    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/claude" if name == "claude" else None

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", fake_which)

    statuses = {
        name: status for name, status, _ in mcp_install.install_mcp_registrations(("claude",))
    }

    assert statuses["claude"] == "installed"
    assert calls == [
        [
            "/usr/local/bin/claude",
            "mcp",
            "add",
            "--scope",
            "user",
            "ssgrep",
            "--",
            SSGREP_BIN,
            "mcp",
        ]
    ]
