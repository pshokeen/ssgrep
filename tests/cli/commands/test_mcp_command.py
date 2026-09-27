"""Tests for the MCP server command."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ssgrep.cli.commands import mcp_command
from ssgrep.services import mcp_install, mcp_server


def command() -> mcp_command.McpCommand:
    return object.__new__(mcp_command.McpCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "mcp"
    assert instance.description() == "Start the MCP stdio server (or install client registrations)"


def test_unknown_action_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        command().handle(action="remove")

    assert raised.value.code == 2


def test_install_unknown_client_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        command().handle(action="install", clients=["nope"])

    assert raised.value.code == 2


def test_install_invalid_launcher_env_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The real (unmocked) installer rejects a bad SSGREP_MCP_LAUNCHER cleanly."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SSGREP_MCP_LAUNCHER", "bogus")

    with pytest.raises(SystemExit) as raised:
        command().handle(action="install", clients=["cursor"])

    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "Usage error" in err
    assert "SSGREP_MCP_LAUNCHER" in err
    assert "bogus" in err
    assert not (tmp_path / ".cursor" / "mcp.json").exists()


def test_install_json_returns_status_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    results = (("cursor", "installed", Path("/tmp/mcp.json")),)
    installer = Mock(return_value=results)
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", installer)
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: True)

    payload = command().handle(action="install", clients=["cursor"])

    installer.assert_called_once_with(("cursor",))
    assert payload == {
        "ok": True,
        "clients": {"cursor": {"status": "installed", "config": "/tmp/mcp.json"}},
    }
    json.dumps(payload)  # the strict usecli serializer must accept it


def test_install_without_clients_installs_every_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    installer = Mock(return_value=())
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", installer)
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: True)

    command().handle(action="install")

    installer.assert_called_once_with(())


def test_status_style_variants() -> None:
    """Every branch of the registration status styler is reachable."""
    style = mcp_command._status_style

    assert style("error: disk full") == "[red]error: disk full[/red]"
    assert style("skipped: claude CLI not found") == "[gray]skipped: claude CLI not found[/gray]"
    assert style("already_installed") == "[gray]already installed[/gray]"
    assert style("updated") == "[yellow]updated[/yellow]"
    assert style("installed") == "[green]installed[/green]"
    assert style("something_else") == "something_else"


def test_display_registrations_renders_name_status_and_location(capsys) -> None:
    results = (
        ("cursor", "installed", Path("/tmp/mcp.json")),
        ("claude", "installed", None),
    )

    mcp_command._display_registrations(results)

    out = capsys.readouterr().out
    assert "cursor" in out and "installed" in out and "/tmp/mcp.json" in out
    assert "claude" in out


def test_unknown_action_writes_stderr_in_json_mode(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: True)

    with pytest.raises(SystemExit) as raised:
        command().handle(action="remove")

    assert raised.value.code == 2
    assert "unknown mcp action" in capsys.readouterr().err


def test_install_text_mode_renders_and_honors_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    installer = Mock(return_value=(("cursor", "installed", Path("/tmp/mcp.json")),))
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", installer)
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: False)

    monkeypatch.setattr(mcp_command, "is_quiet", lambda: False)
    assert command().handle(action="install") is None
    out = capsys.readouterr().out
    assert "cursor" in out and "installed" in out and "/tmp/mcp.json" in out

    monkeypatch.setattr(mcp_command, "is_quiet", lambda: True)
    assert command().handle(action="install") is None
    assert capsys.readouterr().out == ""


def test_handle_evicts_shadow_module_and_runs_server(monkeypatch: pytest.MonkeyPatch) -> None:
    shadow = SimpleNamespace()
    server = SimpleNamespace(run=Mock())
    get_server = Mock(return_value=server)
    monkeypatch.setitem(sys.modules, "mcp", shadow)
    monkeypatch.setattr(mcp_server, "get_mcp_server", get_server)

    assert command().handle() is None

    assert sys.modules.get("mcp") is not shadow
    get_server.assert_called_once_with()
    server.run.assert_called_once_with()


def test_handle_preserves_real_mcp_module(monkeypatch: pytest.MonkeyPatch) -> None:
    real_like = SimpleNamespace(types=object())
    server = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "mcp", real_like)
    monkeypatch.setattr(mcp_server, "get_mcp_server", lambda: server)

    command().handle()

    assert sys.modules["mcp"] is real_like
    server.run.assert_called_once_with()
