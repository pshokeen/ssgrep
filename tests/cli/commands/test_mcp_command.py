"""Tests for the MCP server command."""

from __future__ import annotations

import io
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
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

    results = (("cursor", "installed", Path("/tmp/mcp.json")),)
    installer = Mock(return_value=results)
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", installer)
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: True)

    payload = command().handle(action="install", clients=["cursor"])

    installer.assert_called_once_with(("cursor",))
    assert payload == {
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


def _stub_serving(monkeypatch: pytest.MonkeyPatch, server: object) -> Mock:
    """Serve ``server`` without touching real signal handlers or reconcile state."""
    install_handler = Mock()
    monkeypatch.setattr(mcp_server.signal, "signal", install_handler)
    monkeypatch.setattr(mcp_server, "_reconcile_thread", None)
    monkeypatch.setattr(mcp_server, "get_mcp_server", lambda: server)
    return install_handler


def test_handle_evicts_shadow_module_and_runs_server(monkeypatch: pytest.MonkeyPatch) -> None:
    shadow = SimpleNamespace()
    server = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "mcp", shadow)
    install_handler = _stub_serving(monkeypatch, server)

    assert command().handle() is None

    assert sys.modules.get("mcp") is not shadow
    server.run.assert_called_once_with()
    installed = {call.args[0] for call in install_handler.call_args_list}
    assert installed == {signal.SIGINT, signal.SIGTERM}


def test_handle_preserves_real_mcp_module(monkeypatch: pytest.MonkeyPatch) -> None:
    real_like = SimpleNamespace(types=object())
    server = SimpleNamespace(run=Mock())
    monkeypatch.setitem(sys.modules, "mcp", real_like)
    _stub_serving(monkeypatch, server)

    command().handle()

    assert sys.modules["mcp"] is real_like
    server.run.assert_called_once_with()


def _results(*rows: tuple[str, str]) -> tuple[mcp_install.InstallResult, ...]:
    return tuple((name, status, Path(f"/cfg/{name}.json")) for name, status in rows)


def test_install_with_a_failed_client_still_reports_everyone_then_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    results = _results(
        ("claude", "skipped: claude CLI not found"),
        ("cursor", "error: [Errno 13] Permission denied"),
        ("zed", "installed"),
    )
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", Mock(return_value=results))
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(mcp_command, "is_quiet", lambda: False)

    with pytest.raises(SystemExit) as raised:
        command().handle(action="install")

    assert raised.value.code == exit_codes.INTERNAL_FAILURE
    captured = capsys.readouterr()
    assert "claude" in captured.out and "cursor" in captured.out and "zed" in captured.out
    assert "Permission denied" in captured.out
    assert captured.err == "registration failed for: cursor\n"


def test_install_skipped_clients_alone_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    results = _results(("claude", "skipped: claude CLI not found"))
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", Mock(return_value=results))
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(mcp_command, "is_quiet", lambda: True)

    assert command().handle(action="install") is None


def test_install_failure_in_quiet_mode_still_fails_and_names_the_client(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    results = _results(("zed", "error: settings.json is not strict JSON"))
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", Mock(return_value=results))
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(mcp_command, "is_quiet", lambda: True)

    with pytest.raises(SystemExit) as raised:
        command().handle(action="install")

    assert raised.value.code == exit_codes.INTERNAL_FAILURE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "registration failed for: zed\n"


def test_install_failure_json_reports_every_client_with_ok_false_and_hard_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _results(("claude", "skipped: claude CLI not found"), ("zed", "error: bad json"))
    monkeypatch.setattr(mcp_install, "install_mcp_registrations", Mock(return_value=results))
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: True)
    stdout = io.StringIO()
    monkeypatch.setattr(mcp_command.sys, "__stdout__", stdout)
    exit_mock = Mock(side_effect=SystemExit(99))
    monkeypatch.setattr(mcp_command.os, "_exit", exit_mock)

    with pytest.raises(SystemExit):
        command().handle(action="install")

    exit_mock.assert_called_once_with(exit_codes.INTERNAL_FAILURE)
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "client_registration_failed",
        "message": "registration failed for: zed",
        "command": None,
        "clients": {
            "claude": {"status": "skipped: claude CLI not found", "config": "/cfg/claude.json"},
            "zed": {"status": "error: bad json", "config": "/cfg/zed.json"},
        },
    }


def test_install_end_to_end_with_commented_zed_settings_fails_without_clobbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    zed_dir = tmp_path / "xdg" / "zed"
    zed_dir.mkdir(parents=True)
    settings = zed_dir / "settings.json"
    original = '// my settings\n{"theme": "dark"}\n'
    settings.write_text(original)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("OMP_AGENT_DIR", str(tmp_path / "omp"))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "opencode"))
    monkeypatch.delenv("SSGREP_MCP_LAUNCHER", raising=False)
    monkeypatch.setattr(mcp_install.shutil, "which", lambda name: None)
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(mcp_command, "is_quiet", lambda: False)

    with pytest.raises(SystemExit) as raised:
        command().handle(action="install", clients=["zed", "cursor"])

    assert raised.value.code == exit_codes.INTERNAL_FAILURE
    captured = capsys.readouterr()
    assert "not strict JSON" in captured.out
    assert "installed" in captured.out  # cursor was still registered
    assert (tmp_path / ".cursor" / "mcp.json").exists()
    assert settings.read_text() == original
