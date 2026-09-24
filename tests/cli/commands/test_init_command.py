"""Tests for the restored multi-runtime init command."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import init_command
from ssgrep.utilities.types import IndexNotReadyError


def command() -> init_command.InitCommand:
    return object.__new__(init_command.InitCommand)


def _stub_mcp(monkeypatch: pytest.MonkeyPatch, results: tuple = ()) -> Mock:
    """Stub the MCP installer so tests never touch real client configs."""
    installer = Mock(return_value=results)
    monkeypatch.setattr(init_command.mcp_install, "install_mcp_registrations", installer)
    return installer


def test_metadata() -> None:
    instance = command()
    assert instance.visible() is True
    assert instance.signature() == "init"
    assert (
        instance.description()
        == "Install agent skills, register MCP clients, and build the global multi-runtime index"
    )


def test_handle_json_document(sample_stats, monkeypatch: pytest.MonkeyPatch) -> None:
    stats = replace(sample_stats, runtime_counts=(("claude", 2), ("pi", 1)))
    monkeypatch.setattr(
        init_command.integrations,
        "install_skills",
        Mock(
            return_value=(
                ("claude", "user_modified"),
                ("opencode", "installed"),
            )
        ),
    )
    _stub_mcp(monkeypatch, (("cursor", "updated", Path("/tmp/mcp.json")),))
    monkeypatch.setattr(init_command.api, "index", Mock(return_value=stats))
    monkeypatch.setattr(init_command, "is_json_mode", lambda: True)
    serialize = Mock(wraps=lambda value: {"serialized": value.session_count})
    monkeypatch.setattr(init_command, "to_jsonable", serialize)

    result = command().handle()

    assert result == {
        "ok": True,
        "skills": {"claude": "user_modified", "opencode": "installed"},
        "mcp": {"cursor": {"status": "updated", "config": "/tmp/mcp.json"}},
        "sources": {"claude": 2, "pi": 1},
        "index": {"serialized": sample_stats.session_count},
    }
    serialize.assert_called_once_with(stats)


def test_handle_plain_output(sample_stats, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    stats = replace(sample_stats, runtime_counts=(("opencode", 4),))
    monkeypatch.setattr(
        init_command.integrations,
        "install_skills",
        Mock(
            return_value=(
                ("claude", "already_installed"),
                ("opencode", "already_installed"),
                ("pi", "already_installed"),
            )
        ),
    )
    monkeypatch.setattr(init_command.api, "index", Mock(return_value=stats))
    _stub_mcp(monkeypatch, (("cursor", "installed", None),))
    monkeypatch.setattr(init_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(init_command, "is_quiet", lambda: False)

    assert command().handle() is None
    out = capsys.readouterr().out
    assert "Agent skills" in out
    assert "claude" in out and "already installed" in out
    assert "opencode" in out and "already installed" in out
    assert "pi" in out and "already installed" in out
    assert "MCP clients" in out
    assert "cursor" in out and "installed" in out
    assert "Sessions" in out and "1" in out
    assert "Runtimes" in out and "opencode=4" in out


def test_handle_plain_no_sources(sample_stats, monkeypatch, capsys) -> None:
    stats = replace(sample_stats, runtime_counts=())
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=()))
    monkeypatch.setattr(init_command.api, "index", Mock(return_value=stats))
    _stub_mcp(monkeypatch)
    monkeypatch.setattr(init_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(init_command, "is_quiet", lambda: False)

    command().handle()

    out = capsys.readouterr().out
    # The skill section is empty, the table shows up without Runtimes row.
    assert "Sessions" in out
    assert "Runtimes" not in out


def test_handle_quiet_suppresses_output(sample_stats, monkeypatch, capsys) -> None:
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=()))
    _stub_mcp(monkeypatch)
    monkeypatch.setattr(init_command.api, "index", Mock(return_value=sample_stats))
    monkeypatch.setattr(init_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(init_command, "is_quiet", lambda: True)

    assert command().handle() is None
    assert capsys.readouterr().out == ""


def test_skill_style_variants() -> None:
    """Every status branch in _skill_style returns the expected styled string."""
    from ssgrep.cli.commands.init_command import _skill_style

    assert _skill_style("installed") == "[green]installed[/green]"
    assert _skill_style("already_installed") == "[gray]already installed[/gray]"
    assert _skill_style("user_modified") == "[yellow]kept your edits[/yellow]"
    assert _skill_style("error: disk full") == "[red]error: disk full[/red]"
    assert _skill_style("something_else") == "something_else"


def test_index_not_ready_fails_with_missing_index_code(monkeypatch) -> None:
    error = IndexNotReadyError("schema changed")
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=()))
    _stub_mcp(monkeypatch)
    monkeypatch.setattr(init_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(init_command, "is_json_mode", lambda: False)

    class ExitCalled(RuntimeError):
        def __init__(self, code: int) -> None:
            self.code = code
            super().__init__(str(code))

    def hard_exit(code: int) -> None:
        raise ExitCalled(code)

    monkeypatch.setattr(init_command, "_fail", lambda error, code: hard_exit(code))

    with pytest.raises(ExitCalled) as caught:
        command().handle()
    assert caught.value.code == exit_codes.MISSING_INDEX


def test_unexpected_index_failure_exits_internal_and_json_reraises(
    sample_stats, monkeypatch
) -> None:
    error = RuntimeError("boom")
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=()))
    _stub_mcp(monkeypatch)
    monkeypatch.setattr(init_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(init_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle()
    assert caught.value.code == exit_codes.INTERNAL_FAILURE

    monkeypatch.setattr(init_command, "is_json_mode", lambda: True)
    with pytest.raises(RuntimeError, match="boom"):
        command().handle()
