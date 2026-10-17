"""Tests for the restored multi-runtime init command."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import init_command, mcp_command
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


def test_invalid_launcher_env_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The real (unmocked) installer rejects a bad SSGREP_MCP_LAUNCHER cleanly, not a traceback."""
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=()))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SSGREP_MCP_LAUNCHER", "bogus")
    monkeypatch.setattr(init_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as raised:
        command().handle()

    assert raised.value.code == exit_codes.USAGE_ERROR
    err = capsys.readouterr().err
    assert "Usage error" in err
    assert "SSGREP_MCP_LAUNCHER" in err
    assert "bogus" in err
    assert not (tmp_path / ".cursor" / "mcp.json").exists()


def test_invalid_launcher_env_is_a_usage_error_in_json_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=()))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SSGREP_MCP_LAUNCHER", "bogus")
    monkeypatch.setattr(init_command, "is_json_mode", lambda: True)

    with pytest.raises(SystemExit) as raised:
        command().handle()

    assert raised.value.code == exit_codes.USAGE_ERROR
    err = capsys.readouterr().err
    assert "SSGREP_MCP_LAUNCHER" in err
    assert "bogus" in err


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


def _stub_for_failures(
    monkeypatch: pytest.MonkeyPatch, sample_stats, *, skills: tuple, mcp: tuple, json_mode: bool
) -> None:
    stats = replace(sample_stats, runtime_counts=(("claude", 2),))
    monkeypatch.setattr(init_command.integrations, "install_skills", Mock(return_value=skills))
    _stub_mcp(monkeypatch, mcp)
    monkeypatch.setattr(init_command.api, "index", Mock(return_value=stats))
    monkeypatch.setattr(init_command, "is_json_mode", lambda: json_mode)
    monkeypatch.setattr(init_command, "is_quiet", lambda: False)


def test_failed_mcp_client_is_reported_with_everything_else_then_exits_nonzero(
    sample_stats, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _stub_for_failures(
        monkeypatch,
        sample_stats,
        skills=(("claude", "installed"),),
        mcp=(
            ("claude", "skipped: claude CLI not found", None),
            ("zed", "error: settings.json is not strict JSON", Path("/cfg/zed.json")),
            ("cursor", "installed", Path("/cfg/cursor.json")),
        ),
        json_mode=False,
    )

    with pytest.raises(SystemExit) as raised:
        command().handle()

    assert raised.value.code == exit_codes.INTERNAL_FAILURE
    captured = capsys.readouterr()
    assert "not strict JSON" in captured.out  # every client is still reported
    assert "cursor" in captured.out and "Sessions" in captured.out
    assert captured.err == "registration failed for: mcp zed\n"


def test_failed_skill_install_also_exits_nonzero(
    sample_stats, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _stub_for_failures(
        monkeypatch,
        sample_stats,
        skills=(("claude", "error: [Errno 13] Permission denied"), ("pi", "installed")),
        mcp=(("cursor", "installed", None),),
        json_mode=False,
    )

    with pytest.raises(SystemExit) as raised:
        command().handle()

    assert raised.value.code == exit_codes.INTERNAL_FAILURE
    assert capsys.readouterr().err == "registration failed for: skill claude\n"


def test_failed_client_json_document_is_ok_false_and_complete(
    sample_stats, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_for_failures(
        monkeypatch,
        sample_stats,
        skills=(("claude", "installed"),),
        mcp=(("zed", "error: bad json", Path("/cfg/zed.json")),),
        json_mode=True,
    )
    monkeypatch.setattr(
        init_command, "to_jsonable", lambda stats: {"sessions": stats.session_count}
    )
    stdout = io.StringIO()
    monkeypatch.setattr(init_command.sys, "__stdout__", stdout)
    exit_mock = Mock(side_effect=SystemExit(99))
    monkeypatch.setattr(mcp_command.os, "_exit", exit_mock)
    monkeypatch.setattr(mcp_command, "is_json_mode", lambda: True)

    with pytest.raises(SystemExit):
        command().handle()

    exit_mock.assert_called_once_with(exit_codes.INTERNAL_FAILURE)
    document = json.loads(stdout.getvalue())
    assert document["ok"] is False
    assert document["condition"] == "client_registration_failed"
    assert document["message"] == "registration failed for: mcp zed"
    assert document["skills"] == {"claude": "installed"}
    assert document["mcp"] == {"zed": {"status": "error: bad json", "config": "/cfg/zed.json"}}
    assert document["sources"] == {"claude": 2}
    assert document["index"] == {"sessions": sample_stats.session_count}
