"""Tests for the `ssgrep rules` command.

`rules` is the surface that carries ssgrep's operating rules to harnesses that
do not load skills, so the properties pinned here are the ones a consumer
depends on: it works with nothing installed but the wheel, its three output
surfaces agree exactly, and it never leaks the template placeholder.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from ssgrep.cli.commands.rules_command import RulesCommand
from ssgrep.services import integrations


def _command() -> RulesCommand:
    return RulesCommand(typer.Typer())


def test_rules_prints_the_full_guidance_body(capsys) -> None:
    _command().handle()

    printed = capsys.readouterr().out
    assert printed == integrations.guidance_body()
    assert "Operating rules" in printed


def test_rules_short_prints_only_the_fenced_block(capsys) -> None:
    _command().handle(short=True)

    printed = capsys.readouterr().out
    assert printed == integrations.guidance_short()
    assert "<!--" not in printed
    assert "## Commands" not in printed


def test_rules_never_leaks_the_template_placeholder(capsys) -> None:
    _command().handle()

    assert "SSGREP_BIN_PLACEHOLDER" not in capsys.readouterr().out


def test_rules_json_payload_carries_provenance_and_both_forms(monkeypatch) -> None:
    from usecli.cli.core import runtime

    monkeypatch.setattr(runtime, "is_json_mode", lambda: True)
    monkeypatch.setattr("ssgrep.cli.commands.rules_command.is_json_mode", lambda: True)

    payload = _command().handle()

    assert payload is not None
    assert set(payload) == {"version", "executable", "short", "full"}
    assert payload["short"] == integrations.guidance_short()
    assert payload["full"] == integrations.guidance_body()
    assert payload["version"] == integrations.version("ssgrep")
    assert "SSGREP_BIN_PLACEHOLDER" not in str(payload["full"])


def test_rules_touches_no_index_model_or_data_directory(tmp_path, monkeypatch, capsys) -> None:
    """It must work on a fresh install: no index, no model, no data dir."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(data_dir))

    _command().handle()

    assert capsys.readouterr().out
    assert not data_dir.exists()


def test_rules_module_pulls_in_no_index_or_model_machinery() -> None:
    """A fresh interpreter importing the command loads none of the heavy stack."""
    probe = (
        "import sys;"
        "import ssgrep.cli.commands.rules_command;"
        "heavy=[m for m in ('lancedb','torch','pylate','sentence_transformers',"
        "'ssgrep.services.api','ssgrep.store') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        check=True,
    )

    assert result.stdout.strip() == ""


def test_rules_is_visible_in_help_with_a_description() -> None:
    command = _command()

    assert command.visible() is True
    assert command.signature() == "rules"
    assert "rules" in command.description().lower()


def test_rules_was_not_added_to_the_mcp_tool_surface() -> None:
    """`rules` is a CLI surface; the MCP server stays read-only and unchanged.

    (The exact registration list is pinned in tests/services/test_mcp_server.py;
    this guards only that THIS change did not grow that surface.)
    """
    from ssgrep.services import mcp_server

    exported = {
        name
        for name, value in vars(mcp_server).items()
        if callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", "") == mcp_server.__name__
    }

    assert {"search_sessions", "show_session", "index_status"} <= exported
    assert not any("rule" in name for name in exported)
