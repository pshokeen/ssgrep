"""Tests for the version command."""

from __future__ import annotations

from importlib.metadata import version

from ssgrep.cli.commands import version_command


def command() -> version_command.VersionCommand:
    return object.__new__(version_command.VersionCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "version"
    assert instance.description() == "Display the installed ssgrep version"


def test_handle_prints_version_in_text_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(version_command, "is_json_mode", lambda: False)

    assert command().handle() is None
    assert capsys.readouterr().out == f"ssgrep {version('ssgrep')}\n"


def test_handle_returns_json_document(monkeypatch) -> None:
    monkeypatch.setattr(version_command, "is_json_mode", lambda: True)

    assert command().handle() == {"name": "ssgrep", "version": version("ssgrep")}
