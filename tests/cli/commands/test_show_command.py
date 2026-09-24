"""Tests for the episode detail command."""

from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import show_command
from ssgrep.utilities.types import EpisodeDetail, IndexNotFoundError, IndexNotReadyError


class ExitCalled(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(str(code))


def hard_exit(code: int) -> None:
    raise ExitCalled(code)


def command() -> show_command.ShowCommand:
    return object.__new__(show_command.ShowCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "show"
    assert instance.description() == "Show bounded prompt and response context for one episode"


def test_handle_renders_plain_detail(
    sample_detail: EpisodeDetail,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_show = Mock(return_value=sample_detail)
    render = Mock(return_value="full detail")
    monkeypatch.setattr(show_command.api, "show", api_show)
    monkeypatch.setattr(show_command, "render_episode_detail", render)
    monkeypatch.setattr(show_command, "is_json_mode", lambda: False)

    assert command().handle("session-1:ep:0") is None

    api_show.assert_called_once_with("session-1:ep:0")
    render.assert_called_once_with(sample_detail)
    assert capsys.readouterr().out == "\nfull detail\n"


def test_handle_returns_jsonable_detail(
    sample_detail: EpisodeDetail, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"episode_id": "serialized"}
    monkeypatch.setattr(show_command.api, "show", Mock(return_value=sample_detail))
    monkeypatch.setattr(show_command, "is_json_mode", lambda: True)
    serialize = Mock(return_value=payload)
    monkeypatch.setattr(show_command, "to_jsonable", serialize)

    assert command().handle("ref") is payload
    serialize.assert_called_once_with(sample_detail)


@pytest.mark.parametrize(
    "error",
    [IndexNotFoundError("missing"), IndexNotReadyError("not ready")],
)
def test_index_errors_plain(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(show_command.api, "show", Mock(side_effect=error))
    monkeypatch.setattr(show_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle("ref")

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert caught.value.__cause__ is error
    assert capsys.readouterr().err == f"{error}\n"


def test_index_error_json_writes_document_and_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    error = IndexNotReadyError("rebuild needed")
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(show_command.api, "show", Mock(side_effect=error))
    monkeypatch.setattr(show_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(show_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(show_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle("ref")

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "corrupt_index",
        "message": "rebuild needed",
        "command": "ssgrep index --rebuild",
    }
    exit_mock.assert_called_once_with(exit_codes.MISSING_INDEX)


def test_unknown_ref_plain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(show_command.api, "show", Mock(return_value=None))
    monkeypatch.setattr(show_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle("not-real")

    assert caught.value.code == exit_codes.NO_MATCHING_DATA
    assert capsys.readouterr().err == (
        "Episode not found: not-real. Run `ssgrep search <query>` to find valid refs.\n"
    )


def test_unknown_ref_json_writes_document_and_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(show_command.api, "show", Mock(return_value=None))
    monkeypatch.setattr(show_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(show_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(show_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle("not-real")

    assert caught.value.code == exit_codes.NO_MATCHING_DATA
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "unknown_ref",
        "message": ("Episode not found: not-real. Run `ssgrep search <query>` to find valid refs."),
        "command": "ssgrep search <query>",
    }
    exit_mock.assert_called_once_with(exit_codes.NO_MATCHING_DATA)
