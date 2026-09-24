"""Tests for the index CLI command."""

from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import index_command
from ssgrep.utilities.types import (
    IndexNotReadyError,
    IndexStats,
    RebuildWouldShrinkError,
    SearchException,
)


class ExitCalled(RuntimeError):
    """Stand in for os._exit without terminating pytest."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"os._exit({code})")


def hard_exit(code: int) -> None:
    raise ExitCalled(code)


def command() -> index_command.IndexCommand:
    return object.__new__(index_command.IndexCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "index"
    assert instance.description() == "Build or update the global all-project session index"


def test_fail_plain_writes_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = SearchException("broken", condition="bad", command="repair")
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        index_command._fail(error, 7)

    assert caught.value.code == 7
    assert capsys.readouterr().err == "broken\n"


def test_fail_json_writes_and_flushes_before_hard_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    error = SearchException("broken", condition="bad", command="repair")
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(index_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(index_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(index_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        index_command._fail(error, 7)

    assert caught.value.code == 7
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "bad",
        "message": "broken",
        "command": "repair",
    }
    exit_mock.assert_called_once_with(7)


def test_handle_forwards_options_and_prints_stats_table(
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_index = Mock(return_value=sample_stats)
    monkeypatch.setattr(index_command.api, "index", api_index)
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(index_command, "is_quiet", lambda: False)

    result = command().handle(
        rebuild=True,
        no_subagents=True,
        quiet=False,
        allow_shrink=True,
        scope="/safe/scope",
        live=True,
        full_reprocess=True,
    )

    assert result is None
    api_index.assert_called_once_with(
        rebuild=True,
        no_subagents=True,
        allow_shrink=True,
        scope="/safe/scope",
        quiet=False,
        live=True,
        full_reprocess=True,
    )
    output = capsys.readouterr().out
    assert "Sessions" in output and "1" in output
    assert "Episodes" in output and "1" in output
    assert "Chunks" in output and "2" in output
    assert "lightonai/answerai-colbert-small-v1 (dim 96)" in output
    assert "0 skipped, 0 malformed" in output


@pytest.mark.parametrize(
    ("explicit_quiet", "runtime_quiet"),
    [(True, False), (False, True)],
)
def test_handle_suppresses_summary_when_quiet(
    explicit_quiet: bool,
    runtime_quiet: bool,
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(index_command.api, "index", lambda **_kwargs: sample_stats)
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(index_command, "is_quiet", lambda: runtime_quiet)

    assert command().handle(quiet=explicit_quiet) is None
    assert capsys.readouterr().out == ""


def test_handle_returns_jsonable_stats(
    sample_stats: IndexStats, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"serialized": True}
    monkeypatch.setattr(index_command.api, "index", lambda **_kwargs: sample_stats)
    monkeypatch.setattr(index_command, "is_json_mode", lambda: True)
    serialize = Mock(return_value=payload)
    monkeypatch.setattr(index_command, "to_jsonable", serialize)

    assert command().handle() is payload
    serialize.assert_called_once_with(sample_stats)


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (
            RebuildWouldShrinkError("refused", old_counts=(10, 20, 30), new_counts=(1, 2, 3)),
            exit_codes.USAGE_ERROR,
        ),
        (IndexNotReadyError("not ready"), exit_codes.MISSING_INDEX),
    ],
)
def test_handle_maps_known_failures(
    error: Exception,
    expected_code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(index_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle()

    assert caught.value.code == expected_code
    assert str(error) in capsys.readouterr().err


def test_handle_reraises_unexpected_error_in_json_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("surprise")
    monkeypatch.setattr(index_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(index_command, "is_json_mode", lambda: True)

    with pytest.raises(RuntimeError, match="surprise"):
        command().handle()


def test_handle_reports_unexpected_error_and_rebuild_advice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = RuntimeError("surprise")
    monkeypatch.setattr(index_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle(rebuild=False)

    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    assert caught.value.__cause__ is error
    assert capsys.readouterr().err == (
        "Index operation failed: surprise\nTry `ssgrep index --rebuild` to force a full rebuild.\n"
    )


def test_handle_omits_rebuild_advice_for_rebuild_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(index_command.api, "index", Mock(side_effect=RuntimeError("again")))
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit):
        command().handle(rebuild=True)

    assert capsys.readouterr().err == "Index operation failed: again\n"
