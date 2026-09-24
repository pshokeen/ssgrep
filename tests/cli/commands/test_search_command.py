"""Tests for the search CLI command."""

from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import search_command
from ssgrep.utilities.types import (
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    InvalidPredicateError,
    SearchResponse,
)


class ExitCalled(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(str(code))


def hard_exit(code: int) -> None:
    raise ExitCalled(code)


def command() -> search_command.SearchCommand:
    return object.__new__(search_command.SearchCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "search"
    assert instance.description() == "Search transcripts across all projects"


def test_handle_forwards_options_and_renders_plain_response(
    sample_response: SearchResponse,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_search = Mock(return_value=sample_response)
    render = Mock(return_value="rendered cards")
    monkeypatch.setattr(search_command.api, "search", api_search)
    monkeypatch.setattr(search_command, "render_search_response", render)
    monkeypatch.setattr(search_command, "is_json_mode", lambda: False)

    result = command().handle("pytest", limit=4, token_budget=900, where="project = 'safe'")

    assert result is None
    api_search.assert_called_once_with(
        "pytest", limit=4, token_budget=900, where="project = 'safe'"
    )
    render.assert_called_once_with(sample_response, query="pytest")
    assert capsys.readouterr().out == "\nrendered cards\n"


def test_handle_returns_jsonable_response(
    sample_response: SearchResponse, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"results": ["serialized"]}
    monkeypatch.setattr(search_command.api, "search", Mock(return_value=sample_response))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: True)
    serialize = Mock(return_value=payload)
    monkeypatch.setattr(search_command, "to_jsonable", serialize)

    assert command().handle("pytest") is payload
    serialize.assert_called_once_with(sample_response)


@pytest.mark.parametrize(
    "error",
    [EmptyQueryError("empty"), InvalidPredicateError("bad where")],
)
def test_usage_errors_plain(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(search_command.api, "search", Mock(side_effect=error))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle("query")

    assert caught.value.code == exit_codes.USAGE_ERROR
    assert caught.value.__cause__ is error
    assert capsys.readouterr().err == f"{error}\n"


def test_usage_error_json_writes_document_and_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    error = EmptyQueryError("query cannot be blank", command="ssgrep search QUERY")
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(search_command.api, "search", Mock(side_effect=error))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(search_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(search_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle(" ")

    assert caught.value.code == exit_codes.USAGE_ERROR
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "empty_query",
        "message": "query cannot be blank",
        "command": "ssgrep search QUERY",
    }
    exit_mock.assert_called_once_with(exit_codes.USAGE_ERROR)


@pytest.mark.parametrize(
    "error",
    [IndexNotFoundError("missing"), IndexNotReadyError("not ready")],
)
def test_index_errors_plain(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(search_command.api, "search", Mock(side_effect=error))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle("query")

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert caught.value.__cause__ is error
    assert capsys.readouterr().err == f"{error}\n"


def test_index_error_json_writes_document_and_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    error = IndexNotReadyError("rebuild needed")
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(search_command.api, "search", Mock(side_effect=error))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(search_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(search_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle("query")

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "corrupt_index",
        "message": "rebuild needed",
        "command": "ssgrep index --rebuild",
    }
    exit_mock.assert_called_once_with(exit_codes.MISSING_INDEX)


def test_empty_index_json_writes_diagnostic_and_hard_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SearchResponse(results=[], index_empty=True, total_matches=0)
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(search_command.api, "search", Mock(return_value=response))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(search_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(search_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle("unmatched")

    assert caught.value.code == exit_codes.NO_MATCHING_DATA
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "index_empty",
        "message": "The global index contains no searchable chunks.",
        "command": "ssgrep index",
    }
    exit_mock.assert_called_once_with(exit_codes.NO_MATCHING_DATA)


def test_empty_index_plain_reports_specific_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = SearchResponse(results=[], index_empty=True, total_matches=0)
    monkeypatch.setattr(search_command.api, "search", Mock(return_value=response))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle("unmatched")

    assert caught.value.code == exit_codes.NO_MATCHING_DATA
    assert capsys.readouterr().err == "The global index contains no searchable chunks.\n"


def test_no_matches_plain_reports_no_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    response = SearchResponse(results=[], index_empty=False, total_matches=0)
    monkeypatch.setattr(search_command.api, "search", Mock(return_value=response))
    monkeypatch.setattr(search_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle("unmatched")

    assert caught.value.code == exit_codes.NO_MATCHING_DATA
    assert capsys.readouterr().err == "No results found for this query.\n"
