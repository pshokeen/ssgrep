"""Tests for the durable note command."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import note_command
from ssgrep.utilities.types import IndexNotReadyError, IndexStats


class ExitCalled(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(str(code))


def hard_exit(code: int) -> None:
    raise ExitCalled(code)


def command() -> note_command.NoteCommand:
    return object.__new__(note_command.NoteCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "note"
    assert instance.description() == "Write a durable, searchable note into the global index"


def test_handle_rejects_blank_title(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_note = Mock()
    monkeypatch.setattr(note_command.notes, "write_note", write_note)

    with pytest.raises(SystemExit) as caught:
        command().handle(title="  ", body="content")

    assert caught.value.code == exit_codes.USAGE_ERROR
    assert "--title is required" in capsys.readouterr().err
    write_note.assert_not_called()


def test_handle_rejects_blank_body(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(note_command.sys, "stdin", io.StringIO(" \n"))

    with pytest.raises(SystemExit) as caught:
        command().handle(title="question", body="-")

    assert caught.value.code == exit_codes.USAGE_ERROR
    assert "--body is required" in capsys.readouterr().err


def test_handle_writes_stdin_note_reindexes_and_prints(
    tmp_path: Path,
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shard = tmp_path / "notes" / "note.jsonl"
    write_note = Mock(return_value=shard)
    index_api = Mock(return_value=sample_stats)
    resolved_project = tmp_path / "project"
    monkeypatch.setattr(note_command.sys, "stdin", io.StringIO("body from stdin\n"))
    resolve_mock = Mock(return_value=resolved_project)
    monkeypatch.setattr(note_command, "resolve_project_dir", resolve_mock)
    monkeypatch.setattr(note_command.notes, "write_note", write_note)
    monkeypatch.setattr(note_command.api, "index", index_api)
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(note_command, "is_quiet", lambda: False)

    result = command().handle(
        title="searchable question",
        body="-",
        project_dir="relative-project",
    )

    assert result is None
    resolve_mock.assert_called_once_with("relative-project")
    write_note.assert_called_once_with(resolved_project, "searchable question", "body from stdin\n")
    index_api.assert_called_once_with()
    assert capsys.readouterr().out == (
        f"Note written to {shard}\nIndexed 1 sessions, 1 episodes, 2 chunks.\n"
    )


def test_handle_returns_json_document(
    tmp_path: Path, sample_stats: IndexStats, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "note.jsonl"
    serialized = {"session_count": 1}
    monkeypatch.setattr(note_command.notes, "write_note", Mock(return_value=shard))
    monkeypatch.setattr(note_command.api, "index", Mock(return_value=sample_stats))
    monkeypatch.setattr(note_command, "is_json_mode", lambda: True)
    serialize = Mock(return_value=serialized)
    monkeypatch.setattr(note_command, "to_jsonable", serialize)

    result = command().handle(title="question", body="body")

    assert result == {"ok": True, "note_file": str(shard), "index": serialized}
    serialize.assert_called_once_with(sample_stats)


def test_handle_quiet_success_has_no_output(
    tmp_path: Path,
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        note_command.notes, "write_note", Mock(return_value=tmp_path / "note.jsonl")
    )
    monkeypatch.setattr(note_command.api, "index", Mock(return_value=sample_stats))
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(note_command, "is_quiet", lambda: True)

    assert command().handle(title="question", body="body") is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_handle_plain_reindex_failure_explains_note_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shard = tmp_path / "note.jsonl"
    error = RuntimeError("database unavailable")
    monkeypatch.setattr(note_command.notes, "write_note", Mock(return_value=shard))
    monkeypatch.setattr(note_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle(title="question", body="body")

    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    assert caught.value.__cause__ is error
    stderr = capsys.readouterr().err
    assert f"Note written to {shard}" in stderr
    assert "reindexing failed: database unavailable" in stderr
    assert "note is saved" in stderr


def test_handle_json_reindex_failure_writes_document_and_hard_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "note.jsonl"
    error = IndexNotReadyError("database unavailable")
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(note_command.notes, "write_note", Mock(return_value=shard))
    monkeypatch.setattr(note_command.api, "index", Mock(side_effect=error))
    monkeypatch.setattr(note_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(note_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(note_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle(title="question", body="body")

    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "corrupt_index",
        "message": f"Note written to {shard}, but reindexing failed: database unavailable",
        "note_file": str(shard),
        "command": "ssgrep index --rebuild",
    }
    exit_mock.assert_called_once_with(exit_codes.INTERNAL_FAILURE)
