"""Tests for index status output."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import status_command
from ssgrep.utilities.types import IndexNotReadyError, IndexStats


class ExitCalled(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(str(code))


def hard_exit(code: int) -> None:
    raise ExitCalled(code)


def command() -> status_command.StatusCommand:
    return object.__new__(status_command.StatusCommand)


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "status"
    assert instance.description() == "Show global index counts, size, model, and archive status"


def test_handle_returns_jsonable_stats(
    sample_stats: IndexStats, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"status": "serialized"}
    monkeypatch.setattr(status_command.api, "status", Mock(return_value=sample_stats))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: True)
    serialize = Mock(return_value=payload)
    monkeypatch.setattr(status_command, "to_jsonable", serialize)

    assert command().handle() is payload
    serialize.assert_called_once_with(sample_stats)


def test_missing_index_is_successful_and_actionable(
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stats = replace(sample_stats, index_exists=False)
    monkeypatch.setattr(status_command.api, "status", Mock(return_value=stats))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: False)

    assert command().handle() is None
    assert capsys.readouterr().out == "No global index found. Run `ssgrep index` to build it.\n"


def test_populated_status_prints_every_field_and_tombstones(
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stats = replace(
        sample_stats,
        index_size_bytes=1_572_864,
        tombstoned_source_count=3,
        tombstoned_chunk_count=7,
        runtime_counts=(("claude", 12), ("opencode", 3)),
    )
    monkeypatch.setattr(status_command.api, "status", Mock(return_value=stats))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: False)

    assert command().handle() is None

    output = capsys.readouterr().out
    assert "Sessions" in output and "1" in output
    assert "Episodes" in output and "1" in output
    assert "Chunks" in output and "2" in output
    assert "1572864 bytes (1.5 MiB)" in output
    assert "2025-01-02T03:04:00+00:00" in output
    assert "lightonai/answerai-colbert-small-v1 (dim 96)" in output
    assert "0 skipped, 0 malformed" in output
    assert "claude=12, opencode=3" in output
    assert "3 sources, 7 chunks" in output


def test_status_with_no_index_time_prints_never_without_tombstone_line(
    sample_stats: IndexStats,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stats = replace(sample_stats, last_index_time=None, tombstoned_chunk_count=0)
    monkeypatch.setattr(status_command.api, "status", Mock(return_value=stats))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: False)

    assert command().handle() is None

    output = capsys.readouterr().out
    assert "Indexed" in output and "never" in output
    assert "Tombstone" not in output


def test_plain_status_failure_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = RuntimeError("unavailable")
    monkeypatch.setattr(status_command.api, "status", Mock(side_effect=error))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle()

    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    assert caught.value.__cause__ is error
    assert capsys.readouterr().err == "Status failed: unavailable\n"


def test_json_status_failure_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    error = RuntimeError("unavailable")
    monkeypatch.setattr(status_command.api, "status", Mock(side_effect=error))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: True)

    with pytest.raises(RuntimeError, match="unavailable"):
        command().handle()


def test_stale_index_plain_exits_missing_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = IndexNotReadyError(
        "The global index schema or embedding model is incompatible; run `ssgrep index --rebuild`."
    )
    monkeypatch.setattr(status_command.api, "status", Mock(side_effect=error))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle()

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert caught.value.__cause__ is error
    assert capsys.readouterr().err == f"{error}\n"


def test_stale_index_json_writes_document_and_hard_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = IndexNotReadyError(
        "The global index schema or embedding model is incompatible; run `ssgrep index --rebuild`."
    )
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    monkeypatch.setattr(status_command.api, "status", Mock(side_effect=error))
    monkeypatch.setattr(status_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(status_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(status_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle()

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "corrupt_index",
        "message": str(error),
        "command": "ssgrep index --rebuild",
    }
    exit_mock.assert_called_once_with(exit_codes.MISSING_INDEX)
