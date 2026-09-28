"""Tests for the tombstone pruning command."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import prune_command
from ssgrep.pipeline.sources import read_registry
from ssgrep.services import api
from ssgrep.store import (
    CHUNKS_TABLE,
    CURSORS_TABLE,
    EPISODES_TABLE,
    SESSIONS_TABLE,
    SOURCES_TABLE,
    LanceStore,
)
from tests.pipeline.test_app import fake_models as fake_models, write_claude_session


class ExitCalled(RuntimeError):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(str(code))


def hard_exit(code: int) -> None:
    raise ExitCalled(code)


def command() -> prune_command.PruneCommand:
    return object.__new__(prune_command.PruneCommand)


def session(
    session_id: str = "session-1",
    *,
    path: str = "/safe/transcript.jsonl",
    chunks: int = 2,
    episodes: int = 1,
    absent_since: float | None = 1.0,
) -> prune_command.TombstonedSession:
    return {
        "session_id": session_id,
        "path": path,
        "chunk_count": chunks,
        "episode_count": episodes,
        "absent_since": absent_since,
    }


def repository(tmp_path: Path, *, exists: bool = True) -> MagicMock:
    result = MagicMock()
    result.root = tmp_path
    result.exists.return_value = exists
    return result


def install_repository(monkeypatch: pytest.MonkeyPatch, value: MagicMock) -> Mock:
    constructor = Mock(return_value=value)
    monkeypatch.setattr(prune_command, "LanceStore", constructor)
    return constructor


def test_metadata() -> None:
    instance = command()

    assert instance.visible() is True
    assert instance.signature() == "prune"
    assert (
        instance.description()
        == "Permanently delete globally indexed content whose source vanished"
    )


def test_tombstoned_sessions_normalizes_dates_and_counts(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    when = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    repo.rows.return_value = [
        {"session_id": "datetime", "path": "/a", "absent_since": when},
        {"session_id": "string", "path": "/b", "absent_since": when.isoformat()},
        {"session_id": "missing", "path": "/c", "absent_since": 123},
    ]

    def count_rows(table: str, predicate: str) -> int:
        assert "session_id" in predicate
        return 3 if table == CHUNKS_TABLE else 2

    repo.count.side_effect = count_rows

    result = prune_command._tombstoned_sessions(repo)

    repo.rows.assert_called_once_with(
        SESSIONS_TABLE, where="source_status = 'absent'", limit=100_000
    )
    assert [item["session_id"] for item in result] == ["datetime", "string", "missing"]
    assert result[0]["absent_since"] == when.timestamp()
    assert result[1]["absent_since"] == when.timestamp()
    assert result[2]["absent_since"] is None
    assert all(item["chunk_count"] == 3 for item in result)
    assert all(item["episode_count"] == 2 for item in result)


def test_filter_by_age_returns_all_for_nonpositive_threshold() -> None:
    sessions = [session(absent_since=None), session("session-2", absent_since=10.0)]

    assert prune_command._filter_by_age(sessions, 0) is sessions
    assert prune_command._filter_by_age(sessions, -2) is sessions


def test_filter_by_age_excludes_recent_and_unknown_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prune_command.time, "time", lambda: 10 * prune_command._DAY_SECONDS)
    sessions = [
        session("old", absent_since=1.0),
        session("recent", absent_since=9 * prune_command._DAY_SECONDS + 1),
        session("unknown", absent_since=None),
    ]

    assert prune_command._filter_by_age(sessions, 2) == [sessions[0]]


def test_missing_index_plain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path, exists=False)
    constructor = install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: False)

    with pytest.raises(SystemExit) as caught:
        command().handle()

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert caught.value.__cause__ is not None
    assert "No global index found" in capsys.readouterr().err
    constructor.assert_called_once_with()


def test_missing_index_json_hard_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path, exists=False)
    stdout = io.StringIO()
    exit_mock = Mock(side_effect=hard_exit)
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(prune_command.sys, "__stdout__", stdout)
    monkeypatch.setattr(prune_command.os, "_exit", exit_mock)

    with pytest.raises(ExitCalled) as caught:
        command().handle()

    assert caught.value.code == exit_codes.MISSING_INDEX
    assert json.loads(stdout.getvalue()) == {
        "ok": False,
        "condition": "missing_index",
        "message": "No global index found. Run `ssgrep index` first.",
        "command": "ssgrep index",
    }
    exit_mock.assert_called_once_with(exit_codes.MISSING_INDEX)


def test_empty_selection_plain_reports_nothing_to_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path)
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", lambda _repo: [])
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: False)

    assert command().handle(older_than=30) is None
    assert capsys.readouterr().err == "Nothing to prune: no tombstoned content matches.\n"


def test_dry_run_plain_prints_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path)
    sessions = [session(), session("session-2", chunks=4, episodes=2)]
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", lambda _repo: sessions)
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: False)

    assert command().handle(dry_run=True) is None
    assert capsys.readouterr().err == "Would prune 2 sessions (6 chunks).\n"
    repo.delete.assert_not_called()


def test_dry_run_json_returns_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path)
    sessions = [session(absent_since=5.0)]
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", lambda _repo: sessions)
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: True)

    result = command().handle(older_than=0, dry_run=True)

    assert result == {
        "ok": True,
        "dry_run": True,
        "older_than": 0,
        "session_count": 1,
        "chunk_count": 2,
        "episode_count": 1,
        "sessions": [
            {
                "session_id": "session-1",
                "path": "/safe/transcript.jsonl",
                "chunk_count": 2,
                "episode_count": 1,
            }
        ],
    }


@pytest.mark.parametrize("json_mode", [False, True])
def test_noninteractive_prune_requires_yes(
    json_mode: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path)
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", lambda _repo: [session()])
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: json_mode)
    monkeypatch.setattr(prune_command.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    with pytest.raises(SystemExit) as caught:
        command().handle(yes=False)

    assert caught.value.code == exit_codes.USAGE_ERROR
    assert capsys.readouterr().err == "Prune requires --yes when non-interactive.\n"


def test_interactive_rejection_cancels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path)
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", lambda _repo: [session()])
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(prune_command.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", Mock(return_value="no"))

    assert command().handle(yes=False) is None
    assert capsys.readouterr().err == "Cancelled.\n"
    repo.delete.assert_not_called()


def test_interactive_confirmation_revalidates_and_deletes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = repository(tmp_path)
    initial = [session("before")]
    current = [session("current", path="/current.jsonl")]
    tombstones = Mock(side_effect=[initial, current])
    flock = Mock()
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", tombstones)
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: False)
    monkeypatch.setattr(prune_command.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", Mock(return_value=" YES "))
    monkeypatch.setattr(prune_command.fcntl, "flock", flock)

    assert command().handle(yes=False) is None

    assert tombstones.call_count == 2
    flock.assert_called_once()
    predicate = f"session_id = {prune_command.quote('current')}"
    assert repo.delete.call_args_list == [
        call(CHUNKS_TABLE, predicate),
        call(EPISODES_TABLE, predicate),
        call(SESSIONS_TABLE, predicate),
        call(CURSORS_TABLE, f"path = {prune_command.quote('/current.jsonl')}"),
        call(SOURCES_TABLE, f"key = {prune_command.quote('/current.jsonl')}"),
    ]
    assert capsys.readouterr().err == "Pruned 1 tombstoned sessions.\n"


def test_confirmed_json_prune_returns_revalidated_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = repository(tmp_path)
    initial = [session("old", chunks=9, episodes=8)]
    current = [session("current", chunks=3, episodes=2, absent_since=10.0)]
    install_repository(monkeypatch, repo)
    monkeypatch.setattr(prune_command, "_tombstoned_sessions", Mock(side_effect=[initial, current]))
    monkeypatch.setattr(prune_command, "is_json_mode", lambda: True)
    monkeypatch.setattr(prune_command.fcntl, "flock", Mock())

    result = command().handle(yes=True)

    assert result is not None
    assert result["dry_run"] is False
    assert result["session_count"] == 1
    assert result["chunk_count"] == 3
    assert result["episode_count"] == 2
    assert result["sessions"] == [
        {
            "session_id": "current",
            "path": "/safe/transcript.jsonl",
            "chunk_count": 3,
            "episode_count": 2,
        }
    ]


def test_pruning_a_tombstoned_source_lets_a_later_reindex_succeed(
    fake_models, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for ssgrep#5's fail-closed lockout (real store, no mocks).

    Before this fix, ``prune`` deleted a source's data rows but left its
    ``sources`` registry row behind. That zombie entry is exactly the shape
    ``pipeline/archive.py``'s ``retained_rows`` fails closed on -- and
    ``capture_archives`` scans it on EVERY subsequent reconcile, not only one
    following a memo-invalidating pipeline-code change -- so a single pruned
    source would lock every later ``ssgrep index``/``note``/MCP-startup run
    for the WHOLE corpus. This exercises the real ``LanceStore`` and the real
    ``PruneCommand``, not a mocked delete spy.
    """
    pruned_path = write_claude_session("pruned")
    write_claude_session("live")
    api.index()
    pruned_path.unlink()
    api.index()  # tombstones "pruned": rows retained, source_status='absent'

    repo = LanceStore()
    registry_key = str(pruned_path.absolute())
    assert registry_key in read_registry(repo)
    assert (
        repo.rows(SESSIONS_TABLE, where="session_id = 'pruned'", limit=1)[0]["source_status"]
        == "absent"
    )

    monkeypatch.setattr(prune_command, "is_json_mode", lambda: False)
    assert command().handle(yes=True) is None

    # The fix: the pruned source's registry row is gone, not just its data.
    assert registry_key not in read_registry(repo)
    assert repo.count(SESSIONS_TABLE, "session_id = 'pruned'") == 0

    # Without the fix, this next reconcile raises ValueError("... incomplete
    # archive (session identity mismatch)") because capture_archives() still
    # finds the pruned key in the registry (union_descriptors merges registry
    # + fresh discovery) but its data rows are gone.
    stats = api.index()
    assert stats.session_count == 1
    assert stats.tombstoned_source_count == 0
    live_rows = repo.rows(EPISODES_TABLE, where="session_id = 'live'", limit=100)
    assert live_rows and any("exponential backoff" in row["response_text"] for row in live_rows)
