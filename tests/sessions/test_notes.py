"""Unit tests for native note writing and discovery."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.sessions import notes


class _FixedDateTime:
    @classmethod
    def now(cls, timezone):
        assert timezone is UTC
        return datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)


class _FixedUuid:
    hex = "123456789abcEXTRA"


def test_notes_dir_uses_isolated_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "private-data"))
    assert notes.notes_dir() == tmp_path / "private-data" / "notes"


@pytest.mark.parametrize(
    ("title", "body", "message"),
    [
        ("", "body", "title"),
        ("  \t", "body", "title"),
        ("title", "", "body"),
        ("title", " \n ", "body"),
    ],
)
def test_write_note_rejects_blank_fields(tmp_path: Path, title: str, body: str, message: str):
    with pytest.raises(ValueError, match=message):
        notes.write_note(tmp_path, title, body)


def test_write_note_creates_and_appends_native_record_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(notes, "datetime", _FixedDateTime)
    monkeypatch.setattr(notes.uuid, "uuid4", lambda: _FixedUuid())
    locks: list[int] = []
    monkeypatch.setattr(notes.fcntl, "flock", lambda _fd, operation: locks.append(operation))

    first = notes.write_note(project, "How is retry configured?", "Use jitter.")
    second = notes.write_note(project, "What is the cap?", "Thirty seconds.")

    assert first == second == notes.notes_dir() / "notes-202502.jsonl"
    assert first.exists()
    assert first.parent.stat().st_mode & 0o777 == 0o700
    assert locks == [notes.fcntl.LOCK_EX, notes.fcntl.LOCK_UN] * 2

    records = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(records) == 4
    user_record, assistant_record = records[:2]
    assert user_record == {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": "How is retry configured?"},
        "uuid": "note-123456789abc-u",
        "timestamp": "2025-02-03T04:05:06Z",
        "cwd": str(project),
        "sessionId": "note-202502",
        "gitBranch": "",
    }
    assert assistant_record["parentUuid"] == user_record["uuid"]
    assert assistant_record["type"] == "assistant"
    assert assistant_record["message"] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Use jitter."}],
    }
    assert assistant_record["uuid"] == "note-123456789abc-a"
    assert assistant_record["timestamp"] == user_record["timestamp"]
    assert assistant_record["cwd"] == str(project)
    assert assistant_record["sessionId"] == "note-202502"
    assert assistant_record["gitBranch"] == ""
    assert records[2]["message"]["content"] == "What is the cap?"
    assert records[3]["message"]["content"][0]["text"] == "Thirty seconds."


def test_discover_notes_missing_and_sorted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "note-root"
    monkeypatch.setattr(notes, "notes_dir", lambda: root)
    assert notes.discover_notes() == []

    root.mkdir()
    later = root / "notes-202512.jsonl"
    earlier = root / "notes-202501.jsonl"
    ignored = root / "other.jsonl"
    for path in (later, earlier, ignored):
        path.write_text("{}\n")

    found = notes.discover_notes()
    assert [session.path for session in found] == [earlier, later]
    assert [session.session_id for session in found] == ["notes-202501", "notes-202512"]
    assert all(session.is_main for session in found)
    assert all(session.parent_session_id is None for session in found)
