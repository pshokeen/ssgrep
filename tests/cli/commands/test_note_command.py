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


def test_handle_note_write_failure_records_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ssgrep.accounting.capture import journal, journal_reader
    from ssgrep.accounting.ledger.contracts import Outcome, Phase
    from ssgrep.accounting.privacy import policy

    policy.enable_collection("personal-local")
    error = OSError("note storage unavailable")
    monkeypatch.setattr(note_command.notes, "write_note", Mock(side_effect=error))
    try:
        with pytest.raises(OSError) as caught:
            command().handle(title="question", body="body")
        assert caught.value is error
        events = journal_reader.read_journal("personal-local", quarantine=False).events
        assert [event.phase for event in events] == [Phase.STARTED, Phase.COMPLETED]
        assert events[-1].outcome is Outcome.ERROR
        assert events[0].execution_id == events[-1].execution_id
    finally:
        journal.close_all()


def test_note_with_work_metadata_reports_a_storage_failure_without_raw_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from ssgrep.accounting.privacy import policy
    from ssgrep.sessions import notes

    policy.enable_collection("personal-local")
    metadata = tmp_path / "work.json"
    metadata.write_text(json.dumps({"work_id": "task-1", "elapsed_seconds": 90}))
    monkeypatch.setattr(
        notes, "append_note", Mock(side_effect=OSError("/private/root unavailable"))
    )
    with pytest.raises(SystemExit) as caught:
        command().handle(title="question", body="answer", work=str(metadata))
    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    err = capsys.readouterr().err
    assert "note could not be saved (OSError)" in err and "/private/root" not in err


def test_legacy_damaged_note_lines_do_not_shift_new_refs_or_block_saves(tmp_path):
    from ssgrep.sessions import notes
    from ssgrep.sessions.adapters.base import read_jsonl
    from ssgrep.sessions.episodes import segment_episode_groups

    shard = notes.write_note(tmp_path, "old question", "old answer")
    original = shard.read_bytes()
    damaged = b'42\n{"type":"user","message":{"content":"' + b"x" * 500001 + b'"}}\n\xff\n{broken'
    with shard.open("ab") as stream:
        stream.write(damaged)
    saved, ref = notes.append_note(tmp_path, "new question", "new answer")
    episodes = segment_episode_groups(list(read_jsonl(saved).records), saved.stem)
    assert saved.read_bytes().startswith(original + damaged)
    assert [(ep.prompt_text, ep.response_text) for ep, _ in episodes] == [
        ("old question", "old answer"),
        ("new question", "new answer"),
    ]
    assert ref == episodes[-1][0].episode_id


def test_note_work_file_and_reference_enrichment_share_metadata(
    tmp_path, monkeypatch, sample_detail, capsys
):
    from dataclasses import replace

    from ssgrep.search import detail
    from ssgrep.services import knowledge
    from ssgrep.sessions.work import read_work, subject_digest

    metadata = tmp_path / "work.json"
    metadata.write_text(
        json.dumps(
            {
                "work_id": "task-1",
                "provider": "provider",
                "model": "model",
                "elapsed_seconds": 60,
                "usage": {"input_tokens": 100},
            }
        )
    )
    monkeypatch.setattr(knowledge, "_index_saved_note", lambda: None)
    monkeypatch.setattr(note_command, "is_json_mode", lambda: True)
    saved = command().handle(title="question", body="answer", work=str(metadata))
    assert saved is not None and saved["indexed"]
    ref = saved["ref"]
    assert isinstance(ref, str)
    indexed = replace(
        sample_detail,
        episode_id=ref,
        runtime="ssgrep",
        prompt_text="question",
        response_text="answer",
    )
    first = read_work(ref, subject_digest(indexed))
    assert first is not None and first["work"]["usage"] == {"input_tokens": 100}
    monkeypatch.setattr(detail, "show", lambda ref: indexed)
    metadata.write_text(json.dumps({"work_id": "task-1", "elapsed_seconds": 90}))
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)
    command().handle(ref=ref, work=str(metadata))
    updated = read_work(ref, subject_digest(indexed))
    assert updated is not None and updated["work"]["elapsed_seconds"] == 90
    assert "revision 2" in capsys.readouterr().out


@pytest.mark.parametrize(
    "payload",
    ["[]", "{", "x" * 65537, '{"work_id":"x","elapsed_seconds":-1}'],
    ids=["nonobject", "invalid-json", "oversized", "negative-duration"],
)
def test_invalid_metadata_file_never_persists_a_note(tmp_path, monkeypatch, payload):
    metadata = tmp_path / "work.json"
    metadata.write_text(payload)
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)
    with pytest.raises(SystemExit) as caught:
        command().handle(title="question", body="answer", work=str(metadata))
    assert caught.value.code == exit_codes.USAGE_ERROR
    assert not (tmp_path / "data" / "notes").exists()


def test_metadata_file_cannot_be_a_fifo_or_symlink(tmp_path):
    import os

    from ssgrep.services.knowledge import load_work_file

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        load_work_file(str(fifo))
    original = tmp_path / "original.json"
    original.write_text('{"work_id":"source"}')
    link = tmp_path / "link.json"
    link.symlink_to(original)
    with pytest.raises(OSError):
        load_work_file(str(link))
    assert load_work_file(str(original)) == {"work_id": "source"}


def test_metadata_enrichment_handles_missing_index_and_unknown_ref(
    tmp_path, monkeypatch, sample_detail
):
    from ssgrep.search import detail
    from ssgrep.utilities.types import IndexNotFoundError

    metadata = tmp_path / "work.json"
    metadata.write_text('{"work_id":"existing"}')
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)

    def missing(ref):
        raise IndexNotFoundError("no index")

    monkeypatch.setattr(detail, "show", missing)
    with pytest.raises(SystemExit) as absent:
        command().handle(ref="session:ep:0", work=str(metadata))
    assert absent.value.code == exit_codes.MISSING_INDEX
    monkeypatch.setattr(detail, "show", lambda ref: None)
    with pytest.raises(SystemExit) as unknown:
        command().handle(ref="session:ep:0", work=str(metadata))
    assert unknown.value.code == exit_codes.NO_MATCHING_DATA
    assert not (tmp_path / "data" / "work-metadata").exists()


def test_metadata_failure_after_note_write_reports_the_saved_reference(
    tmp_path, monkeypatch, capsys
):
    import os

    root = tmp_path / "data"
    root.mkdir()
    (root / "work-metadata").mkdir()
    (root / "work-metadata").chmod(0o777)
    metadata = tmp_path / "work.json"
    metadata.write_text('{"work_id":"blocked"}')
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)
    with pytest.raises(SystemExit) as caught:
        command().handle(title="already saved", body="keep this", work=str(metadata))
    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    shard = next((root / "notes").glob("*.jsonl"))
    assert "keep this" in shard.read_text()
    assert "Saved ref:" in capsys.readouterr().err
    os.chmod(root / "work-metadata", 0o700)


def test_enrichment_cannot_silently_change_the_project(tmp_path, monkeypatch):
    metadata = tmp_path / "work.json"
    metadata.write_text('{"work_id":"x"}')
    monkeypatch.setattr(note_command, "is_json_mode", lambda: False)
    with pytest.raises(SystemExit) as caught:
        command().handle(ref="old:ep:0", project_dir=str(tmp_path), work=str(metadata))
    assert caught.value.code == exit_codes.USAGE_ERROR
    assert not (tmp_path / "data").exists()


def test_note_rejects_oversized_and_hardlinked_shards_without_losing_prior_notes(tmp_path):
    import os

    from ssgrep.sessions import notes

    with pytest.raises(ValueError, match="size limit"):
        notes.write_note(tmp_path, "too long", "x" * 500_001)
    assert not (tmp_path / "data").exists()
    shard = notes.write_note(tmp_path, "first", "preserved")
    before = shard.read_bytes()
    alias = tmp_path / "alias"
    os.link(shard, alias)
    with pytest.raises(ValueError, match="owned regular"):
        notes.write_note(tmp_path, "rejected", "not written")
    assert shard.read_bytes() == alias.read_bytes() == before
    alias.unlink()
    notes.write_note(tmp_path, "second", "accepted")
    assert shard.read_bytes().startswith(before)
    assert "accepted" in shard.read_text()
