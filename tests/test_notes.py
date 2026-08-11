"""Tests for ssgrep note (notes.py): authored content in the index's own root.

The design's guard-inversion property is the centerpiece: ssgrep's first
intentional write path to indexed content must write ONLY under
`<index_dir>/notes/` -- never under the transcript root the session guards
protect. test_note_never_writes_under_the_transcript_root pins it with a
positive assertion on the real write target plus a full-tree scan of the
fake corpus root.
"""

from __future__ import annotations

import pytest

from ssgrep import api, indexer, notes, staleness
from ssgrep import search as search_module


@pytest.fixture
def note_project(tmp_path, monkeypatch):
    """An isolated project + empty fake corpus root."""
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return project, home


def test_note_field_scenario_end_to_end(note_project):
    """Write a note phrased as a question, index, search it -- rank 1.

    The field deployment measured exactly this shape retrieving rank-1 on
    every probe; this is that scenario, native."""
    project, _home = note_project
    notes.write_note(
        project,
        "how do we configure the frobnicator retry backoff",
        "Set frobnicator.retry_backoff_ms in settings; exponential with jitter.",
    )
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count == 1, "the note shard must be discovered as a session"
    assert stats.chunk_count >= 1

    response = api.search(project, "frobnicator retry backoff")
    assert response.results, "the note must be searchable"
    assert response.results[0].ref.startswith(
        "notes-"
    ), f"the note episode must rank first; got {response.results[0].ref}"


def test_note_never_writes_under_the_transcript_root(note_project):
    """Guard inversion: the write target is index-side, and the corpus root
    stays byte-empty. Positive assertion first (the note landed where
    designed), then the negative sweep (nothing appeared under ~/.claude)."""
    project, home = note_project
    shard = notes.write_note(project, "a title question", "a body answer")

    expected_root = project / ".ssgrep" / "notes"
    assert shard.is_relative_to(
        expected_root
    ), f"note written to {shard}, outside the designed root {expected_root}"
    assert shard.exists()

    corpus_root = home / ".claude" / "projects"
    debris = [p for p in corpus_root.rglob("*") if p.is_file()]
    assert debris == [], f"ssgrep note must NEVER write under the transcript root; found {debris}"


def test_note_staleness_lifecycle(note_project):
    """Notes ride the normal staleness machinery: fresh after index, APPENDED
    (never VANISHED) after a second note lands in the same shard, fresh
    again after reindex."""
    project, _home = note_project
    notes.write_note(project, "first question", "first answer")
    indexer.index(project, quiet=True, index_dir=project / ".ssgrep")

    report = search_module.staleness_summary(project)
    assert not staleness.is_index_stale(report), "freshly indexed note must not be stale"

    notes.write_note(project, "second question", "second answer")
    report = search_module.staleness_summary(project)
    assert staleness.is_index_stale(report), "a new note must read as stale (appended)"
    vanished = [f for f in report.stale_files if f.status == staleness.FileStatus.VANISHED]
    assert vanished == [], "an appended note shard must never read as VANISHED"

    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.episode_count == 2
    report = search_module.staleness_summary(project)
    assert not staleness.is_index_stale(report)


def test_note_validation(note_project):
    project, _home = note_project
    with pytest.raises(ValueError):
        notes.write_note(project, "", "body")
    with pytest.raises(ValueError):
        notes.write_note(project, "title", "   ")


def test_no_notes_dir_is_zero_behavior_change(note_project):
    project, _home = note_project
    assert notes.discover_notes(project / ".ssgrep") == []
    stats = indexer.index(project, quiet=True, index_dir=project / ".ssgrep")
    assert stats.session_count == 0


def test_concurrent_note_writes_never_corrupt_episodes(note_project):
    """The blind-review blocker's repro, pinned: two concurrent write_note
    calls forced to overlap must yield two INTACT episodes (each title with
    its own answer) -- the unlocked two-write version interleaved at the
    line level and segmentation (record-order-based) lost one answer while
    contaminating the other."""
    import threading

    from ssgrep import episodes as ep_mod
    from ssgrep import records as rec_mod

    project, _home = note_project
    barrier = threading.Barrier(2)
    errors = []

    def write(title, body):
        try:
            barrier.wait(timeout=5)
            for _ in range(20):  # amplify interleave pressure
                notes.write_note(project, title, body)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t1 = threading.Thread(target=write, args=("QUESTION_A", "ANSWER_A"))
    t2 = threading.Thread(target=write, args=("QUESTION_B", "ANSWER_B"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []

    shards = notes.discover_notes(project / ".ssgrep")
    assert len(shards) == 1
    recs = list(rec_mod.read_records(shards[0].path))
    eps = ep_mod.segment_episodes(recs, shards[0].session_id)
    assert len(eps) == 40
    for ep in eps:
        if "QUESTION_A" in ep.prompt_text:
            assert (
                "ANSWER_A" in ep.response_text and "ANSWER_B" not in ep.response_text
            ), f"episode corrupted: prompt={ep.prompt_text!r} response={ep.response_text!r}"
        else:
            assert "QUESTION_B" in ep.prompt_text
            assert (
                "ANSWER_B" in ep.response_text and "ANSWER_A" not in ep.response_text
            ), f"episode corrupted: prompt={ep.prompt_text!r} response={ep.response_text!r}"


def test_note_command_reports_cleanly_when_reindex_fails(note_project, monkeypatch, capsys):
    """Blind-review minor: api.index failure after a successful write must
    produce the IndexCommand-style clean report -- note-saved stated, no
    raw traceback, SystemExit with INTERNAL_FAILURE -- never a bare crash."""
    from unittest.mock import MagicMock

    from ssgrep import api
    from ssgrep.cli import exit_codes
    from ssgrep.cli.commands.note import NoteCommand
    from ssgrep.types import IndexNotFoundError

    project, _home = note_project

    def boom(*a, **k):
        raise IndexNotFoundError("no transcripts", condition="missing_index")

    monkeypatch.setattr(api, "index", boom)
    with pytest.raises(SystemExit) as excinfo:
        NoteCommand(MagicMock()).handle(
            title="t question", body="b answer", project_dir=str(project)
        )
    assert excinfo.value.code == exit_codes.INTERNAL_FAILURE
    err = capsys.readouterr().err
    assert "Note written to" in err, "the user must learn the note is SAFE"
    assert "reindexing failed" in err
    # and the note really is on disk
    assert notes.discover_notes(project / ".ssgrep"), "note must survive the index failure"
