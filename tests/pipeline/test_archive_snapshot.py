"""Workers must not read an archive while sibling workers reconcile its tables."""

import asyncio
import sqlite3
from unittest.mock import Mock, call

import pytest

from ssgrep.pipeline import archive as archive_module, components
from ssgrep.pipeline.archive import capture_archives
from ssgrep.pipeline.sources import SourceDescriptor, read_registry
from ssgrep.services import api
from ssgrep.store import LanceStore
from tests.pipeline.test_app import fake_models as fake_models, write_claude_session
from tests.sessions.adapters.test_opencode import (
    _database as _opencode_database,
    _insert_message as _insert_opencode_message,
    _insert_part as _insert_opencode_part,
    _insert_session as _insert_opencode_session,
)


def test_capture_archives_skips_presence_check_for_freshly_discovered_keys(monkeypatch):
    """``fresh_keys`` sources were just found by this run's own discovery, so
    they're present by construction: ``capture_archives`` must skip the
    per-source presence check for them rather than pay it for every live
    source on every reconcile -- costliest for OpenCode, whose ``present()``
    opens a real sqlite connection per session (see the docstring).
    """
    checked: list[str] = []

    def fake_source_present(source):
        checked.append(source.key)
        return False

    monkeypatch.setattr(archive_module.transcript_adapters, "source_present", fake_source_present)
    monkeypatch.setattr(archive_module, "retained_rows", lambda source, repo=None: [[], [], []])

    def descriptor(key: str) -> SourceDescriptor:
        return SourceDescriptor(
            adapter="native", key=key, path=f"/tmp/{key}.jsonl", size=0, mtime=0.0, digest=""
        )

    entries = {"fresh-1": descriptor("fresh-1"), "stale-1": descriptor("stale-1")}
    result = capture_archives(entries, fresh_keys={"fresh-1"})

    assert checked == ["stale-1"]
    assert result == {"stale-1": [[], [], []]}


def test_worker_redeclares_frozen_rows_after_database_changes(monkeypatch):
    path = write_claude_session("frozen")
    api.index()
    repo = LanceStore()
    descriptor = read_registry(repo)[str(path)]
    path.unlink()
    snapshot = capture_archives({descriptor.key: descriptor})
    assert all(snapshot[descriptor.key]), "the fixture must own rows in every data table"
    for table in ("sessions", "episodes", "chunks"):
        repo.delete(table, "session_id = 'frozen'")
        assert repo.count(table) == 0
    monkeypatch.setattr(components.coco, "use_context", lambda key: snapshot)
    chunk_target, episode_target, session_target = Mock(), Mock(), Mock()
    asyncio.run(components.process_source(descriptor, chunk_target, episode_target, session_target))
    for target, expected in zip(
        (session_target, episode_target, chunk_target), snapshot[descriptor.key], strict=True
    ):
        assert target.declare_row.call_args_list == [call(row=row) for row in expected]
    repo.close()


def test_worker_redeclares_frozen_rows_after_opencode_row_deletion(tmp_path, monkeypatch):
    """Issue #8 regression: OpenCode's database backs many sessions, so a row
    deleted from an otherwise-intact ``opencode.db`` must be recognized as
    missing too (not just a deleted database), or the next pipeline-code
    change reconciles that session's episodes/chunks away instead of
    preserving them.
    """
    db_path = tmp_path / "opencode.db"
    connection = _opencode_database(db_path)
    _insert_opencode_session(connection, "keep")
    _insert_opencode_message(
        connection,
        "message-user",
        "keep",
        {"role": "user", "time": {"created": 1_700_000_002_000}},
        created=1_700_000_002_000,
    )
    _insert_opencode_message(
        connection,
        "message-assistant",
        "keep",
        {"role": "assistant", "parentID": "message-user", "time": {"created": 1_700_000_005_000}},
        created=1_700_000_005_000,
    )
    _insert_opencode_part(
        connection,
        "part-01",
        "message-user",
        "keep",
        {"type": "text", "text": "user prompt"},
        created=1_700_000_010_001,
    )
    _insert_opencode_part(
        connection,
        "part-02",
        "message-assistant",
        "keep",
        {"type": "text", "text": "assistant response"},
        created=1_700_000_010_002,
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(db_path))

    api.index()
    repo = LanceStore()
    descriptor = read_registry(repo)["opencode:keep"]

    connection = sqlite3.connect(db_path)
    connection.execute("DELETE FROM session WHERE id = 'keep'")
    connection.commit()
    connection.close()

    snapshot = capture_archives({descriptor.key: descriptor})
    assert all(snapshot[descriptor.key]), "the fixture must own rows in every data table"
    for table in ("sessions", "episodes", "chunks"):
        repo.delete(table, "session_id = 'opencode:keep'")
        assert repo.count(table) == 0
    monkeypatch.setattr(components.coco, "use_context", lambda key: snapshot)
    chunk_target, episode_target, session_target = Mock(), Mock(), Mock()
    asyncio.run(components.process_source(descriptor, chunk_target, episode_target, session_target))
    for target, expected in zip(
        (session_target, episode_target, chunk_target), snapshot[descriptor.key], strict=True
    ):
        assert target.declare_row.call_args_list == [call(row=row) for row in expected]
    repo.close()


def test_source_disappearing_after_snapshot_fails_closed(monkeypatch):
    path = write_claude_session("late")
    api.index()
    repo = LanceStore()
    descriptor = read_registry(repo)[str(path)]
    snapshot = capture_archives({descriptor.key: descriptor})
    assert snapshot == {}
    path.unlink()
    monkeypatch.setattr(components.coco, "use_context", lambda key: snapshot)
    targets = [Mock(), Mock(), Mock()]
    with pytest.raises(ValueError, match="disappeared after archive snapshot"):
        asyncio.run(components.process_source(descriptor, *targets))
    assert all(not target.declare_row.called for target in targets)
    repo.close()
