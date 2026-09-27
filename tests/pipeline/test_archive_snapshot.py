"""Workers must not read an archive while sibling workers reconcile its tables."""

import asyncio
from unittest.mock import Mock, call

import pytest

from ssgrep.pipeline import components
from ssgrep.pipeline.archive import capture_archives
from ssgrep.pipeline.sources import read_registry
from ssgrep.services import api
from ssgrep.store import LanceStore
from tests.pipeline.test_app import fake_models as fake_models, write_claude_session


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
