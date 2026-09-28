"""Workers must not read an archive while sibling workers reconcile its tables."""

import asyncio
import gc
import sqlite3
from contextlib import contextmanager
from unittest.mock import Mock, call

import pytest

from ssgrep.pipeline import app as app_module, archive as archive_module, components
from ssgrep.pipeline.archive import capture_archives
from ssgrep.pipeline.sources import SourceDescriptor, read_registry
from ssgrep.services import api
from ssgrep.sessions.adapters import opencode as opencode_adapter
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore
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


def _seed_opencode_session(connection: sqlite3.Connection, session_id: str, word: str) -> None:
    _insert_opencode_session(connection, session_id)
    user, assistant = f"{session_id}-user", f"{session_id}-assistant"
    _insert_opencode_message(
        connection,
        user,
        session_id,
        {"role": "user", "time": {"created": 1_700_000_002_000}},
        created=1_700_000_002_000,
    )
    _insert_opencode_message(
        connection,
        assistant,
        session_id,
        {"role": "assistant", "parentID": user, "time": {"created": 1_700_000_005_000}},
        created=1_700_000_005_000,
    )
    _insert_opencode_part(
        connection,
        f"{session_id}-p1",
        user,
        session_id,
        {"type": "text", "text": f"{word} prompt"},
        created=1_700_000_010_001,
    )
    _insert_opencode_part(
        connection,
        f"{session_id}-p2",
        assistant,
        session_id,
        {"type": "text", "text": f"{word} response"},
        created=1_700_000_010_002,
    )


def test_transient_opencode_read_error_does_not_reconcile_rows_away(
    tmp_path, monkeypatch, fake_models
):
    """Issue #13: a locked ``opencode.db`` during a memo-invalidating run must
    not remove any indexed row, for a tombstoned session (row deleted from the
    db, so the index holds its only copy) or a live one.

    Before the fix ``read()`` returned an empty result on the sqlite error,
    ``present()`` said "present" (so no archive snapshot was captured), and
    ``process_source`` reconciled the session's episodes and chunks to nothing.
    """
    db_path = tmp_path / "opencode.db"
    connection = _opencode_database(db_path)
    _seed_opencode_session(connection, "gone", "tombstoned")
    _seed_opencode_session(connection, "live", "surviving")
    connection.commit()
    connection.close()
    monkeypatch.setenv("SSGREP_OPENCODE_DB", str(db_path))
    original_drive = app_module._drive_update

    async def bounded_drive(*args, **kwargs):
        # A raising component must fail the run, not retry forever.
        await asyncio.wait_for(original_drive(*args, **kwargs), timeout=30)

    monkeypatch.setattr(app_module, "_drive_update", bounded_drive)

    api.index()
    repo = LanceStore()
    tables = {
        SESSIONS_TABLE: "session_id",
        EPISODES_TABLE: "episode_id",
        CHUNKS_TABLE: "chunk_id",
    }

    def rows(session_id: str) -> dict[str, list[dict]]:
        return {
            table: sorted(
                repo.rows(table, where=f"session_id = '{session_id}'", limit=repo.count(table)),
                key=lambda row: row[key],
            )
            for table, key in tables.items()
        }

    before = {name: rows(f"opencode:{name}") for name in ("gone", "live")}
    for name, per_table in before.items():
        assert all(per_table.values()), f"{name} must own rows in every data table"

    connection = sqlite3.connect(db_path)
    connection.execute("DELETE FROM session WHERE id = 'gone'")
    connection.commit()
    connection.close()

    fault_hits: list[str] = []

    @contextmanager
    def locked_snapshot(_path):
        fault_hits.append("locked")
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover

    real_snapshot = opencode_adapter._snapshot
    monkeypatch.setattr(opencode_adapter, "_snapshot", locked_snapshot)
    # ``full_reprocess`` invalidates every memo, exactly like a pipeline-code change.
    # Keep only the message: holding the exception would pin its traceback
    # (and the run's CocoIndex environment), so the next run could not reopen
    # the same journal in this process.
    failure: str | None = None
    try:
        api.index(full_reprocess=True)
    except RuntimeError as error:
        failure = str(error)
    # Dropping ``error`` is not enough: the failed run's traceback frames sit in
    # a reference cycle that keeps its CocoIndex environment open until the
    # cyclic GC happens to run, and on CPython 3.11 that timing varies -- the
    # follow-up run below then fails with "environment already open". Collect
    # explicitly so the release is deterministic.
    gc.collect()
    # The property: no indexed row of either session is removed.
    assert {name: rows(f"opencode:{name}") for name in ("gone", "live")} == before
    # Positive: the fault was actually hit while reading (present + read + discover),
    # and the run reported it as a failure to retry rather than succeeding silently.
    assert len(fault_hits) >= 2
    assert failure is not None
    assert "component errors" in failure

    monkeypatch.setattr(opencode_adapter, "_snapshot", real_snapshot)
    api.index(full_reprocess=True)
    after = {name: rows(f"opencode:{name}") for name in ("gone", "live")}
    assert after["live"] == before["live"]
    for table in tables:
        for old, new in zip(before["gone"][table], after["gone"][table], strict=True):
            tombstone = {"source_status", "absent_since"}
            assert {k: v for k, v in new.items() if k not in tombstone} == {
                k: v for k, v in old.items() if k not in tombstone
            }
    assert {row["source_status"] for row in after["gone"][CHUNKS_TABLE]} == {"absent"}
