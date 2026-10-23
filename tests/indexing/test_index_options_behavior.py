"""Public indexing options against the real offline CocoIndex/Lance pipeline.

These belong to the heavyweight ``slow`` lane, not the model-download lane.
Only the embedding boundary and an explicit source-read fault are substituted.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from ssgrep.cli import exit_codes
from ssgrep.cli.commands import index_command
from ssgrep.pipeline import app as app_mod
from ssgrep.pipeline.update import ComponentUpdateError
from ssgrep.services import api
from ssgrep.sessions import adapters
from ssgrep.store import CHUNKS_TABLE, EPISODES_TABLE, SESSIONS_TABLE, LanceStore
from ssgrep.utilities.types import RebuildWouldShrinkError
from tests.pipeline.test_app import fake_models as fake_models

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def offline_pipeline(fake_models, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse integration model fixtures without an inherited journal or CLI mode."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("SSGREP_COCOINDEX_DB", str(tmp_path / "journal" / "state.db"))
    monkeypatch.setattr(index_command, "is_json_mode", lambda: False)
    original_drive = app_mod._drive_update

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    async def bounded_drive(*args, **kwargs):
        # A regression must fail instead of indefinitely retrying a broken source.
        await asyncio.wait_for(original_drive(*args, **kwargs), timeout=30)

    monkeypatch.setattr(app_mod, "_drive_update", bounded_drive)


def write_session(session_id: str, cwd: Path, *, subagent: bool = False) -> Path:
    """Write a native prompt/response pair in Claude's real discovery layout."""
    project = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "fixture-project"
    relative = (
        Path("parent/subagents/agent-child.jsonl") if subagent else Path(f"{session_id}.jsonl")
    )
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    records = (
        {
            "type": "user",
            "cwd": str(cwd),
            "message": {"content": f"Question for {session_id}"},
            "timestamp": "2025-01-01T00:00:00Z",
            "sessionId": session_id,
        },
        {
            "type": "assistant",
            "message": {"content": f"Answer for {session_id}"},
            "timestamp": "2025-01-01T00:00:01Z",
            "sessionId": session_id,
        },
    )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def invoke(entry: str, **options):
    if entry == "api":
        return api.index(quiet=True, **options)
    # The framework-free CLI handler is the same public boundary used by the
    # existing CLI suite; discovery, API, lock, engine and storage remain real.
    return object.__new__(index_command.IndexCommand).handle(quiet=True, **options)


def indexed_paths() -> set[str]:
    return {row["path"] for row in LanceStore().rows(SESSIONS_TABLE, limit=100)}


def stored_rows() -> dict[str, list[dict]]:
    repository = LanceStore()
    return {
        table: sorted(repository.rows(table, limit=100), key=lambda row: row[key])
        for table, key in (
            (SESSIONS_TABLE, "session_id"),
            (EPISODES_TABLE, "episode_id"),
            (CHUNKS_TABLE, "chunk_id"),
        )
    }


@pytest.mark.parametrize("entry", ["api", "cli"])
@pytest.mark.parametrize("no_subagents", [False, True])
def test_scope_includes_descendants_not_sibling_prefixes_or_excluded_subagents(
    entry: str, no_subagents: bool, tmp_path: Path
) -> None:
    scope = tmp_path / "work" / "app"
    exact = write_session("exact", scope)
    descendant = write_session("nested", scope / "src")
    write_session("sibling", scope.parent / "application")
    write_session("elsewhere", tmp_path / "elsewhere")
    child = write_session("child", scope / "src", subagent=True)

    invoke(entry, scope=str(scope), no_subagents=no_subagents)

    expected = {str(exact), str(descendant)}
    if not no_subagents:
        expected.add(str(child))
    assert indexed_paths() == expected
    repository = LanceStore()
    assert repository.count(EPISODES_TABLE) == len(expected)
    session_ids = {row["session_id"] for row in repository.rows(SESSIONS_TABLE, limit=100)}
    assert {row["session_id"] for row in repository.rows(CHUNKS_TABLE, limit=100)} == session_ids
    assert all(row["source_status"] == "available" for row in repository.rows(SESSIONS_TABLE))
    assert repository.get_meta("index_state") == "ready"


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_rebuild_removes_archived_rows_instead_of_incrementally_retaining_them(
    entry: str, tmp_path: Path
) -> None:
    keep = write_session("keep", tmp_path)
    removed = write_session("removed", tmp_path)
    api.index(quiet=True)
    removed.unlink()
    api.index(quiet=True)
    assert indexed_paths() == {str(keep), str(removed)}
    assert LanceStore().count(SESSIONS_TABLE, "source_status = 'absent'") == 1

    # One of two sessions remains: above the shrink refusal threshold, so this
    # distinguishes rebuild from a normal update independently of allow_shrink.
    invoke(entry, rebuild=True)

    assert indexed_paths() == {str(keep)}
    rows = stored_rows()
    assert len(rows[SESSIONS_TABLE]) == len(rows[EPISODES_TABLE]) == 1
    assert rows[CHUNKS_TABLE]
    assert all(row["session_id"] == "keep" for table in rows.values() for row in table)
    assert LanceStore().count(SESSIONS_TABLE, "source_status = 'absent'") == 0


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_shrinking_rebuild_refuses_without_destroying_rows_then_accepts_override(
    entry: str, tmp_path: Path
) -> None:
    keep = write_session("keep", tmp_path)
    removed = [write_session(f"removed-{number}", tmp_path) for number in range(2)]
    api.index(quiet=True)
    before = stored_rows()
    for path in removed:
        path.unlink()

    if entry == "api":
        with pytest.raises(RebuildWouldShrinkError) as refusal:
            invoke(entry, rebuild=True)
        assert refusal.value.old_counts == tuple(len(rows) for rows in before.values())
        assert refusal.value.new_counts[0] == 1
    else:
        with pytest.raises(SystemExit) as refusal:
            invoke(entry, rebuild=True)
        assert refusal.value.code == exit_codes.USAGE_ERROR
    assert stored_rows() == before
    assert LanceStore().get_meta("index_state") == "ready"

    invoke(entry, rebuild=True, allow_shrink=True)

    assert indexed_paths() == {str(keep)}
    rows = stored_rows()
    assert len(rows[SESSIONS_TABLE]) == len(rows[EPISODES_TABLE]) == 1
    assert rows[CHUNKS_TABLE]
    assert all(row["session_id"] == "keep" for table in rows.values() for row in table)


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_forced_component_read_failure_reaches_public_error_boundary(
    entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = write_session("unreadable", tmp_path)
    api.index(quiet=True)
    before = stored_rows()
    original_read = adapters.read_source
    faults: list[str] = []

    def unreadable(source):
        if source.session.path == source_path:
            faults.append(source.key)
            raise PermissionError("fixture source read denied")
        return original_read(source)

    monkeypatch.setattr(adapters, "read_source", unreadable)
    # Unchanged bytes normally memo-hit. A dropped full_reprocess option would
    # never exercise the read fault and would incorrectly return success.
    if entry == "api":
        with pytest.raises(ComponentUpdateError):
            invoke(entry, full_reprocess=True)
    else:
        with pytest.raises(SystemExit) as failure:
            invoke(entry, full_reprocess=True)
        assert failure.value.code == exit_codes.INTERNAL_FAILURE
        assert isinstance(failure.value.__cause__, ComponentUpdateError)
    assert faults, "the failing adapter read was not exercised"
    assert source_path.is_file()
    assert stored_rows() == before
    assert LanceStore().get_meta("index_state") != "ready"
