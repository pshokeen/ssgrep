"""Two ways a generation swap could destroy the buyer's index without saying so.

Both are failures of the same kind: an operation that REPLACES the live
generation, deciding what to replace it with from something that is not the
live generation's actual contents.

1. `.ssgrep/.manifest` becomes unreadable. It is a ~100-byte JSON file next to
   a multi-hundred-megabyte index, so a backup that skips dotfiles, a
   zero-length restore, a sync exclusion, or a user clearing what looks like a
   stale state file is enough. GenerationalStore then reported generation 0 --
   fail-open -- and stage_generation() unlinked the live index.db.N before a
   single transcript had been read, with rebuild_guard blinded by the same
   read.

2. `shutil.copy2()` of a WAL-mode database. `ssgrep revectorize` and
   `ssgrep prune` each build the next generation from a copy of the live one
   and neither goes through rebuild_guard. copy2 takes index.db and not
   index.db-wal, so every commit since the last checkpoint was silently
   absent from the copy that then became live, with the original unlinked
   behind it.

Both were silent, both exited 0, and both destroyed content that for a
tombstoned session exists nowhere else on the machine.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssgrep import embed, indexer, revectorize, store

DIM = 256


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return home


def _install_fake_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embed,
        "encode",
        lambda texts: np.full((len(texts), DIM), 1.0 / (DIM**0.5), dtype=np.float32),
    )
    monkeypatch.setattr(embed, "get_model_info", lambda: (embed.MODEL_ID, DIM))


def _write_transcript(home: Path, project_dir: Path, session_id: str) -> Path:
    path = home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    records: list[dict[str, Any]] = [
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": f"how do I fix the {session_id} problem"},
            "uuid": f"{session_id}-u1",
            "timestamp": "2026-07-01T10:00:00.000Z",
            "cwd": str(project_dir),
            "sessionId": session_id,
            "gitBranch": "main",
        },
        {
            "parentUuid": f"{session_id}-u1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"resolve the {session_id} problem by rebuilding the widget "
                        "and confirming the counts still line up afterwards",
                    }
                ],
            },
            "uuid": f"{session_id}-a1",
            "timestamp": "2026-07-01T10:01:00.000Z",
            "cwd": str(project_dir),
            "sessionId": session_id,
            "gitBranch": "main",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def _write_corpus(home: Path, project_dir: Path, count: int) -> list[Path]:
    return [
        _write_transcript(home, project_dir, f"aaaaaaaa-0000-4000-8000-00000000000{i}")
        for i in range(count)
    ]


def _counts(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        return (
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1. An unreadable manifest must never cost the buyer their index.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("damage", ["truncated", "deleted"])
def test_an_unreadable_manifest_never_unlinks_the_live_generation(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """A rebuild against a store whose manifest is gone must not delete the
    index, and must not report success having done so.

    Both damage shapes are load-bearing. A DELETED manifest is at least as
    plausible as a corrupt one -- `_load_manifest` returned (0, 0) for a
    missing file too -- and is what a backup or sync that skips dotfiles
    inside .ssgrep produces.

    The corpus is made unreproducible (the project moved) because that is
    what turns the unlink into permanent loss rather than something the same
    rebuild immediately repairs. The guard that should have caught the empty
    result was defeated by the same fail-open read: it measures the retiring
    generation at get_index_path(), which had become the nonexistent
    index.db, so it saw (0, 0, 0) and had nothing to refuse.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 6)
    indexer.index(project, index_dir=index_dir, quiet=True)
    indexer.index(project, index_dir=index_dir, rebuild=True, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    assert gen_store.current_generation == 1, "precondition: the live generation must be N >= 1"
    live_db = gen_store.get_index_path()
    live_vec = gen_store.get_vector_path()
    before = _counts(live_db)
    assert before[0] == 6 and before[1] > 0, "precondition: a populated live generation"

    manifest = index_dir / ".manifest"
    if damage == "truncated":
        manifest.write_text("")
    else:
        manifest.unlink()

    # The project moved, so this rebuild cannot reproduce what it retires.
    moved = tmp_path / "moved"
    moved.mkdir()
    with pytest.raises(Exception):  # noqa: B017 - any refusal beats silent deletion
        indexer.index(moved, index_dir=index_dir, rebuild=True, quiet=True)

    assert live_db.exists(), "the live index.db.1 must survive an unreadable manifest"
    assert live_vec.exists(), "the live vectors.f32.1 must survive it too"
    assert _counts(live_db) == before, "and must still hold every session and chunk"


def test_residue_above_a_known_live_generation_is_still_cleared(
    tmp_path: Path,
) -> None:
    """Refusing on an unreadable manifest must not disable residue clearing.

    A rebuild that dies partway leaves index.db.N behind with nothing
    pointing at it. Before that residue was unlinked, the next rebuild
    reopened those exact files and appended to them, leaving orphaned vector
    rows that commit_generation() then refused -- permanently, for every
    later rebuild, with the only escape being to delete .ssgrep/ by hand.
    A READABLE manifest is what makes residue provably residue, and that case
    must keep working.
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir()
    (index_dir / "index.db").write_bytes(b"the live index")
    (index_dir / ".manifest").write_text(json.dumps({"generation": 0, "vector_row_count": 0}))
    residue = index_dir / "index.db.1"
    residue.write_bytes(b"abandoned")

    store.GenerationalStore(index_dir).stage_generation(1)

    assert not residue.exists(), "residue above a known live generation must be cleared"
    assert (index_dir / "index.db").read_bytes() == b"the live index"


# ---------------------------------------------------------------------------
# 2. A generation built by copying the live database must include its WAL.
# ---------------------------------------------------------------------------


def _strand_a_session_in_the_wal(live_db: Path) -> sqlite3.Connection:
    """Commit a session that lives only in index.db-wal, and return the reader
    holding it there.

    A WAL checkpoint normally happens when the last connection closes. A
    concurrent reader with an open snapshot blocks that checkpoint, which is
    exactly the state ssgrep's own design produces: hook-driven indexing runs
    asynchronously while searches and the MCP server read (indexer.py's module
    docstring calls this out), and a writer killed after commit leaves the
    same state.

    The caller must close the returned connection.
    """
    reader = sqlite3.connect(str(live_db))
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM sessions").fetchone()

    writer = sqlite3.connect(str(live_db))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute(
        "INSERT INTO sessions (session_id, path, is_main, source_status) "
        "VALUES ('wal-only-session', '/nonexistent/wal-only.jsonl', 1, 'available')"
    )
    writer.commit()
    writer.close()

    assert live_db.with_name(
        live_db.name + "-wal"
    ).exists(), "precondition: the commit must still be sitting in the WAL"
    return reader


def test_revectorize_does_not_drop_sessions_that_live_only_in_the_wal(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ssgrep revectorize` builds the next generation from a copy of the live
    database and then unlinks the original. A copy that omits the WAL
    therefore destroys every commit since the last checkpoint -- reporting
    success and exiting 0, with only a quietly smaller count to show for it.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 4)
    indexer.index(project, index_dir=index_dir, quiet=True)

    live_db = store.GenerationalStore(index_dir).get_index_path()
    reader = _strand_a_session_in_the_wal(live_db)
    try:
        expected_sessions = _counts(live_db)[0]
        revectorize.revectorize(project, index_dir=index_dir, quiet=True)
    finally:
        reader.close()

    new_live = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(new_live))
    try:
        survived = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = 'wal-only-session'"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()

    assert survived == 1, "the WAL-resident session must be carried into the new generation"
    assert total == expected_sessions, "and nothing else may be dropped alongside it"


def test_prune_compaction_does_not_drop_sessions_that_live_only_in_the_wal(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ssgrep prune` reaches the same copy through
    store.cleanup_orphaned_vectors(). It is the worse of the two: prune is
    precisely the command a buyer runs once Claude Code's retention has
    deleted the transcripts behind their tombstoned sessions, so the index is
    by then the last surviving copy of that history.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 4)
    indexer.index(project, index_dir=index_dir, quiet=True)

    live_db = store.GenerationalStore(index_dir).get_index_path()
    reader = _strand_a_session_in_the_wal(live_db)
    try:
        expected_sessions = _counts(live_db)[0]
        conn = sqlite3.connect(str(live_db))
        store.cleanup_orphaned_vectors(conn, store.GenerationalStore(index_dir))
    finally:
        reader.close()

    new_live = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(new_live))
    try:
        survived = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = 'wal-only-session'"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()

    assert survived == 1, "the WAL-resident session must survive compaction"
    assert total == expected_sessions, "and nothing else may be dropped alongside it"


def test_revectorize_refuses_to_commit_a_generation_that_lost_rows(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backstop, independent of how the copy is made.

    Re-vectorizing re-embeds text that is already stored, so it must not add
    or drop a row. rebuild_guard gates indexer.index()'s rebuild but has
    never covered this path, which replaces the live generation just as
    completely -- so anything that makes the staged copy diverge has to be
    caught here, at the last moment before the original is unlinked.
    """
    _install_fake_encode(monkeypatch)
    project = tmp_path / "project"
    index_dir = tmp_path / "idx"
    _write_corpus(fake_home, project, 4)
    indexer.index(project, index_dir=index_dir, quiet=True)

    live_db = store.GenerationalStore(index_dir).get_index_path()
    before = _counts(live_db)

    real_copy = store.copy_database

    def lossy_copy(source: Path, destination: Path) -> None:
        real_copy(source, destination)
        conn = sqlite3.connect(str(destination))
        try:
            conn.execute(
                "DELETE FROM sessions WHERE session_id IN "
                "(SELECT session_id FROM sessions LIMIT 2)"
            )
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(store, "copy_database", lossy_copy)

    with pytest.raises(Exception, match="does not match the existing one"):
        revectorize.revectorize(project, index_dir=index_dir, quiet=True)

    assert (
        _counts(store.GenerationalStore(index_dir).get_index_path()) == before
    ), "a refused re-vectorization must leave the live generation intact and current"
