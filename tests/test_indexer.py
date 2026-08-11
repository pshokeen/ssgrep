"""Tests for the indexer pipeline hub.

discovery.discover_sessions() reads real transcripts from Path.home() /
".claude" / "projects" with no way to redirect that root, so every isolated
test here monkeypatches Path.home() to a tmp_path-rooted fake home and places
synthetic transcripts under it, matched into scope via the `cwd` field on
each record (per D11, scope matching is cwd-based, not directory-name-based).
`.ssgrep/` output is always redirected to a tmp_path via indexer.index()'s
index_dir override, so nothing is ever written under the real repo or home.

embed.encode() is monkeypatched to a fast deterministic stand-in everywhere
in this file -- these tests exercise pipeline wiring (discovery through
store/vectors), not embedding quality, and the real model2vec model is not
guaranteed to be available or fast in every environment.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssgrep import embed, indexer, store, vectors
from tests.conftest import build_file_cursor, require_scoped_real_corpus

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


def _install_fake_encode(
    monkeypatch: pytest.MonkeyPatch, counter: dict[str, int] | None = None
) -> dict[str, int]:
    if counter is None:
        counter = {"calls": 0}

    def fake_encode(texts: list[str]) -> np.ndarray:
        counter["calls"] += 1
        return np.full((len(texts), 256), 1.0 / (256**0.5), dtype=np.float32)

    monkeypatch.setattr(embed, "encode", fake_encode)
    return counter


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _user_record(
    uid: str, text: str, cwd: str, session_id: str, parent_uuid: str | None = None
) -> dict[str, Any]:
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": uid,
        "timestamp": "2026-07-01T10:00:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _assistant_record(
    uid: str, text: str, cwd: str, session_id: str, parent_uuid: str
) -> dict[str, Any]:
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "uuid": uid,
        "timestamp": "2026-07-01T10:01:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _tool_result_user_record(
    uid: str, cwd: str, session_id: str, parent_uuid: str | None = None
) -> dict[str, Any]:
    """A `user`-type record carrying only a tool_result echo -- no "text"
    block, so its extracted prompt text is empty. Real transcripts produce
    these constantly (every tool call's result is echoed back as a `user`
    record); this is the shape that exercises episodes.py's empty-content
    merge quirk when two land back to back with no assistant text between.
    """
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": uid, "content": "ok"}],
        },
        "uuid": uid,
        "timestamp": "2026-07-01T10:00:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _fts_hits(index_dir: Path, query: str) -> list[tuple[str, float]]:
    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        return store.search_fts(conn, query)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests: the rewrite/truncation guard itself
# ---------------------------------------------------------------------------


def test_needs_full_reparse_no_cursor_means_first_time():
    assert indexer._needs_full_reparse(None, 100, "h") is True


def test_needs_full_reparse_shrunk_size_triggers_reparse():
    cursor = build_file_cursor(size=100, first_line_hash="h")
    assert indexer._needs_full_reparse(cursor, 50, "h") is True


def test_needs_full_reparse_hash_mismatch_triggers_reparse_even_if_grown():
    cursor = build_file_cursor(size=100, first_line_hash="h1")
    assert indexer._needs_full_reparse(cursor, 500, "h2") is True


def test_needs_full_reparse_safe_resume_when_both_checks_pass():
    cursor = build_file_cursor(size=100, first_line_hash="h")
    assert indexer._needs_full_reparse(cursor, 150, "h") is False
    assert indexer._needs_full_reparse(cursor, 100, "h") is False


# ---------------------------------------------------------------------------
# Full pipeline wiring
# ---------------------------------------------------------------------------


def test_full_pipeline_wires_discovery_to_vectors(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record(
                "u1", "How do I configure the widget cache TTL?", str(project_dir), session_id
            ),
            _assistant_record(
                "a1",
                "Set CACHE_TTL_SECONDS in the cache settings module.",
                str(project_dir),
                session_id,
                "u1",
            ),
        ],
    )
    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == 1
    assert stats.episode_count == 1
    assert stats.chunk_count == 2  # one prompt chunk, one response chunk
    assert stats.model_id == embed.MODEL_ID
    assert stats.vector_dimension == 256
    assert stats.tombstoned_source_count == 0

    assert len(_fts_hits(index_dir, "widget")) > 0
    valid, msg = vectors.validate_alignment(index_dir / "index.db", index_dir / "vectors.f32")
    assert valid, msg


def test_subagent_meta_sidecar_attributes_episode(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    parent_id = "99999999-2222-4333-8444-555555555555"
    sub_path = (
        fake_home
        / ".claude"
        / "projects"
        / "proj"
        / parent_id
        / "subagents"
        / "agent-deadbeef.jsonl"
    )
    _write_jsonl(
        sub_path,
        [_user_record("su1", "investigate the dependency risk", str(project_dir), parent_id)],
    )
    meta_path = sub_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "agentType": "general-purpose",
                "name": "risk-investigator",
                "model": "claude-x",
                "description": "Investigate dependency risk before upgrade",
            }
        )
    )
    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    conn = sqlite3.connect(str(index_dir / "index.db"))
    row = conn.execute("SELECT is_subagent, agent_name, agent_description FROM episodes").fetchone()
    conn.close()
    assert row == (1, "risk-investigator", "Investigate dependency risk before upgrade")


# ---------------------------------------------------------------------------
# No-change re-index: zero re-embedding, sub-second (mutation-tested)
# ---------------------------------------------------------------------------


def test_no_change_reindex_is_instant_and_reembeds_nothing(fake_home, tmp_path, monkeypatch):
    import time

    project_dir = tmp_path / "project"
    session_id = "22222222-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record(
                "u1", "Investigate the allocator frobnication bug.", str(project_dir), session_id
            ),
            _assistant_record(
                "a1",
                "Found it: empty pools now return early.",
                str(project_dir),
                session_id,
                "u1",
            ),
        ],
    )
    index_dir = tmp_path / "idx"
    counter = _install_fake_encode(monkeypatch)

    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert counter["calls"] == 1
    assert stats1.chunk_count > 0

    counter["calls"] = 0
    start = time.perf_counter()
    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    elapsed = time.perf_counter() - start

    assert counter["calls"] == 0, "unchanged corpus must not call the encoder at all"
    assert elapsed < 1.0
    assert stats2.chunk_count == stats1.chunk_count


def test_no_change_reindex_makes_no_store_writes(fake_home, tmp_path, monkeypatch):
    """The no-op early return's own comment claims 'no reads, no encode(),
    no writes' for an unchanged file (_index_file's
    `if not need_full and start_offset >= disk_size: return`). The read and
    encode halves are covered by test_no_change_reindex_is_instant_and_
    reembeds_nothing above; nothing previously checked the write half --
    store.insert_session() and store.upsert_session_file() sit downstream of
    that same early return and are only reached when it does NOT fire.

    This can't be a state-based assertion: both calls are idempotent, so the
    database ends up byte-for-byte identical whether the early return fired
    correctly or was bypassed and let them redundantly re-run. Only a
    call-count spy can distinguish "correctly skipped" from "ran anyway" --
    which is exactly the distinction a `>=` -> `>` boundary mutation on the
    early return's condition collapses (equality, the only-ever-reached
    exact-no-op case, stops satisfying `>` while still satisfying `>=`).

    A spy is only as good as the guarantee that indexer.py's calls actually
    route through it, so the patch is installed BEFORE the first (fresh-file)
    index() call too, with a positive floor assertion that it really did
    observe calls there. Without that positive signal, `== 0` on the second
    call is indistinguishable from a spy that silently never intercepts
    anything (e.g. indexer.py binding `store.insert_session` some other way
    that bypasses monkeypatch.setattr(store, ...)) -- confirmed empirically:
    reverting just the floor assertions below reproduces a version of this
    test that stays green even when indexer.py's `store` reference is
    swapped out from under the spy entirely.
    """
    project_dir = tmp_path / "project"
    session_id = "22222222-3333-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record(
                "u1", "Investigate the allocator frobnication bug.", str(project_dir), session_id
            ),
            _assistant_record(
                "a1",
                "Found it: empty pools now return early.",
                str(project_dir),
                session_id,
                "u1",
            ),
        ],
    )
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)

    calls = {"insert_session": 0, "upsert_session_file": 0}
    real_insert_session = store.insert_session
    real_upsert_session_file = store.upsert_session_file

    def spy_insert_session(*args: Any, **kwargs: Any) -> None:
        calls["insert_session"] += 1
        return real_insert_session(*args, **kwargs)

    def spy_upsert_session_file(*args: Any, **kwargs: Any) -> None:
        calls["upsert_session_file"] += 1
        return real_upsert_session_file(*args, **kwargs)

    monkeypatch.setattr(store, "insert_session", spy_insert_session)
    monkeypatch.setattr(store, "upsert_session_file", spy_upsert_session_file)

    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats1.chunk_count > 0
    assert calls["insert_session"] > 0, (
        "positive signal: a brand-new file's first index() call MUST insert its "
        "session row -- proves the spy is genuinely wired to indexer.py's real call "
        "sites, so the zero-count assertions below are meaningful rather than vacuous"
    )
    assert calls["upsert_session_file"] > 0

    calls["insert_session"] = 0
    calls["upsert_session_file"] = 0

    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats2.chunk_count == stats1.chunk_count
    assert calls["insert_session"] == 0, (
        "an unchanged file's session row must not be redundantly re-inserted "
        "-- the early return exists specifically to skip this"
    )
    assert calls["upsert_session_file"] == 0, (
        "an unchanged file's cursor must not be redundantly re-written -- "
        "the early return exists specifically to skip this"
    )


# ---------------------------------------------------------------------------
# Incremental append: bounded bytes read (mutation-tested)
# ---------------------------------------------------------------------------


class _CountingFile:
    """Wraps a real binary file object, tallying bytes returned by reads.

    A plain Python class (not a monkeypatched method on the C-implemented
    io.BufferedReader) so `for line in f:` reliably goes through __next__
    below rather than a C-level iteration slot that could bypass an
    instance-attribute override.
    """

    def __init__(self, f: Any, counter: dict[str, int]) -> None:
        self._f = f
        self._counter = counter

    def __enter__(self) -> _CountingFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self._f.close()

    def seek(self, *a: Any, **k: Any) -> int:
        return self._f.seek(*a, **k)  # type: ignore[no-any-return]

    def tell(self) -> int:
        return self._f.tell()  # type: ignore[no-any-return]

    def readline(self, *a: Any, **k: Any) -> bytes:
        data = self._f.readline(*a, **k)
        self._counter["bytes"] += len(data)
        return data  # type: ignore[no-any-return]

    def __iter__(self) -> _CountingFile:
        return self

    def __next__(self) -> bytes:
        data = self._f.readline()
        if not data:
            raise StopIteration
        self._counter["bytes"] += len(data)
        return data  # type: ignore[no-any-return]

    def close(self) -> None:
        self._f.close()


def _write_large_transcript(path: Path, cwd: str, session_id: str, target_bytes: int) -> str:
    """Write a large valid transcript; returns the uuid of its last record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_uuid = None
    with open(path, "w") as f:
        written = 0
        i = 0
        parent: str | None = None
        while written < target_bytes:
            uid = f"u{i}"
            record = _user_record(uid, "filler content " + "x" * 400, cwd, session_id, parent)
            line = json.dumps(record) + "\n"
            f.write(line)
            written += len(line)
            parent = uid
            last_uuid = uid
            i += 1
    assert last_uuid is not None
    return last_uuid


def test_incremental_append_reads_bounded_bytes_not_whole_file(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    session_id = "33333333-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    last_uuid = _write_large_transcript(session_path, str(project_dir), session_id, 34_000_000)
    file_size_before = session_path.stat().st_size
    assert file_size_before >= 33_600_000, "reference scenario is a >=33.6MB transcript"

    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats1.chunk_count > 0

    appended = _user_record(
        "uNEW",
        "brand new appended prompt after the big file " + "x" * 3800,
        str(project_dir),
        session_id,
        last_uuid,
    )
    appended_line = json.dumps(appended) + "\n"
    with open(session_path, "a") as f:
        f.write(appended_line)
    appended_bytes = len(appended_line.encode("utf-8"))
    assert 1000 < appended_bytes < 10_000, "reference scenario appends a few KB"

    counter = {"bytes": 0}
    real_open = open

    def fake_open(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        f = real_open(file, mode, *a, **k)
        if "b" in mode and Path(file) == session_path:
            return _CountingFile(f, counter)
        return f

    monkeypatch.setattr(indexer, "open", fake_open, raising=False)
    _install_fake_encode(monkeypatch)

    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats2.chunk_count > stats1.chunk_count, "the appended prompt must be indexed"
    assert counter["bytes"] > 0, (
        "the counting wrapper never observed any read at all -- either indexer.py's "
        "incremental read no longer goes through indexer.open(), or fake_open's own "
        "path-matching condition silently stopped matching. Without this floor, the "
        "bound check below would pass vacuously (0 < 20_000) without having verified "
        "anything about the size of the read"
    )
    assert counter["bytes"] < 20_000, (
        f"read {counter['bytes']} bytes for a {appended_bytes}-byte append into a "
        f"{file_size_before} byte file -- should be a small bounded read, not the whole file"
    )
    print(
        f"\nMEASURED: {counter['bytes']} bytes read for a {appended_bytes}-byte append "
        f"to a {file_size_before}-byte ({file_size_before / 1_000_000:.1f}MB) transcript"
    )


# ---------------------------------------------------------------------------
# Truncation guard (mutation-tested)
# ---------------------------------------------------------------------------


def test_truncation_triggers_full_reparse(fake_home, tmp_path, monkeypatch):
    """A file that shrinks below its stored cursor size must be fully
    re-parsed, even when its first line is unchanged (so the hash check
    alone would not catch it -- this isolates the size-shrink check).
    """
    project_dir = tmp_path / "project"
    session_id = "44444444-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    shared_first = _user_record("u0", "shared first line content", str(project_dir), session_id)
    long_records = [
        shared_first,
        _assistant_record(
            "a0",
            "beforetruncationmarker " + "long detail " * 100,
            str(project_dir),
            session_id,
            "u0",
        ),
    ]
    _write_jsonl(session_path, long_records)
    original_size = session_path.stat().st_size

    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats1.chunk_count > 0
    assert len(_fts_hits(index_dir, "beforetruncationmarker")) > 0

    # Truncate: identical first line (same hash), but everything after it is
    # gone, so only the size-shrink check -- not the hash check -- can catch
    # this.
    _write_jsonl(session_path, [shared_first])
    new_size = session_path.stat().st_size
    assert new_size < original_size, "fixture must actually shrink to exercise truncation"

    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert (
        len(_fts_hits(index_dir, "beforetruncationmarker")) == 0
    ), "stale pre-truncation content must be purged, not left mixed in"


# ---------------------------------------------------------------------------
# Rewrite-in-place guard (mutation-tested)
# ---------------------------------------------------------------------------


def test_rewrite_triggers_full_reparse(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    session_id = "55555555-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    original = [
        {"type": "custom-title", "custom-title": "original", "sessionId": "orig-sess"},
        _user_record("u1", "beforerewritemarker widget cache detail", str(project_dir), session_id),
    ]
    _write_jsonl(session_path, original)
    original_size = session_path.stat().st_size

    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert len(_fts_hits(index_dir, "beforerewritemarker")) == 1

    # Same-or-larger size, but a *different first line* -- a rewrite the size
    # check alone cannot see.
    rewritten = [
        {
            "type": "custom-title",
            "custom-title": "totally different session",
            "sessionId": "new-sess",
        },
        _user_record(
            "u2",
            "afterrewritemarker replaced content entirely " + ("padding " * 60),
            str(project_dir),
            session_id,
        ),
    ]
    _write_jsonl(session_path, rewritten)
    assert session_path.stat().st_size >= original_size

    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert (
        len(_fts_hits(index_dir, "beforerewritemarker")) == 0
    ), "pre-rewrite content must be purged, not left mixed in with the new session"
    assert len(_fts_hits(index_dir, "afterrewritemarker")) == 1


# ---------------------------------------------------------------------------
# Tombstoning, not cascade delete (mutation-tested -- the most important one)
# ---------------------------------------------------------------------------


def test_vanished_source_tombstoned_not_deleted(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    session_id = "66666666-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record(
                "u1", "vanishguardmarker unique searchable prompt", str(project_dir), session_id
            ),
            _assistant_record(
                "a1", "vanishguardmarker response detail too", str(project_dir), session_id, "u1"
            ),
        ],
    )
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats1.chunk_count > 0
    assert stats1.tombstoned_source_count == 0

    session_path.unlink()

    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats2.tombstoned_source_count == 1
    assert stats2.tombstoned_chunk_count == stats1.chunk_count
    assert stats2.chunk_count == stats1.chunk_count, "vanished source's chunks must be RETAINED"

    hits = _fts_hits(index_dir, "vanishguardmarker")
    assert len(hits) > 0, "tombstoned content must remain searchable"

    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        status = conn.execute(
            "SELECT source_status FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert status == ("absent",)
        cursor_row = conn.execute(
            "SELECT path FROM session_files WHERE path = ?", (str(session_path),)
        ).fetchone()
        assert cursor_row is not None, "cursor must survive so a reappearance can resume"
    finally:
        conn.close()


def test_reappeared_source_clears_tombstone(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    session_id = "77777777-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    records = [_user_record("u1", "reappearmarker prompt text", str(project_dir), session_id)]
    _write_jsonl(session_path, records)
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    session_path.unlink()
    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats2.tombstoned_source_count == 1

    _write_jsonl(session_path, records)  # identical content: cursor stays consistent
    stats3 = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats3.tombstoned_source_count == 0
    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        status = conn.execute(
            "SELECT source_status FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert status == ("available",)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Model/schema mismatch forces rebuild
# ---------------------------------------------------------------------------


def test_model_mismatch_triggers_rebuild(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    session_id = "88888888-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [_user_record("u1", "model mismatch rebuild marker", str(project_dir), session_id)],
    )
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    gen_before = store.GenerationalStore(index_dir).current_generation
    conn = sqlite3.connect(str(index_dir / "index.db"))
    conn.execute("UPDATE meta SET value = ? WHERE key = 'model_id'", ("stale-model-v0",))
    conn.commit()
    conn.close()

    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    gen_after = store.GenerationalStore(index_dir).current_generation

    assert gen_after > gen_before, "a model_id mismatch must force a fresh generation"
    assert stats2.model_id == embed.MODEL_ID
    live_db = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(live_db))
    try:
        assert store.get_meta(conn, "model_id") == embed.MODEL_ID
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# no_subagents flag
# ---------------------------------------------------------------------------


def test_no_subagents_flag_restricts_to_main(fake_home, tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    main_id = "aaaaaaaa-2222-4333-8444-555555555555"
    main_path = fake_home / ".claude" / "projects" / "proj" / f"{main_id}.jsonl"
    _write_jsonl(
        main_path, [_user_record("u1", "mainsessionmarker text", str(project_dir), main_id)]
    )

    sub_path = (
        fake_home / ".claude" / "projects" / "proj" / main_id / "subagents" / "agent-deadbeef.jsonl"
    )
    _write_jsonl(sub_path, [_user_record("su1", "subagentmarker text", str(project_dir), main_id)])

    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    stats = indexer.index(project_dir, index_dir=index_dir, no_subagents=True, quiet=True)

    assert stats.session_count == 1
    assert len(_fts_hits(index_dir, "mainsessionmarker")) == 1
    assert len(_fts_hits(index_dir, "subagentmarker")) == 0


# ---------------------------------------------------------------------------
# Crash recovery must be WIRED into index(), not merely available on the
# store. GenerationalStore.recover()/checkpoint() are independently unit
# tested in test_atomicity.py, which passes whether or not index() ever
# calls them -- so these tests assert the integration itself, and each is
# mutation-tested by deleting its call site from indexer.index().
# ---------------------------------------------------------------------------


def _vector_row_count(vec_path: Path) -> int:
    return vec_path.stat().st_size // store.GenerationalStore.VECTOR_ROW_BYTES


def _live_fts_hits(index_dir: Path, query: str) -> list[tuple[str, float]]:
    """FTS hits against whichever generation the manifest currently points at.

    _fts_hits() hardcodes generation 0's index.db, which is correct for the
    tests above (routine indexing never leaves generation 0) but would make a
    rebuild look like an unrelated "no such table" error rather than the
    content loss it actually is.
    """
    live_db = store.GenerationalStore(index_dir).get_index_path()
    if not live_db.exists():
        return []
    conn = sqlite3.connect(str(live_db))
    try:
        return store.search_fts(conn, query)
    finally:
        conn.close()


def _simulate_crash_after_vector_append(vec_path: Path, n: int) -> None:
    """Reproduce the crash window index() exists to close: vectors.append()
    reached disk, but the SQLite commit that would reference those rows --
    and the checkpoint that would journal them -- never landed.
    """
    with open(vec_path, "ab") as f:
        f.write(b"\xab" * (n * store.GenerationalStore.VECTOR_ROW_BYTES))


def test_crash_residue_is_recovered_in_place_not_by_full_rebuild(fake_home, tmp_path, monkeypatch):
    """Guards the gen_store.recover() call at the top of index().

    Without it, the orphaned trailing rows a crash leaves behind are seen by
    rebuild_guard.needs_rebuild()'s validate_alignment() as corruption, which escalates a
    1024-byte truncation into a full re-embed of the entire corpus into a new
    generation. Asserting only "the store ends up consistent" would pass
    either way -- a rebuild is also consistent -- so this pins the mechanism:
    same generation, and the encoder is never called.
    """
    project_dir = tmp_path / "project"
    session_id = "bbbbbbbb-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record("u1", "crashrecoverymarker prompt text", str(project_dir), session_id),
            _assistant_record(
                "a1", "crashrecoverymarker response text", str(project_dir), session_id, "u1"
            ),
        ],
    )
    index_dir = tmp_path / "idx"
    counter = _install_fake_encode(monkeypatch)
    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats1.chunk_count > 0

    vec_path = store.GenerationalStore(index_dir).get_vector_path()
    rows_before = _vector_row_count(vec_path)
    gen_before = store.GenerationalStore(index_dir).current_generation

    _simulate_crash_after_vector_append(vec_path, 2)
    assert _vector_row_count(vec_path) == rows_before + 2
    valid, _ = vectors.validate_alignment(index_dir / "index.db", vec_path)
    assert not valid, "fixture must actually leave the store inconsistent"

    counter["calls"] = 0
    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert counter["calls"] == 0, (
        "recovering from a crash must not re-embed the corpus -- a full rebuild "
        "would, which is exactly what an unwired recover() falls back to"
    )
    assert store.GenerationalStore(index_dir).current_generation == gen_before, (
        "crash recovery must happen in place; bumping the generation means the "
        "orphaned rows were treated as corruption and the whole store was rebuilt"
    )
    assert _vector_row_count(vec_path) == rows_before, "orphaned rows must be truncated away"
    valid, msg = vectors.validate_alignment(index_dir / "index.db", vec_path)
    assert valid, msg
    assert stats2.chunk_count == stats1.chunk_count
    assert len(_fts_hits(index_dir, "crashrecoverymarker")) > 0


def test_index_journals_durable_vector_watermark_after_commit(fake_home, tmp_path, monkeypatch):
    """Guards the gen_store.checkpoint() call after index()'s conn.commit().

    Without it the incremental path never writes a manifest at all, so the
    journal permanently claims zero rows are known-good while the file holds
    N -- recover()'s cheap fast path can then never engage, and nothing
    records where durability actually ended.
    """
    project_dir = tmp_path / "project"
    session_id = "cccccccc-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [_user_record("u1", "watermarkmarker prompt text", str(project_dir), session_id)],
    )
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats.chunk_count > 0

    manifest_path = index_dir / ".manifest"
    assert manifest_path.exists(), "a routine incremental index must journal a manifest"

    reopened = store.GenerationalStore(index_dir)
    actual_rows = _vector_row_count(reopened.get_vector_path())
    assert actual_rows == stats.chunk_count
    assert reopened.committed_vector_rows == actual_rows, (
        "the journalled watermark must match the rows actually on disk, or "
        "recover() cannot tell what was durable at the last good commit"
    )
    assert reopened.current_generation == 0, "an incremental index must not bump the generation"
    assert reopened.recover() == (0, 0), "a checkpointed store has nothing to repair"


def test_recovery_preserves_tombstoned_content_a_rebuild_would_destroy(
    fake_home, tmp_path, monkeypatch
):
    """The sharpest consequence of leaving recover() unwired.

    Falling back to a rebuild is not merely a slower route to the same state.
    A rebuild re-derives the index from the transcripts still on disk, so any
    session whose source has vanished -- content ssgrep deliberately retains
    via tombstoning -- is silently destroyed by it. In-place recovery keeps
    that content; the rebuild an unwired recover() triggers does not.
    """
    project_dir = tmp_path / "project"
    session_id = "dddddddd-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [_user_record("u1", "tombstonesurvivalmarker prompt", str(project_dir), session_id)],
    )
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    session_path.unlink()
    stats2 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats2.tombstoned_source_count == 1
    assert len(_live_fts_hits(index_dir, "tombstonesurvivalmarker")) > 0

    vec_path = store.GenerationalStore(index_dir).get_vector_path()
    _simulate_crash_after_vector_append(vec_path, 1)

    indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert len(_live_fts_hits(index_dir, "tombstonesurvivalmarker")) > 0, (
        "crash recovery must not destroy tombstoned content whose source is "
        "gone -- a rebuild cannot re-derive it from a transcript that no "
        "longer exists, so it would be lost for good"
    )


# ---------------------------------------------------------------------------
# Episode/chunk duplication regression (the whole-batch fallback defect).
# Before the fix, whenever _split_into_episode_groups's group count
# disagreed with episodes.segment_episodes's episode count -- e.g. on the
# empty-content merge quirk exercised below -- EVERY episode got chunked
# from the entire batch of new_records, multiplying the index by the
# episode count (measured 733x on this repo's own main session; see
# episodes.segment_episode_groups's docstring and indexer._build_episodes).
# ---------------------------------------------------------------------------


def test_no_cross_episode_chunk_duplication(fake_home, tmp_path, monkeypatch):
    """Index a synthetic multi-episode session and assert (a) no chunk text
    is duplicated anywhere in the corpus -- total row count equals distinct
    text count -- and (b) every episode's chunks derive only from its own
    records: no chunk anywhere carries another episode's marker.
    """
    project_dir = tmp_path / "project"
    session_id = "aaaaaaaa-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    def long_text(marker: str) -> str:
        # Every word is unique (marker + running index), not a repeated
        # filler phrase: guarantees no two windows anywhere -- including a
        # coincidental prompt-vs-response window within the same episode
        # -- ever land on byte-identical text, and puts the marker in
        # every chunk rather than only at the start/end.
        return " ".join(f"{marker}_seg{i}" for i in range(220))

    markers = ["ALPHA_MARK", "BRAVO_MARK", "CHARLIE_MARK"]
    records: list[dict[str, Any]] = []
    parent: str | None = None
    for i, marker in enumerate(markers):
        uid_q, uid_a = f"q{i}", f"a{i}"
        records.append(
            _user_record(uid_q, long_text(f"PROMPT_{marker}"), str(project_dir), session_id, parent)
        )
        records.append(
            _assistant_record(
                uid_a, long_text(f"RESPONSE_{marker}"), str(project_dir), session_id, uid_q
            )
        )
        parent = uid_a
    _write_jsonl(session_path, records)

    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.episode_count == 3
    assert stats.chunk_count > 3, "fixture must actually produce multiple chunks per episode"

    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        distinct = conn.execute("SELECT COUNT(DISTINCT text) FROM chunks").fetchone()[0]
        assert (
            total == distinct
        ), f"chunk duplication detected: {total} rows but only {distinct} distinct texts"
        rows = conn.execute("SELECT episode_id, text FROM chunks").fetchall()
    finally:
        conn.close()

    assert len(rows) == stats.chunk_count
    for episode_id, text in rows:
        own_index = int(episode_id.split(":ep:")[-1])
        for other_index, other_marker in enumerate(markers):
            if other_index == own_index:
                continue
            assert (
                other_marker not in text
            ), f"{episode_id} contains foreign marker {other_marker!r}: {text[:200]!r}"


def test_empty_content_merge_quirk_end_to_end_no_duplication(fake_home, tmp_path, monkeypatch):
    """Force, through the real indexer pipeline, the exact record shape
    that used to trigger the whole-batch fallback: two consecutive
    tool-result-only `user` turns (empty extracted text -- see
    _tool_result_user_record) followed by a real assistant response, then a
    normal second episode. Under the old code this produced a
    segment_episodes()/_split_into_episode_groups() count mismatch (1
    episode vs. 2 groups) and every episode was chunked from the whole
    batch. Assert the same invariants as the marker test above: no
    duplication, no cross-episode content bleed.
    """
    project_dir = tmp_path / "project"
    session_id = "bbbbbbbb-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    def long_text(marker: str) -> str:
        # Every word is unique (marker + running index), not a repeated
        # filler phrase: guarantees no two windows anywhere -- including a
        # coincidental prompt-vs-response window within the same episode
        # -- ever land on byte-identical text, and puts the marker in
        # every chunk rather than only at the start/end.
        return " ".join(f"{marker}_seg{i}" for i in range(220))

    records = [
        _tool_result_user_record("tr0", str(project_dir), session_id),
        _tool_result_user_record("tr1", str(project_dir), session_id, "tr0"),
        _assistant_record(
            "a0", long_text("FIRST_EPISODE_MARK"), str(project_dir), session_id, "tr1"
        ),
        _user_record("q1", long_text("SECOND_PROMPT_MARK"), str(project_dir), session_id, "a0"),
        _assistant_record(
            "a1", long_text("SECOND_RESPONSE_MARK"), str(project_dir), session_id, "q1"
        ),
    ]
    _write_jsonl(session_path, records)

    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)
    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.episode_count == 2, "the two empty tool-result turns must fold into one episode"

    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        distinct = conn.execute("SELECT COUNT(DISTINCT text) FROM chunks").fetchone()[0]
        assert (
            total == distinct
        ), f"chunk duplication detected: {total} rows but only {distinct} distinct texts"
        rows = conn.execute("SELECT episode_id, text FROM chunks").fetchall()
    finally:
        conn.close()

    ep0_id, ep1_id = f"{session_id}:ep:0", f"{session_id}:ep:1"
    for episode_id, text in rows:
        assert episode_id in (ep0_id, ep1_id)
        if episode_id == ep0_id:
            assert "SECOND_PROMPT_MARK" not in text
            assert "SECOND_RESPONSE_MARK" not in text
        else:
            assert "FIRST_EPISODE_MARK" not in text


def test_chunk_ids_stable_and_do_not_grow_across_independent_full_rebuilds(
    fake_home, tmp_path, monkeypatch
):
    """Two independent full builds over byte-identical input must produce
    the exact same set of chunk ids and the exact same chunk count -- not
    merely 'a store that looks consistent', but content-derived ids that
    are actually reproducible. rebuild=True forces a real full re-parse
    and re-chunk into a fresh generation (unlike a routine incremental run,
    which would just skip already-processed bytes and not exercise
    chunk_id computation a second time at all).
    """
    project_dir = tmp_path / "project"
    session_id = "cccccccc-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record(
                "u1", "rebuildstabilitymarker prompt text here", str(project_dir), session_id
            ),
            _assistant_record(
                "a1",
                "rebuildstabilitymarker response text with enough content to chunk nicely",
                str(project_dir),
                session_id,
                "u1",
            ),
        ],
    )
    index_dir = tmp_path / "idx"
    _install_fake_encode(monkeypatch)

    stats1 = indexer.index(project_dir, index_dir=index_dir, quiet=True)
    assert stats1.chunk_count > 0
    db1 = store.GenerationalStore(index_dir).get_index_path()
    conn = sqlite3.connect(str(db1))
    try:
        ids1 = sorted(r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall())
    finally:
        conn.close()

    stats2 = indexer.index(project_dir, index_dir=index_dir, rebuild=True, quiet=True)
    db2 = store.GenerationalStore(index_dir).get_index_path()
    assert db2 != db1, "rebuild=True must actually land in a new generation"
    conn = sqlite3.connect(str(db2))
    try:
        ids2 = sorted(r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall())
    finally:
        conn.close()

    assert (
        stats2.chunk_count == stats1.chunk_count
    ), "re-indexing unchanged content from scratch must not grow the chunk count"
    assert ids2 == ids1, "chunk ids must be stable across two independent runs over identical input"


# ---------------------------------------------------------------------------
# Real corpus (skips cleanly if absent)
# ---------------------------------------------------------------------------


def test_real_corpus_full_index_completes(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    require_scoped_real_corpus(repo_root)
    _install_fake_encode(monkeypatch)

    stats = indexer.index(repo_root, index_dir=tmp_path / "idx", quiet=True)

    assert stats.index_exists is True
    assert stats.schema_version == store.SCHEMA_VERSION
    assert stats.chunk_count > 0, "Real corpus should have at least one chunk"
