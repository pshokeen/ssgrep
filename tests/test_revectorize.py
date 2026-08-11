"""Tests for re-vectorization: re-embedding without re-parsing."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from ssgrep import embed, revectorize, store, vectors
from ssgrep.chunker import chunk_episode
from ssgrep.types import Chunk, Episode


def _build_index(index_dir: Path, episodes: list[Episode]) -> None:
    """Build a real on-disk index from episodes via the real sibling modules."""
    conn = store.init_db(index_dir / "index.db")
    vstore = vectors.open_vectors(index_dir / "vectors.f32", dimension=embed.DIMENSION)

    all_chunks: list[Chunk] = []
    for ep in episodes:
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)
        all_chunks.extend(chunk_episode(ep))

    if all_chunks:
        vecs = embed.encode([c.text for c in all_chunks])
        rows = vectors.append(vstore, vecs)
        for chunk, row in zip(all_chunks, rows, strict=True):
            store.insert_chunk(conn, chunk, row)

    # Create a session for these episodes
    session = store.SessionFile(
        path=Path("/tmp/test-session.jsonl"),
        session_id="test-session-001",
        is_main=True,
        size=1000,
        mtime=1234567890.0,
    )
    store.insert_session(conn, session)

    # Set model binding
    model_id, dimension = embed.get_model_info()
    store.set_meta(conn, "model_id", model_id)
    store.set_meta(conn, "vector_dimension", str(dimension))

    # Set last_index_time
    now = datetime.now(UTC)
    store.set_meta(conn, "last_index_time", now.isoformat())

    conn.commit()
    conn.close()
    vectors.close(vstore)


# Test fixture episodes
FIXTURE_EPISODES: list[Episode] = [
    Episode(
        episode_id="test-session-001:ep:0",
        session_id="test-session-001",
        prompt_text="How do I test Python code?",
        response_text="You can test Python code using pytest.",
        title="Testing Python",
        timestamp=datetime(2026, 7, 1, 10, 5, tzinfo=UTC),
        files_touched=("test_example.py",),
        git_branch="main",
    ),
    Episode(
        episode_id="test-session-001:ep:1",
        session_id="test-session-001",
        prompt_text="What is vectorization?",
        response_text=(
            "Vectorization is the process of converting data into vectors for "
            "machine learning models. Re-vectorization means re-encoding existing data."
        ),
        title="Vectorization basics",
        timestamp=datetime(2026, 7, 2, 10, 5, tzinfo=UTC),
        files_touched=("embed.py",),
        git_branch="main",
    ),
    Episode(
        episode_id="test-session-001:ep:2",
        session_id="test-session-001",
        prompt_text="How does indexing work in databases?",
        response_text=(
            "Indexing creates data structures like B-trees to speed up lookups. "
            "When you search for a value, the index helps find it quickly without "
            "scanning every row."
        ),
        title="Database indexing",
        timestamp=datetime(2026, 7, 3, 10, 5, tzinfo=UTC),
        files_touched=("index.py", "store.py"),
        git_branch="feature/indexing",
    ),
]


@pytest.fixture
def indexed_project(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small indexed project for re-vectorization testing.

    Returns (project_dir, index_dir)
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    index_dir = project_dir / ".ssgrep"
    index_dir.mkdir(parents=True, mode=0o700)

    _build_index(index_dir, FIXTURE_EPISODES)

    return project_dir, index_dir


def test_revectorize_produces_stats(indexed_project: tuple[Path, Path]) -> None:
    """Revectorization returns IndexStats with correct counts."""
    project_dir, index_dir = indexed_project

    # Get original counts
    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    chunk_count_before = conn_before.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    episode_count_before = conn_before.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    session_count_before = conn_before.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn_before.close()

    stats = revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == session_count_before
    assert stats.episode_count == episode_count_before
    assert stats.chunk_count == chunk_count_before
    assert stats.model_id == embed.MODEL_ID
    assert stats.vector_dimension == embed.DIMENSION


def test_revectorize_preserves_chunk_ids(indexed_project: tuple[Path, Path]) -> None:
    """Chunk ids remain byte-identical after re-vectorization."""
    project_dir, index_dir = indexed_project

    # Capture chunk ids before revectorization
    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    chunks_before = set(
        conn_before.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
    )
    conn_before.close()

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    # Verify chunk ids are identical after revectorization
    # Need to re-open after revectorization to get the current generation
    gen_store = store.GenerationalStore(index_dir)
    current_db = gen_store.get_index_path()
    conn_after = sqlite3.connect(str(current_db))
    chunks_after = set(
        conn_after.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
    )
    conn_after.close()

    assert chunks_before == chunks_after


def test_revectorize_preserves_episode_metadata(indexed_project: tuple[Path, Path]) -> None:
    """Episode metadata is byte-identical after re-vectorization."""
    project_dir, index_dir = indexed_project

    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    episodes_before = set(
        conn_before.execute(
            "SELECT episode_id, title, git_branch, cwd FROM episodes ORDER BY episode_id"
        ).fetchall()
    )
    conn_before.close()

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    current_db = gen_store.get_index_path()
    conn_after = sqlite3.connect(str(current_db))
    episodes_after = set(
        conn_after.execute(
            "SELECT episode_id, title, git_branch, cwd FROM episodes ORDER BY episode_id"
        ).fetchall()
    )
    conn_after.close()

    assert episodes_before == episodes_after


def test_revectorize_no_parsing(indexed_project: tuple[Path, Path]) -> None:
    """Parsing PROVABLY does not run during revectorization.

    This is the load-bearing test: monkeypatch the transcript-reading entry
    point in records.py to raise, and assert revectorize still succeeds.
    """
    project_dir, index_dir = indexed_project

    # Get chunk count before
    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    chunk_count_before = conn_before.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn_before.close()

    # Monkeypatch records.read_records to raise, proving parsing doesn't happen
    with mock.patch("ssgrep.records.read_records", side_effect=RuntimeError("PARSE MUST NOT RUN")):
        stats = revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    assert stats.chunk_count == chunk_count_before


def test_revectorize_updates_model_binding(indexed_project: tuple[Path, Path]) -> None:
    """Model id and dimension are updated in meta after re-vectorization."""
    project_dir, index_dir = indexed_project

    # Verify initial state
    db_path = index_dir / "index.db"
    conn = sqlite3.connect(str(db_path))
    initial_model = store.get_meta(conn, "model_id")
    initial_dim = store.get_meta(conn, "vector_dimension")
    conn.close()

    assert initial_model == embed.MODEL_ID
    assert initial_dim == str(embed.DIMENSION)

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    # Verify model binding is rewritten (even though it's the same model)
    gen_store = store.GenerationalStore(index_dir)
    current_db = gen_store.get_index_path()
    conn = sqlite3.connect(str(current_db))
    final_model = store.get_meta(conn, "model_id")
    final_dim = store.get_meta(conn, "vector_dimension")
    conn.close()

    assert final_model == embed.MODEL_ID
    assert final_dim == str(embed.DIMENSION)


def test_revectorize_vector_alignment(indexed_project: tuple[Path, Path]) -> None:
    """Vector matrix and chunk rows maintain alignment after re-vectorization."""
    project_dir, index_dir = indexed_project

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    db_path = gen_store.get_index_path()
    vec_path = gen_store.get_vector_path()

    # Validate alignment
    valid, msg = vectors.validate_alignment(db_path, vec_path)
    assert valid, f"Vector alignment failed: {msg}"


def test_revectorize_vector_generation_advance(indexed_project: tuple[Path, Path]) -> None:
    """Vector generation advances to ensure atomic commit."""
    project_dir, index_dir = indexed_project

    gen_store_before = store.GenerationalStore(index_dir)
    gen_before = gen_store_before.current_generation
    vec_path_before = gen_store_before.get_vector_path()

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    gen_store_after = store.GenerationalStore(index_dir)
    gen_after = gen_store_after.current_generation
    vec_path_after = gen_store_after.get_vector_path()

    # A new generation should be created
    assert gen_after > gen_before
    # The new vectors path should be different
    assert vec_path_after != vec_path_before


def test_revectorize_index_not_found(tmp_path: Path) -> None:
    """Revectorization raises IndexNotFoundError when no index exists."""
    project_dir = tmp_path / "no_index"
    project_dir.mkdir()

    with pytest.raises(Exception) as exc_info:
        revectorize.revectorize(project_dir, quiet=True)

    # The exception should indicate missing index
    error_msg = str(exc_info.value).lower()
    assert "not found" in error_msg or "no index" in error_msg


def test_revectorize_preserves_chunk_content(indexed_project: tuple[Path, Path]) -> None:
    """Chunk text content is byte-identical after re-vectorization."""
    project_dir, index_dir = indexed_project

    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    chunks_before = dict(
        conn_before.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    )
    conn_before.close()

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    current_db = gen_store.get_index_path()
    conn_after = sqlite3.connect(str(current_db))
    chunks_after = dict(
        conn_after.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    )
    conn_after.close()

    assert chunks_before == chunks_after


def test_revectorize_vec_row_remapped(indexed_project: tuple[Path, Path]) -> None:
    """Vec_row values are remapped to reflect new vector positions.

    After re-vectorization, chunks should have vec_rows pointing to valid rows
    in the newly written vectors file.
    """
    project_dir, index_dir = indexed_project

    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    vec_rows_before = set(
        conn_before.execute("SELECT vec_row FROM chunks WHERE vec_row IS NOT NULL").fetchall()
    )
    chunk_count_before = len(vec_rows_before)
    conn_before.close()

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    current_db = gen_store.get_index_path()
    conn_after = sqlite3.connect(str(current_db))
    vec_rows_after = set(
        conn_after.execute("SELECT vec_row FROM chunks WHERE vec_row IS NOT NULL").fetchall()
    )
    conn_after.close()

    # Both should have the same count of vec_rows
    assert len(vec_rows_after) == chunk_count_before

    # All vec_rows should be in range [0, chunk_count)
    for (row,) in vec_rows_after:
        assert 0 <= row < chunk_count_before


def test_revectorize_updates_last_index_time(indexed_project: tuple[Path, Path]) -> None:
    """last_index_time in meta is updated during revectorization."""
    project_dir, index_dir = indexed_project

    db_path = index_dir / "index.db"
    conn_before = sqlite3.connect(str(db_path))
    time_before = store.get_meta(conn_before, "last_index_time")
    conn_before.close()

    import time as time_module

    time_module.sleep(0.01)  # Ensure time advances

    revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)

    gen_store = store.GenerationalStore(index_dir)
    current_db = gen_store.get_index_path()
    conn_after = sqlite3.connect(str(current_db))
    time_after = store.get_meta(conn_after, "last_index_time")
    conn_after.close()

    assert time_before != time_after
    # Verify the new time is later
    assert time_after > time_before


def test_revectorize_mutation_fallback_to_full_reindex(indexed_project):
    """Mutation test: prove test catches if revectorize falls back to full re-index.

    This test verifies that revectorize performs only vector operations, never
    re-reading transcripts. If revectorize calls records.read_records, the
    monkeypatch causes it to raise, and this test fails.
    """
    project_dir, index_dir = indexed_project

    # Get the original chunk count before revectorize
    import sqlite3

    gen_store = store.GenerationalStore(index_dir)
    db_path = gen_store.get_index_path()
    conn = sqlite3.connect(str(db_path))
    original_chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    with mock.patch("ssgrep.records.read_records", side_effect=RuntimeError("PARSE MUST NOT RUN")):
        # This should succeed with the correct implementation
        stats = revectorize.revectorize(project_dir, index_dir=index_dir, quiet=True)
        assert stats.chunk_count == original_chunk_count

    # If the mutation causes revectorize to call records.read_records,
    # the monkeypatch would have raised and this assertion would not be reached.
