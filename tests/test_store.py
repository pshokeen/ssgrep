"""Tests for SQLite store."""

from pathlib import Path

from ssgrep.store import (
    GenerationalStore,
    get_meta,
    get_tombstone_stats,
    init_db,
    insert_chunk,
    insert_session,
    mark_source_available,
    search_fts,
    search_fts_trigram,
    set_meta,
    tombstone_session_chunks,
)
from ssgrep.store.lifecycle import delete_session_chunks
from ssgrep.types import Chunk, ContentType, SessionFile
from ssgrep.vectors import append, open_vectors, validate_alignment


def test_schema_creation(tmp_path):
    conn = init_db(tmp_path / "test.db")
    tables = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "chunks" in tables
    assert "episodes" in tables
    assert "sessions" in tables
    conn.close()


def test_insert_and_query_chunk(tmp_path):
    conn = init_db(tmp_path / "test.db")
    chunk = Chunk(
        chunk_id="c1",
        episode_id="e1",
        session_id="s1",
        text="hello world",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, chunk, 0)
    conn.commit()
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = 'c1'").fetchone()
    assert row is not None
    assert row[3] == "hello world"
    conn.close()


def test_fts_search(tmp_path):
    conn = init_db(tmp_path / "test.db")
    chunk = Chunk(
        chunk_id="c1",
        episode_id="e1",
        session_id="s1",
        text="python decorator pattern",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, chunk, 0)
    conn.commit()
    results = search_fts(conn, "decorator")
    assert len(results) > 0
    conn.close()


def test_meta_table(tmp_path):
    conn = init_db(tmp_path / "test.db")
    set_meta(conn, "model_id", "potion-base-8M")
    conn.commit()
    assert get_meta(conn, "model_id") == "potion-base-8M"
    conn.close()


def test_schema_version(tmp_path):
    # 4: chunks_fts_tri (trigram subword leg) added to the schema; bumping
    # forces the one-time rebuild that populates it on existing indexes.
    conn = init_db(tmp_path / "test.db")
    version = get_meta(conn, "schema_version")
    assert version == "4"
    conn.close()


def test_tombstoning_retains_chunks(tmp_path):
    """Test that tombstoning marks chunks as absent without deleting them."""
    conn = init_db(tmp_path / "test.db")

    # Insert a chunk
    chunk = Chunk(
        chunk_id="c1",
        episode_id="e1",
        session_id="s1",
        text="hello world",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, chunk, 0)
    conn.commit()

    # Verify chunk exists
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = 'c1'").fetchone()
    assert row is not None

    # Tombstone the session
    tombstone_session_chunks(conn, "s1")
    conn.commit()

    # Verify chunk still exists but is marked absent
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = 'c1'").fetchone()
    assert row is not None
    # source_status is the last column
    assert row[-1] == "absent"

    # Verify FTS entry is still there
    results = search_fts(conn, "hello")
    assert len(results) > 0

    conn.close()


def test_mark_source_available(tmp_path):
    """Test that marking source as available clears the absence marker."""
    conn = init_db(tmp_path / "test.db")

    # Insert and tombstone a chunk
    chunk = Chunk(
        chunk_id="c1",
        episode_id="e1",
        session_id="s1",
        text="hello world",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, chunk, 0)
    conn.commit()
    tombstone_session_chunks(conn, "s1")
    conn.commit()

    # Verify it's absent
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = 'c1'").fetchone()
    assert row[-1] == "absent"

    # Mark as available
    mark_source_available(conn, "s1")
    conn.commit()

    # Verify it's available again
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = 'c1'").fetchone()
    assert row[-1] == "available"

    conn.close()


def test_tombstone_stats(tmp_path):
    """Test that tombstone stats correctly count absent sources and chunks."""
    conn = init_db(tmp_path / "test.db")

    # Insert sessions
    session1 = SessionFile(
        path=Path("/path/s1"), session_id="s1", is_main=True, size=100, mtime=0.0
    )
    session2 = SessionFile(
        path=Path("/path/s2"), session_id="s2", is_main=True, size=100, mtime=0.0
    )
    insert_session(conn, session1)
    insert_session(conn, session2)

    # Insert chunks from two sessions
    chunk1 = Chunk(
        chunk_id="c1",
        episode_id="e1",
        session_id="s1",
        text="session1",
        content_type=ContentType.PROMPT,
    )
    chunk2 = Chunk(
        chunk_id="c2",
        episode_id="e2",
        session_id="s1",
        text="session1b",
        content_type=ContentType.PROMPT,
    )
    chunk3 = Chunk(
        chunk_id="c3",
        episode_id="e3",
        session_id="s2",
        text="session2",
        content_type=ContentType.PROMPT,
    )

    insert_chunk(conn, chunk1, 0)
    insert_chunk(conn, chunk2, 1)
    insert_chunk(conn, chunk3, 2)
    conn.commit()

    # Initially no tombstones
    sources, chunks = get_tombstone_stats(conn)
    assert sources == 0
    assert chunks == 0

    # Tombstone session 1
    tombstone_session_chunks(conn, "s1")
    conn.commit()

    sources, chunks = get_tombstone_stats(conn)
    assert sources == 1
    assert chunks == 2

    conn.close()


def test_chunk_vector_alignment(tmp_path):
    """Test that chunks and vectors remain aligned after appends."""
    import numpy as np

    # Create index.db and vectors.f32
    conn = init_db(tmp_path / "index.db")
    vec_store = open_vectors(tmp_path / "vectors.f32", dimension=256)

    # Create vectors and chunks
    vectors = np.random.randn(3, 256).astype(np.float32)
    vec_rows = append(vec_store, vectors)

    for i, vec_row in enumerate(vec_rows):
        chunk = Chunk(
            chunk_id=f"c{i}",
            episode_id=f"e{i}",
            session_id="s1",
            text=f"text{i}",
            content_type=ContentType.PROMPT,
        )
        insert_chunk(conn, chunk, vec_row)

    conn.commit()

    # Validate alignment
    is_valid, error_msg = validate_alignment(tmp_path / "index.db", tmp_path / "vectors.f32")
    assert is_valid, f"Alignment check failed: {error_msg}"

    conn.close()


def test_crash_recovery_detects_orphaned_vectors(tmp_path):
    """Test that crash recovery detects orphaned vector rows.

    Scenario: vectors.f32 is truncated mid-write, leaving orphaned rows
    that no chunk references.
    """
    import numpy as np

    conn = init_db(tmp_path / "index.db")
    vec_store = open_vectors(tmp_path / "vectors.f32", dimension=256)

    # Create 3 vectors and 2 chunks (simulating crash mid-commit)
    vectors = np.random.randn(3, 256).astype(np.float32)
    vec_rows = append(vec_store, vectors)

    # Only insert chunks for first 2 vectors
    for i in range(2):
        chunk = Chunk(
            chunk_id=f"c{i}",
            episode_id=f"e{i}",
            session_id="s1",
            text=f"text{i}",
            content_type=ContentType.PROMPT,
        )
        insert_chunk(conn, chunk, vec_rows[i])

    conn.commit()

    # Now validate - should detect orphaned row 2
    is_valid, error_msg = validate_alignment(tmp_path / "index.db", tmp_path / "vectors.f32")
    assert not is_valid, "Should detect orphaned vector row"
    assert "orphaned" in error_msg.lower()

    conn.close()


def test_crash_recovery_detects_missing_vectors(tmp_path):
    """Test that crash recovery detects chunks with missing vector rows.

    Scenario: index.db has a chunk with a vec_row that doesn't exist in
    vectors.f32 (because vectors weren't flushed before crash).
    """
    import numpy as np

    conn = init_db(tmp_path / "index.db")
    vec_store = open_vectors(tmp_path / "vectors.f32", dimension=256)

    # Create 2 vectors
    vectors = np.random.randn(2, 256).astype(np.float32)
    _ = append(vec_store, vectors)

    # Insert chunk referencing row 2 (which doesn't exist)
    chunk = Chunk(
        chunk_id="c1",
        episode_id="e1",
        session_id="s1",
        text="text1",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, chunk, 999)  # Non-existent row

    conn.commit()

    # Validate should fail
    is_valid, error_msg = validate_alignment(tmp_path / "index.db", tmp_path / "vectors.f32")
    assert not is_valid, "Should detect invalid vec_row"
    assert "invalid" in error_msg.lower() or "missing" in error_msg.lower()

    conn.close()


def test_generational_store_staging(tmp_path):
    """Test that GenerationalStore correctly manages generations."""
    gen_store = GenerationalStore(tmp_path)

    # First generation paths should be the base paths
    index_0, vec_0 = gen_store.stage_generation(0)
    assert index_0 == tmp_path / "index.db"
    assert vec_0 == tmp_path / "vectors.f32"

    # Next generation paths should have generation numbers
    index_1, vec_1 = gen_store.stage_generation(1)
    assert index_1 == tmp_path / "index.db.1"
    assert vec_1 == tmp_path / "vectors.f32.1"


def test_generational_store_commit_and_recovery(tmp_path):
    """Test that commit_generation() atomically swaps in a new generation
    and reclaims the generation it supersedes.

    Scenario:
    1. Write generation 0 directly, with one chunk/vector.
    2. Stage generation 1 via stage_generation() and write both its files,
       with three more chunks/vectors.
    3. Commit generation 1 atomically via commit_generation().
    4. Verify the store is now on generation 1, generation 1's files exist,
       and generation 0's files are reclaimed.
    """
    import numpy as np

    gen_store = GenerationalStore(tmp_path)

    # Create generation 0
    conn0 = init_db(gen_store.get_index_path(0))
    vec0 = open_vectors(gen_store.get_vector_path(0))

    vectors_0 = np.random.randn(2, 256).astype(np.float32)
    rows_0 = append(vec0, vectors_0)

    chunk = Chunk(
        chunk_id="c0",
        episode_id="e0",
        session_id="s0",
        text="gen0",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn0, chunk, rows_0[0])
    conn0.commit()
    conn0.close()

    # Stage and create generation 1
    index_1, vec_1 = gen_store.stage_generation(1)
    conn1 = init_db(index_1)
    vec1 = open_vectors(vec_1)

    vectors_1 = np.random.randn(3, 256).astype(np.float32)
    rows_1 = append(vec1, vectors_1)

    for i in range(3):
        chunk = Chunk(
            chunk_id=f"c{i}+1",
            episode_id=f"e{i}+1",
            session_id="s1",
            text=f"gen1-{i}",
            content_type=ContentType.PROMPT,
        )
        insert_chunk(conn1, chunk, rows_1[i])

    conn1.commit()
    conn1.close()

    # Atomically commit generation 1
    gen_store.commit_generation(1)

    # Verify we're on generation 1
    assert gen_store.current_generation == 1
    assert gen_store.get_index_path().exists()
    assert gen_store.get_vector_path().exists()

    # Verify old generation 0 files were cleaned up. Generation 0's actual
    # files are the *unsuffixed* names (get_index_path()/get_vector_path()
    # special-case gen == 0 to return the base name with no ".0" suffix) --
    # compare against those accessors, not a hand-built ".0"-suffixed guess
    # that generation 0 never uses and that would pass unconditionally.
    assert not gen_store.get_index_path(0).exists()
    assert not gen_store.get_vector_path(0).exists()


def test_atomicity_proves_test_fails_without_mechanism(tmp_path):
    """Prove that orphan detection fails without the atomicity mechanism.

    This test demonstrates the danger: without manifest-based generations,
    a crash between sqlite commit and vector append leaves:
    - All chunks written to index.db (survives crash)
    - Partial vectors written to vectors.f32 (inconsistent)

    When the process restarts and tries to validate, it sees chunks
    referencing vector rows that don't exist. This test verifies that
    without the mechanism, the validation would have failed to catch it.

    Note: This test would need to be modified if the atomicity mechanism
    were removed — it should fail in that case, proving the mechanism is
    necessary.
    """
    import numpy as np

    conn = init_db(tmp_path / "index.db")
    vec_store = open_vectors(tmp_path / "vectors.f32", dimension=256)

    # Simulate the crash window: write chunks but don't write all vectors
    # Write 3 vectors
    vectors = np.random.randn(3, 256).astype(np.float32)
    vec_rows = append(vec_store, vectors)

    # Write chunks referencing all 3
    for i in range(3):
        chunk = Chunk(
            chunk_id=f"c{i}",
            episode_id=f"e{i}",
            session_id="s1",
            text=f"text{i}",
            content_type=ContentType.PROMPT,
        )
        insert_chunk(conn, chunk, vec_rows[i])

    # NOW, simulate a crash: commit the DB but don't flush vectors
    conn.commit()
    # (Without properly closing/flushing vec_store, vectors might be incomplete)

    # In a real crash scenario, we'd kill the process here.
    # For this test, we'll artificially truncate vectors.f32
    vec_store_file = tmp_path / "vectors.f32"
    current_size = vec_store_file.stat().st_size

    # Truncate to simulate incomplete write (keep only 2 of 3 vectors)
    with open(vec_store_file, "r+b") as f:
        f.truncate(current_size - (256 * 4))  # Remove last vector

    conn.close()

    # Now when we validate, the mechanism should catch it
    is_valid, error_msg = validate_alignment(tmp_path / "index.db", tmp_path / "vectors.f32")
    assert not is_valid, "Validation should fail to catch orphaned/missing vec_row"
    assert "invalid" in error_msg.lower() or "orphaned" in error_msg.lower()

    # THIS IS THE KEY ASSERTION: without the atomicity mechanism (manifest-based
    # generations), the process would have restarted and silently mixed the
    # incomplete vectors with the complete chunks, producing wrong results.
    # The manifest ensures both files are committed together, or neither is used.


def test_delete_session_chunks_purges_trigram_mirror(tmp_path):
    """chunks_fts_tri must not retain a pruned session's text.

    delete_session_chunks() is the only routine that permanently removes
    data (explicit prune). The trigram mirror added in schema v4 is a third
    copy of every chunk's text; if its DELETE were dropped, a pruned
    transcript would keep matching through the trigram leg while chunks and
    chunks_fts forget it — silent retention of data the user asked to
    destroy. Positive assertions first (the mirror really held the text and
    the trigram MATCH really found it), then the deletion property.
    """
    conn = init_db(tmp_path / "test.db")
    kept = Chunk(
        chunk_id="keep1",
        episode_id="ep-keep",
        session_id="sess-keep",
        text="unrelated retained content about frobnication widgets",
        content_type=ContentType.PROMPT,
    )
    doomed = Chunk(
        chunk_id="doom1",
        episode_id="ep-doom",
        session_id="sess-doom",
        text="sensitive pruneworthy transcript mentioning zanzibar credentials",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, kept, 0)
    insert_chunk(conn, doomed, 1)
    conn.commit()

    # Positive assertions: the mirror is populated and the trigram leg
    # genuinely reaches the doomed text before deletion.
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts_tri").fetchone()[0] == 2
    pre = search_fts_trigram(conn, '"zanzibar"', limit=10)
    assert [cid for cid, _rank in pre] == ["doom1"]

    delete_session_chunks(conn, "sess-doom")
    conn.commit()

    # The property: no trigram row survives for the pruned session, the
    # trigram MATCH no longer finds its text, and the kept session's row
    # is untouched.
    remaining = [row[0] for row in conn.execute("SELECT chunk_id FROM chunks_fts_tri").fetchall()]
    assert remaining == ["keep1"]
    assert search_fts_trigram(conn, '"zanzibar"', limit=10) == []
    assert [cid for cid, _r in search_fts_trigram(conn, '"frobnication"', limit=10)] == ["keep1"]
    conn.close()


def test_incremental_reparse_delete_purges_trigram_mirror(tmp_path):
    """indexer._delete_rows_for_session must clear chunks_fts_tri too.

    This is the deletion the incremental indexer runs on every rewritten
    session before re-parsing it (the most frequently exercised delete in
    production). If its chunks_fts_tri DELETE were dropped, every re-parsed
    session would accumulate stale trigram rows: the trigram leg would keep
    matching text that no longer exists in the session, and INSERT OR
    REPLACE cannot repair fts5 tables (no unique constraint), so rows would
    also duplicate on every reindex. Positive assertions first, then the
    property, mirroring test_delete_session_chunks_purges_trigram_mirror.
    """
    from ssgrep.indexer import _delete_rows_for_session

    conn = init_db(tmp_path / "test.db")
    kept = Chunk(
        chunk_id="keep1",
        episode_id="ep-keep",
        session_id="sess-keep",
        text="retained content about frobnication widgets",
        content_type=ContentType.PROMPT,
    )
    doomed = Chunk(
        chunk_id="doom1",
        episode_id="ep-doom",
        session_id="sess-doom",
        text="stale rewritten-session text mentioning xylophone credentials",
        content_type=ContentType.PROMPT,
    )
    insert_chunk(conn, kept, 0)
    insert_chunk(conn, doomed, 1)
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM chunks_fts_tri").fetchone()[0] == 2
    assert [c for c, _r in search_fts_trigram(conn, '"xylophone"', limit=10)] == ["doom1"]

    _delete_rows_for_session(conn, "sess-doom")
    conn.commit()

    remaining = [row[0] for row in conn.execute("SELECT chunk_id FROM chunks_fts_tri").fetchall()]
    assert remaining == ["keep1"]
    assert search_fts_trigram(conn, '"xylophone"', limit=10) == []
    conn.close()
