"""Tests for episode drill-down (show command)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from ssgrep import chunker, detail, discovery, search, staleness, store
from ssgrep.types import Episode, FileCursor


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory for testing."""
    return tmp_path


@pytest.fixture
def index_with_episodes(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build an index containing test episodes.

    Returns (project_dir, {episode_id: session_id})
    """
    index_dir = tmp_path / ".ssgrep"
    index_dir.mkdir(parents=True, mode=0o700)

    db_path = index_dir / "index.db"
    conn = store.init_db(db_path)

    session_id = "test-session-001"
    test_ts = datetime(2026, 7, 25, 10, 30, 0)

    # Insert session
    test_session = store.SessionFile(
        path=Path("/home/test/transcript.jsonl"),
        session_id=session_id,
        is_main=True,
        size=1000,
        mtime=1234567890.0,
    )
    store.insert_session(conn, test_session)

    # Create episodes with content
    episode_ids: dict[str, str] = {}

    # Episode 0: short content
    ep0 = Episode(
        episode_id=f"{session_id}:ep:0",
        session_id=session_id,
        prompt_text="What is Python?",
        response_text="Python is a programming language.",
        title="Python basics",
        timestamp=test_ts,
        git_branch="main",
        cwd="/home/test",
        files_touched=("script.py", "test.py"),
        tool_names=("Read", "Write"),
        is_subagent=False,
    )
    store.insert_episode(conn, ep0, ep0.prompt_text, ep0.response_text)
    episode_ids["short"] = ep0.episode_id

    # Episode 1: content that will exceed prompt bound (> 50k chars)
    long_prompt = "X" * 60000  # Exceeds MAX_PROMPT_CHARS
    ep1 = Episode(
        episode_id=f"{session_id}:ep:1",
        session_id=session_id,
        prompt_text=long_prompt,
        response_text="Response to long prompt",
        title="Long prompt episode",
        timestamp=test_ts,
        git_branch="main",
        cwd="/home/test",
        files_touched=(),
        tool_names=(),
        is_subagent=False,
    )
    store.insert_episode(conn, ep1, ep1.prompt_text, ep1.response_text)
    episode_ids["long_prompt"] = ep1.episode_id

    # Episode 2: content that will exceed response bound (> 150k chars)
    long_response = "Y" * 200000  # Exceeds MAX_RESPONSE_CHARS
    ep2 = Episode(
        episode_id=f"{session_id}:ep:2",
        session_id=session_id,
        prompt_text="Brief question",
        response_text=long_response,
        title="Long response episode",
        timestamp=test_ts,
        git_branch="develop",
        cwd="/home/test/subdir",
        files_touched=("large_file.py",),
        tool_names=("Bash", "Edit"),
        is_subagent=False,
    )
    store.insert_episode(conn, ep2, ep2.prompt_text, ep2.response_text)
    episode_ids["long_response"] = ep2.episode_id

    # Episode 3: subagent episode
    ep3 = Episode(
        episode_id=f"{session_id}:ep:3",
        session_id=session_id,
        prompt_text="Investigate the issue",
        response_text="Found and fixed the bug in module.py",
        title="Subagent investigation",
        timestamp=test_ts,
        git_branch="main",
        cwd="/home/test",
        files_touched=("module.py",),
        tool_names=(),
        is_subagent=True,
        agent_type="critic",
        agent_name="error-finder",
        agent_description="Finds errors in code",
        parent_session_id="parent-session-id",
    )
    store.insert_episode(conn, ep3, ep3.prompt_text, ep3.response_text)
    episode_ids["subagent"] = ep3.episode_id

    # Insert chunks for the episodes using the actual chunker to split large texts
    for _ep_id, episode in [
        (ep0.episode_id, ep0),
        (ep1.episode_id, ep1),
        (ep2.episode_id, ep2),
        (ep3.episode_id, ep3),
    ]:
        # Use the actual chunker to split episode text into chunks
        chunks = chunker.chunk_episode(episode)
        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)

    conn.commit()
    conn.close()

    return tmp_path, episode_ids


class TestShowValidRef:
    """Tests for show() with valid refs."""

    def test_valid_ref_returns_episode_detail(self, index_with_episodes: tuple):
        """Valid ref retrieves full episode context."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["short"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert result.episode_id == ep_id
        assert result.session_id == "test-session-001"
        assert result.title == "Python basics"
        assert result.prompt_text == "What is Python?"
        assert result.response_text == "Python is a programming language."
        assert result.files_touched == ("script.py", "test.py")
        assert result.tool_names == ("Read", "Write")
        assert result.is_subagent is False

    def test_valid_ref_round_trips_from_search(self, index_with_episodes: tuple):
        """A ref from search.py's format resolves correctly."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["short"]

        # search.py emits ref=episode_id (line 472)
        # Verify it round-trips through show()
        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert result.episode_id == ep_id

    def test_shows_full_content_within_bounds(self, index_with_episodes: tuple):
        """Short content is shown in full, untruncated."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["short"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert "..." not in result.prompt_text
        assert "..." not in result.response_text

    def test_subagent_attributes_populated(self, index_with_episodes: tuple):
        """Subagent episodes include attribution fields."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["subagent"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert result.is_subagent is True
        assert result.agent_type == "critic"
        assert result.agent_name == "error-finder"
        assert result.agent_description == "Finds errors in code"
        assert result.parent_session_id == "parent-session-id"

    def test_metadata_fields_populated(self, index_with_episodes: tuple):
        """All metadata fields are correctly populated."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["short"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert result.git_branch == "main"
        assert result.cwd == "/home/test"
        assert result.timestamp is not None


class TestShowBounds:
    """Tests for bounded output with explicit truncation."""

    def test_prompt_truncated_when_exceeds_bound(self, index_with_episodes: tuple):
        """Prompt exceeding MAX_PROMPT_CHARS is truncated with marker."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["long_prompt"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert len(result.prompt_text) <= detail.MAX_PROMPT_CHARS + len(detail.TRUNCATION_MARKER)
        # Verify explicit truncation marker is present
        assert "[... truncated" in result.prompt_text

    def test_response_truncated_when_exceeds_bound(self, index_with_episodes: tuple):
        """Response exceeding MAX_RESPONSE_CHARS is truncated with marker."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["long_response"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert len(result.response_text) <= detail.MAX_RESPONSE_CHARS + len(
            detail.TRUNCATION_MARKER
        )
        # Verify explicit truncation marker is present
        assert "[... truncated" in result.response_text

    def test_truncation_marker_includes_max_chars(self, index_with_episodes: tuple):
        """Truncation marker states the bound that was applied."""
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["long_response"]

        result = detail.show(project_dir, ep_id)

        assert result is not None
        assert f"{detail.MAX_RESPONSE_CHARS}" in result.response_text


class TestShowInvalidRef:
    """Tests for show() with invalid refs.

    Note: These tests create an empty index so that ref validation can proceed
    without hitting the "no index" error. The unit tests focus on ref validation
    behavior in isolation. The CLI tests verify that missing index takes
    precedence over malformed refs (exit 4 vs exit 3).
    """

    def test_malformed_ref_returns_none(self, project_dir: Path):
        """Malformed refs return None without raising."""
        # Create empty index so ref validation can be tested
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)
        conn.commit()
        conn.close()

        result = detail.show(project_dir, "not-a-valid-ref")
        assert result is None

    def test_empty_ref_returns_none(self, project_dir: Path):
        """Empty ref returns None."""
        # Create empty index
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)
        conn.commit()
        conn.close()

        result = detail.show(project_dir, "")
        assert result is None

    def test_ref_without_ep_separator_returns_none(self, project_dir: Path):
        """Ref missing :ep: separator returns None."""
        # Create empty index
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)
        conn.commit()
        conn.close()

        result = detail.show(project_dir, "session-123:episode:0")
        assert result is None

    def test_ref_with_non_numeric_index_returns_none(self, project_dir: Path):
        """Ref with non-numeric index returns None."""
        # Create empty index
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)
        conn.commit()
        conn.close()

        result = detail.show(project_dir, "session-123:ep:abc")
        assert result is None

    def test_validate_episode_id_rejects_non_numeric_index(self):
        """_validate_episode_id itself rejects a non-numeric index.

        test_ref_with_non_numeric_index_returns_none (above) can't prove
        *why* show() returns None: with an empty index, a ref rejected at
        format validation and a well-formed-but-absent ref both make show()
        return None -- one via the format check's early `return None`, the
        other via an empty-table DB lookup that simply finds nothing.
        Populating the fixture with real episodes doesn't resolve this
        either: show()'s DB query matches on the *entire* ref string
        (`WHERE episode_id = ?`, bound to the raw ref, not the extracted
        session_id), and a ref with a bogus, non-numeric index can never
        literally equal a real numeric-indexed episode_id -- so the lookup
        still misses either way, regardless of whether format validation
        ran. The only way to prove format validation itself does the
        rejecting -- not a downstream DB miss -- is to call it directly.
        """
        # Well-formed: accepted, session_id extracted.
        assert detail._validate_episode_id("session-123:ep:0") == "session-123"
        # Malformed: non-numeric index must be rejected at this layer.
        assert detail._validate_episode_id("session-123:ep:abc") is None

    def test_unknown_ref_returns_none(self, index_with_episodes: tuple):
        """Well-formed but non-existent ref returns None."""
        project_dir, _episode_ids = index_with_episodes
        # Format is valid but episode doesn't exist
        result = detail.show(project_dir, "nonexistent-session:ep:99999")
        assert result is None


class TestShowMissingIndex:
    """Tests for show() behavior with missing or corrupt index."""

    def test_missing_index_raises_error(self, project_dir: Path):
        """Missing index raises IndexNotFoundError."""
        from ssgrep.types import IndexNotFoundError

        # project_dir has no .ssgrep directory
        with pytest.raises(IndexNotFoundError):
            detail.show(project_dir, "session:ep:0")

    def test_corrupt_index_raises_error(self, project_dir: Path):
        """Corrupt index raises IndexNotReadyError."""
        from ssgrep.types import IndexNotReadyError

        # Create a corrupt db file
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        db_path.write_text("not a valid sqlite database")

        # Should raise IndexNotReadyError
        with pytest.raises(IndexNotReadyError):
            detail.show(project_dir, "session:ep:0")


class TestShowMutationTests:
    """Mutation tests to verify bounds are actually enforced."""

    def test_prompt_bound_is_enforced(self, index_with_episodes: tuple):
        """Removing the prompt bound causes test to fail (mutation test).

        This test verifies that the MAX_PROMPT_CHARS bound is actually
        being applied by confirming truncation happens for oversized text.
        """
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["long_prompt"]

        result = detail.show(project_dir, ep_id)
        assert result is not None

        # The prompt is 60k chars, bound is 50k
        # After truncation with marker, it should be bounded
        assert len(result.prompt_text) < 60000
        assert "[... truncated" in result.prompt_text

    def test_response_bound_is_enforced(self, index_with_episodes: tuple):
        """Removing the response bound causes test to fail (mutation test).

        This test verifies that the MAX_RESPONSE_CHARS bound is actually
        being applied by confirming truncation happens for oversized text.
        """
        project_dir, episode_ids = index_with_episodes
        ep_id = episode_ids["long_response"]

        result = detail.show(project_dir, ep_id)
        assert result is not None

        # The response is 200k chars, bound is 150k
        # After truncation, it should be bounded
        assert len(result.response_text) < 200000
        assert "[... truncated" in result.response_text


class TestShowGenerationalIndex:
    """Tests for show() on generational indices (generation >= 1).

    These tests verify the critical real-world workflow: search finds a ref,
    then show retrieves it. This must work on indices that have been pruned
    or rebuilt (generation >= 1).
    """

    @pytest.fixture
    def index_after_prune(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Build a generation-0 index, prune to create generation 1, return project info.

        Returns (project_dir, episode_id_of_untouched_session, episode_id_of_tombstoned_session)
        """
        index_dir = tmp_path / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)

        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)

        # Create two sessions
        session_id_keep = "session-keep-001"
        session_id_prune = "session-prune-001"
        test_ts = datetime(2026, 7, 25, 10, 30, 0)

        # Session 1: Will be kept (still exists on disk)
        keep_session = store.SessionFile(
            path=tmp_path / "keep_transcript.jsonl",
            session_id=session_id_keep,
            is_main=True,
            size=1000,
            mtime=1234567890.0,
        )
        store.insert_session(conn, keep_session)

        # Create episode in keep session
        ep_keep = Episode(
            episode_id=f"{session_id_keep}:ep:0",
            session_id=session_id_keep,
            prompt_text="Keep this",
            response_text="Keeping this content",
            title="Episode to keep",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=(),
            tool_names=(),
            is_subagent=False,
        )
        store.insert_episode(conn, ep_keep, ep_keep.prompt_text, ep_keep.response_text)
        chunks = chunker.chunk_episode(ep_keep)
        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)

        # Session 2: Will be tombstoned (disk file vanishes)
        prune_session = store.SessionFile(
            path=tmp_path / "prune_transcript.jsonl",
            session_id=session_id_prune,
            is_main=True,
            size=1000,
            mtime=1200000000.0,  # Old timestamp so it passes --older-than
        )
        store.insert_session(conn, prune_session)

        # Create episode in prune session
        ep_prune = Episode(
            episode_id=f"{session_id_prune}:ep:0",
            session_id=session_id_prune,
            prompt_text="Will be pruned",
            response_text="This will be pruned",
            title="Episode to prune",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=(),
            tool_names=(),
            is_subagent=False,
        )
        store.insert_episode(conn, ep_prune, ep_prune.prompt_text, ep_prune.response_text)
        chunks = chunker.chunk_episode(ep_prune)
        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)

        conn.commit()
        conn.close()

        # Mark the prune session as tombstoned by deleting the source file
        # The file was never created, so it's implicitly absent
        # Run prune with --older-than 0 to delete old sessions
        # This will advance the generation
        gen_store = store.GenerationalStore(index_dir)
        db_path_gen0 = gen_store.get_index_path()
        conn = store.sqlite3.connect(str(db_path_gen0))

        # Mark the prune session as absent (simulating disk deletion)
        # First update the session's source_status
        conn.execute(
            "UPDATE sessions SET source_status = 'absent' WHERE session_id = ?",
            (session_id_prune,),
        )
        # Also update its chunks and episodes
        conn.execute(
            "UPDATE chunks SET source_status = 'absent' " "WHERE session_id = ?",
            (session_id_prune,),
        )
        conn.execute(
            "UPDATE episodes SET source_status = 'absent' " "WHERE session_id = ?",
            (session_id_prune,),
        )
        conn.commit()

        # Call cleanup_orphaned_vectors which stages and commits a new generation
        store.cleanup_orphaned_vectors(conn, gen_store)

        return tmp_path, ep_keep.episode_id, ep_prune.episode_id

    def test_show_works_on_generation_1_index(self, index_after_prune: tuple):
        """show() retrieves episodes from generation >= 1 indices."""
        project_dir, ep_id_keep, _ep_id_prune = index_after_prune

        # Verify we're on a higher generation
        gen_store = store.GenerationalStore(project_dir / ".ssgrep")
        assert gen_store.current_generation > 0, "Test fixture should create generation > 0"

        # show() must work on the current generation
        result = detail.show(project_dir, ep_id_keep)

        assert result is not None
        assert result.episode_id == ep_id_keep
        assert result.prompt_text == "Keep this"
        assert result.response_text == "Keeping this content"

    def test_unknown_ref_still_returns_none_at_gen_1(self, index_after_prune: tuple):
        """Unknown ref returns None (not an index error) on generation >= 1."""
        project_dir, _ep_id_keep, _ep_id_prune = index_after_prune

        # Well-formed but non-existent ref must return None
        result = detail.show(project_dir, "nonexistent-session:ep:99999")
        assert result is None

    def test_malformed_ref_returns_none_at_gen_1(self, index_after_prune: tuple):
        """Malformed refs return None even on generation >= 1."""
        project_dir, _ep_id_keep, _ep_id_prune = index_after_prune

        result = detail.show(project_dir, "not-a-valid-ref")
        assert result is None


# ---------------------------------------------------------------------------
# Staleness wiring: show() must carry the same signal as search()/status()
# ---------------------------------------------------------------------------


class TestShowStalenessWiring:
    """Verify staleness is detected and reported in EpisodeDetail.

    Mirrors TestSearchStalenessWiring in test_search.py. show() is "the only
    command that returns untruncated transcript content" (EpisodeDetail
    docstring), so a user drilling into full episode detail from a stale
    index is at least as liable to be misled as one reading search results.
    Tests construct distinguishable worlds (fresh vs. stale) and assert
    exact staleness values. The mutation test proves the wiring is
    load-bearing.
    """

    def test_show_fresh_index_returns_stale_false_count_zero(
        self, index_with_episodes: tuple, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fresh index: stale=False, stale_count=0."""
        project_dir, episode_ids = index_with_episodes

        def mock_discover(project_dir, scope=None):
            return []

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        result = detail.show(project_dir, episode_ids["short"])
        assert result is not None
        assert result.stale is False
        assert result.stale_count == 0

    def test_show_stale_index_returns_exact_count(
        self, index_with_episodes: tuple, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stale index: stale=True with stale_count == exact number of changed files."""
        project_dir, episode_ids = index_with_episodes

        file_a = project_dir / "session-a.jsonl"
        file_a.write_text("x" * 100)

        db_path = project_dir / ".ssgrep" / "index.db"
        conn = sqlite3.connect(str(db_path))
        cursor = FileCursor(
            path=file_a, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
        )
        store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        result = detail.show(project_dir, episode_ids["short"])
        assert result is not None
        assert result.stale is True
        assert result.stale_count == 1

    def test_show_mutation_staleness_call_removed_produces_stale_false_count_zero(
        self, index_with_episodes: tuple, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation test: removing staleness call leaves stale=False, count=0.

        Proves the wiring is load-bearing by showing before/after differ.
        """
        project_dir, episode_ids = index_with_episodes

        file_a = project_dir / "session-a.jsonl"
        file_a.write_text("x" * 100)

        db_path = project_dir / ".ssgrep" / "index.db"
        conn = sqlite3.connect(str(db_path))
        cursor = FileCursor(
            path=file_a, size=100, mtime=1000.0, byte_offset=0, first_line_hash="old"
        )
        store.upsert_session_file(conn, cursor)
        conn.commit()
        conn.close()

        def mock_discover(project_dir, scope=None):
            return [
                discovery.SessionFile(
                    path=file_a, session_id="s-a", is_main=True, size=200, mtime=2000.0
                ),
            ]

        monkeypatch.setattr(discovery, "discover_sessions", mock_discover)

        # Baseline: staleness detected
        result_before = detail.show(project_dir, episode_ids["short"])
        assert result_before is not None
        assert result_before.stale is True
        assert result_before.stale_count == 1

        # Simulate mutation: replace staleness_summary with one that always returns unchanged
        def mock_staleness_summary_unchanged(project_dir, discovered_files=None):
            return staleness.StalenessReport(
                unchanged_count=1,
                appended_count=0,
                rewritten_count=0,
                vanished_count=0,
                stale_files=[],
            )

        monkeypatch.setattr(search, "staleness_summary", mock_staleness_summary_unchanged)

        # Mutated: staleness not detected
        result_after = detail.show(project_dir, episode_ids["short"])
        assert result_after is not None
        assert result_after.stale is False
        assert result_after.stale_count == 0

        # Verify before/after are different (proves the call matters)
        assert result_before.stale != result_after.stale
        assert result_before.stale_count != result_after.stale_count


class TestShowExactRoundTrip:
    """Tests verifying exact round-trip of varied, multi-chunk content.

    These tests construct episodes with VARIED, non-repeating text that spans
    multiple chunks (to exercise chunking and reconstruction), then assert that
    show() returns EXACTLY the original text — not a proxy, not a length check,
    but character-for-character identity. These tests would fail against the old
    chunk-concatenation logic (which duplicates overlap regions and loses
    indentation) and pass against the canonical-text-storage fix.
    """

    def test_exact_round_trip_varied_content_single_prompt_chunk(self, tmp_path: Path) -> None:
        """Exact round-trip for a small, single-chunk prompt+response."""
        index_dir = tmp_path / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)

        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)

        session_id = "test-session-roundtrip-001"
        test_ts = datetime(2026, 7, 25, 10, 30, 0)

        # Insert session
        test_session = store.SessionFile(
            path=Path("/home/test/transcript.jsonl"),
            session_id=session_id,
            is_main=True,
            size=1000,
            mtime=1234567890.0,
        )
        store.insert_session(conn, test_session)

        # Create episode with varied content (single chunk each)
        original_prompt = "What is the Python GIL?"
        original_response = (
            "The Global Interpreter Lock (GIL) is a mutex that prevents "
            "multiple threads from executing Python bytecode simultaneously."
        )

        ep = Episode(
            episode_id=f"{session_id}:ep:0",
            session_id=session_id,
            prompt_text=original_prompt,
            response_text=original_response,
            title="GIL explanation",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=("test.py",),
            tool_names=("Read",),
            is_subagent=False,
        )
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)

        # Insert chunks using the chunker
        chunks = chunker.chunk_episode(ep)
        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)

        conn.commit()
        conn.close()

        # Retrieve via show() and verify EXACT match
        result = detail.show(tmp_path, ep.episode_id)

        assert result is not None
        assert result.prompt_text == original_prompt, (
            f"Prompt mismatch:\n"
            f"Expected: {repr(original_prompt)}\n"
            f"Got:      {repr(result.prompt_text)}"
        )
        assert result.response_text == original_response, (
            f"Response mismatch:\n"
            f"Expected: {repr(original_response)}\n"
            f"Got:      {repr(result.response_text)}"
        )

    def test_exact_round_trip_varied_content_multi_chunk_prompt(self, tmp_path: Path) -> None:
        """Exact round-trip for a multi-chunk prompt with varied, non-repeating content.

        This prompt is large enough (>1200 chars) to span multiple chunks with overlap.
        The old chunk-concatenation logic would duplicate the overlap region and
        potentially lose leading/trailing whitespace, causing this test to fail.
        """
        index_dir = tmp_path / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)

        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)

        session_id = "test-session-multi-chunk-001"
        test_ts = datetime(2026, 7, 25, 10, 30, 0)

        # Insert session
        test_session = store.SessionFile(
            path=Path("/home/test/transcript.jsonl"),
            session_id=session_id,
            is_main=True,
            size=1000,
            mtime=1234567890.0,
        )
        store.insert_session(conn, test_session)

        # Create a varied, non-repeating prompt that spans multiple chunks
        # Using descriptive prose to ensure uniqueness, not repeated characters
        original_prompt = (
            "I'm working on a distributed cache system and need to understand "
            "consistency models. Specifically, I want to know the differences between "
            "strong consistency and eventual consistency, including their tradeoffs in "
            "terms of performance, latency, and availability. Please provide concrete "
            "examples using real-world systems like Dynamo, Cassandra, and Redis. "
            "Also explain how vector clocks and causal consistency fit into this picture. "
            "What are the best practices for choosing a consistency model for a given use case? "
            "Include code examples if possible."
        )

        original_response = (
            "Strong consistency ensures that all reads return the most "
            "recent write. This is enforced through mechanisms like quorum "
            "reads/writes or master-based replication. "
            "Examples: PostgreSQL (with synchronous replication), "
            "Google Spanner, and ZooKeeper. "
            "Eventual consistency allows temporary inconsistency but "
            "guarantees convergence. "
            "Examples: DynamoDB, Cassandra, and Riak. "
            "Vector clocks track causal relationships between events. "
            "Causal consistency is weaker than strong but stronger than "
            "eventual. Choose based on your consistency/availability/"
            "latency triangle needs."
        )

        ep = Episode(
            episode_id=f"{session_id}:ep:0",
            session_id=session_id,
            prompt_text=original_prompt,
            response_text=original_response,
            title="Distributed consistency models",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=("design.md",),
            tool_names=("Read", "Edit"),
            is_subagent=False,
        )
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)

        # Insert chunks using the chunker (will produce multiple chunks due to size)
        chunks = chunker.chunk_episode(ep)
        assert len(chunks) >= 2, (
            "Expected multiple chunks to test overlap handling; " f"got {len(chunks)} chunks"
        )
        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)

        conn.commit()
        conn.close()

        # Retrieve via show() and verify EXACT match
        result = detail.show(tmp_path, ep.episode_id)

        assert result is not None
        assert result.prompt_text == original_prompt, (
            f"Prompt mismatch after {len(chunks)} chunks:\n"
            f"Expected ({len(original_prompt)} chars): {repr(original_prompt[:100])}\n"
            f"Got ({len(result.prompt_text)} chars):      {repr(result.prompt_text[:100])}"
        )
        assert result.response_text == original_response, (
            f"Response mismatch after {len(chunks)} chunks:\n"
            f"Expected ({len(original_response)} chars): {repr(original_response[:100])}\n"
            f"Got ({len(result.response_text)} chars):      {repr(result.response_text[:100])}"
        )

    def test_exact_round_trip_preserves_leading_trailing_whitespace(self, tmp_path: Path) -> None:
        """Verify that leading and trailing whitespace is preserved exactly.

        The old chunk-reconstruction logic called .strip() on each chunk, which
        would lose leading/trailing whitespace. This test verifies the fix
        preserves it exactly.
        """
        index_dir = tmp_path / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)

        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)

        session_id = "test-session-whitespace-001"
        test_ts = datetime(2026, 7, 25, 10, 30, 0)

        # Insert session
        test_session = store.SessionFile(
            path=Path("/home/test/transcript.jsonl"),
            session_id=session_id,
            is_main=True,
            size=1000,
            mtime=1234567890.0,
        )
        store.insert_session(conn, test_session)

        # Create content with intentional leading/trailing whitespace
        original_prompt = "  \n  Leading and trailing whitespace matter.  \n  "
        original_response = (
            "\n\nMultiple leading newlines.\n\n"
            "Indented lines:\n"
            "    Line 1\n"
            "    Line 2\n"
            "\nTrailing newlines.\n\n"
        )

        ep = Episode(
            episode_id=f"{session_id}:ep:0",
            session_id=session_id,
            prompt_text=original_prompt,
            response_text=original_response,
            title="Whitespace test",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=(),
            tool_names=(),
            is_subagent=False,
        )
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)

        # Insert chunks
        chunks = chunker.chunk_episode(ep)
        for chunk in chunks:
            store.insert_chunk(conn, chunk, vec_row=None)

        conn.commit()
        conn.close()

        # Retrieve and verify whitespace is preserved EXACTLY
        result = detail.show(tmp_path, ep.episode_id)

        assert result is not None
        assert result.prompt_text == original_prompt, (
            f"Prompt whitespace not preserved:\n"
            f"Expected: {repr(original_prompt)}\n"
            f"Got:      {repr(result.prompt_text)}"
        )
        assert result.response_text == original_response, (
            f"Response whitespace not preserved:\n"
            f"Expected: {repr(original_response)}\n"
            f"Got:      {repr(result.response_text)}"
        )


# ============================================================================
# show()'s own schema gate: must reject stale indices BEFORE returning data
#
# The detail layer's _open_ready_index() checks schema_version and raises
# IndexNotReadyError if it doesn't match store.SCHEMA_VERSION. This test
# verifies that show() correctly gates on the schema version and that the
# exception carries the exact message the user needs to recover.
# ============================================================================


class TestShowSchemaGate:
    """Tests for show()'s schema version validation.

    show() must call _open_ready_index() which validates schema_version,
    raising IndexNotReadyError if there's a mismatch. This gates the
    entire operation before any data is returned.
    """

    def test_show_rejects_old_schema_version(self, tmp_path: Path) -> None:
        """show() raises IndexNotReadyError when schema_version is stale.

        This directly tests the detail._open_ready_index gate by creating a
        valid index, downgrading its schema_version, then verifying show()
        raises with the correct exception type and helpful message.
        """
        from ssgrep.types import IndexNotReadyError

        # Build a minimal valid index
        index_dir = tmp_path / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)

        session_id = "test-session-001"
        test_ts = datetime(2026, 7, 25, 10, 30, 0)

        # Insert a valid session and episode
        test_session = store.SessionFile(
            path=tmp_path / "transcript.jsonl",
            session_id=session_id,
            is_main=True,
            size=1000,
            mtime=1234567890.0,
        )
        store.insert_session(conn, test_session)

        ep = Episode(
            episode_id=f"{session_id}:ep:0",
            session_id=session_id,
            prompt_text="Test prompt",
            response_text="Test response",
            title="Test episode",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=(),
            tool_names=(),
            is_subagent=False,
        )
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)
        conn.commit()
        conn.close()

        # Now downgrade the schema_version to simulate a stale index
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = ?",
            ("2", "schema_version"),  # Old version, current is 3
        )
        conn.commit()
        conn.close()

        # show() must raise IndexNotReadyError, not return data
        with pytest.raises(IndexNotReadyError) as exc_info:
            detail.show(tmp_path, ep.episode_id)

        # Verify the exception message mentions the rebuild remedy
        error_msg = str(exc_info.value)
        assert "schema version" in error_msg.lower(), "Message must mention schema version"
        assert "rebuild" in error_msg.lower(), "Message must mention rebuild remedy"
        assert "--rebuild" in error_msg, "Message should mention --rebuild flag"
        # The actual exception must have the correct condition and command
        assert exc_info.value.condition == "corrupt_index"
        assert "rebuild" in exc_info.value.command.lower()

    def test_show_works_with_current_schema_version(self, tmp_path: Path) -> None:
        """show() succeeds when schema_version matches store.SCHEMA_VERSION.

        This is the positive control: verifies that when schema is current,
        show() works normally (not just that it doesn't raise).
        """
        # Build a minimal valid index with current schema
        index_dir = tmp_path / ".ssgrep"
        index_dir.mkdir(parents=True, mode=0o700)
        db_path = index_dir / "index.db"
        conn = store.init_db(db_path)  # init_db sets current schema

        session_id = "test-session-001"
        test_ts = datetime(2026, 7, 25, 10, 30, 0)

        # Insert a valid session and episode
        test_session = store.SessionFile(
            path=tmp_path / "transcript.jsonl",
            session_id=session_id,
            is_main=True,
            size=1000,
            mtime=1234567890.0,
        )
        store.insert_session(conn, test_session)

        ep = Episode(
            episode_id=f"{session_id}:ep:0",
            session_id=session_id,
            prompt_text="Current schema test",
            response_text="Works fine",
            title="Schema current",
            timestamp=test_ts,
            git_branch="main",
            cwd="/home/test",
            files_touched=("test.py",),
            tool_names=("Read",),
            is_subagent=False,
        )
        store.insert_episode(conn, ep, ep.prompt_text, ep.response_text)
        conn.commit()
        conn.close()

        # show() must NOT raise and must return the episode
        result = detail.show(tmp_path, ep.episode_id)

        assert result is not None, "show() must succeed with current schema"
        assert result.episode_id == ep.episode_id
        assert result.prompt_text == "Current schema test"
        assert result.response_text == "Works fine"
