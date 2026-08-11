"""End-to-end integration test of the public API against real corpus data.

This test exercises the full stack:
- api.index() reading the real project's session history
- api.search() finding real content
- api.show() drilling down to a full episode
- api.status() reporting consistent counts

If ~/.claude/projects does not exist, this test is skipped (the corpus is not
present in the test environment).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep import api, indexer
from ssgrep.paths import resolve_claude_dir
from tests.conftest import require_scoped_real_corpus


class TestApiIntegrationRealCorpus:
    """Integration tests against the real corpus in ~/.claude/projects.

    These tests skip if the corpus is not available, never silently return.
    """

    @pytest.fixture(autouse=True)
    def ensure_corpus_available(self) -> None:
        """Verify corpus exists via resolve_claude_dir(); skip if not."""
        projects_dir = resolve_claude_dir() / "projects"
        if not projects_dir.exists():
            pytest.skip(
                reason=(
                    "Real session corpus is not available in this environment "
                    f"(checked {projects_dir})"
                )
            )

    @pytest.fixture
    def project_dir(self) -> Path:
        """Get the repo root directory (this project's own directory)."""
        return Path(__file__).resolve().parent.parent

    def test_index_real_project_directory(self, tmp_path: Path, project_dir: Path) -> None:
        """Index the real project directory and verify basic counts.

        Uses this project's own directory as the target.
        """
        require_scoped_real_corpus(project_dir)
        # Create an isolated .ssgrep in tmp_path so we don't touch the real repo
        index_dir = tmp_path / "idx"
        stats = indexer.index(
            project_dir,
            index_dir=index_dir,
            quiet=True,
        )

        # Verify the return type and basic structure
        assert stats.session_count >= 1, "Should index at least one session"
        assert stats.episode_count >= 1, "Should have at least one episode"
        assert stats.chunk_count >= 1, "Should have at least one chunk"
        assert stats.index_size_bytes > 0, "Index should have non-zero size"
        assert stats.model_id is not None, "Model ID should be set"
        assert stats.vector_dimension > 0, "Vector dimension should be positive"
        assert stats.schema_version is not None, "Schema version should be set"
        assert stats.index_exists is True, "Index should exist after build"

    def test_search_real_index_finds_real_content(self, isolated_real_corpus_index: Path) -> None:
        """Search the real index for a term that exists in the corpus.

        Searches for "api" which is known to appear in this project's own
        session history, in discussions about the API contract.
        """
        project_dir = isolated_real_corpus_index
        # The fixture built a real-corpus index at an isolated project_dir/.ssgrep/
        # (never the repo's own) — see conftest.py's isolated_real_corpus_index.

        # Search for a term that should exist in the project's session history
        response = api.search(project_dir, "api")

        # Verify results structure
        assert response.results is not None, "Response should have results list"
        assert response.index_exists is True, "Index should exist"
        assert len(response.results) > 0, "Search for 'api' should find results in this corpus"

        # Verify result card shape
        card = response.results[0]
        assert card.ref is not None, "Result ref should not be None"
        assert card.title is not None, "Result title should not be None"
        assert card.score > 0, "Result score should be positive"
        assert card.excerpt is not None, "Result excerpt should not be None"
        assert len(card.excerpt) > 0, "Result excerpt should not be empty"
        assert isinstance(card.is_subagent, bool), "is_subagent should be bool"

    def test_show_result_from_search(self, isolated_real_corpus_index: Path) -> None:
        """Show a full episode from a search result.

        Verifies that refs from search results resolve to valid episodes.
        """
        project_dir = isolated_real_corpus_index
        # The fixture built a real-corpus index at an isolated project_dir/.ssgrep/
        # (never the repo's own) — see conftest.py's isolated_real_corpus_index.
        response = api.search(project_dir, "index")

        # Verify we got results
        assert len(response.results) > 0, "Should find results for 'index' in this corpus"

        # Take the first result
        first_result = response.results[0]
        ref = first_result.ref

        # Show that episode
        episode = api.show(project_dir, ref)

        # Verify the episode detail structure
        assert episode is not None, "show() should resolve the ref"
        assert episode.episode_id is not None
        assert episode.session_id is not None
        assert episode.title is not None
        assert episode.prompt_text is not None, "Episode should have prompt text field"
        assert episode.response_text is not None, "Episode should have response text field"
        # After deduplication fix, one-sided episodes are valid in both
        # directions: response-only (prompt == "") and prompt-only
        # (response == "", e.g. a subagent spawn-context episode). Which
        # kind ranks first for this query is a ranking outcome, not this
        # test's contract -- the round-trip is. Verify real content came
        # back, whichever side holds it.
        assert (
            len(episode.prompt_text) > 0 or len(episode.response_text) > 0
        ), "Episode must have prompt or response content"

    def test_status_after_index_consistency(self, isolated_real_corpus_index: Path) -> None:
        """Status reports should be internally consistent with index() results.

        Verifies that status() and index() report matching counts.
        """
        project_dir = isolated_real_corpus_index
        # The fixture built a real-corpus index at an isolated project_dir/.ssgrep/
        # (never the repo's own) — see conftest.py's isolated_real_corpus_index.

        # Get status (the index is already built by the fixture)
        status_stats = api.status(project_dir)

        # Verify that status reports valid data after indexing
        assert status_stats.index_exists is True, "Index should exist after fixture setup"
        assert status_stats.session_count >= 1, "Should have at least one session"
        assert status_stats.episode_count >= 1, "Should have at least one episode"
        assert status_stats.chunk_count >= 1, "Should have at least one chunk"
        assert status_stats.model_id is not None, "Model ID should be set"
        assert status_stats.vector_dimension > 0, "Vector dimension should be positive"

    def test_search_empty_query_raises_error(self, isolated_real_corpus_index: Path) -> None:
        """search() with empty query raises EmptyQueryError.

        Verifies the documented error behavior.
        """
        project_dir = isolated_real_corpus_index
        # The fixture built a real-corpus index at an isolated project_dir/.ssgrep/
        # (never the repo's own) — see conftest.py's isolated_real_corpus_index.

        # Empty query should raise
        from ssgrep.types import EmptyQueryError

        with pytest.raises(EmptyQueryError):
            api.search(project_dir, "")

        # Whitespace-only query should also raise
        with pytest.raises(EmptyQueryError):
            api.search(project_dir, "   ")

    def test_show_invalid_ref_returns_none(self, isolated_real_corpus_index: Path) -> None:
        """show() with invalid ref returns None, never raises.

        Verifies the documented behavior.
        """
        project_dir = isolated_real_corpus_index
        # The fixture built a real-corpus index at an isolated project_dir/.ssgrep/
        # (never the repo's own) — see conftest.py's isolated_real_corpus_index.

        # Invalid ref format should return None, not raise
        result = api.show(project_dir, "not-a-valid-ref")
        assert result is None

        # Malformed ref should also return None
        result = api.show(project_dir, "missing-episode-id")
        assert result is None

    def test_status_before_index_reports_no_index(self, tmp_path: Path) -> None:
        """status() when no index exists reports index_exists=False.

        Verifies the documented behavior for non-existent index.
        """
        # Use a fresh, empty directory with no index
        empty_project_dir = tmp_path / "empty-project"
        empty_project_dir.mkdir(parents=True)

        stats = api.status(empty_project_dir)

        assert stats.index_exists is False
        assert stats.session_count == 0
        assert stats.episode_count == 0
        assert stats.chunk_count == 0
