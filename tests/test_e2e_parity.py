"""Cross-surface parity tests: CLI vs API vs MCP.

Verifies that CLI, API, and MCP return identical results for equivalent inputs,
tested on real transcripts from a clean state with no pre-existing index.

The test focuses on:
1. Index → search → show → status on real transcripts from clean state
2. CLI and MCP return identical results for the same query
3. Refs returned by one surface resolve through the other's show
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from ssgrep import api, indexer
from ssgrep.paths import resolve_claude_dir
from tests.conftest import get_ssgrep_binary, require_scoped_real_corpus


class TestE2EParityCleanState:
    """Integration tests on real corpus from clean state."""

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

    def test_index_from_clean_state(self, tmp_path: Path, project_dir: Path) -> None:
        """Verify index builds from clean state with no pre-existing index.

        Indexes the real project and verifies basic counts. Uses an explicit
        index_dir under tmp_path (never project_dir/.ssgrep) so "clean state"
        means tmp_path starting empty, not the repo's real index — api.index()
        has no index_dir override, so this calls indexer.index() directly, the
        same substitution test_api_integration.py's test_index_real_project_directory
        already uses. See tests/conftest.py's guard_real_index_unchanged.
        """
        require_scoped_real_corpus(project_dir)
        stats = indexer.index(project_dir, index_dir=tmp_path / "idx", quiet=True)

        # Verify basic structure
        assert stats.session_count >= 1, "Should index at least one session"
        assert stats.episode_count >= 1, "Should have at least one episode"
        assert stats.chunk_count >= 1, "Should have at least one chunk"
        assert stats.index_size_bytes > 0, "Index should have non-zero size"
        assert stats.model_id is not None, "Model ID should be set"
        assert stats.vector_dimension > 0, "Vector dimension should be positive"
        assert stats.index_exists is True, "Index should exist after build"

    def test_cli_and_mcp_search_parity(self, isolated_real_corpus_index: Path) -> None:
        """CLI and MCP search return identical results for the same query.

        Both surfaces call ssgrep.api, but shape output independently, so
        they can and do diverge. This test verifies they return identical
        result sets with identical order.
        """
        project_dir = isolated_real_corpus_index
        # The fixture already built a real-corpus index at project_dir/.ssgrep,
        # where project_dir is an isolated directory that is never the repo
        # itself. Neither the CLI's --project-dir nor MCP's cwd-based resolution
        # has an index-location override, so both surfaces read whatever is
        # physically at project_dir/.ssgrep — pointing every surface (API, CLI,
        # MCP) at this same isolated project_dir, rather than at the repo, is
        # what keeps all three off the real index. See tests/conftest.py's
        # isolated_real_corpus_index docstring.

        # Use a query that should return results from this repo's corpus
        query = "index"
        limit = 10

        # Search via API (what both CLI and MCP call)
        api_response = api.search(project_dir, query, limit=limit)

        # Search via CLI subprocess
        cli_stdout, cli_stderr, cli_code = self._run_cli_search(project_dir, query, limit)

        # Search via MCP
        mcp_result = self._run_mcp_search(project_dir, query, limit)

        # Verify CLI search succeeded
        assert cli_code == 0, f"CLI search failed with code {cli_code}: {cli_stderr}"

        # Parse CLI JSON response
        cli_response_dict = json.loads(cli_stdout)
        assert cli_response_dict["ok"] is True, "CLI should return ok=True"

        # Verify result set identity: same refs in same order
        api_refs = [card.ref for card in api_response.results]
        cli_refs = [card["ref"] for card in cli_response_dict["data"]["results"]]
        mcp_refs = [card["ref"] for card in mcp_result["results"]]

        assert api_refs == cli_refs, f"CLI refs diverged from API. API: {api_refs}, CLI: {cli_refs}"
        assert api_refs == mcp_refs, f"MCP refs diverged from API. API: {api_refs}, MCP: {mcp_refs}"

        # Verify omitted counts
        api_omitted = api_response.omitted_count
        cli_omitted = cli_response_dict["data"]["omitted_count"]
        mcp_omitted = mcp_result["omitted_count"]

        assert (
            api_omitted == cli_omitted
        ), f"CLI omitted_count diverged. API: {api_omitted}, CLI: {cli_omitted}"
        assert (
            api_omitted == mcp_omitted
        ), f"MCP omitted_count diverged. API: {api_omitted}, MCP: {mcp_omitted}"

    def test_cli_mcp_show_ref_parity(self, isolated_real_corpus_index: Path) -> None:
        """Refs from one surface resolve through the other's show.

        This catches a real defect: search handed out refs that show
        refused on any index at generation >= 1.
        """
        project_dir = isolated_real_corpus_index
        # The fixture already built a real-corpus index at project_dir/.ssgrep,
        # where project_dir is an isolated directory, never the repo itself.

        # Search to get a ref
        response = api.search(project_dir, "index", limit=5)
        if not response.results:
            pytest.skip("No search results to test show parity")

        first_ref = response.results[0].ref

        # Show via API
        api_detail = api.show(project_dir, first_ref)
        assert api_detail is not None, f"API show resolved ref {first_ref}"

        # Show via CLI
        cli_detail = self._run_cli_show(project_dir, first_ref)
        assert cli_detail is not None, f"CLI show resolved ref {first_ref}"

        # Show via MCP
        mcp_detail = self._run_mcp_show(project_dir, first_ref)
        assert mcp_detail is not None, f"MCP show resolved ref {first_ref}"

        # Verify all three agree on episode_id (proof the ref resolved to same episode)
        assert (
            api_detail.episode_id == cli_detail["episode_id"]
        ), "CLI show returned different episode"
        assert (
            api_detail.episode_id == mcp_detail["episode_id"]
        ), "MCP show returned different episode"

    def _run_cli_search(self, project_dir: Path, query: str, limit: int) -> tuple[str, str, int]:
        """Run ssgrep search via CLI subprocess.

        Returns: (stdout, stderr, exit_code)
        """
        ssgrep_bin = get_ssgrep_binary()

        with tempfile.TemporaryDirectory(prefix="ssgrep-cli-test-") as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir
            Path(tmpdir, ".claude", "projects").mkdir(parents=True, exist_ok=True)

            cmd = [
                str(ssgrep_bin),
                "search",
                query,
                "--limit",
                str(limit),
                "--project-dir",
                str(project_dir),
                "--json",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            return result.stdout, result.stderr, result.returncode

    def _run_cli_show(self, project_dir: Path, ref: str) -> dict | None:
        """Run ssgrep show via CLI and return parsed JSON response."""
        ssgrep_bin = get_ssgrep_binary()

        with tempfile.TemporaryDirectory(prefix="ssgrep-cli-show-") as tmpdir:
            env = os.environ.copy()
            env["HOME"] = tmpdir
            Path(tmpdir, ".claude", "projects").mkdir(parents=True, exist_ok=True)

            cmd = [
                str(ssgrep_bin),
                "show",
                ref,
                "--project-dir",
                str(project_dir),
                "--json",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                return None

            response_dict = json.loads(result.stdout)
            if response_dict.get("ok"):
                # Extract detail from the wrapped response
                return response_dict.get("data")
            return None

    def _run_mcp_search(self, project_dir: Path, query: str, limit: int) -> dict:
        """Call MCP search_sessions directly.

        Returns the search result dict.
        """
        import ssgrep.mcp_server

        # Temporarily change cwd for the MCP server's project resolution
        original_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            result = ssgrep.mcp_server.search_sessions(query, limit=limit)
            return result
        finally:
            os.chdir(original_cwd)

    def _run_mcp_show(self, project_dir: Path, ref: str) -> dict | None:
        """Call MCP show_session directly.

        Returns the episode detail dict or None if not found.
        """
        import ssgrep.mcp_server

        # Temporarily change cwd for the MCP server's project resolution
        original_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            result = ssgrep.mcp_server.show_session(ref)
            # MCP returns {"error": "..."} on error, or the detail dict
            if "error" in result:
                return None
            return result
        finally:
            os.chdir(original_cwd)
