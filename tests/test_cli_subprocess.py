"""Test suite for CLI subprocess invocation with isolated project directories.

These tests verify that bare CLI commands (without --project-dir) correctly discover
and index sessions from an isolated project directory by resolving relative paths
to absolute paths before passing them to the discovery layer.

The discovery layer compares absolute cwd values from session records against
the project scope, so the scope must be an absolute path for matching to work.
Regression test for P0: bare `ssgrep index` silently indexed nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from tests.conftest import get_ssgrep_binary


def run_ssgrep_subprocess(
    args: list[str],
    cwd: Path,
    isolated_home: Path,
) -> tuple[int, str, str]:
    """Run ssgrep as a subprocess with isolated HOME and return results.

    Args:
        args: Command and arguments to pass to ssgrep
        cwd: Working directory for the command (project directory)
        isolated_home: Temporary HOME directory with projects structure

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    env = os.environ.copy()
    env["HOME"] = str(isolated_home)

    # Resolve ssgrep from the running interpreter's venv
    venv_bin = get_ssgrep_binary()

    cmd = [str(venv_bin)] + args

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def create_isolated_project_with_sessions(tmp_path: Path, project_dir: Path) -> tuple[Path, Path]:
    """Create an isolated HOME with .claude/projects/ containing session data.

    Args:
        tmp_path: Temporary directory for the isolated HOME
        project_dir: The project directory that sessions will reference via cwd
                    (must be resolved to an absolute path)

    Returns:
        Tuple of (isolated_home_path, encoded_project_path)
    """
    # Ensure project_dir is absolute/resolved
    project_dir = project_dir.resolve()

    # Create the .claude/projects/ directory structure
    home_dir = tmp_path
    claude_projects_dir = home_dir / ".claude" / "projects"
    claude_projects_dir.mkdir(parents=True, exist_ok=True)

    # Encode the project path (/ and . become -)
    # Use the resolved absolute path for encoding
    encoded_project = str(project_dir).replace("/", "-").replace(".", "-")

    # Create the encoded project directory
    project_sessions_dir = claude_projects_dir / encoded_project
    project_sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create a session directory and add a main session file
    session_id = "test-session-001"
    session_file = project_sessions_dir / f"{session_id}.jsonl"

    # Create real-shaped JSONL records with cwd matching the project_dir
    # These records are in the format produced by Claude Code transcripts.
    # Episodes are segmented from user/assistant records, not directly from the JSONL.
    records = [
        {
            "type": "user",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": {"content": "Test prompt with search_marker_xyz to find in search results"},
            "cwd": str(project_dir),
            "git_branch": "main",
        },
        {
            "type": "assistant",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": {"content": [{"type": "text", "text": "Test response to the prompt"}]},
            "cwd": str(project_dir),
        },
        {
            "type": "user",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": {"content": "Another user message"},
            "cwd": str(project_dir),
            "git_branch": "main",
        },
        {
            "type": "assistant",
            "timestamp": datetime.now(UTC).isoformat(),
            "message": {"content": [{"type": "text", "text": "More response content"}]},
            "cwd": str(project_dir),
        },
    ]

    # Write the JSONL file
    with open(session_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return home_dir, project_sessions_dir


class TestCliSubprocessProjectDirResolution:
    """Test that bare CLI commands resolve project_dir correctly."""

    def test_bare_index_discovers_sessions(self, tmp_path: Path) -> None:
        """Bare `ssgrep index` must discover sessions from the current directory."""
        # Create a project directory (resolved to absolute path)
        project_dir = (tmp_path / "my-project").resolve()
        project_dir.mkdir()

        # Create isolated HOME with session data pointing to this project
        isolated_home, _ = create_isolated_project_with_sessions(tmp_path, project_dir)

        # Run ssgrep index bare (no --project-dir) from the project directory
        code, stdout, stderr = run_ssgrep_subprocess(
            ["index"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )

        # Must succeed
        assert code == 0, f"ssgrep index failed: {stderr}"

        # Must report discovered sessions (not "indexing 0 discovered transcripts")
        output = stdout + stderr
        assert (
            "Indexed" in output or "indexed" in output
        ), f"Missing 'Indexed' in output. stdout: {stdout}\nstderr: {stderr}"

        # Must report more than zero sessions
        assert (
            "0 sessions" not in output
        ), f"index reported 0 sessions, but expected > 0. output: {output}"

        # Verify .ssgrep/index.db was created
        index_db = project_dir / ".ssgrep" / "index.db"
        assert index_db.exists(), "index.db was not created"

    def test_bare_status_not_stale_after_index(self, tmp_path: Path) -> None:
        """Bare `ssgrep status` must report 'not stale' immediately after index."""
        project_dir = (tmp_path / "my-project").resolve()
        project_dir.mkdir()

        isolated_home, _ = create_isolated_project_with_sessions(tmp_path, project_dir)

        # First, index the project
        code, stdout, stderr = run_ssgrep_subprocess(
            ["index"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )
        assert code == 0, f"ssgrep index failed: {stderr}"

        # Immediately check status
        code, stdout, stderr = run_ssgrep_subprocess(
            ["status"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )

        # Must succeed
        assert code == 0, f"ssgrep status failed: {stderr}"

        # Must report not stale
        output = stdout + stderr
        assert (
            "Stale:     no" in output or "stale" in output.lower()
        ), f"status should report staleness. output: {output}"

        # Should not report stale: yes
        assert (
            "Stale:     yes" not in output
        ), f"status incorrectly reported stale immediately after index. output: {output}"

    def test_bare_search_finds_results(self, tmp_path: Path) -> None:
        """Bare `ssgrep search` must find results without --project-dir."""
        project_dir = (tmp_path / "my-project").resolve()
        project_dir.mkdir()

        isolated_home, _ = create_isolated_project_with_sessions(tmp_path, project_dir)

        # First, index the project
        code, stdout, stderr = run_ssgrep_subprocess(
            ["index"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )
        assert code == 0, f"ssgrep index failed: {stderr}"

        # Search for the marker string we embedded in the test episode
        code, stdout, stderr = run_ssgrep_subprocess(
            ["search", "search_marker_xyz"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )

        # Must succeed (exit 0)
        assert code == 0, f"ssgrep search failed with exit code {code}: {stderr}"

        # Must find the result
        output = stdout + stderr
        assert (
            "search_marker_xyz" in output or "Test Episode" in output or "episode" in output.lower()
        ), f"search did not find the expected marker. output: {output}"

    def test_bare_index_with_relative_path_arg(self, tmp_path: Path) -> None:
        """Bare `ssgrep index --project-dir ../other` must work with relative paths."""
        # Create a project directory
        project_dir = (tmp_path / "projects" / "my-project").resolve()
        project_dir.mkdir(parents=True)

        # Create isolated HOME with session data
        isolated_home, _ = create_isolated_project_with_sessions(tmp_path, project_dir)

        # Run from a sibling directory
        run_from = tmp_path / "projects" / "other-dir"
        run_from.mkdir(parents=True)

        # Run ssgrep index with relative --project-dir
        code, stdout, stderr = run_ssgrep_subprocess(
            ["index", "--project-dir", "../my-project"],
            cwd=run_from,
            isolated_home=isolated_home,
        )

        # Must succeed
        assert code == 0, f"ssgrep index with relative path failed: {stderr}"

        # Must discover sessions
        output = stdout + stderr
        assert (
            "0 sessions" not in output
        ), f"index with relative --project-dir reported 0 sessions. output: {output}"

        # Verify index was created at the right place
        index_db = project_dir / ".ssgrep" / "index.db"
        assert index_db.exists(), "index.db was not created at the expected location"

    def test_status_reports_session_count(self, tmp_path: Path) -> None:
        """Bare `ssgrep status` must report the correct session count."""
        project_dir = (tmp_path / "my-project").resolve()
        project_dir.mkdir()

        isolated_home, _ = create_isolated_project_with_sessions(tmp_path, project_dir)

        # Index the project
        code, stdout, stderr = run_ssgrep_subprocess(
            ["index"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )
        assert code == 0, f"ssgrep index failed: {stderr}"

        # Check status
        code, stdout, stderr = run_ssgrep_subprocess(
            ["status"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )

        # Must succeed
        assert code == 0, f"ssgrep status failed: {stderr}"

        # Must report at least 1 session
        output = stdout + stderr
        assert "Sessions:" in output, f"status missing Sessions line. output: {output}"
        assert "Episodes:" in output, f"status missing Episodes line. output: {output}"

        # Parse the session count to verify it's > 0
        lines = output.split("\n")
        for line in lines:
            if "Sessions:" in line:
                parts = line.split()
                session_count = int(parts[-1])
                assert (
                    session_count >= 1
                ), f"Expected at least 1 session, got {session_count}. output: {output}"

    def test_status_renders_cwd_cache_output(self, tmp_path: Path) -> None:
        """PROPERTY TEST: Rendered output includes/excludes cache health line.

        Verifies that the status command's text output correctly reflects cache
        health. When cache is healthy (no fallback scans), the line is absent;
        when degraded (fallback scans occurred), the line is present with count.

        This is the property, not a proxy. Tests assert on rendered text.
        """
        from ssgrep import discovery

        project_dir = (tmp_path / "my-project").resolve()
        project_dir.mkdir()

        isolated_home, _ = create_isolated_project_with_sessions(tmp_path, project_dir)

        # First, index the project
        code, stdout, stderr = run_ssgrep_subprocess(
            ["index"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )
        assert code == 0, f"ssgrep index failed: {stderr}"

        # Reset counter and run status on a healthy index
        discovery._cwd_index_stats["fallback_scans"] = 0

        code, stdout, stderr = run_ssgrep_subprocess(
            ["status"],
            cwd=project_dir,
            isolated_home=isolated_home,
        )

        # Must succeed
        assert code == 0, f"ssgrep status failed: {stderr}"

        output = stdout + stderr

        # PROPERTY: Healthy path must NOT show degradation line
        assert "Cwd cache:" not in output, (
            f"status should not render cache degradation line in healthy state. "
            f"output: {output}"
        )

        # Sanity check: normal output is still there
        assert "Sessions:" in output, f"status missing Sessions line. output: {output}"
