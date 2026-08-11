"""Error-path UX tests — assert on actionable messages, not just exit codes.

Each degenerate path must produce an actionable message with a
suggested next command. These tests run the CLI via subprocess to exercise
the real error paths end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import get_ssgrep_binary

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(
    args: list[str],
    cwd: str | Path | None = None,
    env: dict | None = None,
    isolated_home: bool = True,
) -> subprocess.CompletedProcess:
    """Run ssgrep with optional custom environment.

    Args:
        args: Command arguments
        cwd: Working directory
        env: Additional environment variables to set (merged with os.environ)
        isolated_home: If True and HOME not in env, create a temp HOME with
                      .claude/projects to avoid touching the real ~/.claude/settings.json
                      and to ensure indexer.py's first-run guard is satisfied

    Returns:
        CompletedProcess result
    """
    ssgrep_bin = get_ssgrep_binary()
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # If isolated_home and HOME wasn't explicitly set, create a temp home with .claude/projects
    if isolated_home and (env is None or "HOME" not in env):
        fake_home = Path(tempfile.mkdtemp(prefix="ssgrep-test-home-"))
        # Create .claude/projects directory structure so indexer.py's first-run guard passes
        (fake_home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)
        run_env["HOME"] = str(fake_home)

    return subprocess.run(
        [str(ssgrep_bin)] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=run_env,
        timeout=30,
    )


class TestMissingIndex:
    """When no index exists, commands must suggest 'ssgrep index' with actionable text."""

    def test_search_no_index_shows_actionable_message(self):
        """Search with no index must mention 'ssgrep index' and identify the problem."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            result = _run(["search", "test query"], cwd=tmpdir)
            combined = result.stdout + result.stderr
            # Must mention the command to run
            assert "ssgrep index" in combined, (
                f"Expected 'ssgrep index' in message, got:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )
            # Must mention that no index was found
            assert "no index" in combined.lower() or "not found" in combined.lower(), (
                f"Expected mention of missing index, got:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )

    def test_show_no_index_shows_actionable_message(self):
        """Show with no index must suggest 'ssgrep index'."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            # Use a well-formed ref so validation passes and we hit the index check
            result = _run(["show", "session:ep:0"], cwd=tmpdir)
            combined = result.stdout + result.stderr
            assert "ssgrep index" in combined, (
                f"Expected 'ssgrep index' in message, got:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )
            assert (
                "no index" in combined.lower()
            ), f"Expected mention of missing index, got:\n{combined}"

    def test_prune_no_index_shows_actionable_message(self):
        """Prune with no index must suggest 'ssgrep index'."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            result = _run(["prune", "--yes"], cwd=tmpdir)
            combined = result.stdout + result.stderr
            assert "ssgrep index" in combined, (
                f"Expected 'ssgrep index' in message, got:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )

    def test_search_no_index_json_mode(self):
        """JSON mode search must have structured error with condition code.

        Error is now flat in the document (no nested data field).
        """
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            result = _run(["search", "test", "--json"], cwd=tmpdir)
            # Single JSON document with error at top level
            lines = result.stdout.strip().split("\n")
            assert len(lines) == 1, f"Expected single JSON document, got {len(lines)}"
            try:
                doc = json.loads(lines[0])
            except (json.JSONDecodeError, IndexError):
                pytest.fail(f"JSON in stdout is not valid JSON: {result.stdout!r}")
            assert isinstance(doc, dict), f"Expected JSON object, got {type(doc)}"
            # Error is now at top level, not nested in data
            assert doc.get("ok") is False, f"Expected ok=False, got: {doc!r}"
            assert (
                doc.get("condition") == "missing_index"
            ), f"Expected condition='missing_index', got: {doc.get('condition')}"
            assert "ssgrep index" in doc.get(
                "command", ""
            ), f"Expected command to mention 'ssgrep index', got: {doc.get('command')}"


class TestEmptyScope:
    """When the corpus directory is missing, status must be informative."""

    def test_status_missing_projects_dir(self):
        """status on a project with no ~/.claude/projects must not crash."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noscope-") as tmpdir:
            # Override HOME so ~/.claude/projects doesn't exist
            fake_home = Path(tmpdir) / "fake_home"
            fake_home.mkdir()
            result = _run(["status"], cwd=tmpdir, env={"HOME": str(fake_home)})
            # Should succeed (exit 0) with a message about no index or 0 sessions
            combined = result.stdout + result.stderr
            # Either says "no index" or reports 0 — both acceptable
            assert result.returncode == 0 or "index" in combined.lower(), (
                f"Expected graceful handling, got:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )

    def test_index_missing_projects_dir_exits_4(self):
        """index exits 4 when ~/.claude/projects doesn't exist."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noscope-") as tmpdir:
            fake_home = Path(tmpdir) / "fake_home"
            fake_home.mkdir()
            result = _run(["index"], cwd=tmpdir, env={"HOME": str(fake_home)})
            assert result.returncode == 4, (
                f"index with missing ~/.claude/projects should exit 4, got {result.returncode}:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )

    def test_index_missing_projects_dir_shows_actionable_message(self):
        """index with missing ~/.claude/projects must show actionable message."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noscope-") as tmpdir:
            fake_home = Path(tmpdir) / "fake_home"
            fake_home.mkdir()
            result = _run(["index"], cwd=tmpdir, env={"HOME": str(fake_home)})
            combined = result.stdout + result.stderr
            assert ".claude" in combined or "claude" in combined.lower(), (
                f"Message must mention .claude directory, got:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )

    def test_index_empty_projects_dir_succeeds(self):
        """index succeeds with 0 sessions when ~/.claude/projects exists but is empty."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noscope-") as tmpdir:
            fake_home = Path(tmpdir) / "fake_home"
            fake_home.mkdir()
            # Create the projects directory but leave it empty
            (fake_home / ".claude" / "projects").mkdir(parents=True)
            result = _run(["index"], cwd=tmpdir, env={"HOME": str(fake_home)})
            assert result.returncode == 0, (
                f"index with empty ~/.claude/projects should succeed, got {result.returncode}:\n"
                f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
            )


class TestCorruptIndex:
    """When the index is corrupt, commands must provide actionable recovery steps."""

    def test_search_corrupt_index_shows_rebuild_suggestion(self):
        """Corrupt index in search must suggest 'ssgrep index --rebuild'."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-corrupt-") as tmpdir:
            project = Path(tmpdir)
            # Create a corrupt index
            ssgrep_dir = project / ".ssgrep"
            ssgrep_dir.mkdir()
            # Write garbage as the SQLite file
            (ssgrep_dir / "index.db").write_bytes(b"not-a-sqlite-db")
            (ssgrep_dir / "index.db-shm").write_bytes(b"")
            (ssgrep_dir / "index.db-wal").write_bytes(b"")

            result = _run(["search", "test"], cwd=tmpdir)
            combined = result.stdout + result.stderr
            # Must not show a Python traceback
            assert (
                "Traceback" not in combined
            ), f"Corrupt index must not produce a traceback:\n{combined}"
            # Must mention the rebuild command
            assert (
                "index --rebuild" in combined or "rebuild" in combined.lower()
            ), f"Expected rebuild suggestion for corrupt index, got:\n{combined}"
            # Must identify the problem
            assert (
                "corrupt" in combined.lower()
            ), f"Expected mention of corruption, got:\n{combined}"

    def test_show_corrupt_index_shows_rebuild_suggestion(self):
        """Corrupt index in show must suggest 'ssgrep index --rebuild'."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-corrupt-") as tmpdir:
            project = Path(tmpdir)
            ssgrep_dir = project / ".ssgrep"
            ssgrep_dir.mkdir()
            (ssgrep_dir / "index.db").write_bytes(b"corrupt")

            # Use a well-formed ref so validation passes and we hit the corrupt index check
            result = _run(["show", "session:ep:0"], cwd=tmpdir)
            combined = result.stdout + result.stderr
            assert (
                "Traceback" not in combined
            ), f"Corrupt index must not produce a traceback:\n{combined}"
            assert (
                "index --rebuild" in combined or "rebuild" in combined.lower()
            ), f"Expected rebuild suggestion, got:\n{combined}"


class TestEmptyQuery:
    """Empty query must produce an actionable message."""

    def test_search_empty_query_shows_message(self):
        """Empty query must mention that query is required."""
        result = _run(["search", ""])
        combined = result.stdout + result.stderr
        # Should mention query is required or empty
        assert (
            "query" in combined.lower() or "empty" in combined.lower()
        ), f"Expected mention of query requirement, got:\n{combined}"
        assert len(combined.strip()) > 0, "Empty query should produce a message"


class TestJsonModeErrorPaths:
    """JSON mode error paths must produce parseable JSON with structured error info."""

    def test_search_no_index_json_has_structured_error(self):
        """JSON mode with no index must have structured error with condition."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-json-err-") as tmpdir:
            result = _run(["search", "test", "--json"], cwd=tmpdir)
            # JSON envelope contains error at top level (flat structure)
            lines = result.stdout.strip().split("\n")
            try:
                doc = json.loads(lines[0])
                assert isinstance(doc, dict), f"Expected JSON object, got {type(doc)}"
            except (json.JSONDecodeError, IndexError):
                pytest.fail(f"JSON error in stdout is not valid JSON: {result.stdout!r}")
            # Error is at top level, not nested
            # Verify structured error with condition code
            assert doc.get("ok") is False, "Expected ok=False in error"
            assert (
                doc.get("condition") == "missing_index"
            ), f"Expected condition='missing_index', got {doc.get('condition')}"
            # Guard: ok must NOT be true when an error condition is present
            assert not (doc.get("ok") is True and doc.get("condition")), (
                f"DEFECT: ok=true with condition={doc.get('condition')}. "
                "The outer envelope must not claim success when wrapping a failure."
            )

    def test_status_no_index_json_is_valid(self):
        """JSON mode status with no index must return valid JSON."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-json-err-") as tmpdir:
            result = _run(["status", "--json"], cwd=tmpdir)
            try:
                doc = json.loads(result.stdout)
                assert isinstance(doc, dict), f"Expected JSON object, got {type(doc)}"
            except json.JSONDecodeError:
                pytest.fail(f"JSON mode stdout is not valid JSON: {result.stdout!r}")
            # Verify the response is not empty
            assert len(doc) > 0, "Status JSON response should not be empty"

    def test_corrupt_index_json_has_condition_code(self):
        """Corrupt index error must have condition='corrupt_index' in JSON."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-corrupt-json-") as tmpdir:
            project = Path(tmpdir)
            ssgrep_dir = project / ".ssgrep"
            ssgrep_dir.mkdir()
            (ssgrep_dir / "index.db").write_bytes(b"corrupt data")

            result = _run(["search", "test", "--json"], cwd=tmpdir)
            lines = result.stdout.strip().split("\n")
            try:
                doc = json.loads(lines[0])
            except (json.JSONDecodeError, IndexError):
                pytest.fail(f"JSON error in stdout is not valid JSON: {result.stdout!r}")
            # Error is at top level, not nested
            assert (
                doc.get("condition") == "corrupt_index"
            ), f"Expected condition='corrupt_index', got {doc.get('condition')}"
            # Guard: ok must NOT be true when an error condition is present
            assert not (doc.get("ok") is True and doc.get("condition")), (
                f"DEFECT: ok=true with condition={doc.get('condition')}. "
                "The outer envelope must not claim success when wrapping a failure."
            )

    def test_empty_query_json_has_condition_code(self):
        """Empty query error must have condition='empty_query' in JSON."""
        result = _run(["search", "", "--json"])
        lines = result.stdout.strip().split("\n")
        try:
            doc = json.loads(lines[0])
        except (json.JSONDecodeError, IndexError):
            pytest.fail(f"JSON error in stdout is not valid JSON: {result.stdout!r}")
        # Error is at top level, not nested
        assert (
            doc.get("condition") == "empty_query"
        ), f"Expected condition='empty_query', got {doc.get('condition')}"
        # Guard: ok must NOT be true when an error condition is present
        assert not (doc.get("ok") is True and doc.get("condition")), (
            f"DEFECT: ok=true with condition={doc.get('condition')}. "
            f"The outer envelope must not claim success when wrapping a failure."
        )


class TestShowJsonEnvelope:
    """Show command JSON envelope and exit code contract tests."""

    def test_show_missing_index_json_exit_code_matches_plain(self):
        """Show missing index must return same exit code in JSON and plain."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            # Plain mode
            plain_result = _run(["show", "session:ep:0"], cwd=tmpdir)
            # JSON mode
            json_result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            # Exit codes must match
            assert json_result.returncode == plain_result.returncode, (
                f"Exit codes differ: JSON={json_result.returncode}, "
                f"plain={plain_result.returncode}"
            )
            assert json_result.returncode == 4, "Missing index should exit 4"

    def test_show_missing_index_json_has_flat_envelope(self):
        """Show missing index in JSON must have flat envelope with ok:false."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            lines = result.stdout.strip().split("\n")
            assert len(lines) == 1, f"Expected single JSON line, got {len(lines)}"
            try:
                doc = json.loads(lines[0])
            except json.JSONDecodeError:
                pytest.fail(f"JSON not valid: {result.stdout!r}")
            # Flat envelope: ok at top level (not nested)
            assert doc.get("ok") is False, f"Expected ok:false, got {doc}"
            assert doc.get("condition") == "missing_index"
            assert "ssgrep index" in doc.get("command", "")

    def test_show_corrupt_index_json_exit_code_matches_plain(self):
        """Show corrupt index must return same exit code in JSON and plain."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-corrupt-") as tmpdir:
            project = Path(tmpdir)
            ssgrep_dir = project / ".ssgrep"
            ssgrep_dir.mkdir()
            (ssgrep_dir / "index.db").write_bytes(b"corrupt")
            # Plain mode
            plain_result = _run(["show", "session:ep:0"], cwd=tmpdir)
            # JSON mode
            json_result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            # Exit codes must match
            assert json_result.returncode == plain_result.returncode, (
                f"Exit codes differ: JSON={json_result.returncode}, "
                f"plain={plain_result.returncode}"
            )
            assert json_result.returncode == 4, "Corrupt index should exit 4"

    def test_show_corrupt_index_json_has_flat_envelope(self):
        """Show corrupt index in JSON must have flat envelope with ok:false."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-corrupt-") as tmpdir:
            project = Path(tmpdir)
            ssgrep_dir = project / ".ssgrep"
            ssgrep_dir.mkdir()
            (ssgrep_dir / "index.db").write_bytes(b"corrupt")
            result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            lines = result.stdout.strip().split("\n")
            assert len(lines) == 1, f"Expected single JSON line, got {len(lines)}"
            try:
                doc = json.loads(lines[0])
            except json.JSONDecodeError:
                pytest.fail(f"JSON not valid: {result.stdout!r}")
            # Flat envelope: ok at top level
            assert doc.get("ok") is False, f"Expected ok:false, got {doc}"
            assert doc.get("condition") in ("corrupt_index", "index_not_ready")
            assert "rebuild" in doc.get("message", "").lower()

    def test_show_unknown_ref_json_exit_code_matches_plain(self):
        """Show unknown ref must return same exit code in JSON and plain (exit 3)."""
        # First create an empty index
        with tempfile.TemporaryDirectory(prefix="ssgrep-empty-") as tmpdir:
            # Create empty index using _run to ensure isolated HOME with .claude/projects
            index_result = _run(["index", "--quiet"], cwd=tmpdir)
            assert index_result.returncode == 0, (
                f"Index setup failed with exit {index_result.returncode}\n"
                f"stdout: {index_result.stdout}\n"
                f"stderr: {index_result.stderr}"
            )
            # Plain mode with unknown ref
            plain_result = _run(["show", "unknown:ep:0"], cwd=tmpdir)
            # JSON mode with unknown ref
            json_result = _run(["show", "unknown:ep:0", "--json"], cwd=tmpdir)
            # Exit codes must match and be 3 (NO_MATCHING_DATA)
            assert json_result.returncode == plain_result.returncode, (
                f"Exit codes differ: JSON={json_result.returncode}, "
                f"plain={plain_result.returncode}"
            )
            assert json_result.returncode == 3, "Unknown ref should exit 3"

    def test_show_unknown_ref_json_has_flat_envelope(self):
        """Show unknown ref in JSON must have flat envelope with ok:false."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-empty-") as tmpdir:
            # Create empty index using _run to ensure isolated HOME with .claude/projects
            index_result = _run(["index", "--quiet"], cwd=tmpdir)
            assert index_result.returncode == 0, (
                f"Index setup failed with exit {index_result.returncode}\n"
                f"stdout: {index_result.stdout}\n"
                f"stderr: {index_result.stderr}"
            )
            result = _run(["show", "unknown:ep:0", "--json"], cwd=tmpdir)
            lines = result.stdout.strip().split("\n")
            assert len(lines) == 1, f"Expected single JSON line, got {len(lines)}"
            try:
                doc = json.loads(lines[0])
            except json.JSONDecodeError:
                pytest.fail(f"JSON not valid: {result.stdout!r}")
            # Flat envelope: ok at top level
            assert doc.get("ok") is False, f"Expected ok:false, got {doc}"
            assert doc.get("condition") == "unknown_ref"
            assert "not found" in doc.get("message", "").lower()

    def test_show_distinguishes_missing_index_from_unknown_ref_json(self):
        """Show must distinguish missing_index from unknown_ref in exit codes."""
        with tempfile.TemporaryDirectory(prefix="ssgrep-empty-") as tmpdir:
            # Create empty index using _run to ensure isolated HOME with .claude/projects
            index_result = _run(["index", "--quiet"], cwd=tmpdir)
            assert index_result.returncode == 0, (
                f"Index setup failed with exit {index_result.returncode}\n"
                f"stdout: {index_result.stdout}\n"
                f"stderr: {index_result.stderr}"
            )
            # Unknown ref: exit 3
            unknown_result = _run(["show", "unknown:ep:0", "--json"], cwd=tmpdir)
            assert unknown_result.returncode == 3

        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            # No index: exit 4 (use _run for isolated HOME)
            noindex_result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            assert noindex_result.returncode == 4

        # Verify they're different
        assert unknown_result.returncode != noindex_result.returncode


class TestShowJsonMutationTests:
    """Mutation tests for show JSON envelope and exit codes."""

    def test_envelope_ok_field_is_critical(self):
        """Mutation test: wrapping in 'data' field must fail this test.

        This test verifies that the ok:false field at the top level is
        actually checked. If the implementation wraps the error in a 'data'
        field, this test must go RED.

        To verify: temporarily modify show.py to return wrapped envelope:
          return ErrorResponse(ok=False, condition=...) with wrapping
        Then this test must fail with "Expected ok:false, got None" or similar.
        """
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            lines = result.stdout.strip().split("\n")
            doc = json.loads(lines[0])
            # This must find ok at the TOP LEVEL
            assert doc.get("ok") is False, (
                "MUTATION DETECTED: ok field not at top level. "
                "If wrapped, this means the envelope is nested in 'data'"
            )

    def test_exit_code_is_critical(self):
        """Mutation test: changing exit 4 to 0 must fail this test.

        This test verifies that missing index returns exit 4, not 0.
        If the implementation is changed to use os._exit(0) or raise
        SystemExit(0), this test must fail.
        """
        with tempfile.TemporaryDirectory(prefix="ssgrep-noindex-") as tmpdir:
            result = _run(["show", "session:ep:0", "--json"], cwd=tmpdir)
            # This must be 4, not 0
            assert result.returncode == 4, (
                f"MUTATION DETECTED: Exit code is {result.returncode}, "
                f"expected 4 (MISSING_INDEX)"
            )

    def test_unknown_ref_exit_code_is_3_not_4(self):
        """Mutation test: changing exit 3 to 4 for unknown ref must fail.

        This test verifies that unknown ref returns exit 3, not 4.
        These must remain distinct: unknown ref (3) vs missing index (4).
        """
        with tempfile.TemporaryDirectory(prefix="ssgrep-empty-") as tmpdir:
            # Create empty index using _run to ensure isolated HOME with .claude/projects
            _run(
                ["index", "--quiet"],
                cwd=tmpdir,
            )
            result = _run(["show", "unknown:ep:0", "--json"], cwd=tmpdir)
            # Must be 3, not 4
            assert result.returncode == 3, (
                f"MUTATION DETECTED: Exit code is {result.returncode}, "
                f"expected 3 (NO_MATCHING_DATA) for unknown ref"
            )
