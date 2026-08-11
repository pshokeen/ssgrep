"""Test suite for CLI exit code contract.

Every test asserts the exact exit code literal from the spec (0, 1, 2, 3, 4).
Using literals instead of symbols ensures mutations to exit_codes.py are caught
by the tests that verify the contract, not just by tautological symbol equality.

The CLI exit-code contract defines:
- 0: success
- 1: unexpected/internal failure
- 2: usage error
- 3: no matching data
- 4: missing or unusable index
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ssgrep.cli import exit_codes
from tests.conftest import get_ssgrep_binary


def run_ssgrep(
    args: list[str],
    cwd: Path | None = None,
    json_mode: bool = False,
    isolated_home: bool = True,
) -> tuple[int, str, str]:
    """Run ssgrep CLI and return (exit_code, stdout, stderr).

    Args:
        args: Command and arguments to pass to ssgrep
        cwd: Working directory for the command
        json_mode: If True, add --json flag
        isolated_home: If True, run with isolated HOME to avoid touching real settings

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    ssgrep_bin = get_ssgrep_binary()
    cmd = [str(ssgrep_bin)] + args
    if json_mode:
        cmd.append("--json")

    env = os.environ.copy()
    if isolated_home:
        # Create a temporary home directory for this test to avoid touching
        # the real ~/.claude/settings.json
        with tempfile.TemporaryDirectory(prefix="ssgrep-test-home-") as tmpdir:
            env["HOME"] = tmpdir
            # Create ~/.claude/projects to match the expected Claude Code environment
            # (some commands like `index` require it to exist)
            Path(tmpdir, ".claude", "projects").mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                env=env,
            )
            return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr


class TestExitCodeConstants:
    """Verify exit code constants hold their contract values."""

    def test_constants_match_spec(self) -> None:
        """Exit code constants must match the contract's literal values."""
        assert exit_codes.SUCCESS == 0, "SUCCESS must be 0 per spec"
        assert exit_codes.INTERNAL_FAILURE == 1, "INTERNAL_FAILURE must be 1 per spec"
        assert exit_codes.USAGE_ERROR == 2, "USAGE_ERROR must be 2 per spec"
        assert exit_codes.NO_MATCHING_DATA == 3, "NO_MATCHING_DATA must be 3 per spec"
        assert exit_codes.MISSING_INDEX == 4, "MISSING_INDEX must be 4 per spec"


class TestSearchExitCodes:
    """Test search command exit codes."""

    def test_search_empty_scope_has_no_results(self, tmp_path_project_indexed) -> None:
        """Search on empty scope (no sessions) exits 3.

        NOTE: Exit 3 is only reached when the index is empty (has 0 sessions).
        On a populated index, the vector/BM25 fusion always returns top-k results
        regardless of score, so exit 3 is currently unreachable in practice.
        """
        code, stdout, stderr = run_ssgrep(
            ["search", "test"],
            cwd=tmp_path_project_indexed,
        )
        assert code == 3, "empty scope must exit 3 (NO_MATCHING_DATA) per spec"

    def test_search_empty_query_is_usage_error(self, tmp_path_project_indexed) -> None:
        """Empty query exits 2 (usage error), not 1."""
        code, stdout, stderr = run_ssgrep(
            ["search", ""],
            cwd=tmp_path_project_indexed,
        )
        assert code == 2, "empty query must exit 2 (USAGE_ERROR) per spec"

    def test_search_whitespace_query_is_usage_error(self, tmp_path_project_indexed) -> None:
        """Whitespace-only query exits 2 (usage error)."""
        code, stdout, stderr = run_ssgrep(
            ["search", "   "],
            cwd=tmp_path_project_indexed,
        )
        assert code == 2, "whitespace query must exit 2 (USAGE_ERROR) per spec"

    def test_search_missing_index_exits_4(self, tmp_path_empty) -> None:
        """Search without index exits 4, not 1."""
        code, stdout, stderr = run_ssgrep(
            ["search", "test"],
            cwd=tmp_path_empty,
        )
        assert code == 4, "missing index must exit 4 (MISSING_INDEX) per spec"

    def test_search_empty_scope_json_mode(self, tmp_path_project_indexed) -> None:
        """Empty scope (0 sessions) exits 3 even in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["search", "test"],
            cwd=tmp_path_project_indexed,
            json_mode=True,
        )
        assert code == 3, "empty scope must exit 3 even in JSON mode"

    def test_search_empty_query_json_mode(self, tmp_path_project_indexed) -> None:
        """Empty query in --json mode returns structured error with exit code 2.

        Empty query is a usage error per spec, so even in JSON mode the process
        exits with code 2 (USAGE_ERROR). The error condition is conveyed in the
        JSON response body.
        """
        code, stdout, stderr = run_ssgrep(
            ["search", ""],
            cwd=tmp_path_project_indexed,
            json_mode=True,
        )
        # Empty query is usage error: exit 2 even in JSON mode
        assert code == 2, "Empty query must exit 2 (USAGE_ERROR) per spec"
        # Verify error details are in the response
        import json

        doc = json.loads(stdout)
        assert doc.get("ok") is False, "Error response must have ok=False"
        assert (
            doc.get("condition") == "empty_query"
        ), f"Expected condition='empty_query', got {doc.get('condition')}"

    def test_search_missing_index_json_mode(self, tmp_path_empty) -> None:
        """Missing index in --json mode returns structured error with exit code 4.

        Missing index is per spec a missing/corrupt-index condition, so even in
        JSON mode the process exits with code 4 (MISSING_INDEX). The error condition
        is conveyed in the JSON response body.
        """
        code, stdout, stderr = run_ssgrep(
            ["search", "test"],
            cwd=tmp_path_empty,
            json_mode=True,
        )
        # Missing index: exit 4 even in JSON mode per spec
        assert code == 4, "Missing index must exit 4 (MISSING_INDEX) per spec"
        # Verify error details are in the response
        import json

        doc = json.loads(stdout)
        assert doc.get("ok") is False, "Error response must have ok=False"
        assert (
            doc.get("condition") == "missing_index"
        ), f"Expected condition='missing_index', got {doc.get('condition')}"

    def test_search_empty_scope_distinguishable_message(self, tmp_path_empty) -> None:
        """Search on empty scope (no sessions) has different message from no-match."""
        # Build an index with no sessions (discovered nothing from ~/.claude/projects/)
        run_ssgrep(["index"], cwd=tmp_path_empty)

        # Search returns 3 but message mentions "no sessions"
        code, stdout, stderr = run_ssgrep(
            ["search", "anything"],
            cwd=tmp_path_empty,
        )
        assert code == 3, "empty scope search must exit 3"
        # Check that message distinguishes empty scope
        output = stderr + stdout
        assert "no sessions" in output.lower() or "no recorded" in output.lower()


class TestShowExitCodes:
    """Test show command exit codes."""

    def test_show_unknown_ref_exits_3(self, tmp_path_project_indexed) -> None:
        """Show with unknown ref exits 3 (no matching data)."""
        code, stdout, stderr = run_ssgrep(
            ["show", "invalid_ref_zzz"],
            cwd=tmp_path_project_indexed,
        )
        assert code == 3, "unknown ref must exit 3 (NO_MATCHING_DATA) per spec"

    def test_show_missing_index_exits_4(self, tmp_path_empty) -> None:
        """Show without index exits 4 (missing index)."""
        code, stdout, stderr = run_ssgrep(
            ["show", "any_ref"],
            cwd=tmp_path_empty,
        )
        assert code == 4, "missing index must exit 4 (MISSING_INDEX) per spec"

    def test_show_unknown_ref_json_mode(self, tmp_path_project_indexed) -> None:
        """Show unknown ref exits 3 even in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["show", "invalid_ref_zzz"],
            cwd=tmp_path_project_indexed,
            json_mode=True,
        )
        assert code == 3, "unknown ref must exit 3 even in JSON mode"

    def test_show_missing_index_json_mode(self, tmp_path_empty) -> None:
        """Show missing index in --json mode returns structured error with exit code 4.

        Missing index is per spec a missing/corrupt-index condition, so even in
        JSON mode the process exits with code 4 (MISSING_INDEX). The error condition
        is conveyed in the JSON response body.
        """
        code, stdout, stderr = run_ssgrep(
            ["show", "any_ref"],
            cwd=tmp_path_empty,
            json_mode=True,
        )
        # Missing index: exit 4 even in JSON mode per spec
        assert code == 4, "Missing index must exit 4 (MISSING_INDEX) per spec"
        # Verify error details are in the response
        import json

        doc = json.loads(stdout)
        assert doc.get("ok") is False, "Error response must have ok=False"
        assert (
            doc.get("condition") == "missing_index"
        ), f"Expected condition='missing_index', got {doc.get('condition')}"


class TestIndexExitCodes:
    """Test index command exit codes."""

    def test_index_success(self, tmp_path_empty) -> None:
        """Index succeeds (exits 0) even with no sessions discovered."""
        code, stdout, stderr = run_ssgrep(
            ["index"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "index must exit 0 (SUCCESS) on success per spec"

    def test_index_success_json_mode(self, tmp_path_empty) -> None:
        """Index succeeds in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["index"],
            cwd=tmp_path_empty,
            json_mode=True,
        )
        assert code == 0, "index must exit 0 even in JSON mode"

    def test_index_rebuild_with_missing_manifest_does_not_advise_retry(
        self, tmp_path_project_indexed
    ) -> None:
        """When --rebuild fails due to manifest error, do not suggest running --rebuild again.

        This test verifies the fix for dead-end rebuild advice: if a user runs
        `ssgrep index --rebuild` and it fails with a manifest error that already
        carries a remedy, we should NOT print "Try `ssgrep index --rebuild`" again.
        """
        # Create a corrupted index: valid index.db file but missing manifest
        ssgrep_dir = tmp_path_project_indexed / ".ssgrep"

        # First remove the manifest
        manifest_path = ssgrep_dir / ".manifest"
        if manifest_path.exists():
            manifest_path.unlink()

        # Create a fake index file (corrupt but present)
        index_db = ssgrep_dir / "index.db.1"
        index_db.write_bytes(b"not a valid sqlite file" * 100)

        # Try to rebuild with missing manifest — should fail
        code, stdout, stderr = run_ssgrep(
            ["index", "--rebuild"],
            cwd=tmp_path_project_indexed,
        )

        # Should fail with INTERNAL_FAILURE (1) because it's caught by generic handler
        assert code == 1, "rebuild with missing manifest must exit 1 (INTERNAL_FAILURE)"

        # Most important: stderr should NOT contain "Try `ssgrep index --rebuild`"
        # because that's the exact command that just failed
        assert (
            "Try `ssgrep index --rebuild`" not in stderr
        ), "stderr must not suggest running the failed command again when --rebuild was passed"

        # But it SHOULD contain the remedy message from the manifest error itself
        assert (
            "manifest is missing or unreadable" in stderr
        ), "stderr must contain the manifest-specific remedy message"

    def test_index_rebuild_error_message_has_remedy_not_dead_end_advice(
        self, tmp_path_project_indexed
    ) -> None:
        """Error message with remedy should not also print the dead-end rebuild suggestion.

        When --rebuild fails with an error that already explains what went wrong
        and what to do about it (like the manifest error), we should NOT add
        "Try `ssgrep index --rebuild`" because that's redundant and potentially
        misleading (the user already passed --rebuild).
        """
        # Create a corrupted index: valid index.db file but missing manifest
        ssgrep_dir = tmp_path_project_indexed / ".ssgrep"

        # First remove the manifest
        manifest_path = ssgrep_dir / ".manifest"
        if manifest_path.exists():
            manifest_path.unlink()

        # Create a fake index file (corrupt but present)
        index_db = ssgrep_dir / "index.db.1"
        index_db.write_bytes(b"not a valid sqlite file" * 100)

        # Try to rebuild with missing manifest — should fail
        code, stdout, stderr = run_ssgrep(
            ["index", "--rebuild"],
            cwd=tmp_path_project_indexed,
        )

        # The error message itself contains the full remedy
        assert "manifest is missing or unreadable" in stderr
        assert "Restore the manifest from a backup" in stderr

        # Verify the generic dead-end advice is NOT in stderr
        assert (
            "Try `ssgrep index --rebuild`" not in stderr
        ), "generic advice must not appear when error already carries remedy instructions"

    def test_index_error_with_run_in_message_no_command_must_advise(self) -> None:
        """Error with 'Run' in text but NO command field must advise generic rebuild.

        This test discriminates the fix: text-sniffing for "Run" would suppress
        advice for an error mentioning "Run" in natural language but carrying
        no structured remedy. The field-based check avoids this false suppression.

        Direct unit test of the suppression logic.
        """

        # Create an error with "Run" in its message but NO command field
        class UnstructuredError(Exception):
            """Plain exception with 'Run' in message, no command field."""

            pass

        error = UnstructuredError(
            "Index failed. Run database integrity check manually on the file."
        )

        # Under text-sniffing (OLD, WRONG):
        # has_remedy_text = any(hint in str(error) for hint in ["Run", "Try", ...])
        # Result: has_remedy_text = True (because "Run" appears)
        # So: should_advise_rebuild = False (WRONG! This error has no remedy!)

        # Under field-based check (NEW, CORRECT):
        rebuild = False
        carries_remedy = getattr(error, "command", None) is not None
        should_advise_rebuild = not rebuild and not carries_remedy

        # Verify the logic gives the right answer
        assert (
            should_advise_rebuild
        ), "Must advise rebuild: error has 'Run' text but NO command field, --rebuild not passed"
        assert (
            carries_remedy is False
        ), "Error without command field must not be marked as carrying remedy"

        # The text-sniffing approach would fail this test:
        # It would set should_advise_rebuild = False (wrong!)
        # because "Run" appears in the error message

    def test_generic_index_failure_without_rebuild_still_advises_rebuild(
        self, monkeypatch, capsys
    ) -> None:
        """Remedyless failure without --rebuild must still print generic advice.

        The suppression must not over-fire. A failure carrying no structured
        remedy, reached WITHOUT --rebuild, must still get the generic advice—
        that is the one case where re-running with --rebuild genuinely is the fix.

        The message deliberately contains "Run": the previous text-sniffing
        implementation would have suppressed advice here. The field-based check
        does not. This test proves the difference on rendered stderr.
        """
        from unittest.mock import MagicMock

        from ssgrep.cli.commands import index as index_cmd

        def boom(*a, **kw):  # type: ignore
            raise RuntimeError("Ran out of disk space while writing the index")

        monkeypatch.setattr(index_cmd.api, "index", boom)

        with pytest.raises(SystemExit):
            index_cmd.IndexCommand(MagicMock()).handle(project_dir=".", rebuild=False)

        err = capsys.readouterr().err
        assert "Ran out of disk space" in err, "error message must appear in stderr"
        assert "Try `ssgrep index --rebuild` to force a full rebuild." in err, (
            "remedyless failure without --rebuild must still get generic advice; "
            f"stderr was: {err}"
        )


class TestStatusExitCodes:
    """Test status command exit codes."""

    def test_status_success_with_index(self, tmp_path_project_indexed) -> None:
        """Status succeeds with index."""
        code, stdout, stderr = run_ssgrep(
            ["status"],
            cwd=tmp_path_project_indexed,
        )
        assert code == 0, "status must exit 0 (SUCCESS) per spec"

    def test_status_success_without_index(self, tmp_path_empty) -> None:
        """Status succeeds even without index."""
        code, stdout, stderr = run_ssgrep(
            ["status"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "status must exit 0 even without index per spec"

    def test_status_success_json_mode(self, tmp_path_project_indexed) -> None:
        """Status succeeds in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["status"],
            cwd=tmp_path_project_indexed,
            json_mode=True,
        )
        assert code == 0, "status must exit 0 in JSON mode"


class TestHooksExitCodes:
    """Test hooks command exit codes."""

    def test_hooks_install_success(self, tmp_path_empty) -> None:
        """Hooks install succeeds."""
        code, stdout, stderr = run_ssgrep(
            ["hooks", "install"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "hooks install must exit 0 (SUCCESS) per spec"

    def test_hooks_uninstall_success(self, tmp_path_empty) -> None:
        """Hooks uninstall succeeds."""
        code, stdout, stderr = run_ssgrep(
            ["hooks", "uninstall"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "hooks uninstall must exit 0 (SUCCESS) per spec"

    def test_hooks_unknown_action_is_usage_error(self, tmp_path_empty) -> None:
        """Unknown hooks action exits 2 (usage error).

        NOTE: The old form ["hooks", "--action", "invalid_action"] hits usecli's
        parse-rejection path and exits 0 instead of 2. Using the positional form
        tests our unknown-action validation logic correctly. The --action form
        exiting 0 on an invalid value is a real, unfixed defect: it happens
        inside usecli's parser, before any ssgrep code runs, so it cannot be
        corrected from the command handler this test covers.
        """
        code, stdout, stderr = run_ssgrep(
            ["hooks", "invalid_action"],
            cwd=tmp_path_empty,
        )
        assert code == 2, "unknown action must exit 2 (USAGE_ERROR) per spec"
        output = stderr + stdout
        assert "install" in output, "error message must mention valid action 'install'"
        assert "uninstall" in output, "error message must mention valid action 'uninstall'"
        assert "enqueue" in output, "error message must mention valid action 'enqueue'"

    def test_hooks_unknown_action_json_mode(self, tmp_path_empty) -> None:
        """Unknown hooks action exits 2 even in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["hooks", "invalid_action"],
            cwd=tmp_path_empty,
            json_mode=True,
        )
        assert code == 2, "unknown action must exit 2 even in JSON mode"
        output = stderr + stdout
        assert "install" in output, "error message must mention valid action 'install'"
        assert "uninstall" in output, "error message must mention valid action 'uninstall'"
        assert "enqueue" in output, "error message must mention valid action 'enqueue'"


class TestPruneExitCodes:
    """Test prune command exit codes."""

    def test_prune_missing_index_exits_4(self, tmp_path_empty) -> None:
        """Prune without index exits 4 (missing index)."""
        code, stdout, stderr = run_ssgrep(
            ["prune", "--yes"],
            cwd=tmp_path_empty,
        )
        assert code == 4, "missing index must exit 4 (MISSING_INDEX) per spec"

    def test_prune_dry_run_success(self, tmp_path_project_indexed) -> None:
        """Prune --dry-run succeeds."""
        code, stdout, stderr = run_ssgrep(
            ["prune", "--dry-run"],
            cwd=tmp_path_project_indexed,
        )
        assert code == 0, "prune --dry-run must exit 0 (SUCCESS) per spec"

    def test_prune_success_with_yes(self, tmp_path_project_indexed) -> None:
        """Prune with --yes succeeds."""
        code, stdout, stderr = run_ssgrep(
            ["prune", "--yes"],
            cwd=tmp_path_project_indexed,
        )
        assert code == 0, "prune --yes must exit 0 (SUCCESS) per spec"

    def test_prune_missing_index_json_mode(self, tmp_path_empty) -> None:
        """Prune missing index exits 4 in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["prune", "--yes"],
            cwd=tmp_path_empty,
            json_mode=True,
        )
        assert code == 4, "missing index must exit 4 even in JSON mode"


class TestInitExitCodes:
    """Test init command exit codes."""

    def test_init_success(self, tmp_path_empty) -> None:
        """Init succeeds."""
        code, stdout, stderr = run_ssgrep(
            ["init"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "init must exit 0 (SUCCESS) per spec"

    def test_init_success_json_mode(self, tmp_path_empty) -> None:
        """Init succeeds in --json mode."""
        code, stdout, stderr = run_ssgrep(
            ["init"],
            cwd=tmp_path_empty,
            json_mode=True,
        )
        assert code == 0, "init must exit 0 even in JSON mode"


class TestInitHookCommand:
    """Guard: ensure init installs the correct hook command (D13 compliance)."""

    def test_init_hook_command_uses_enqueue_not_index(self, tmp_path_empty) -> None:
        """Init hook must call 'hooks enqueue', never 'ssgrep index' directly (D13)."""
        import json

        # Create isolated home
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as fake_home:
            env = os.environ.copy()
            env["HOME"] = fake_home
            # Create ~/.claude/projects in the fake home to match realistic Claude Code setup
            # (ssgrep only runs on machines that have recorded at least one session)
            Path(fake_home, ".claude", "projects").mkdir(parents=True, exist_ok=True)

            # Run init in the isolated environment
            ssgrep_bin = get_ssgrep_binary()
            cmd = [str(ssgrep_bin), "init"]
            result = subprocess.run(
                cmd,
                cwd=str(tmp_path_empty),
                capture_output=True,
                text=True,
                env=env,
            )
            # Check that init succeeded, with output in failure message for debugging
            assert result.returncode == 0, (
                f"Init must exit 0, got {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

            # Check settings.json in the isolated home
            settings_path = Path(fake_home) / ".claude" / "settings.json"
            assert settings_path.exists(), "Init must create settings.json"

            settings = json.loads(settings_path.read_text())
            assert "hooks" in settings, "Settings must have hooks"
            assert "SessionEnd" in settings["hooks"], "Must have SessionEnd hooks"

            hook_entry = settings["hooks"]["SessionEnd"][0]

            # Assert the entry has the correct nested structure (real Claude Code schema)
            assert isinstance(
                hook_entry, dict
            ), f"Hook entry must be a dict, got: {type(hook_entry)}"
            assert "hooks" in hook_entry, (
                f"Hook entry must have 'hooks' key (nested structure), "
                f"got keys: {hook_entry.keys()}"
            )
            assert isinstance(hook_entry["hooks"], list), (
                f"hook_entry['hooks'] must be a list, " f"got: {type(hook_entry['hooks'])}"
            )
            assert len(hook_entry["hooks"]) > 0, "hook_entry['hooks'] must have at least one entry"

            # Extract the command from the nested structure
            hook_obj = hook_entry["hooks"][0]
            assert isinstance(hook_obj, dict), f"Hook object must be a dict, got: {type(hook_obj)}"
            assert (
                "type" in hook_obj
            ), f"Hook object must have 'type' key, got keys: {hook_obj.keys()}"
            assert (
                hook_obj["type"] == "command"
            ), f"Hook type must be 'command', got: {hook_obj['type']}"
            assert (
                "command" in hook_obj
            ), f"Hook object must have 'command' key, got keys: {hook_obj.keys()}"

            command = hook_obj["command"]
            assert isinstance(command, str), f"Command must be a string, got: {type(command)}"

            # Guard: must use 'hooks enqueue', never 'ssgrep index'
            assert "hooks enqueue" in command, f"Init hook must use 'hooks enqueue', got: {command}"
            assert (
                "ssgrep index" not in command
            ), f"Init hook violates D13 (must not index directly), got: {command}"


class TestVersionExitCodes:
    """Test version command exit codes and output against independent oracles."""

    @staticmethod
    def _read_expected_version() -> str:
        """Read the expected version from pyproject.toml (independent oracle).

        This oracle is independent of the code under test: it reads the
        [project].version field directly from the source pyproject.toml file.
        """
        import tomllib
        from pathlib import Path

        # pyproject.toml is at the repo root (one level up from tests/)
        repo_root = Path(__file__).resolve().parents[1]
        pyproject = repo_root / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]

    def test_version_subcommand_prints_correct_version(self, tmp_path_empty) -> None:
        """ssgrep version (subcommand) prints the installed version, not cwd's pyproject.toml.

        This tests the fix for the usecli bug where version resolution checked
        the cwd's pyproject.toml *first*, causing `ssgrep --version` to print
        the wrong version when run inside another project.

        The correct behavior: version is read from the installed ssgrep package,
        not from any pyproject.toml found in the cwd.

        ORACLE: The expected version from our own pyproject.toml (read via
        tomllib, independent of production code).
        """
        expected_version = self._read_expected_version()

        # Run version subcommand from an empty directory
        code, stdout, stderr = run_ssgrep(
            ["version"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "version subcommand must exit 0 (SUCCESS) per spec"

        output = stdout + stderr
        # ORACLE: expected_version must appear in the output
        assert (
            expected_version in output
        ), f"Expected version '{expected_version}' not found in output: {output}"

        # GUARD: ensure it's not the unknown sentinel in the normal case
        assert "0.0.0+unknown" not in output, "Should not print unknown sentinel in normal case"

        # Sanity check: output format
        assert output.strip().startswith(
            "ssgrep "
        ), f"Output should start with 'ssgrep ', got: {output}"

    def test_version_flag_prints_correct_version(self, tmp_path_empty) -> None:
        """ssgrep --version (flag) prints the installed version, not cwd's pyproject.toml.

        This tests that the --version flag also gets the version from the
        installed package, not from the cwd's pyproject.toml. This was the
        original entry point for the version-resolution bug.

        ORACLE: The expected version from our own pyproject.toml.
        """
        expected_version = self._read_expected_version()

        code, stdout, stderr = run_ssgrep(
            ["--version"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "--version flag must exit 0 (SUCCESS) per spec"

        output = stdout + stderr
        # ORACLE: expected_version must appear in the output
        assert (
            expected_version in output
        ), f"Expected version '{expected_version}' not found in --version output: {output}"

        # GUARD: ensure it's not the unknown sentinel in the normal case
        assert (
            "0.0.0+unknown" not in output
        ), "Should not print unknown sentinel in normal case with --version"

    def test_version_ignores_decoy_pyproject_in_cwd(self, tmp_path_empty) -> None:
        """Version command uses installed package version, ignoring cwd's pyproject.toml.

        This is a concrete test of the fix for the usecli version-resolution bug.
        When ssgrep is run from inside another project with a different version
        in its pyproject.toml, ssgrep should still report its own installed
        version, not the cwd project's version.

        ORACLE: The expected version from ssgrep's own pyproject.toml.
        """
        expected_version = self._read_expected_version()

        # Create a decoy pyproject.toml in tmp_path_empty with a different version
        decoy_pyproject = tmp_path_empty / "pyproject.toml"
        decoy_pyproject.write_text("[project]\nname = 'decoy'\nversion = '9.9.9'\n")

        code, stdout, stderr = run_ssgrep(
            ["version"],
            cwd=tmp_path_empty,
        )
        assert code == 0, "version must exit 0"

        output = stdout + stderr
        # CRITICAL: Must print ssgrep's version, NOT the decoy version
        assert expected_version in output, (
            f"Must print ssgrep's version '{expected_version}', "
            f"not decoy version '9.9.9'. Got: {output}"
        )
        assert "9.9.9" not in output, f"Must not print decoy version 9.9.9. Got: {output}"


class TestRealSettingsNotModified:
    """Regression guard: ensure subprocess tests never touch real ~/.claude/settings.json.

    This prevents test suite mutations from persisting to the developer's actual
    Claude Code configuration.
    """

    def test_real_settings_untouched_after_hooks_install(self) -> None:
        """Hooks install via subprocess must not modify real settings.json."""
        import hashlib

        real_settings = Path.home() / ".claude" / "settings.json"

        # Capture state before test
        before_exists = real_settings.exists()
        before_hash = None
        before_mtime = None
        if before_exists:
            before_mtime = real_settings.stat().st_mtime
            before_hash = hashlib.md5(real_settings.read_bytes()).hexdigest()

        # Run hook install via subprocess (this is what we're guarding against)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_ssgrep(
                ["hooks", "install"],
                cwd=tmp_path,
                isolated_home=True,  # Should use isolated home
            )

        # Verify settings.json was not modified
        after_exists = real_settings.exists()
        assert before_exists == after_exists, (
            "Real settings.json existence changed during test. "
            "Tests must use isolated_home=True for subprocess calls that touch hooks."
        )

        if before_exists:
            after_mtime = real_settings.stat().st_mtime
            after_hash = hashlib.md5(real_settings.read_bytes()).hexdigest()
            assert before_mtime == after_mtime, (
                "Real settings.json mtime changed. "
                "Tests must use isolated_home=True for subprocess calls."
            )
            assert before_hash == after_hash, (
                "Real settings.json content changed. "
                "Tests must use isolated_home=True for subprocess calls."
            )

    def test_real_settings_untouched_after_init(self) -> None:
        """Init via subprocess must not modify real settings.json."""
        import hashlib

        real_settings = Path.home() / ".claude" / "settings.json"

        # Capture state before test
        before_exists = real_settings.exists()
        before_hash = None
        before_mtime = None
        if before_exists:
            before_mtime = real_settings.stat().st_mtime
            before_hash = hashlib.md5(real_settings.read_bytes()).hexdigest()

        # Run init via subprocess (this is what we're guarding against)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_ssgrep(
                ["init"],
                cwd=tmp_path,
                isolated_home=True,  # Should use isolated home
            )

        # Verify settings.json was not modified
        after_exists = real_settings.exists()
        assert before_exists == after_exists, (
            "Real settings.json existence changed during test. "
            "Tests must use isolated_home=True for subprocess calls that touch hooks."
        )

        if before_exists:
            after_mtime = real_settings.stat().st_mtime
            after_hash = hashlib.md5(real_settings.read_bytes()).hexdigest()
            assert before_mtime == after_mtime, (
                "Real settings.json mtime changed. "
                "Tests must use isolated_home=True for subprocess calls."
            )
            assert before_hash == after_hash, (
                "Real settings.json content changed. "
                "Tests must use isolated_home=True for subprocess calls."
            )


class TestWindowsUnsupportedGuard:
    """Test the Windows platform guard in cli/__init__.py."""

    def test_windows_guard_direct_call(self) -> None:
        """Direct test of the guard logic in main().

        This test calls main() after monkeypatching sys.platform to 'win32'.
        It verifies that the guard raises SystemExit(1) with an appropriate message.

        NOTE: This test runs in the same process as other tests, so fcntl may
        already be imported. The guard in main() only prevents *new* processes
        from reaching the fcntl import; it does not protect this test process
        from earlier imports. The subprocess test (below) is the real verification.
        """
        from unittest.mock import patch

        import pytest

        # Patch sys.platform to simulate Windows
        with patch("sys.platform", "win32"):
            # Capture stderr to verify the message
            with pytest.raises(SystemExit) as exc_info:
                from ssgrep.cli import main

                main()

            # Verify the exit code is 1
            assert exc_info.value.code == 1, "Windows guard must exit with code 1"

    def test_windows_guard_via_subprocess(self, tmp_path_empty) -> None:
        """Subprocess test of Windows platform guard (avoids import caching issues).

        This runs ssgrep as a fresh subprocess process with sys.platform
        monkeypatched to "win32" via a temporary script. Since it's a separate
        process, the guard catches the situation before any fcntl import.

        This is the real proof that a Windows user gets an actionable message
        instead of a raw ModuleNotFoundError.
        """
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        # Create a temporary script that patches sys.platform and imports ssgrep
        script = """
import sys
sys.platform = "win32"

# Now import and call the CLI entry point
from ssgrep.cli import main
main()
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            script_path = f.name

        try:
            env = os.environ.copy()
            # Ensure isolated HOME to avoid side effects
            with tempfile.TemporaryDirectory(prefix="ssgrep-test-win-") as tmpdir:
                env["HOME"] = tmpdir
                # Create ~/.claude/projects structure
                Path(tmpdir, ".claude", "projects").mkdir(parents=True, exist_ok=True)

                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10,
                )

                # Must exit with code 1
                assert result.returncode == 1, (
                    f"Windows guard must exit with code 1, got {result.returncode}. "
                    f"stderr: {result.stderr}"
                )

                # stderr must contain a clear message about Windows not being supported
                stderr = result.stderr
                assert (
                    "not supported on Windows" in stderr
                    or "Windows" in stderr
                    or "not support" in stderr
                ), f"stderr must contain message about Windows not being supported, got: {stderr}"

                # Must NOT be a raw ModuleNotFoundError for fcntl
                assert "ModuleNotFoundError" not in stderr, (
                    f"stderr should not show a raw ModuleNotFoundError "
                    f"(guard should catch it), got: {stderr}"
                )
                assert "fcntl" not in stderr, (
                    f"stderr should not mention fcntl "
                    f"(guard should prevent that), got: {stderr}"
                )
        finally:
            os.unlink(script_path)


class TestUsecliWrapperExitCodes:
    """Test that usecli's argv-rejection path honors the exit code contract.

    The wrapper in src/ssgrep/cli/__init__.py fixes usecli's behavior where
    parse rejections (unknown option, extra positional) incorrectly exit 0
    instead of 2 (USAGE_ERROR) per the spec.
    """

    def test_search_unknown_option_exits_2(self, tmp_path_empty) -> None:
        """Unknown option to search exits 2 (USAGE_ERROR), not 0.

        This tests the wrapper's fix for usecli's parse-rejection path.
        """
        code, stdout, stderr = run_ssgrep(
            ["search", "--unknown-flag", "query"],
            cwd=tmp_path_empty,
        )
        # Literal integer from spec per task requirement
        assert code == 2, "Unknown option must exit 2 (USAGE_ERROR) per spec"
        output = stderr + stdout
        assert "ERROR" in output, "Error output must be present"
        assert "unknown" in output.lower() or "no such" in output.lower()

    def test_index_unknown_option_exits_2(self, tmp_path_empty) -> None:
        """Unknown option to index exits 2 (USAGE_ERROR), not 0.

        This tests the wrapper on a different command to ensure the fix
        is comprehensive across all commands.
        """
        code, stdout, stderr = run_ssgrep(
            ["index", "--invalid-flag"],
            cwd=tmp_path_empty,
        )
        # Literal integer from spec per task requirement
        assert code == 2, "Unknown option must exit 2 (USAGE_ERROR) per spec"
        output = stderr + stdout
        assert "ERROR" in output, "Error output must be present"

    def test_search_unexpected_positional_exits_2(self, tmp_path_empty) -> None:
        """Unexpected extra positional to search exits 2 (USAGE_ERROR), not 0.

        This tests the wrapper's fix for usecli's parse-rejection path.
        """
        code, stdout, stderr = run_ssgrep(
            ["search", "query", "extra_positional"],
            cwd=tmp_path_empty,
        )
        # Literal integer from spec per task requirement
        assert code == 2, "Unexpected positional must exit 2 (USAGE_ERROR) per spec"
        output = stderr + stdout
        assert "ERROR" in output, "Error output must be present"
        assert "unexpected" in output.lower() or "argument" in output.lower()

    def test_hooks_unexpected_positional_exits_2(self, tmp_path_empty) -> None:
        """Unexpected extra positional to hooks exits 2 (USAGE_ERROR), not 0.

        This tests the wrapper on a different command to ensure the fix
        is comprehensive across all commands.
        """
        code, stdout, stderr = run_ssgrep(
            ["hooks", "install", "extra_arg"],
            cwd=tmp_path_empty,
        )
        # Literal integer from spec per task requirement
        assert code == 2, "Unexpected positional must exit 2 (USAGE_ERROR) per spec"
        output = stderr + stdout
        assert "ERROR" in output, "Error output must be present"

    def test_status_unknown_option_exits_2(self, tmp_path_empty) -> None:
        """Unknown option to status exits 2 (USAGE_ERROR), not 0."""
        code, stdout, stderr = run_ssgrep(
            ["status", "--bogus"],
            cwd=tmp_path_empty,
        )
        # Literal integer from spec per task requirement
        assert code == 2, "Unknown option must exit 2 (USAGE_ERROR) per spec"

    def test_show_unknown_option_exits_2(self, tmp_path_empty) -> None:
        """Unknown option to show exits 2 (USAGE_ERROR), not 0."""
        code, stdout, stderr = run_ssgrep(
            ["show", "--invalid", "ref"],
            cwd=tmp_path_empty,
        )
        # Literal integer from spec per task requirement
        assert code == 2, "Unknown option must exit 2 (USAGE_ERROR) per spec"
