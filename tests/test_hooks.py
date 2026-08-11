"""Tests for the hooks command: install, uninstall, and enqueue operations.

Tests verify:
1. Install emits the real Claude Code schema (matcher groups with hooks array)
2. Install is idempotent (twice = exactly one hook)
3. Install then uninstall returns file to byte-identical prior state
4. Uninstall when not installed is safe
5. Pre-existing unrelated hooks survive install/uninstall byte-identically
6. Installed command does NOT index (must not contain 'ssgrep index')
7. Invoking hook enqueues work items (idempotent deduplication)
8. Malformed settings file produces actionable error message
9. Legacy flat-shape entries are migrated away
10. Installed hook command actually executes (not just round-tripped)

Tests never touch the real ~/.claude/settings.json. All tests redirect HOME.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ssgrep.cli.commands.hooks import (
    HOOK_COMMAND_FRAGMENT,
    HooksCommand,
    _build_hook_matcher_group,
    _enqueue_sessions,
    _hook_command,
    _install_hook,
    _is_our_hook_matcher_group,
    _read_settings,
    _session_end_hooks,
    _uninstall_hook,
)
from ssgrep.workqueue import WorkQueue

REPO_ROOT = Path(__file__).resolve().parent.parent
# The PostToolUse size guard ships as a pair: a thin hook wrapper that Claude
# Code invokes, and the checker in scripts/ that does the counting. Both are
# tracked in scripts/, so the tests below resolve them by path and fail loudly
# if either goes missing rather than skipping -- a silent skip here is how the
# wrapper's regression coverage would rot.
SIZE_HOOK_WRAPPER = REPO_ROOT / "scripts" / "check-file-size.py"
SIZE_HOOK_CHECKER = REPO_ROOT / "scripts" / "check_file_size.py"


@pytest.fixture
def temp_home(monkeypatch, tmp_path):
    """Redirect HOME to a temporary directory for hook tests.

    This ensures we never touch the real ~/.claude/settings.json.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    return fake_home


@pytest.fixture
def settings_file(temp_home):
    """Path to the settings file in the temporary home."""
    path = temp_home / ".claude" / "settings.json"
    return path


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    project = tmp_path / "test_project"
    project.mkdir()
    return project


class TestRealSchemaEmission:
    """Test that installed hooks use the real Claude Code schema."""

    def test_install_emits_real_schema_structure(self, settings_file, temp_project):
        """Installing must emit the real Claude Code schema: matcher group with hooks array."""
        _install_hook(temp_project)
        settings = _read_settings(settings_file)

        # Must have SessionEnd key
        assert "SessionEnd" in settings["hooks"]
        entries = settings["hooks"]["SessionEnd"]
        assert len(entries) == 1

        # Installed entry must be a matcher group with hooks array
        entry = entries[0]
        assert isinstance(entry, dict)
        assert "hooks" in entry, "Entry must have 'hooks' key (real schema)"
        assert isinstance(entry["hooks"], list)
        assert len(entry["hooks"]) > 0

        # Each hook must have type and command
        hook = entry["hooks"][0]
        assert hook.get("type") == "command", f"Hook must have type='command', got {hook}"
        assert "command" in hook, f"Hook must have 'command' key, got {hook}"
        assert isinstance(hook["command"], str)
        assert len(hook["command"]) > 0

    def test_installed_schema_has_no_flat_description(self, settings_file, temp_project):
        """Real schema must not have description key at entry level (that was flat schema)."""
        _install_hook(temp_project)
        settings = _read_settings(settings_file)
        entry = settings["hooks"]["SessionEnd"][0]

        # Real schema should not have description at top level
        # (description was only in the flat, broken schema)
        assert (
            "description" not in entry
        ), "Entry must not have 'description' key (that was the broken flat schema)"

    def test_build_hook_matcher_group_structure(self, temp_project):
        """_build_hook_matcher_group must produce the correct structure."""
        group = _build_hook_matcher_group(temp_project)

        assert isinstance(group, dict)
        assert "hooks" in group
        assert isinstance(group["hooks"], list)
        assert len(group["hooks"]) == 1

        hook = group["hooks"][0]
        assert hook["type"] == "command"
        assert isinstance(hook["command"], str)
        assert HOOK_COMMAND_FRAGMENT in hook["command"]


class TestInstallIdempotency:
    """Test 1: Install twice yields exactly one hook entry."""

    def test_install_twice_is_idempotent(self, settings_file, temp_project):
        """Installing the hook twice must result in exactly one entry."""
        # First install
        result1 = _install_hook(temp_project)
        assert result1["installed"] is True
        assert result1["changed"] is True

        # Read settings after first install
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 1
        first_entry = settings["hooks"]["SessionEnd"][0]

        # Second install (same project)
        result2 = _install_hook(temp_project)
        assert result2["installed"] is True

        # Read settings after second install
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 1

        # The command should be the same
        second_entry = settings["hooks"]["SessionEnd"][0]
        first_cmd = first_entry.get("hooks", [{}])[0].get("command", "")
        second_cmd = second_entry.get("hooks", [{}])[0].get("command", "")
        assert first_cmd == second_cmd


class TestInstallUninstallRoundtrip:
    """Test 2: Install then uninstall returns file to byte-identical state."""

    def test_uninstall_restores_pristine_empty(self, settings_file, temp_project):
        """Uninstalling when starting from empty/nonexistent returns to empty."""
        # Verify file doesn't exist
        assert not settings_file.exists()

        # Install
        _install_hook(temp_project)
        assert settings_file.exists()

        # Uninstall
        _uninstall_hook(temp_project)

        # File should not exist (returned to pristine state)
        assert not settings_file.exists()

    def test_uninstall_preserves_foreign_hook_byte_identical(self, settings_file, temp_project):
        """Uninstalling must preserve foreign hooks byte-identically.

        This seeds a genuine WorktreeCreate hook (real schema) and verifies
        it survives install/uninstall untouched.
        """
        # Create a settings file with a genuine foreign hook (WorktreeCreate in real schema)
        foreign_hook = {
            "hooks": [
                {
                    "type": "command",
                    "command": "/home/dev/.claude/hooks/worktree-create.sh",
                }
            ]
        }
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "WorktreeCreate": [foreign_hook],
                        "SessionEnd": [foreign_hook],  # Add a genuine foreign SessionEnd hook too
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Install our hook
        _install_hook(temp_project)

        # Verify foreign hook still exists
        settings = _read_settings(settings_file)
        assert "WorktreeCreate" in settings["hooks"]
        assert settings["hooks"]["WorktreeCreate"] == [foreign_hook]

        # Now uninstall
        _uninstall_hook(temp_project)

        # WorktreeCreate must survive byte-identically
        settings = _read_settings(settings_file)
        assert "WorktreeCreate" in settings["hooks"]
        assert settings["hooks"]["WorktreeCreate"] == [foreign_hook]

    def test_uninstall_preserves_other_hooks(self, settings_file, temp_project):
        """Uninstalling must not remove unrelated SessionEnd hooks."""
        # Create a settings file with a pre-existing foreign hook
        foreign_hook = {"hooks": [{"type": "command", "command": "some-other-tool --something"}]}
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [foreign_hook],
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Install our hook (before_content would be compared, but the file is modified by install)
        _install_hook(temp_project)
        assert len(_read_settings(settings_file)["hooks"]["SessionEnd"]) == 2

        # Uninstall our hook
        _uninstall_hook(temp_project)

        # Should have exactly one entry (the other hook)
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 1
        assert settings["hooks"]["SessionEnd"][0] == foreign_hook

    def test_uninstall_preserves_other_hook_types(self, settings_file, temp_project):
        """Uninstalling must not affect other hook types (not SessionEnd)."""
        # Create settings with SessionStart hook
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {"hooks": [{"type": "command", "command": "echo starting"}]}
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Install our SessionEnd hook
        _install_hook(temp_project)
        settings = _read_settings(settings_file)
        assert "SessionStart" in settings["hooks"]
        assert "SessionEnd" in settings["hooks"]

        # Uninstall our hook
        _uninstall_hook(temp_project)

        # SessionStart must still exist, SessionEnd must be gone
        settings = _read_settings(settings_file)
        assert "SessionStart" in settings["hooks"]
        assert "SessionEnd" not in settings["hooks"]

    def test_uninstall_removes_empty_sessionend_key(self, settings_file, temp_project):
        """Uninstalling must remove the SessionEnd key when it becomes empty."""
        # Install
        _install_hook(temp_project)
        assert "SessionEnd" in _read_settings(settings_file)["hooks"]

        # Uninstall
        _uninstall_hook(temp_project)

        # SessionEnd key should be gone
        settings = _read_settings(settings_file)
        assert "SessionEnd" not in settings.get("hooks", {})

    def test_uninstall_removes_empty_hooks_dict(self, settings_file, temp_project):
        """Uninstalling must remove the hooks dict if it becomes empty."""
        # Install (creates hooks dict)
        _install_hook(temp_project)
        assert "hooks" in _read_settings(settings_file)

        # Uninstall
        _uninstall_hook(temp_project)

        # hooks key should be gone
        settings = _read_settings(settings_file)
        assert "hooks" not in settings

    def test_uninstall_with_preserve_other_keys(self, settings_file, temp_project):
        """Uninstalling must preserve other top-level settings keys."""
        # Create settings with other keys
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "model": "claude-opus-5",
                    "hooks": {"SessionEnd": [_build_hook_matcher_group(temp_project)]},
                },
                indent=2,
            )
            + "\n"
        )

        # Uninstall
        _uninstall_hook(temp_project)

        # Other keys must survive
        settings = _read_settings(settings_file)
        assert settings["theme"] == "dark"
        assert settings["model"] == "claude-opus-5"
        assert "hooks" not in settings


class TestUninstallWhenNotInstalled:
    """Test 3: Uninstall when not installed does not error and doesn't modify file."""

    def test_uninstall_not_installed_no_file(self, settings_file, temp_project):
        """Uninstalling when no settings file exists is safe."""
        assert not settings_file.exists()

        # Uninstall must not error
        result = _uninstall_hook(temp_project)
        assert result["uninstalled"] is False
        assert result["removed"] == 0

        # File still shouldn't exist
        assert not settings_file.exists()

    def test_uninstall_not_installed_no_hooks(self, settings_file, temp_project):
        """Uninstalling when no hooks exist is safe."""
        # Create settings without hooks
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps({"theme": "dark"}, indent=2) + "\n")
        before_content = settings_file.read_text()

        # Uninstall must not error
        result = _uninstall_hook(temp_project)
        assert result["uninstalled"] is False
        assert result["removed"] == 0

        # File content must not change
        assert settings_file.read_text() == before_content

    def test_uninstall_not_installed_no_sessionend(self, settings_file, temp_project):
        """Uninstalling when SessionEnd key doesn't exist is safe."""
        # Create settings with hooks but no SessionEnd
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [{"hooks": [{"type": "command", "command": "echo start"}]}]
                    }
                },
                indent=2,
            )
            + "\n"
        )
        before_content = settings_file.read_text()

        # Uninstall must not error
        result = _uninstall_hook(temp_project)
        assert result["uninstalled"] is False
        assert result["removed"] == 0

        # File content must not change
        assert settings_file.read_text() == before_content


class TestUninstallGuardRegression:
    """Regression tripwires for the `if removed:` guard in _uninstall_hook.

    An auditor demonstrated that weakening this guard (e.g. to
    `if removed >= 0:`, always true) cascades into deleting the user's
    settings.json outright when the file's only content is an empty
    SessionEnd array, and into silently dropping the "hooks"/"SessionEnd"
    keys from a file that has other real content. These pin the *correct*,
    guarded behavior: nothing to remove must mean nothing is touched — not
    reformatted, not stripped, not deleted.

    Fixtures are written with non-canonical (4-space) indentation on
    purpose: _write_settings always emits indent=2, so any accidental
    rewrite changes these bytes even when the *logical* content it writes
    would be identical to the input. A 2-space fixture lets a spurious
    rewrite hide behind coincidental round-trip fidelity — real settings
    files are hand-edited or written by other tools and don't reliably
    round-trip through json.dumps(indent=2) unchanged.
    """

    def test_survives_when_only_content_is_empty_sessionend(self, settings_file, temp_project):
        """The auditor's exact probe: a file whose only content is an empty
        SessionEnd array must survive uninstall — not be deleted outright.
        """
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps({"hooks": {"SessionEnd": []}}, indent=4) + "\n")
        before_content = settings_file.read_text()

        result = _uninstall_hook(temp_project)

        assert result == {"uninstalled": False, "removed": 0}
        assert settings_file.exists(), "Must not delete settings.json when nothing needs removing"
        assert settings_file.read_text() == before_content, "File must be untouched, byte-for-byte"

    def test_survives_when_other_keys_present_with_empty_sessionend(
        self, settings_file, temp_project
    ):
        """A file with real, unrelated content plus an empty SessionEnd
        array must keep both — the "hooks"/"SessionEnd" keys must not be
        silently dropped just because there was nothing of ours to remove.
        """
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "model": "claude-opus-5",
                    "hooks": {
                        "SessionEnd": [],
                        "PreToolUse": [{"type": "command", "command": "x"}],
                    },
                },
                indent=4,
            )
            + "\n"
        )
        before_content = settings_file.read_text()

        result = _uninstall_hook(temp_project)

        assert result == {"uninstalled": False, "removed": 0}
        assert settings_file.read_text() == before_content, "File must be untouched, byte-for-byte"


class TestPreExistingHooksSurvive:
    """Test 4: Pre-existing unrelated hooks survive install and uninstall untouched."""

    def test_other_sessionend_hooks_survive(self, settings_file, temp_project):
        """Other SessionEnd hooks must survive our install/uninstall."""
        # Set up pre-existing hook (real schema)
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        other_hook = {"hooks": [{"type": "command", "command": "custom-tool --do-something"}]}
        settings_file.write_text(
            json.dumps({"hooks": {"SessionEnd": [other_hook]}}, indent=2) + "\n"
        )

        # Install our hook
        _install_hook(temp_project)
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 2

        # Verify our hook is present
        has_our_hook = any(_is_our_hook_matcher_group(e) for e in settings["hooks"]["SessionEnd"])
        assert has_our_hook

        # Uninstall our hook
        _uninstall_hook(temp_project)
        settings = _read_settings(settings_file)
        # Other hook must remain
        assert len(settings["hooks"]["SessionEnd"]) == 1
        assert settings["hooks"]["SessionEnd"][0] == other_hook


class TestSharedMatcherGroupSurvives:
    """Regression tests for the group-replacement data-loss bug.

    ssgrep's own hook can end up sharing a single Claude Code matcher group
    with a third-party hook (e.g. both registered under matcher "*", as a
    live probe against the real schema demonstrated). Install and uninstall
    must only ever touch ssgrep's own hook dict within that group — never
    the group as a whole, and never any other key on the group (notably
    "matcher" must survive byte-for-byte).

    Fixtures here are realistic on purpose: other top-level settings keys
    (model, permissions), a "matcher" field, and a genuine sibling hook
    co-located in the same array entry as ours. The old minimal fixtures
    elsewhere in this file (a matcher group containing only our hook) can't
    exercise this bug at all — there's nothing else in the group to lose.
    """

    @staticmethod
    def _seed_shared_group(settings_file: Path, project_a: Path) -> None:
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "model": "claude-opus-5",
                    "permissions": {"allow": ["Bash(git *)"]},
                    "hooks": {
                        "SessionEnd": [
                            {
                                "matcher": "*",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": (
                                            f"ssgrep hooks enqueue --project-dir {project_a}"
                                        ),
                                    },
                                    {
                                        "type": "command",
                                        "command": "my-other-tool --important",
                                    },
                                ],
                            }
                        ]
                    },
                },
                indent=2,
            )
            + "\n"
        )

    def test_install_from_project_b_preserves_sibling_hook_and_matcher(
        self, settings_file, tmp_path
    ):
        """Install from project A (seeded), then install again from project B.

        Must not destroy the co-located third-party hook or the group's
        "matcher" field — only ssgrep's own command may change.
        """
        project_a = tmp_path / "project_a"
        project_a.mkdir()
        project_b = tmp_path / "project_b"
        project_b.mkdir()
        self._seed_shared_group(settings_file, project_a)

        result = _install_hook(project_b)
        assert result == {"installed": True, "changed": True}

        # Re-read from disk — never trust the value our own call returned.
        settings = _read_settings(settings_file)
        assert settings["model"] == "claude-opus-5"
        assert settings["permissions"] == {"allow": ["Bash(git *)"]}

        groups = settings["hooks"]["SessionEnd"]
        assert len(groups) == 1, "Must stay one shared group, not split into two"
        group = groups[0]
        assert group["matcher"] == "*", "The group's 'matcher' field must survive"

        hooks = group["hooks"]
        assert len(hooks) == 2, "Both the third-party hook and ours must remain"
        commands = {h["command"] for h in hooks}
        assert "my-other-tool --important" in commands, "Third-party sibling hook must survive"
        assert _hook_command(project_b) in commands
        assert (
            _hook_command(project_a) not in commands
        ), "ssgrep's own command must be updated, not left stale"

    def test_uninstall_preserves_sibling_hook_and_matcher_in_shared_group(
        self, settings_file, tmp_path
    ):
        """Uninstalling must remove only ssgrep's own hook from a shared
        matcher group — the sibling hook and "matcher" field must survive.
        """
        project_a = tmp_path / "project_a"
        project_a.mkdir()
        self._seed_shared_group(settings_file, project_a)

        result = _uninstall_hook(project_a)
        assert result == {"uninstalled": True, "removed": 1}

        settings = _read_settings(settings_file)
        assert settings["model"] == "claude-opus-5"
        assert settings["permissions"] == {"allow": ["Bash(git *)"]}

        groups = settings["hooks"]["SessionEnd"]
        assert len(groups) == 1, "The group itself must survive (sibling hook remains)"
        group = groups[0]
        assert group["matcher"] == "*", "The group's 'matcher' field must survive"
        assert group["hooks"] == [
            {"type": "command", "command": "my-other-tool --important"}
        ], "Only ssgrep's own hook may be removed from the group"


class TestHookCommandDoesNotIndex:
    """Test 5: The installed command does NOT index (no 'ssgrep index')."""

    def test_hook_command_string_no_index(self, temp_project):
        """The hook command must not contain 'ssgrep index'."""
        command = _hook_command(temp_project)
        assert "ssgrep index" not in command, (
            f"Hook command violates D13: must not call index directly. " f"Got: {command}"
        )

    def test_hook_command_uses_enqueue(self, temp_project):
        """The hook command must use the enqueue action."""
        command = _hook_command(temp_project)
        assert (
            "hooks enqueue" in command
        ), f"Hook command must enqueue work, not index. Got: {command}"

    def test_hook_command_includes_project_dir(self, temp_project):
        """The hook command must pass the project directory."""
        command = _hook_command(temp_project)
        assert (
            str(temp_project) in command
        ), f"Hook command must include project directory. Got: {command}"


class TestEnqueueingWorks:
    """Test 6: Invoking the hook path enqueues exactly one work item per session."""

    @patch("ssgrep.cli.commands.hooks.discover_sessions")
    def test_enqueue_creates_work_items(self, mock_discover, temp_project):
        """Enqueueing must create work items for discovered sessions."""
        from ssgrep.types import SessionFile

        # Mock discovering a session
        session1 = SessionFile(
            path=Path("/home/.claude/projects/proj-abc/session-001.jsonl"),
            session_id="session-001",
            is_main=True,
            size=1000,
            mtime=1234567890.0,
            parent_session_id=None,
            agent_hash=None,
        )
        mock_discover.return_value = [session1]

        # Enqueue
        result = _enqueue_sessions(temp_project)
        assert result["enqueued"] == 1
        assert result["error"] is None

        # Verify work was queued
        index_dir = temp_project / ".ssgrep"
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        queue.close()

        assert len(pending) == 1
        assert pending[0].session_id == "session-001"

    @patch("ssgrep.cli.commands.hooks.discover_sessions")
    def test_enqueue_deduplicates(self, mock_discover, temp_project):
        """Invoking hook twice for same session deduplicates via workqueue."""
        from ssgrep.types import SessionFile

        session1 = SessionFile(
            path=Path("/home/.claude/projects/proj-abc/session-001.jsonl"),
            session_id="session-001",
            is_main=True,
            size=1000,
            mtime=1234567890.0,
            parent_session_id=None,
            agent_hash=None,
        )
        mock_discover.return_value = [session1]

        # First enqueue
        result1 = _enqueue_sessions(temp_project)
        assert result1["enqueued"] == 1

        # Second enqueue (same session)
        result2 = _enqueue_sessions(temp_project)
        assert result2["enqueued"] == 1

        # Verify queue has exactly one item (deduplicated)
        index_dir = temp_project / ".ssgrep"
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        queue.close()

        assert len(pending) == 1

    @patch("ssgrep.cli.commands.hooks.discover_sessions")
    def test_enqueue_handles_no_sessions(self, mock_discover, temp_project):
        """Enqueueing with no sessions is safe."""
        mock_discover.return_value = []

        result = _enqueue_sessions(temp_project)
        assert result["enqueued"] == 0
        assert result["error"] is None

    @patch("ssgrep.cli.commands.hooks.discover_sessions")
    def test_enqueue_silently_handles_errors(self, mock_discover, temp_project):
        """Enqueue must never raise, errors are silent."""
        mock_discover.side_effect = RuntimeError("Simulated discovery failure")

        # Must not raise
        result = _enqueue_sessions(temp_project)
        assert result["enqueued"] == 0
        assert result["error"] is not None


class TestSessionEndHooksReadOnly:
    """_session_end_hooks must never mutate settings as a side effect of reading.

    A prior version used dict.setdefault(), which materialized
    'hooks': {'SessionEnd': []} into settings on a bare read. Combined with
    _uninstall_hook's "if settings is now empty, delete the file" cleanup,
    that was one weakened guard away from deleting the user's global
    settings.json even for a settings file that had never mentioned hooks
    at all (see TestUninstallGuardRegression for the guard-side tripwires).
    These test the accessor directly and in isolation.
    """

    def test_no_hooks_key_present(self):
        """Reading with no 'hooks' key at all must not create one."""
        settings = {"model": "claude-opus-5"}

        entries = _session_end_hooks(settings)

        assert entries == []
        assert settings == {
            "model": "claude-opus-5"
        }, "Reading must not materialize a 'hooks' key that was never there"

    def test_hooks_present_without_sessionend(self):
        """Reading with 'hooks' present but no 'SessionEnd' must not add it."""
        settings = {"hooks": {"PreToolUse": [{"x": 1}]}}

        entries = _session_end_hooks(settings)

        assert entries == []
        assert settings == {
            "hooks": {"PreToolUse": [{"x": 1}]}
        }, "Reading must not materialize a 'SessionEnd' key that was never there"

    def test_completely_empty_settings(self):
        """Reading an empty settings dict must not materialize anything into it."""
        settings: dict = {}

        entries = _session_end_hooks(settings)

        assert entries == []
        assert settings == {}, "Reading must not materialize anything into an empty dict"


class TestMalformedSettingsFile:
    """Test 7: Malformed settings file produces actionable error message."""

    def test_corrupt_json_produces_error(self, settings_file, temp_project):
        """Corrupt JSON in settings.json must raise ValueError with message."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("{invalid json")

        # Must raise ValueError, not JSONDecodeError
        with pytest.raises(ValueError) as exc_info:
            _read_settings(settings_file)

        # Message must mention the file and the error
        assert str(settings_file) in str(exc_info.value)
        assert "JSON" in str(exc_info.value)

    def test_non_dict_toplevel_produces_error(self, settings_file, temp_project):
        """Non-dict JSON at top level must raise ValueError."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps(["not", "a", "dict"]) + "\n")

        with pytest.raises(ValueError) as exc_info:
            _read_settings(settings_file)

        assert "must contain a JSON object" in str(exc_info.value)

    def test_install_with_corrupt_settings_returns_error(self, settings_file, temp_project):
        """Install with corrupt settings must return actionable error."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("{bad json")

        with pytest.raises(ValueError):
            _install_hook(temp_project)

    def test_uninstall_with_corrupt_settings_returns_error(self, settings_file, temp_project):
        """Uninstall with corrupt settings must return actionable error."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("{bad json")

        with pytest.raises(ValueError):
            _uninstall_hook(temp_project)


class TestLegacyFlatSchemaMigration:
    """Test 8: Legacy flat-schema hooks are recognized and migrated to real schema."""

    def test_install_removes_legacy_flat_ssgrep_index(self, settings_file, temp_project):
        """Install must remove legacy flat-schema hooks with 'ssgrep index'."""
        # Create settings with a legacy flat-schema hook that violates D13
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "command": "ssgrep index --project-dir /some/path",
                                "description": "old-ssgrep-hook",
                            }
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Install the new hook
        _install_hook(temp_project)

        # Verify exactly one hook remains and it's in the real schema
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 1
        entry = settings["hooks"]["SessionEnd"][0]
        assert "hooks" in entry, "Must be real schema with hooks array"
        assert entry["hooks"][0]["type"] == "command"
        assert "hooks enqueue" in entry["hooks"][0]["command"]
        assert "ssgrep index" not in entry["hooks"][0]["command"]

    def test_install_removes_legacy_flat_schema_entry(self, settings_file, temp_project):
        """Install must remove legacy flat-schema ssgrep-session-index entries."""
        # Create settings with legacy flat schema (our old broken format)
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "command": "ssgrep hooks enqueue --project-dir /old/path",
                                "description": "ssgrep-session-index",
                            }
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Install
        _install_hook(temp_project)

        # Exactly one hook should remain in the real schema
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 1
        entry = settings["hooks"]["SessionEnd"][0]
        assert "hooks" in entry, "Must be migrated to real schema"
        # Should not have flat description key
        assert "description" not in entry

    def test_install_preserves_non_ssgrep_flat_hooks(self, settings_file, temp_project):
        """Install must not remove unrelated flat-schema hooks."""
        # Create settings with a flat-schema hook that is not ssgrep
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "command": "echo 'running workflow'",
                                "description": "my-custom-workflow",
                            }
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Install our hook
        _install_hook(temp_project)

        # Both hooks should be present
        settings = _read_settings(settings_file)
        assert len(settings["hooks"]["SessionEnd"]) == 2

        # Find the custom workflow hook (it should still be flat if it's not ours)
        custom_found = False
        for entry in settings["hooks"]["SessionEnd"]:
            if isinstance(entry, dict) and entry.get("description") == "my-custom-workflow":
                custom_found = True
        assert custom_found, "Custom hook should be preserved"

    def test_uninstall_removes_legacy_flat_entries(self, settings_file, temp_project):
        """Uninstall must remove both current and legacy flat-schema ssgrep hooks."""
        # Create settings with both current-schema and legacy flat-schema hooks
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "command": "ssgrep index --project-dir /old/path",
                                "description": "Index session transcripts for ssgrep",
                            },
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "ssgrep hooks enqueue --project-dir /new/path",
                                    }
                                ]
                            },
                            {"hooks": [{"type": "command", "command": "other-tool"}]},
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Uninstall
        result = _uninstall_hook(temp_project)

        # Both current and legacy should be removed, other hook should survive
        settings = _read_settings(settings_file)
        assert result["removed"] == 2
        assert len(settings["hooks"]["SessionEnd"]) == 1
        # Only the other-tool hook should remain
        assert "other-tool" in settings["hooks"]["SessionEnd"][0]["hooks"][0]["command"]

    def test_uninstall_legacy_flat_not_installed_by_this_version(self, settings_file, temp_project):
        """Uninstall must remove flat legacy entries even if not installed by current version."""
        # Create settings with ONLY legacy flat entries (no current-schema entry)
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "command": "ssgrep index --project-dir /old/path",
                                "description": "Index session transcripts for ssgrep",
                            }
                        ]
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Uninstall (without ever installing current version)
        result = _uninstall_hook(temp_project)

        # Legacy entry should be removed
        assert result["removed"] == 1
        # File should be cleaned up to pristine state
        assert not settings_file.exists()


class TestHooksCommandIntegration:
    """Integration tests for the HooksCommand handle method."""

    def test_install_via_handle_method(self, settings_file, temp_project):
        """Install via handle() method works end-to-end."""
        command = HooksCommand(app=MagicMock())
        command.handle(action="install", project_dir=str(temp_project))

        # Verify hook was installed in real schema
        settings = _read_settings(settings_file)
        assert "hooks" in settings
        assert "SessionEnd" in settings["hooks"]
        assert len(settings["hooks"]["SessionEnd"]) == 1
        # Verify real schema
        entry = settings["hooks"]["SessionEnd"][0]
        assert "hooks" in entry

    def test_uninstall_via_handle_method(self, settings_file, temp_project):
        """Uninstall via handle() method works end-to-end."""
        command = HooksCommand(app=MagicMock())

        # First install
        command.handle(action="install", project_dir=str(temp_project))
        assert "hooks" in _read_settings(settings_file)

        # Then uninstall
        command.handle(action="uninstall", project_dir=str(temp_project))
        assert "hooks" not in _read_settings(settings_file)

    @patch("ssgrep.cli.commands.hooks.discover_sessions")
    def test_enqueue_via_handle_method(self, mock_discover, settings_file, temp_project):
        """Enqueue via handle() method works end-to-end."""
        from ssgrep.types import SessionFile

        session1 = SessionFile(
            path=Path("/home/.claude/projects/proj-abc/session-001.jsonl"),
            session_id="session-001",
            is_main=True,
            size=1000,
            mtime=1234567890.0,
            parent_session_id=None,
            agent_hash=None,
        )
        mock_discover.return_value = [session1]

        # Enqueue via handle method
        command = HooksCommand(app=MagicMock())
        command.handle(action="enqueue", project_dir=str(temp_project))

        # Verify work was enqueued (no error should be raised)
        index_dir = temp_project / ".ssgrep"
        queue = WorkQueue(index_dir)
        queue.open()
        pending = queue.pending()
        queue.close()
        assert len(pending) == 1


class TestSurvivingMutations:
    """Tests to catch the three surviving mutations identified in the full sweep.

    These tests specifically verify file state on disk after operations, not returned values,
    to catch mutations in conditional writes and cleanup logic.
    """

    def test_install_with_legacy_and_no_command_change_persists_to_disk(
        self, settings_file, temp_project
    ):
        """Mutation #25: Install must persist even when found_current=True and command unchanged.

        Scenario: Current hook exists (from same project), legacy hook exists, command unchanged.
        The bug: write is conditional on (changed or removed_legacy), so if removed_legacy is False,
        memory is modified but not written to disk.

        This test catches the mutation where the `or removed_legacy` part of the write condition
        is deleted, causing the write to be skipped when command is unchanged.
        """
        # Setup: Create settings with a current-schema hook (same project) and a legacy flat hook
        our_hook = _build_hook_matcher_group(temp_project)
        legacy_hook = {
            "command": "ssgrep index --project-dir /old/path",
            "description": "ssgrep-session-index",
        }
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [our_hook, legacy_hook],
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Record the original hook command
        original_command = our_hook["hooks"][0]["command"]

        # Read original file state
        before_content = settings_file.read_text()
        before_mtime = settings_file.stat().st_mtime_ns

        # Now install from the same project (command should be unchanged)
        _install_hook(temp_project)

        # Read settings back from disk (critical: reload from disk, not returned value)
        settings = _read_settings(settings_file)

        # Verify the legacy hook was removed
        entries = settings["hooks"]["SessionEnd"]
        assert len(entries) == 1, "Legacy hook should be removed"
        assert _is_our_hook_matcher_group(entries[0]), "Remaining hook should be ours"

        # The new hook command should be the same as the original (same project)
        new_command = entries[0]["hooks"][0]["command"]
        assert new_command == original_command, "Command should be unchanged"

        # CRITICAL: The file MUST have been written to disk (it was modified by removing legacy)
        after_content = settings_file.read_text()
        after_mtime = settings_file.stat().st_mtime_ns

        assert after_content != before_content, (
            "File content should have changed (legacy hook removed). "
            "Mutation: write conditional on 'if changed or removed_legacy' being deleted."
        )
        assert after_mtime >= before_mtime, (
            "File mtime should be updated (write should have happened). "
            "Mutation: conditional write being skipped when command unchanged but legacy exists."
        )

    def test_uninstall_does_not_write_when_removed_zero(self, settings_file, temp_project):
        """Mutation #24: Uninstall must NOT write to disk when removed=0.

        When no ssgrep hooks are present, uninstall should return {uninstalled: False, removed: 0}
        and leave the file untouched (st_mtime_ns unchanged).

        The bug: `if removed >= 0:` mutation causes cleanup code to execute even when removed=0,
        rewriting the file with json.dumps(indent=2) formatting, different from the input.

        The fixture is written with indent=4 (not the indent=2 that
        _write_settings always emits) so the "content unchanged" assertion
        below is real evidence, not dead code: with a 2-space fixture, a
        spurious rewrite of this exact (unchanged) dict reproduces the
        input byte-for-byte, and only the mtime check would ever catch a
        regression. Real settings files don't reliably round-trip through
        json.dumps(indent=2) unchanged, so the test fixture shouldn't either.
        """
        # Setup: Create settings with a non-ssgrep hook
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [{"hooks": [{"type": "command", "command": "other-tool"}]}]
                    }
                },
                indent=4,
            )
            + "\n"
        )

        before_mtime = settings_file.stat().st_mtime_ns
        before_content = settings_file.read_text()

        # Try to uninstall (no ssgrep hook to remove)
        result = _uninstall_hook(temp_project)

        assert result["removed"] == 0, "No ssgrep hook to remove"
        assert result["uninstalled"] is False

        # CRITICAL: content must be byte-identical (file was not rewritten).
        # This is now a real assertion (see docstring) — a rewrite of the
        # unchanged dict would reformat indent=4 input to indent=2 output.
        after_content = settings_file.read_text()
        assert after_content == before_content, (
            "File content should be byte-identical when removed=0. "
            "Mutation: conditional write being executed when it shouldn't."
        )

        # Secondary corroborating signal, not the primary one.
        after_mtime = settings_file.stat().st_mtime_ns
        assert after_mtime == before_mtime, (
            "File mtime should be unchanged when removed=0. "
            "Mutation: 'if removed >= 0:' causes write even when removed=0."
        )

    def test_uninstall_with_nested_legacy_hook_removes_it(self, settings_file, temp_project):
        """Mutation #26: Uninstall must detect and remove nested-shape legacy ssgrep hooks.

        Legacy nested-shape hooks have a 'hooks' array containing 'ssgrep index' commands.
        The bug: the nested-shape detection in _is_legacy_ssgrep_hook is deleted or broken,
        so nested legacy hooks are not recognized and not removed.

        This test verifies that uninstall removes nested-shape legacy hooks even though
        they have the 'hooks' key (real schema) with a legacy 'ssgrep index' command.
        """
        # Setup: Create settings with a nested-shape legacy hook (has 'hooks' array but old command)
        nested_legacy_hook = {
            "hooks": [
                {
                    "type": "command",
                    "command": "ssgrep index --project-dir /old/path",
                }
            ]
        }
        other_hook = {"hooks": [{"type": "command", "command": "other-tool"}]}
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [nested_legacy_hook, other_hook],
                    }
                },
                indent=2,
            )
            + "\n"
        )

        # Uninstall (should remove the nested legacy hook)
        result = _uninstall_hook(temp_project)

        # Verify by reloading from disk
        settings = _read_settings(settings_file)
        entries = settings["hooks"]["SessionEnd"]

        # Should have exactly one hook left (the other-tool one)
        assert len(entries) == 1, (
            f"Expected 1 hook (other-tool), got {len(entries)}. "
            "Mutation: nested-shape legacy detection broken or deleted, legacy hook not removed."
        )
        assert "other-tool" in entries[0]["hooks"][0]["command"], "Other hook should remain"
        assert result["removed"] == 1, "Should have removed 1 nested legacy hook"


class TestHookMissingScriptResilience:
    """Test that the hook command gracefully handles missing script files.

    When CLAUDE_PROJECT_DIR points to a stale location or the script has been
    moved, the hook must exit 0 silently instead of erroring on every Write/Edit.
    The resilience lives at the invocation layer (the settings.json command),
    not inside the script (which already handles missing imports gracefully).
    """

    def test_missing_script_silent_exit_zero(self, tmp_path):
        """When the hook script doesn't exist, the command must exit 0 silently.

        Simulates the exact invocation specified in settings.json with a stale
        CLAUDE_PROJECT_DIR path, verifying the guard silently skips the script
        instead of erroring on every Write/Edit.
        """
        # Create a mock project directory
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Set CLAUDE_PROJECT_DIR to point to an old location (script missing)
        old_project_dir = tmp_path / "old_location"
        old_project_dir.mkdir()

        # The hook script path that doesn't exist
        nonexistent_script = old_project_dir / ".claude" / "hooks" / "check-file-size.py"

        # Build the command exactly as specified in settings.json with shell guard
        # This is the hardened form that should silently exit 0
        command = (
            f'sh -c \'test -f "{nonexistent_script}" && python3 "{nonexistent_script}" || exit 0\''
        )

        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )

        # Must exit 0 (success, silent)
        assert result.returncode == 0, (
            f"Missing script must exit 0, got {result.returncode}. "
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # No error output
        assert result.stderr == "", f"Missing script must produce no stderr, got: {result.stderr}"
        # No stdout output
        assert result.stdout == "", f"Missing script must produce no stdout, got: {result.stdout}"

    def test_present_script_oversized_file_warns(self, tmp_path):
        """When the script IS present and file is oversized, it warns and exits 0.

        This is the complement to the missing-script test: verify the script
        still functions normally when it does exist.
        """
        assert SIZE_HOOK_WRAPPER.exists(), (
            f"size-guard hook wrapper missing at {SIZE_HOOK_WRAPPER} -- it is "
            f"a tracked file; restore it rather than skipping this test."
        )

        # Create project structure with actual hook script
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".claude" / "hooks").mkdir(parents=True)

        # Copy the real check-file-size.py script
        hook_script = project_dir / ".claude" / "hooks" / "check-file-size.py"
        shutil.copy(SIZE_HOOK_WRAPPER, hook_script)

        # Also need the helper scripts/check_file_size.py
        scripts_dir = project_dir / "scripts"
        scripts_dir.mkdir()
        shutil.copy(SIZE_HOOK_CHECKER, scripts_dir / "check_file_size.py")

        # Create src/ and a python file that's oversized
        src_dir = project_dir / "src"
        src_dir.mkdir()
        oversized_file = src_dir / "oversized.py"

        # Generate 410 code lines (exceeds 400 ceiling)
        code_lines = ['def func():\n    """Docstring."""\n    pass\n']
        for i in range(410):
            code_lines.append(f"    x{i} = {i}\n")
        oversized_file.write_text("".join(code_lines))

        # Build the hardened command
        script_path = hook_script
        command = f'sh -c \'test -f "{script_path}" && python3 "{script_path}" || exit 0\''

        # Prepare stdin (PostToolUse hook payload for the oversized file)
        hook_input = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(oversized_file),
                },
            }
        )

        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            input=hook_input,
            capture_output=True,
            text=True,
            cwd=str(project_dir),  # So the script can resolve paths relative to project
        )

        # Must exit 0 (never blocks, even with warnings)
        assert result.returncode == 0, (
            f"Script must exit 0 even with warnings, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )

        # Should have a warning in stdout (JSON format)
        if result.stdout:
            try:
                output = json.loads(result.stdout)
                assert "additionalContext" in output.get(
                    "hookSpecificOutput", {}
                ), f"Expected warning in output, got: {result.stdout}"
            except json.JSONDecodeError:
                pass  # Warnings may not be JSON if script doesn't emit to stdout

    def test_present_script_compliant_file_silent(self, tmp_path):
        """When the script IS present and file is compliant, it's silent.

        Verify the nominal case: script runs, file is fine, no output.
        """
        assert SIZE_HOOK_WRAPPER.exists(), (
            f"size-guard hook wrapper missing at {SIZE_HOOK_WRAPPER} -- it is "
            f"a tracked file; restore it rather than skipping this test."
        )

        # Create project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".claude" / "hooks").mkdir(parents=True)

        # Copy the real scripts
        hook_script = project_dir / ".claude" / "hooks" / "check-file-size.py"
        shutil.copy(SIZE_HOOK_WRAPPER, hook_script)

        scripts_dir = project_dir / "scripts"
        scripts_dir.mkdir()
        shutil.copy(SIZE_HOOK_CHECKER, scripts_dir / "check_file_size.py")

        # Create a compliant Python file (under 300 lines)
        src_dir = project_dir / "src"
        src_dir.mkdir()
        compliant_file = src_dir / "compliant.py"
        compliant_file.write_text("def hello():\n    pass\n")

        # Build the hardened command
        script_path = hook_script
        command = f'sh -c \'test -f "{script_path}" && python3 "{script_path}" || exit 0\''

        # Prepare stdin
        hook_input = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(compliant_file),
                },
            }
        )

        # Execute
        result = subprocess.run(
            command,
            shell=True,
            input=hook_input,
            capture_output=True,
            text=True,
            cwd=str(project_dir),
        )

        # Must exit 0
        assert result.returncode == 0, (
            f"Script must exit 0 for compliant file, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )
        # No output for compliant file (silent)
        assert result.stdout == "", f"Compliant file should produce no output, got: {result.stdout}"


class TestHookCommandExecution:
    """Test that the installed hook command actually executes (not just round-tripped).

    This is the critical test that verifies the hook command string stored in
    settings.json is not merely syntactically present but genuinely executable.
    """

    def test_installed_hook_command_executes_and_enqueues(
        self, temp_home, settings_file, temp_project
    ):
        """Install hook, read the command string, execute it via subprocess, verify it works.

        This is the only test that validates:
        1. The command string round-trips correctly from settings.json
        2. The command can be executed as a subprocess (not just in-process)
        3. Exit code is 0 (success)
        4. A workqueue item is actually created (proof of execution, not just exit 0)
        5. The stored command does not have broken forms
        """
        # Step 1: Install the hook
        result = _install_hook(temp_project)
        assert result["installed"]
        assert settings_file.exists()

        # Step 2: Read the command string back from settings.json
        settings = _read_settings(settings_file)
        assert "hooks" in settings
        assert "SessionEnd" in settings["hooks"]
        assert len(settings["hooks"]["SessionEnd"]) > 0

        entry = settings["hooks"]["SessionEnd"][0]
        assert "hooks" in entry, "Must be real schema with hooks array"
        assert len(entry["hooks"]) > 0

        command_string = entry["hooks"][0].get("command", "")
        assert command_string, "Must have a non-empty command string"

        # Step 3: Verify command shape
        assert (
            "--project-dir" in command_string
        ), f"Command must include --project-dir, got: {command_string}"
        assert "enqueue" in command_string, f"Command must include 'enqueue', got: {command_string}"

        # Step 4: Execute the stored string VERBATIM. No substitution of any
        # kind -- rewriting the command before running it is what let a hook
        # that Claude Code could never execute pass this test for so long.
        executable_command = command_string

        # Create session files in temp_home/.claude/projects so discover_sessions finds them
        # Claude Code stores sessions at ~/.claude/projects/PROJECT_ID/session-*.jsonl
        # The session file must contain a cwd field that matches the project_dir for discovery
        project_id = temp_project.name  # Use temp_project name as project ID
        sessions_dir = temp_home / ".claude" / "projects" / project_id
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal session file with cwd for discovery
        session_file = sessions_dir / "session-001.jsonl"
        session_file.write_text(f'{{"session_id": "session-001", "cwd": "{temp_project}"}}\n')

        # Step 6: Run the command with isolated HOME so discover_sessions finds our test sessions
        ssgrep_dir = temp_project / ".ssgrep"
        result = subprocess.run(
            executable_command,
            shell=True,
            capture_output=True,
            text=True,
            # A bare PATH, deliberately. Claude Code runs SessionEnd hooks
            # through a plain non-interactive shell that inherits no venv
            # activation, and none of the documented installs put `ssgrep` on
            # a global PATH -- so the stored command has to name an
            # executable that resolves without help. Leaving the test
            # runner's own PATH here (which has .venv/bin on it) is what hid
            # `sh: ssgrep: command not found` from every run of this suite.
            env={"HOME": str(temp_home), "PATH": "/usr/bin:/bin"},
        )

        # Step 7: Assert exit code is 0 (success)
        assert result.returncode == 0, (
            f"Hook command exited with {result.returncode}, not 0. "
            f"Command: {executable_command}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Step 8: Assert work queue now contains an item (proof it executed)
        # This is the CRITICAL proof that the command ran, not just exited 0
        queue = WorkQueue(ssgrep_dir)
        queue.open()
        pending = queue.pending()
        queue.close()

        assert len(pending) > 0, (
            "Hook command exited 0 but created no workqueue items. "
            f"Command: {executable_command}\n"
            f"This means the command syntax is accepted but logic never ran."
        )

    def test_hook_command_names_an_executable_that_resolves_without_a_venv(
        self, temp_home, settings_file, temp_project
    ):
        """The stored command's executable must resolve for the shell Claude
        Code actually uses.

        This test previously asserted the OPPOSITE -- that the command starts
        with a bare "ssgrep " and must not be an absolute path, in the name of
        "portability". That requirement was the defect written down as a
        rule: none of the three documented installs (uvx --from, uv add,
        .venv pip install) puts an `ssgrep` executable on a global PATH, and
        a SessionEnd hook inherits no venv activation, so the "portable" form
        produced `sh: ssgrep: command not found` (rc=127) on every session
        while `ssgrep init` reported "SessionEnd hook installed."

        Portability is a real concern, but it is a property of RESOLUTION,
        not of spelling: the executable must be findable from a bare PATH.
        """
        _install_hook(temp_project)
        settings = _read_settings(settings_file)

        entry = settings["hooks"]["SessionEnd"][0]
        command_string = entry["hooks"][0].get("command", "")

        executable = shlex.split(command_string)[0]
        # Resolved against a BARE PATH, not the test runner's. pytest runs
        # with .venv/bin on PATH, so resolving against os.environ would find
        # `ssgrep` for the bare-name form too and the assertion would never
        # bite -- which is precisely how the broken form survived review.
        resolved = (
            executable
            if os.path.isabs(executable)
            else shutil.which(executable, path="/usr/bin:/bin")
        )
        assert resolved is not None and os.access(resolved, os.X_OK), (
            f"The hook's executable {executable!r} does not resolve to something "
            f"runnable from a shell with no venv on PATH, so Claude Code cannot "
            f"run this hook: {command_string}"
        )

    def test_hook_command_survives_a_project_path_containing_spaces(
        self, temp_home, settings_file, tmp_path
    ):
        """A project under `~/Documents/My Project` must produce a hook that
        runs, not one that word-splits.

        The command is stored as a single unquoted shell string. Interpolating
        the path without shlex.quote made the shell split on the space, so
        ssgrep saw `Project` as a stray positional argument and exited 2 with
        a usage dump -- on every session, for a hook install that reported
        success. Directory names with spaces are ordinary on macOS.
        """
        project = tmp_path / "My Project"
        project.mkdir()
        sessions_dir = temp_home / ".claude" / "projects" / "encoded"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "session-001.jsonl").write_text(
            json.dumps({"session_id": "session-001", "cwd": str(project)}) + "\n"
        )

        _install_hook(project)
        command_string = _read_settings(settings_file)["hooks"]["SessionEnd"][0]["hooks"][0][
            "command"
        ]

        # The shell must see exactly the argv the builder intended: any
        # word-splitting shows up here as extra arguments.
        assert shlex.split(command_string)[-1] == str(project), (
            "the project path must survive the shell as ONE argument: "
            f"{shlex.split(command_string)}"
        )

        result = subprocess.run(
            command_string,
            shell=True,
            capture_output=True,
            text=True,
            env={"HOME": str(temp_home), "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, (
            f"Hook command exited {result.returncode} for a path with a space.\n"
            f"Command: {command_string}\nstderr: {result.stderr}"
        )

        queue = WorkQueue(project / ".ssgrep")
        queue.open()
        pending = queue.pending()
        queue.close()
        assert len(pending) > 0, (
            "the hook exited 0 but enqueued nothing, so the command was accepted "
            "without its arguments arriving intact"
        )
