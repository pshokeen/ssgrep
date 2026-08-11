"""Tests for Claude Code template installation during init.

Tests verify:
1. init installs skill and command templates into CLAUDE_CONFIG_DIR
2. CLAUDE_CONFIG_DIR environment variable is respected
3. Installation is idempotent (no duplication, corruption)
4. User-modified files are not overwritten
5. Templates are present in the built wheel
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ssgrep.cli.commands.hooks import _install_claude_code_templates
from ssgrep.cli.commands.init import InitCommand


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """Redirect HOME and CLAUDE_CONFIG_DIR to temporary directories."""
    fake_home_dir = tmp_path / "home"
    fake_home_dir.mkdir()
    fake_claude_dir = tmp_path / "claude"
    fake_claude_dir.mkdir()
    monkeypatch.setenv("HOME", str(fake_home_dir))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(fake_claude_dir))
    return fake_claude_dir


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory."""
    project = tmp_path / "test_project"
    project.mkdir()
    return project


class TestTemplateInstallation:
    """Test that templates are installed into Claude Code config."""

    def test_install_templates_creates_skill_at_correct_path(self, fake_home):
        """Installing templates creates skill at CLAUDE_CONFIG_DIR/skills/ssgrep/SKILL.md."""
        result = _install_claude_code_templates()

        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        assert skill_path.exists(), f"Skill should be created at {skill_path}"
        assert result["skill"] == "installed"
        assert skill_path.read_text().strip() != ""

    def test_install_templates_creates_command_at_correct_path(self, fake_home):
        """Installing templates creates command at CLAUDE_CONFIG_DIR/commands/ssgrep/search.md."""
        result = _install_claude_code_templates()

        command_path = fake_home / "commands" / "ssgrep" / "search.md"
        assert command_path.exists(), f"Command should be created at {command_path}"
        assert result["command"] == "installed"
        assert command_path.read_text().strip() != ""

    def test_install_templates_respects_claude_config_dir(self, monkeypatch, tmp_path):
        """CLAUDE_CONFIG_DIR environment variable is respected."""
        custom_claude_dir = tmp_path / "my_claude_config"
        custom_claude_dir.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom_claude_dir))

        _install_claude_code_templates()

        skill_path = custom_claude_dir / "skills" / "ssgrep" / "SKILL.md"
        command_path = custom_claude_dir / "commands" / "ssgrep" / "search.md"
        assert skill_path.exists()
        assert command_path.exists()

    def test_install_templates_idempotent_no_duplication(self, fake_home):
        """Re-running install is idempotent — no duplication."""
        _install_claude_code_templates()
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        skill_mtime1 = skill_path.stat().st_mtime_ns

        result2 = _install_claude_code_templates()
        skill_mtime2 = skill_path.stat().st_mtime_ns

        # Should report already_installed on second run
        assert result2["skill"] == "already_installed"
        assert result2["command"] == "already_installed"

        # File should not be rewritten (mtime unchanged)
        assert skill_mtime1 == skill_mtime2

    def test_install_templates_does_not_overwrite_user_edits(self, fake_home):
        """User-modified files are NOT overwritten."""
        # First install
        result1 = _install_claude_code_templates()
        assert result1["skill"] == "installed"

        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        original_content = skill_path.read_text()

        # User edits the file (modify content, not just replacing the executable path)
        user_edit = original_content.replace("Search AI coding", "My custom text")
        skill_path.write_text(user_edit)

        # Second install should detect the edit
        result2 = _install_claude_code_templates()
        assert result2["skill"] == "user_modified"

        # User's edit should be preserved (not overwritten)
        assert skill_path.read_text() == user_edit
        assert "My custom text" in skill_path.read_text()

    def test_templates_have_content(self, fake_home):
        """Installed templates have meaningful content (not empty)."""
        _install_claude_code_templates()

        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        command_path = fake_home / "commands" / "ssgrep" / "search.md"

        skill_content = skill_path.read_text()
        command_content = command_path.read_text()

        # Both should have content and be valid markdown
        assert len(skill_content) > 100, "Skill should have substantial content"
        assert len(command_content) > 100, "Command should have substantial content"

        # Both should reference ssgrep
        assert "ssgrep" in skill_content.lower()
        assert "ssgrep" in command_content.lower()

    def test_installed_templates_contain_resolved_executable_path(self, fake_home):
        """Installed templates have placeholder substituted with resolved ssgrep path."""
        from ssgrep.cli.commands.hooks import ssgrep_executable

        _install_claude_code_templates()

        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        command_path = fake_home / "commands" / "ssgrep" / "search.md"

        skill_content = skill_path.read_text()
        command_content = command_path.read_text()

        # Placeholder must not be present in installed files
        assert "SSGREP_BIN_PLACEHOLDER" not in skill_content
        assert "SSGREP_BIN_PLACEHOLDER" not in command_content

        # Resolved executable path must be present
        resolved_path = ssgrep_executable()
        assert resolved_path in skill_content
        assert resolved_path in command_content

    def test_template_idempotency_with_resolved_path(self, fake_home):
        """Re-installing templates with same resolved path reports already_installed."""
        _install_claude_code_templates()

        # Second install should report already_installed
        result2 = _install_claude_code_templates()
        assert result2["skill"] == "already_installed"
        assert result2["command"] == "already_installed"


class TestInitWithTemplateInstallation:
    """Test init command with template installation."""

    @patch("ssgrep.cli.commands.init.api.index")
    def test_init_installs_templates(self, mock_index, fake_home, temp_project):
        """Running init installs Claude Code templates."""
        from ssgrep.types import IndexStats

        mock_index.return_value = IndexStats(
            session_count=0,
            episode_count=0,
            chunk_count=0,
            index_size_bytes=0,
            last_index_time=None,
            model_id="test-model",
            vector_dimension=768,
            skipped_records=0,
            malformed_records=0,
            schema_version=1,
        )

        command = InitCommand(app=MagicMock())
        command.handle(project_dir=str(temp_project))

        # Verify templates were installed
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        command_path = fake_home / "commands" / "ssgrep" / "search.md"

        assert skill_path.exists(), "Skill should be installed by init"
        assert command_path.exists(), "Command should be installed by init"

    @patch("ssgrep.cli.commands.init.api.index")
    def test_init_is_idempotent_with_templates(self, mock_index, fake_home, temp_project):
        """Re-running init does not duplicate or corrupt templates."""
        from ssgrep.types import IndexStats

        mock_index.return_value = IndexStats(
            session_count=0,
            episode_count=0,
            chunk_count=0,
            index_size_bytes=0,
            last_index_time=None,
            model_id="test-model",
            vector_dimension=768,
            skipped_records=0,
            malformed_records=0,
            schema_version=1,
        )

        command = InitCommand(app=MagicMock())

        # First run
        command.handle(project_dir=str(temp_project))
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        content1 = skill_path.read_text()
        mtime1 = skill_path.stat().st_mtime_ns

        # Second run
        command.handle(project_dir=str(temp_project))
        content2 = skill_path.read_text()
        mtime2 = skill_path.stat().st_mtime_ns

        # Content should be identical and mtime unchanged
        assert content1 == content2
        assert mtime1 == mtime2


class TestInitResilience:
    """Test that init is resilient to indexing failures."""

    @patch("ssgrep.cli.commands.init.api.index")
    def test_init_installs_setup_even_when_indexing_fails(
        self, mock_index, fake_home, temp_project
    ):
        """Init installs hook and templates even when indexing fails.

        This is critical: a customer who runs init on a project without
        transcripts yet must still get a working hook and installed templates,
        so that future sessions are captured. Without this, ssgrep never starts
        working until they happen to re-run init at the right moment.
        """
        from ssgrep.types import IndexNotReadyError

        # Simulate indexing failure (no transcripts)
        mock_index.side_effect = IndexNotReadyError(
            condition="no_transcripts",
            message="Cannot build index — no Claude Code transcripts",
            command="ssgrep index",
        )

        command = InitCommand(app=MagicMock())
        # init should exit with error code, but NOT before running setup
        with pytest.raises(SystemExit):
            command.handle(project_dir=str(temp_project))

        # Despite the failure, templates should be installed
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        command_path = fake_home / "commands" / "ssgrep" / "search.md"

        assert skill_path.exists(), "Skill should be installed even if indexing fails"
        assert command_path.exists(), "Command should be installed even if indexing fails"

        # Hook should also be installed
        settings_path = fake_home / "settings.json"
        assert settings_path.exists(), "Settings file should be created with hook"
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        assert "SessionEnd" in settings["hooks"]


class TestTemplatesShipInWheel:
    """Test that templates are present in the built wheel."""

    def test_skill_template_exists_in_package(self):
        """Skill template exists in src/ssgrep/templates/."""
        template_path = (
            Path(__file__).parent.parent
            / "src"
            / "ssgrep"
            / "templates"
            / "claude"
            / "skills"
            / "ssgrep"
            / "SKILL.md"
        )
        assert template_path.exists(), f"Skill template should exist at {template_path}"
        content = template_path.read_text()
        assert len(content) > 0
        assert "ssgrep" in content.lower()

    def test_command_template_exists_in_package(self):
        """Command template exists in src/ssgrep/templates/."""
        template_path = (
            Path(__file__).parent.parent
            / "src"
            / "ssgrep"
            / "templates"
            / "claude"
            / "commands"
            / "ssgrep"
            / "search.md"
        )
        assert template_path.exists(), f"Command template should exist at {template_path}"
        content = template_path.read_text()
        assert len(content) > 0
        assert "ssgrep" in content.lower()


class TestTemplateUpgradePath:
    """Marker-based installs: upgrades and path changes are not 'user edits'."""

    def test_shipped_template_upgrade_applies(self, fake_home, monkeypatch):
        """A newer shipped template overwrites an unedited install."""
        from ssgrep.cli.commands import hooks

        result1 = _install_claude_code_templates()
        assert result1["skill"] == "installed"
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        old_installed = skill_path.read_text()

        # Simulate a shipped-template upgrade by pointing the installer at a
        # patched read: the packaged template gains new content.
        real_read_text = Path.read_text

        def upgraded_read_text(self, *args, **kwargs):
            content = real_read_text(self, *args, **kwargs)
            if self.name == "SKILL.md" and "templates" in str(self):
                return content + "\n\nNEW UPSTREAM SECTION\n"
            return content

        monkeypatch.setattr(Path, "read_text", upgraded_read_text)
        result2 = hooks._install_claude_code_templates()
        assert result2["skill"] == "installed", "an unedited install must accept upgrades"
        monkeypatch.undo()
        new_installed = skill_path.read_text()
        assert "NEW UPSTREAM SECTION" in new_installed
        assert new_installed != old_installed

    def test_user_edit_still_protected(self, fake_home):
        """A genuine user edit is never overwritten, marker or not."""
        _install_claude_code_templates()
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        edited = skill_path.read_text().replace("Search AI coding", "My private notes")
        skill_path.write_text(edited)

        result = _install_claude_code_templates()
        assert result["skill"] == "user_modified"
        assert skill_path.read_text() == edited

    def test_moved_venv_path_only_change_is_not_user_modified(self, fake_home, monkeypatch):
        """A changed resolved executable path re-renders instead of refusing."""
        from ssgrep.cli.commands import hooks

        result1 = _install_claude_code_templates()
        assert result1["skill"] == "installed"
        skill_path = fake_home / "skills" / "ssgrep" / "SKILL.md"
        original_bin = hooks.ssgrep_executable()

        monkeypatch.setattr(hooks, "ssgrep_executable", lambda: "/moved/venv/bin/ssgrep")
        result2 = hooks._install_claude_code_templates()
        assert (
            result2["skill"] == "installed"
        ), "a moved-venv path change must re-render, not read as user_modified"
        content = skill_path.read_text()
        assert "/moved/venv/bin/ssgrep" in content
        assert original_bin not in content
