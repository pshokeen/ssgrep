"""Tests for idempotent skill installation."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

import pytest

from ssgrep.services import integrations
from ssgrep.sessions.adapters import registry

CLAUDE_BODY = 'SSGREP_BIN_PLACEHOLDER search "q" --json\n'
STANDARD_MARKER = "---\nname: ssgrep\n"


def _standard_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    template = tmp_path / "SKILL.md"
    template.write_text(STANDARD_MARKER + "body " + CLAUDE_BODY)
    monkeypatch.setattr(integrations, "_skill_template", lambda: template.read_text())
    return template.read_text(), template


def _configure_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point every agent skill root at isolated temp directories and map names."""
    roots = {
        "claude": tmp_path / "claude",
        "opencode": tmp_path / "opencode",
        "codex": tmp_path / "codex",
        "pi": tmp_path / "pi",
        "prime-agent": tmp_path / "prime-agent",
    }
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(roots["claude"]))
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(roots["opencode"]))
    monkeypatch.setenv("CODEX_HOME", str(roots["codex"]))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(roots["pi"]))
    monkeypatch.setenv("PRIME_AGENT_CODING_AGENT_DIR", str(roots["prime-agent"]))
    return roots


def _dest_file(root: Path) -> Path:
    return root / "skills" / "ssgrep" / "SKILL.md"


def test_skill_template_ships_standard_front_matter() -> None:
    content = integrations._skill_template()
    assert "name: ssgrep" in content
    assert "SSGREP_BIN_PLACEHOLDER" in content


def test_guidance_short_requires_fenced_region() -> None:
    """A template without the short block fails loudly instead of returning junk."""
    with pytest.raises(ValueError, match="ssgrep-rules:short"):
        integrations.guidance_short("---\nname: ssgrep\n---\nbody without a fenced block\n")


def test_install_skills_writes_all_five_destinations(tmp_path, monkeypatch) -> None:
    template = _standard_template(tmp_path, monkeypatch)[0]
    roots = _configure_env(tmp_path, monkeypatch)

    rendered = integrations._render_template(template)
    results = dict(integrations.install_skills())

    assert set(results) == {"claude", "opencode", "codex", "pi", "prime-agent"}
    for name, status in results.items():
        assert status == "installed"
        claude_style = name == "claude"
        content = _dest_file(roots[name]).read_text()
        if claude_style:
            assert "name: ssgrep" not in content
        else:
            assert content == rendered


def test_install_skills_is_idempotent_and_updates_marker(tmp_path, monkeypatch) -> None:
    _standard_template(tmp_path, monkeypatch)
    roots = _configure_env(tmp_path, monkeypatch)

    results = dict(integrations.install_skills())
    assert results["opencode"] == "installed"
    dest = _dest_file(roots["opencode"])
    before_mtime = dest.stat().st_mtime

    results = dict(integrations.install_skills())
    assert results["opencode"] == "already_installed"
    assert dest.stat().st_mtime == before_mtime

    # Upgrade: change the template and reinstall.
    _standard_template(tmp_path, monkeypatch)
    (tmp_path / "SKILL.md").write_text(STANDARD_MARKER + "new body " + CLAUDE_BODY)
    results = dict(integrations.install_skills())
    assert results["opencode"] == "updated"
    assert "new body" in dest.read_text()


def test_user_edited_skill_is_preserved(tmp_path, monkeypatch) -> None:
    _standard_template(tmp_path, monkeypatch)
    roots = _configure_env(tmp_path, monkeypatch)
    integrations.install_skills()

    dest = _dest_file(roots["opencode"])
    dest.write_text(dest.read_text() + "\n# my custom instructions\n")

    results = dict(integrations.install_skills())
    assert results["opencode"] == "user_modified"
    assert "# my custom instructions" in dest.read_text()


def test_existing_unmarked_managed_file_is_adopted(tmp_path, monkeypatch) -> None:
    template = _standard_template(tmp_path, monkeypatch)[0]
    roots = _configure_env(tmp_path, monkeypatch)
    executable = integrations.ssgrep_executable()
    dest = _dest_file(roots["pi"])
    dest.parent.mkdir(parents=True)
    dest.write_text(template.replace("SSGREP_BIN_PLACEHOLDER", executable))

    results = dict(integrations.install_skills())

    assert results["pi"] == "adopted"
    assert integrations._MARKER_RE.search(dest.read_text()) is not None


def test_unmarked_unrelated_file_is_user_modified(tmp_path, monkeypatch) -> None:
    _standard_template(tmp_path, monkeypatch)
    roots = _configure_env(tmp_path, monkeypatch)
    dest = _dest_file(roots["prime-agent"])
    dest.parent.mkdir(parents=True)
    dest.write_text("# completely custom instructions\n")

    results = dict(integrations.install_skills())

    assert results["prime-agent"] == "user_modified"
    assert dest.read_text() == "# completely custom instructions\n"


def test_install_errors_are_reported_not_raised(tmp_path, monkeypatch) -> None:
    _standard_template(tmp_path, monkeypatch)
    _configure_env(tmp_path, monkeypatch)

    def boom(content, destination):
        raise OSError("permission denied")

    monkeypatch.setattr(integrations, "_install_template", boom)
    results = dict(integrations.install_skills())
    for status in results.values():
        assert status.startswith("error: permission denied")


def test_ssgrep_executable_prefers_python_sibling(tmp_path, monkeypatch) -> None:
    import shlex

    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\n")
    (tmp_path / "venv" / "bin" / "ssgrep").write_text("#!/bin/sh\n")
    monkeypatch.setattr(integrations.sys, "executable", str(fake_python))
    monkeypatch.setattr(integrations.shutil, "which", lambda name: "/unused/ssgrep")

    assert integrations.ssgrep_executable() == shlex.quote(
        str(tmp_path / "venv" / "bin" / "ssgrep")
    )


def test_executable_falls_back_to_which(tmp_path, monkeypatch) -> None:
    fake_which = tmp_path / "ssgrep"
    fake_which.write_text("#!/bin/sh\n")
    monkeypatch.setattr(integrations.sys, "executable", str(tmp_path / "missing-python"))
    monkeypatch.setattr(integrations.shutil, "which", lambda name: str(fake_which))
    import shlex

    assert integrations.ssgrep_executable() == shlex.quote(str(fake_which))


def test_executable_falls_back_to_bare_name(monkeypatch) -> None:
    monkeypatch.setattr(integrations.sys, "executable", "/missing/python")
    monkeypatch.setattr(integrations.shutil, "which", lambda name: None)
    assert integrations.ssgrep_executable() == "ssgrep"


# --- Guidance accuracy -------------------------------------------------------
#
# These tests exist because guidance that drifts from the shipped behaviour is
# worse than no guidance: an agent trusts it and loses a session to a claim that
# stopped being true. Every factual assertion in SKILL.md is therefore pinned to
# the live code here, not to a copy of it.

_COMMAND_RE = re.compile(r"^SSGREP_BIN_PLACEHOLDER ([a-z-]+)(.*)$", re.M)
#: usecli supplies these to every command, so they are legal in any example.
_GLOBAL_FLAGS = frozenset({"--json", "--quiet"})


def _registered_commands() -> dict[str, Any]:
    """Map command name -> command class by loading the real command modules.

    Commands are constructed by usecli with a live Typer app, so the classes are
    instantiated here against a real ``typer.Typer`` rather than a stub -- the
    registration these tests read is the same registration usecli performs.
    """
    import typer

    package = importlib.import_module("ssgrep.cli.commands")
    found: dict[str, Any] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"ssgrep.cli.commands.{module_info.name}")
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and value.__name__.endswith("Command")
                and value.__module__ == module.__name__
            ):
                found[value(typer.Typer()).signature()] = value
    return found


def _flags_of(command: Any) -> set[str]:
    """Every ``--flag`` the command's handle() actually accepts."""
    flags: set[str] = set()
    for parameter in inspect.signature(command.handle).parameters.values():
        for meta in getattr(parameter.annotation, "__metadata__", ()):
            for candidate in (getattr(meta, "name", None), *getattr(meta, "param_decls", ())):
                if isinstance(candidate, str) and candidate.startswith("--"):
                    flags.add(candidate)
        flags.add(f"--{parameter.name.replace('_', '-')}")
    return flags


def _guidance_invocations() -> list[tuple[str, list[str]]]:
    template = integrations._skill_template()
    invocations = []
    for command, tail in _COMMAND_RE.findall(template):
        invocations.append((command, re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", tail)))
    return invocations


def test_guidance_names_only_real_commands() -> None:
    """Every `ssgrep <command>` the guidance shows is a registered command."""
    registered = _registered_commands()
    named = {command for command, _ in _guidance_invocations()}

    assert named, "guidance shows no command examples at all"
    assert named <= set(registered), (
        f"guidance names unregistered commands: {named - set(registered)}"
    )


def test_guidance_uses_only_real_flags() -> None:
    """Every `--flag` the guidance attaches to a command is accepted by it."""
    registered = _registered_commands()

    for command, flags in _guidance_invocations():
        accepted = _flags_of(registered[command]) | _GLOBAL_FLAGS
        unknown = set(flags) - accepted
        assert not unknown, f"guidance uses {unknown} with `{command}`, which does not accept them"


def test_guidance_flag_check_would_catch_a_bogus_flag() -> None:
    """The flag check is only meaningful if a wrong flag actually fails it."""
    registered = _registered_commands()

    accepted = _flags_of(registered["search"]) | _GLOBAL_FLAGS

    assert "--limit" in accepted
    assert "--no-such-flag" not in accepted


def test_guidance_runtime_labels_match_the_adapter_registry() -> None:
    """A new or removed adapter must be reflected in the shipped labels."""
    assert set(integrations.RUNTIME_LABELS) == set(registry.adapter_names())


def test_guidance_names_every_supported_runtime() -> None:
    """Each runtime's product name appears in the guidance body."""
    body = integrations.guidance_body()

    missing = [label for label in integrations.RUNTIME_LABELS.values() if label not in body]
    assert not missing, f"guidance never names these runtimes: {missing}"


def test_guidance_does_not_require_a_rebuild_for_new_content() -> None:
    """A hand-rolled `index --rebuild` recipe must never reappear."""
    body = integrations.guidance_body()

    assert "--rebuild" not in body
    assert "--full-reprocess" not in body


def test_guidance_does_not_claim_claude_only_discovery() -> None:
    """ssgrep reads five runtimes; the old 'only ~/.claude' claim is a trap."""
    body = integrations.guidance_body().lower()

    assert "only claude" not in body
    assert "only ~/.claude" not in body
    assert "only reads ~/.claude" not in body


def test_guidance_quotes_no_numeric_score_threshold() -> None:
    """Scores are corpus-relative; a shipped number is always wrong somewhere."""
    body = integrations.guidance_body()

    assert not re.search(r"\d+\.\d+", body), "guidance ships a numeric threshold"
    assert "control query" in body


def test_guidance_short_region_holds_exactly_the_seven_rules() -> None:
    body = integrations.guidance_body()
    short = integrations.guidance_short()

    assert body.count("<!-- ssgrep-rules:short -->") == 1
    assert body.count("<!-- /ssgrep-rules:short -->") == 1
    assert [int(n) for n in re.findall(r"^(\d+)\. ", short, re.M)] == [1, 2, 3, 4, 5, 6, 7]
    assert "<!--" not in short


def test_guidance_body_stays_skill_sized() -> None:
    """Agents load short skills whole; a sprawling one gets summarized away."""
    assert len(integrations.guidance_body().strip().splitlines()) <= 120


def test_installed_bodies_are_identical_across_runtimes(tmp_path, monkeypatch) -> None:
    """Frontmatter varies per runtime; the rules themselves never do."""
    roots = _configure_env(tmp_path, monkeypatch)

    integrations.install_skills()

    bodies = {
        name: integrations.guidance_body(_dest_file(root).read_text())
        for name, root in roots.items()
    }
    assert len(set(bodies.values())) == 1, "installed guidance bodies differ between runtimes"
    assert next(iter(bodies.values())) == integrations.guidance_body()


def test_claude_frontmatter_carries_triggers_without_a_name(tmp_path, monkeypatch) -> None:
    roots = _configure_env(tmp_path, monkeypatch)

    integrations.install_skills()

    claude = _dest_file(roots["claude"]).read_text()
    other = _dest_file(roots["codex"]).read_text()
    assert "name: ssgrep" not in claude
    assert "triggers:" in claude
    assert "name: ssgrep" in other
    assert "triggers:" not in other


def test_upgrade_refreshes_an_unmodified_skill(tmp_path, monkeypatch) -> None:
    """The whole delivery story: upgrading ssgrep updates the installed rules."""
    roots = _configure_env(tmp_path, monkeypatch)
    old_template = "---\nname: ssgrep\n---\n\nold guidance SSGREP_BIN_PLACEHOLDER search\n"
    monkeypatch.setattr(integrations, "_skill_template", lambda: old_template)
    integrations.install_skills()
    installed = _dest_file(roots["codex"])
    assert "old guidance" in installed.read_text()

    monkeypatch.undo()
    _configure_env(tmp_path, monkeypatch)
    results = dict(integrations.install_skills())

    assert results["codex"] == "updated"
    assert integrations.guidance_body(installed.read_text()) == integrations.guidance_body()


def test_upgrade_preserves_a_user_edited_skill(tmp_path, monkeypatch) -> None:
    roots = _configure_env(tmp_path, monkeypatch)
    integrations.install_skills()
    installed = _dest_file(roots["codex"])
    edited = installed.read_text() + "\nmy own note\n"
    installed.write_text(edited)

    results = dict(integrations.install_skills())

    assert results["codex"] == "user_modified"
    assert installed.read_text() == edited
