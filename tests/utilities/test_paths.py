"""Tests for filesystem-aware path normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ssgrep.utilities import paths


def test_run_probe_closes_tests_and_removes_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    created = "/tmp/ssgrep-CASEPROBE-ABC"
    closed: list[int] = []
    unlinked: list[str] = []
    existence_checks: list[str] = []
    monkeypatch.setattr(paths.tempfile, "mkstemp", lambda **_kwargs: (42, created))
    monkeypatch.setattr(paths.os, "close", closed.append)
    monkeypatch.setattr(
        paths.os.path,
        "exists",
        lambda candidate: existence_checks.append(candidate) or True,
    )
    monkeypatch.setattr(paths.os, "unlink", unlinked.append)

    assert paths._run_probe("/tmp") is True
    assert closed == [42]
    assert existence_checks == ["/tmp/ssgrep-caseprobe-ABC"]
    assert unlinked == [created]


def test_run_probe_cleans_up_when_existence_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    created = "/tmp/ssgrep-CASEPROBE-ABC"
    unlinked: list[str] = []
    monkeypatch.setattr(paths.tempfile, "mkstemp", lambda **_kwargs: (42, created))
    monkeypatch.setattr(paths.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        paths.os.path, "exists", lambda _candidate: (_ for _ in ()).throw(OSError("boom"))
    )
    monkeypatch.setattr(paths.os, "unlink", unlinked.append)

    with pytest.raises(OSError, match="boom"):
        paths._run_probe("/tmp")
    assert unlinked == [created]


def test_real_probe_does_not_leave_its_temporary_file(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    assert isinstance(paths._run_probe(str(tmp_path)), bool)
    assert set(tmp_path.iterdir()) == before


def test_filesystem_probe_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(paths, "_probe_result", None)
    monkeypatch.setattr(paths.tempfile, "gettempdir", lambda: "/probe-here")
    monkeypatch.setattr(paths, "_run_probe", lambda directory: calls.append(directory) or True)

    assert paths.filesystem_is_case_insensitive() is True
    assert paths.filesystem_is_case_insensitive() is True
    assert calls == ["/probe-here"]


@pytest.mark.parametrize(
    ("error", "platform", "expected"),
    [(OSError("denied"), "darwin", True), (RuntimeError("bad probe"), "linux", False)],
)
def test_filesystem_probe_falls_back_to_platform(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    platform: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(paths, "_probe_result", None)
    monkeypatch.setattr(paths, "_run_probe", lambda _directory: (_ for _ in ()).throw(error))
    monkeypatch.setattr(sys, "platform", platform)

    assert paths.filesystem_is_case_insensitive() is expected


def test_fold_case_obeys_probe_and_normcase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths.os.path, "normcase", lambda text: f"Normalized:{text}")
    monkeypatch.setattr(paths, "filesystem_is_case_insensitive", lambda: False)
    assert paths.fold_case("MiXeD") == "Normalized:MiXeD"

    monkeypatch.setattr(paths, "filesystem_is_case_insensitive", lambda: True)
    assert paths.fold_case("MiXeD") == "normalized:mixed"


def test_canonical_expands_and_normalizes_without_resolving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(paths, "filesystem_is_case_insensitive", lambda: False)
    historical = paths.canonical("~/project/gone/../recorded")

    assert historical == home / "project" / "recorded"
    assert historical.is_absolute()


def test_resolve_live_resolves_symlinks_and_expands_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    real = home / "real"
    real.mkdir(parents=True)
    link = home / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))

    assert paths.resolve_live("~/link") == real.resolve()


def test_resolve_claude_dir_uses_default_override_and_strips_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "   ")
    assert paths.resolve_claude_dir() == (home / ".claude").resolve()

    configured = tmp_path / "configured"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", f"  {configured}  ")
    assert paths.resolve_claude_dir() == configured.resolve()


def test_resolve_opencode_dir_defaults_to_xdg_config(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG_DIR", raising=False)
    assert paths.resolve_opencode_dir() == (home / ".config" / "opencode").resolve()

    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert paths.resolve_opencode_dir() == (xdg / "opencode").resolve()

    override = tmp_path / "custom"
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", f"  {override}  ")
    assert paths.resolve_opencode_dir() == override.resolve()


def test_resolve_pi_agent_dir_defaults_and_overrides(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    assert paths.resolve_pi_agent_dir() == (home / ".pi" / "agent").resolve()

    override = tmp_path / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", f"  {override}  ")
    assert paths.resolve_pi_agent_dir() == override.resolve()


def test_resolve_prime_agent_dir_defaults_and_overrides(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("PRIME_AGENT_CODING_AGENT_DIR", raising=False)
    assert paths.resolve_prime_agent_dir() == (home / ".prime" / "agent").resolve()

    override = tmp_path / "prime-agent"
    monkeypatch.setenv("PRIME_AGENT_CODING_AGENT_DIR", f"  {override}  ")
    assert paths.resolve_prime_agent_dir() == override.resolve()


def test_resolve_codex_dir_defaults_and_overrides(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert paths.resolve_codex_dir() == (home / ".codex").resolve()

    override = tmp_path / "custom-codex"
    monkeypatch.setenv("CODEX_HOME", f"  {override}  ")
    assert paths.resolve_codex_dir() == override.resolve()


def test_is_at_or_beneath_canonicalizes_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paths, "filesystem_is_case_insensitive", lambda: True)
    assert paths.is_at_or_beneath("/WORK/Project", "/work/project")
    assert paths.is_at_or_beneath("/WORK/Project/src/module.py", "/work/project")
    assert not paths.is_at_or_beneath("/work/project-other", "/work/project")
