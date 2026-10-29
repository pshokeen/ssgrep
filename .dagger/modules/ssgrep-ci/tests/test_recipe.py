"""Behavior of the dependency-environment policy, executed for real where it is shell or Python."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_RECIPE = Path(__file__).resolve().parents[1] / "src" / "ssgrep_ci" / "recipe.py"
_spec = importlib.util.spec_from_file_location("ssgrep_ci_recipe", _RECIPE)
assert _spec is not None and _spec.loader is not None
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)

BASE = "python:3.13-slim@sha256:" + "a" * 64
NEWER_BASE = "python:3.13-slim@sha256:" + "b" * 64
SEED = "ghcr.io/example/ssgrep-ci-deps@sha256:" + "c" * 64
PACKAGES = ("git", "procps")


def test_seed_built_from_the_current_base_is_admitted() -> None:
    assert recipe.admitted_seed(SEED, BASE, BASE) == SEED


@pytest.mark.parametrize(
    ("image", "built_base"),
    [
        (SEED, NEWER_BASE),  # the checkout moved to a patched base after the seed was built
        (SEED, ""),  # provenance missing
        ("", BASE),  # nothing published yet
        ("", ""),
    ],
)
def test_seed_that_does_not_match_the_current_base_is_ignored(image: str, built_base: str) -> None:
    assert recipe.admitted_seed(image, built_base, BASE) == ""


def _run_apt(tmp_path: Path, *, dpkg_ok: bool, update_status: int, install_status: int = 0):
    """Run the real command with stand-in `dpkg` and `apt-get`; return (exit status, log lines)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "index").write_text("cached")
    (bin_dir / "dpkg").write_text(f"#!/bin/sh\nexit {0 if dpkg_ok else 1}\n")
    (bin_dir / "apt-get").write_text(
        "#!/bin/sh\n"
        f'echo "apt-get $*" >> "{log}"\n'
        f'[ "$1" = update ] && exit {update_status}\n'
        f"exit {install_status}\n"
    )
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        recipe.apt_install(PACKAGES, lists_dir=str(lists)), env=env, capture_output=True
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result.returncode, calls, lists.exists()


def test_apt_is_skipped_when_every_package_is_already_installed(tmp_path: Path) -> None:
    status, calls, lists_kept = _run_apt(tmp_path, dpkg_ok=True, update_status=100)
    assert (status, calls, lists_kept) == (0, [], True)


def test_apt_installs_missing_packages_and_clears_the_lists(tmp_path: Path) -> None:
    status, calls, lists_kept = _run_apt(tmp_path, dpkg_ok=False, update_status=0)
    assert status == 0
    assert calls == ["apt-get update", "apt-get install -y --no-install-recommends git procps"]
    assert not lists_kept


def test_failed_apt_update_fails_the_step_and_installs_nothing(tmp_path: Path) -> None:
    status, calls, lists_kept = _run_apt(tmp_path, dpkg_ok=False, update_status=100)
    assert status != 0
    assert calls == ["apt-get update"]
    assert lists_kept


def test_failed_apt_install_fails_the_step_and_keeps_the_lists(tmp_path: Path) -> None:
    status, _, lists_kept = _run_apt(tmp_path, dpkg_ok=False, update_status=0, install_status=100)
    assert status != 0
    assert lists_kept


def _interpreter_check(wanted: str) -> subprocess.CompletedProcess[bytes]:
    env = {**os.environ, "UV_PYTHON": wanted}
    return subprocess.run(
        [sys.executable, "-c", recipe.INTERPRETER_CHECK], env=env, capture_output=True
    )


def test_interpreter_check_accepts_the_requested_minor_version() -> None:
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert _interpreter_check(running).returncode == 0


def test_interpreter_check_rejects_a_different_minor_version() -> None:
    other = f"{sys.version_info.major}.{sys.version_info.minor + 1}"
    result = _interpreter_check(other)
    assert result.returncode != 0
    assert f"expected {other}".encode() in result.stderr
