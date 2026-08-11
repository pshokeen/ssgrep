"""Packaging smoke test — defuses usecli's silent command-discovery failure.

usecli's console script always points at usecli:main. The framework then
discovers commands by scanning the filesystem at runtime. If the importable
top-level package name does not exactly match the distribution name,
discovery fails silently: custom commands vanish, --version reports the
framework's own version, and no error is raised.

This test builds a wheel, installs it non-editable into a clean venv,
and runs from an unrelated working directory to prove the names are
correct. Verification inside the source tree is worthless — usecli's
cwd fallback masks the failure.

The test is marked slow (it builds a wheel and installs into a fresh venv)
and runs in CI as part of the standard test suite.
"""

from __future__ import annotations

import subprocess
import tempfile
import venv
import zipfile
from importlib.metadata import version as _installed_version
from pathlib import Path

import pytest

from tests.conftest import get_ssgrep_binary

REPO_ROOT = Path(__file__).resolve().parent.parent
_DECOY_PYPROJECT = '[project]\nname = "totally-different-app"\nversion = "9.9.9"\n'


@pytest.mark.slow
@pytest.mark.skipif(
    not REPO_ROOT.joinpath("pyproject.toml").exists(),
    reason="Must run from the ssgrep repo root",
)
def test_packaging_names_match_and_commands_discoverable():
    """Build wheel, install non-editable, run from unrelated dir.

    Asserts:
    1. ssgrep --version reports "ssgrep <version>", not "usecli <version>"
    2. All 5 commands are discoverable in ssgrep --help
    3. Each command is callable without error
    """
    with tempfile.TemporaryDirectory(prefix="ssgrep-pkgtest-") as tmpdir:
        tmpdir = Path(tmpdir)

        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()
        build_result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        assert build_result.returncode == 0, f"Wheel build failed:\n{build_result.stderr}"
        wheels = list(dist_dir.glob("*.whl"))
        assert len(wheels) == 1, f"Expected 1 wheel, found {len(wheels)}: {wheels}"
        wheel_path = wheels[0]

        # Step 2: Create a clean venv and install the wheel non-editable
        venv_dir = tmpdir / "venv"
        venv.create(str(venv_dir), with_pip=True)
        pip_path = venv_dir / "bin" / "pip"

        install_result = subprocess.run(
            [str(pip_path), "install", str(wheel_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert install_result.returncode == 0, f"pip install failed:\n{install_result.stderr}"

        # Step 3: Run from an UNRELATED directory (not the repo)
        run_dir = tmpdir / "unrelated_dir"
        run_dir.mkdir()
        ssgrep_path = venv_dir / "bin" / "ssgrep"

        # 3a: --version must report ssgrep, not usecli
        version_result = subprocess.run(
            [str(ssgrep_path), "--version"],
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=30,
        )
        assert version_result.returncode == 0, f"--version failed:\n{version_result.stderr}"
        version_output = version_result.stdout.strip()
        assert (
            "ssgrep" in version_output.lower()
        ), f"--version output must contain 'ssgrep', got: {version_output!r}"
        assert (
            "usecli" not in version_output.lower()
        ), f"--version output must NOT contain 'usecli', got: {version_output!r}"

        # 3b: --help must list all 5 commands
        help_result = subprocess.run(
            [str(ssgrep_path), "--help"],
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=30,
        )
        assert help_result.returncode == 0, f"--help failed:\n{help_result.stderr}"
        help_output = help_result.stdout
        for cmd in ("index", "search", "show", "status", "mcp"):
            assert cmd in help_output, f"Command '{cmd}' not found in --help output"

        # 3c: Each command is callable (smoke — just check it doesn't crash on --help)
        for cmd in ("index", "search", "show", "status", "mcp"):
            cmd_result = subprocess.run(
                [str(ssgrep_path), cmd, "--help"],
                capture_output=True,
                text=True,
                cwd=str(run_dir),
                timeout=30,
            )
            assert cmd_result.returncode == 0, f"ssgrep {cmd} --help failed:\n{cmd_result.stderr}"


@pytest.mark.slow
@pytest.mark.skipif(
    not REPO_ROOT.joinpath("pyproject.toml").exists(),
    reason="Must run from the ssgrep repo root",
)
def test_no_inspire_command_leaks():
    """The 'inspire' demo command must not appear in production help.

    usecli's default inspire command is a demo; our config sets
    hide_inspire = true. This test verifies it actually worked.
    """
    with tempfile.TemporaryDirectory(prefix="ssgrep-pkgtest-") as tmpdir:
        tmpdir = Path(tmpdir)

        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()
        build_result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        assert build_result.returncode == 0
        wheels = list(dist_dir.glob("*.whl"))
        wheel_path = wheels[0]

        venv_dir = tmpdir / "venv"
        venv.create(str(venv_dir), with_pip=True)
        pip_path = venv_dir / "bin" / "pip"
        subprocess.run(
            [str(pip_path), "install", str(wheel_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        run_dir = tmpdir / "unrelated_dir"
        run_dir.mkdir()
        ssgrep_path = venv_dir / "bin" / "ssgrep"

        help_result = subprocess.run(
            [str(ssgrep_path), "--help"],
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=30,
        )
        assert help_result.returncode == 0
        assert (
            "inspire" not in help_result.stdout.lower()
        ), "'inspire' command should be hidden but appeared in --help"


@pytest.mark.slow
@pytest.mark.skipif(
    not REPO_ROOT.joinpath("pyproject.toml").exists(),
    reason="Must run from the ssgrep repo root",
)
def test_no_backup_files_in_wheel():
    """Verify that only allowed file types are included under ssgrep/ in the wheel.

    Uses an allowlist approach: every file under ssgrep/ must be either:
    - A .py file (Python source)
    - usecli.config.toml (configuration file)

    Note: Files matching *.bak, *.orig, and *.fixed are already filtered at
    build time by pyproject.toml's exclude patterns. This test is a second
    line of defense against other stray files (.tmp, .swp, etc.) that the
    build-time exclude patterns do not cover.
    """
    with tempfile.TemporaryDirectory(prefix="ssgrep-pkgtest-") as tmpdir:
        tmpdir = Path(tmpdir)

        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()
        build_result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        assert build_result.returncode == 0, f"Wheel build failed:\n{build_result.stderr}"
        wheels = list(dist_dir.glob("*.whl"))
        assert len(wheels) == 1, f"Expected 1 wheel, found {len(wheels)}: {wheels}"
        wheel_path = wheels[0]

        # Extract and inspect wheel contents under ssgrep/
        # Valid files: *.py, usecli.config.toml, or templates (markdown files)
        illegal_files = []
        with zipfile.ZipFile(wheel_path, "r") as whl:
            for name in whl.namelist():
                # Only check files under ssgrep/ (not dist-info, etc.)
                if name.startswith("ssgrep/") and not name.endswith("/"):
                    # File is under ssgrep/; check if it's allowed
                    is_python = name.endswith(".py")
                    is_config = name == "ssgrep/usecli.config.toml"
                    is_template = name.startswith("ssgrep/templates/") and name.endswith(".md")
                    if not (is_python or is_config or is_template):
                        illegal_files.append(name)

        assert len(illegal_files) == 0, (
            "Wheel should only contain .py files, usecli.config.toml, or "
            f"templates under ssgrep/, but found: {illegal_files}"
        )


@pytest.mark.skipif(
    not REPO_ROOT.joinpath("pyproject.toml").exists(),
    reason="Must run from the ssgrep repo root",
)
def test_version_flag_ignores_decoy_pyproject_in_cwd():
    """`ssgrep --version` must report ssgrep's own version, not a decoy's.

    Regression test for: usecli's `--version` flag resolves the version via
    CommandService._load_version(), which calls ConfigManager.get_project_version()
    FIRST -- a search that walks up from the current working directory for the
    nearest pyproject.toml and reads its [project].version, with no check that
    the file it finds has anything to do with ssgrep. Only if that search comes
    up empty does it fall back to the installed-distribution lookup.

    A buyer running `ssgrep --version` inside their own Python project (the
    normal case for an installed CLI tool) got *that* project's version
    printed back at them instead of ssgrep's.

    Every prior verification of --version in this project ran from /tmp or an
    unrelated directory containing NO pyproject.toml at all. That accidentally
    exercises usecli's fallback path (nothing found -> correct distribution
    lookup) and hides the bug completely. This test plants a decoy
    pyproject.toml -- a different name, a different version -- to force the
    buggy first branch and prove it no longer wins.

    ssgrep's `about` command is included here too: it happens to call the
    distribution lookup *before* the same cwd-walk fallback (the reverse,
    correct order), so it was never actually broken -- this locks that in.
    """
    real_version = _installed_version("ssgrep")
    ssgrep_path = get_ssgrep_binary()

    with tempfile.TemporaryDirectory(prefix="ssgrep-decoy-") as tmpdir:
        decoy_dir = Path(tmpdir)
        (decoy_dir / "pyproject.toml").write_text(_DECOY_PYPROJECT)

        version_result = subprocess.run(
            [str(ssgrep_path), "--version"],
            capture_output=True,
            text=True,
            cwd=str(decoy_dir),
            timeout=30,
        )
        assert version_result.returncode == 0, f"--version failed:\n{version_result.stderr}"
        output = version_result.stdout
        assert "9.9.9" not in output, (
            f"--version leaked the decoy project's version instead of ssgrep's "
            f"own, got: {output!r}"
        )
        assert real_version in output, (
            f"--version must report ssgrep's own installed version "
            f"{real_version!r}, got: {output!r}"
        )

        # -v is the same eager option under a short flag -- same code path.
        short_flag_result = subprocess.run(
            [str(ssgrep_path), "-v"],
            capture_output=True,
            text=True,
            cwd=str(decoy_dir),
            timeout=30,
        )
        assert short_flag_result.returncode == 0, f"-v failed:\n{short_flag_result.stderr}"
        assert "9.9.9" not in short_flag_result.stdout
        assert real_version in short_flag_result.stdout

        about_result = subprocess.run(
            [str(ssgrep_path), "about"],
            capture_output=True,
            text=True,
            cwd=str(decoy_dir),
            timeout=30,
        )
        assert about_result.returncode == 0, f"about failed:\n{about_result.stderr}"
        about_output = about_result.stdout
        assert "9.9.9" not in about_output, (
            f"about leaked the decoy project's version instead of ssgrep's own, "
            f"got: {about_output!r}"
        )
        assert real_version in about_output, (
            f"about must report ssgrep's own installed version {real_version!r}, "
            f"got: {about_output!r}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not REPO_ROOT.joinpath("pyproject.toml").exists(),
    reason="Must run from the ssgrep repo root",
)
def test_version_flag_ignores_decoy_pyproject_wheel_install():
    """Same regression as test_version_flag_ignores_decoy_pyproject_in_cwd,
    but against a real non-editable wheel install run from a directory with a
    decoy pyproject.toml -- the exact shape a paying customer's environment
    takes, not just the dev editable install.

    test_packaging_names_match_and_commands_discoverable (above) already
    builds a wheel and runs from an "unrelated_dir", but that directory has
    no pyproject.toml at all, so it cannot catch this defect -- usecli's
    fallback path happens to be correct when nothing is found. This test
    closes that exact gap by planting a decoy pyproject.toml in the run
    directory.
    """
    with tempfile.TemporaryDirectory(prefix="ssgrep-pkgtest-") as tmpdir:
        tmpdir = Path(tmpdir)

        # The wheel is built from this checkout's current pyproject.toml, so
        # read the expected version from the same source uv build will use --
        # not hardcoded, so this stays correct across version bumps.
        import tomllib

        with open(REPO_ROOT / "pyproject.toml", "rb") as f:
            expected_version = tomllib.load(f)["project"]["version"]

        dist_dir = tmpdir / "dist"
        dist_dir.mkdir()
        build_result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        assert build_result.returncode == 0, f"Wheel build failed:\n{build_result.stderr}"
        wheels = list(dist_dir.glob("*.whl"))
        assert len(wheels) == 1, f"Expected 1 wheel, found {len(wheels)}: {wheels}"
        wheel_path = wheels[0]

        venv_dir = tmpdir / "venv"
        venv.create(str(venv_dir), with_pip=True)
        pip_path = venv_dir / "bin" / "pip"
        install_result = subprocess.run(
            [str(pip_path), "install", str(wheel_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert install_result.returncode == 0, f"pip install failed:\n{install_result.stderr}"

        decoy_dir = tmpdir / "decoy_project"
        decoy_dir.mkdir()
        (decoy_dir / "pyproject.toml").write_text(_DECOY_PYPROJECT)

        ssgrep_path = venv_dir / "bin" / "ssgrep"
        version_result = subprocess.run(
            [str(ssgrep_path), "--version"],
            capture_output=True,
            text=True,
            cwd=str(decoy_dir),
            timeout=30,
        )
        assert version_result.returncode == 0, f"--version failed:\n{version_result.stderr}"
        output = version_result.stdout
        assert "9.9.9" not in output, (
            f"--version leaked the decoy project's version instead of ssgrep's "
            f"own, got: {output!r}"
        )
        assert expected_version in output, (
            f"--version must report ssgrep's own version {expected_version!r}, " f"got: {output!r}"
        )
