"""The installed ssgrep CLI's public surface under the pinned usecli.

These run the real console scripts from pytest's environment with an isolated HOME, so they
see what a user sees. Neither test invokes an upgrade check or installer.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import get_ssgrep_binary

SSGREP = get_ssgrep_binary()
SG = SSGREP.with_name("sg")
_STYLE_VARS = ("FORCE_COLOR", "CLICOLOR_FORCE", "GITHUB_ACTIONS", "PY_COLORS")


def _run(binary: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in _STYLE_VARS}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), NO_COLOR="1", COLUMNS="200")
    return subprocess.run(
        [str(binary), *args], capture_output=True, text=True, env=env, timeout=60, check=False
    )


@pytest.mark.parametrize("binary", [SSGREP, SG], ids=["ssgrep", "sg"])
def test_json_usage_names_ssgrep_not_usecli(binary: Path, tmp_path: Path) -> None:
    result = _run(binary, tmp_path, "--json")

    document = json.loads(result.stdout)
    assert document["ok"] is True
    assert document["data"]["usage"] == "ssgrep [COMMAND] [OPTIONS] [ARGUMENTS]"


def test_the_upstream_self_upgrade_command_is_not_exposed(tmp_path: Path) -> None:
    """usecli 0.1.81 adds a default-enabled `upgrade` command; ssgrep ships none."""
    help_text = _run(SSGREP, tmp_path, "--help")
    assert help_text.returncode == 0
    commands = [line.split()[0] for line in help_text.stdout.splitlines() if line.strip()]
    assert "upgrade" not in commands

    # `--help` only (never `--check` or an install): an unknown command exits 2.
    probe = _run(SSGREP, tmp_path, "upgrade", "--help")
    assert probe.returncode == 2
    assert "No such command" in probe.stderr + probe.stdout
