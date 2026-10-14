"""Read-only accounting startup stays machine-readable with networking denied."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("args", "exit_code", "state"),
    [
        (("savings",), 0, "disabled"),
        (("metrics", "status"), 0, "disabled"),
        (("metrics", "census"), 0, "partial"),
        (("savings", "--runtime", "invalid-runtime"), 2, "invalid_filter"),
        (("metrics", "status", "--profile", "../invalid"), 2, "invalid_profile"),
    ],
)
def test_offline_accounting_startup_emits_one_json_document(
    tmp_path: Path, args: tuple[str, ...], exit_code: int, state: str
) -> None:
    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if sandbox is None:
        pytest.skip("Requires macOS sandbox-exec to deny native dependency network access")
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    # Deliberately do not inherit logging or telemetry opt-outs: importing an
    # unrelated network-enabled dependency must not corrupt the JSON contract.
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
        "SSGREP_DATA_DIR": str(data),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "SSGREP_CODEX_SESSIONS_DIR": str(home / ".codex" / "sessions"),
        "SSGREP_OMP_SESSIONS_DIR": str(home / ".omp" / "agent" / "sessions"),
        "SSGREP_PI_SESSIONS_DIR": str(home / ".pi" / "agent" / "sessions"),
        "SSGREP_PRIME_AGENT_SESSIONS_DIR": str(home / ".prime" / "agent" / "sessions"),
        "SSGREP_OPENCODE_DB": str(home / ".local" / "share" / "opencode" / "opencode.db"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
    }
    result = subprocess.run(
        [
            sandbox,
            "-p",
            "(version 1) (allow default) (deny network*)",
            str(Path(sys.executable).parent / "ssgrep"),
            "--json",
            *args,
        ],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == exit_code, result.stderr
    assert result.stderr == ""
    document = json.loads(result.stdout)
    if exit_code == 0:
        assert document["ok"] is True
        assert document["data"]["state"] == state
        if args == ("metrics", "census"):
            assert document["data"]["data"]["coverage"] == {
                "runtimes_total": 6,
                "runtimes_covered": 0,
            }
        elif args == ("savings",):
            assert document["data"]["data"]["collection_enabled"] is False
        else:
            assert document["data"]["data"]["capture"] == "off"
    else:
        assert document["ok"] is False
        assert document["condition"] == state
        assert document["message"] and document["command"]
    assert not data.exists()
    assert not tuple(home.iterdir())
    assert not tuple(workspace.iterdir())
