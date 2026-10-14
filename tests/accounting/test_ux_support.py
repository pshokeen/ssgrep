"""Support for the UX suites (``test_ux_*``): a REAL collected dataset and a real-binary runner.

``collected_scene`` writes native Claude transcripts, journal events and a pinned frequency view,
then runs the real collector so receipts, history, ledger state and the snapshot are exactly what
production writes. Nothing here hand-builds accounting files. This module holds no tests.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.test_savings_cli import BINARY

CLOCK = scene.at(hour=9)
NOW = scene.at(hour=20)
PAIRS = (0, 1, 2)


def collected_scene(root: Path, pairs: tuple[int, ...] = PAIRS) -> None:
    """Capture on, ``pairs`` of fact-source + consumer sessions, one published generation."""
    vs.enable(pin=True)
    vs.start_capture()
    for n in pairs:
        vs.pair(vs.projects(root), n, CLOCK)
    result = scene.collect(NOW)
    assert result.state == "published", result


class Binary:
    """The installed ``ssgrep`` binary in an isolated HOME/data layout (never the real one)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(
        self, *args: str, columns: int = 80, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = {**vs.env_for(self.root), "COLUMNS": str(columns), "NO_COLOR": "1"}
        environment.update(env or {})
        return subprocess.run(
            [str(BINARY), *args],
            input="",
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
            check=False,
        )

    def json(self, *args: str) -> tuple[int, dict]:
        done = self(*args, "--json")
        last = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "{}"
        return done.returncode, json.loads(last)

    def data(self, *args: str) -> dict:
        code, body = self.json(*args)
        assert code == 0, (args, body, code)
        return body["data"]


def fingerprint(directory: Path) -> dict[str, tuple[int, int, bytes]]:
    """Every file under ``directory``: (mtime_ns, size, bytes) so any touch is visible."""
    out: dict[str, tuple[int, int, bytes]] = {}
    for path in sorted(directory.rglob("*")):
        info = path.lstat()
        key = str(path.relative_to(directory))
        out[key] = (info.st_mtime_ns, info.st_size, b"" if path.is_dir() else path.read_bytes())
    return out


def data_root() -> Path:
    return Path(os.environ["SSGREP_DATA_DIR"])
