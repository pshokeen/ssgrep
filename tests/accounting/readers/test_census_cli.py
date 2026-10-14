"""4.8 actual installed ``ssgrep metrics census`` over an isolated six-runtime home."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.accounting.readers.test_census import tree_digest
from tests.accounting.readers_fixtures import build_home

BIN = Path(sys.executable).parent / "ssgrep"


def runtime_env(home, tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        **home.env,
        "SSGREP_DATA_DIR": str(tmp_path / "data"),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
        "NO_COLOR": "1",
    }


def run(home, *args: str, tmp_path: Path, cwd: Path | None = None):
    return subprocess.run(
        [str(BIN), *args],
        env=runtime_env(home, tmp_path),
        cwd=cwd or tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def warm_process_loader(home, tmp_path: Path) -> None:
    """Initialize Python/SQLite loader caches, without importing or running ssgrep."""
    started = subprocess.run(
        [sys.executable, "-c", "import sqlite3"],
        env=runtime_env(home, tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert started.returncode == 0, started.stderr


def test_json_census_covers_all_six_formats_without_side_effects(tmp_path: Path) -> None:
    home = build_home(tmp_path / "home")
    # Rosetta may initialize HOME/.cache on process startup, even for plain
    # Python. Warm only the interpreter/SQLite runtime, never the census or CLI:
    # application writes must remain observable, not become part of the baseline.
    sources = tree_digest(home.root)
    warm_process_loader(home, tmp_path)
    warmed = tree_digest(home.root)
    assert all(warmed[path] == digest for path, digest in sources.items())
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "xdg-data").exists()
    before = tree_digest(tmp_path)
    done = run(home, "--json", "metrics", "census", "--root", str(home.code), tmp_path=tmp_path)
    assert done.returncode == 0, done.stderr
    body = json.loads(done.stdout)
    assert body["ok"] is True
    document = body["data"]
    assert {"schema_version", "state", "data", "warnings", "next_actions", "provenance"} <= set(
        document
    )
    data = document["data"]
    assert document["state"] == "ok" and data["coverage"] == {
        "runtimes_total": 6,
        "runtimes_covered": 6,
    }
    sources = {name: row["sources"] for name, row in data["runtimes"].items()}
    assert sources == {"claude": 4, "codex": 1, "omp": 2, "opencode": 1, "pi": 1, "prime-agent": 3}
    assert data["totals"]["recognized_calls"] == 12
    assert data["repositories"]["found"] == 4 and data["repositories"]["symlinks_skipped"] == 1
    assert document["provenance"] == {"read_only": True, "content_included": False}
    assert str(home.root) not in done.stdout and "evil.invalid" not in done.stdout
    # Include HOME, explicit data/index roots, and loader caches in the snapshot.
    # No cache paths are excluded: any census-time creation/change still fails.
    after = tree_digest(tmp_path)
    created = set(after) - set(before)
    assert created <= {f"{home.paths['opencode']}-shm", f"{home.paths['opencode']}-wal"}
    assert all(after[path] == digest for path, digest in before.items())
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "xdg-data").exists()


def test_text_census_default_root_is_cwd(tmp_path: Path) -> None:
    home = build_home(tmp_path / "home")
    done = run(home, "metrics", "census", tmp_path=tmp_path, cwd=home.code / "beta")
    assert done.returncode == 0, done.stderr
    assert "ssgrep metrics census" in done.stdout
    assert "claude" in done.stdout and "prime-agent" in done.stdout
    assert "1 found" in done.stdout.replace("\n", " ")


def test_invalid_profile_is_rejected_before_scanning(tmp_path: Path) -> None:
    home = build_home(tmp_path / "home")
    done = run(home, "--json", "metrics", "census", "--profile", "../etc", tmp_path=tmp_path)
    assert done.returncode == 2
    assert json.loads(done.stdout)["ok"] is False
