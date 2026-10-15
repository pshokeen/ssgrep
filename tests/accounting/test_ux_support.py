"""Support for the UX suites (``test_ux_*``): a REAL collected dataset and a real-binary runner.

``collected_scene`` writes native Claude transcripts and journal events, then runs the real
collector so receipts, history, ledger state and the snapshot are exactly what production
writes. Numeric tests explicitly install ``current_rarity_index``; absent-index tests do not.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.test_savings_cli import BINARY

CLOCK = scene.at(hour=9)
NOW = scene.at(hour=20)
PAIRS = (0, 1, 2)


def collected_scene(root: Path, pairs: tuple[int, ...] = PAIRS) -> None:
    """Capture on, ``pairs`` of fact-source + consumer sessions, one published generation."""
    vs.enable()
    vs.start_capture()
    for n in pairs:
        vs.pair(vs.projects(root), n, CLOCK)
    result = scene.collect(NOW)
    assert result.state == "published", result


def current_rarity_index(root: Path, monkeypatch) -> Path:
    """A named synthetic current-index arithmetic fixture, not production calibration.

    Real SQL metadata supplies the existing read-only repository seam. Every
    path/code has a guaranteed singleton lexical component among 511 authorized
    chunks: log2((511+1)/(1+1)) = 8 bits. The collector/reducer are not replaced.
    Only tests explicitly requesting numeric all-band evidence install this.
    """
    from ssgrep.accounting import index_frequency
    from tests.accounting.test_frequency_generations import MetadataRepository, database

    target = root / "synthetic-ux-index.sqlite"
    documents = 511
    database(target, documents)
    terms = {name: 1 for name in vs.NAMES}
    terms.update({vs.code(n).lower(): 1 for n in range(len(vs.NAMES))})
    terms["fixture_filler"] = documents - 1
    with sqlite3.connect(target) as connection:
        connection.execute("UPDATE chunks SET cwd='/work/demo'")
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='last_index_time'",
            ("synthetic-ux-arithmetic-v1",),
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='lexical_stats'",
            (
                json.dumps(
                    {
                        "n_docs": documents,
                        "df": terms,
                        "avgdl": (2 * len(vs.NAMES) + documents - 1) / documents,
                    }
                ),
            ),
        )
    connection.close()
    original = index_frequency.read_index_view

    def current(profile, **kwargs):
        repository = MetadataRepository(target)
        try:
            return original(profile, repository=repository, **kwargs)
        finally:
            repository.close()

    monkeypatch.setattr(index_frequency, "read_index_view", current)
    return target


def current_fact_consumer() -> list[dict]:
    """The existing consumer shape with vocabulary authorized by ``current_rarity_index``."""
    from tests.accounting.reduce_builders import card, results_json
    from tests.accounting.test_reduce_evidence import consumer

    return consumer(
        action=f"vim {vs.path(0)}",
        body=results_json(card("claude:src1:ep:0", vs.fact(0))),
    )


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
