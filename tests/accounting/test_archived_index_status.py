"""The human index status must retain the archive count exposed by JSON status."""

from __future__ import annotations

import json
import os
import subprocess

from ssgrep.services import api
from tests.accounting.test_savings_cli import BINARY
from tests.pipeline.test_app import fake_models as fake_models, write_claude_session


def test_archived_source_count_is_visible_in_real_cli_status() -> None:
    source = write_claude_session("archived-status")
    api.index()
    assert api.status().archived_source_count == 0
    source.unlink()
    archived = api.index(full_reprocess=True)
    assert archived.archived_source_count == 1
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "120"}
    machine = subprocess.run(
        [str(BINARY), "status", "--json"],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert machine.returncode == 0, machine.stderr
    data = json.loads(machine.stdout)["data"]
    assert data["archived_source_count"] == 1
    assert data["session_count"] == 1 and data["episode_count"] == 1
    human = subprocess.run(
        [str(BINARY), "status"],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert human.returncode == 0, human.stderr
    archive_row = next(line for line in human.stdout.splitlines() if "Archived" in line)
    assert "1 sources" in archive_row
    assert api.status().archived_source_count == 1
