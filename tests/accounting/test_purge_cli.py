"""``ssgrep metrics purge`` through the real installed binary, on state a real collector wrote.

Every run is a subprocess with an isolated HOME and ``SSGREP_DATA_DIR``; confirmation prompts are
exercised on a real pseudo-terminal. Trees are compared byte-for-byte including mtimes.
"""

from __future__ import annotations

import json
import os
import pty
import subprocess
from pathlib import Path

import pytest

from ssgrep.accounting import purge_files
from ssgrep.accounting.purge_model import (
    LocalDeleted,
    Obligation,
    PurgeReport,
    TombstoneLine,
    WorkerView,
)
from ssgrep.cli import purge_render
from tests.accounting.collector_scene import accounting_dir
from tests.accounting.test_collect_cli import hold_lease, release
from tests.accounting.test_purge import (
    ALPHA,
    CUTOFF,
    build_scene,
    data_root,
    evidence_by_session,
    generation_names,
    periods,
    tree,
)
from tests.accounting.test_savings_cli import BINARY, Run, document
from tests.accounting.test_savings_render import flat

ANSI = "\x1b"
TIERS = (
    "1. LOCAL DELETION",
    "2. DELETION REQUESTS QUEUED",
    "3. ACCEPTED BY THE COLLECTOR",
    "4. ERASED FROM THE BACKEND",
)
DOCUMENT_KEYS = {"schema_version", "state", "data", "warnings", "next_actions", "provenance"}
DATA_KEYS = {
    "profile",
    "performed",
    "scope",
    "local_deleted",
    "queued_tombstones",
    "accepted_by_collector",
    "backend_erased",
    "operator_obligations",
    "non_reversible_metrics",
    "retained",
    "worker",
}


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def interactive(run: Run, answer: bytes, *args: str) -> tuple[int, str, str]:
    """Run on a real pseudo-terminal stdin and feed ``answer`` to the prompt."""
    master, slave = pty.openpty()
    environment = {
        **os.environ,
        "HOME": str(run.home),
        "SSGREP_DATA_DIR": str(run.data),
        "COLUMNS": "80",
        "NO_COLOR": "1",
    }
    process = subprocess.Popen(
        [str(BINARY), "metrics", "purge", *args],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    os.write(master, answer)
    stdout, stderr = process.communicate(timeout=120)
    os.close(master)
    os.close(slave)
    return process.returncode, stdout, stderr


# ---- validation and confirmation: exit 2 before any side effect -------------------------------


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize(
    ("args", "condition"),
    [
        ((), "invalid_interval"),
        (("--before", "yesterday"), "invalid_interval"),
        (("--before", "2026-02-30"), "invalid_interval"),
        (("--before", CUTOFF, "--profile", "../etc"), "invalid_profile"),
        (("--before", CUTOFF, "--workspace", "relative/dir"), "invalid_workspace"),
        (("--before", CUTOFF, "--workspace", "/"), "invalid_workspace"),
    ],
)
def test_bad_arguments_exit_2_and_leave_the_data_tree_untouched(
    run: Run, tmp_path: Path, args: tuple[str, ...], condition: str, as_json: bool
) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    result = run("metrics", "purge", *args, "--yes", *(("--json",) if as_json else ()))
    assert result.returncode == 2, result.stderr
    if as_json:
        assert document(result)["condition"] == condition
    else:
        assert result.stdout == "" and result.stderr.strip()
    assert tree(data_root(tmp_path)) == before


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_missing_confirmation_exits_2_before_side_effects(
    run: Run, tmp_path: Path, as_json: bool
) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    result = run("metrics", "purge", "--before", CUTOFF, *(("--json",) if as_json else ()))
    assert result.returncode == 2
    if as_json:
        body = document(result)
        assert body["condition"] == "confirmation_required" and body["ok"] is False
    else:
        assert "--yes" in result.stderr and "--dry-run" in result.stderr
    assert tree(data_root(tmp_path)) == before
    assert periods() == {"2026-10-05", "2026-10-06", "2026-10-15"}


def test_a_json_run_never_prompts_even_on_a_terminal_and_needs_yes(
    run: Run, tmp_path: Path
) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    code, stdout, _ = interactive(run, b"y\n", "--before", CUTOFF, "--json")
    assert code == 2 and json.loads(stdout.splitlines()[-1])["condition"] == "confirmation_required"
    assert tree(data_root(tmp_path)) == before


# ---- dry run ----------------------------------------------------------------------------------


def test_dry_run_on_an_empty_data_dir_exits_0_and_creates_nothing(run: Run) -> None:
    for extra in ((), ("--json",)):
        result = run("metrics", "purge", "--before", CUTOFF, "--dry-run", *extra)
        assert result.returncode == 0, result.stderr
    assert run.files() == []
    body = document(run("metrics", "purge", "--before", CUTOFF, "--dry-run", "--json"))
    assert body["data"]["state"] == "nothing_to_purge"


def test_dry_run_changes_no_byte_and_describes_the_four_tiers(run: Run, tmp_path: Path) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    result = run("metrics", "purge", "--before", CUTOFF, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert tree(data_root(tmp_path)) == before
    for tier in TIERS:
        assert tier in result.stdout
    assert "Nothing was changed" in result.stdout and "Would delete" in result.stdout
    assert "unknown operator responsibility" in result.stdout
    assert "Source transcripts were not touched" in result.stdout
    assert periods() == {"2026-10-05", "2026-10-06", "2026-10-15"}


@pytest.mark.parametrize("columns", [40, 80, 120])
def test_output_wraps_at_every_width_without_ansi_or_stray_control_bytes(
    run: Run, tmp_path: Path, columns: int
) -> None:
    build_scene(tmp_path)
    result = run("metrics", "purge", "--before", CUTOFF, "--dry-run", columns=columns)
    assert result.returncode == 0
    assert ANSI not in result.stdout
    assert max(len(line) for line in result.stdout.splitlines()) <= columns


def test_a_hostile_workspace_name_is_escaped_in_human_output(run: Run, tmp_path: Path) -> None:
    hostile = f"/work/{ANSI}[2Jevil\nforged line"
    result = run("metrics", "purge", "--before", CUTOFF, "--workspace", hostile, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert ANSI not in result.stdout and "forged line" in result.stdout
    assert "\nforged line" not in result.stdout


# ---- real runs --------------------------------------------------------------------------------


def test_yes_purges_through_the_installed_cli_and_json_has_the_documented_shape(
    run: Run, tmp_path: Path
) -> None:
    build_scene(tmp_path)
    result = run("metrics", "purge", "--before", CUTOFF, "--yes", "--json")
    assert result.returncode == 0, result.stderr
    outer = document(result)
    assert outer["ok"] is True
    body = outer["data"]
    assert set(body) == DOCUMENT_KEYS and body["state"] == "purged"
    data = body["data"]
    assert set(data) == DATA_KEYS and data["performed"] is True
    assert data["backend_erased"] == "unknown_operator_responsibility"
    assert data["accepted_by_collector"]["count"] == 0
    assert data["scope"] == {
        "before": f"{CUTOFF}T00:00:00Z",
        "cutoff_day": CUTOFF,
        "workspace": None,
    }
    assert data["local_deleted"]["ledger_period_rows"] == 2
    assert data["local_deleted"]["snapshot_generations"] == 1
    assert "backend_erasure_is_operator_responsibility" in body["warnings"]
    assert {a["command"] for a in body["next_actions"]} >= {"ssgrep metrics status"}
    assert not any(isinstance(v, float) for v in _flat(data))
    assert periods() == {"2026-10-15"} and set(evidence_by_session()) == {"a2", "src-a2"}
    assert generation_names() == [g for g in generation_names() if g.startswith("gen-0000000002-")]
    assert not (accounting_dir() / "collector" / purge_files.MARKER).exists()
    # a second invocation is a no-op report and exit 0
    again = document(run("metrics", "purge", "--before", CUTOFF, "--yes", "--json"))
    assert again["data"]["state"] == "nothing_to_purge"


def test_workspace_scoped_yes_run_keeps_the_other_workspace(run: Run, tmp_path: Path) -> None:
    build_scene(tmp_path)
    result = run("metrics", "purge", "--before", CUTOFF, "--workspace", ALPHA, "--yes")
    assert result.returncode == 0, result.stderr
    assert "Purge complete." in result.stdout and f"workspace {ALPHA}" in result.stdout
    assert periods() == {"2026-10-06", "2026-10-15"}


def test_the_interactive_prompt_cancel_changes_nothing_and_confirm_purges(
    run: Run, tmp_path: Path
) -> None:
    build_scene(tmp_path)
    before = tree(data_root(tmp_path))
    code, stdout, stderr = interactive(run, b"n\n", "--before", CUTOFF)
    assert code == 0 and "Cancelled. Nothing was changed." in stderr
    assert "Purge preview" in stdout  # the plan was shown before asking
    assert tree(data_root(tmp_path)) == before

    code, stdout, stderr = interactive(run, b"y\n", "--before", CUTOFF)
    assert code == 0, stderr
    assert "Purge complete." in stdout
    assert periods() == {"2026-10-15"}


# ---- refusals ---------------------------------------------------------------------------------


def test_a_live_worker_makes_a_real_purge_exit_1_already_running_and_changes_nothing(
    run: Run, tmp_path: Path
) -> None:
    build_scene(tmp_path)
    holder = hold_lease(run)
    try:
        before = tree(data_root(tmp_path))
        result = run("metrics", "purge", "--before", CUTOFF, "--yes", "--json")
        assert result.returncode == 1
        assert document(result)["condition"] == "already_running"
        assert tree(data_root(tmp_path)) == before
        preview = run("metrics", "purge", "--before", CUTOFF, "--dry-run")
        assert preview.returncode == 0 and "collect is running" in preview.stdout
        assert tree(data_root(tmp_path)) == before
    finally:
        release(holder)
    assert run("metrics", "purge", "--before", CUTOFF, "--yes").returncode == 0


def test_a_symlinked_collector_directory_exits_1_storage_unsafe_and_deletes_nothing_outside(
    run: Run, tmp_path: Path
) -> None:
    build_scene(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "evidence").mkdir(parents=True)
    victim = elsewhere / "evidence" / "seg-victim.jsonl"
    victim.write_bytes(b"not ours\n")
    real = accounting_dir() / "collector"
    real.rename(tmp_path / "moved")
    real.symlink_to(elsewhere)
    before, outside = tree(data_root(tmp_path)), tree(elsewhere)
    for extra in (("--dry-run",), ("--yes",)):
        result = run("metrics", "purge", "--before", CUTOFF, *extra, "--json")
        assert result.returncode == 1, result.stderr
        failure = document(result)
        assert failure["condition"] == "storage_unsafe" and str(tmp_path) not in result.stdout
    assert tree(data_root(tmp_path)) == before and tree(elsewhere) == outside
    assert victim.read_bytes() == b"not ours\n"


def test_sentinel_files_survive_a_real_cli_purge_byte_for_byte(run: Run, tmp_path: Path) -> None:
    from tests.accounting.test_purge import _sentinels

    build_scene(tmp_path)
    sentinels = _sentinels(tmp_path)
    result = run("metrics", "purge", "--before", CUTOFF, "--yes")
    assert result.returncode == 0, result.stderr
    for path, body in sentinels.items():
        assert path.read_bytes() == body
    assert periods() == {"2026-10-15"}


def _flat(value: object):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flat(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flat(item)
    else:
        yield value


@pytest.mark.parametrize("performed", [False, True])
def test_purge_json_and_text_separate_local_queued_and_operator_erasure(performed: bool):
    report = PurgeReport(
        "acme",
        performed,
        "2026-10-01T00:00:00Z",
        "2026-10-01",
        "/w/team",
        LocalDeleted(receipt_files=13),
        tombstones=(
            TombstoneLine("allowed", 17, 2, "queued" if performed else "would_queue", None),
            TombstoneLine("blocked", 19, 0, "not_permitted", "export_disabled"),
        ),
        obligations=(Obligation("blocked", "export_disabled", 19, ("private-id",)),),
        non_reversible_destinations=("aggregate",),
        pending_retention_days=11,
        worker=WorkerView(True, "serve") if not performed else None,
    )
    report_doc = purge_render.report_document(report, "acme")
    data = report_doc["data"]
    assert isinstance(data, dict)
    assert data["local_deleted"]["receipt_files"] == 13
    assert data["queued_tombstones"]["records"] == 17
    assert data["accepted_by_collector"]["count"] == 0
    assert data["backend_erased"] == "unknown_operator_responsibility"
    warnings = report_doc["warnings"]
    actions = report_doc["next_actions"]
    assert isinstance(warnings, list) and isinstance(actions, list)
    assert "operator_must_erase_exported_records" in warnings
    assert "aggregate_metrics_cannot_be_recalled" in warnings
    assert ("worker_running" in warnings) is (not performed)
    assert all(action["command"].endswith("--profile acme") for action in actions)
    text = flat(purge_render.render_report(report, 80).plain)
    assert "13" in text and "allowed: 17 record(s)" in text and "blocked: 19 record(s)" in text
    assert "11 days" in text and "aggregate" in text and "operator responsibility" in text
    assert "private-id" not in text
    assert "blocked: 19 exported record(s)" in text
