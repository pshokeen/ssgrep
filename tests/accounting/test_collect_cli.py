"""``metrics collect`` / ``metrics serve``: real subprocess runs of the installed binary.

Validation, pause and disabled paths never import the collector, so those tests run
without it; flows that actually collect or serve skip with a named reason until
``ssgrep.accounting.collector`` is importable.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import snapshots
from ssgrep.accounting.collector_types import CollectOptions, CollectResult, CycleReport
from ssgrep.accounting.lease import Lease, inspect_lease
from ssgrep.cli.collect_json import (
    collect_data,
    collect_document,
    owner_view,
    plain,
    resume_command,
)
from ssgrep.cli.collect_render import render_collect, render_running
from tests.accounting.readers_fixtures import (
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    write_records,
)
from tests.accounting.snapshot_builders import write_snapshot
from tests.accounting.test_savings_cli import BINARY, Run, document

PROFILE = "personal-local"
HAS_COLLECTOR = importlib.util.find_spec("ssgrep.accounting.collector") is not None
needs_collector = pytest.mark.skipif(
    not HAS_COLLECTOR, reason="ssgrep.accounting.collector (collect_once) is not importable yet"
)
ANSI = "\x1b"


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def tree(root: Path) -> dict[str, str]:
    """Every path under ``root`` with its mode and content digest (byte-for-byte)."""
    if not root.exists():
        return {}
    out = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        mode = oct(path.stat().st_mode & 0o7777)
        if path.is_file():
            out[key] = f"{mode} {hashlib.sha256(path.read_bytes()).hexdigest()}"
        else:
            out[key] = f"{mode} dir"
    return out


def enable(run: Run, profile: str = PROFILE) -> None:
    result = run("metrics", "setup", "--yes", "--profile", profile)
    assert result.returncode == 0, result.stderr


def pause(run: Run) -> None:
    assert run("metrics", "pause").returncode == 0


def accounting(run: Run, *parts: str) -> Path:
    return run.data.joinpath("accounting", PROFILE, *parts)


# ---------------------------------------------------------------- validation (exit 2)

MATRIX = [
    (("--backfill",), "backfill_requires_interval"),
    (("--backfill", "--since", "2026-10-01"), "backfill_requires_interval"),
    (("--backfill", "--until", "2026-10-02"), "backfill_requires_interval"),
    (("--since", "2026-10-01"), "interval_requires_mode"),
    (("--until", "2026-10-02"), "interval_requires_mode"),
    (("--new-epoch", "--since", "2026-10-01", "--until", "2026-10-02"), "interval_requires_mode"),
    (
        ("--backfill", "--recompute", "--since", "2026-10-01", "--until", "2026-10-02"),
        "conflicting_modes",
    ),
    (("--recompute", "--since", "yesterday"), "invalid_interval"),
    (("--recompute", "--since", "2026-10-02", "--until", "2026-10-01"), "invalid_interval"),
    (("--backfill", "--since", "2026-10-02", "--until", "2026-10-02"), "invalid_interval"),
    (("--profile", "../etc"), "invalid_profile"),
    (("--profile", "A/B"), "invalid_profile"),
    (("--profile", f"{ANSI}[31mred"), "invalid_profile"),
]


@pytest.mark.parametrize("prepared", [False, True], ids=["pristine", "enabled-and-paused"])
@pytest.mark.parametrize(("args", "condition"), MATRIX)
def test_validation_exits_2_in_json_and_text_and_leaves_the_data_dir_untouched(
    run: Run, args: tuple[str, ...], condition: str, prepared: bool
) -> None:
    if prepared:  # validation outranks the pause check: still exit 2, not exit 1
        enable(run)
        pause(run)
    before = tree(run.data)
    as_json = run("metrics", "collect", *args, "--json")
    assert as_json.returncode == 2, as_json.stdout
    error = document(as_json)
    assert error["ok"] is False and error["condition"] == condition
    assert error["message"] and error["command"].startswith("ssgrep ")
    as_text = run("metrics", "collect", *args)
    assert as_text.returncode == 2 and as_text.stdout == ""
    assert error["command"] in as_text.stderr and "Try:" in as_text.stderr
    assert ANSI not in as_text.stderr + as_json.stdout
    assert tree(run.data) == before
    if not prepared:
        assert not run.data.exists()


def test_valid_interval_with_mode_passes_validation_up_to_the_disabled_check(run: Run) -> None:
    for args in (
        ("--backfill", "--since", "2026-10-01", "--until", "2026-10-02"),
        ("--recompute", "--since", "2026-10-01T00:00:00+02:00"),
        ("--new-epoch",),
        (),
    ):
        result = run("metrics", "collect", *args, "--json")
        assert result.returncode == 1, (args, result.stdout)
        assert document(result)["condition"] == "collection_disabled"
    assert not run.data.exists()


# ------------------------------------------------------------- pause / disabled (exit 1)


def test_paused_profile_fails_with_collection_paused_and_creates_nothing(run: Run) -> None:
    enable(run)
    pause(run)
    before = tree(run.data)
    result = run("metrics", "collect", "--json")
    assert result.returncode == 1
    error = document(result)
    assert error["ok"] is False and error["condition"] == "collection_paused"
    assert error["command"] == "ssgrep metrics resume --profile personal-local"
    assert tree(run.data) == before
    for name in ("worker", "collector", "snapshots"):
        assert not accounting(run, name).exists()
    text = run("metrics", "collect", "--new-epoch")
    assert text.returncode == 1 and "paused" in text.stderr and text.stdout == ""
    assert tree(run.data) == before


def test_paused_profile_keeps_its_snapshot_readable_and_unchanged(run: Run) -> None:
    snapshot = write_snapshot(run.data)
    enable(run)
    pause(run)
    before = tree(run.data)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert run("metrics", "collect", "--json").returncode == 1
    backfill = run(
        "metrics", "collect", "--backfill", "--since", "2026-10-01", "--until", "2026-10-02"
    )
    assert backfill.returncode == 1 and "paused" in backfill.stderr
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
    assert tree(run.data) == before
    summary = run("savings", "--json")
    assert summary.returncode == 0
    assert document(summary)["data"]["provenance"]["snapshot_id"] == "a" * 16


def test_pause_applies_per_profile(run: Run) -> None:
    enable(run)
    pause(run)
    paused = run("metrics", "collect", "--json")
    assert document(paused)["condition"] == "collection_paused"
    other = run("metrics", "collect", "--profile", "acme", "--json")  # never enabled, not paused
    assert other.returncode == 1 and document(other)["condition"] == "collection_disabled"


def test_disabled_profile_fails_with_collection_disabled_and_creates_nothing(run: Run) -> None:
    result = run("metrics", "collect", "--json")
    assert result.returncode == 1
    assert document(result) == {
        "ok": False,
        "condition": "collection_disabled",
        "message": "Local measurement is not enabled for this profile; nothing was collected.",
        "command": "ssgrep metrics setup",
    }
    text = run("metrics", "collect", "--profile", "acme")
    assert text.returncode == 1 and "ssgrep metrics setup --profile acme" in text.stderr
    assert not run.data.exists()


# -------------------------------------------------------------------- already running


def hold_lease(run: Run, profile: str = PROFILE) -> subprocess.Popen[str]:
    code = (
        "import sys\n"
        "from ssgrep.accounting.lease import Lease\n"
        "lease = Lease.acquire(sys.argv[1], 'collect')\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
    )
    environment = {**os.environ, "HOME": str(run.home), "SSGREP_DATA_DIR": str(run.data)}
    holder = subprocess.Popen(
        [sys.executable, "-c", code, profile],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
    return holder


def release(holder: subprocess.Popen[str]) -> None:
    assert holder.stdin is not None
    holder.stdin.write("\n")
    holder.stdin.flush()
    assert holder.wait(timeout=30) == 0


def write_progress(run: Run, pid: int, **overrides: object) -> Path:
    body = {
        "schema": 1,
        "profile": PROFILE,
        "pid": pid,
        "kind": "collect",
        "phase": "reading",
        "mode": "live",
        "cycle_started_at": "2026-10-13T08:00:00Z",
        "updated_at": "2026-10-13T08:00:05Z",
        "cycles": 3,
        "bytes_read": 4_194_304,
        "sources_done": 7,
        "sources_total": 19,
        "lag_bytes": 1_048_576,
        "last_cycle": None,
        **overrides,
    }
    target = accounting(run, "worker", "progress.json")
    target.write_text(json.dumps(body))
    target.chmod(0o600)
    return target


@needs_collector
def test_second_collect_reports_the_running_owner_with_exit_0_and_changes_nothing(
    run: Run,
) -> None:
    enable(run)
    write_snapshot(run.data)
    holder = hold_lease(run)
    try:
        write_progress(run, holder.pid)
        before = tree(accounting(run, "snapshots"))
        as_json = run("metrics", "collect", "--json")
        assert as_json.returncode == 0, as_json.stderr
        body = document(as_json)["data"]
        assert body["state"] == "already_running"
        owner = body["data"]["owner"]
        assert owner["pid"] == holder.pid and owner["kind"] == "collect"
        assert owner["alive"] is True and owner["phase"] == "reading"
        assert (owner["sources_done"], owner["sources_total"]) == (7, 19)
        assert (owner["bytes_read"], owner["lag_bytes"]) == (4_194_304, 1_048_576)
        assert body["data"]["generation"] is None and body["data"]["snapshot_id"] is None
        assert body["next_actions"][0]["command"] == "ssgrep metrics status"
        text = run("metrics", "collect", columns=40)
        assert text.returncode == 0 and f"pid {holder.pid}" in text.stdout
        assert "7 of 19" in text.stdout and "4.0 MiB" in text.stdout
        assert "ssgrep metrics status" in text.stdout and ANSI not in text.stdout
        assert max(len(line) for line in text.stdout.splitlines()) <= 40
        assert tree(accounting(run, "snapshots")) == before
    finally:
        release(holder)
    assert Lease.acquire(PROFILE, "collect") is not None


# ------------------------------------------------ pure documents/renderers (no collector)


def result_of(state: str = "published", mode: str = "live", **fields: Any) -> CollectResult:
    report = CycleReport(
        status="published",
        generation=4,
        snapshot_id="0123456789abcdef0123456789abcdef",
        complete=state == "published",
        stopped_by="complete" if state == "published" else "byte_budget",
        bytes_read=33_554_432,
        lag_bytes=0 if state == "published" else 7_340_032,
        new_evidence=12,
        sources=3,
        elapsed_ms=1500,
    )
    return CollectResult(state=state, mode=mode, report=report, epoch=2, **fields)


def test_json_documents_carry_only_ints_strings_bools_and_null() -> None:
    options = CollectOptions(
        since="2026-09-01T00:00:00Z", until="2026-10-01T00:00:00Z", backfill=True, new_epoch=True
    )
    result = result_of(
        "partial",
        "backfill",
        backfill={"since": "a", "until": "b", "complete": False, "counts": [["x", 3]], "f": 1.5},
        unavailable_periods=("2026-09-01=evidence_expired", "2026-09-02"),
    )
    document_, data, actions = collect_document(PROFILE, options, result)
    encoded = json.dumps(document_)
    assert document_["state"] == "partial" and data["generation"] == 4
    assert data["backfill"]["f"] is None  # floats dropped, never forwarded
    assert data["unavailable_periods"] == [
        {"period": "2026-09-01", "reason": "evidence_expired"},
        {"period": "2026-09-02", "reason": "evidence_expired"},
    ]
    assert data["new_epoch"] is True and data["epoch"] == 2
    assert [a.command for a in actions] == [
        "ssgrep metrics collect --backfill --since 2026-09-01T00:00:00Z"
        " --until 2026-10-01T00:00:00Z",
        "ssgrep savings",
    ]
    assert "--new-epoch" not in actions[0].command  # resuming never re-declares a new epoch
    assert json.loads(encoded) == document_
    assert _no_floats(document_)


def _no_floats(value: object) -> bool:
    if isinstance(value, float):
        return False
    if isinstance(value, dict):
        return all(_no_floats(v) for v in value.values())
    if isinstance(value, list):
        return all(_no_floats(v) for v in value)
    return True


def test_plain_drops_floats_objects_and_deep_nesting() -> None:
    assert plain({"a": 1, "b": 1.5, "c": object(), "d": [1, "x", None, True]}) == {
        "a": 1,
        "b": None,
        "c": None,
        "d": [1, "x", None, True],
    }
    deep: object = 1
    for _ in range(10):
        deep = [deep]
    assert plain(deep) == [[[[None]]]]  # bounded depth: the rest collapses to null


def test_resume_command_preserves_profile_and_mode() -> None:
    live = CollectOptions()
    assert resume_command(PROFILE, live) == "ssgrep metrics collect"
    assert resume_command("acme", live) == "ssgrep metrics collect --profile acme"
    recompute = CollectOptions(since="2026-09-01T00:00:00Z", recompute=True)
    assert resume_command("acme", recompute) == (
        "ssgrep metrics collect --recompute --since 2026-09-01T00:00:00Z --profile acme"
    )


@pytest.mark.parametrize("width", [40, 80])
def test_published_partial_and_backfill_render_the_required_facts(width: int) -> None:
    published = result_of()
    options = CollectOptions(new_epoch=True)
    data = collect_data(PROFILE, options, published)
    document_, _, actions = collect_document(PROFILE, options, published)
    text = render_collect(data, (), actions, width).plain
    assert "Collection complete." in text and "0123456789ab" in text
    assert "0123456789abcdef" not in text  # short id only
    assert "32.0 MiB" in text and "Generation" in text and "4" in text
    assert "New counter epoch 2 declared" in text and "ssgrep savings" in text
    assert "run again to resume" not in text
    assert max(len(line) for line in text.splitlines()) <= width

    partial = result_of("partial")
    data = collect_data("acme", CollectOptions(), partial)
    _, _, actions = collect_document("acme", CollectOptions(), partial)
    text = render_collect(data, (), actions, width).plain
    assert "Stopped by byte budget -- run again to resume." in " ".join(text.split())
    assert "7.0 MiB" in text and "ssgrep savings --profile acme" in text
    assert "ssgrep metrics collect --profile acme" in " ".join(text.split())

    backfill = result_of(
        "published", "backfill", backfill={"complete": True, "counts": [["search", 9]]}
    )
    options = CollectOptions(
        since="2026-09-01T00:00:00Z", until="2026-10-01T00:00:00Z", backfill=True
    )
    data = collect_data(PROFILE, options, backfill)
    _, _, actions = collect_document(PROFILE, options, backfill)
    text = render_collect(data, (), actions, width).plain
    assert "HISTORICAL BACKFILL (not live executions)" in " ".join(text.split())
    assert "search" in text and "9" in text

    recompute = result_of(
        "published", "recompute", unavailable_periods=("2026-08-30=evidence_expired",)
    )
    data = collect_data(PROFILE, CollectOptions(recompute=True), recompute)
    text = render_collect(data, (), (), width).plain
    assert "2026-08-30" in text and "evidence expired" in text and "RECOMPUTE" in text


def test_hostile_text_in_warnings_and_owner_is_escaped_in_human_output() -> None:
    data = collect_data(PROFILE, CollectOptions(), result_of())
    text = render_collect(data, (f"{ANSI}[31mbad\x07",), (), 80).plain
    assert ANSI not in text and "\x07" not in text and "\\x1b" in text
    running = {
        "state": "already_running",
        "profile": PROFILE,
        "owner": {
            "pid": 42,
            "kind": f"{ANSI}[2Jx",
            "phase": "rea\nding",
            "sources_done": None,
            "sources_total": None,
            "bytes_read": None,
            "lag_bytes": None,
            "updated_at": None,
        },
    }
    shown = render_running(running, (), 40).plain
    assert ANSI not in shown and "unknown" in shown and "\\n" in shown


def test_owner_view_ignores_progress_that_belongs_to_another_process(run: Run) -> None:
    with Lease.acquire(PROFILE, "collect"):
        pid = os.getpid()
        base: dict[str, object] = {
            "pid": pid,
            "phase": "reading",
            "sources_done": 2,
            "sources_total": 5,
            "bytes_read": 10,
            "lag_bytes": 3,
            "cycles": True,
            "updated_at": 1.5,
        }
        mine = owner_view(PROFILE, base)
        assert mine["pid"] == pid and mine["kind"] == "collect" and mine["alive"] is True
        assert (mine["sources_done"], mine["sources_total"], mine["bytes_read"]) == (2, 5, 10)
        assert mine["cycles"] is None and mine["updated_at"] is None  # bool / float rejected
        stale = owner_view(PROFILE, {**base, "pid": pid + 1})
        assert stale["pid"] == pid and stale["phase"] is None and stale["sources_done"] is None
        assert owner_view(PROFILE, None)["phase"] is None


@needs_collector
def test_serve_startup_line_states_cadence_and_auth_file_path(run: Run) -> None:
    enable(run)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    process = start_serve(run, "--port", str(port))
    try:
        banner = read_serve_startup(process, port)
        token_path = accounting(run, "endpoint.token")
        token = token_path.read_text().strip()
        assert f"http://127.0.0.1:{port}" in banner
        assert "Profile: personal-local" in banner
        assert str(token_path) in banner.replace("\n", "").replace(" ", "")
        assert "60 s" in banner and "300 s" in banner
        assert token and token not in banner
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 0, err
    assert "Worker stopped." in out
    assert token not in banner + out + err and "Traceback" not in err


# ---------------------------------------------------------------------- flows (collector)


def stamp(offset_seconds: int = 0) -> str:
    moment = datetime.now(UTC) + timedelta(seconds=offset_seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def capture_one_search(run: Run) -> None:
    """A real ``ssgrep search``: with no index it fails, which is still a captured execution."""
    result = run("search", "find the notes")
    assert result.returncode == 4, result.stderr  # missing index; the journal event exists anyway


def claude_transcript(tmp_path: Path, session: str = "s1", extra_bytes: int = 0) -> Path:
    """A native Claude transcript in the isolated CLAUDE_CONFIG_DIR, dated now (after capture)."""
    target = tmp_path / "claude" / "projects" / "-work-demo" / f"{session}.jsonl"
    records = [
        claude_prompt(session, "find the notes", ts=stamp(0)),
        claude_tool_use(
            session, "toolu_1", "mcp__ssgrep__search_sessions", {"query": "notes"}, ts=stamp(0)
        ),
        claude_tool_result(session, "toolu_1", [{"type": "text", "text": "[]"}], ts=stamp(1)),
    ]
    write_records(target, records)
    if extra_bytes:
        fill = claude_prompt(session, "x" * 80, uuid="u-fill", ts=stamp(0))
        line = (json.dumps(fill) + "\n").encode()
        with target.open("ab") as handle:
            for _ in range(extra_bytes // len(line) + 1):
                handle.write(line)
    return target


def published_generation(run: Run, generation: int) -> snapshots.SnapshotEnvelope:
    found = sorted(accounting(run, "snapshots").glob(f"gen-{generation:010d}-*.json"))
    assert len(found) == 1, sorted(p.name for p in accounting(run, "snapshots").iterdir())
    envelope = snapshots.decode(found[0].read_bytes())
    assert envelope.generation == generation and found[0].name.endswith(
        f"-{envelope.snapshot_id}.json"
    )
    return envelope


def is_current(run: Run, generation: int) -> bool:
    current = snapshots.decode(accounting(run, "snapshots", "current.json").read_bytes())
    return current.generation == generation


@needs_collector
def test_first_collect_publishes_a_generation_that_savings_reads(run: Run, tmp_path: Path) -> None:
    enable(run)
    capture_one_search(run)
    claude_transcript(tmp_path)
    result = run("metrics", "collect", "--json")
    assert result.returncode == 0, result.stderr
    body = document(result)["data"]
    data = body["data"]
    assert body["state"] == "published" and data["mode"] == "live" and data["epoch"] >= 1
    assert data["generation"] >= 1 and data["stopped_by"] == "complete" and data["bytes_read"] > 0
    envelope = published_generation(run, data["generation"])
    assert envelope.snapshot_id == data["snapshot_id"] and is_current(run, data["generation"])
    assert ANSI not in result.stdout and _no_floats(json.loads(result.stdout.splitlines()[-1]))
    summary = document(run("savings", "--json"))["data"]
    assert summary["provenance"]["snapshot_id"] == data["snapshot_id"]
    for columns in (40, 80):
        text = run("metrics", "collect", columns=columns)
        assert text.returncode == 0 and "Collection complete." in text.stdout
        assert "ssgrep savings" in text.stdout and ANSI not in text.stdout
        assert max(len(line) for line in text.stdout.splitlines()) <= columns
        assert str(tmp_path) not in text.stdout + text.stderr  # never a transcript path


@needs_collector
def test_byte_budget_stops_partial_and_the_next_run_resumes_to_published(
    run: Run, tmp_path: Path
) -> None:
    enable(run)
    capture_one_search(run)
    claude_transcript(tmp_path, extra_bytes=40 * 1024 * 1024)
    first = run("metrics", "collect", "--json")
    assert first.returncode == 0, first.stderr
    one = document(first)["data"]
    assert one["state"] == "partial" and one["data"]["stopped_by"] == "byte_budget"
    assert one["data"]["lag_bytes"] > 0 and one["data"]["complete"] is False
    assert published_generation(run, one["data"]["generation"])
    text = run("metrics", "collect")
    assert text.returncode == 0
    second = document(run("metrics", "collect", "--json"))["data"]
    assert second["state"] == "published" and second["data"]["lag_bytes"] == 0
    assert second["data"]["generation"] > one["data"]["generation"]


@needs_collector
def test_new_epoch_starts_a_fresh_counter_epoch_and_keeps_history(run: Run, tmp_path: Path) -> None:
    enable(run)
    capture_one_search(run)
    claude_transcript(tmp_path)
    first = document(run("metrics", "collect", "--json"))["data"]["data"]
    second = document(run("metrics", "collect", "--new-epoch", "--json"))["data"]
    assert second["state"] == "published" and second["data"]["new_epoch"] is True
    assert second["data"]["epoch"] == first["epoch"] + 1
    assert second["data"]["generation"] > first["generation"]
    published_generation(run, first["generation"])  # earlier generation is immutable history
    text = run("metrics", "collect", "--new-epoch")
    assert text.returncode == 0 and "New counter epoch" in " ".join(text.stdout.split())


@needs_collector
def test_backfill_is_a_separate_series_not_live_executions(run: Run, tmp_path: Path) -> None:
    enable(run)
    capture_one_search(run)
    claude_transcript(tmp_path)
    live = document(run("metrics", "collect", "--json"))["data"]["data"]
    args = ("--backfill", "--since", "2026-09-01", "--until", "2026-11-01")
    result = run("metrics", "collect", *args, "--json")
    assert result.returncode == 0, result.stderr
    body = document(result)["data"]
    assert body["data"]["mode"] == "backfill" and body["data"]["backfill"] is not None
    assert body["data"]["interval"] == {
        "since": "2026-09-01T00:00:00Z",
        "until": "2026-11-01T00:00:00Z",
    }
    after = published_generation(run, body["data"]["generation"])
    series: Any = after.observations["backfill"]
    assert series and series[0]["since"] == "2026-09-01T00:00:00Z"
    assert body["data"]["epoch"] == live["epoch"]  # backfill never starts a new epoch
    text = run("metrics", "collect", *args)
    assert text.returncode == 0
    assert "HISTORICAL BACKFILL (not live executions)" in " ".join(text.stdout.split())


@needs_collector
def test_recompute_reports_unavailable_periods_with_reasons(run: Run, tmp_path: Path) -> None:
    enable(run)
    capture_one_search(run)
    claude_transcript(tmp_path)
    assert run("metrics", "collect").returncode == 0
    result = run("metrics", "collect", "--recompute", "--since", "2020-01-01", "--json")
    assert result.returncode == 0, result.stderr
    data = document(result)["data"]["data"]
    assert data["mode"] == "recompute"
    for period in data["unavailable_periods"]:
        assert period["period"] and period["reason"]
    text = run("metrics", "collect", "--recompute", "--since", "2020-01-01")
    assert text.returncode == 0 and "RECOMPUTE" in text.stdout
    assert not (tmp_path / "data" / "lancedb").exists()  # never rebuilds the search index


# ------------------------------------------------------------------------------- serve


def start_serve(run: Run, *extra: str) -> subprocess.Popen[str]:
    environment = {
        **os.environ,
        "HOME": str(run.home),
        "SSGREP_DATA_DIR": str(run.data),
        "NO_COLOR": "1",
        "COLUMNS": "80",
    }
    return subprocess.Popen(
        [str(BINARY), "metrics", "serve", *extra],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def read_serve_startup(process: subprocess.Popen[str], port: int) -> str:
    assert process.stdout is not None
    url = f"http://127.0.0.1:{port}"
    banner = ""
    deadline = time.monotonic() + 30
    timer = threading.Timer(30, process.send_signal, args=(signal.SIGTERM,))
    timer.daemon = True
    timer.start()
    try:
        ready, _, _ = select.select([process.stdout], [], [], max(0.0, deadline - time.monotonic()))
        assert ready, f"serve did not announce its startup: {banner!r}"
        # The text stream may buffer the remaining startup lines after the first read.
        while "Foreground only:" not in banner:
            line = process.stdout.readline()
            assert line, f"serve exited before startup (status {process.poll()})"
            banner += line
    finally:
        timer.cancel()
    assert url in banner
    return banner


def get_http(port: int, path: str, token: str | None = None) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


def assert_http_port_stopped(port: int) -> None:
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
    except OSError:
        return
    connection.close()
    pytest.fail(f"metrics HTTP listener is still accepting connections on {port}")


def wait_for(predicate, timeout: float = 30.0, what: str = "condition") -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def lease_alive_for(pid: int) -> bool:
    info = inspect_lease(PROFILE)
    return info is not None and info.pid == pid and info.alive and info.kind == "serve"


def listening_sockets(pid: int) -> str | None:
    if shutil.which("lsof") is None:
        return None
    result = subprocess.run(
        ["lsof", "-nP", "-a", "-p", str(pid), "-i"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.stdout


def test_lsof_probe_detects_a_listening_socket_when_one_exists() -> None:
    """Positive twin of the 'binds no port' assertion: the probe really sees listeners."""
    if shutil.which("lsof") is None:
        pytest.skip("lsof is not installed: cannot probe sockets of the serve process")
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert "LISTEN" in (listening_sockets(os.getpid()) or "")


@needs_collector
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_serve_holds_the_lease_serves_loopback_http_and_stops_cleanly(
    run: Run, sig: signal.Signals
) -> None:
    enable(run)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    process = start_serve(run, "--port", str(port))
    token_path = accounting(run, "endpoint.token")
    try:
        banner = read_serve_startup(process, port)
        token = token_path.read_text().strip()
        assert f"http://127.0.0.1:{port}" in banner
        assert "personal-local" in banner
        assert str(token_path) in banner.replace("\n", "").replace(" ", "")
        assert "60 s" in banner and "300 s" in banner
        assert token and token not in banner
        assert token_path.stat().st_mode & 0o777 == 0o600
        assert get_http(port, "/healthz")[0] == 200
        assert get_http(port, "/metrics")[0] == 401
        status, content_type, body = get_http(port, "/metrics", token)
        assert status == 200
        assert content_type == "application/openmetrics-text; version=1.0.0; charset=utf-8"
        assert body.endswith(b"# EOF\n")

        wait_for(lambda: lease_alive_for(process.pid), what="lease held by serve")
        progress = accounting(run, "worker", "progress.json")
        wait_for(progress.is_file, what="progress.json after first cycle")
        recorded = json.loads(progress.read_text())
        assert recorded["pid"] == process.pid and recorded["kind"] == "serve"
        sockets = listening_sockets(process.pid)
        if sockets is not None:
            assert "LISTEN" in sockets and str(port) in sockets and "UDP" not in sockets
        with pytest.raises(Exception, match="held|Running|running"):
            Lease.acquire(PROFILE, "collect")
        process.send_signal(sig)
        out, err = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 0, err
    assert f"http://127.0.0.1:{port}" in out
    assert str(token_path) in out.replace("\n", "").replace(" ", "")
    assert "Worker stopped." in out and "lease is released" in " ".join(out.split())
    assert ANSI not in banner + out and "Traceback" not in err and token not in banner + out + err
    assert_http_port_stopped(port)
    with Lease.acquire(PROFILE, "collect"):
        pass

    second = start_serve(run, "--port", str(port))
    try:
        second_banner = read_serve_startup(second, port)
        assert f"http://127.0.0.1:{port}" in second_banner
        assert get_http(port, "/healthz")[0] == 200
        wait_for(lambda: lease_alive_for(second.pid), what="lease held by rebound serve")
        second.send_signal(signal.SIGTERM)
        second_out, second_err = second.communicate(timeout=60)
    finally:
        if second.poll() is None:
            second.kill()
            second.wait()
    assert second.returncode == 0, second_err
    assert "Worker stopped." in second_out and "Traceback" not in second_err
    assert token not in second_banner + second_out + second_err
    assert_http_port_stopped(port)
    with Lease.acquire(PROFILE, "collect"):
        pass
    assert not [p for p in run.home.rglob("*") if p.suffix in {".plist", ".service", ".timer"}]
    assert not (run.home / "Library").exists() and not (run.home / ".config" / "systemd").exists()


@needs_collector
def test_second_serve_exits_1_already_running_and_leaves_the_first_alone(run: Run) -> None:
    enable(run)
    first = start_serve(run)
    try:
        wait_for(lambda: lease_alive_for(first.pid), what="first serve holding the lease")
        second = run("metrics", "serve", "--json")
        assert second.returncode == 1, second.stdout
        error = document(second)
        assert error["condition"] == "already_running" and str(first.pid) in error["message"]
        assert error["command"] == "ssgrep metrics status"
        text = run("metrics", "serve", "--profile", PROFILE)
        assert text.returncode == 1 and str(first.pid) in text.stderr
        assert first.poll() is None and lease_alive_for(first.pid)
        first.send_signal(signal.SIGTERM)
        assert first.wait(timeout=60) == 0
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()


def test_serve_rejects_a_hostile_profile_before_starting_anything(run: Run) -> None:
    for bad in ("../x", f"{ANSI}[31m", "A/B"):
        result = run("metrics", "serve", "--profile", bad, "--json")
        assert result.returncode == 2 and document(result)["condition"] == "invalid_profile"
        text = run("metrics", "serve", "--profile", bad)
        assert text.returncode == 2 and ANSI not in text.stdout + text.stderr
    assert not run.data.exists()


def test_serve_and_collect_never_import_an_http_stack_or_the_mcp_server() -> None:
    code = (
        "import sys\n"
        "import ssgrep.cli.commands.metrics_serve_command\n"
        "import ssgrep.cli.commands.metrics_collect_command\n"
        "names = ('http.server', 'socketserver', 'ssgrep.services.mcp_server')\n"
        "bad = [n for n in names if n in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stderr
