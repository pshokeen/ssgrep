# ---- Survivor source: test_status.py ----
"""``metrics status`` worker/lag/missing/storage: real files, real lease holders, real CLI."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.collect import collector as collector_progress
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.storage import data_root
from ssgrep.accounting.serve import status_worker
from ssgrep.accounting.serve.lease import Lease, process_start_token
from ssgrep.accounting.serve.status_worker import Watermark
from ssgrep.accounting.views.service import read_status
from ssgrep.accounting.views.types import DeliveryLine
from ssgrep.cli.metrics_json import status_document
from ssgrep.cli.metrics_render import render_status
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.views.snapshot_builders import COLLECTED_AT, enable_policy, write_snapshot
from tests.accounting.views.test_savings_commands import Run, document, flat

PROFILE = "personal-local"
NOW = datetime(2026, 10, 12, 12, 5, 0, tzinfo=UTC)
NOON = datetime(2026, 10, 12, 12, tzinfo=UTC)
SCENE_NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)
ROOT = Path(__file__).resolve().parents[3]
MIB = 1024 * 1024
HOSTILE = "\x1b[2J[red]boom[/red]\nfake line: all clear"

LAST_CYCLE = {
    "finished_at": "2026-10-12T11:59:00Z",
    "status": "published",
    "stopped_by": "byte_budget",
    "error": None,
    "generation": 7,
    "elapsed_ms": 1500,
    "bytes_read": 4096,
    "complete": False,
}

MARKS = [
    {
        "source": "0123456789abcdef",
        "runtime": "claude",
        "offset": 100,
        "size": 1100,
        "lag_bytes": 1000,
        "pending_tail_bytes": 0,
        "status": "ok",
    },
    {
        "source": "89abcdef01234567",
        "runtime": "codex",
        "offset": 500,
        "size": 500,
        "lag_bytes": 0,
        "pending_tail_bytes": 12,
        "status": "ok",
    },
    {
        "source": "fedcba9876543210",
        "runtime": "claude",
        "offset": 0,
        "size": 7_339_032,
        "lag_bytes": 7_339_032,
        "pending_tail_bytes": 0,
        "status": "missing",
    },
]

COVERAGE = {
    "complete": False,
    "partial": True,
    "capture_completeness": "unknown",
    "sources": {
        "total": 5,
        "complete": 2,
        "lag_bytes": 7_340_032,
        "pending_tail_bytes": 12,
        "missing": 1,
        "baseline_skipped_bytes": 0,
        "by_runtime": {
            "claude": {
                "sources": 3,
                "processed_bytes": 100,
                "total_bytes": 7_340_232,
                "lag_bytes": 7_340_032,
                "missing": 1,
            },
            "codex": {
                "sources": 2,
                "processed_bytes": 500,
                "total_bytes": 500,
                "lag_bytes": 0,
                "missing": 0,
            },
        },
    },
    "watermarks": MARKS,
    "gaps": {"oversized_record": 2, "malformed_json": 4, "journal_incomplete_tail": 1},
    "journal": {"segments": 3, "incomplete_tails": 1, "quarantined": 6, "conflicts": 2},
    "stopped_by": "byte_budget",
    "lag_bytes": 7_340_032,
    "retention": {
        "evidence_expired_before": "2026-09-01",
        "aggregates_expired_before": None,
        "frozen_roots": 3,
    },
}


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def progress(**overrides: Any) -> dict[str, Any]:
    document_: dict[str, Any] = {
        "schema": 1,
        "profile": PROFILE,
        "pid": os.getpid(),
        "kind": "collect",
        "phase": "reading",
        "mode": "live",
        "cycle_started_at": "2026-10-12T12:04:00Z",
        "updated_at": "2026-10-12T12:04:30Z",
        "cycles": 4,
        "bytes_read": MIB,
        "sources_done": 3,
        "sources_total": 5,
        "lag_bytes": 7_340_032,
        "last_cycle": LAST_CYCLE,
    }
    document_.update(overrides)
    return document_


def put(rel: str, data: bytes | str, *, mode: int = 0o600) -> Path:
    """A private file under ``<data>/accounting/<profile>/`` in 0700 directories."""
    payload = data.encode() if isinstance(data, str) else data
    target = data_root() / "accounting" / PROFILE / rel
    current = data_root()
    current.mkdir(exist_ok=True, mode=0o700)
    current.chmod(0o700)
    for part in target.relative_to(data_root()).parts[:-1]:
        current = current / part
        current.mkdir(exist_ok=True, mode=0o700)
        current.chmod(0o700)
    target.write_bytes(payload)
    target.chmod(mode)
    return target


def put_progress(**overrides: Any) -> Path:
    return put("worker/progress.json", json.dumps(progress(**overrides)))


def put_snapshot(coverage: dict[str, Any] | None = None, **overrides: Any) -> None:
    """The standard coverage with ``overrides`` applied; ``coverage=`` replaces it wholesale."""
    write_snapshot(
        data_root(), coverage={**COVERAGE, **overrides} if coverage is None else coverage
    )


def tree(root: Path) -> dict[str, tuple[int, int, int, bytes | None]]:
    """Every entry's inode, mtime and (for files) bytes: any write, touch or replace shows."""
    if not root.exists():
        return {}
    entries = {}
    for entry in sorted(root.rglob("*")):
        info = entry.lstat()
        body = entry.read_bytes() if entry.is_file() and not entry.is_symlink() else None
        entries[str(entry.relative_to(root))] = (info.st_ino, info.st_mtime_ns, info.st_size, body)
    return entries


def dead_owner() -> tuple[int, str]:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    token = process_start_token(child.pid)
    child.kill()
    child.wait()
    assert token is not None
    return child.pid, token


def put_lease(pid: int, token: str | None, kind: str = "serve") -> None:
    record = {
        "pid": pid,
        "process_start": token,
        "kind": kind,
        "acquired_at": "2026-10-12T11:00:00Z",
    }
    put("worker/lease.lock", json.dumps(record))


@contextmanager
def live_holder(kind: str = "serve") -> Iterator[int]:
    """A real process holding the worker lease (flock) until the block ends."""
    script = (
        "import sys\nfrom ssgrep.accounting.serve.lease import Lease\n"
        f"lease = Lease.acquire({PROFILE!r}, {kind!r})\n"
        "print('ready', flush=True)\nsys.stdin.readline()\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
        yield holder.pid
    finally:
        holder.kill()
        holder.wait()


def status_data(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    return document(result)["data"]["data"]


# --- 1. fresh profile ----------------------------------------------------------------------


def test_fresh_profile_reports_unknowns_with_reasons_and_creates_nothing(run: Run) -> None:
    result = run("savings", "status", "--json")
    data = status_data(result)
    assert data["worker"]["state"] == "never_run" and data["worker"]["reason"] == "no_worker_record"
    assert data["lag"]["lag_bytes"] is None and data["lag"]["reason"] == "no_snapshot_yet"
    assert data["watermarks"]["count"] is None and data["watermarks"]["items"] is None
    assert data["watermarks"]["reason"] == "no_snapshot_yet"
    assert data["missing_evidence"]["missing_sources"] is None
    assert data["missing_evidence"]["gaps"] is None
    assert data["missing_evidence"]["reason"] == "no_snapshot_yet"
    assert data["missing_evidence"]["capture_completeness"] == "unknown"
    storage = data["storage_pressure"]
    assert storage["total_bytes"] is None and storage["reason"] == "not_created_yet"
    assert set(storage["components"]) == {
        "journal",
        "collector",
        "snapshots",
        "receipts",
        "history",
    }
    assert all(value is None for value in storage["components"].values())
    assert run.files() == []
    text = run("savings", "status")
    assert text.returncode == 0 and "never run" in text.stdout and "unknown" in text.stdout
    assert run.files() == []


def test_status_with_policy_only_creates_no_worker_dir_and_takes_no_lock() -> None:
    enable_policy()
    before = tree(data_root())
    view = read_status(PROFILE, now=NOW)
    assert view.worker.state == "never_run"
    assert tree(data_root()) == before and not (data_root() / "accounting" / PROFILE).exists()


def test_status_never_needs_the_flock_and_leaves_it_free() -> None:
    import fcntl

    put_snapshot()
    put_lease(os.getpid(), process_start_token(os.getpid()))
    lock = data_root() / "accounting" / PROFILE / "worker" / "lease.lock"
    holder = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a "live worker" holding the lock
        assert (
            read_status(PROFILE, now=NOW).worker.state == "running"
        )  # did not block, did not fail
    finally:
        os.close(holder)
    other = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # status left nothing locked behind
    finally:
        os.close(other)


# --- 2. exact numbers: facade and real CLI -------------------------------------------------


def test_read_status_reports_exact_lag_watermarks_gaps_and_stop_reason() -> None:
    put_snapshot()
    put_progress()
    view = read_status(PROFILE, now=NOW)
    assert view.lag.lag_bytes == 7_340_032 and view.lag.basis == "snapshot"
    assert view.lag.sources_total == 5 and view.lag.sources_behind == 2
    assert view.lag.stopped_by == "byte_budget"
    assert [(r.runtime, r.lag_bytes, r.sources) for r in view.lag.by_runtime] == [
        ("claude", 7_340_032, 3),
        ("codex", 0, 2),
    ]
    assert view.lag.watermark_count == 3 and view.lag.watermarks is not None
    assert [m.source for m in view.lag.watermarks] == sorted(m["source"] for m in MARKS)
    assert view.missing.missing_sources == 1
    assert dict(view.missing.gaps or ()) == COVERAGE["gaps"]
    assert (view.missing.incomplete_tails, view.missing.quarantined) == (1, 6)
    assert view.missing.conflicts == 2 and view.missing.frozen_roots == 3
    assert view.missing.evidence_expired_before == "2026-09-01"
    assert view.missing.capture_completeness == "unknown"
    assert view.worker.state == "idle" and view.worker.progress is not None
    assert view.worker.progress.cycles == 4


def test_cli_json_has_exact_numbers_and_no_paths(run: Run) -> None:
    put_snapshot()
    put_progress()
    before = run.files()
    result = run("savings", "status", "--json")
    data = status_data(result)
    assert data["lag"]["lag_bytes"] == 7_340_032 and data["lag"]["sources_behind"] == 2
    assert data["lag"]["stopped_by"] == "byte_budget" and data["lag"]["basis"] == "snapshot"
    assert data["lag"]["by_runtime"][0] == {
        "runtime": "claude",
        "sources": 3,
        "processed_bytes": 100,
        "total_bytes": 7_340_232,
        "lag_bytes": 7_340_032,
        "missing": 1,
    }
    assert data["watermarks"]["count"] == 3 and data["watermarks"]["truncated"] is False
    first = data["watermarks"]["items"][0]
    assert first == {**MARKS[0]} and set(first) == set(MARKS[0])
    assert [m["source"] for m in data["watermarks"]["items"]] == sorted(m["source"] for m in MARKS)
    missing = data["missing_evidence"]
    assert missing["missing_sources"] == 1 and missing["gaps"] == COVERAGE["gaps"]
    assert missing["journal"] == {"incomplete_tails": 1, "quarantined": 6, "conflicts": 2}
    assert missing["evidence_expired_before"] == "2026-09-01" and missing["frozen_roots"] == 3
    assert missing["reason"] is None and missing["capture_completeness"] == "unknown"
    worker = data["worker"]
    assert worker["state"] == "idle" and worker["cycles"] == 4
    assert worker["last_cycle"]["stopped_by"] == "byte_budget"
    assert str(run.data) not in result.stdout and str(run.home) not in result.stdout
    assert "\\/" not in result.stdout and "progress.json" not in result.stdout
    assert run.files() == before


def test_unknown_coverage_pieces_are_null_never_zero() -> None:
    put_snapshot(
        coverage={"sources": {"total": 2}, "watermarks": "garbage", "gaps": None, "retention": 1}
    )
    document_: Any = status_document(read_status(PROFILE, now=NOW))["data"]  # type: ignore[index]
    assert document_["lag"]["lag_bytes"] is None and document_["lag"]["sources_behind"] is None
    assert document_["lag"]["sources_total"] == 2
    assert document_["watermarks"]["count"] is None and document_["watermarks"]["items"] is None
    missing = document_["missing_evidence"]
    assert missing["missing_sources"] is None and missing["gaps"] is None
    assert missing["journal"] == {"incomplete_tails": None, "quarantined": None, "conflicts": None}
    assert missing["frozen_roots"] is None and missing["reason"] == "coverage_not_recorded"


def test_lag_from_progress_when_no_snapshot_exists_yet() -> None:
    put_progress(last_cycle=None)
    view = read_status(PROFILE, now=NOW)
    assert (view.lag.lag_bytes, view.lag.basis) == (7_340_032, "progress")
    assert view.missing.missing_sources is None  # progress does not describe missing sources
    assert view.worker.state == "never_run" and view.worker.reason == "no_cycle_finished"


def test_malformed_watermarks_are_counted_as_ignored_not_trusted() -> None:
    bad = [
        {**MARKS[0], "source": "/home/me/secret/path.jsonl"},  # a path is not a source hash
        {**MARKS[0], "offset": -1},
        {**MARKS[0], "lag_bytes": True},
        {**MARKS[0], "status": "who knows"},
        "nonsense",
    ]
    put_snapshot(watermarks=[MARKS[1], *bad])
    lag = read_status(PROFILE, now=NOW).lag
    assert lag.watermark_count == 1 and lag.ignored_entries == 5 and lag.sources_behind == 0


# --- 3. worker states ----------------------------------------------------------------------


def test_live_lease_holder_is_running_with_its_pid_and_progress(run: Run) -> None:
    put_snapshot()
    with live_holder("serve") as pid:
        put_progress(pid=pid, kind="serve")
        view = read_status(PROFILE, now=NOW)
        worker = status_data(run("savings", "status", "--json"))["worker"]
        text = render_status(view, 80).plain
    assert (view.worker.state, view.worker.pid, view.worker.kind) == ("running", pid, "serve")
    assert view.worker.recoverable is False and view.worker.acquired_at is not None
    assert (worker["state"], worker["pid"], worker["kind"]) == ("running", pid, "serve")
    assert (worker["phase"], worker["mode"], worker["cycles"]) == ("reading", "live", 4)
    assert (worker["sources_done"], worker["sources_total"]) == (3, 5)
    assert worker["bytes_read"] == MIB
    assert f"running (serve, pid {pid})" in text and "3 of 5 sources" in text
    assert "1,048,576 B" in text


def test_progress_from_another_process_is_not_attributed_to_the_live_owner() -> None:
    with live_holder("collect") as pid:
        put_progress(pid=pid + 1, cycles=99)
        view = read_status(PROFILE, now=NOW)
    assert view.worker.state == "running" and view.worker.pid == pid
    assert view.worker.progress is None and view.worker.reason == "progress_not_from_this_worker"


def test_dead_owner_without_flock_holder_is_a_recoverable_stale_lease(run: Run) -> None:
    put_snapshot()
    pid, token = dead_owner()
    put_lease(pid, token, "collect")
    view = read_status(PROFILE, now=NOW)
    assert (view.worker.state, view.worker.pid, view.worker.recoverable) == (
        "stale_lease",
        pid,
        True,
    )
    assert view.worker.kind == "collect" and view.worker.reason == "owner_not_running"
    worker = status_data(run("savings", "status", "--json"))["worker"]
    assert (worker["state"], worker["pid"], worker["recoverable"]) == ("stale_lease", pid, True)
    text = render_status(view, 80).plain
    assert "stale lease" in text and "abandoned" in text and "takes it over" in text


def test_pid_reuse_is_not_a_live_owner() -> None:
    put_lease(os.getpid(), "Thu Jan  1 00:00:00 1970")
    worker = read_status(PROFILE, now=NOW).worker
    assert worker.state == "stale_lease" and worker.pid == os.getpid()


def test_orderly_release_with_a_published_cycle_is_idle() -> None:
    put_snapshot()
    put_progress()
    Lease.acquire(PROFILE, "collect").release()  # lock file exists, owner fields cleared
    worker = read_status(PROFILE, now=NOW).worker
    assert worker.state == "idle" and worker.pid is None and worker.reason is None


def test_failed_last_cycle_shows_only_the_stable_condition(run: Run) -> None:
    failed = {**LAST_CYCLE, "status": "failed", "error": "collection_failed", "generation": None}
    put_snapshot()
    put_progress(last_cycle=failed)
    worker = status_data(run("savings", "status", "--json"))["worker"]
    assert worker["state"] == "failed_last_cycle" and worker["error"] == "collection_failed"
    assert worker["last_cycle"]["status"] == "failed"
    assert "collection failed" in render_status(read_status(PROFILE, now=NOW), 80).plain


@pytest.mark.parametrize(
    "payload",
    [b"{not json", b"[1, 2]", b'{"schema": 2}', b'{"schema": 1, "cycles": 1.5}', b"x" * 70_000],
)
def test_corrupt_or_foreign_progress_is_unknown_with_a_reason(payload: bytes) -> None:
    put("worker/progress.json", payload)
    worker = read_status(PROFILE, now=NOW).worker
    assert worker.state == "unknown" and worker.reason in {
        "progress_unreadable",
        "progress_unsupported_schema",
    }


def test_group_writable_progress_is_not_trusted_and_status_still_succeeds(run: Run) -> None:
    path = put("worker/progress.json", json.dumps(progress()))
    path.chmod(0o666)
    worker = status_data(run("savings", "status", "--json"))["worker"]
    assert worker["state"] == "unknown" and worker["reason"] == "progress_unreadable"


def test_snapshot_without_any_progress_is_unknown_not_never_run() -> None:
    put_snapshot()
    worker = read_status(PROFILE, now=NOW).worker
    assert (worker.state, worker.reason) == ("unknown", "no_progress_record")


# --- 4. storage ----------------------------------------------------------------------------


def build_storage_tree() -> None:
    put("journal/seg-1.jsonl", b"j" * 1000)
    put("journal/nested/seg-2.jsonl", b"j" * 234)
    put("journal-quarantine/q.jsonl", b"q" * 66)
    put("collector/checkpoint.json", b"c" * 300)
    put("collector/evidence/e1.jsonl", b"e" * 700)
    put("snapshots/gen-000000000001.json", b"s" * 3000)
    put("snapshots/gen-000000000002.json", b"s" * 1000)
    put("receipts/rcpt-1.json", b"r" * 55)
    put("history/periods.json", b"h" * 77)
    put("worker/progress.json", b"{}")  # worker state is not a storage component


def test_component_bytes_match_real_file_sizes() -> None:
    build_storage_tree()
    pressure = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 50 * MIB * 1000)
    sizes = {c.name: c.bytes for c in pressure.components}
    assert sizes == {
        "journal": 1000 + 234 + 66,
        "collector": 1000,
        "snapshots": 4000,
        "receipts": 55,
        "history": 77,
    }
    assert pressure.total_bytes == 6432 and pressure.reason is None
    assert pressure.level == "ok" and pressure.free_bytes == 50 * MIB * 1000


def test_symlinks_are_not_followed_or_counted_and_regular_twin_is_counted(tmp_path: Path) -> None:
    build_storage_tree()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "huge.bin").write_bytes(b"x" * 5000)
    base = data_root() / "accounting" / PROFILE
    (base / "receipts" / "link.json").symlink_to(outside / "huge.bin")
    (base / "history" / "dirlink").symlink_to(outside, target_is_directory=True)
    pressure = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    sizes = {c.name: c.bytes for c in pressure.components}
    assert sizes["receipts"] == 55 and sizes["history"] == 77
    (base / "receipts" / "regular.json").write_bytes(b"x" * 5000)  # same size, regular file
    again = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert {c.name: c.bytes for c in again.components}["receipts"] == 55 + 5000


def test_a_symlinked_component_directory_is_not_walked(tmp_path: Path) -> None:
    build_storage_tree()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 9000)
    base = data_root() / "accounting" / PROFILE
    (base / "history" / "periods.json").unlink()
    (base / "history").rmdir()
    (base / "history").symlink_to(outside, target_is_directory=True)
    sizes = {
        c.name: c.bytes
        for c in status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12).components
    }
    assert sizes["history"] == 0 and sizes["snapshots"] == 4000


@pytest.mark.parametrize(
    ("free", "level", "reason"),
    [
        (100 * MIB - 1, "critical", "free_space_below_100_mib"),
        (100 * MIB, "high", "free_space_below_1_gib"),
        (1024 * MIB - 1, "high", "free_space_below_1_gib"),
        (1024 * MIB, "ok", None),
    ],
)
def test_free_space_thresholds_are_exact(free: int, level: str, reason: str | None) -> None:
    build_storage_tree()
    pressure = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: free)
    assert (pressure.level, pressure.level_reason, pressure.free_bytes) == (level, reason, free)


def test_large_accounting_total_raises_pressure_even_with_free_space() -> None:
    build_storage_tree()
    big = data_root() / "accounting" / PROFILE / "collector" / "evidence" / "big.jsonl"
    with big.open("wb") as handle:
        handle.truncate(1024 * MIB)  # sparse: the size counts, the disk does not
    plenty = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**13)
    assert plenty.total_bytes == 1024 * MIB + 6432 and plenty.level == "high"
    assert plenty.level_reason == "accounting_data_above_1_gib"
    with big.open("r+b") as handle:
        handle.truncate(1024 * MIB - 6432)  # exactly 1 GiB in total: not above the threshold
    edge = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**13)
    assert edge.total_bytes == 1024 * MIB and edge.level == "ok"


def test_unreadable_free_space_and_walk_limit_are_unknown_not_zero() -> None:
    build_storage_tree()

    def broken(_: Path) -> int:
        raise OSError("no such volume")

    unknown_free = status_worker.storage_pressure(PROFILE, free_bytes_fn=broken)
    assert unknown_free.level is None and unknown_free.level_reason == "free_space_unknown"
    assert unknown_free.free_bytes is None and unknown_free.total_bytes == 6432
    limited = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12, entry_limit=4)
    assert limited.total_bytes is None and limited.reason == "walk_limit_exceeded"
    assert any(c.bytes is None for c in limited.components)


def test_default_free_bytes_use_the_real_data_volume() -> None:
    build_storage_tree()
    real = status_worker.storage_pressure(PROFILE)
    assert (
        real.free_bytes is not None
        and real.free_bytes > 0
        and real.level in {"ok", "high", "critical"}
    )


# --- 5. read-only: nothing changes, age semantics stay -------------------------------------


def test_repeated_reads_change_no_bytes_mtimes_or_inodes_and_age_follows_collected_at() -> None:
    enable_policy()
    build_storage_tree()
    put_snapshot()
    put_progress()
    put_lease(os.getpid(), process_start_token(os.getpid()))
    before = tree(data_root())
    collected = datetime.strptime(COLLECTED_AT, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    ages = []
    for step in range(3):
        view = read_status(PROFILE, now=collected + timedelta(seconds=100 * (step + 1)))
        ages.append(view.age_seconds)
        render_status(view, 40)
        status_document(view)
    assert ages == [100, 200, 300]
    assert tree(data_root()) == before


# --- 6. rendering --------------------------------------------------------------------------


@pytest.mark.parametrize("width", [40, 80])
def test_rendered_text_has_the_real_numbers_and_no_paths(width: int, tmp_path: Path) -> None:
    build_storage_tree()
    put_snapshot()
    put_progress()
    snapshot_bytes = (
        4000 + (data_root() / "accounting" / PROFILE / "snapshots/current.json").stat().st_size
    )
    view = read_status(PROFILE, now=NOW)
    text = render_status(view, width).plain
    assert max(len(line) for line in text.splitlines()) <= max(width, 40)
    flat = " ".join(text.replace(":", " ").split())  # narrow widths stack "label: value"
    for needle in (
        "7,340,032 B",  # lag bytes
        "2 sources behind",
        "stopped by byte budget",
        "Watermarks 3",
        "oversized record",
        "malformed json",
        "journal incomplete tail",
        "Missing sources 1",
        "incomplete tails 1, quarantined 6, conflicts 2",
        "2026-09-01",
        "unknown: a finished snapshot does not prove everything was captured",
        "journal 1,300 B",
        f"snapshots {snapshot_bytes:,} B",
        "receipts 55 B",
        "history 77 B",
    ):
        assert " ".join(needle.replace(":", " ").split()) in flat, needle
    assert str(tmp_path) not in text and "/accounting/" not in text and ".jsonl" not in text
    assert "0123456789abcdef" in text  # source hashes appear, never source paths


def test_hostile_strings_in_progress_lease_and_snapshot_are_sanitized(run: Run) -> None:
    hostile_cycle = {**LAST_CYCLE, "status": "failed", "error": HOSTILE, "stopped_by": HOSTILE}
    put_snapshot(gaps={HOSTILE: 3, "malformed_json": 1}, stopped_by=HOSTILE)
    put_progress(phase=HOSTILE, mode=HOSTILE, kind=HOSTILE, last_cycle=hostile_cycle)
    put_lease(os.getpid(), "Thu Jan  1 00:00:00 1970", kind=HOSTILE)
    for width in (40, 80):
        view = read_status(PROFILE, now=NOW)
        text = render_status(view, width).plain
        assert "\x1b" not in text and "[red]" not in text and "fake line" not in text
        assert "boom" not in text
    result = run("savings", "status", "--json")
    data = status_data(result)
    assert "boom" not in result.stdout and "fake line" not in result.stdout
    assert data["worker"]["phase"] == "unrecognized" and data["worker"]["mode"] == "unrecognized"
    assert data["worker"]["kind"] == "unrecognized"
    assert data["worker"]["last_cycle"]["error"] == "unrecognized_error"
    assert data["missing_evidence"]["gaps"] == {"malformed_json": 1}  # hostile reason dropped
    assert data["lag"]["stopped_by"] == "unrecognized"
    plain = run("savings", "status")
    assert plain.returncode == 0 and "\x1b" not in plain.stdout and "boom" not in plain.stdout


# --- 7. real collector state: unsafe, unreadable and racing storage; live progress ----------


def seeded(root: Path) -> None:
    """One creditable pair collected into generation 1 (counter total 2)."""
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(SCENE_NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def seeded_pending(root: Path) -> None:
    """Two creditable pairs on disk, nothing collected yet."""
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)


class Holder:
    """A real worker process parked at a hook step with the lease held."""

    def __init__(self, root: Path, step: str, *, now: datetime = SCENE_NOW) -> None:
        self.hold = root / f"hold-{step}"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.accounting.collect.verify_driver",
                "collect",
                step,
                now.isoformat(),
                "--hold",
                str(self.hold),
            ],
            env=vs.env_for(root),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 120
        while not (self.hold / "reached").exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"holder never parked: {self.process.communicate()}")
            time.sleep(0.02)

    def release(self) -> dict[str, Any]:
        (self.hold / "release").write_text("go")
        out, err = self.process.communicate(timeout=120)
        assert self.process.returncode == 0, err[-2000:]
        return json.loads(out.splitlines()[-1])

    def kill(self) -> None:
        self.process.send_signal(signal.SIGKILL)
        self.process.communicate(timeout=60)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


def Holder_backfill(root: Path) -> Holder:
    holder = Holder.__new__(Holder)
    holder.hold = root / "hold-backfill"
    holder.process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.accounting.collect.verify_driver",
            "collect",
            "evidence_written",
            SCENE_NOW.isoformat(),
            "--hold",
            str(holder.hold),
            "--backfill",
            "2026-10-01T00:00:00Z",
            "2026-11-01T00:00:00Z",
        ],
        env=vs.env_for(root),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    while not (holder.hold / "reached").exists():
        if holder.process.poll() is not None or time.monotonic() > deadline:
            raise AssertionError(f"holder never parked: {holder.process.communicate()}")
        time.sleep(0.02)
    return holder


def count(progress: dict[str, object], key: str) -> int:
    """One integer progress field of the running owner (fails loudly if it is not one)."""
    value = progress[key]
    assert isinstance(value, int), (key, value)
    return value


def test_status_keeps_watermark_count_distinct_from_visible_rows_and_delivery_age():
    from tests.accounting.export.test_consent import DEST

    base = read_status(now=NOON)
    marks = tuple(Watermark(f"{i:016x}", "omp", 100, 120, 20, 0, "ready") for i in range(10))
    lag = replace(base.lag, lag_bytes=200, watermark_count=10, watermarks=marks)
    line = DeliveryLine(
        DEST,
        "logs",
        True,
        "acknowledged",
        "collector accepted",
        last_acked_at="2026-10-10T12:00:00Z",
    )
    view = replace(
        base,
        lag=lag,
        delivery_lines=(line,),
        delivery_age_seconds=172800,
        policy_error="invalid\x1b[2Jpolicy",
    )
    text = flat(render_status(view, 80).plain)
    assert "200 B" in text and "2d ago" in text
    assert "2026-10-10T12:00:00Z" in text
    assert marks[7].source in text and marks[8].source not in text
    assert "2 more" in text and "--json" in text
    assert "\x1b" not in text and "\\x1b[2J" in text
    unknown = replace(view, lag=replace(lag, watermark_count=None, watermarks=None))
    text = flat(render_status(unknown, 80).plain)
    assert "200 B" in text and "Watermarks unknown" in text


def test_status_counts_all_admitted_watermarks_but_bounds_display_and_raw_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(status_worker, "MAX_WATERMARKS", 2)
    monkeypatch.setattr(status_worker, "MAX_RAW_WATERMARKS", 4)
    marks = [
        {
            "source": f"{n:016x}",
            "runtime": "claude",
            "offset": 10,
            "size": 12,
            "lag_bytes": 2,
            "pending_tail_bytes": 0,
            "status": "ok",
        }
        for n in (4, 3, 2, 1, 0)
    ]
    marks[2] = {**marks[2], "offset": -1}
    coverage = {"lag_bytes": 10, "watermarks": marks, "stopped_by": "stop_requested"}
    lag = status_worker.lag_status(coverage, None, None)
    assert lag.lag_bytes == 10 and lag.basis == "snapshot"
    assert lag.sources_behind == lag.watermark_count == 3
    assert lag.watermarks_truncated and lag.ignored_entries == 2
    assert lag.watermarks is not None
    assert [m.source for m in lag.watermarks] == [f"{n:016x}" for n in (1, 3)]
    assert lag.stopped_by == "stop_requested"
    complete = status_worker.lag_status({"lag_bytes": 0, "watermarks": []}, None, None)
    assert complete.watermarks == () and complete.watermark_count == complete.sources_behind == 0
    unknown = status_worker.lag_status({}, "no_snapshot", None)
    assert unknown.lag_bytes is None and unknown.watermarks is None
    assert unknown.sources_behind is None and unknown.reason == "no_snapshot"


def test_unreadable_lease_does_not_turn_last_success_into_an_idle_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    progress, problem = status_worker.read_progress(PROFILE)
    assert progress is not None and problem is None
    assert status_worker.worker_status(PROFILE, progress, None, has_snapshot=True).state == "idle"

    def unreadable(_profile: str) -> Any:
        raise PermissionError("private path must not escape status")

    monkeypatch.setattr(status_worker, "inspect_lease", unreadable)
    state = status_worker.worker_status(PROFILE, progress, None, has_snapshot=True)
    assert (state.state, state.reason) == ("unknown", "lease_unreadable")
    assert state.progress == progress and state.pid is None


def test_progress_write_failure_does_not_fail_a_real_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "worker" / collector_progress.PROGRESS_FILE
    previous = path.read_bytes()
    original = storage.atomic_write
    failed_writes: list[str] = []

    def write(directory: int, name: str, data: bytes) -> None:
        if name == collector_progress.PROGRESS_FILE:
            failed_writes.append(name)
            raise OSError("advisory progress volume failure")
        original(directory, name, data)

    scene.journal_events(scene.execution("new", "2026-10-12T20:00:01Z"))
    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_write", write)
        assert scene.report(scene.collect(scene.at(hour=21))).generation == 2
    assert failed_writes and path.read_bytes() == previous
    assert scene.counter_total(scene.current()) == 3
    assert scene.report(scene.collect(scene.at(hour=21))).generation == 3
    assert path.read_bytes() != previous
    assert scene.counter_total(scene.current()) == 3


def test_progress_directory_open_failure_is_advisory_after_lease_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    original = storage.open_profile_dir
    failures = []

    def unavailable(profile: str, *parts: str, **options: Any) -> int:
        if parts == ("worker",):
            failures.append(parts)
            raise PermissionError("worker progress unavailable")
        return original(profile, *parts, **options)

    with monkeypatch.context() as patch:

        def fault(step: str) -> None:
            if step == "started":
                patch.setattr(storage, "open_profile_dir", unavailable)

        result = scene.collect(SCENE_NOW, on_step=fault)
    assert failures and scene.report(result).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert scene.report(scene.collect(SCENE_NOW)).generation == 3


@pytest.mark.parametrize("payload", [b"{", b'{"cycles":{}}'])
def test_bad_advisory_progress_does_not_reset_committed_counters(
    tmp_path: Path, payload: bytes
) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / "worker" / collector_progress.PROGRESS_FILE
    path.write_bytes(payload)
    assert collector_progress.load_previous(PROFILE) is None
    result = scene.collect(SCENE_NOW)
    assert result.epoch == 1 and scene.counter_total(scene.current()) == 2
    restored = collector_progress.read_progress(PROFILE)
    assert restored is not None and restored["last_cycle"]["generation"] == 2


def test_status_refuses_a_symlinked_profile_without_counting_its_target(tmp_path: Path) -> None:
    seeded(tmp_path)
    measured = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert measured.total_bytes is not None and measured.total_bytes > 0
    directory = vs.accounting_dir()
    saved = tmp_path / "saved-profile"
    directory.rename(saved)
    directory.symlink_to(saved, target_is_directory=True)
    try:
        unknown = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
        assert unknown.total_bytes is None and unknown.reason == "unsafe_path"
        assert all(
            part.bytes is None and part.reason == "unsafe_path" for part in unknown.components
        )
    finally:
        directory.unlink()
        saved.rename(directory)
    assert (
        status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12).total_bytes
        == measured.total_bytes
    )


def test_unreadable_profile_metadata_is_not_reported_as_an_empty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir()
    original = os.lstat

    def denied(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if Path(path) == directory:
            raise PermissionError("profile metadata denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(status_worker.os, "lstat", denied)
        unknown = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
        assert unknown.total_bytes is None and unknown.reason == "unreadable"
        assert all(part.bytes is None for part in unknown.components)
    measured = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert measured.total_bytes is not None and measured.total_bytes > 0


def test_file_removed_between_directory_listing_and_stat_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    victim = vs.accounting_dir() / "history" / "vanishing.json"
    victim.write_bytes(b"123456789")
    victim.chmod(0o600)
    baseline = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert baseline.total_bytes is not None
    original = os.scandir

    class VanishingEntry:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self.entry = entry

        def __getattr__(self, name: str) -> Any:
            return getattr(self.entry, name)

        def stat(self, **kwargs: Any) -> os.stat_result:
            victim.unlink()
            return self.entry.stat(**kwargs)

    @contextmanager
    def racing_scan(path: Any):
        with original(path) as entries:
            yield (
                VanishingEntry(entry) if Path(entry.path) == victim else entry for entry in entries
            )

    monkeypatch.setattr(status_worker.os, "scandir", racing_scan)
    observed = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    assert not victim.exists()
    assert observed.reason is None
    assert observed.total_bytes == baseline.total_bytes - 9


def test_component_failure_is_unknown_without_discarding_other_measured_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    baseline = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    before = {c.name: c.bytes for c in baseline.components}
    assert before["collector"] and before["snapshots"]
    original = os.scandir
    blocked = vs.accounting_dir() / "collector"

    def scandir(path: Any) -> Any:
        if Path(path) == blocked:
            raise PermissionError("blocked collector directory")
        return original(path)

    monkeypatch.setattr(status_worker.os, "scandir", scandir)
    partial = status_worker.storage_pressure(PROFILE, free_bytes_fn=lambda _: 10**12)
    parts = {c.name: c for c in partial.components}
    assert parts["collector"].bytes is None and parts["collector"].reason == "unreadable"
    assert parts["snapshots"].bytes == before["snapshots"]
    assert partial.total_bytes is None and partial.reason == "unreadable"


def test_progress_of_a_pending_cycle_reports_the_bytes_it_already_read(tmp_path: Path) -> None:
    """Regression: ``bytes_read`` stayed 0 until the cycle ended, so 'current progress' lied."""
    seeded_pending(tmp_path)
    holder = Holder(tmp_path, "evidence_written")  # reading is over, reduction pending
    try:
        answer = scene.collect(SCENE_NOW)
        assert answer.state == "already_running"
        progress = answer.progress
        assert progress is not None and progress["phase"] == "reducing"
        on_disk = sum(p.stat().st_size for p in (vs.projects(tmp_path)).rglob("*.jsonl"))
        assert (
            0 < count(progress, "bytes_read") <= on_disk
        )  # what the cycle consumed, never invented
        assert count(progress, "sources_done") == count(progress, "sources_total") > 0
    finally:
        holder.close()
    finished = scene.collect(SCENE_NOW)
    assert finished.state == "published"
    final = json.loads((vs.accounting_dir() / "worker" / "progress.json").read_text())
    assert final["bytes_read"] == scene.report(finished).bytes_read > 0 or final["phase"] == "idle"


def test_progress_of_a_pending_backfill_also_reports_sources_and_bytes(tmp_path: Path) -> None:
    """Regression: a backfill cycle never updated sources_done/bytes_read while it ran."""
    seeded_pending(tmp_path)
    assert scene.collect(SCENE_NOW).state == "published"
    holder = Holder_backfill(tmp_path)
    try:
        answer = scene.collect(SCENE_NOW)
        assert answer.state == "already_running"
        progress = answer.progress
        assert progress is not None and progress["mode"] == "backfill"
        assert count(progress, "sources_total") > 0
        assert count(progress, "sources_done") == count(progress, "sources_total")
        assert count(progress, "bytes_read") > 0
    finally:
        holder.close()
