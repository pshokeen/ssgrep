"""Bounded worker failures remain distinguishable from idle, zero lag, or successful delivery."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_progress, status_worker, storage
from ssgrep.accounting.collector import run_worker_cycle
from ssgrep.accounting.collector_types import CycleReport
from ssgrep.accounting.lease import Lease
from ssgrep.accounting.serve_loop import LoopControl, Worker
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_guard_collector_recovery import seeded


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


def test_stop_wakes_a_worker_even_when_its_notification_pipe_is_full(tmp_path: Path) -> None:
    seeded(tmp_path)
    calls: list[str] = []
    collected = run_worker_cycle(PROFILE)

    def control(_profile: str) -> LoopControl:
        return LoopControl(True, False, False, 60, 300)

    def cycle(stop: Callable[[], bool]) -> CycleReport:
        calls.append("collect")
        return collected(stop)

    worker = Worker(PROFILE, cycle=cycle, control=control, wait=lambda _: True)
    with pytest.raises(BlockingIOError):
        while True:
            os.write(worker._stop._write_fd, b"x" * 4096)
    worker.stop()
    assert worker.run().phase == "stopped"
    assert calls == []
    with Lease.acquire(PROFILE, "collect", retry_seconds=0):
        pass
    twin = Worker(PROFILE, cycle=cycle, control=control, wait=lambda _: True)
    assert twin.run().cycles == 1
    assert calls == ["collect"]
    assert scene.current().generation == 2


def test_fatal_worker_thread_exit_still_releases_the_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = threading.Event()
    errors: list[type[BaseException]] = []

    def hook(args: threading.ExceptHookArgs) -> None:
        errors.append(args.exc_type)
        failed.set()

    def cycle(_stop: Any) -> Any:
        raise SystemExit("worker interrupted after startup")

    monkeypatch.setattr(threading, "excepthook", hook)
    worker = Worker(
        PROFILE,
        cycle=cycle,
        control=lambda _profile: LoopControl(True, False, False, 60, 300),
    )
    worker.start()
    assert failed.wait(5)
    assert worker.join(5) and worker.state().phase == "stopped"
    assert errors == [SystemExit]
    assert worker.state().last_error == "SystemExit"
    with Lease.acquire(PROFILE, "collect", retry_seconds=0):
        pass
