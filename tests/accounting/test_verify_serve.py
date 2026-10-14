"""9.7: the serve loop with the REAL collector underneath, on an injected monotonic clock.

``test_collect_serve.py`` covers scheduling with a stub cycle. Here the cycle is
``collector.run_worker_cycle`` (real reducer, real snapshots), the lease is the real flock, the
policy is the real policy file, and time is a fake monotonic clock whose ``wait`` advances it, so
ten simulated minutes run in milliseconds and every claim is read off real generations on disk.
"""

from __future__ import annotations

import datetime as datetime_module
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector, policy, serve_loop, snapshots
from ssgrep.accounting.collector_types import CycleReport
from ssgrep.accounting.lease import Lease, inspect_lease
from ssgrep.accounting.serve_loop import LoopConfig, Worker
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE

ROOT = Path(__file__).resolve().parents[2]
CLOCK = scene.at(hour=9)
ORIGIN = 5000.25


BASE = scene.at(12, 20)  # the simulated wall clock at monotonic offset 0


@dataclass
class Sim:
    """A real Worker over the real collector on a fake monotonic clock.

    The collection (wall) clock is simulated too, so no scene depends on the real date: it reads
    ``BASE + elapsed monotonic seconds + skew`` and ``skew`` is what tests change to jump it.
    """

    end: int
    monkeypatch: pytest.MonkeyPatch
    exports: list[float] = field(default_factory=list)
    reports: list[tuple[float, CycleReport]] = field(default_factory=list)
    now: float = ORIGIN
    skew: datetime_module.timedelta = datetime_module.timedelta(0)
    scripted: list[tuple[float, Any]] = field(default_factory=list)

    def wall(self) -> datetime_module.datetime:
        return BASE + datetime_module.timedelta(seconds=self.now - ORIGIN) + self.skew

    def __post_init__(self) -> None:
        self.monkeypatch.setattr(collector, "_now", self.wall)
        real_cycle = collector.run_worker_cycle(PROFILE, clock=lambda: self.now)

        def cycle(stop: Any) -> CycleReport:
            report = real_cycle(stop)
            self.reports.append((self.now - ORIGIN, report))
            return report

        self.worker = Worker(
            PROFILE,
            cycle=cycle,
            export=lambda: self.exports.append(self.now - ORIGIN),
            clock=lambda: self.now,
            wait=self._wait,
        )

    def at(self, offset: float, action: Any) -> None:
        self.scripted.append((ORIGIN + offset, action))

    def _wait(self, seconds: float) -> bool:
        target = self.now + seconds
        if target > ORIGIN + self.end:
            return True
        for when, action in sorted(self.scripted, key=lambda item: item[0]):
            if when <= target:
                self.scripted.remove((when, action))
                self.now = max(self.now, when)
                action()
        self.now = target
        return False

    def run(self) -> serve_loop.WorkerState:
        return self.worker.run()

    @property
    def generations(self) -> list[int]:
        return [r.generation for _, r in self.reports if r.generation is not None]


def scene_ready(root: Path, *, extra: int = 0) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in range(2 + extra):
        vs.pair(vs.projects(root), n, CLOCK)


def collected(generation: int) -> datetime_module.datetime:
    envelope = snapshots.read_generation(PROFILE, generation)
    assert envelope is not None
    return datetime_module.datetime.fromisoformat(envelope.collected_at.replace("Z", "+00:00"))


def test_ten_simulated_minutes_publish_real_generations_on_a_60_second_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_ready(tmp_path)
    sim = Sim(600, monkeypatch)
    final = sim.run()
    assert [round(at) for at, _ in sim.reports] == list(range(0, 601, 60))
    assert sim.exports == [300.0, 600.0]  # the 300 s export cadence, first one a full interval in
    assert sim.generations == list(range(1, 12))  # eleven real generations, strictly in order
    assert snapshots.generations(PROFILE) == list(range(1, 12))
    assert final.phase == "stopped" and final.cycles == 11
    assert all(r.status == "published" and r.complete for _, r in sim.reports)
    # every generation is stamped with the collection clock of its own tick, 60 s apart
    stamps = [collected(g) for g in sim.generations]
    assert [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:], strict=False)] == [
        60.0
    ] * 10
    # clock-only cycles after the first change no ledger content: one snapshot identity throughout
    first, last = snapshots.read_generation(PROFILE, 1), snapshots.read_generation(PROFILE, 11)
    assert first is not None and last is not None and first.snapshot_id == last.snapshot_id
    assert first.collected_at != last.collected_at
    held = inspect_lease(PROFILE)
    assert held is None or held.pid is None or not held.alive  # the lease was released at the end


def test_wall_clock_jumps_change_neither_the_cadence_nor_the_number_of_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_ready(tmp_path)
    baseline = Sim(600, monkeypatch)
    baseline.run()
    base_grid = [round(at) for at, _ in baseline.reports]
    base_exports = list(baseline.exports)
    vs.wipe(tmp_path)  # a fresh data root so the second run starts from the same state
    scene_ready(tmp_path)

    jumped = Sim(600, monkeypatch)
    jumped.at(130, lambda: setattr(jumped, "skew", datetime_module.timedelta(days=3)))  # forward
    jumped.at(370, lambda: setattr(jumped, "skew", datetime_module.timedelta(days=-2)))  # back
    jumped.run()
    assert [round(at) for at, _ in jumped.reports] == base_grid  # same simulated ticks
    assert jumped.exports == base_exports
    assert jumped.generations == list(range(1, len(base_grid) + 1))
    # the wall clock really moved, and shows up ONLY as the label of each generation
    day = datetime_module.timedelta(days=1)
    offsets = {g: collected(g) - BASE for g in (1, 3, 4, 7, 8, 11)}
    assert offsets[1] == datetime_module.timedelta(seconds=0)
    assert offsets[3] == datetime_module.timedelta(seconds=120)  # before the jump
    assert offsets[4] == 3 * day + datetime_module.timedelta(seconds=180)  # first tick after +3d
    assert offsets[7] == 3 * day + datetime_module.timedelta(seconds=360)
    assert offsets[8] == -2 * day + datetime_module.timedelta(seconds=420)  # after the step back
    assert offsets[11] == -2 * day + datetime_module.timedelta(seconds=600)


def test_pause_is_honored_by_the_next_tick_and_resume_never_imports_the_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_ready(tmp_path)
    sim = Sim(600, monkeypatch)
    gap_start, gap_end = scene.at(12, 12), scene.at(12, 14)  # the recorded pause interval
    sim.at(130, lambda: policy.pause(PROFILE, now=gap_start))
    seen: dict[str, Any] = {}

    def during_pause() -> None:
        vs.pair(vs.projects(tmp_path), 4, CLOCK)  # its events are dated 13:40, inside the gap
        seen["generation"] = snapshots.current_generation(PROFILE)

    def after_resume() -> None:
        vs.pair(vs.projects(tmp_path), 5, CLOCK)  # dated 14:50: after the gap

    sim.at(250, during_pause)
    sim.at(400, lambda: policy.resume(PROFILE, now=gap_end))
    sim.at(430, after_resume)
    sim.run()
    ticks = [round(at) for at, _ in sim.reports]
    assert ticks == [0, 60, 120, 420, 480, 540, 600]  # nothing between pause and resume
    assert seen["generation"] == 3  # no generation was published during the pause
    assert sim.exports == [600.0]  # the export tick inside the pause was dropped, not replayed
    sessions = {e[1] for e in vs.evidence_identities()}
    assert {"cons5", "src5"} <= sessions  # evidence after the resume IS collected ...
    assert "cons4" not in sessions  # ... the paused interval is NOT silently imported
    assert snapshots.current_generation(PROFILE) == 7  # three before + four after, no burst
    dropped = scene.coverage(scene.current())["gaps"]
    assert dropped.get("dropped_paused_interval", 0) >= 1  # and the omission is visible, not silent


def test_a_config_change_between_ticks_is_honored_by_the_next_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_ready(tmp_path)
    live = {"collect": 60, "export": 300}

    def control(_profile: str) -> serve_loop.LoopControl:
        return serve_loop.LoopControl(True, False, False, live["collect"], live["export"])

    sim = Sim(400, monkeypatch)
    sim.worker = Worker(
        PROFILE,
        cycle=collector.run_worker_cycle(PROFILE, clock=lambda: sim.now),
        export=lambda: sim.exports.append(sim.now - ORIGIN),
        control=control,
        clock=lambda: sim.now,
        wait=sim._wait,
    )
    seen: list[float] = []
    inner = sim.worker._cycle
    assert inner is not None
    real_cycle: serve_loop.Cycle = inner

    def recording_cycle(stop: Callable[[], bool]) -> CycleReport:
        seen.append(sim.now - ORIGIN)
        return real_cycle(stop)

    sim.worker._cycle = recording_cycle
    sim.at(65, lambda: live.update(collect=100, export=1000))
    sim.run()
    assert seen[:2] == [0.0, 60.0]
    assert seen[2:] == [160.0, 260.0, 360.0]  # 100 s from the last tick, not every 60 s
    assert sim.exports == []  # the 1000 s export interval replaced the 300 s one


def test_sigterm_during_a_real_cycle_finishes_it_publishes_and_exits_cleanly(
    tmp_path: Path,
) -> None:
    """A real serve process stops on SIGTERM that lands while a cycle is pending."""
    scene_ready(tmp_path)
    hold = tmp_path / "hold"
    code = """
import sys
from pathlib import Path
from ssgrep.accounting import collector
from ssgrep.accounting.serve_loop import LoopConfig, Worker

hold = Path(sys.argv[1])

def on_step(name):
    if name == "before_stage" and not (hold / "reached").exists():
        hold.mkdir(parents=True, exist_ok=True)
        (hold / "reached").write_text("1")
        while not (hold / "release").exists():
            import time; time.sleep(0.01)

cycle = collector.run_worker_cycle("personal-local", on_step=on_step)
worker = Worker("personal-local", config=LoopConfig(1, 1), cycle=cycle)
worker.install_signal_handlers()
final = worker.run()
print(final.phase, final.cycles, final.last_report.status if final.last_report else None)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(hold)],
        env=vs.env_for(tmp_path),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not (hold / "reached").exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        held = inspect_lease(PROFILE)
        assert held is not None and held.kind == "serve" and held.pid == process.pid and held.alive
        process.send_signal(signal.SIGTERM)  # arrives mid-cycle
        time.sleep(0.2)
        assert process.poll() is None  # the cycle in flight is allowed to finish
        (hold / "release").write_text("go")
        out, err = process.communicate(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, err[-2000:]
    assert out.split() == ["stopped", "1", "published"]  # exactly the one cycle, completed
    assert snapshots.current_generation(PROFILE) == 1  # its generation really is published
    with Lease.acquire(PROFILE, "probe", retry_seconds=0):  # and the lease is free again
        pass


def test_nothing_attaches_a_reducer_or_worker_to_index_mcp_or_search_startup() -> None:
    """Observed at runtime in a fresh interpreter, not by reading imports."""
    script = """
import json, sys, threading
from ssgrep.services import mcp_server, api
mcp_server.get_mcp_server.__wrapped__ if hasattr(mcp_server.get_mcp_server, "__wrapped__") else None
mcp_server._start_reconcile_background = lambda: None  # no real indexing in a unit run
mcp_server.get_mcp_server()
import ssgrep.indexing.indexer  # noqa: F401
from ssgrep.cli.commands import search_command, show_command  # noqa: F401
heavy = {"collector", "collector_cycle", "serve_loop", "lease", "reduce", "reduce_io",
         "snapshots", "windows", "counters", "status_worker"}
loaded = [m for m in sys.modules
          if m.startswith("ssgrep.accounting.") and m.split(".")[2] in heavy]
names = sorted(t.name for t in threading.enumerate())
print(json.dumps({"loaded": loaded, "threads": names}))
"""
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=dict(os.environ),
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.splitlines()[-1])
    assert report["loaded"] == []  # the worker, reducer and lease are never imported
    assert not [n for n in report["threads"] if "serve" in n or "collect" in n]


def test_starting_mcp_and_reconciliation_never_takes_the_worker_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.services import mcp_server

    before = inspect_lease(PROFILE)
    started = threading.Event()
    monkeypatch.setattr(mcp_server, "_start_reconcile_background", started.set)
    mcp_server._mcp = None
    mcp_server.get_mcp_server()
    assert started.is_set()
    assert inspect_lease(PROFILE) == before  # no lease file appeared or changed
    assert not (vs.accounting_dir() / "worker").exists()
    assert not (vs.accounting_dir() / "collector").exists()
    with Lease.acquire(PROFILE, "collect"):  # the lease is free for the real worker
        pass


def test_the_default_cycle_of_a_worker_is_the_collectors_and_is_built_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_ready(tmp_path)
    sim = Sim(0, monkeypatch)
    default = Worker(
        PROFILE, config=LoopConfig(60, 300), clock=lambda: sim.now, wait=lambda s: True
    )
    assert default._cycle is None  # nothing built at construction
    final = default.run()  # builds collector.run_worker_cycle on first use, runs one real cycle
    assert final.cycles == 1 and final.last_report is not None
    assert final.last_report.status == "published"
    assert snapshots.current_generation(PROFILE) == 1
