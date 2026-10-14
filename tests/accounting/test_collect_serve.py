"""Serve loop scheduling (task 9.7): cadence, pause/config, failure isolation, stop, signals.

Cadence tests drive a fake *monotonic* clock whose ``wait`` advances it, so ten simulated
minutes run instantly and exact tick times are asserted. Lease behavior uses the real
``Lease`` on the isolated data root; signals are delivered to a real subprocess.
"""

from __future__ import annotations

import datetime
import inspect
import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import policy, serve_loop
from ssgrep.accounting.collector_types import CycleReport
from ssgrep.accounting.lease import AlreadyRunning, Lease, LeaseInfo
from ssgrep.accounting.serve_loop import (
    DEFAULT_CONFIG,
    LeaseHandle,
    LoopConfig,
    LoopControl,
    Worker,
    WorkerState,
    read_control,
)

PROFILE = "personal-local"
ORIGIN = 1000.5  # monotonic clocks have an arbitrary origin: nothing may assume it is zero
ENABLED = LoopControl(True, False, False, 60, 300)
SLOW = LoopControl(True, False, False, 3600, 3600)  # only the immediate first cycle runs
INVALID = {"profile_id": PROFILE, "roots": "no"}  # rejected: holds sending paused
VALID = {"profile_id": PROFILE}  # accepted: clears the sending pause


def report(generation: int | None = 1, **overrides: Any) -> CycleReport:
    fields: dict[str, Any] = {
        "status": "published",
        "generation": generation,
        "snapshot_id": f"snap-{generation}",
        "complete": True,
        "stopped_by": "complete",
    }
    fields.update(overrides)
    return CycleReport(**fields)


def ticks(start: int, stop: int, step: int) -> list[int]:
    return list(range(start, stop + 1, step))


class SpyLeases:
    """Stands in for ``Lease.acquire``: records every acquire and release."""

    def __init__(self) -> None:
        self.acquired: list[tuple[str, str]] = []
        self.released = 0

    def acquire(self, profile: str, kind: str) -> SpyLeases:
        self.acquired.append((profile, kind))
        return self

    def release(self) -> None:
        self.released += 1


class Harness:
    """A Worker on a fake monotonic clock; ``at`` scripts actions at exact simulated offsets."""

    def __init__(
        self,
        end: int,
        *,
        config: LoopConfig = DEFAULT_CONFIG,
        export: Callable[[], None] | None | bool = True,
        control: Callable[[str], LoopControl] | None = None,
        acquire: Callable[[str, str], LeaseHandle] | None = None,
        cycle: Callable[[Callable[[], bool]], CycleReport] | None = None,
        on_event: Callable[[WorkerState], None] | None = None,
    ) -> None:
        self.now = ORIGIN
        self.end = ORIGIN + end
        self.leases = SpyLeases()
        self.collects: list[float] = []
        self.exports: list[float] = []
        self.snapshots: list[tuple[float, WorkerState]] = []
        self._script: list[tuple[float, Callable[[], object]]] = []
        self._stretch: dict[float, float] = {}
        send = self._record_export if export is True else (export or None)
        self.worker = Worker(
            PROFILE,
            config=config,
            cycle=cycle or self._cycle,
            export=send,
            control=control or read_control,  # the real reader unless a test scripts control
            acquire=acquire or self.leases.acquire,
            clock=lambda: self.now,
            wait=self._wait,
            on_event=on_event,
        )

    @property
    def offset(self) -> float:
        return self.now - ORIGIN

    def at(self, offset: float, action: Callable[[], object]) -> None:
        self._script.append((ORIGIN + offset, action))

    def oversleep(self, offset: float, seconds: float) -> None:
        """The wait that starts at ``offset`` lasts ``seconds`` (a suspended laptop)."""
        self._stretch[ORIGIN + offset] = seconds

    def _record_export(self) -> None:
        self.exports.append(self.offset)

    def _cycle(self, should_stop: Callable[[], bool]) -> CycleReport:
        self.collects.append(self.offset)
        return report(len(self.collects))

    def _wait(self, seconds: float) -> bool:
        self.snapshots.append((self.offset, self.worker.state()))
        target = self.now + self._stretch.get(self.now, seconds)
        if target > self.end:
            return True  # the simulated session is over: behave like a stop request
        for when, action in sorted(self._script, key=lambda item: item[0]):
            if when <= target:
                self._script.remove((when, action))
                self.now = max(self.now, when)
                action()
        self.now = target
        return False

    def run(self) -> WorkerState:
        return self.worker.run()

    def phase_at(self, offset: float) -> str:
        return next(state.phase for at, state in self.snapshots if at == offset)


# ---- cadence ---------------------------------------------------------------------------------


def test_ten_simulated_minutes_collect_every_60s_and_export_every_300s() -> None:
    h = Harness(600, control=lambda _p: ENABLED)
    before = h.worker.state()
    final = h.run()
    assert h.collects == ticks(0, 600, 60)  # first collect immediately
    assert h.exports == [300, 600]  # first export one full interval after start
    assert (before.phase, before.cycles, before.next_collect_in) == ("starting", 0, None)
    assert (final.phase, final.cycles, final.next_collect_in) == ("stopped", 11, None)
    assert final.last_report == report(11) and final.pid == os.getpid()
    assert h.leases.acquired == [(PROFILE, "serve")] and h.leases.released == 1
    # the countdowns are whole seconds on the monotonic clock
    assert [(o, s.next_collect_in, s.next_export_in) for o, s in h.snapshots[:3]] == [
        (0, 60, 300),
        (60, 60, 240),
        (120, 60, 180),
    ]
    assert {s.phase for _o, s in h.snapshots} == {"idle"}


def test_default_cadence_is_60s_collect_300s_export_on_the_monotonic_clock() -> None:
    assert LoopConfig() == LoopConfig(collect_seconds=60, export_seconds=300)
    assert inspect.signature(Worker.__init__).parameters["clock"].default is time.monotonic
    with pytest.raises(ValueError):
        LoopConfig(collect_seconds=0)
    h = Harness(600)  # real read_control + default config: the same grid without any script
    policy.enable_collection(PROFILE)
    h.run()
    assert h.collects == ticks(0, 600, 60) and h.exports == [300, 600]


def test_phases_are_visible_while_a_cycle_and_an_export_run() -> None:
    seen: dict[str, str] = {}
    events: list[WorkerState] = []

    def cycle(_stop: Callable[[], bool]) -> CycleReport:
        seen["cycle"] = h.worker.state().phase
        return report()

    h = Harness(
        300,
        control=lambda _p: ENABLED,
        cycle=cycle,
        export=lambda: seen.update(export=h.worker.state().phase),
        on_event=events.append,
    )
    h.run()
    phases = [event.phase for event in events]
    assert seen == {"cycle": "collecting", "export": "exporting"}
    assert phases[0] == "collecting" and phases[-1] == "stopped"
    assert phases.index("idle") > phases.index("collecting")
    assert "exporting" in phases[phases.index("idle") :]


def test_wall_clock_jumps_do_not_change_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = Harness(600, control=lambda _p: ENABLED)
    baseline.run()

    skew = {"seconds": 0}
    real_time, real_now = time.time, datetime.datetime.now

    def not_from_the_loop() -> None:
        caller = sys._getframe(2).f_globals.get("__name__")
        assert caller != serve_loop.__name__, "the loop must not read the wall clock"

    def skewed_time() -> float:
        not_from_the_loop()
        return real_time() + skew["seconds"]

    class SkewedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> SkewedDatetime:
            not_from_the_loop()
            return super().now(tz) + datetime.timedelta(seconds=skew["seconds"])

    monkeypatch.setattr(time, "time", skewed_time)
    monkeypatch.setattr(datetime, "datetime", SkewedDatetime)
    jumped = Harness(600, control=lambda _p: ENABLED)
    jumped.at(150, lambda: skew.update(seconds=86_400))  # +1 day mid-run
    jumped.at(350, lambda: skew.update(seconds=-86_400))  # then -1 day
    jumped.run()
    # the shims really moved the wall clock a day backwards by the end
    assert abs((real_time() - time.time()) - 86_400) < 5
    assert abs((real_now() - datetime.datetime.now()).total_seconds() - 86_400) < 5
    assert (jumped.collects, jumped.exports) == (baseline.collects, baseline.exports)
    assert jumped.collects == ticks(0, 600, 60) and jumped.exports == [300, 600]


def test_oversleep_runs_one_catch_up_cycle_not_a_burst() -> None:
    h = Harness(1000, control=lambda _p: ENABLED)
    h.oversleep(120, 600)  # the machine slept from t=120 to t=720
    h.run()
    # ticks 180..660 were missed: one cycle at 720, then the normal grid resumes
    assert h.collects == [0, 60, 120, 720, 780, 840, 900, 960]
    assert h.exports == [720, 900]  # the missed 300/600 exports are dropped, not replayed


# ---- pause, disabled, sending paused ---------------------------------------------------------


def test_pause_stops_ticks_and_resume_continues_without_replaying_the_gap() -> None:
    policy.enable_collection(PROFILE)
    h = Harness(600)
    h.at(130, lambda: policy.pause(PROFILE))
    h.at(400, lambda: policy.resume(PROFILE))
    h.run()
    assert h.collects == [0, 60, 120, 420, 480, 540, 600]  # nothing in 130..400, no burst
    assert h.exports == [600]  # the 300 export fell inside the pause and was dropped
    paused = [(o, s) for o, s in h.snapshots if 130 < o < 400]
    assert [o for o, _s in paused] == [180, 240, 300, 360]
    assert {s.phase for _o, s in paused} == {"paused"}
    assert all(s.next_collect_in is None and s.next_export_in is None for _o, s in paused)
    assert (h.phase_at(120), h.phase_at(420)) == ("idle", "idle")
    assert h.snapshots[0][1].next_collect_in == 60  # live countdowns return after resume


def test_disabled_collection_runs_nothing_until_enabled() -> None:
    h = Harness(600)  # no policy file at all: the real reader answers "disabled"
    h.at(250, lambda: policy.enable_collection(PROFILE))
    h.run()
    disabled = [s for o, s in h.snapshots if o < 250]
    assert len(disabled) == 5 and {s.phase for s in disabled} == {"disabled"}
    assert all(s.next_collect_in is None for s in disabled)
    assert h.collects == [300, 360, 420, 480, 540, 600] and h.exports == [300, 600]


def test_sending_paused_blocks_export_but_not_collection() -> None:
    policy.enable_collection(PROFILE)
    assert policy.apply_managed_update(PROFILE, INVALID)[0] is False
    control = read_control(PROFILE)
    assert control.sending_paused and control.collection_enabled and not control.paused
    blocked = Harness(600)
    blocked.run()
    assert blocked.collects == ticks(0, 600, 60) and blocked.exports == []
    assert {s.phase for _o, s in blocked.snapshots} == {"idle"}
    # positive twin: correcting the policy at t=250 lets the 300 s export tick through
    assert policy.apply_managed_update(PROFILE, INVALID)[0] is False
    corrected = Harness(600)
    corrected.at(250, lambda: policy.apply_managed_update(PROFILE, VALID))
    corrected.run()
    assert corrected.collects == ticks(0, 600, 60) and corrected.exports == [300, 600]


def test_export_is_skipped_when_no_export_callable_is_configured() -> None:
    h = Harness(600, export=False, control=lambda _p: ENABLED)
    h.run()
    assert h.collects == ticks(0, 600, 60) and h.exports == []
    assert all(s.next_export_in is None for _o, s in h.snapshots)


def test_read_control_reflects_policy_and_config() -> None:
    assert read_control(PROFILE) == LoopControl(False, False, False, 60, 300)
    policy.enable_collection(PROFILE)
    assert read_control(PROFILE, LoopConfig(5, 7)) == LoopControl(True, False, False, 5, 7)
    policy.pause(PROFILE)
    assert read_control(PROFILE).paused and read_control(PROFILE).collection_enabled
    policy.resume(PROFILE)
    assert not read_control(PROFILE).paused
    policy.apply_managed_update(PROFILE, INVALID)
    invalid = read_control(PROFILE)
    assert (invalid.invalid_policy, invalid.sending_paused) == ("invalid_policy", True)


def test_unreadable_policy_file_reads_as_disabled_with_sending_paused() -> None:
    policy.enable_collection(PROFILE)
    policy.policy_path().write_text("{ not json")
    control = read_control(PROFILE)
    assert control.collection_enabled is False and control.sending_paused is True
    assert control.invalid_policy == "policy_corrupt"


def test_default_control_applies_the_workers_config() -> None:
    policy.enable_collection(PROFILE)
    h = Harness(20, config=LoopConfig(10, 15))
    h.run()
    assert h.collects == [0, 10, 20] and h.exports == [15]


# ---- live config -----------------------------------------------------------------------------


def _config_harness(end: int, changes: dict[float, tuple[int, int]]) -> Harness:
    live = {"collect": 60, "export": 300}

    def control(_profile: str) -> LoopControl:
        return LoopControl(True, False, False, live["collect"], live["export"])

    h = Harness(end, control=control)
    for offset, (collect, export) in changes.items():
        h.at(offset, lambda c=collect, e=export: live.update(collect=c, export=e))
    return h


def test_shorter_intervals_take_effect_by_the_next_tick() -> None:
    h = _config_harness(600, {125: (30, 120)})
    h.run()
    # the loop slept until 180 when the change landed; from there the 30 s grid applies
    assert h.collects == [0, 60, 120, *ticks(180, 600, 30)]
    assert h.exports == [180, 240, 360, 480, 600]  # 120 s grid, anchored at the overdue 120


def test_longer_intervals_take_effect_by_the_next_tick() -> None:
    h = _config_harness(600, {65: (120, 600)})
    h.run()
    assert h.collects == [0, 60, 180, 300, 420, 540]  # 120 s after the 60 tick, not every 60
    assert h.exports == [600]  # one 600 s interval after start (the old 300 s grid is gone)


# ---- failure isolation -----------------------------------------------------------------------


def test_cycle_exception_is_recorded_by_class_name_and_the_loop_continues() -> None:
    calls: list[float] = []

    def cycle(_stop: Callable[[], bool]) -> CycleReport:
        calls.append(h.offset)
        if len(calls) == 2:
            raise RuntimeError("secret path /home/someone/token")
        return report(len(calls))

    h = Harness(180, control=lambda _p: ENABLED, cycle=cycle)
    final = h.run()
    assert calls == [0, 60, 120, 180]  # the tick after the failure still ran
    failed = h.snapshots[1][1]  # observed right after the failing cycle at t=60
    assert (failed.last_error, failed.cycles, failed.last_report) == ("RuntimeError", 2, report(1))
    assert "secret" not in repr(h.snapshots) and "secret" not in repr(final)
    assert (final.cycles, final.last_report) == (4, report(4))  # later successes still counted
    assert final.last_error == "RuntimeError"  # the latest failure is kept, not erased


def test_failed_report_surfaces_its_stable_condition() -> None:
    failing = report(None, status="failed", snapshot_id=None, complete=False, error="source_locked")
    h = Harness(60, control=lambda _p: ENABLED, cycle=lambda _s: failing)
    final = h.run()
    assert (final.last_error, final.last_report, final.cycles) == ("source_locked", failing, 2)
    healthy = Harness(60, control=lambda _p: ENABLED).run()
    assert healthy.last_error is None  # a published report is not an error


def test_export_exception_is_recorded_and_the_next_export_runs() -> None:
    attempts: list[float] = []

    def export() -> None:
        attempts.append(h.offset)
        if len(attempts) == 1:
            raise ValueError("boom with detail")

    h = Harness(900, control=lambda _p: ENABLED, export=export)
    final = h.run()
    assert attempts == [300, 600, 900]
    assert final.last_error == "ValueError" and "detail" not in repr(final)
    assert h.collects == ticks(0, 900, 60)


def test_unreadable_control_ends_the_run_but_still_releases_the_lease() -> None:
    reads: list[int] = []

    def control(_profile: str) -> LoopControl:
        reads.append(1)
        if len(reads) == 3:
            raise OSError("disk gone")
        return ENABLED

    h = Harness(600, control=control)
    with pytest.raises(OSError):
        h.run()
    assert h.leases.released == 1 and h.collects == [0]
    assert (h.worker.state().phase, h.worker.state().last_error) == ("stopped", "OSError")


# ---- stop ------------------------------------------------------------------------------------


def test_stop_between_ticks_ends_run_and_releases_the_lease() -> None:
    h = Harness(600, control=lambda _p: ENABLED)
    h.at(90, h.worker.stop)
    final = h.run()
    assert (final.phase, h.collects, h.exports) == ("stopped", [0, 60], [])
    assert h.leases.released == 1


def test_stop_during_a_cycle_flags_it_and_run_finishes_after_it() -> None:
    seen: list[object] = []

    def cycle(should_stop: Callable[[], bool]) -> CycleReport:
        seen.append(should_stop())
        h.worker.stop()
        seen.extend([should_stop(), h.worker.state().phase])
        return report(7)

    h = Harness(600, control=lambda _p: ENABLED, cycle=cycle)
    final = h.run()
    assert seen == [False, True, "stopping"]
    assert (final.phase, final.cycles, final.last_report) == ("stopped", 1, report(7))
    assert h.exports == [] and h.leases.released == 1


def test_double_stop_is_harmless_and_stop_before_run_runs_no_cycle() -> None:
    h = Harness(600, control=lambda _p: ENABLED)
    h.worker.stop()
    h.worker.stop()
    final = h.run()
    assert (final.phase, final.cycles, h.collects) == ("stopped", 0, [])
    assert (h.leases.acquired, h.leases.released) == ([(PROFILE, "serve")], 1)
    h.worker.stop()  # after the run: still harmless
    assert h.worker.state().phase == "stopped"


# ---- helpers for workers on real threads and the real lease ----------------------------------


@pytest.fixture
def started() -> Iterator[list[Worker]]:
    workers: list[Worker] = []
    yield workers
    for worker in workers:
        worker.stop()
        worker.join(5)


@dataclass
class Live:
    worker: Worker
    ran: threading.Event  # set the moment the first cycle begins
    calls: list[int]


def live_worker(
    started: list[Worker],
    *,
    control: Callable[[str], LoopControl] = lambda _p: SLOW,
    fail_on: int | None = None,
) -> Live:
    ran, calls = threading.Event(), []

    def cycle(_stop: Callable[[], bool]) -> CycleReport:
        calls.append(1)
        ran.set()
        if fail_on == len(calls):
            raise KeyError("detail")
        return report(len(calls))

    worker = Worker(PROFILE, control=control, cycle=cycle)
    started.append(worker)
    return Live(worker, ran, calls)


def hourly(profile: str) -> LoopControl:
    return read_control(profile, LoopConfig(3600, 3600))


def test_stop_wakes_a_real_wait_immediately(started: list[Worker]) -> None:
    live = live_worker(started)
    thread = live.worker.start()
    assert live.ran.wait(5)
    assert thread.name == f"ssgrep-serve-{PROFILE}" and thread.daemon
    began = time.monotonic()
    live.worker.stop()
    live.worker.stop()
    assert live.worker.join(5) and time.monotonic() - began < 2  # not the 3600 s wait
    assert live.worker.state().phase == "stopped" and not thread.is_alive()
    with Lease.acquire(PROFILE, "collect", retry_seconds=0):
        pass  # the real lease is free again


def test_second_worker_on_one_profile_raises_already_running_and_runs_nothing(
    started: list[Worker],
) -> None:
    first = live_worker(started)
    first.worker.start()
    assert first.ran.wait(5)
    cycles: list[int] = []
    exports: list[int] = []
    second = Worker(
        PROFILE,
        control=lambda _p: ENABLED,
        cycle=lambda _s: (cycles.append(1), report())[1],
        export=lambda: exports.append(1),
    )
    with pytest.raises(AlreadyRunning) as caught:
        second.run()
    assert caught.value.info.pid == os.getpid() and caught.value.info.kind == "serve"
    assert (cycles, exports, second.state().cycles) == ([], [], 0)
    assert first.worker.state().cycles == 1  # the owner is untouched
    first.worker.stop()
    assert first.worker.join(5)
    third = Worker(
        PROFILE,
        control=lambda _p: ENABLED,
        cycle=lambda _s: (cycles.append(2), report())[1],
        clock=lambda: 0.0,
        wait=lambda _s: True,
    )
    assert third.run().phase == "stopped" and cycles == [2]  # twin: free lease, same call works


def test_already_running_from_the_lease_propagates_before_any_work() -> None:
    owner = LeaseInfo(pid=4242, process_start="t", kind="collect", acquired_at=None, alive=True)

    def refuse(_profile: str, _kind: str) -> LeaseHandle:
        raise AlreadyRunning(owner)

    h = Harness(600, control=lambda _p: ENABLED, acquire=refuse)
    with pytest.raises(AlreadyRunning) as caught:
        h.run()
    assert caught.value.info is owner
    assert (h.collects, h.exports, h.worker.state().cycles) == ([], [], 0)


def test_start_raises_already_running_synchronously_and_runs_nothing() -> None:
    cycles: list[int] = []
    with Lease.acquire(PROFILE, "collect"):
        worker = Worker(
            PROFILE,
            control=lambda _p: ENABLED,
            cycle=lambda _s: (cycles.append(1), report())[1],
        )
        with pytest.raises(AlreadyRunning):
            worker.start()
        assert cycles == [] and worker.join(1) and worker.state().phase == "stopped"
        with pytest.raises(RuntimeError, match="^worker_not_running$"):
            worker.run_cycle()
    assert cycles == []


def test_start_stop_join_releases_the_real_lease_and_double_start_is_rejected(
    started: list[Worker],
) -> None:
    live = live_worker(started)
    thread = live.worker.start()
    assert live.ran.wait(5)
    with pytest.raises(AlreadyRunning):
        Lease.acquire(PROFILE, "collect", retry_seconds=0)  # held while the worker runs
    with pytest.raises(RuntimeError, match="^worker_already_started$"):
        live.worker.start()
    live.worker.stop()
    assert live.worker.join(5) and not thread.is_alive()
    with Lease.acquire(PROFILE, "collect", retry_seconds=0):
        pass


def test_join_times_out_while_running_and_is_true_when_never_started(
    started: list[Worker],
) -> None:
    live = live_worker(started)
    live.worker.start()
    assert live.worker.join(0.01) is False
    assert Worker(PROFILE).join() is True


def test_status_is_the_state_snapshot() -> None:
    h = Harness(60, control=lambda _p: ENABLED)
    h.run()
    assert h.worker.status() == h.worker.state()
    assert h.worker.status().phase == "stopped"


def test_state_is_readable_while_a_cycle_is_in_flight(started: list[Worker]) -> None:
    gate, entered = threading.Event(), threading.Event()

    def cycle(_stop: Callable[[], bool]) -> CycleReport:
        entered.set()
        gate.wait(5)
        return report(3)

    worker = Worker(PROFILE, control=lambda _p: SLOW, cycle=cycle)
    started.append(worker)
    worker.start()
    assert entered.wait(5)
    mid = worker.state()  # must not wait for the cycle
    assert (mid.phase, mid.cycles, mid.last_report) == ("collecting", 0, None)
    gate.set()
    deadline = time.monotonic() + 5
    while worker.state().cycles == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert (worker.state().cycles, worker.state().last_report) == (1, report(3))


# ---- run_cycle -------------------------------------------------------------------------------


def test_run_cycle_runs_one_cycle_now_and_counts_it(started: list[Worker]) -> None:
    policy.enable_collection(PROFILE)
    live = live_worker(started, control=hourly)
    live.worker.start()
    assert live.ran.wait(5)
    result = live.worker.run_cycle()
    state = live.worker.state()
    assert result == report(2) and (state.cycles, state.last_report) == (2, report(2))
    assert live.calls == [1, 1] and state.phase == "idle"


def test_run_cycle_refuses_while_paused_disabled_or_not_running(started: list[Worker]) -> None:
    policy.enable_collection(PROFILE)
    live = live_worker(started, control=hourly)
    with pytest.raises(RuntimeError, match="^worker_not_running$"):
        live.worker.run_cycle()  # lease not held yet
    live.worker.start()
    assert live.ran.wait(5)
    policy.pause(PROFILE)
    with pytest.raises(RuntimeError, match="^worker_not_running$"):
        live.worker.run_cycle()
    policy.resume(PROFILE)
    assert live.worker.run_cycle() == report(2)  # allowed again after resume
    policy.save_profile(replace(policy.load_profile_policy(PROFILE), collection_enabled=False))
    with pytest.raises(RuntimeError, match="^worker_not_running$"):
        live.worker.run_cycle()  # disabled
    assert live.calls == [1, 1]  # the refused calls ran nothing
    live.worker.stop()
    assert live.worker.join(5)
    with pytest.raises(RuntimeError, match="^worker_not_running$"):
        live.worker.run_cycle()  # stopped


def test_run_cycle_failure_is_recorded_and_raised(started: list[Worker]) -> None:
    live = live_worker(started, fail_on=2)
    live.worker.start()
    assert live.ran.wait(5)
    with pytest.raises(KeyError):
        live.worker.run_cycle()
    assert (live.worker.state().last_error, live.worker.state().cycles) == ("KeyError", 2)


def test_run_cycle_is_serialized_with_the_loops_cycle(started: list[Worker]) -> None:
    guard = threading.Lock()
    gate, entered = threading.Event(), threading.Event()
    active = {"now": 0, "peak": 0, "calls": 0}

    def cycle(_stop: Callable[[], bool]) -> CycleReport:
        with guard:
            active["now"] += 1
            active["calls"] += 1
            active["peak"] = max(active["peak"], active["now"])
            number = active["calls"]
        entered.set()
        gate.wait(5)
        with guard:
            active["now"] -= 1
        return report(number)

    worker = Worker(PROFILE, control=lambda _p: SLOW, cycle=cycle)
    started.append(worker)
    worker.start()
    assert entered.wait(5)  # the loop's own cycle is in flight
    results: list[CycleReport] = []
    caller = threading.Thread(target=lambda: results.append(worker.run_cycle()))
    caller.start()
    caller.join(0.05)
    assert caller.is_alive() and active["calls"] == 1  # queued behind the loop's cycle
    gate.set()
    caller.join(5)
    assert (active["peak"], active["calls"], results) == (1, 2, [report(2)])


# ---- default cycle wiring --------------------------------------------------------------------


def test_default_cycle_is_built_lazily_from_the_collector(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []
    flags: list[bool] = []

    def run_worker_cycle(profile: str) -> Callable[[Callable[[], bool]], CycleReport]:
        built.append(profile)

        def cycle(should_stop: Callable[[], bool]) -> CycleReport:
            flags.append(should_stop())
            return report(9)

        return cycle

    fake = types.SimpleNamespace(run_worker_cycle=run_worker_cycle)
    monkeypatch.setitem(sys.modules, "ssgrep.accounting.collector", fake)
    worker = Worker(
        PROFILE,
        control=lambda _p: ENABLED,
        clock=lambda: 0.0,
        wait=lambda _s: True,
        acquire=SpyLeases().acquire,
    )
    assert built == []  # construction builds nothing
    final = worker.run()
    assert built == [PROFILE] and flags == [False] and final.last_report == report(9)


# ---- real signals ----------------------------------------------------------------------------

SIGNAL_SCRIPT = """
import json, signal, sys
from pathlib import Path
from ssgrep.accounting import policy
from ssgrep.accounting.collector_types import CycleReport
from ssgrep.accounting.lease import Lease
from ssgrep.accounting.serve_loop import LoopConfig, Worker

marker = Path(sys.argv[1])
policy.enable_collection("personal-local")
signums = (signal.SIGTERM, signal.SIGINT)
before = {s: signal.getsignal(s) for s in signums}

def cycle(should_stop):
    with open(marker / "cycles.log", "a") as log:
        log.write("c\\n")
    return CycleReport("published", 1, "snap-1", True, "complete")

worker = Worker("personal-local", config=LoopConfig(1, 1), cycle=cycle)
worker.install_signal_handlers()
installed = all(signal.getsignal(s) != before[s] for s in signums)
final = worker.run()
with Lease.acquire("personal-local", "probe", retry_seconds=0):
    lease_free = True
(marker / "final.json").write_text(json.dumps({
    "phase": final.phase,
    "cycles": final.cycles,
    "installed": installed,
    "restored": all(signal.getsignal(s) == before[s] for s in signums),
    "lease_free": lease_free,
}))
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_real_signal_stops_the_worker_cleanly(signum: signal.Signals, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    marker.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", SIGNAL_SCRIPT, str(marker)],
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        log = marker / "cycles.log"
        while not (log.exists() and log.stat().st_size) and proc.poll() is None:
            assert time.monotonic() < deadline, "worker never ran its first cycle"
            time.sleep(0.02)
        proc.send_signal(signum)
        _out, err = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, err.decode()
    final = json.loads((marker / "final.json").read_text())
    assert final["phase"] == "stopped" and final["cycles"] >= 1
    assert final["installed"] and final["restored"] and final["lease_free"]


def test_handlers_are_installed_then_restored_after_a_started_worker_stops(
    started: list[Worker],
) -> None:
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        live = live_worker(started)
        live.worker.install_signal_handlers()
        assert all(signal.getsignal(s) != before[s] for s in before)
        live.worker.start()
        assert live.ran.wait(5)
        live.worker.stop()
        assert live.worker.join(5)  # restores from the main thread: ``run`` is on another
        assert {s: signal.getsignal(s) for s in before} == before
    finally:
        for signum, handler in before.items():
            signal.signal(signum, handler)


# ---- isolation -------------------------------------------------------------------------------

ISOLATION_SCRIPT = """
import json, sys
import ssgrep.accounting.serve_loop as serve_loop
worker = serve_loop.Worker("personal-local")
watched = ["ssgrep.services", "ssgrep.services.mcp_server", "ssgrep.pipeline", "ssgrep.store",
           "http.server", "socketserver", "ssgrep.accounting.collector",
           "ssgrep.accounting.lease", "ssgrep.accounting.policy"]
print(json.dumps([m for m in watched if m in sys.modules]))
"""


def test_importing_and_constructing_a_worker_loads_no_heavy_or_http_modules() -> None:
    out = subprocess.run(
        [sys.executable, "-c", ISOLATION_SCRIPT],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(out.stdout) == []


def test_a_full_session_writes_no_launchd_or_systemd_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    policy.enable_collection(PROFILE)
    h = Harness(600, acquire=lambda profile, kind: Lease.acquire(profile, kind))
    h.run()
    assert h.collects == ticks(0, 600, 60) and h.exports == [300, 600]
    tree = list(tmp_path.rglob("*"))
    names = {p.name for p in tree if p.is_file()}
    assert {"policy.json", "lease.lock"} <= names  # the walk sees what the session wrote
    assert not [p for p in tree if p.suffix in {".plist", ".service", ".timer", ".socket"}]
    assert not [p for p in tree if {"LaunchAgents", "LaunchDaemons", "systemd"} & set(p.parts)]
    assert list(home.iterdir()) == []
