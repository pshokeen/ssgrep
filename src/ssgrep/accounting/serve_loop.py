"""Foreground scheduling for the collection worker (no HTTP, no daemon install).

``Worker.run`` holds the per-profile lease and ticks two independent timers on an
injected *monotonic* clock: collection (60 s) and export (300 s, first one a full
export interval after start). Policy is re-read on every wake-up, so pause/resume
and interval changes take effect by the next tick. A tick that falls due while
paused/disabled, or while a long cycle ran, is consumed and never replayed: resuming
runs at most one cycle. Nothing here imports ``services``, the index/pipeline
packages or an HTTP stack; policy, lease and collector load on first use, so
importing this module and constructing a ``Worker`` stay free.
"""

from __future__ import annotations

import math
import os
import select
import signal
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from types import FrameType
from typing import Any, Protocol

from ssgrep.accounting.collector_types import CycleReport
from ssgrep.accounting.ledger_contracts import DEFAULT_COLLECT_SECONDS, DEFAULT_EXPORT_SECONDS

__all__ = [
    "DEFAULT_CONFIG",
    "LEASE_KIND",
    "Cycle",
    "LeaseHandle",
    "LoopConfig",
    "LoopControl",
    "Worker",
    "WorkerState",
    "read_control",
]

LEASE_KIND = "serve"
SIGNALS = (signal.SIGTERM, signal.SIGINT)


def _check_intervals(collect_seconds: int, export_seconds: int) -> None:
    if collect_seconds < 1 or export_seconds < 1:
        raise ValueError("loop intervals must be at least one second")


@dataclass(frozen=True)
class LoopConfig:
    collect_seconds: int = DEFAULT_COLLECT_SECONDS
    export_seconds: int = DEFAULT_EXPORT_SECONDS

    def __post_init__(self) -> None:
        _check_intervals(self.collect_seconds, self.export_seconds)


DEFAULT_CONFIG = LoopConfig()


@dataclass(frozen=True)
class LoopControl:
    """What the loop may do right now; re-read on every wake-up."""

    collection_enabled: bool
    paused: bool
    sending_paused: bool
    collect_seconds: int
    export_seconds: int
    invalid_policy: str | None = None  # stable condition when the policy blocks sending

    def __post_init__(self) -> None:
        _check_intervals(self.collect_seconds, self.export_seconds)


def read_control(profile: str, config: LoopConfig = DEFAULT_CONFIG) -> LoopControl:
    """Current control for ``profile``, read-only; an unreadable policy yields a disabled one.

    Policy stores no cadence, so the intervals come from ``config``.
    """
    from ssgrep.accounting import policy

    status = policy.policy_status(profile)
    reason = status.sending_paused_reason
    return LoopControl(
        collection_enabled=status.profile.collection_enabled,
        paused=status.profile.paused,
        sending_paused=reason is not None,
        collect_seconds=config.collect_seconds,
        export_seconds=config.export_seconds,
        invalid_policy=reason,
    )


@dataclass(frozen=True)
class WorkerState:
    # starting|idle|collecting|exporting|paused|disabled|stopping|stopped
    phase: str
    profile: str
    pid: int
    cycles: int  # collection cycles finished, whatever their outcome
    last_report: CycleReport | None
    last_error: str | None  # latest failure (exception class name or stable condition); kept
    next_collect_in: int | None  # whole seconds on the monotonic clock; None: will not run
    next_export_in: int | None


class LeaseHandle(Protocol):
    def release(self) -> None: ...


Cycle = Callable[[Callable[[], bool]], CycleReport]


def _default_acquire(profile: str, kind: str) -> LeaseHandle:
    from ssgrep.accounting.lease import Lease

    return Lease.acquire(profile, kind)


def _default_cycle(profile: str) -> Cycle:
    from ssgrep.accounting.collector import run_worker_cycle

    return run_worker_cycle(profile)


def _close_fds(*fds: int) -> None:
    for fd in fds:
        os.close(fd)


class _StopFlag:
    """A stop request that signal handlers and other threads may set, and that wakes a wait.

    A signal handler calling ``threading.Event.set`` can deadlock when it interrupts the main
    thread while that thread holds the Event's non-reentrant lock inside ``Event.wait``. A plain
    flag plus a self-pipe byte has no lock; ``select`` sees the byte even when the stop landed
    between the flag check and the sleep.
    """

    def __init__(self) -> None:
        self._requested = False
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._read_fd, self._write_fd = read_fd, write_fd
        weakref.finalize(self, _close_fds, read_fd, write_fd)

    def set(self) -> None:
        self._requested = True
        try:
            os.write(self._write_fd, b"\0")
        except BlockingIOError:
            pass  # a full pipe is already readable: a sleeper is already awake

    def is_set(self) -> bool:
        return self._requested

    def wait(self, seconds: float) -> bool:
        if not self._requested:
            select.select([self._read_fd], [], [], max(0.0, seconds))
        return self._requested


class _Timer:
    """Fixed-grid schedule: ticks fall at ``anchor + n * interval``; ``anchor`` None = due now.

    The interval is supplied per call, so a changed interval applies to the next tick.
    """

    def __init__(self, anchor: float | None) -> None:
        self.anchor = anchor

    def next_at(self, interval: int, now: float) -> float:
        return now if self.anchor is None else self.anchor + interval

    def due_tick(self, interval: int, now: float) -> float | None:
        """The scheduled time of the tick that is due at ``now``, else None."""
        at = self.next_at(interval, now)
        return at if now >= at else None

    def settle(self, interval: int, scheduled: float, now: float) -> None:
        """Tick ``scheduled`` was handled; skip every grid point missed up to ``now``."""
        self.anchor = scheduled + ((now - scheduled) // interval) * interval


def _remaining(deadline: float | None, now: float) -> int | None:
    return None if deadline is None else max(0, math.ceil(deadline - now))


def _may_collect(control: LoopControl) -> bool:
    return control.collection_enabled and not control.paused


def _may_export(control: LoopControl) -> bool:
    return _may_collect(control) and not control.sending_paused


def _resting_phase(control: LoopControl) -> str:
    if control.paused:
        return "paused"
    return "idle" if control.collection_enabled else "disabled"


class Worker:
    """One profile's foreground collect/export loop. Single use: ``run`` blocks until stopped."""

    def __init__(
        self,
        profile: str,
        *,
        config: LoopConfig = DEFAULT_CONFIG,
        cycle: Cycle | None = None,
        export: Callable[[], None] | None = None,
        control: Callable[[str], LoopControl] = read_control,
        acquire: Callable[[str, str], LeaseHandle] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
        on_event: Callable[[WorkerState], None] | None = None,
    ) -> None:
        self._profile = profile
        self._cycle = cycle
        self._export = export
        self._control: Callable[[str], LoopControl] = (
            partial(read_control, config=config) if control is read_control else control
        )
        self._acquire = acquire if acquire is not None else _default_acquire
        self._clock = clock
        self._stop = _StopFlag()
        self._wait = wait if wait is not None else self._stop.wait
        self._on_event = on_event
        self._lock = threading.Lock()
        self._deadlines: tuple[float | None, float | None] = (None, None)
        self._state = WorkerState("starting", profile, os.getpid(), 0, None, None, None, None)
        self._previous_handlers: dict[int, Any] = {}
        self._work_lock = threading.Lock()  # one cycle/export/phase change at a time
        self._active_cycle: Cycle | None = None  # set only while the lease is held
        self._ready = threading.Event()  # lease held (or startup failed): ``start`` may return
        self._startup_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def stop(self) -> None:
        """Request shutdown: thread- and signal-safe, idempotent, wakes the default wait."""
        self._stop.set()

    def state(self) -> WorkerState:
        with self._lock:
            base, (collect_at, export_at) = self._state, self._deadlines
        phase = "stopping" if self._stop.is_set() and base.phase != "stopped" else base.phase
        now = self._clock()
        return replace(
            base,
            phase=phase,
            next_collect_in=_remaining(collect_at, now),
            next_export_in=_remaining(export_at, now),
        )

    def status(self) -> WorkerState:
        """Alias of ``state`` for callers composing the worker with other components."""
        return self.state()

    def install_signal_handlers(self) -> None:
        """SIGTERM/SIGINT request a clean stop; ``run`` restores the previous handlers on return.

        Main thread only (``signal.signal`` raises ``ValueError`` elsewhere).
        """
        for signum in SIGNALS:
            if signum not in self._previous_handlers:
                self._previous_handlers[signum] = signal.signal(signum, self._on_signal)

    def run(self) -> WorkerState:
        """Hold the lease and tick until stopped; ``AlreadyRunning`` propagates untouched."""
        try:
            lease = self._acquire(self._profile, LEASE_KIND)
            try:
                cycle = self._cycle or _default_cycle(self._profile)
                with self._work_lock:
                    self._active_cycle = cycle
                self._ready.set()
                self._loop(cycle)
            finally:
                with self._work_lock:  # waits out a ``run_cycle`` still in flight
                    self._active_cycle = None
                lease.release()
        except BaseException as error:
            if not self._ready.is_set():
                self._startup_error = error
            else:
                self._publish(last_error=type(error).__name__)
            raise
        finally:
            self._ready.set()
            self._restore_signal_handlers()
            self._publish(phase="stopped", deadlines=(None, None))
        return self.state()

    def start(self) -> threading.Thread:
        """Run the loop on a daemon thread; return once the lease is held.

        A startup failure (``AlreadyRunning``, unsafe storage, ...) is raised here, before
        returning, and nothing keeps running. Signal handlers installed beforehand are
        restored by ``join`` (main thread), since ``run`` cannot do so from this thread.
        """
        if self._thread is not None:
            raise RuntimeError("worker_already_started")
        thread = threading.Thread(
            target=self._thread_main, name=f"ssgrep-serve-{self._profile}", daemon=True
        )
        self._thread = thread
        thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            thread.join()
            raise self._startup_error
        return thread

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the ``start`` thread; True once it has ended (or none was started)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                return False
        self._restore_signal_handlers()
        return True

    def run_cycle(self) -> CycleReport:
        """Run one cycle now, serialized with the loop's cycles and exports; schedule unchanged.

        ``RuntimeError("worker_not_running")`` unless the lease is held and collection is
        currently allowed (not stopping, paused or disabled). A cycle exception is recorded in
        ``last_error`` and propagates. Never call it from inside the cycle or ``on_event``.
        """
        with self._work_lock:
            cycle = self._active_cycle
            if cycle is None or self._stop.is_set():
                raise RuntimeError("worker_not_running")
            control = self._control(self._profile)
            if not _may_collect(control):
                raise RuntimeError("worker_not_running")
            try:
                return self._execute(cycle)
            finally:
                self._publish(phase=_resting_phase(control))

    def _thread_main(self) -> None:
        try:
            self.run()
        except BaseException:
            if self._startup_error is None:  # startup failures belong to ``start``'s caller
                raise

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        self.stop()

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # ``signal.signal`` is main-thread only; ``join`` restores from there
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
        self._previous_handlers.clear()

    def _publish(
        self, *, deadlines: tuple[float | None, float | None] | None = None, **changes: Any
    ) -> None:
        with self._lock:
            self._state = replace(self._state, **changes)
            if deadlines is not None:
                self._deadlines = deadlines
        if self._on_event is not None:
            self._on_event(self.state())

    def _loop(self, cycle: Cycle) -> None:
        collect, export = _Timer(None), _Timer(self._clock())
        while not self._stop.is_set():
            control = self._control(self._profile)
            if self._tick(
                collect,
                control.collect_seconds,
                _may_collect(control),
                partial(self._collect, cycle),
            ):
                control = self._control(self._profile)  # a long cycle: a pause may have arrived
            if self._export is not None:
                send = partial(self._send, self._export)
                self._tick(export, control.export_seconds, _may_export(control), send)
            if self._stop.is_set():
                break
            if self._wait(self._rest(control, collect, export)):
                self._stop.set()

    def _tick(
        self, timer: _Timer, interval: int, allowed: bool, action: Callable[[], None]
    ) -> bool:
        """If ``timer`` is due, run ``action`` when allowed; the tick is consumed either way.

        Returns whether ``action`` ran.
        """
        scheduled = timer.due_tick(interval, self._clock())
        if scheduled is None or self._stop.is_set():
            return False
        if allowed:
            action()
        timer.settle(interval, scheduled, self._clock())
        return allowed

    def _rest(self, control: LoopControl, collect: _Timer, export: _Timer) -> float:
        """Publish the between-ticks state; return how long to sleep until the next tick."""
        now = self._clock()
        collect_at = collect.next_at(control.collect_seconds, now)
        export_at = export.next_at(control.export_seconds, now)
        wake = collect_at if self._export is None else min(collect_at, export_at)
        with self._work_lock:  # not while ``run_cycle`` is mid-cycle
            self._publish(
                phase=_resting_phase(control),
                deadlines=(
                    collect_at if _may_collect(control) else None,
                    export_at if self._export is not None and _may_export(control) else None,
                ),
            )
        return max(0.0, wake - now)

    def _collect(self, cycle: Cycle) -> None:
        with self._work_lock:
            try:
                self._execute(cycle)
            except Exception:
                pass  # already recorded in ``last_error``; the loop keeps ticking

    def _execute(self, cycle: Cycle) -> CycleReport:
        """Run ``cycle`` (cycle lock held), recording the report or the failure's class name."""
        self._publish(phase="collecting")
        cycles = self._state.cycles + 1
        try:
            report = cycle(self._stop.is_set)
        except Exception as error:
            self._publish(cycles=cycles, last_error=type(error).__name__)
            raise
        failure = (report.error or "cycle_failed") if report.status == "failed" else None
        self._publish(
            cycles=cycles, last_report=report, last_error=failure or self._state.last_error
        )
        return report

    def _send(self, export: Callable[[], None]) -> None:
        with self._work_lock:
            self._publish(phase="exporting")
            try:
                export()
            except Exception as error:
                self._publish(last_error=type(error).__name__)
