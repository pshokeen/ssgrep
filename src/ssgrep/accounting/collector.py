"""The collection worker's public entry points (design D8, tasks 9.1-9.6).

``collect_once`` is the bounded one-shot used by ``metrics collect``: it takes the profile's
exclusive lease (a concurrent collect or serve reports ``already_running`` with the owner's
progress instead of reducing twice), runs exactly one cycle, and releases the lease.
``run_worker_cycle`` is the cycle the foreground serve loop runs while it already holds the
lease. Both go through ``collector_cycle.run_cycle``; failures leave the previous committed
generation untouched and surface as a stable ``CollectError.condition`` (never raw text).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from ssgrep.accounting import plainfile, storage
from ssgrep.accounting.collector_cycle import CycleContext, CycleOutcome, run_cycle
from ssgrep.accounting.collector_progress import (
    Progress,
    last_cycle_document,
    load_previous,
    read_progress,
)
from ssgrep.accounting.collector_reduce import Reducer
from ssgrep.accounting.collector_types import (
    CollectError,
    CollectOptions,
    CollectResult,
    CycleReport,
)
from ssgrep.accounting.lease import AlreadyRunning, Lease

__all__ = ["CollectError", "CollectOptions", "collect_once", "run_worker_cycle"]

Clock = Callable[[], float]
StopCheck = Callable[[], bool]


def _failure(error: BaseException) -> CollectError:
    """Map any cycle failure to a stable condition; the original stays chained for debugging."""
    if isinstance(error, CollectError):
        return error
    if isinstance(error, storage.UnsafeStorageError | plainfile.UnsafePathError):
        return CollectError(
            "storage_unsafe",
            "Accounting storage is not private to this user; collection is disabled until fixed.",
        )
    return CollectError("collection_failed", f"Collection failed ({type(error).__name__}).")


def _now() -> datetime:
    return datetime.now(UTC)


def _elapsed(clock: Clock, started: float) -> int:
    return int((clock() - started) * 1000)


def _failed_report(condition: str, elapsed_ms: int) -> CycleReport:
    return CycleReport(
        status="failed",
        generation=None,
        snapshot_id=None,
        complete=False,
        stopped_by="complete",
        elapsed_ms=elapsed_ms,
        error=condition,
    )


def _begin(progress: Progress, mode: str, now: datetime) -> None:
    progress.mode = mode
    progress.phase = "reading"
    progress.cycle_started_at = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    progress.sources_done = progress.sources_total = progress.bytes_read = 0
    progress.write(now)


def _finish(progress: Progress, report: CycleReport, now: datetime) -> None:
    progress.phase = "idle"
    progress.cycles += 1
    progress.bytes_read = report.bytes_read
    progress.lag_bytes = report.lag_bytes
    progress.cycle_started_at = None
    progress.last_cycle = last_cycle_document(report, now)
    progress.write(now)


def _run(
    progress: Progress,
    context: CycleContext,
    clock: Clock,
) -> CycleOutcome:
    """One cycle with progress bookkeeping; failure is recorded then re-raised as CollectError."""
    started = clock()
    _begin(progress, context.options.mode, context.now)
    try:
        outcome = run_cycle(context)
    except Exception as error:
        failure = _failure(error)
        _finish(progress, _failed_report(failure.condition, _elapsed(clock, started)), _now())
        if failure is error:
            raise
        raise failure from error
    except BaseException:  # interrupts propagate untouched, but the failure is still recorded
        _finish(progress, _failed_report("interrupted", _elapsed(clock, started)), _now())
        raise
    _finish(progress, outcome.report, _now())
    return outcome


def collect_once(
    profile: str,
    options: CollectOptions,
    *,
    now: datetime | None = None,
    clock: Clock = time.monotonic,
    should_stop: StopCheck = lambda: False,
    reducer: Reducer | None = None,
    env: Mapping[str, str] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> CollectResult:
    """Run one bounded cycle under the profile's lease, or report who already holds it.

    Does not check pause/enabled: the CLI layer does that first. ``now`` is the collection
    clock (``collected_at`` and window boundaries); ``clock`` is the monotonic budget clock.
    """
    try:
        lease = Lease.acquire(profile, "collect")
    except AlreadyRunning:
        return CollectResult("already_running", options.mode, progress=read_progress(profile))
    except (storage.UnsafeStorageError, plainfile.UnsafePathError) as error:
        raise _failure(error) from error
    try:
        moment = now or _now()
        progress = load_previous(profile) or Progress(profile, "collect")
        progress.kind = "collect"
        progress.pid = os.getpid()
        context = CycleContext(
            profile=profile,
            options=options,
            now=moment,
            clock=clock,
            should_stop=should_stop,
            reducer=reducer,
            env=env,
            progress=progress,
            on_step=on_step,
        )
        outcome = _run(progress, context, clock)
        return outcome.result
    finally:
        lease.release()


def run_worker_cycle(
    profile: str,
    *,
    options: CollectOptions | None = None,
    clock: Clock = time.monotonic,
    reducer: Reducer | None = None,
    env: Mapping[str, str] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> Callable[[StopCheck], CycleReport]:
    """The serve loop's cycle: the caller already holds the lease (kind ``serve``).

    One ``Progress`` lives across cycles so ``cycles`` and ``last_cycle`` stay cumulative.
    A failed cycle returns a ``failed`` report carrying the stable condition; the loop keeps
    ticking and the previous committed generation stays what readers see.
    """
    progress = load_previous(profile) or Progress(profile, "serve")
    progress.kind = "serve"
    progress.pid = os.getpid()
    chosen = options or CollectOptions()

    def cycle(should_stop: StopCheck) -> CycleReport:
        context = CycleContext(
            profile=profile,
            options=chosen,
            now=_now(),
            clock=clock,
            should_stop=should_stop,
            reducer=reducer,
            env=env,
            progress=progress,
            on_step=on_step,
        )
        try:
            return _run(progress, context, clock).report
        except CollectError as error:
            return _failed_report(error.condition, 0)

    return cycle
