"""The collection worker's public entry points (design D8, tasks 9.1-9.6).

``collect_once`` is the bounded one-shot used by ``savings collect``: it takes the profile's
exclusive lease (a concurrent collect or serve reports ``already_running`` with the owner's
progress instead of reducing twice), runs exactly one cycle, and releases the lease.
``run_worker_cycle`` is the cycle the foreground serve loop runs while it already holds the
lease. Both go through ``collector_cycle.run_cycle``; failures leave the previous committed
generation untouched and surface as a stable ``CollectError.condition`` (never raw text).
"""

from __future__ import annotations

import os
import re
import time
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.frequency import FrequencyView
from ssgrep.accounting.collect.replay import Ledger
from ssgrep.accounting.collect.store import CollectError, SourceRecord, dumps, loads
from ssgrep.accounting.ledger.contracts import ExecutionEvent, RawEvidence, digest_of
from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.ledger.valuation_inputs import Methodology
from ssgrep.accounting.privacy import storage, storage as plainfile
from ssgrep.accounting.serve.lease import AlreadyRunning, Lease

if TYPE_CHECKING:
    from ssgrep.accounting.collect.cycle import CycleContext, CycleOutcome
    from ssgrep.accounting.collect.read import Discovered, LazyLedger, ReadOutcome

MIB = 1024 * 1024
DEFAULT_CYCLE_SECONDS = 10
DEFAULT_CYCLE_BYTES = 32 * MIB
CAPTURE_GRACE_SECONDS = 300
MAX_REPORTED_DAYS = 3660
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class CycleBudget:
    """Cooperative per-cycle limits; an already-started bounded record may overshoot."""

    seconds: int = DEFAULT_CYCLE_SECONDS
    bytes: int = DEFAULT_CYCLE_BYTES


@dataclass(frozen=True)
class CycleReport:
    """What one finished collection cycle did (also the serve loop's view of it)."""

    status: str
    generation: int | None
    snapshot_id: str | None
    complete: bool
    stopped_by: str
    bytes_read: int = 0
    lag_bytes: int = 0
    new_evidence: int = 0
    sources: int = 0
    elapsed_ms: int = 0
    error: str | None = None
    configured_budget_seconds: int | None = None
    required_budget_seconds: int | None = None
    resumable: bool = field(init=False)
    next_action: str | None = field(init=False)

    def __post_init__(self) -> None:
        for name, value in collection_transition(self.stopped_by).items():
            object.__setattr__(self, name, value)


def collection_transition(stopped_by: str) -> dict[str, Any]:
    """Separate resumable work from accepted partial evidence needing intervention."""
    action = {
        "reduction_blocked": "increase_cycle_budget",
        "replay_unavailable": "restore_missing_sources",
        "backfill_blocked": "restore_backfill_sources",
    }.get(stopped_by)
    resumable = stopped_by != "complete" and action is None
    return {"resumable": resumable, "next_action": action or ("resume" if resumable else None)}


@dataclass(frozen=True)
class RetentionCoverage:
    """What retention has already removed (drives ``lifetime`` coverage statements)."""

    evidence_expired_before: str | None = None
    aggregates_expired_before: str | None = None
    frozen_roots: int = 0


@dataclass(frozen=True)
class CollectOptions:
    """Validated arguments of one ``savings collect`` run."""

    since: str | None = None
    until: str | None = None
    backfill: bool = False
    recompute: bool = False
    new_epoch: bool = False
    repin_frequency: bool = False
    budget: CycleBudget = CycleBudget()
    journal_legacy_writers_stopped: bool = False
    reset_frequency_authority: bool = False

    @property
    def mode(self) -> str:
        return "backfill" if self.backfill else ("recompute" if self.recompute else "live")


@dataclass(frozen=True)
class CollectResult:
    """Outcome of ``collect_once``: a published cycle or the live worker's progress."""

    state: str
    mode: str
    report: CycleReport | None = None
    progress: dict[str, object] | None = None
    epoch: int | None = None
    unavailable_periods: tuple[str, ...] = ()
    backfill: dict[str, object] | None = None
    warnings: tuple[str, ...] = ()
    pending_backfill: tuple[str, str] | None = None


@dataclass(frozen=True)
class Reducer:
    reduce_profile: Callable[..., Any]
    methodology_for: Callable[[FrequencyView], Methodology]

    def reduce_affected(self, profile, checkpoint, **kwargs):
        """The worker's incremental seam; retain the same pure reducer implementation."""
        from ssgrep.accounting.collect.partitions import reduce_partitions

        return reduce_partitions(profile, checkpoint, reducer=self, **kwargs)


def default_reducer() -> Reducer:
    from ssgrep.accounting.ledger import reduce as reduce_module

    return Reducer(reduce_module.reduce_profile, reduce_module.methodology_for)


PROGRESS_FILE = "progress.json"
MAX_PROGRESS_BYTES = 64 * 1024
_UTC = "%Y-%m-%dT%H:%M:%SZ"


def utc_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_UTC)


def last_cycle_document(report: CycleReport, finished_at: datetime) -> dict[str, Any]:
    return {
        "finished_at": utc_stamp(finished_at),
        "status": report.status,
        "stopped_by": report.stopped_by,
        "error": report.error,
        "generation": report.generation,
        "elapsed_ms": report.elapsed_ms,
        "bytes_read": report.bytes_read,
        "complete": report.complete,
        "resumable": report.resumable,
        "next_action": report.next_action,
        "configured_budget_seconds": report.configured_budget_seconds,
        "required_budget_seconds": report.required_budget_seconds,
    }


@dataclass
class Progress:
    """Mutable progress of the cycle in flight plus the last finished cycle."""

    profile: str
    kind: str
    pid: int = field(default_factory=os.getpid)
    phase: str = "idle"
    mode: str = "live"
    cycle_started_at: str | None = None
    updated_at: str = ""
    cycles: int = 0
    bytes_read: int = 0
    sources_done: int = 0
    sources_total: int = 0
    lag_bytes: int = 0
    last_cycle: dict[str, Any] | None = None
    sources_complete: int = 0
    sources_unvisited: int = 0
    sources_pending: int = 0
    rows_visited: int = 0
    pending_records: int | None = None
    abandoned_records: int = 0
    unsupported_sources: int = 0
    known_pending_lower_bound: int = 0
    pending_count_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "profile": self.profile,
            "pid": self.pid,
            "kind": self.kind,
            "phase": self.phase,
            "mode": self.mode,
            "cycle_started_at": self.cycle_started_at,
            "updated_at": self.updated_at,
            "cycles": self.cycles,
            "bytes_read": self.bytes_read,
            "sources_done": self.sources_done,
            "sources_total": self.sources_total,
            "lag_bytes": self.lag_bytes,
            "sources_complete": self.sources_complete,
            "sources_unvisited": self.sources_unvisited,
            "sources_pending": self.sources_pending,
            "rows_visited": self.rows_visited,
            "pending_records": self.pending_records,
            "abandoned_records": self.abandoned_records,
            "unsupported_sources": self.unsupported_sources,
            "known_pending_lower_bound": self.known_pending_lower_bound,
            "pending_count_complete": self.pending_count_complete,
            "last_cycle": self.last_cycle,
        }

    def write(self, now: datetime) -> None:
        """Persist atomically; progress is advisory, so a failed write never fails a cycle."""
        self.updated_at = utc_stamp(now)
        try:
            directory = storage.open_profile_dir(self.profile, "worker")
        except OSError:
            return
        try:
            storage.atomic_write(directory, PROGRESS_FILE, dumps(self.to_dict()) + b"\n")
        except OSError:
            pass
        finally:
            os.close(directory)


def load_previous(profile: str) -> Progress | None:
    """Carry cycle counters and ``last_cycle`` over when the next owner starts."""
    raw = read_progress(profile)
    if raw is None:
        return None
    last = raw.get("last_cycle")
    try:
        return Progress(
            profile=profile,
            kind=str(raw.get("kind", "collect")),
            cycles=int(raw.get("cycles", 0)),
            last_cycle=last if isinstance(last, dict) else None,
        )
    except (TypeError, ValueError):
        return None


def read_progress(profile: str) -> dict[str, Any] | None:
    """The progress document, or None when absent/unreadable/not an object (never raises)."""
    try:
        data = plainfile.read_private(
            storage.data_root(),
            "accounting",
            profile,
            "worker",
            PROGRESS_FILE,
            max_bytes=MAX_PROGRESS_BYTES,
        )
        if data is None:
            return None
        raw = loads(data)
    except (plainfile.UnsafePathError, ValueError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


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
    from ssgrep.accounting.collect.cycle import run_cycle

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
        from ssgrep.accounting.collect.cycle import CycleContext

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
    from ssgrep.accounting.collect.cycle import CycleContext

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


@dataclass(frozen=True)
class ExpiryPlan:
    """Which roots age out now (sorted, disjoint) and the horizons used to decide."""

    evidence_horizon_day: str
    aggregate_horizon_day: str
    expire_roots: tuple[str, ...]
    keep_roots: tuple[str, ...]


@dataclass(frozen=True)
class AggregateExpiry:
    """Exactly what ``expire_aggregates`` removed (all lists sorted, no duplicates)."""

    before_day: str
    removed_periods: tuple[str, ...]
    removed_history_rows: int
    removed_sealed_days: tuple[str, ...]

    @property
    def aggregates_expired_before(self) -> str | None:
        """The statement the snapshot may make; ``None`` when nothing was removed."""
        if self.removed_history_rows or self.removed_sealed_days:
            return self.before_day
        return None


def _utc_day(now: datetime, days: int) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if days < 0:
        raise ValueError("retention days must not be negative")
    return (now.astimezone(UTC) - timedelta(days=days)).date().isoformat()


def evidence_horizon_day(now: datetime, policy: ProfilePolicy) -> str:
    """UTC day ``evidence_retention_days`` before ``now``; earlier-dated evidence expires."""
    return _utc_day(now, policy.evidence_retention_days)


def aggregate_horizon_day(now: datetime, policy: ProfilePolicy) -> str:
    """UTC day ``receipt_retention_days`` before ``now``; earlier aggregates expire."""
    return _utc_day(now, policy.receipt_retention_days)


def _ts_day(ts: str) -> str:
    match = _TS.match(ts) if isinstance(ts, str) else None
    if match is None:
        raise ValueError("timestamp must be UTC YYYY-MM-DDTHH:MM:SS[.fff]Z")
    return match.group(1)


def _check_day(day: str) -> str:
    if not isinstance(day, str) or _DAY.match(day) is None:
        raise ValueError("day must be YYYY-MM-DD")
    return day


def plan_expiry(
    *, now: datetime, policy: ProfilePolicy, root_last_evidence: Mapping[str, str]
) -> ExpiryPlan:
    """Roots whose latest evidence of ANY record is dated before the evidence horizon."""
    horizon = evidence_horizon_day(now, policy)
    expire: list[str] = []
    keep: list[str] = []
    for root in sorted(root_last_evidence):
        (expire if _ts_day(root_last_evidence[root]) < horizon else keep).append(root)
    return ExpiryPlan(horizon, aggregate_horizon_day(now, policy), tuple(expire), tuple(keep))


def apply_expiry(ledger: Ledger, plan: ExpiryPlan) -> list[str]:
    """Freeze the planned roots the ledger actually knows; returns the roots frozen."""
    return ledger.expire_roots(plan.expire_roots)


def split_evidence(
    evidence: Iterable[RawEvidence], expired_sessions: AbstractSet[tuple[str, str]]
) -> tuple[list[RawEvidence], int]:
    """Drop records of expired conversations so a new tail is never reduced from a prefix."""
    kept: list[RawEvidence] = []
    dropped = 0
    for record in evidence:
        if (record.runtime, record.session_id) in expired_sessions:
            dropped += 1
        else:
            kept.append(record)
    return kept, dropped


def expire_journal_events(
    events: Iterable[ExecutionEvent], horizon_day: str
) -> tuple[list[ExecutionEvent], int]:
    """Retain current phases plus minimal starts of executions crossing the UTC floor."""
    from ssgrep.accounting.capture.journal_codec import minimal_started
    from ssgrep.accounting.ledger.contracts import Phase

    horizon = _check_day(horizon_day)
    items = list(events)
    live = {event.execution_id for event in items if _ts_day(event.ts_utc) >= horizon}
    kept: list[ExecutionEvent] = []
    dropped = 0
    for event in items:
        if _ts_day(event.ts_utc) >= horizon:
            kept.append(event)
        elif event.phase is Phase.STARTED and event.execution_id in live:
            kept.append(minimal_started(event))
        else:
            dropped += 1
    return kept, dropped


def expire_aggregates(
    history_rows: Sequence[Mapping[str, object]],
    sealed_days: Mapping[str, object],
    before_day: str,
) -> tuple[list[dict], dict, AggregateExpiry]:
    """Drop whole history rows / sealed days dated before ``before_day`` (aggregate expiry).

    This states what aggregate retention removed; it recomputes nothing. Rows keep their
    original order; the report names exactly the periods, row count and sealed days gone.
    """
    horizon = _check_day(before_day)
    kept_rows: list[dict] = []
    removed_periods: set[str] = set()
    removed_rows = 0
    for row in history_rows:
        period = row.get("period")
        if not isinstance(period, str):
            raise ValueError("history row without a period")
        if _check_day(period) < horizon:
            removed_rows += 1
            removed_periods.add(period)
        else:
            kept_rows.append(dict(row))
    kept_days = {day: v for day, v in sealed_days.items() if _check_day(day) >= horizon}
    removed_days = sorted(day for day in sealed_days if day not in kept_days)
    report = AggregateExpiry(
        horizon, tuple(sorted(removed_periods)), removed_rows, tuple(removed_days)
    )
    return kept_rows, kept_days, report


def _later(first: str | None, second: str | None) -> str | None:
    present = [d for d in (first, second) if d is not None]
    return max(present) if present else None


def coverage_after(
    previous: RetentionCoverage,
    plan: ExpiryPlan,
    frozen: Sequence[str],
    aggregates_before: str | None,
    total_frozen: int,
) -> RetentionCoverage:
    """New coverage: horizons only move forward; ``frozen_roots`` is the running total.

    ``evidence_expired_before`` advances to the plan's horizon only when roots were
    actually frozen this cycle (a horizon that removed nothing states nothing).
    """
    if total_frozen < len(set(frozen)):
        raise ValueError("total_frozen cannot be less than the roots frozen now")
    evidence = _later(
        previous.evidence_expired_before, plan.evidence_horizon_day if frozen else None
    )
    aggregates = _later(previous.aggregates_expired_before, aggregates_before)
    return RetentionCoverage(evidence, aggregates, total_frozen)


KEY = "published_retention"
PREFIX = "identity-archive-"


@dataclass(frozen=True)
class Work:
    pending: bool = False
    changed: bool = False
    delete_receipts: tuple[str, ...] = ()


def _signature(checkpoint):
    return digest_of(
        sorted((name, value) for name, value in checkpoint.blobs.items() if name.startswith(PREFIX))
    )


def _size(profile, name, sub=store.STATE_DIR):
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\0" in name:
        raise CollectError("state_corrupt", "Original retention child reference is invalid.")
    directory = storage.open_profile_dir(profile, "collector", sub, create=False)
    try:
        descriptor = storage.open_regular(directory, name)
        try:
            return os.fstat(descriptor).st_size
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _remaining_public(profile, rows, receipt_ids):
    """Check <=32 names; canonical receipt tuples are ordered by ID and revision."""
    from ssgrep.accounting.views.records import RECEIPT_ID_RE

    if not isinstance(receipt_ids, (tuple, list)) or len(receipt_ids) > 32:
        raise CollectError("state_corrupt", "Original receipt cleanup intent is invalid.")
    if not receipt_ids:
        return ()
    groups = (
        (rows.receipts, *(archive.receipts for archive in rows.archives))
        if rows is not None
        else ()
    )
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return ()
    remaining = []
    try:
        for receipt in receipt_ids:
            if not isinstance(receipt, str) or RECEIPT_ID_RE.fullmatch(receipt) is None:
                raise CollectError("state_corrupt", "Original receipt cleanup identity is invalid.")
            protected = False
            for group in groups:
                position = bisect_left(group, receipt, key=lambda row: row.receipt_id)
                if position < len(group) and group[position].receipt_id == receipt:
                    protected = True
                    break
            if protected:
                continue
            try:
                descriptor = storage.open_regular(directory, receipt + ".json")
            except FileNotFoundError:
                continue
            os.close(descriptor)
            remaining.append(receipt)
    finally:
        os.close(directory)
    return tuple(remaining)


def _clean(documents, key, source, cutoff):
    from ssgrep.accounting.privacy.published import day

    indexes = documents.get("retention_child_dates")
    index = indexes.get(key) if isinstance(indexes, dict) else None
    if not isinstance(index, dict) or index.get("schema") != 1 or index.get("source") != source:
        return False
    checked = day(index.get("clean_before"))
    if checked is not None and cutoff <= checked:
        return True
    minimum = index.get("minimum")
    return index.get("complete") is True and (
        minimum is None or day(minimum) is not None and cutoff <= minimum
    )


def _next(manifest, name, job):
    """Advance one empty/clean metadata edge, or return the next data quantum."""
    from ssgrep.accounting.privacy.published import day

    documents, phase, after = manifest["published_documents"], job["phase"], job["after"]
    if phase == "raw":
        public_receipts = documents.get("receipt_blobs", {}).values()
        candidates = sorted(
            (key, value, store.STATE_DIR)
            for key, value in manifest.get("blobs", {}).items()
            if key > after
            and not key.startswith("published-")
            and value != documents.get("history_blob")
            and value not in public_receipts
        )
        field, following = "blob_names", "segments"
    elif phase == "segments":
        candidates = sorted(
            (ref["name"], ref["name"], store.EVIDENCE_DIR)
            for ref in manifest.get("segments", [])
            if ref["name"] > after
        )
        field, following = "segment_names", "history"
    elif phase == "history":
        candidates = (
            [("history", documents["history_blob"], store.STATE_DIR)]
            if documents.get("history_blob")
            else []
        )
        field, following = "history", "receipts"
    else:
        minimum = day(documents.get("receipt_min_period"))
        receipts = documents.get("receipt_blobs", {})
        if (
            minimum is not None
            and minimum >= job["before_day"]
            and len(documents.get("receipt_periods", {})) == len(receipts)
        ):
            job.update(archive_after=name, phase="raw", after="")
            return None
        candidates = sorted(
            (key, value, store.STATE_DIR) for key, value in receipts.items() if key > after
        )
        field, following = "receipt_ids", None
    if not candidates:
        if following:
            job.update(phase=following, after="")
        else:
            job.update(archive_after=name, phase="raw", after="")
        return None
    first, source, _ = candidates[0]
    cutoff = (
        job["evidence_before_day"]
        if phase == "raw" and first.split("-scope-", 1)[0] == "counters"
        else job["before_day"]
    )
    cache_key = (
        "blob:" + first
        if phase == "raw"
        else "segment:" + first
        if phase == "segments"
        else "history"
    )
    if phase != "receipts" and _clean(documents, cache_key, source, cutoff):
        if phase == "history":
            job.update(phase="receipts", after="")
        else:
            job["after"] = first
        return None
    return field, candidates[: 32 if phase == "receipts" else 1], cutoff


def prepare(
    profile, checkpoint, *, rows, before_day, evidence_before_day, clock, deadline, byte_budget
):
    """One data quantum, <=64 metadata edges, and a <=32-ID postcommit cleanup intent."""
    from ssgrep.accounting.privacy.published import day
    from ssgrep.accounting.privacy.retention import stage_children

    names = sorted(name for name in checkpoint.blobs if name.startswith(PREFIX))
    old = checkpoint.retention.get(KEY, {})
    if not names and not old.get("cleanup_ids"):
        return Work()
    signature = _signature(checkpoint)
    if (
        old.get("before_day") == before_day
        and old.get("evidence_before_day") == evidence_before_day
        and old.get("signature") == signature
    ):
        job = dict(old)
        if not job.get("pending") and not job.get("cleanup_ids"):
            return Work()
    else:
        job = {
            "before_day": before_day,
            "evidence_before_day": evidence_before_day,
            "signature": signature,
            "archive_after": "",
            "phase": "raw",
            "after": "",
            "pending": bool(names),
            "cleanup_ids": old.get("cleanup_ids", []),
        }
    queued = _remaining_public(profile, rows, job.get("cleanup_ids", []))
    job["cleanup_ids"] = list(queued)
    if queued:
        ready = clock() < deadline
        checkpoint.retention[KEY] = job
        return Work(bool(job["pending"] or not ready), False, queued if ready else ())
    changed, consumed, expired_receipts, used_data, edges = False, 0, (), False, 0
    job.pop("blocked_child_bytes", None)
    stop = False
    while job["pending"] and edges < 64 and clock() < deadline and not stop:
        name = next((name for name in names if name > job["archive_after"]), None)
        if name is None:
            job["pending"] = False
            break
        size = _size(profile, checkpoint.blobs[name])
        if consumed + size > byte_budget:
            job["blocked_child_bytes"] = size
            break
        consumed += size
        manifest = store.read_blob(profile, checkpoint.blobs[name])
        if not isinstance(manifest, dict):
            raise CollectError("state_corrupt", "Original publication manifest is invalid.")
        edges += 1
        if manifest.get("published_documents") is None:
            job.update(archive_after=name, phase="raw", after="")
            continue
        while edges < 64 and clock() < deadline and job["archive_after"] != name:
            candidate = _next(manifest, name, job)
            edges += 1
            if candidate is None:
                continue
            if used_data:
                stop = True
                break
            field, candidates, cutoff = candidate
            selected = []
            documents = manifest["published_documents"]
            for key, filename, sub in candidates:
                indexed = (
                    field == "receipt_ids"
                    and day(documents.get("receipt_periods", {}).get(key)) is not None
                    and isinstance(documents.get("receipt_roots", {}).get(key), str)
                )
                size = (
                    0
                    if indexed
                    else _size(profile, filename, sub) * (2 if field == "segment_names" else 1)
                )
                if consumed + size > byte_budget or clock() >= deadline:
                    job["blocked_child_bytes"] = size
                    break
                selected.append(key)
                consumed += size
            if not selected:
                stop = True
                break
            result = stage_children(
                profile,
                checkpoint,
                name=name,
                manifest=manifest,
                before_day=cutoff,
                **{field: True if field == "history" else tuple(selected)},
            )
            manifest = result.manifest
            changed = changed or result.changed
            expired_receipts = result.expired_receipts
            used_data = True
            if field == "history":
                job.update(phase="receipts", after="")
            else:
                job["after"] = selected[-1]
        if job["archive_after"] == name and not any(other > name for other in names):
            job["pending"] = False
    deletions = _remaining_public(profile, rows, expired_receipts)
    job.update(
        cleanup_ids=list(deletions),
        signature=_signature(checkpoint),
        bytes_read_bound=consumed,
        metadata_edges=edges,
    )
    checkpoint.retention[KEY] = job
    return Work(bool(job["pending"]), changed, deletions)


def finish(profile, work):
    """Only explicitly proven stale public receipt files are unlinked after commit."""
    from ssgrep.accounting.privacy.purge_files import delete_receipts

    return delete_receipts(profile, set(work.delete_receipts)) if work.delete_receipts else 0


def parse_stamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def pre_baseline(
    discovered: Discovered, known: dict[str, SourceRecord], capture_start: datetime
) -> int:
    """Baseline new sources not modified since capture began; returns how many were skipped."""
    from ssgrep.accounting.collect.read import baseline_record

    skipped = 0
    cutoff = int(capture_start.timestamp() * 1_000_000_000)
    for key, (reader, source) in discovered.items():
        if key in known or reader.runtime == "opencode":
            continue
        try:
            modified = os.stat(source.path).st_mtime_ns
        except OSError:
            continue
        if modified < cutoff:
            known[key] = baseline_record(reader, source)
            skipped += 1
    return skipped


def backfill_job_for(job: Mapping[str, Any], since: str, until: str) -> dict[str, Any]:
    """The persisted job when it is for this exact interval; otherwise a fresh one."""
    if job.get("since") == since and job.get("until") == until:
        return dict(job)
    return {"since": since, "until": until, "status": "running", "sources": {}}


def backfill_read(
    *,
    discovered: Discovered,
    job: dict[str, Any],
    ledger: LazyLedger,
    expired_sessions: frozenset[tuple[str, str]],
    now: datetime,
    budget: CycleBudget,
    clock: Callable[[], float],
    should_stop: Callable[[], bool],
    policy_horizon: Any,
    on_source: Callable[[str, SourceRecord], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
) -> tuple[ReadOutcome, dict[str, Any]]:
    """Read the job's interval from each source; returns the outcome and the updated job."""
    from ssgrep.accounting.collect.read import AdmitRules, read_sources

    # aggregate_horizon_day is local so mode helpers stay in the collector owner.

    known = {k: SourceRecord.from_dict(v) for k, v in job.get("sources", {}).items()}
    rules = AdmitRules(
        expired_sessions=expired_sessions,
        horizon_day=aggregate_horizon_day(now, policy_horizon),
        window=(job["since"], job["until"]),
    )
    outcome = read_sources(
        discovered=discovered,
        known=known,
        admit=rules,
        ledger=ledger,
        byte_budget=budget.bytes,
        time_budget=float(budget.seconds),
        clock=clock,
        should_stop=should_stop,
        baseline_new=False,
        on_source=on_source,
        on_bytes=on_bytes,
    )
    finished = outcome.stopped_by == "complete" and all(
        r.complete for r in outcome.sources.values()
    )
    updated = dict(job)
    updated["sources"] = {k: r.to_dict() for k, r in sorted(outcome.sources.items())}
    blocked = any(
        r.status in {"missing", "unreadable", "unsupported"} for r in outcome.sources.values()
    )
    updated["status"] = "complete" if finished else "blocked" if blocked else "running"
    return outcome, updated


def require_interval(options: Any) -> tuple[str, str]:
    if not (options.since and options.until):
        raise CollectError(
            "backfill_requires_interval",
            "Backfill needs both positional FROM and TO dates; nothing was imported.",
            "ssgrep savings collect --backfill <from> <to>",
        )
    return options.since, options.until
