"""Read-only worker, lag, missing-evidence and storage readers for ``savings status``.

Inputs (all existing files, all bounded and tolerant): the lease metadata
(``lease.inspect_lease``; the flock itself is never taken), ``worker/progress.json``
and the ``coverage`` block of the current snapshot. Anything absent, corrupt or
hostile becomes ``None``/``unknown`` with a stable reason: never an exception and
never a zero standing in for "unknown". No path, transcript text or free-form worker
string leaves this module; every string is validated against a closed set or a
strict pattern first.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ssgrep.accounting.privacy import storage
from ssgrep.accounting.serve.lease import inspect_lease

PROGRESS_MAX_BYTES = 65536
MAX_WATERMARKS = 5000
MAX_RAW_WATERMARKS = 100_000
MAX_RUNTIMES = 64
MAX_GAP_REASONS = 64
MAX_WALK_ENTRIES = 200_000

MIB = 1024 * 1024
GIB = 1024 * MIB
CRITICAL_FREE_BYTES = 100 * MIB  # free space below this: level "critical"
HIGH_FREE_BYTES = 1 * GIB  # free space below this: level "high"
HIGH_TOTAL_BYTES = 1 * GIB  # accounting data above this: level "high"

KINDS = ("collect", "serve")
PHASES = ("idle", "reading", "reducing", "publishing")
MODES = ("live", "backfill", "recompute")
STOPPED_BY = (
    "complete",
    "byte_budget",
    "time_budget",
    "stop_requested",
    "reduction_budget",
    "reduction_blocked",
    "replay_unavailable",
    "backfill_pending",
    "backfill_blocked",
    "evidence_maintenance_budget",
    "root_state_migration_budget",
    "frequency_maintenance_pending",
    "published_retention_budget",
    "cycle_budget_exceeded",  # older persisted reports remain interpretable
)
CYCLE_STATUSES = ("published", "failed")
UNRECOGNIZED = "unrecognized"
COMPONENTS = (
    ("journal", ("journal", "journal-quarantine")),
    ("collector", ("collector",)),
    ("snapshots", ("snapshots",)),
    ("receipts", ("receipts",)),
    ("history", ("history",)),
)

_HEX16 = re.compile(r"[0-9a-f]{16}")
_TOKEN = re.compile(r"[a-z][a-z0-9_.-]{0,47}")
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?")


@dataclass(frozen=True)
class LastCycle:
    finished_at: str | None
    status: str
    stopped_by: str | None
    error: str | None  # a stable condition token, never raw exception text
    generation: int | None
    elapsed_ms: int | None
    bytes_read: int | None
    complete: bool | None


@dataclass(frozen=True)
class Progress:
    pid: int | None
    kind: str | None
    phase: str | None
    mode: str | None
    cycle_started_at: str | None
    updated_at: str | None
    cycles: int | None
    bytes_read: int | None
    sources_done: int | None
    sources_total: int | None
    lag_bytes: int | None
    last_cycle: LastCycle | None


@dataclass(frozen=True)
class WorkerStatus:
    state: str  # running | idle | never_run | stale_lease | failed_last_cycle | unknown
    reason: str | None = None
    kind: str | None = None
    pid: int | None = None
    acquired_at: str | None = None
    recoverable: bool = False  # abandoned lease: the next worker takes over
    progress: Progress | None = None


@dataclass(frozen=True)
class RuntimeLag:
    runtime: str
    sources: int | None
    processed_bytes: int | None
    total_bytes: int | None
    lag_bytes: int | None
    missing: int | None


@dataclass(frozen=True)
class Watermark:
    source: str  # hash of the source id, never a path
    runtime: str
    offset: int
    size: int
    lag_bytes: int
    pending_tail_bytes: int
    status: str


@dataclass(frozen=True)
class LagStatus:
    reason: str | None
    lag_bytes: int | None
    basis: str | None  # snapshot | progress
    sources_total: int | None
    sources_behind: int | None
    stopped_by: str | None
    by_runtime: tuple[RuntimeLag, ...]
    watermarks: tuple[Watermark, ...] | None
    watermark_count: int | None
    watermarks_truncated: bool
    ignored_entries: int


@dataclass(frozen=True)
class MissingEvidence:
    reason: str | None
    missing_sources: int | None
    gaps: tuple[tuple[str, int], ...] | None
    incomplete_tails: int | None
    quarantined: int | None
    conflicts: int | None
    evidence_expired_before: str | None
    frozen_roots: int | None
    capture_completeness: str = "unknown"  # a completed snapshot never proves complete capture


@dataclass(frozen=True)
class StorageComponent:
    name: str
    bytes: int | None
    reason: str | None


@dataclass(frozen=True)
class StoragePressure:
    level: str | None  # ok | high | critical; None when it cannot be judged
    level_reason: str | None
    free_bytes: int | None
    total_bytes: int | None
    components: tuple[StorageComponent, ...]
    reason: str | None


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _choice(value: object, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and value in allowed else UNRECOGNIZED


def _stamp(value: object) -> str | None:
    return value if isinstance(value, str) and _STAMP.fullmatch(value) else None


def _token(value: object) -> str | None:
    return value if isinstance(value, str) and _TOKEN.fullmatch(value) else None


def _obj(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _reject_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _last_cycle(raw: object) -> LastCycle | None:
    if not isinstance(raw, dict):
        return None
    error = raw.get("error")
    return LastCycle(
        finished_at=_stamp(raw.get("finished_at")),
        status=_choice(raw.get("status"), CYCLE_STATUSES) or UNRECOGNIZED,
        stopped_by=_choice(raw.get("stopped_by"), STOPPED_BY),
        error=None if error is None else (_token(error) or "unrecognized_error"),
        generation=_int(raw.get("generation")),
        elapsed_ms=_int(raw.get("elapsed_ms")),
        bytes_read=_int(raw.get("bytes_read")),
        complete=raw["complete"] if isinstance(raw.get("complete"), bool) else None,
    )


def read_progress(profile: str) -> tuple[Progress | None, str | None]:
    """(progress, None) | (None, None) when absent | (None, reason) when unreadable."""
    try:
        raw = storage.read_private(
            storage.data_root(),
            storage.ACCOUNTING_DIR,
            profile,
            "worker",
            "progress.json",
            max_bytes=PROGRESS_MAX_BYTES,
        )
        if raw is None:
            return None, None
        document = json.loads(raw, parse_float=_reject_float)
    except (OSError, ValueError, RecursionError):
        return None, "progress_unreadable"
    if not isinstance(document, dict) or document.get("schema") != 1:
        return None, "progress_unsupported_schema"
    return (
        Progress(
            pid=_int(document.get("pid")),
            kind=_choice(document.get("kind"), KINDS),
            phase=_choice(document.get("phase"), PHASES),
            mode=_choice(document.get("mode"), MODES),
            cycle_started_at=_stamp(document.get("cycle_started_at")),
            updated_at=_stamp(document.get("updated_at")),
            cycles=_int(document.get("cycles")),
            bytes_read=_int(document.get("bytes_read")),
            sources_done=_int(document.get("sources_done")),
            sources_total=_int(document.get("sources_total")),
            lag_bytes=_int(document.get("lag_bytes")),
            last_cycle=_last_cycle(document.get("last_cycle")),
        ),
        None,
    )


def worker_status(
    profile: str, progress: Progress | None, problem: str | None, *, has_snapshot: bool
) -> WorkerStatus:
    """Truth order: a recorded live owner, an abandoned owner, then the last cycle's record."""
    try:
        owner = inspect_lease(profile)
    except OSError:
        return WorkerStatus("unknown", "lease_unreadable", progress=progress)
    if owner is not None and owner.pid is not None:
        mismatch = progress is not None and progress.pid not in (None, owner.pid)
        kept = None if mismatch else progress
        reason = "progress_not_from_this_worker" if mismatch else problem
        state = WorkerStatus(
            "running" if owner.alive else "stale_lease",
            reason if owner.alive else "owner_not_running",
            kind=_choice(owner.kind, KINDS),
            pid=owner.pid,
            acquired_at=_stamp(owner.acquired_at),
            recoverable=not owner.alive,
            progress=kept,
        )
        return state
    if problem is not None:
        return WorkerStatus("unknown", problem)
    if progress is None:
        if has_snapshot:
            return WorkerStatus("unknown", "no_progress_record")
        return WorkerStatus("never_run", "no_worker_record")
    last = progress.last_cycle
    if last is None:
        return WorkerStatus("never_run", "no_cycle_finished", progress=progress)
    if last.status == "failed":
        return WorkerStatus("failed_last_cycle", "last_cycle_failed", progress=progress)
    return WorkerStatus("idle", progress=progress)


def _runtimes(raw: object) -> tuple[RuntimeLag, ...]:
    table = _obj(raw)
    rows = []
    for runtime in sorted(table)[:MAX_RUNTIMES]:
        name, entry = _token(runtime), _obj(table[runtime])
        if name is not None:
            rows.append(
                RuntimeLag(
                    name,
                    _int(entry.get("sources")),
                    _int(entry.get("processed_bytes")),
                    _int(entry.get("total_bytes")),
                    _int(entry.get("lag_bytes")),
                    _int(entry.get("missing")),
                )
            )
    return tuple(rows)


def _watermark(raw: object) -> Watermark | None:
    entry = _obj(raw)
    source, runtime, status = entry.get("source"), _token(entry.get("runtime")), entry.get("status")
    offset, size = _int(entry.get("offset")), _int(entry.get("size"))
    lag, tail = _int(entry.get("lag_bytes")), _int(entry.get("pending_tail_bytes"))
    if not (isinstance(source, str) and _HEX16.fullmatch(source) and runtime):
        return None
    if offset is None or size is None or lag is None or tail is None:
        return None
    if status not in ("ok", "missing"):
        return None
    return Watermark(source, runtime, offset, size, lag, tail, status)


def lag_status(
    coverage: Mapping[str, Any] | None, why: str | None, progress: Progress | None
) -> LagStatus:
    """Lag and watermarks from the published snapshot; live progress only fills a missing lag."""
    coverage = coverage or {}
    sources = _obj(coverage.get("sources"))
    lag = _int(sources.get("lag_bytes"))
    lag = _int(coverage.get("lag_bytes")) if lag is None else lag
    basis = "snapshot" if lag is not None else None
    if lag is None and progress is not None and progress.lag_bytes is not None:
        lag, basis = progress.lag_bytes, "progress"
    raw = coverage.get("watermarks")
    marks: list[Watermark] = []
    ignored = behind = 0
    if isinstance(raw, list):
        for item in raw[:MAX_RAW_WATERMARKS]:
            mark = _watermark(item)
            if mark is None:
                ignored += 1
                continue
            marks.append(mark)
            behind += mark.lag_bytes > 0
        ignored += max(len(raw) - MAX_RAW_WATERMARKS, 0)
    known = isinstance(raw, list)
    marks.sort(key=lambda m: (m.source, m.runtime))
    reason = None
    if lag is None:
        reason = why or "lag_not_recorded"
    return LagStatus(
        reason=reason,
        lag_bytes=lag,
        basis=basis,
        sources_total=_int(sources.get("total")),
        sources_behind=behind if known else None,
        stopped_by=_choice(coverage.get("stopped_by"), STOPPED_BY),
        by_runtime=_runtimes(sources.get("by_runtime")),
        watermarks=tuple(marks[:MAX_WATERMARKS]) if known else None,
        watermark_count=len(marks) if known else None,
        watermarks_truncated=len(marks) > MAX_WATERMARKS,
        ignored_entries=ignored,
    )


def missing_evidence(coverage: Mapping[str, Any] | None, why: str | None) -> MissingEvidence:
    """Gaps and journal damage as recorded by the collector; unknown stays unknown."""
    coverage = coverage or {}
    journal, retention = _obj(coverage.get("journal")), _obj(coverage.get("retention"))
    raw_gaps = coverage.get("gaps")
    gaps = None
    if isinstance(raw_gaps, dict):
        pairs = [(g, n) for g, n in raw_gaps.items() if _token(g) and _int(n) is not None]
        gaps = tuple(sorted(pairs)[:MAX_GAP_REASONS])
    expired = retention.get("evidence_expired_before")
    unknown = _int(_obj(coverage.get("sources")).get("missing")) is None or gaps is None
    return MissingEvidence(
        reason=(why or "coverage_not_recorded") if unknown else None,
        missing_sources=_int(_obj(coverage.get("sources")).get("missing")),
        gaps=gaps,
        incomplete_tails=_int(journal.get("incomplete_tails")),
        quarantined=_int(journal.get("quarantined")),
        conflicts=_int(journal.get("conflicts")),
        evidence_expired_before=(
            expired if isinstance(expired, str) and _DAY.fullmatch(expired) else None
        ),
        frozen_roots=_int(retention.get("frozen_roots")),
    )


class _WalkLimit(Exception):
    pass


def _tree_bytes(root: Path, budget: list[int]) -> int:
    """Regular-file bytes under ``root``; symlinks are neither followed nor counted."""
    if not os.path.isdir(root) or os.path.islink(root):
        return 0
    total, stack = 0, [root]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                budget[0] -= 1
                if budget[0] < 0:
                    raise _WalkLimit
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    continue  # removed while walking
    return total


def _component(
    base: Path, name: str, subdirs: tuple[str, ...], budget: list[int]
) -> StorageComponent:
    try:
        return StorageComponent(name, sum(_tree_bytes(base / s, budget) for s in subdirs), None)
    except _WalkLimit:
        return StorageComponent(name, None, "walk_limit_exceeded")
    except OSError:
        return StorageComponent(name, None, "unreadable")


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _level(free: int | None, total: int | None) -> tuple[str | None, str | None]:
    if free is not None and free < CRITICAL_FREE_BYTES:
        return "critical", "free_space_below_100_mib"
    if free is not None and free < HIGH_FREE_BYTES:
        return "high", "free_space_below_1_gib"
    if total is not None and total > HIGH_TOTAL_BYTES:
        return "high", "accounting_data_above_1_gib"
    return ("ok", None) if free is not None else (None, "free_space_unknown")


def storage_pressure(
    profile: str,
    *,
    free_bytes_fn: Callable[[Path], int] = _free_bytes,
    entry_limit: int = MAX_WALK_ENTRIES,
) -> StoragePressure:
    """Bytes per component, free space on the data volume, and a documented pressure level.

    Component sizes come from a bounded ``scandir`` walk of the profile directory that
    counts regular files only. ``free_bytes_fn`` (given the nearest existing ancestor of
    the data root) is injectable so thresholds are testable without a full disk.
    """
    root = storage.data_root()
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free: int | None = free_bytes_fn(probe)
    except OSError:
        free = None
    base = root / storage.ACCOUNTING_DIR / profile
    reason = None
    try:
        for directory in (root / storage.ACCOUNTING_DIR, base):  # a symlinked dir is not walked
            if not stat.S_ISDIR(os.lstat(directory).st_mode):
                reason = "unsafe_path"
    except FileNotFoundError:
        reason = "not_created_yet"
    except OSError:
        reason = "unreadable"
    if reason is not None:
        parts = tuple(StorageComponent(name, None, reason) for name, _ in COMPONENTS)
        level, level_reason = _level(free, None)
        return StoragePressure(level, level_reason, free, None, parts, reason)
    budget = [entry_limit]
    parts = tuple(_component(base, name, subdirs, budget) for name, subdirs in COMPONENTS)
    unknown = next((c.reason for c in parts if c.bytes is None), None)
    total = None if unknown else sum(c.bytes or 0 for c in parts)
    level, level_reason = _level(free, total)
    return StoragePressure(level, level_reason, free, total, parts, unknown)
