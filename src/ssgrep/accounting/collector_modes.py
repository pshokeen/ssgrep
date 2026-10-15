"""Collection modes beyond the plain live read: pre-baselining, backfill reads, recompute.

* ``pre_baseline``: on the very first cycle, sources untouched since capture began are pure
  history. They are positioned at their end instantly (never read), so ordinary collection
  can never silently become a historical import.
* ``backfill_read``: the explicit, interval-bounded import. It keeps its own resumable job
  (per-source offsets) in the checkpoint, admits only evidence inside ``[since, until)``,
  and never touches live source offsets or live counters.
* ``unavailable_periods``: recompute reports periods whose retained frozen evidence is gone.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ssgrep.accounting.collector_baseline import baseline_record
from ssgrep.accounting.collector_inputs import AdmitRules, LazyLedger
from ssgrep.accounting.collector_read import Discovered, ReadOutcome, read_sources
from ssgrep.accounting.collector_store import SourceRecord
from ssgrep.accounting.collector_types import CollectError, CycleBudget
from ssgrep.accounting.retention_hooks import aggregate_horizon_day

MAX_REPORTED_DAYS = 3660


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


def _days(since: str | None, until: str | None, fallback: list[str]) -> list[str]:
    if since is None or until is None:
        return fallback
    start, end = date.fromisoformat(since[:10]), date.fromisoformat(until[:10])
    count = min((end - start).days + (until[10:] > "T00:00:00Z"), MAX_REPORTED_DAYS)
    return [(start + timedelta(days=n)).isoformat() for n in range(max(count, 0))]


def unavailable_periods(
    *,
    since: str | None,
    until: str | None,
    horizon_day: str,
    retained_days: set[str],
    frozen_days: set[str],
    sealed_days: set[str],
) -> tuple[str, ...]:
    """``<day>=<reason>`` for every closed period that cannot be recomputed from evidence.

    A day is unavailable when aggregates were retained for it (a frozen root or a sealed
    day) but its evidence is gone: frozen by expiry, or older than the evidence horizon with
    nothing retained.
    """
    candidates = sorted(frozen_days | sealed_days)
    out: list[str] = []
    for day in _days(since, until, candidates):
        if day in frozen_days:
            out.append(f"{day}=evidence_expired")
        elif day in sealed_days and day < horizon_day and day not in retained_days:
            out.append(f"{day}=evidence_expired")
    return tuple(out)


def require_interval(options: Any) -> tuple[str, str]:
    if not (options.since and options.until):
        raise CollectError(
            "backfill_requires_interval",
            "Backfill needs both --since and --until; nothing was imported.",
            "ssgrep metrics collect --backfill --since <date> --until <date>",
        )
    return options.since, options.until
