"""The read stage of a cycle: live read, explicit backfill, or nothing (recompute).

Recompute reads no source at all: it reduces what is already retained. A live read starts at
the capture baseline (never importing history), honors pause gaps and retention, and keeps its
own source watermarks; an explicit backfill keeps a separate resumable job and never touches
live source offsets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from ssgrep.accounting import collector_steps as steps
from ssgrep.accounting.collector_inputs import AdmitRules, LazyLedger
from ssgrep.accounting.collector_modes import (
    backfill_job_for,
    backfill_read,
    parse_stamp,
    pre_baseline,
    require_interval,
)
from ssgrep.accounting.collector_read import ReadOutcome, discover, make_readers, read_sources
from ssgrep.accounting.collector_store import Checkpoint
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.policy import policy_status

Phase = Callable[..., None]


def read_stage(
    *,
    profile: str,
    options: CollectOptions,
    now: datetime,
    clock: Callable[[], float],
    should_stop: Callable[[], bool],
    env: Mapping[str, str] | None,
    checkpoint: Checkpoint,
    ledger: LazyLedger,
    expired: frozenset[tuple[str, str]],
    policy: ProfilePolicy,
    horizon: str,
    phase: Phase,
) -> tuple[ReadOutcome, list[RawEvidence], bool]:
    """(outcome, newly accepted evidence, a backfill job just finished).

    Mutates ``checkpoint`` in memory only (sources / backfill job / cumulative drops); nothing
    is durable until the cycle commits it.
    """
    if options.recompute:
        return ReadOutcome(), [], False
    discovered = discover(make_readers(env))
    phase("reading", sources_total=len(discovered), sources_done=0)
    done = 0

    def on_source(key: str, record: Any) -> None:
        nonlocal done
        done += 1
        phase("reading", sources_done=done)

    def on_bytes(total: int) -> None:
        phase("reading", bytes_read=total)

    if options.backfill:
        since, until = require_interval(options)
        job = backfill_job_for(checkpoint.backfill_job, since, until)
        outcome, job = backfill_read(
            discovered=discovered,
            job=job,
            ledger=ledger,
            expired_sessions=expired,
            now=now,
            budget=options.budget,
            clock=clock,
            should_stop=should_stop,
            policy_horizon=policy,
            on_source=on_source,
            on_bytes=on_bytes,
        )
        checkpoint.backfill_job = job
        return outcome, outcome.evidence, job["status"] == "complete"
    known = dict(checkpoint.sources)
    pre_baseline(discovered, known, parse_stamp(checkpoint.baseline_ts) or now)
    rules = AdmitRules(
        paused=policy_status(profile).gaps,
        expired_sessions=expired,
        horizon_day=horizon,
        capture_start=checkpoint.baseline_ts,
    )
    outcome = read_sources(
        discovered=discovered,
        known=known,
        admit=rules,
        ledger=ledger,
        byte_budget=options.budget.bytes,
        time_budget=float(options.budget.seconds),
        clock=clock,
        should_stop=should_stop,
        on_source=on_source,
        on_bytes=on_bytes,
    )
    checkpoint.sources = dict(outcome.sources)
    checkpoint.dropped = steps.merge_dropped(checkpoint.dropped, outcome.dropped)
    return outcome, outcome.evidence, False
