"""The post-purge snapshot generation.

Aggregate snapshots cannot be split per workspace or date, so every older generation is deleted.
A collection cycle insists that the generation its checkpoint names exists on disk, so simply
deleting them all would break the next ``metrics collect``. The purge therefore builds ONE new
generation (``checkpoint.generation + 1``) from the already purged state with the collector's own
``build_snapshot``; it is staged before the checkpoint commit and made current after it, and
nothing older survives. Generations stay monotonic, which acknowledged delivery state relies on.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ssgrep.accounting import snapshots
from ssgrep.accounting.collector_assemble import SnapshotInputs, build_snapshot
from ssgrep.accounting.collector_expiry import previous_coverage
from ssgrep.accounting.collector_inputs import journal_last_ts, journal_stats
from ssgrep.accounting.collector_state import ReductionDoc
from ssgrep.accounting.collector_steps import source_gaps
from ssgrep.accounting.collector_store import Checkpoint
from ssgrep.accounting.counters import empty_counters
from ssgrep.accounting.journal_reader import read_journal
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope
from ssgrep.accounting.policy import load_profile_policy
from ssgrep.accounting.purge_state import Namespace, StatePlan, sealed_after, stamp_of


def build(profile: str, plan: StatePlan, checkpoint: Checkpoint, now: datetime) -> SnapshotEnvelope:
    """The purged state as the next generation (``checkpoint`` is the NEW checkpoint value)."""
    current: Namespace | None = plan.current
    if current is None:
        raise ValueError("no ledger of the current methodology")
    journal = read_journal(profile, quarantine=False)
    doc: ReductionDoc | None = plan.reduction_after
    counters = plan.state.counters or empty_counters(1, stamp_of(now.astimezone(UTC)))
    return build_snapshot(
        SnapshotInputs(
            profile=profile,
            now=now,
            generation=checkpoint.generation,
            policy=load_profile_policy(profile),
            methodology=current.after.methodology,
            executions=doc.executions if doc else (),
            sealed=sealed_after(plan),
            rows=current.rows_after,
            counters=counters,
            backfills=checkpoint.backfills,
            reduction_coverage=doc.coverage if doc else {},
            warnings=(*(doc.warnings if doc else ()), "purged: older data was deleted on request"),
            sources=checkpoint.sources,
            journal=journal_stats(journal),
            journal_last_ts=journal_last_ts(journal),
            read_gaps=source_gaps(checkpoint.sources),
            dropped=checkpoint.dropped,
            stopped_by="complete",
            retention=previous_coverage(checkpoint.retention),
            since_day=checkpoint.since_day,
            baseline_ts=checkpoint.baseline_ts,
            collection_scope=checkpoint.collection_scope,
            scope_epoch=checkpoint.scope_epoch,
        )
    )


def stage(profile: str, envelope: SnapshotEnvelope) -> None:
    snapshots.stage_generation(profile, envelope)


def make_current(profile: str, generation: int) -> None:
    snapshots.make_current(profile, generation)
