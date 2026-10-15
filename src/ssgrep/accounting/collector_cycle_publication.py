"""Publish one collected cycle through its existing checkpoint acceptance point."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from ssgrep.accounting import (
    collector_published_retention as published_retention,
    collector_steps as steps,
    collector_store as store,
    snapshots,
)
from ssgrep.accounting.collector_assemble import SnapshotInputs, build_snapshot, utc_stamp
from ssgrep.accounting.collector_cycle_store import stage
from ssgrep.accounting.collector_frequency_gc import finish as finish_frequency_gc
from ssgrep.accounting.collector_inputs import journal_last_ts, journal_stats
from ssgrep.accounting.collector_modes import unavailable_periods
from ssgrep.accounting.collector_publish import (
    history_document,
    reuse_blocks,
    write_history,
    write_receipts,
)
from ssgrep.accounting.collector_types import CollectResult, CycleReport
from ssgrep.accounting.counters import backfill_series
from ssgrep.accounting.journal_retention import finish_journal_retention

if TYPE_CHECKING:
    from datetime import datetime

    from ssgrep.accounting.collector_cycle import CycleContext, CycleOutcome
    from ssgrep.accounting.collector_cycle_reduction import ReductionStage
    from ssgrep.accounting.collector_state import ReductionDoc
    from ssgrep.accounting.collector_types import CollectOptions


def publish_cycle(
    profile: str,
    checkpoint: store.Checkpoint,
    *,
    ctx: CycleContext,
    options: CollectOptions,
    now: datetime,
    today: str,
    started: float,
    phase: Callable[..., None],
    step: Callable[[str], None],
    backfill_done,
    prepared: ReductionStage,
    doc: ReductionDoc,
    executions,
    ledger,
    state,
    prior_state,
    counters,
    sealed,
    scope,
    frequencies,
    frequency,
    policy,
    methodology,
    expiry_out,
    journal,
    previous_journal_retention,
    horizon,
    evidence_job,
    outcome,
    new_items,
    migration,
    freq,
    segments,
    deadline,
    late,
) -> CycleOutcome:
    """Keep staging, commit hooks, maintenance and derived publication in order."""
    backfills = list(checkpoint.backfills)
    series = None
    if (
        backfill_done
        and not prepared.partitions.pending
        and not prepared.partitions.pending_projections
        and not prepared.partitions.unavailable
    ):
        since, until = checkpoint.backfill_job["since"], checkpoint.backfill_job["until"]
        trimmed = [
            replace(r, ts_utc=(r.ts_utc or "")[:19] + "Z") for r in doc.invocations if r.ts_utc
        ]
        series = backfill_series(trimmed, since, until, utc_stamp(now)).to_dict()
        backfills = [b for b in backfills if (b.get("since"), b.get("until")) != (since, until)]
        backfills.append(series)
        checkpoint.backfill_job = {}
    rows = ledger.rows()
    state = replace(
        state,
        ledger=ledger.state(),
        superseded=ledger.superseded_state(),
        counters=counters,
        sealed=sealed,
        reduction=doc,
        collection_scope=scope,
        frequency_assignments=frequencies.records(),
        archives=ledger.archives_state(),
        frequency_probe=frequency if not frequency.known else None,
    )
    generation = checkpoint.generation + 1
    phase("published_retention")
    blobs, retention, frequency_pending, publication_work = stage(
        profile,
        checkpoint,
        state=state,
        prior_state=prior_state,
        methodology=methodology,
        clock=ctx.clock,
        deadline=deadline,
        affected_roots=prepared.affected_roots,
        changed=prepared.changed,
        partitions=prepared.partitions,
        ledger=ledger,
        expiry_out=expiry_out,
        journal=journal,
        previous_journal_retention=previous_journal_retention,
        horizon=horizon,
        now=now,
        policy=policy,
        evidence_job=evidence_job,
        today=today,
        rows=rows,
        maintenance_budget=max(0, options.budget.bytes - outcome.bytes_read),
    )
    step("state_written")
    stopped = outcome.stopped_by
    if stopped == "complete" and (
        prepared.partitions.pending or prepared.partitions.pending_projections
    ):
        stopped = "reduction_budget"
    elif stopped == "complete" and prepared.partitions.unavailable:
        stopped = "replay_unavailable"
    elif stopped == "complete" and (
        evidence_job.get("pending") or evidence_job.get("compaction_pending")
    ):
        stopped = "evidence_maintenance_budget"
    if stopped == "complete" and state.root_migration_pending:
        stopped = "root_state_migration_budget"
    if stopped == "complete" and frequency_pending:
        stopped = "frequency_maintenance_pending"
    if stopped == "complete" and publication_work.pending:
        stopped = "published_retention_budget"
    if stopped == "complete" and ctx.clock() >= deadline:
        stopped = "cycle_budget_exceeded"
    from ssgrep.accounting.published_documents import merge_history

    published_history, publication_warnings = merge_history(
        profile, history_document(rows, methodology.units[0], today), blobs
    )
    envelope = build_snapshot(
        SnapshotInputs(
            profile=profile,
            now=now,
            generation=generation,
            policy=policy,
            methodology=methodology,
            executions=executions,
            sealed=sealed,
            rows=rows,
            counters=counters,
            backfills=backfills,
            reduction_coverage=doc.coverage,
            warnings=(*doc.warnings, *freq.standing_warnings(), *publication_warnings),
            frequency=freq.coverage(),
            sources=checkpoint.sources,
            journal=journal_stats(journal),
            journal_last_ts=journal_last_ts(journal),
            read_gaps=steps.source_gaps(checkpoint.sources),
            dropped=checkpoint.dropped,
            stopped_by=stopped,
            retention=expiry_out.coverage,
            since_day=checkpoint.since_day,
            baseline_ts=checkpoint.baseline_ts,
            suppressed=late,
            collection_scope=scope,
            scope_epoch=checkpoint.scope_epoch,
        )
    )
    step("before_stage")
    snapshots.stage_generation(profile, envelope, ctx.on_step)
    checkpoint.generation = generation
    checkpoint.cycles += 1
    checkpoint.methodology = methodology.hash
    checkpoint.input_digest = prepared.digest
    checkpoint.journal_sig = prepared.journal_sig
    checkpoint.segments = segments
    checkpoint.blobs = blobs
    checkpoint.journal = journal_stats(journal)
    checkpoint.retention = retention
    checkpoint.backfills = backfills
    step("before_commit")
    store.commit_checkpoint(profile, checkpoint)
    step("checkpoint_committed")
    frequency_maintenance = finish_frequency_gc(profile, generation=checkpoint.generation)
    maintenance = finish_journal_retention(
        profile,
        generation=checkpoint.generation,
        legacy_writers_stopped=options.journal_legacy_writers_stopped,
    )
    phase("publishing")
    reuse = reuse_blocks(ledger.state().candidates, methodology, ledger.superseded_state())
    write_receipts(profile, rows, reuse)
    write_history(profile, published_history)
    published_retention.finish(profile, publication_work)
    expired_before = steps.aggregates_expired_before(checkpoint)
    if expired_before is not None:
        steps.prune_receipt_files(profile, expired_before)
    step("receipts_written")
    snapshots.make_current(profile, generation, ctx.on_step)
    snapshots.prune(profile, keep=steps.KEEP_GENERATIONS)
    lag = steps.lag_bytes(envelope.coverage)
    report = CycleReport(
        status="published",
        generation=generation,
        snapshot_id=envelope.snapshot_id,
        complete=bool(envelope.coverage["complete"]),
        stopped_by=stopped,
        bytes_read=outcome.bytes_read,
        lag_bytes=lag,
        new_evidence=len(new_items),
        sources=len(checkpoint.sources),
        elapsed_ms=int((ctx.clock() - started) * 1000),
    )
    periods: tuple[str, ...] = ()
    if options.recompute:
        periods = unavailable_periods(
            since=options.since,
            until=options.until,
            horizon_day=horizon,
            retained_days={e.day for e in executions},
            frozen_days={leaf.day for f in ledger.state().frozen for leaf in f.leaves},
            sealed_days=set(sealed),
        )
    result = CollectResult(
        state="published" if stopped == "complete" and not checkpoint.backfill_job else "partial",
        mode=options.mode,
        report=report,
        epoch=counters.epoch,
        unavailable_periods=periods,
        backfill=series,
        warnings=(
            *(doc.warnings if doc else ()),
            *migration.warnings,
            *freq.standing_warnings(),
            *freq.events,
            *maintenance.warnings,
            *frequency_maintenance.warnings,
            *publication_warnings,
            *((maintenance.recovery_action,) if maintenance.recovery_action else ()),
        ),
    )
    from ssgrep.accounting.collector_cycle import CycleOutcome

    return CycleOutcome(report, result)
