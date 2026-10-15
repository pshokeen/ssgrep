"""Collection's affected-closure phase before counters and retention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_partitions import PartitionState, load_partitions

if TYPE_CHECKING:
    from datetime import datetime

    from ssgrep.accounting.collector_cycle import CycleContext
    from ssgrep.accounting.collector_reduce import Reducer
    from ssgrep.accounting.collector_state import ReductionDoc
    from ssgrep.accounting.collector_types import CollectOptions
    from ssgrep.accounting.journal_reader import JournalRead


class ReductionStage(NamedTuple):
    changed: bool
    affected_roots: set[str]
    doc: ReductionDoc
    last: dict[tuple[str, str], str]
    segments: list[store.SegmentRef]
    scoped_journal: JournalRead
    partitions: PartitionState
    journal_sig: str
    digest: str


def prepare_reduction(
    profile: str,
    checkpoint: store.Checkpoint,
    *,
    ctx: CycleContext,
    reducer: Reducer,
    options: CollectOptions,
    state,
    methodology,
    frequency,
    fingerprint_key,
    new_items,
    backfill_done,
    journal,
    policy,
    ownership,
    scoper,
    ledger,
    frequencies,
    migration_changed,
    source_changes,
    expired,
    horizon,
    deadline,
    today,
    now: datetime,
    phase: Callable[..., None],
    step: Callable[[str], None],
    bytes_read: int,
) -> ReductionStage:
    """Prepare collection's affected closures before counters or retention run."""
    from ssgrep.accounting import collector_expiry as expiry, collector_steps as steps
    from ssgrep.accounting.collector_incremental import partition_coverage, pending_document
    from ssgrep.accounting.collector_inputs import journal_signature
    from ssgrep.accounting.collector_state import ReductionDoc
    from ssgrep.accounting.fingerprint import key_id
    from ssgrep.accounting.retention_hooks import expire_journal_events

    journal_sig = journal_signature(journal)
    digest = steps.input_digest(policy, frequency, key_id(fingerprint_key))
    partitions = load_partitions(profile, checkpoint)
    changed = bool(
        new_items
        or backfill_done
        or journal_sig != checkpoint.journal_sig
        or digest != checkpoint.input_digest
        or methodology.hash != checkpoint.methodology
        or state.reduction is None
        or options.recompute
        or options.repin_frequency
        or migration_changed
        or frequencies.completed_pending
        or source_changes
        or partitions.pending
        or partitions.pending_projections
        or not partitions.methodology
    )
    phase("reducing", bytes_read=bytes_read)
    new_refs = store.write_segments(
        profile, new_items, origin="backfill" if options.backfill else "live"
    )
    segments = [*checkpoint.segments, *new_refs]
    step("evidence_written")
    reduction = None
    affected_roots: set[str] = set()
    if changed:
        events, _ = expire_journal_events(journal.events, horizon)
        from ssgrep.accounting.reduce_io import native_source_adapter

        force = (
            "frequency_replacement"
            if options.recompute or options.repin_frequency
            else "frequency_build_complete"
            if frequencies.completed_pending
            else "scope_replay"
            if checkpoint.input_digest and digest != checkpoint.input_digest
            else "identity_replay"
            if migration_changed
            else None
        )
        partitioned = reducer.reduce_affected(
            profile,
            checkpoint,
            new_items=new_items,
            events=events,
            policy=policy,
            frequency=frequency,
            fingerprint_key=fingerprint_key,
            ownership=ownership,
            ledger=ledger,
            frequency_assignments=frequencies,
            force=force,
            changed_sources=source_changes,
            expired_sessions=expired,
            env=ctx.env,
            clock=ctx.clock,
            deadline=deadline,
            should_stop=ctx.should_stop,
            previous_doc=state.reduction,
            native_sources=native_source_adapter(
                policy,
                ctx.env,
                now_ms=int(now.timestamp() * 1000),
                scoper=scoper,
            ),
        )
        partitions, reduction = partitioned.state, partitioned.reduction
        affected_roots = set(partitioned.desired)
        ledger.reconcile(partitioned.desired, today=today, exhaustive=False)
    seen = steps.session_seen(new_items)
    last = expiry.merge_session_last(state.reduction.session_last if state.reduction else (), seen)
    doc = steps.reduction_doc(reduction, last) if reduction is not None else state.reduction
    doc = pending_document(state.reduction, doc or ReductionDoc(), partitions)
    doc = partition_coverage(doc, partitions)
    from ssgrep.accounting.join_scope import effective_scoped_journal

    scoped_journal = replace(
        journal,
        events=effective_scoped_journal(journal.events, doc.executions, scoper),
    )
    if reduction is None:
        doc = replace(doc, session_last=steps.last_tuple(last))
    return ReductionStage(
        changed=changed,
        affected_roots=affected_roots,
        doc=doc,
        last=last,
        segments=segments,
        scoped_journal=scoped_journal,
        partitions=partitions,
        journal_sig=journal_sig,
        digest=digest,
    )
