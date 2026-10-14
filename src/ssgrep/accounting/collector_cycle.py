"""One bounded collection cycle (design D8): read, reduce, value, expire, publish, commit.

Order is the crash-safety argument. New evidence segments and state blobs are written first
(nothing references them yet); receipts/history are refreshed (derived, idempotent); the
snapshot generation is staged; the single checkpoint is committed (the exactly-once point);
only then ``current.json`` is swapped. A crash before the commit leaves the previous
checkpoint authoritative, so evidence is read again and reduced once and orphans are swept;
a crash after it is repaired by the next lease holder (``snapshots.repair_current``).
The caller holds the profile's worker lease for the whole cycle.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from ssgrep.accounting import (
    collector_expiry as expiry,
    collector_steps as steps,
    collector_store as store,
    snapshots,
)
from ssgrep.accounting.collector_assemble import SnapshotInputs, build_snapshot, utc_stamp
from ssgrep.accounting.collector_frequency import load_pinned, resolve as resolve_frequency
from ssgrep.accounting.collector_inputs import (
    LazyLedger,
    Retained,
    journal_last_ts,
    journal_signature,
    journal_stats,
    load_fingerprint_key,
)
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_live import read_stage
from ssgrep.accounting.collector_modes import unavailable_periods
from ssgrep.accounting.collector_progress import Progress
from ssgrep.accounting.collector_publish import (
    history_document,
    reuse_blocks,
    write_history,
    write_receipts,
)
from ssgrep.accounting.collector_reduce import Reducer, default_reducer
from ssgrep.accounting.collector_scope import prepare_scope
from ssgrep.accounting.collector_state import load_state, save_state
from ssgrep.accounting.collector_types import CollectOptions, CollectResult, CycleReport
from ssgrep.accounting.counters import backfill_series
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.journal_reader import read_journal
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.policy import collection_scope, read_policy_file
from ssgrep.accounting.retention_hooks import evidence_horizon_day, expire_journal_events
from ssgrep.accounting.revisions import PROVISIONAL_DAYS
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.windows import seal_days

MAX_SEGMENTS = store.MAX_SEGMENTS
PROGRESS_INTERVAL = 0.5


@dataclass
class CycleContext:
    profile: str
    options: CollectOptions
    now: datetime
    progress: Progress
    clock: Callable[[], float] = time.monotonic
    should_stop: Callable[[], bool] = lambda: False
    reducer: Reducer | None = None
    env: Mapping[str, str] | None = None
    on_step: Callable[[str], None] | None = None  # test fault hook around the commit point


@dataclass
class CycleOutcome:
    report: CycleReport
    result: CollectResult


class Cycle:
    """State and steps of one cycle; ``run`` is the whole protocol."""

    def __init__(self, ctx: CycleContext) -> None:
        self.ctx = ctx
        self.profile = ctx.profile
        self.now = ctx.now
        self.today = ctx.now.astimezone(UTC).date().isoformat()
        self.started = ctx.clock()
        self._last_progress = float("-inf")

    # ---- plumbing --------------------------------------------------------------------

    def _step(self, name: str) -> None:
        if self.ctx.on_step is not None:
            self.ctx.on_step(name)

    def _phase(self, phase: str, **fields: int) -> None:
        progress = self.ctx.progress
        progress.phase = phase
        for name, value in fields.items():
            setattr(progress, name, value)
        moment = self.ctx.clock()
        if phase != "reading" or moment - self._last_progress >= PROGRESS_INTERVAL:
            self._last_progress = moment
            progress.write(self.now)

    # ---- the protocol ----------------------------------------------------------------

    def run(self) -> CycleOutcome:
        ctx, profile, options = self.ctx, self.profile, self.ctx.options
        self._step("started")
        steps.guard_schema(profile)
        stored = store.load_checkpoint(profile)
        steps.recover(profile, stored)
        ownership = read_policy_file()
        policy = ownership.profiles.get(profile, ProfilePolicy(profile_id=profile))
        scope = collection_scope(policy, ownership)
        journal = read_journal(profile)
        checkpoint = stored or steps.new_checkpoint(profile, journal, self.now)
        prepare_scope(profile, checkpoint, scope, journal, self.now)
        scoper = Scoper(policy, ownership)
        scoped_journal = replace(
            journal, events=[e for e in journal.events if scoper.in_scope(e.cwd)]
        )
        reducer = ctx.reducer or default_reducer()
        pinned = load_pinned(profile)
        methodology = reducer.methodology_for(pinned or FrequencyView.unknown())
        state = load_state(profile, checkpoint, methodology.hash)
        expired = expiry.decode_sessions(checkpoint.retention.get("expired_sessions", []))
        lazy = LazyLedger(profile, checkpoint)
        horizon = evidence_horizon_day(self.now, policy)
        self._phase("reading")
        outcome, new_items, backfill_done = read_stage(
            profile=profile,
            options=options,
            now=self.now,
            clock=ctx.clock,
            should_stop=ctx.should_stop,
            env=ctx.env,
            checkpoint=checkpoint,
            ledger=lazy,
            expired=expired,
            policy=policy,
            horizon=horizon,
            phase=self._phase,
        )
        # ---- the frequency view: pinned once, immutable, repinned only on request ----
        retained = Retained(profile, checkpoint.segments)
        freq = resolve_frequency(
            profile,
            pinned=pinned,
            last_documents=checkpoint.frequency.get("documents"),
            evidence=lambda: [*retained.items(), *new_items],
            scan=bool(new_items),
            repin=options.repin_frequency,
            on_step=ctx.on_step,
        )
        frequency = freq.view
        checkpoint.frequency = {"documents": freq.documents} if freq.state != "pinned" else {}
        if frequency.version_id != (pinned.version_id if pinned else "frequency-unknown"):
            methodology = reducer.methodology_for(frequency)
            state = load_state(profile, checkpoint, methodology.hash)  # a new namespace
        ledger = (
            VersionedLedger.restore_versioned(state.ledger, state.superseded)
            if state.ledger is not None
            else VersionedLedger(methodology)
        )
        if ledger.methodology.hash != methodology.hash:
            raise steps.CollectError("state_corrupt", "Ledger methodology does not match state.")
        # ---- reduce (everything retained + what arrived) only when something changed ----
        journal_sig = journal_signature(journal)
        key_present = load_fingerprint_key(profile) is not None
        digest = steps.input_digest(policy, frequency, key_present)
        changed = bool(
            new_items
            or backfill_done
            or journal_sig != checkpoint.journal_sig
            or digest != checkpoint.input_digest
            or methodology.hash != checkpoint.methodology
            or state.reduction is None
            or options.recompute
        )
        self._phase("reducing", bytes_read=outcome.bytes_read)
        new_refs = store.write_segments(
            profile, new_items, origin="backfill" if options.backfill else "live"
        )
        segments = [*checkpoint.segments, *new_refs]
        self._step("evidence_written")
        reduction = None
        if changed:
            evidence = [*retained.items(), *new_items]
            events, _ = expire_journal_events(journal.events, horizon)
            reduction = reducer.reduce_profile(
                profile,
                evidence,
                events,
                policy=policy,
                frequency=frequency,
                fingerprint_key=load_fingerprint_key(profile),
                ownership=ownership,
            )
            ledger.reconcile(  # the reduction covers ALL retained evidence: it is exhaustive
                dict(zip(reduction.bundle_roots, reduction.bundles, strict=True)),
                today=self.today,
                exhaustive=True,
            )
        seen = steps.session_seen(new_items)
        last = expiry.merge_session_last(
            state.reduction.session_last if state.reduction else (), seen
        )
        doc = steps.reduction_doc(reduction, last) if reduction is not None else state.reduction
        assert doc is not None  # a missing cached reduction always triggers a fresh reduction
        if reduction is None:
            doc = replace(doc, session_last=steps.last_tuple(last))
        executions = doc.executions
        # ---- counters, sealing, retention ----
        counters = steps.counters_step(
            state.counters,
            scoped_journal,
            now=self.now,
            horizon_day=horizon,
            new_epoch=options.new_epoch,
            observing=True,
        )
        committed_floor = steps.aggregates_expired_before(checkpoint)
        if committed_floor is not None:
            executions = tuple(e for e in executions if e.day >= committed_floor)
        seal_before = (
            self.now.astimezone(UTC).date() - timedelta(days=PROVISIONAL_DAYS)
        ).isoformat()
        sealed, late = seal_days(executions, before_day=seal_before, already=state.sealed)
        expiry_out, kept_days = expiry.apply_expiry(
            profile,
            now=self.now,
            policy=policy,
            ledger=ledger,
            doc=doc,
            session_last=last,
            already=expired,
            previous=expiry.previous_coverage(checkpoint.retention),
            segments=segments,
            sealed_days=sealed,
        )
        sealed = {day: sealed[day] for day in kept_days}
        expired_before = expiry_out.coverage.aggregates_expired_before
        if expired_before is not None:
            executions = tuple(e for e in executions if e.day >= expired_before)
            doc = replace(doc, executions=executions)
        if expiry_out.segments is not None:
            segments = expiry_out.segments
        if len(segments) > MAX_SEGMENTS:
            segments = store.write_segments(profile, store.iter_evidence(profile, segments))
        # ---- backfill series, receipts, history ----
        backfills = list(checkpoint.backfills)
        series = None
        if backfill_done and reduction is not None:
            since, until = checkpoint.backfill_job["since"], checkpoint.backfill_job["until"]
            trimmed = [
                replace(r, ts_utc=(r.ts_utc or "")[:19] + "Z")
                for r in reduction.invocations
                if r.ts_utc
            ]
            series = backfill_series(trimmed, since, until, utc_stamp(self.now)).to_dict()
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
        )
        generation = checkpoint.generation + 1
        blobs = dict(checkpoint.blobs)
        save_state(profile, state, methodology.hash, blobs)
        self._step("state_written")
        stopped = outcome.stopped_by
        envelope = build_snapshot(
            SnapshotInputs(
                profile=profile,
                now=self.now,
                generation=generation,
                policy=policy,
                methodology=methodology,
                executions=executions,
                sealed=sealed,
                rows=rows,
                counters=counters,
                backfills=backfills,
                reduction_coverage=doc.coverage,
                warnings=(*doc.warnings, *freq.standing_warnings()),
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
        self._step("before_stage")
        snapshots.stage_generation(profile, envelope, self.ctx.on_step)
        checkpoint.generation = generation
        checkpoint.cycles += 1
        checkpoint.methodology = methodology.hash
        checkpoint.input_digest = digest
        checkpoint.journal_sig = journal_sig
        checkpoint.segments = segments
        checkpoint.blobs = blobs
        checkpoint.journal = journal_stats(journal)
        checkpoint.retention = expiry.retention_record(expiry_out.coverage, expiry_out.sessions)
        checkpoint.backfills = backfills
        self._step("before_commit")
        store.commit_checkpoint(profile, checkpoint)
        self._step("checkpoint_committed")
        self._phase("publishing")
        reuse = reuse_blocks(ledger.state().candidates, methodology, ledger.superseded_state())
        write_receipts(profile, rows, reuse)
        write_history(profile, history_document(rows, methodology.units[0], self.today))
        expired_before = steps.aggregates_expired_before(checkpoint)
        if expired_before is not None:
            steps.prune_receipt_files(profile, expired_before)
        self._step("receipts_written")
        snapshots.make_current(profile, generation, self.ctx.on_step)
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
            elapsed_ms=int((ctx.clock() - self.started) * 1000),
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
            state="published"
            if stopped == "complete" and not checkpoint.backfill_job
            else "partial",
            mode=options.mode,
            report=report,
            epoch=counters.epoch,
            unavailable_periods=periods,
            backfill=series,
            warnings=(*(doc.warnings if doc else ()), *freq.standing_warnings(), *freq.events),
        )
        return CycleOutcome(report, result)


def run_cycle(ctx: CycleContext) -> CycleOutcome:
    return Cycle(ctx).run()
