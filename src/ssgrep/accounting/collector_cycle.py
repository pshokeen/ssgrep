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
)
from ssgrep.accounting.collector_cycle_publication import publish_cycle
from ssgrep.accounting.collector_cycle_reduction import prepare_reduction
from ssgrep.accounting.collector_frequency import (
    load_pinned,
    load_view,
    resolve as resolve_frequency,
)
from ssgrep.accounting.collector_incremental import changed_sources, source_proof
from ssgrep.accounting.collector_inputs import (
    LazyLedger,
    load_fingerprint_key,
)
from ssgrep.accounting.collector_journal import read_cached_journal
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_live import read_stage
from ssgrep.accounting.collector_maintenance import ensure_indexes, maintain_evidence
from ssgrep.accounting.collector_progress import Progress
from ssgrep.accounting.collector_read import ReadOutcome
from ssgrep.accounting.collector_reduce import Reducer, default_reducer
from ssgrep.accounting.collector_root_cache import restore_cache
from ssgrep.accounting.collector_scope import prepare_scope
from ssgrep.accounting.collector_setup import prepare_evidence, prepare_identity
from ssgrep.accounting.collector_state import load_state
from ssgrep.accounting.collector_types import CollectOptions, CollectResult, CycleReport
from ssgrep.accounting.fingerprint import key_id
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.frequency_assignment import FrequencyAssignments
from ssgrep.accounting.journal_retention import expand_journal_retention, finish_journal_retention
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.policy import collection_scope, read_policy_file
from ssgrep.accounting.retention_hooks import evidence_horizon_day
from ssgrep.accounting.revisions import PROVISIONAL_DAYS
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.windows import seal_days

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

    def _phase(self, phase: str, **fields: int | bool | None) -> None:
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
        from ssgrep.accounting.purge_derived import recover_pending_purge

        recover_pending_purge(profile)
        stored = store.load_checkpoint(profile)
        validated_state = steps.recover(profile, stored)
        validated_blobs = dict(stored.blobs) if stored else {}
        finish_journal_retention(
            profile,
            generation=stored.generation if stored else 0,
            legacy_writers_stopped=options.journal_legacy_writers_stopped,
        )
        if stored is not None:
            stored, blocked = prepare_evidence(ctx, stored)
            if blocked is not None:
                return blocked
        ownership = read_policy_file()
        policy = ownership.profiles.get(profile, ProfilePolicy(profile_id=profile))
        scope = collection_scope(policy, ownership)
        journal = read_cached_journal(profile, stored)
        checkpoint = stored or steps.new_checkpoint(profile, journal, self.now)
        migration, blocked = prepare_identity(ctx, checkpoint, policy, journal)
        if blocked is not None:
            return blocked
        previous_journal_retention = checkpoint.retention.get("journal")
        if previous_journal_retention:
            previous_journal_retention = expand_journal_retention(
                profile, previous_journal_retention
            )
        prepare_scope(profile, checkpoint, scope, journal, self.now)
        scoper = Scoper(policy, ownership)
        reducer = ctx.reducer or default_reducer()
        pinned = load_pinned(profile)
        methodology = reducer.methodology_for(pinned or FrequencyView.unknown())
        state = (
            validated_state
            if validated_state is not None
            and checkpoint.methodology == methodology.hash
            and validated_blobs == checkpoint.blobs
            else load_state(profile, checkpoint, methodology.hash)
        )
        prior_state = state
        deadline = self.started + options.budget.seconds
        _, indexing_pending = ensure_indexes(
            profile,
            checkpoint,
            clock=ctx.clock,
            deadline=deadline,
            byte_budget=options.budget.bytes,
            should_stop=ctx.should_stop,
        )
        previous_sources = {key: source_proof(record) for key, record in checkpoint.sources.items()}
        expired = expiry.decode_sessions(checkpoint.retention.get("expired_sessions", []))
        lazy = LazyLedger(profile, checkpoint)
        horizon = evidence_horizon_day(self.now, policy)
        self._phase("reading")
        if indexing_pending:
            outcome, new_items, backfill_done = (
                ReadOutcome(stopped_by="evidence_index_budget"),
                [],
                False,
            )
        else:
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
                remaining_seconds=max(0.0, deadline - ctx.clock()),
            )
        source_changes = changed_sources(previous_sources, checkpoint.sources)
        # ---- current compatible stats; each lookup keeps its own frozen input ----
        freq = resolve_frequency(
            profile,
            pinned=pinned,
            last_documents=checkpoint.frequency.get("documents"),
            evidence=lambda: new_items,
            scan=bool(new_items),
            repin=options.repin_frequency or options.recompute,
            on_step=ctx.on_step,
            cached=state.frequency_probe or pinned,
            reset_authority=options.reset_frequency_authority,
            scope_epoch=checkpoint.scope_epoch,
            authority=scoper.in_scope,
            authority_id=steps.digest_of([policy, ownership]),
        )
        frequency = freq.view
        checkpoint.frequency = {"documents": freq.documents}
        current_method = reducer.methodology_for(frequency)
        if current_method.hash != methodology.hash:
            methodology = current_method
            state = load_state(profile, checkpoint, methodology.hash)
            prior_state = state
        ledger = (
            VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives)
            if state.ledger is not None
            else VersionedLedger(methodology)
        )
        if ledger.methodology.hash != methodology.hash:
            raise steps.CollectError("state_corrupt", "Ledger methodology does not match state.")
        restore_cache(profile, ledger, checkpoint.blobs)
        fingerprint_key = load_fingerprint_key(profile)
        frequencies = FrequencyAssignments(
            frequency,
            assignments=state.frequency_assignments,
            loader=lambda view_id: load_view(profile, view_id),
            fingerprint_key_id=key_id(fingerprint_key),
        )
        if options.recompute or options.repin_frequency:
            revision_name = f"frequency-recompute:{frequency.version_id}"
            freq = replace(freq, transition="replacement", revision_name=revision_name)
            ledger.archive_current(f"{revision_name}:generation-{checkpoint.generation + 1}")
            frequencies.recompute(revision_name)
        # ---- reduce affected dependency closures; retain untouched canonical rows ----
        prepared = prepare_reduction(
            profile,
            checkpoint,
            ctx=ctx,
            reducer=reducer,
            options=options,
            state=state,
            methodology=methodology,
            frequency=frequency,
            fingerprint_key=fingerprint_key,
            new_items=new_items,
            backfill_done=backfill_done,
            journal=journal,
            policy=policy,
            ownership=ownership,
            scoper=scoper,
            ledger=ledger,
            frequencies=frequencies,
            migration_changed=migration.changed,
            source_changes=source_changes,
            expired=expired,
            horizon=horizon,
            deadline=deadline,
            today=self.today,
            now=self.now,
            phase=self._phase,
            step=self._step,
            bytes_read=outcome.bytes_read,
        )
        doc = prepared.doc
        executions = doc.executions
        # ---- counters, sealing, retention ----
        counters = steps.counters_step(
            state.counters,
            prepared.scoped_journal,
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
        sealing = expiry.sealing_executions(
            state.reduction.executions if state.reduction else (),
            executions,
            evidence_before=horizon,
            before_day=seal_before,
            aggregate_before=committed_floor,
            already=state.sealed,
        )
        sealed, late = seal_days(sealing, before_day=seal_before, already=state.sealed)
        expiry_out, kept_days = expiry.apply_expiry(
            profile,
            now=self.now,
            policy=policy,
            ledger=ledger,
            doc=doc,
            session_last=prepared.last,
            already=expired,
            previous=expiry.previous_coverage(checkpoint.retention),
            segments=prepared.segments,
            sealed_days=sealed,
            compact=False,
        )
        sealed = {day: sealed[day] for day in kept_days}
        expired_before = expiry_out.coverage.aggregates_expired_before
        if expired_before is not None:
            executions = tuple(e for e in executions if e.day >= expired_before)
            doc = replace(doc, executions=executions)
        segments, evidence_job, _, _ = maintain_evidence(
            profile,
            prepared.segments,
            expired_sessions=expiry_out.sessions,
            previous=checkpoint.retention.get("evidence_maintenance"),
            clock=ctx.clock,
            deadline=deadline,
            byte_budget=options.budget.bytes,
            should_stop=ctx.should_stop,
        )
        return publish_cycle(
            profile,
            checkpoint,
            ctx=ctx,
            options=options,
            now=self.now,
            today=self.today,
            started=self.started,
            phase=self._phase,
            step=self._step,
            backfill_done=backfill_done,
            prepared=prepared,
            doc=doc,
            executions=executions,
            ledger=ledger,
            state=state,
            prior_state=prior_state,
            counters=counters,
            sealed=sealed,
            scope=scope,
            frequencies=frequencies,
            frequency=frequency,
            methodology=methodology,
            policy=policy,
            expiry_out=expiry_out,
            journal=journal,
            previous_journal_retention=previous_journal_retention,
            horizon=horizon,
            evidence_job=evidence_job,
            outcome=outcome,
            new_items=new_items,
            migration=migration,
            freq=freq,
            segments=segments,
            deadline=deadline,
            late=late,
        )


def run_cycle(ctx: CycleContext) -> CycleOutcome:
    return Cycle(ctx).run()
