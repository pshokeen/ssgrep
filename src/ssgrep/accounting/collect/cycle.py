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

import hashlib
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from ssgrep.accounting.capture.fingerprint import key_id
from ssgrep.accounting.capture.journal_reader import JournalRead
from ssgrep.accounting.capture.journal_retention import (
    expand_journal_retention,
    finish_journal_retention,
    persist_journal_retention,
    prepare_journal_retention,
)
from ssgrep.accounting.collect import cycle_publish, steps, store
from ssgrep.accounting.collect.collector import (
    AggregateExpiry,
    CollectOptions,
    CollectResult,
    CycleReport,
    Progress,
    Reducer,
    RetentionCoverage,
    aggregate_horizon_day,
    coverage_after,
    default_reducer,
    evidence_horizon_day,
    expire_aggregates,
    plan_expiry,
    prepare as prepare_published_retention,
    split_evidence,
    utc_stamp,
)
from ssgrep.accounting.collect.frequency import (
    FrequencyAssignments,
    FrequencyView,
    load_pinned,
    load_view,
    resolve as resolve_frequency,
)
from ssgrep.accounting.collect.frequency_index import (
    stage as stage_frequency_gc,
)
from ssgrep.accounting.collect.read import (
    LazyLedger,
    ReadOutcome,
    load_fingerprint_key,
    read_cached_journal,
    read_stage,
    save_journal_cache,
)
from ssgrep.accounting.collect.state import (
    COUNTERS,
    ReductionDoc,
    VersionedLedger,
    decode_ledger,
    ledger_blob_name,
    load_state,
    save_state,
)
from ssgrep.accounting.collect.store import (
    CollectError,
    SegmentRef,
    restore_cache,
    save_cache,
)
from ssgrep.accounting.ledger.contracts import RawEvidence, digest_of, target_key
from ssgrep.accounting.ledger.ledger_contracts import LiveCounters, ProfilePolicy, start_epoch
from ssgrep.accounting.ledger.revisions import PROVISIONAL_DAYS
from ssgrep.accounting.privacy.policy import (
    collection_scope,
    read_policy_file,
    valid_collection_scope,
)
from ssgrep.accounting.readers.identity_migration import MigrationResult
from ssgrep.accounting.views.filter import Scoper
from ssgrep.accounting.views.history import seal_days

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
        from ssgrep.accounting.privacy.purge_recovery import recover_pending_purge

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
        reducer = ctx.reducer or default_reducer()
        pinned = load_pinned(profile)
        methodology = reducer.methodology_for(pinned or FrequencyView.unknown())
        steps.retire_methodology(profile, checkpoint, methodology.hash)
        prepare_scope(profile, checkpoint, scope, journal, self.now)
        scoper = Scoper(policy, ownership)
        state = (
            validated_state
            if validated_state is not None
            and checkpoint.methodology == methodology.hash
            and validated_blobs == checkpoint.blobs
            else load_state(profile, checkpoint, methodology.hash)
        )
        prior_state = state
        deadline = self.started + options.budget.seconds
        _, indexing_pending = steps.ensure_indexes(
            profile,
            checkpoint,
            clock=ctx.clock,
            deadline=deadline,
            byte_budget=options.budget.bytes,
            should_stop=ctx.should_stop,
        )
        previous_sources = {
            key: steps.source_proof(record) for key, record in checkpoint.sources.items()
        }
        expired = steps.decode_sessions(checkpoint.retention.get("expired_sessions", []))
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
        source_changes = steps.changed_sources(previous_sources, checkpoint.sources)
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
        request = steps.replacement_request(options, frequency, checkpoint)
        if request is not None and request != store.load_partitions(profile, checkpoint).operation:
            revision_name = f"frequency-recompute:{frequency.version_id}"
            freq = replace(freq, transition="replacement", revision_name=revision_name)
            ledger.archive_current(f"{revision_name}:generation-{checkpoint.generation + 1}")
            frequencies.recompute(revision_name)
        # ---- reduce affected dependency closures; retain untouched canonical rows ----
        prepared = steps.prepare_reduction(
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
        sealing = steps.sealing_executions(
            state.reduction.executions if state.reduction else (),
            executions,
            evidence_before=horizon,
            before_day=seal_before,
            aggregate_before=committed_floor,
            already=state.sealed,
        )
        sealed, late = seal_days(sealing, before_day=seal_before, already=state.sealed)
        expiry_out, kept_days = apply_expiry(
            profile,
            now=self.now,
            policy=policy,
            ledger=ledger,
            doc=doc,
            session_last=prepared.last,
            already=expired,
            previous=steps.previous_coverage(checkpoint.retention),
            segments=prepared.segments,
            sealed_days=sealed,
            compact=False,
        )
        sealed = {day: sealed[day] for day in kept_days}
        expired_before = expiry_out.coverage.aggregates_expired_before
        if expired_before is not None:
            executions = tuple(e for e in executions if e.day >= expired_before)
            doc = replace(doc, executions=executions)
        segments, evidence_job, _, _ = steps.maintain_evidence(
            profile,
            prepared.segments,
            expired_sessions=expiry_out.sessions,
            previous=checkpoint.retention.get("evidence_maintenance"),
            clock=ctx.clock,
            deadline=deadline,
            byte_budget=options.budget.bytes,
            should_stop=ctx.should_stop,
        )
        return cycle_publish.publish_cycle(
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


def prepare_scope(
    profile: str,
    checkpoint: store.Checkpoint,
    scope: dict[str, Any],
    journal: JournalRead,
    now: datetime,
) -> bool:
    """Initialize scope or rotate existing state; callers hold the worker lease."""
    old = checkpoint.collection_scope
    changed = checkpoint.generation > 0 and (not valid_collection_scope(old) or old != scope)
    if checkpoint.generation > 0 and checkpoint.methodology:
        raw = store.read_blob(
            profile, checkpoint.blobs.get(ledger_blob_name(checkpoint.methodology))
        )
        if raw is not None:
            provenance = raw.get("collection_scope") if isinstance(raw, dict) else None
            changed = changed or provenance != scope
            if changed:
                decode_ledger(raw, profile=profile)
    if changed:
        previous = store.read_blob(profile, checkpoint.blobs.get(COUNTERS))
        counters = LiveCounters.from_dict(previous) if previous is not None else None
        epoch = checkpoint.scope_epoch
        for name, file_name in list(checkpoint.blobs.items()):
            if "-scope-" not in name:
                checkpoint.blobs[f"{name}-scope-{epoch}"] = file_name
                del checkpoint.blobs[name]
        reset = start_epoch(counters, journal.events, now)
        checkpoint.blobs[COUNTERS] = store.write_blob(profile, COUNTERS, reset.to_dict())
        checkpoint.scope_epoch += 1
        checkpoint.backfills = []
        checkpoint.input_digest = ""
    checkpoint.collection_scope = scope
    return changed


def prepare_identity(
    ctx: CycleContext,
    checkpoint: store.Checkpoint,
    policy: ProfilePolicy,
    journal: JournalRead,
) -> tuple[MigrationResult, CycleOutcome | None]:
    from ssgrep.accounting.readers.identity_migration import prepare

    migration = prepare(
        ctx.profile,
        checkpoint,
        ctx.now,
        env=ctx.env,
        policy=policy,
        journal=journal,
        byte_budget=ctx.options.budget.bytes,
    )
    if migration.complete:
        return migration, None
    # Persist bounded migration progress, but do not import/read/reduce new inputs.
    store.commit_checkpoint(ctx.profile, checkpoint)
    reason = (
        "identity_migration_unavailable"
        if migration.unavailable_sessions
        else "identity_migration_pending"
    )
    report = CycleReport(
        status="failed",
        generation=checkpoint.generation,
        snapshot_id=None,
        complete=False,
        stopped_by="identity_migration",
        bytes_read=migration.bytes_read,
        error=reason,
    )

    return migration, CycleOutcome(
        report,
        CollectResult(
            state="partial",
            mode=ctx.options.mode,
            report=report,
            warnings=migration.warnings,
        ),
    )


def prepare_evidence(
    ctx: CycleContext, checkpoint: store.Checkpoint
) -> tuple[store.Checkpoint, CycleOutcome | None]:
    """One privacy-compaction quantum; never continue into new-native work when pending."""
    from ssgrep.accounting.collect.store import migrate_evidence

    checkpoint = migrate_evidence(ctx.profile, checkpoint, byte_budget=ctx.options.budget.bytes)
    if not checkpoint.retention.get("evidence_migration_pending"):
        return checkpoint, None
    report = CycleReport(
        status="failed",
        generation=checkpoint.generation,
        snapshot_id=None,
        complete=False,
        stopped_by="evidence_migration",
        error="evidence_migration_pending",
    )

    return checkpoint, CycleOutcome(
        report,
        CollectResult(
            state="partial",
            mode=ctx.options.mode,
            report=report,
            warnings=("evidence_migration_pending",),
        ),
    )


def stage(
    profile,
    checkpoint,
    *,
    state,
    prior_state,
    methodology,
    clock,
    deadline,
    affected_roots,
    changed,
    partitions,
    ledger,
    expiry_out,
    journal,
    previous_journal_retention,
    horizon,
    now,
    policy,
    evidence_job,
    today,
    rows,
    maintenance_budget,
):
    publication_work = prepare_published_retention(
        profile,
        checkpoint,
        rows=rows,
        before_day=max(
            aggregate_horizon_day(now, policy), expiry_out.coverage.aggregates_expired_before or ""
        ),
        evidence_before_day=horizon,
        clock=clock,
        deadline=deadline,
        byte_budget=maintenance_budget,
    )
    blobs = dict(checkpoint.blobs)
    save_state(
        profile,
        state,
        methodology.hash,
        blobs,
        previous=prior_state,
        clock=clock,
        deadline=deadline,
        dirty_roots=affected_roots,
    )
    if changed or "partitions" not in blobs:
        store.save_partitions(profile, partitions, blobs)
    if changed or expiry_out.changed or "root-evaluations" not in blobs:
        save_cache(profile, ledger, blobs)
    retention = {
        **checkpoint.retention,
        **steps.retention_record(
            expiry_out.coverage,
            expiry_out.sessions,
            journal=prepare_journal_retention(
                journal,
                previous=previous_journal_retention,
                horizon_day=horizon,
                aggregate_horizon_day=aggregate_horizon_day(now, policy),
            ),
        ),
        "evidence_maintenance": evidence_job,
    }
    persisted = persist_journal_retention(profile, retention["journal"], blobs=blobs)
    save_journal_cache(
        profile, journal, blobs, retention["journal"], retention_tag=digest_of(persisted)
    )
    retention["journal"] = persisted
    retention["root_state_migration"] = {"pending_roots": state.root_migration_pending}
    frequency_pending = stage_frequency_gc(
        profile,
        checkpoint,
        blobs,
        current_refs={row.view_id for row in state.frequency_assignments},
        before_day=aggregate_horizon_day(now, policy),
        today=today,
    )
    return blobs, retention, frequency_pending, publication_work


"""Explicit resumable compact-evidence indexing, retention and segment compaction.

The quantum is one existing bounded segment, never an unbounded corpus rewrite.
Logical retention floors commit independently of physical cleanup. Pending segment
names are durable backlog; a cycle with backlog cannot claim complete coverage.
"""


"""Retention as one cycle step: freeze expired conversations, drop their evidence, age aggregates.

Wraps ``retention_hooks`` for the worker. Only owned accounting data is ever touched: retained
evidence segments, ledger content, and derived aggregates. Nothing here accepts a path from
evidence or touches a source transcript or an index.
"""


@dataclass
class ExpiryOutcome:
    frozen: list[str] = field(default_factory=list)
    sessions: frozenset[tuple[str, str]] = frozenset()  # every expired session (cumulative)
    dropped_records: int = 0
    segments: list[SegmentRef] | None = None  # set when retained segments were rewritten
    coverage: RetentionCoverage = RetentionCoverage()
    changed: bool = False


def _frozen_days(ledger: VersionedLedger) -> list[str]:
    return sorted({leaf.day for frozen in ledger.state().frozen for leaf in frozen.leaves})


def apply_expiry(
    profile: str,
    *,
    now: datetime,
    policy: ProfilePolicy,
    ledger: VersionedLedger,
    doc: ReductionDoc,
    session_last: Mapping[tuple[str, str], str],
    already: frozenset[tuple[str, str]],
    previous: RetentionCoverage,
    segments: list[SegmentRef],
    sealed_days: Mapping[str, object],
    compact: bool = True,
) -> tuple[ExpiryOutcome, dict[str, object]]:
    """Freeze conversations whose evidence all aged out; compact retained evidence; age totals.

    Returns the outcome and the sealed days that survive aggregate retention.
    """
    plan = plan_expiry(
        now=now, policy=policy, root_last_evidence=steps.root_last_evidence(doc, session_last)
    )
    aggregate_floor = max(plan.aggregate_horizon_day, previous.aggregates_expired_before or "")
    if previous.aggregates_expired_before is not None and ledger.has_before_day(
        previous.aggregates_expired_before
    ):
        # Recomputed evidence cannot restore aggregates already removed by retention.
        ledger.purge(before_day=previous.aggregates_expired_before)
    out = ExpiryOutcome(sessions=already, coverage=previous)
    if plan.expire_roots:
        out.frozen = ledger.expire_roots(plan.expire_roots)
        wanted = {
            (runtime, session)
            for runtime, session, root in doc.session_roots
            if root in set(plan.expire_roots)
        }
        fresh = frozenset(wanted - already)
        if fresh:
            out.sessions = already | fresh
            if compact:
                kept, dropped = split_evidence(store.iter_evidence(profile, segments), out.sessions)
                out.segments = store.write_segments(profile, kept)
                out.dropped_records = dropped
            out.changed = True
    history = [{"period": day} for day in _frozen_days(ledger) if day < aggregate_floor]
    kept_rows, kept_days, report = expire_aggregates(history, sealed_days, aggregate_floor)
    del kept_rows
    frozen_before_aging = len(ledger.state().frozen)
    aged = _age_aggregates(ledger, report)
    out.coverage = coverage_after(
        previous,
        plan,
        out.frozen,
        report.aggregates_expired_before if aged else None,
        frozen_before_aging,
    )
    if aged:
        # Freezing is an event even when the same cycle expires its aggregate.
        # The published count describes only the frozen roots still retained.
        out.coverage = replace(out.coverage, frozen_roots=len(ledger.state().frozen))
    out.changed = out.changed or aged or bool(out.frozen)
    return out, dict(kept_days)


def _age_aggregates(ledger: VersionedLedger, report: AggregateExpiry) -> bool:
    """Remove frozen totals older than the aggregate horizon; True when anything went."""
    if report.aggregates_expired_before is None:
        return False
    ledger.purge(before_day=report.before_day)
    return True


Session = tuple[str, str]
NativeKey = tuple[str, str, str]


def session_name(key: Session) -> str:
    return store.dumps(list(key)).decode()


def identity_hash(key: NativeKey) -> str:
    return hashlib.sha256(store.dumps(list(key))).hexdigest()


@dataclass
class IndexWriter:
    sessions: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    identities: dict[str, dict[str, list]] = field(default_factory=lambda: defaultdict(dict))

    def add(self, item: RawEvidence, offset: int, line: bytes) -> None:
        location = [offset, len(line), hashlib.sha256(line).hexdigest()]
        self.sessions[session_name((item.runtime, item.session_id))].append(location)
        # Claims are fetched with their target, not as separate usage identities.
        key = target_key(item) or item.native_key
        digest = identity_hash(key)
        self.identities[digest[:2]].setdefault(digest, []).append(location)

    def write(self, profile: str) -> str:
        sessions = {
            key: store.write_blob(profile, "segment-session", rows)
            for key, rows in sorted(self.sessions.items())
        }
        identities = {
            key: store.write_blob(profile, "segment-identities", rows)
            for key, rows in sorted(self.identities.items())
        }
        return store.write_blob(
            profile,
            "segment-index",
            {
                "schema": 1,
                "sessions": sessions,
                "identities": identities,
            },
        )


def cached_blob(profile: str, filename: str | None, cache: dict | None):
    if cache is None or filename is None:
        return store.read_blob(profile, filename)
    if filename not in cache:
        cache[filename] = store.read_blob(profile, filename)
    return cache[filename]


def read_index(profile: str, ref: store.SegmentRef, cache: dict | None = None) -> dict | None:
    if ref.index is None:
        return None
    raw = cached_blob(profile, ref.index, cache)
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise CollectError("state_corrupt", "Retained evidence index is invalid.")
    if not isinstance(raw.get("sessions"), dict) or not isinstance(raw.get("identities"), dict):
        raise CollectError("state_corrupt", "Retained evidence index tables are invalid.")
    return raw


def iter_selected(
    profile: str,
    ref: store.SegmentRef,
    *,
    sessions: set[Session] | None = None,
    native_keys: set[NativeKey] | None = None,
    index_cache: dict | None = None,
) -> Iterator[RawEvidence]:
    from ssgrep.accounting.collect.replay import evidence_from_line

    index = read_index(profile, ref, index_cache)
    if index is None:
        # Legacy conversion is explicitly queued by the partition collector; this
        # compatibility read never hydrates native content and cannot claim freshness.
        for item in store.read_segment(profile, ref):
            if sessions is not None and (item.runtime, item.session_id) not in sessions:
                continue
            if native_keys is not None and (target_key(item) or item.native_key) not in native_keys:
                continue
            yield item
        return
    rows: dict[int, list] = {}
    if sessions is not None:
        for key in sorted(sessions):
            filename = index["sessions"].get(session_name(key))
            for row in cached_blob(profile, filename, index_cache) or ():
                rows[row[0]] = row
    elif native_keys is not None:
        shards: dict[str, set[str]] = defaultdict(set)
        for key in native_keys:
            digest = identity_hash(key)
            shards[digest[:2]].add(digest)
        for shard, digests in sorted(shards.items()):
            table = cached_blob(profile, index["identities"].get(shard), index_cache) or {}
            for digest in sorted(digests):
                for row in table.get(digest, ()):
                    rows[row[0]] = row
    else:
        yield from store.read_segment(profile, ref)
        return
    if not rows:
        return
    from ssgrep.accounting.privacy import storage

    directory = storage.open_profile_dir(profile, "collector", store.EVIDENCE_DIR, create=False)
    try:
        fd = storage.open_regular(directory, ref.name)
    except OSError as error:
        raise CollectError("state_corrupt", "Indexed retained evidence is missing.") from error
    finally:
        os.close(directory)
    try:
        if os.fstat(fd).st_size != ref.size:
            raise CollectError("state_corrupt", "Indexed retained evidence size changed.")
        for offset, length, digest in (rows[key] for key in sorted(rows)):
            if offset < 0 or length <= 0 or offset + length > ref.size:
                raise CollectError("state_corrupt", "Retained evidence index range is invalid.")
            line = os.pread(fd, length, offset)
            if len(line) != length or hashlib.sha256(line).hexdigest() != digest:
                raise CollectError("state_corrupt", "Indexed retained evidence digest changed.")
            try:
                item = evidence_from_line(line.rstrip(b"\n"))
            except (ValueError, KeyError, TypeError) as error:
                raise CollectError(
                    "state_corrupt", "Indexed retained evidence is invalid."
                ) from error
            if sessions is not None and (item.runtime, item.session_id) not in sessions:
                raise CollectError("state_corrupt", "Retained evidence session index disagrees.")
            if native_keys is not None and (target_key(item) or item.native_key) not in native_keys:
                raise CollectError("state_corrupt", "Retained evidence identity index disagrees.")
            yield item
    finally:
        os.close(fd)


def index_references(profile: str, refs: Iterable[store.SegmentRef]) -> set[str]:
    keep: set[str] = set()
    for ref in refs:
        index = read_index(profile, ref)
        if index is not None and ref.index is not None:
            keep.add(ref.index)
            keep.update(index["sessions"].values())
            keep.update(index["identities"].values())
    return keep


def _instant(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).replace(microsecond=0)


def coverage_through(
    sources: Mapping[str, store.SourceRecord], journal_last_ts: str | None, fallback: str
) -> str:
    """Latest evidence instant seen (sources + journal), never later than ``fallback``."""
    seen = [_instant(r.last_ts) for r in sources.values()] + [_instant(journal_last_ts)]
    instants = [t for t in seen if t is not None]
    ceiling = _instant(fallback)
    if not instants or ceiling is None:
        return fallback
    return utc_stamp(min(max(instants), ceiling))
