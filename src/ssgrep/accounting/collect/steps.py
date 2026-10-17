"""Helper steps of a collection cycle (recovery, schema guard, checkpoint, counters, inputs).

Kept separate from ``collector_cycle`` so the cycle reads as the protocol it is: recover,
read, reduce, value, expire, publish, commit.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from ssgrep.accounting.collect.collector import CollectOptions, Reducer
    from ssgrep.accounting.collect.cycle import CycleContext
    from ssgrep.accounting.collect.partitions import PartitionState
    from ssgrep.accounting.collect.state import VersionedLedger

from ssgrep.accounting.capture.journal_reader import JournalRead
from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.collector import (
    CAPTURE_GRACE_SECONDS,
    MAX_REPORTED_DAYS,
    RetentionCoverage,
    parse_stamp,
    utc_stamp,
)
from ssgrep.accounting.collect.frequency import FrequencyView, model_identity
from ssgrep.accounting.collect.replay import ArchivedRows
from ssgrep.accounting.collect.state import (
    RETIRED_ARCHIVE_PREFIX,
    RETIRED_VALUATION_VERSION,
    ReductionDoc,
    State,
    load_state,
    save_state,
)
from ssgrep.accounting.collect.store import (
    Checkpoint,
    CollectError,
    SourceRecord,
    restore_cache,
)
from ssgrep.accounting.ledger.contracts import (
    SCHEMA_VERSION,
    Attribution,
    InvocationClass,
    RawEvidence,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_contracts import (
    LiveCounters,
    ProfilePolicy,
    empty_counters,
    observe,
    prune,
    start_epoch,
)
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord, Reduction
from ssgrep.accounting.privacy import storage, storage as plainfile
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.records import MAX_RECEIPT_BYTES
from ssgrep.accounting.views.snapshots import MAX_SNAPSHOT_BYTES

KEEP_GENERATIONS = 64


def guard_schema(profile: str) -> None:
    """Refuse to overwrite a snapshot written by a NEWER ssgrep (older binaries stay read-only)."""
    try:
        raw = plainfile.read_private(
            storage.data_root(),
            "accounting",
            profile,
            "snapshots",
            "current.json",
            max_bytes=MAX_SNAPSHOT_BYTES,
        )
        document = None if raw is None else store.loads(raw)
    except (plainfile.UnsafePathError, ValueError, OSError):
        return  # unreadable: the snapshot recovery step reports or repairs it
    version = document.get("schema_version") if isinstance(document, dict) else None
    if isinstance(version, int) and not isinstance(version, bool) and version > SCHEMA_VERSION:
        raise CollectError(
            "unsupported_schema",
            f"Accounting state uses schema {version}; this ssgrep reads schema {SCHEMA_VERSION}. "
            "Nothing was changed; upgrade ssgrep to collect.",
        )


def recover(profile: str, checkpoint: Checkpoint | None) -> State | None:
    """Validate authoritative state before orphan GC can destroy recovery evidence."""
    validated = None
    legacy = False
    if checkpoint is not None:
        # Cached reductions may not open retained evidence at all. Check required
        # committed directory authority before recovery/GC or any staging creates.
        for component, required in (
            (store.EVIDENCE_DIR, bool(checkpoint.segments)),
            (store.STATE_DIR, bool(checkpoint.blobs)),
        ):
            if not required:
                continue
            try:
                descriptor = storage.open_profile_dir(profile, "collector", component, create=False)
            except FileNotFoundError as error:
                raise CollectError(
                    "state_corrupt", f"Committed collector {component} directory is missing."
                ) from error
            os.close(descriptor)
        from ssgrep.accounting.privacy.published_legacy import validate_checkpoint

        legacy = validate_checkpoint(profile, checkpoint)
    if checkpoint is not None and checkpoint.methodology and not legacy:
        validated = load_state(profile, checkpoint, checkpoint.methodology)
        if (
            validated.ledger is not None
            and validated.ledger.methodology.hash != checkpoint.methodology
        ):
            raise CollectError("state_corrupt", "Ledger methodology does not match the checkpoint.")
    generation = checkpoint.generation if checkpoint else 0
    try:
        snapshots.discard_uncommitted(profile, generation)
        snapshots.sweep_temporaries(profile)
        if generation:
            snapshots.repair_current(profile, generation)
    except (ValueError, FileNotFoundError) as error:
        if str(error) == "uncommitted_pinned_generation":
            raise CollectError(
                "state_corrupt",
                "An uncommitted snapshot is uniquely pinned by delivery state; "
                "restore a coherent checkpoint before collecting. Nothing was deleted.",
            ) from error
        raise CollectError(
            "state_corrupt",
            "Snapshot files do not match the collector checkpoint; nothing was changed.",
        ) from error
    if not legacy:
        store.sweep_orphans(profile, checkpoint)
    return validated


def capture_start(journal: JournalRead, now: datetime) -> datetime:
    """When capture began: the first journal event (minus a grace), else this moment."""
    stamps = [parse_stamp(event.ts_utc) for event in journal.events]
    first = min((s for s in stamps if s is not None), default=None)
    if first is None:
        return now
    return min(first - timedelta(seconds=CAPTURE_GRACE_SECONDS), now)


def new_checkpoint(profile: str, journal: JournalRead, now: datetime) -> Checkpoint:
    start = capture_start(journal, now)
    from ssgrep.accounting.collect.replay import COMPACT

    return Checkpoint(
        profile,
        baseline_ts=utc_stamp(start),
        since_day=utc_stamp(start)[:10],
        retention={"evidence_format": COMPACT, "archive_evidence_format": COMPACT},
    )


def input_digest(
    policy: ProfilePolicy, frequency: FrequencyView, fingerprint_key_id: str | None
) -> str:
    """Everything besides evidence/journal that changes a reduction's output."""
    return digest_of(
        [policy.policy_version, list(policy.roots), model_identity(frequency), fingerprint_key_id]
    )


def counters_step(
    previous: LiveCounters | None,
    journal: JournalRead,
    *,
    now: datetime,
    horizon_day: str,
    new_epoch: bool,
    observing: bool,
) -> LiveCounters:
    """Live counters for this cycle (epoch per spec: restart preserves, --new-epoch resets)."""
    stamp = utc_stamp(now)
    if new_epoch:
        counters = start_epoch(previous, journal.events, now)
    else:
        counters = previous or empty_counters(1, stamp)
    if observing:
        counters = observe(counters, journal.events, horizon_day=horizon_day)
        counters = prune(counters, horizon_day)
    return counters


# Drops that are expected housekeeping (usage records are never retained; a copy is a copy):
# counting them as "gaps" would bury the real ones.
EXPECTED_DROPS = frozenset({"kind_not_consumed", "duplicate"})


def merge_dropped(total: Mapping[str, int], new: Mapping[str, int]) -> dict[str, int]:
    """Cumulative count of evidence deliberately not retained, by reason (live reads only)."""
    merged = dict(total)
    for reason, count in new.items():
        if reason not in EXPECTED_DROPS and count > 0:
            merged[reason] = merged.get(reason, 0) + count
    return dict(sorted(merged.items()))


def source_gaps(sources: Mapping[str, SourceRecord]) -> dict[str, int]:
    """Cumulative reader gaps (malformed, oversized, reset, ...) over every known source."""
    total: dict[str, int] = {}
    for record in sources.values():
        for reason, count in record.gaps.items():
            total[reason] = total.get(reason, 0) + count
    return dict(sorted(total.items()))


def source_manifest_digest(sources: Mapping[str, SourceRecord]) -> str:
    """Digest source identity/state without paths or clocks."""
    entries = sorted(
        [
            record.key,
            record.runtime,
            record.offset,
            max(record.size, 0),
            record.status,
            record.baseline,
        ]
        for record in sources.values()
    )
    return digest_of(entries)


def prune_receipt_files(profile: str, before_day: str) -> int:
    """Expire dated receipts across namespaces; never remove an unexpired archived receipt."""
    try:
        directory = storage.open_profile_dir(profile, "receipts", create=False)
    except FileNotFoundError:
        return 0
    removed = 0
    try:
        for name in storage.list_names(directory):
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                data = storage.read_file(directory, name, MAX_RECEIPT_BYTES)
                raw = store.loads(data) if data is not None else None
                period = raw.get("period") if isinstance(raw, dict) else None
                if not isinstance(period, str) or date.fromisoformat(period).isoformat() != period:
                    continue
            except (ValueError, OSError):
                continue  # an unreadable receipt has no trustworthy expiry date
            if period < before_day:
                removed += storage.delete_file(directory, name)
    finally:
        os.close(directory)
    return removed


def lag_bytes(coverage: Mapping[str, object]) -> int:
    lag = coverage.get("lag_bytes")
    return lag if isinstance(lag, int) and not isinstance(lag, bool) else 0


def aggregates_expired_before(checkpoint: Checkpoint) -> str | None:
    value = checkpoint.retention.get("aggregates_expired_before")
    return value if isinstance(value, str) else None


def session_seen(items: Iterable[RawEvidence]) -> dict[tuple[str, str], str]:
    """Latest evidence timestamp per ``(runtime, session)`` among newly accepted records."""
    seen: dict[tuple[str, str], str] = {}
    for item in items:
        if item.ts_utc and item.ts_utc > seen.get((item.runtime, item.session_id), ""):
            seen[(item.runtime, item.session_id)] = item.ts_utc
    return seen


def last_tuple(last: Mapping[tuple[str, str], str]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((runtime, session, ts) for (runtime, session), ts in last.items()))


def session_cwds(reduction: Reduction) -> tuple[tuple[str, str, str], ...]:
    """Working directory per in-scope session: the earliest invocation that recorded one."""
    earliest: dict[tuple[str, str], tuple[str, str, str]] = {}
    for record in reduction.invocations:
        if not record.cwd:
            continue
        key = (record.runtime, record.session_id)
        order = (record.ts_utc or "", record.invocation_id, record.cwd)
        if key not in earliest or order < earliest[key]:
            earliest[key] = order
    return tuple(
        sorted((runtime, session, found[2]) for (runtime, session), found in earliest.items())
    )


def invocation_sessions(reduction: Reduction) -> tuple[tuple[str, str, str], ...]:
    """(invocation id, runtime, session) of every recognized invocation, sorted."""
    return tuple(sorted((r.invocation_id, r.runtime, r.session_id) for r in reduction.invocations))


def reduction_doc(reduction: Reduction, last: Mapping[tuple[str, str], str]) -> ReductionDoc:
    return ReductionDoc(
        executions=reduction.executions,
        coverage=dict(reduction.coverage),
        warnings=tuple(reduction.warnings),
        digest=reduction.digest,
        session_roots=reduction.session_roots,
        session_last=last_tuple(last),
        session_parents=reduction.session_parents,
        session_cwd=session_cwds(reduction),
        invocation_sessions=invocation_sessions(reduction),
        invocations=reduction.invocations,
        lookup_schema="lookups_v1",
    )


def sealing_executions(
    previous: tuple[ExecutionRecord, ...],
    current: tuple[ExecutionRecord, ...],
    *,
    evidence_before: str,
    before_day: str,
    aggregate_before: str | None,
    already: Mapping[str, object],
) -> tuple[ExecutionRecord, ...]:
    """Carry accepted compact usage into sealing before expired inputs disappear.

    Only old, unsealed observations inside the still-retained aggregate range
    qualify. Current rows win identities; no raw replay or live counter update
    occurs. The caller's state is already the post-purge, current-scope state.
    """
    lost = [
        row
        for row in previous
        if row.in_scope
        and row.day < evidence_before
        and row.day < before_day
        and row.day not in already
        and (aggregate_before is None or row.day >= aggregate_before)
    ]
    if not lost:
        return current
    present = {row.execution_id for row in current}
    return (*current, *(row for row in lost if row.execution_id not in present))


_SEP = "|"


def encode_sessions(pairs: Iterable[tuple[str, str]]) -> list[str]:
    return sorted(f"{runtime}{_SEP}{session}" for runtime, session in pairs)


def decode_sessions(values: Iterable[str]) -> frozenset[tuple[str, str]]:
    out = set()
    for value in values:
        runtime, _, session = str(value).partition(_SEP)
        out.add((runtime, session))
    return frozenset(out)


def previous_coverage(retention: Mapping[str, object]) -> RetentionCoverage:
    def day(name: str) -> str | None:
        value = retention.get(name)
        return value if isinstance(value, str) else None

    frozen = retention.get("frozen_roots")
    return RetentionCoverage(
        day("evidence_expired_before"),
        day("aggregates_expired_before"),
        frozen if isinstance(frozen, int) and not isinstance(frozen, bool) else 0,
    )


def retention_record(
    coverage: RetentionCoverage,
    sessions: Iterable[tuple[str, str]],
    *,
    journal: Mapping[str, object] | None = None,
) -> dict:
    """Stage pure expiry coverage and the journal plan in the SAME acceptance commit.

    Physical journal maintenance runs only after this record is durable; it cannot
    be folded into ``apply_expiry`` before checkpoint publication.
    """
    record: dict[str, object] = {
        "evidence_expired_before": coverage.evidence_expired_before,
        "aggregates_expired_before": coverage.aggregates_expired_before,
        "frozen_roots": coverage.frozen_roots,
        "expired_sessions": encode_sessions(sessions),
    }
    if journal is not None:
        record["journal"] = dict(journal)
    return record


def merge_session_last(
    previous: Iterable[tuple[str, str, str]], seen: Mapping[tuple[str, str], str]
) -> dict[tuple[str, str], str]:
    """Latest evidence timestamp per ``(runtime, session)`` (only ever moves forward)."""
    merged = {(runtime, session): ts for runtime, session, ts in previous}
    for key, ts in seen.items():
        if ts > merged.get(key, ""):
            merged[key] = ts
    return merged


def root_last_evidence(
    doc: ReductionDoc, session_last: Mapping[tuple[str, str], str]
) -> dict[str, str]:
    """Latest evidence timestamp of ANY record of each root conversation."""
    latest: dict[str, str] = {}
    for runtime, session, root in doc.session_roots:
        ts = session_last.get((runtime, session))
        if ts is not None and ts > latest.get(root, ""):
            latest[root] = ts
    return latest


"""Collection's affected-closure phase before counters and retention."""


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
    from ssgrep.accounting.capture.fingerprint import key_id
    from ssgrep.accounting.collect.collector import expire_journal_events
    from ssgrep.accounting.collect.partitions import load_partitions
    from ssgrep.accounting.collect.read import journal_signature
    from ssgrep.accounting.collect.state import ReductionDoc

    journal_sig = journal_signature(journal)
    digest = input_digest(policy, frequency, key_id(fingerprint_key))
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
        from ssgrep.accounting.ledger.reduce import native_source_adapter

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
    seen = session_seen(new_items)
    last = merge_session_last(state.reduction.session_last if state.reduction else (), seen)
    doc = reduction_doc(reduction, last) if reduction is not None else state.reduction
    doc = pending_document(state.reduction, doc or ReductionDoc(), partitions)
    doc = partition_coverage(doc, partitions)
    from ssgrep.accounting.views.filter import effective_scoped_journal

    scoped_journal = replace(
        journal,
        events=effective_scoped_journal(journal.events, doc.executions, scoper),
    )
    if reduction is None:
        doc = replace(doc, session_last=last_tuple(last))
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


def source_progress(records) -> dict[str, Any]:
    """Aggregate logical progress without claiming that a settled source is complete."""
    counts = {
        "sources_complete": 0,
        "sources_unvisited": 0,
        "sources_pending": 0,
        "rows_visited": 0,
        "pending_records": 0,
        "abandoned_records": 0,
        "unsupported_sources": 0,
        "known_pending_lower_bound": 0,
        "pending_count_complete": True,
    }
    for record in records:
        counts["sources_complete"] += int(record.complete)
        if record.runtime != "opencode":
            continue
        stats = record.state.get("extra", {}).get("opencode", {})
        counts["sources_unvisited"] += int(
            stats.get("status", "unvisited") == "unvisited"
            or stats.get("snapshot_checked") is False
        )
        pending = int(stats.get("known_pending_lower_bound", stats.get("pending_records") or 0))
        unknown = (
            stats.get("pending_records") is None
            or record.status != "ok"
            or stats.get("snapshot_checked") is False
        )
        counts["sources_pending"] += int(pending > 0 or stats.get("status") == "pending")
        counts["known_pending_lower_bound"] += pending
        counts["pending_count_complete"] = counts["pending_count_complete"] and not unknown
        counts["pending_records"] += pending
        counts["abandoned_records"] += int(stats.get("abandoned_records", 0))
        counts["unsupported_sources"] += int(stats.get("status") == "unsupported")
        counts["rows_visited"] += sum(int(n) for n in stats.get("rows_visited", {}).values())
    if not counts["pending_count_complete"]:
        counts["pending_records"] = None
    return counts


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


def index_segment(profile, ref):
    from ssgrep.accounting.collect.cycle import IndexWriter
    from ssgrep.accounting.collect.replay import evidence_from_line

    directory = storage.open_profile_dir(profile, "collector", store.EVIDENCE_DIR, create=False)
    try:
        body = storage.read_file(directory, ref.name, ref.size + 1)
    except ValueError as error:
        raise CollectError(
            "state_corrupt", "Legacy retained segment is missing or damaged."
        ) from error
    finally:
        os.close(directory)
    if body is None or len(body) != ref.size:
        raise CollectError("state_corrupt", "Legacy retained segment is missing or damaged.")
    index = IndexWriter()
    offset = 0
    for line in body.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise CollectError("state_corrupt", "Legacy retained segment is incomplete.")
        index.add(evidence_from_line(line.rstrip(b"\n")), offset, line)
        offset += len(line)
    return replace(ref, index=index.write(profile))


def ensure_indexes(profile, checkpoint, *, clock, deadline, byte_budget, should_stop):
    """Run named one-time legacy index work before admitting new native evidence."""
    used = 0
    refs = []
    for ref in checkpoint.segments:
        if ref.index is None and used < byte_budget and not should_stop() and clock() < deadline:
            refs.append(index_segment(profile, ref))
            used += ref.size
        else:
            refs.append(ref)
    checkpoint.segments = refs
    pending = [ref.name for ref in refs if ref.index is None]
    if pending:
        checkpoint.retention["evidence_index_pending"] = pending
    else:
        checkpoint.retention.pop("evidence_index_pending", None)
    return used, bool(pending)


def compact_pair(refs):
    """A count target cannot make already-full segments require impossible work."""
    if len(refs) <= store.MAX_SEGMENTS:
        return None
    for origin in sorted({ref.origin for ref in refs}):
        candidates = sorted(
            (ref for ref in refs if ref.origin == origin), key=lambda ref: (ref.size, ref.name)
        )
        if (
            len(candidates) >= 2
            and candidates[0].size + candidates[1].size <= store.SEGMENT_TARGET_BYTES
        ):
            return candidates[0], candidates[1]
    return None


def maintain_evidence(
    profile, refs, *, expired_sessions, previous, clock, deadline, byte_budget, should_stop
):
    """Return immutable replacement refs + explicit remaining maintenance job."""
    from ssgrep.accounting.collect.cycle import read_index, session_name

    job: dict[str, object] = dict(previous or {})
    expiry_digest = digest_of(sorted(expired_sessions))
    if job.get("expired_digest") != expiry_digest:
        job = {
            "expired_digest": expiry_digest,
            "pending": [ref.name for ref in refs] if expired_sessions else [],
        }
    pending = job.get("pending", [])
    if not isinstance(pending, list):
        raise CollectError("state_corrupt", "Collector evidence maintenance queue is invalid.")
    pending = list(pending)
    refs = list(refs)
    used = dropped = 0
    while pending and used < byte_budget and not should_stop() and clock() < deadline:
        name = pending.pop(0)
        ref = next((ref for ref in refs if ref.name == name), None)
        if ref is None:
            continue
        index = read_index(profile, ref)
        wanted = {session_name(key) for key in expired_sessions}
        if index is not None and not wanted.intersection(index["sessions"]):
            continue
        items = list(store.read_segment(profile, ref))
        kept = [item for item in items if (item.runtime, item.session_id) not in expired_sessions]
        replacements = store.write_segments(profile, kept, origin=ref.origin) if kept else []
        refs = [current for current in refs if current.name != name] + replacements
        used += ref.size
        dropped += len(items) - len(kept)
    # Compaction is incremental too. Preserve live/backfill segment origins.
    pair = compact_pair(refs)
    while (
        not pending
        and pair is not None
        and used < byte_budget
        and not should_stop()
        and clock() < deadline
    ):
        first, second = pair
        size = first.size + second.size
        if used and used + size > byte_budget:
            break
        replacements = store.write_segments(
            profile, store.iter_evidence(profile, (first, second)), origin=first.origin
        )
        if len(replacements) >= 2:
            # Both input segments are already full; no productive bounded merge.
            break
        refs = [ref for ref in refs if ref.name not in {first.name, second.name}] + replacements
        used += size
        pair = compact_pair(refs)
    job["pending"] = pending
    job["compaction_pending"] = compact_pair(refs) is not None
    return refs, job, used, dropped


def source_proof(record):
    return (
        record.source_id,
        record.size,
        record.mtime_ns,
        record.status,
        record.resets,
        digest_of(
            [
                record.state.get("extra", {}).get("collector_source_signature"),
                record.state.get("extra", {}).get("collector_source_generation"),
            ]
        ),
    )


def changed_sources(previous, sources):
    return [record for key, record in sources.items() if previous.get(key) != source_proof(record)]


def pending_document(
    previous: ReductionDoc | None, current: ReductionDoc, partitions
) -> ReductionDoc:
    """A cache bootstrap/purge queue never discards accepted unvisited window rows."""
    if previous is None or (
        not partitions.pending and not partitions.pending_projections and not partitions.unavailable
    ):
        return current
    represented = {member for part in partitions.parts.values() for member in part["members"]}
    sessions = {tuple(name.split(":", 1)) for name in represented if not name.startswith("@")}
    execution_ids = {
        name[len("@journal:") :] for name in represented if name.startswith("@journal:")
    }
    executions = {
        row.execution_id: row
        for row in previous.executions
        if row.execution_id not in execution_ids and (row.runtime, row.session_id) not in sessions
    }
    executions.update((row.execution_id, row) for row in current.executions)
    invocations = {
        row.invocation_id: row
        for row in previous.invocations
        if (row.runtime, row.session_id) not in sessions
    }
    invocations.update((row.invocation_id, row) for row in current.invocations)

    def pairs(field):
        kept = {
            (row[0], row[1]): row
            for row in getattr(previous, field)
            if (row[0], row[1]) not in sessions
        }
        kept.update(((row[0], row[1]), row) for row in getattr(current, field))
        return tuple(kept[key] for key in sorted(kept))

    coverage = dict(current.coverage)
    coverage["recorded_executions"] = len(executions)
    coverage["recorded_lookups"] = sum(
        row.operation in {"search", "show"} for row in executions.values()
    )
    coverage["recognized_invocations"] = len(invocations)
    for attribution in Attribution:
        coverage[attribution.value] = sum(
            row.attribution is attribution for row in executions.values()
        )
    for cls in InvocationClass:
        coverage["inv_" + cls.value] = sum(row.cls is cls for row in invocations.values())
    coverage["ambiguous_joins"] = coverage[Attribution.AMBIGUOUS.value]
    coverage["attributed_lookups"] = sum(
        row.operation in {"search", "show"}
        and row.attribution in {Attribution.EXPLICIT, Attribution.INFERRED_UNIQUE}
        for row in executions.values()
    )
    coverage["tasks_resolved"] = sum(
        row.task is not None and row.task.resolved for row in invocations.values()
    )
    coverage["tasks_unknown"] = len(invocations) - coverage["tasks_resolved"]
    return replace(
        current,
        executions=tuple(
            sorted(executions.values(), key=lambda row: (row.ts_utc, row.execution_id))
        ),
        invocations=tuple(
            sorted(
                invocations.values(),
                key=lambda row: (row.runtime, row.session_id, row.invocation_id),
            )
        ),
        coverage=coverage,
        session_roots=pairs("session_roots"),
        session_parents=pairs("session_parents"),
        session_cwd=pairs("session_cwd"),
        invocation_sessions=tuple(
            sorted((row.invocation_id, row.runtime, row.session_id) for row in invocations.values())
        ),
        warnings=tuple(sorted(set(current.warnings) | set(previous.warnings))),
    )


def partition_coverage(doc: ReductionDoc, state) -> ReductionDoc:
    coverage = dict(doc.coverage)
    coverage.update(
        reduction_pending_nodes=len(state.pending) + len(state.pending_projections),
        replay_unavailable_sessions=len(state.unavailable),
        reduction_cached_partitions=len(state.parts),
    )
    warnings = set(doc.warnings)
    if state.pending or state.pending_projections:
        warnings.add("reduction_backlog")
    else:
        warnings.discard("reduction_backlog")
    if state.unavailable:
        warnings.add("historical_replay_unavailable")
    else:
        warnings.discard("historical_replay_unavailable")
    return replace(doc, coverage=coverage, warnings=tuple(sorted(warnings)))


def archive_retired_rows(profile: str, ledger: VersionedLedger, blobs: Mapping[str, str]) -> None:
    """Attach original accepted v3 rows in memory; read-only callers never persist them."""
    if ledger.methodology.valuation_version != RETIRED_VALUATION_VERSION:
        return
    name = RETIRED_ARCHIVE_PREFIX + ledger.namespace
    if name in ledger.archives:
        return
    restore_cache(profile, ledger, blobs)
    try:
        accepted = ledger._canonical_rows()
    except ValueError as error:
        if str(error) != "methodology_read_only":
            raise
        raise CollectError(
            "methodology_history_unavailable",
            "Accepted v3 root evaluations are unavailable. Restore the matching committed "
            "root-evaluations cache before collecting or purging; "
            "historical amounts were not recomputed.",
        ) from error
    ledger.archives[name] = ArchivedRows(
        name, accepted.namespace, accepted.periods, accepted.receipts
    )


def retire_methodology(profile: str, checkpoint: Checkpoint, current_hash: str) -> None:
    """Keep accepted v3 rows before the collector replaces its current evaluation cache.

    No old valuation runs here: every live subset must already have its exact
    committed evaluation, while expired roots carry their accepted frozen leaves.
    The original archive remains byte-identical; its read-only projection and
    privacy erasure use those accepted leaves, never the corrected v4 allocator.
    """
    from ssgrep.accounting.collect.state import VersionedLedger

    if not checkpoint.methodology or checkpoint.methodology == current_hash:
        return
    previous = load_state(profile, checkpoint, checkpoint.methodology)
    if previous.ledger is None:
        return
    method = previous.ledger.methodology
    if method.hash != checkpoint.methodology:
        raise CollectError("state_corrupt", "Ledger methodology does not match the checkpoint.")
    if method.valuation_version != RETIRED_VALUATION_VERSION:
        return
    name = RETIRED_ARCHIVE_PREFIX + method.hash
    if name in previous.archives:
        return
    ledger = VersionedLedger.restore_versioned(
        previous.ledger, previous.superseded, previous.archives
    )
    archive_retired_rows(profile, ledger, checkpoint.blobs)
    retired = replace(previous, archives=ledger.archives_state())
    save_state(profile, retired, method.hash, checkpoint.blobs, previous=previous)
