"""Build sanitized cycle projections and publish the committed generation."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from ssgrep.accounting.collect import steps, store
from ssgrep.accounting.collect.collector import (
    CollectResult,
    CycleReport,
    RetentionCoverage,
    collection_transition,
    utc_stamp,
)
from ssgrep.accounting.collect.replay import LedgerRows
from ssgrep.accounting.collect.state import ReductionDoc
from ssgrep.accounting.collect.store import CollectError, SourceRecord
from ssgrep.accounting.ledger.contracts import Reuse, canonical_bytes, digest_of
from ssgrep.accounting.ledger.ledger_contracts import (
    LiveCounters,
    ProfilePolicy,
    SnapshotEnvelope,
    backfill_series,
    view,
)
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord, Reduction, summarize
from ssgrep.accounting.ledger.units import UnitSpec
from ssgrep.accounting.ledger.valuation_inputs import Candidate, Methodology
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.storage import from_plain, to_plain
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.history import (
    SealedDay,
    build_windows,
    history_documents,
    quantity_dict,
)
from ssgrep.accounting.views.records import (
    MAX_RECEIPT_BYTES,
    RECEIPT_ID_RE,
    ReceiptRecord,
    receipt_id_for,
)
from ssgrep.accounting.views.snapshots import build_envelope

if TYPE_CHECKING:
    from ssgrep.accounting.collect.collector import CollectOptions
    from ssgrep.accounting.collect.cycle import CycleContext, CycleOutcome
    from ssgrep.accounting.collect.steps import ReductionStage

"""Assemble one snapshot envelope from a cycle's derived state (pure apart from an index probe).

Windows come from event time (``windows.build_windows``); the snapshot ``coverage`` states
processed prefix, lag and gaps (``collector_publish.snapshot_coverage``). The collector's own
completeness verdict is OR-ed into every window's ``coverage.partial`` because the read facade
merges window coverage OVER the snapshot's (a bounded or lagging cycle must never read as a
fully covered 7-day window).
"""


INDEX_DIR = "lancedb"
UNCALIBRATED_NOTE = (
    "thresholds_uncalibrated: rarity thresholds 8/6/4 are uncalibrated; estimates are modeled"
)


def index_state() -> str:
    """``available`` when a search index directory exists; never opened, never created."""
    try:
        return "available" if (storage.data_root() / INDEX_DIR).is_dir() else "unavailable"
    except OSError:
        return "unknown"


@dataclass(frozen=True)
class SnapshotInputs:
    profile: str
    now: datetime
    generation: int
    policy: ProfilePolicy
    methodology: Methodology
    executions: Sequence[ExecutionRecord]
    sealed: Mapping[str, SealedDay]
    rows: LedgerRows
    counters: LiveCounters
    backfills: Sequence[Mapping[str, object]]
    reduction_coverage: Mapping[str, int]
    warnings: Sequence[str]
    sources: Mapping[str, SourceRecord]
    journal: Mapping[str, int]
    journal_last_ts: str | None
    read_gaps: Mapping[str, int]
    dropped: Mapping[str, int]
    stopped_by: str
    retention: RetentionCoverage
    since_day: str | None
    baseline_ts: str | None = None
    frequency: Mapping[str, object] | None = None
    suppressed: int = 0
    collection_scope: Mapping[str, Any] | None = None
    scope_epoch: int = 1


def _enum_counts(executions: Sequence[ExecutionRecord], field: str, names: Sequence[str]) -> dict:
    counts = dict.fromkeys(names, 0)
    for execution in executions:
        counts[getattr(execution, field).value] += 1
    return counts


def _blocks(executions: Sequence[ExecutionRecord]) -> dict[str, Any]:
    """Attribution and reuse-evidence counts over the retained executions."""
    from ssgrep.accounting.ledger.contracts import Attribution, Reuse

    return {
        "attribution": _enum_counts(executions, "attribution", [a.value for a in Attribution]),
        "reuse_evidence": _enum_counts(executions, "reuse", [r.value for r in Reuse]),
    }


def build_snapshot(inputs: SnapshotInputs) -> SnapshotEnvelope:
    from ssgrep.accounting.collect.cycle import coverage_through

    now = inputs.now
    windows = build_windows(
        executions=inputs.executions,
        sealed=inputs.sealed,
        rows=inputs.rows,
        retention=inputs.retention,
        now=now,
        collection_since_day=inputs.since_day,
        evidence_horizon_days=inputs.policy.evidence_retention_days,
    )
    reduction = Reduction(inputs.profile, coverage=dict(inputs.reduction_coverage))
    coverage = snapshot_coverage(
        reduction=reduction,
        sources=inputs.sources,
        journal=inputs.journal,
        read_gaps=inputs.read_gaps,
        dropped=inputs.dropped,
        stopped_by=inputs.stopped_by,
        retention=inputs.retention,
        index=index_state(),
        lag_bytes=0,
    )
    coverage["lag_bytes"] = coverage["sources"]["lag_bytes"]
    coverage.update(_blocks(inputs.executions))
    if inputs.suppressed:
        coverage["suppressed_late_executions"] = inputs.suppressed
    if inputs.frequency is not None:
        coverage["frequency"] = dict(inputs.frequency)
    if inputs.collection_scope is not None:
        coverage["collection_scope"] = dict(inputs.collection_scope)
    coverage["scope_epoch"] = inputs.scope_epoch
    coverage["scope_history_unavailable"] = inputs.scope_epoch > 1
    if inputs.scope_epoch > 1:
        coverage["partial"] = True
        coverage["complete"] = False
    verdict = bool(coverage["partial"])
    for window in windows.values():
        window_coverage = cast("dict[str, Any]", window["coverage"])
        window_coverage["partial"] = bool(window_coverage.get("partial")) or verdict
    warnings = sorted({*inputs.warnings, UNCALIBRATED_NOTE})
    if inputs.stopped_by != "complete":
        warnings.append(
            f"collection_incomplete: stopped by {inputs.stopped_by}; "
            f"{coverage['lag_bytes']} bytes still unread"
        )
    stamp = utc_stamp(now)
    has_evidence = inputs.journal_last_ts is not None or any(
        r.last_ts for r in inputs.sources.values()
    )
    # without evidence the stable capture-start instant stands in: the collection clock
    # must never leak into snapshot identity
    through = (
        coverage_through(inputs.sources, inputs.journal_last_ts, stamp)
        if has_evidence
        else (inputs.baseline_ts or stamp)
    )
    return build_envelope(
        generation=inputs.generation,
        counter_epoch=inputs.counters.epoch,
        methodology_id=inputs.methodology.hash,
        policy_version=inputs.policy.policy_version,
        collected_at=stamp,
        coverage_through=through,
        source_manifest_digest=steps.source_manifest_digest(inputs.sources),
        windows=windows,
        coverage=coverage,
        observations=snapshot_observations(counters=inputs.counters, backfills=inputs.backfills),
        estimates={"representative_receipt": representative_receipt(inputs.rows)},
        warnings=warnings,
    )


"""What the read facade consumes: receipt/history files and snapshot coverage blocks.

Pure builders plus atomic writers, no clocks and no collection logic. Every document
is integers/strings only and must round-trip through ``service_records`` (receipts,
history) and ``service`` (snapshot coverage). Nothing here ever emits a filesystem
path, query, ref, transcript text, receipt content, or witness/action locators:
sources are only ever shown as ``collector_store.source_key`` hashes.
"""


RECEIPT_DIR = "receipts"
HISTORY_DIR = "history"
HISTORY_FILE = "periods.json"
EXCLUSION_PREFIX = "exclusion."
LOOKUP_COUNTERS = (
    "recorded_lookups",
    "attributed_lookups",
    "candidate_receipts",
    "source_costs_available",
    "ambiguous_joins",
    "unavailable_sources",
    "replay_unavailable_sessions",
    "frequency_revision_unavailable",
)
_JOURNAL_GAPS = (
    ("incomplete_tails", "journal_incomplete_tail"),
    ("quarantined", "journal_quarantined"),
    ("conflicts", "journal_conflicts"),
)
_JOURNAL_KEYS = ("segments", "incomplete_tails", "quarantined", "conflicts")
_REUSE_KEYS = frozenset({"class", "relay", "report", "action_ok", "novel", "user_supplied"})


# ---- receipts -----------------------------------------------------------------------------


def receipt_document(record: ReceiptRecord) -> dict[str, Any]:
    """The stored receipt: ``service_records.read_receipt`` shape plus explanatory extras."""
    receipt = record.receipt
    return {
        "schema": receipt.schema,
        "format": "receipt_v2",
        "receipt_id": receipt.receipt_id,
        "revision": receipt.revision,
        "root": receipt.root,
        "task_turn": receipt.task_turn,
        "candidate_type": receipt.candidate_type,
        "atoms": list(receipt.atoms),
        "allocations": [[scenario, list(atoms)] for scenario, atoms in receipt.allocations],
        "exclusions": list(receipt.exclusions),
        "costs": [[name, quantity_dict(amount)] for name, amount in receipt.costs],
        "methodology": receipt.methodology,
        "evidence_available": receipt.evidence_available,
        "execution_ids": list(receipt.execution_ids),
        "exposure_ids": list(receipt.exposure_ids),
        "unit": record.unit,
        "period": record.period,
        "missing_costs": [[name, reason] for name, reason in record.missing_costs],
        "atom_digests": [[atom, digest] for atom, digest in record.atom_digests],
        "credit_raw": [[scenario, raw] for scenario, raw in record.credit_raw],
        "raw_unit": record.raw_unit,
        "debit_raw": record.debit_raw,
        "debit_raw_min": record.debit_raw_min,
        "debit_raw_max": record.debit_raw_max,
        "identities": {
            "frequency": record.frequency_identity,
            "tokenizer": record.tokenizer_identity,
            "pricing": record.pricing_identity,
        },
        "eligibility": receipt.eligibility,
        "eligibility_reasons": list(receipt.eligibility_reasons),
        "provenance": to_plain(receipt.provenance),
        "scenarios": to_plain(receipt.scenarios),
    }


def _latest_receipts(rows: LedgerRows) -> dict[str, ReceiptRecord]:
    latest: dict[str, ReceiptRecord] = {}
    for record in rows.receipts:
        known = latest.get(record.receipt_id)
        if known is None or record.receipt.revision > known.receipt.revision:
            latest[record.receipt_id] = record
    return dict(sorted(latest.items()))


def reuse_block(candidate: Candidate) -> dict[str, Any]:
    """How reuse was established: the reducer's own judgments, enum plus booleans, nothing else."""
    return {
        "class": candidate.reuse.value,
        "relay": candidate.explicit_relay,
        "report": candidate.report_reference,
        "action_ok": candidate.action_outcome_ok,
        "novel": candidate.novel,
        "user_supplied": candidate.user_supplied,
    }


def reuse_blocks(
    candidates: Iterable[Candidate], methodology: Methodology, superseded: Collection[str] = ()
) -> dict[str, dict[str, Any]]:
    """Aggregate observed evidence across associated exposures; never last-writer wins."""
    groups: dict[str, list[Candidate]] = {}
    for cand in candidates:
        if digest_of(cand) not in superseded:
            rid = receipt_id_for(cand.task, cand.candidate_id, methodology)
            groups.setdefault(rid, []).append(cand)
    rank = {
        Reuse.NONE_OBSERVED: 0,
        Reuse.INSPECTED: 1,
        Reuse.ACTION_REFERENCE: 2,
        Reuse.VALIDATED: 3,
        Reuse.CONTRADICTED: 4,
    }
    result = {}
    for receipt_id, members in sorted(groups.items()):
        blocks = [reuse_block(c) for c in members]
        result[receipt_id] = {
            "class": max((c.reuse for c in members), key=rank.__getitem__).value,
            **{key: any(block[key] for block in blocks) for key in sorted(_REUSE_KEYS - {"class"})},
        }
    return result


def _valid_reuse(raw: Any) -> dict[str, Any] | None:
    """An already-stored block, kept only when it has exactly the bounded shape."""
    if not isinstance(raw, dict) or set(raw) != _REUSE_KEYS:
        return None
    if raw["class"] not in {r.value for r in Reuse}:
        return None
    if any(not isinstance(raw[k], bool) for k in _REUSE_KEYS - {"class"}):
        return None
    return {k: raw[k] for k in sorted(raw)}


def _stored_reuse(directory: int, receipt_id: str) -> dict[str, Any] | None:
    """The block of the receipt file already on disk (candidates of frozen roots are gone)."""
    try:
        data = storage.read_file(directory, f"{receipt_id}.json", MAX_RECEIPT_BYTES)
        raw = json.loads(data) if data else None
    except (ValueError, OSError):
        return None
    return _valid_reuse(raw.get("reuse")) if isinstance(raw, dict) else None


def _receipt_files(
    rows: LedgerRows,
    reuse: Mapping[str, Mapping[str, Any]] | None = None,
    directory: int | None = None,
) -> tuple[dict[str, bytes], int]:
    """(file bytes per writable receipt id, number of latest receipts that cannot be written).

    A receipt without a fresh block keeps the one already on disk, so rewriting never erases it.
    """
    files: dict[str, bytes] = {}
    skipped = 0
    for receipt_id, record in _latest_receipts(rows).items():
        document = receipt_document(record)
        block = dict(reuse[receipt_id]) if reuse and receipt_id in reuse else None
        if block is None and directory is not None and RECEIPT_ID_RE.fullmatch(receipt_id):
            block = _stored_reuse(directory, receipt_id)
        if block is not None:
            document["reuse"] = block
        data = canonical_bytes(document) + b"\n"
        if RECEIPT_ID_RE.fullmatch(receipt_id) and len(data) <= MAX_RECEIPT_BYTES:
            files[receipt_id] = data
        else:
            skipped += 1
    return files, skipped


def unwritable_receipts(rows: LedgerRows) -> int:
    """Latest receipts ``write_receipts`` skips (id could be a path, or document too large)."""
    return _receipt_files(rows)[1]


def _write_if_changed(directory: int, name: str, data: bytes) -> bool:
    try:
        existing = storage.read_file(directory, name, len(data))
    except ValueError:  # larger than the new bytes, so certainly different
        existing = None
    if existing == data:
        return False
    storage.atomic_write(directory, name, data)
    return True


def write_receipts(
    profile: str, rows: LedgerRows, reuse: Mapping[str, Mapping[str, Any]] | None = None
) -> int:
    """Write ``receipts/<id>.json`` for each receipt's latest revision; never deletes."""
    if not _receipt_files(rows, reuse)[0]:
        return 0
    directory = storage.open_profile_dir(profile, RECEIPT_DIR)
    try:
        files, _ = _receipt_files(rows, reuse, directory)
        return sum(_write_if_changed(directory, f"{rid}.json", data) for rid, data in files.items())
    finally:
        os.close(directory)


def representative_receipt(rows: LedgerRows) -> str | None:
    """Deterministic receipt for ``savings explain``: biggest positive central credit first."""
    credit: list[tuple[int, str]] = []
    costed: list[str] = []
    for receipt_id, record in _latest_receipts(rows).items():
        if not RECEIPT_ID_RE.fullmatch(receipt_id):
            continue
        central = dict(record.credit_raw).get("central", 0)
        if central > 0:
            credit.append((-central, receipt_id))
        if record.receipt.costs:
            costed.append(receipt_id)
    if credit:
        return min(credit)[1]
    return min(costed) if costed else None


# ---- history ------------------------------------------------------------------------------


def history_document(rows: LedgerRows, unit: UnitSpec, today: str) -> list[dict[str, Any]]:
    """Current and explicitly inactive archived revisions, never summed together."""
    return history_documents(rows, unit, today)


def write_history(profile: str, document: list[dict[str, Any]]) -> bool:
    """Atomically replace ``history/periods.json``; False when the bytes are identical."""
    directory = storage.open_profile_dir(profile, HISTORY_DIR)
    try:
        return _write_if_changed(directory, HISTORY_FILE, canonical_bytes(document) + b"\n")
    finally:
        os.close(directory)


# ---- snapshot blocks ----------------------------------------------------------------------


def _positive(counts: Mapping[str, int]) -> dict[str, int]:
    return {name: int(n) for name, n in counts.items() if int(n) > 0}


def _facts(record: SourceRecord) -> tuple[bool, int, int]:
    """(fully processed, lag bytes, processed bytes) of one source.

    Lag is the unread suffix ``max(size - offset, 0)`` (it includes a torn pending tail),
    except for a baseline-skipped complete source: its pre-existing history was
    deliberately not imported, so nothing of it counts as unread.
    """
    size = max(record.size, 0)
    settled = record.baseline == "skipped" and record.complete
    lag = 0 if settled else max(size - record.offset, 0)
    processed = size if settled else min(record.offset, size)
    done = (
        record.status == "ok" and record.complete and not record.more and record.pending_tail == 0
    )
    return done, lag, processed


def _runtime_blocks(records: Sequence[SourceRecord]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for record in records:
        _, lag, processed = _facts(record)
        block = out.setdefault(
            record.runtime,
            {"sources": 0, "processed_bytes": 0, "total_bytes": 0, "lag_bytes": 0, "missing": 0},
        )
        block["sources"] += 1
        block["processed_bytes"] += processed
        block["total_bytes"] += max(record.size, 0)
        block["lag_bytes"] += lag
        block["missing"] += record.status == "missing"
    return dict(sorted(out.items()))


def _watermark(record: SourceRecord) -> dict[str, Any]:
    _, lag, _ = _facts(record)
    return {
        "source": record.key,
        "runtime": record.runtime,
        "offset": record.offset,
        "size": max(record.size, 0),
        "lag_bytes": lag,
        "pending_tail_bytes": record.pending_tail,
        "status": record.status,
    }


def snapshot_coverage(
    *,
    reduction: Reduction,
    sources: Mapping[str, SourceRecord],
    journal: Mapping[str, int],
    read_gaps: Mapping[str, int],
    dropped: Mapping[str, int],
    stopped_by: str,
    retention: RetentionCoverage,
    index: str,
    lag_bytes: int,
) -> dict[str, Any]:
    """Top-level snapshot ``coverage``: watermarks, lag, gaps, and reducer denominators."""
    records = sorted(sources.values(), key=lambda r: r.key)
    facts = [_facts(r) for r in records]
    journal_block = {k: int(journal[k]) for k in _JOURNAL_KEYS if k in journal}
    gaps: dict[str, int] = {}
    for reason, count in _positive(read_gaps).items():
        gaps[reason] = gaps.get(reason, 0) + count
    for key, reason in _JOURNAL_GAPS:
        if journal_block.get(key, 0) > 0:
            gaps[reason] = gaps.get(reason, 0) + journal_block[key]
    for reason, count in _positive(dropped).items():
        gaps[f"dropped_{reason}"] = gaps.get(f"dropped_{reason}", 0) + count
    complete = (
        all(done for done, _, _ in facts)
        and stopped_by == "complete"
        and journal_block.get("incomplete_tails", 0) == 0
    )
    counters = {k: v for k, v in reduction.coverage.items() if k in LOOKUP_COUNTERS}
    recorded = counters.get("recorded_lookups")
    attributed = counters.get("attributed_lookups")
    partial = (
        not complete
        or counters.get("unavailable_sources", 0) > 0
        or counters.get("ambiguous_joins", 0) > 0
        or counters.get("replay_unavailable_sessions", 0) > 0
        or counters.get("frequency_revision_unavailable", 0) > 0
        or (recorded is not None and attributed is not None and attributed < recorded)
    )
    return {
        "complete": complete,
        "partial": partial,
        "capture_completeness": "unknown",
        "replay": {
            "available": False
            if any(
                counters.get(k, 0) > 0
                for k in ("replay_unavailable_sessions", "frequency_revision_unavailable")
            )
            else True
            if all(
                k in counters
                for k in ("replay_unavailable_sessions", "frequency_revision_unavailable")
            )
            else None,
            "unavailable_sessions": counters.get("replay_unavailable_sessions"),
            "unavailable_frequency_revisions": counters.get("frequency_revision_unavailable"),
        },
        "sources": {
            "total": len(records),
            "complete": sum(done for done, _, _ in facts),
            "lag_bytes": sum(lag for _, lag, _ in facts),
            "pending_tail_bytes": sum(r.pending_tail for r in records),
            "missing": sum(r.status == "missing" for r in records),
            "unreadable": sum(r.status == "unreadable" for r in records),
            "baseline_skipped_bytes": sum(
                r.skipped_bytes for r in records if r.baseline == "skipped"
            ),
            "by_runtime": _runtime_blocks(records),
        },
        "watermarks": [_watermark(r) for r in records],
        "gaps": dict(sorted(gaps.items())),
        "journal": journal_block,
        "stopped_by": stopped_by,
        "collection": collection_transition(stopped_by),
        "lag_bytes": lag_bytes,
        "resets": sum(r.resets for r in records),
        "retention": {
            "evidence_expired_before": retention.evidence_expired_before,
            "aggregates_expired_before": retention.aggregates_expired_before,
            "frozen_roots": retention.frozen_roots,
        },
        **{name: counters[name] for name in LOOKUP_COUNTERS if name in counters},
        "exclusions": dict(
            sorted(
                (name[len(EXCLUSION_PREFIX) :], n)
                for name, n in reduction.coverage.items()
                if name.startswith(EXCLUSION_PREFIX)
            )
        ),
        "index": index,
    }


def snapshot_observations(
    *, counters: LiveCounters | None, backfills: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Top-level snapshot ``observations``: live counters, backfill series, counter epoch."""
    out: dict[str, Any] = {"backfill": [dict(series) for series in backfills]}
    if counters is not None:
        out["counters"] = view(counters)
        out["epoch"] = {
            "counter_epoch": counters.epoch,
            "started_at": counters.epoch_started_at,
        }
    return out


"""Publish one collected cycle through its existing checkpoint acceptance point."""


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
    from ssgrep.accounting.capture.journal_retention import finish_journal_retention
    from ssgrep.accounting.collect.collector import finish as finish_published_retention
    from ssgrep.accounting.collect.cycle import CycleOutcome, stage
    from ssgrep.accounting.collect.frequency_index import finish as finish_frequency_gc
    from ssgrep.accounting.collect.read import journal_last_ts, journal_stats

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
    if prepared.partitions.blocked_reason and stopped == "complete":
        stopped = prepared.partitions.blocked_reason
    elif stopped == "complete" and (
        prepared.partitions.pending or prepared.partitions.pending_projections
    ):
        stopped = "reduction_budget"
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
    if stopped == "complete" and prepared.partitions.unavailable:
        stopped = "replay_unavailable"
        if backfill_done:
            checkpoint.backfill_job["status"] = "partial"
    elif stopped == "complete" and checkpoint.backfill_job:
        stopped = (
            "backfill_blocked"
            if checkpoint.backfill_job.get("status") == "blocked"
            else "backfill_pending"
        )
    from ssgrep.accounting.privacy.published import merge_history

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
    finish_published_retention(profile, publication_work)
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
        configured_budget_seconds=options.budget.seconds,
        required_budget_seconds=(
            max(options.budget.seconds + 1, int(ctx.clock() - started) + 1)
            if stopped == "reduction_blocked"
            else None
        ),
    )
    periods: tuple[str, ...] = ()
    if options.recompute:
        periods = steps.unavailable_periods(
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
        pending_backfill=(
            (checkpoint.backfill_job["since"], checkpoint.backfill_job["until"])
            if checkpoint.backfill_job
            else None
        ),
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

    return CycleOutcome(report, result)


def combine(profile: str, parts: dict[str, dict], frequency_assignments) -> Reduction:
    reductions = [
        from_plain(Reduction, store.read_blob(profile, part["blob"]))
        for _, part in sorted(parts.items())
    ]
    coverage: Counter = Counter()
    bundles = {}
    for reduction in reductions:
        coverage.update(reduction.coverage)
        for root, bundle in zip(reduction.bundle_roots, reduction.bundles, strict=True):
            if root in bundles:
                # A late split can retain an old unavailable component. Never sum its
                # accepted rows with a replacement claiming the same root.
                raise CollectError("state_corrupt", "Collector partition roots overlap.")
            bundles[root] = bundle

    def collect(field):
        return [row for reduction in reductions for row in getattr(reduction, field)]

    executions = tuple(
        sorted(collect("executions"), key=lambda row: (row.ts_utc, row.execution_id))
    )
    invocations = tuple(
        sorted(
            collect("invocations"), key=lambda row: (row.runtime, row.session_id, row.invocation_id)
        )
    )
    candidates = tuple(
        sorted(
            collect("candidates"),
            key=lambda row: (
                row.candidate.task.root,
                row.candidate.task.turn or "",
                row.candidate.candidate_id,
                row.candidate.day,
                row.invocation_id,
            ),
        )
    )
    return Reduction(
        profile,
        executions=executions,
        invocations=invocations,
        candidates=candidates,
        exclusions=tuple(
            sorted(
                set(collect("exclusions")),
                key=lambda row: (row.candidate_id, row.invocation_id, row.reason, row.detail),
            )
        ),
        bundles=tuple(bundles[root] for root in sorted(bundles)),
        bundle_roots=tuple(sorted(bundles)),
        session_roots=tuple(sorted(collect("session_roots"))),
        session_parents=tuple(sorted(collect("session_parents"))),
        coverage=dict(sorted(coverage.items())),
        observations=summarize(executions),
        warnings=tuple(sorted(set(collect("warnings")))),
        frequency_version=reductions[0].frequency_version if reductions else "",
        frequency_assignments=frequency_assignments.records(),
    )
