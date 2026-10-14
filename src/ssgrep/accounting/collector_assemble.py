"""Assemble one snapshot envelope from a cycle's derived state (pure apart from an index probe).

Windows come from event time (``windows.build_windows``); the snapshot ``coverage`` states
processed prefix, lag and gaps (``collector_publish.snapshot_coverage``). The collector's own
completeness verdict is OR-ed into every window's ``coverage.partial`` because the read facade
merges window coverage OVER the snapshot's (a bounded or lagging cycle must never read as a
fully covered 7-day window).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from ssgrep.accounting import storage
from ssgrep.accounting.collector_publish import (
    coverage_through,
    representative_receipt,
    snapshot_coverage,
    snapshot_observations,
    source_manifest_digest,
)
from ssgrep.accounting.collector_store import SourceRecord
from ssgrep.accounting.collector_types import RetentionCoverage
from ssgrep.accounting.counters import LiveCounters
from ssgrep.accounting.ledger_contracts import ProfilePolicy, SnapshotEnvelope
from ssgrep.accounting.reduce_types import ExecutionRecord, Reduction
from ssgrep.accounting.replay import LedgerRows
from ssgrep.accounting.snapshots import build_envelope
from ssgrep.accounting.valuation_inputs import Methodology
from ssgrep.accounting.windows import SealedDay, build_windows

UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
INDEX_DIR = "lancedb"
UNCALIBRATED_NOTE = (
    "thresholds_uncalibrated: rarity thresholds 8/6/4 are uncalibrated; estimates are modeled"
)


def utc_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(UTC_FORMAT)


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
    from ssgrep.accounting.contracts import Attribution, Reuse

    return {
        "attribution": _enum_counts(executions, "attribution", [a.value for a in Attribution]),
        "reuse_evidence": _enum_counts(executions, "reuse", [r.value for r in Reuse]),
    }


def build_snapshot(inputs: SnapshotInputs) -> SnapshotEnvelope:
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
        source_manifest_digest=source_manifest_digest(inputs.sources),
        windows=windows,
        coverage=coverage,
        observations=snapshot_observations(counters=inputs.counters, backfills=inputs.backfills),
        estimates={"representative_receipt": representative_receipt(inputs.rows)},
        warnings=warnings,
    )
