"""JSON documents for ``metrics`` commands (integers/strings only)."""

from __future__ import annotations

from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.policy_codec import profile_to_dict
from ssgrep.accounting.service_status import StatusView
from ssgrep.accounting.service_types import NextAction, command_for
from ssgrep.accounting.status_worker import (
    LagStatus,
    LastCycle,
    MissingEvidence,
    Progress,
    StoragePressure,
    WorkerStatus,
)
from ssgrep.cli.accounting_cli import envelope
from ssgrep.cli.savings_json import condition_data, delivery_data


def policy_data(policy: ProfilePolicy) -> dict[str, object]:
    document = profile_to_dict(policy)
    for destination in document["destinations"]:  # credential refs are references, not secrets
        destination["credential_ref"] = "configured" if destination["credential_ref"] else None
    return document


def _last_cycle(last: LastCycle | None) -> dict[str, object] | None:
    if last is None:
        return None
    return {
        "finished_at": last.finished_at,
        "status": last.status,
        "stopped_by": last.stopped_by,
        "error": last.error,
        "generation": last.generation,
        "elapsed_ms": last.elapsed_ms,
        "bytes_read": last.bytes_read,
        "complete": last.complete,
    }


def _progress(progress: Progress | None) -> dict[str, object]:
    keys = ("phase", "mode", "cycles", "sources_done", "sources_total", "bytes_read")
    keys += ("cycle_started_at", "updated_at")
    return {key: getattr(progress, key, None) for key in keys}


def worker_data(worker: WorkerStatus) -> dict[str, object]:
    progress = worker.progress
    return {
        "state": worker.state,
        "reason": worker.reason,
        "kind": worker.kind or getattr(progress, "kind", None),
        "pid": worker.pid if worker.pid is not None else getattr(progress, "pid", None),
        "acquired_at": worker.acquired_at,
        "recoverable": worker.recoverable,
        **_progress(progress),
        "last_cycle": _last_cycle(progress.last_cycle if progress else None),
        "error": progress.last_cycle.error if progress and progress.last_cycle else None,
    }


def lag_data(lag: LagStatus) -> dict[str, object]:
    return {
        "lag_bytes": lag.lag_bytes,
        "basis": lag.basis,
        "sources_total": lag.sources_total,
        "sources_behind": lag.sources_behind,
        "stopped_by": lag.stopped_by,
        "reason": lag.reason,
        "by_runtime": [
            {
                "runtime": r.runtime,
                "sources": r.sources,
                "processed_bytes": r.processed_bytes,
                "total_bytes": r.total_bytes,
                "lag_bytes": r.lag_bytes,
                "missing": r.missing,
            }
            for r in lag.by_runtime
        ],
    }


def watermarks_data(lag: LagStatus) -> dict[str, object]:
    items = None
    if lag.watermarks is not None:
        items = [
            {
                "source": m.source,
                "runtime": m.runtime,
                "offset": m.offset,
                "size": m.size,
                "lag_bytes": m.lag_bytes,
                "pending_tail_bytes": m.pending_tail_bytes,
                "status": m.status,
            }
            for m in lag.watermarks
        ]
    return {
        "count": lag.watermark_count,
        "items": items,
        "truncated": lag.watermarks_truncated,
        "ignored_entries": lag.ignored_entries,
        "reason": None if items is not None else (lag.reason or "watermarks_not_recorded"),
    }


def missing_data(missing: MissingEvidence) -> dict[str, object]:
    return {
        "reason": missing.reason,
        "missing_sources": missing.missing_sources,
        "gaps": None if missing.gaps is None else dict(missing.gaps),
        "journal": {
            "incomplete_tails": missing.incomplete_tails,
            "quarantined": missing.quarantined,
            "conflicts": missing.conflicts,
        },
        "evidence_expired_before": missing.evidence_expired_before,
        "frozen_roots": missing.frozen_roots,
        "capture_completeness": missing.capture_completeness,
    }


def storage_data(storage: StoragePressure) -> dict[str, object]:
    return {
        "level": storage.level,
        "level_reason": storage.level_reason,
        "free_bytes": storage.free_bytes,
        "total_bytes": storage.total_bytes,
        "components": {c.name: c.bytes for c in storage.components},
        "unknown_components": {c.name: c.reason for c in storage.components if c.bytes is None},
        "reason": storage.reason,
    }


def status_document(view: StatusView) -> dict[str, object]:
    policy = view.policy
    data: dict[str, object] = {
        "profile": view.profile,
        "capture": "paused"
        if policy.paused
        else ("enabled" if policy.collection_enabled else "off"),
        "paused": policy.paused,
        "paused_at": policy.paused_at,
        "roots": list(policy.roots),
        "policy": {
            "version": policy.policy_version,
            "source": policy.policy_source,
            "managed": policy.managed,
            "error": view.policy_error,
            "sending_paused_reason": view.sending_paused_reason,
        },
        "worker": worker_data(view.worker),
        "snapshot": {
            "snapshot_id": view.snapshot_id,
            "generation": view.generation,
            "collected_at": view.collected_at,
            "coverage_through": view.coverage_through,
            "age_seconds": view.age_seconds,
            "stale": view.stale,
            "stale_after_seconds": view.stale_after_seconds,
        },
        "lag": lag_data(view.lag),
        "watermarks": watermarks_data(view.lag),
        "missing_evidence": missing_data(view.missing),
        "storage_pressure": storage_data(view.storage),
        "paused_intervals": [{"since": a, "until": b} for a, b in view.gaps],
        "freshness": {
            "collection": {
                "collected_at": view.collected_at,
                "age_seconds": view.collection_age_seconds,
                "stale": view.stale,
                "stale_after_seconds": view.stale_after_seconds,
            },
            "delivery": {
                "age_seconds": view.delivery_age_seconds,
                "acknowledgment": "collector_acceptance_only",
            },
        },
        "conditions": [condition_data(c) for c in view.conditions],
        "destinations": [
            {
                "destination_id": d.destination_id,
                "signals": list(d.signals),
                "enabled_signals": list(d.enabled_signals),
                "delivery": d.delivery,
                "reason": d.reason,
                "lines": [delivery_data(line) for line in d.lines],
            }
            for d in view.destinations
        ],
    }
    return envelope(
        view.state.value,
        data,
        warnings=view.warnings,
        next_actions=view.next_actions,
        provenance={"snapshot_id": view.snapshot_id, "policy_version": policy.policy_version},
    )


def lifecycle_document(
    state: str, policy: ProfilePolicy, profile: str, next_text: str, why: str
) -> dict[str, object]:
    return envelope(
        state,
        {"profile": profile, "policy": policy_data(policy)},
        next_actions=(NextAction(command_for(profile, next_text), why),),
        provenance={"policy_version": policy.policy_version, "policy_source": policy.policy_source},
    )
