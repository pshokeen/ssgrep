"""JSON documents for ``metrics collect`` and ``metrics serve`` (integers/strings/null only).

The human renderers consume these same ``data`` dicts, so text and JSON always
state the same facts. Values read from ``worker/progress.json`` are untrusted
files: every number is re-checked as a non-negative integer and every string is
bounded, so a hand-edited document can never inject floats or huge text.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.collector_types import CollectOptions, CollectResult
from ssgrep.accounting.service_types import NextAction, command_for
from ssgrep.cli.accounting_cli import envelope

if TYPE_CHECKING:
    from ssgrep.accounting.lease import AlreadyRunning, LeaseInfo
    from ssgrep.accounting.serve_loop import WorkerState

SHORT_ID = 12
TEXT_LIMIT = 64
DEFAULT_UNAVAILABLE_REASON = "evidence_expired"
_MAX_DEPTH = 4


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _str(value: object) -> str | None:
    return value[:TEXT_LIMIT] if isinstance(value, str) and value else None


def plain(value: object, depth: int = 0) -> Any:
    """``value`` reduced to JSON-safe ints/strings/bools/null (floats and objects dropped)."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if depth < _MAX_DEPTH and isinstance(value, dict):
        return {str(key): plain(item, depth + 1) for key, item in value.items()}
    if depth < _MAX_DEPTH and isinstance(value, list | tuple):
        return [plain(item, depth + 1) for item in value]
    return None


def _recorded_lease(profile: str) -> LeaseInfo | None:
    from ssgrep.accounting.lease import inspect_lease

    try:
        return inspect_lease(profile)
    except (OSError, ValueError):
        return None


def owner_view(profile: str, progress: dict[str, object] | None) -> dict[str, object]:
    """The running owner: pid/kind from the lease, numbers from its progress file.

    Progress written by a different pid than the lease owner is stale and ignored.
    """
    lease = _recorded_lease(profile)
    recorded = progress if isinstance(progress, dict) else None
    if recorded is not None and lease is not None and lease.pid is not None:
        if recorded.get("pid") != lease.pid:
            recorded = None
    fresh = recorded or {}
    pid = lease.pid if lease is not None and lease.pid is not None else _int(fresh.get("pid"))
    kind = lease.kind if lease is not None and lease.kind else _str(fresh.get("kind"))
    return {
        "pid": pid,
        "kind": kind,
        "alive": lease.alive if lease is not None else None,
        "acquired_at": lease.acquired_at if lease is not None else None,
        "phase": _str(fresh.get("phase")),
        "mode": _str(fresh.get("mode")),
        "cycle_started_at": _str(fresh.get("cycle_started_at")),
        "updated_at": _str(fresh.get("updated_at")),
        "cycles": _int(fresh.get("cycles")),
        "sources_done": _int(fresh.get("sources_done")),
        "sources_total": _int(fresh.get("sources_total")),
        "bytes_read": _int(fresh.get("bytes_read")),
        "lag_bytes": _int(fresh.get("lag_bytes")),
    }


def _unavailable(entries: tuple[str, ...]) -> list[dict[str, str]]:
    out = []
    for entry in entries:
        period, _, reason = entry.partition("=")
        out.append({"period": period, "reason": reason or DEFAULT_UNAVAILABLE_REASON})
    return out


def resume_command(profile: str, options: CollectOptions) -> str:
    """The collect command that continues ``options`` (never re-declares ``--new-epoch``)."""
    parts = ["metrics collect"]
    if options.backfill:
        parts.append("--backfill")
    if options.recompute:
        parts.append("--recompute")
    if options.since:
        parts.append(f"--since {options.since}")
    if options.until:
        parts.append(f"--until {options.until}")
    return command_for(profile, " ".join(parts))


def next_actions(
    profile: str, options: CollectOptions, result: CollectResult
) -> tuple[NextAction, ...]:
    if result.state == "already_running":
        return (NextAction(command_for(profile, "metrics status"), "See the running worker."),)
    savings = NextAction(command_for(profile, "savings"), "Read the updated snapshot.")
    if result.state == "partial":
        resume = NextAction(resume_command(profile, options), "Resume where this cycle stopped.")
        return (resume, savings)
    return (savings,)


def collect_data(profile: str, options: CollectOptions, result: CollectResult) -> dict[str, Any]:
    report = result.report
    data: dict[str, Any] = {
        "profile": profile,
        "mode": result.mode,
        "state": result.state,
        "interval": {"since": options.since, "until": options.until},
        "new_epoch": options.new_epoch,
        "epoch": _int(result.epoch),
        "generation": report.generation if report else None,
        "snapshot_id": report.snapshot_id if report else None,
        "stopped_by": report.stopped_by if report else None,
        "complete": report.complete if report else None,
        "bytes_read": report.bytes_read if report else None,
        "lag_bytes": report.lag_bytes if report else None,
        "sources": report.sources if report else None,
        "new_evidence": report.new_evidence if report else None,
        "elapsed_ms": report.elapsed_ms if report else None,
        "backfill": plain(result.backfill) if result.backfill is not None else None,
        "unavailable_periods": _unavailable(result.unavailable_periods),
        "owner": owner_view(profile, result.progress)
        if result.state == "already_running"
        else None,
    }
    return data


def collect_document(
    profile: str, options: CollectOptions, result: CollectResult
) -> tuple[dict[str, object], dict[str, Any], tuple[NextAction, ...]]:
    """The JSON envelope, the ``data`` it carries (for the renderers), and the next actions."""
    actions = next_actions(profile, options, result)
    report = result.report
    data = collect_data(profile, options, result)
    document = envelope(
        result.state,
        data,
        warnings=tuple(result.warnings),
        next_actions=actions,
        provenance={
            "snapshot_id": report.snapshot_id if report else None,
            "generation": report.generation if report else None,
        },
    )
    return document, data, actions


def serve_data(profile: str, state: WorkerState, collect_seconds: int, export_seconds: int) -> dict:
    report = state.last_report
    return {
        "profile": profile,
        "phase": state.phase,
        "pid": state.pid,
        "cycles": state.cycles,
        "last_error": state.last_error,
        "cadence": {"collect_seconds": collect_seconds, "export_seconds": export_seconds},
        "last_report": plain(asdict(report)) if report is not None else None,
    }


def already_running_message(error: AlreadyRunning) -> str:
    info = error.info
    who = f"pid {info.pid}" if info.pid is not None else "unknown pid"
    kind = f", kind {info.kind}" if info.kind else ""
    since = f", since {info.acquired_at}" if info.acquired_at else ""
    return f"A worker already holds this profile's lease ({who}{kind}{since}); nothing was started."
