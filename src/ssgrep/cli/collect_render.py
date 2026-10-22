"""Human renderers for ``savings collect`` and ``savings serve`` (width-aware, plain text).

They render command data and startup details; every untrusted string passes through ``safe``.
The startup screen exposes the endpoint token-file path, never its value.
"""

from __future__ import annotations

import shlex
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from rich.text import Text

from ssgrep.accounting.collect.collector import CollectOptions, CollectResult, CycleBudget
from ssgrep.accounting.views.types import NextAction, command_for
from ssgrep.cli.accounting_cli import envelope
from ssgrep.cli.savings_text import Doc, fmt_bytes, fmt_int, safe

if TYPE_CHECKING:
    from ssgrep.accounting.serve.lease import AlreadyRunning, LeaseInfo
    from ssgrep.accounting.serve.serve_loop import WorkerState

SHORT_ID = 12
TEXT_LIMIT = 64
DEFAULT_UNAVAILABLE_REASON = "evidence_expired"
_MAX_DEPTH = 4

_STOPPED = {
    "byte_budget": "byte budget",
    "time_budget": "time budget",
    "backfill_pending": "pending historical backfill",
}


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
    from ssgrep.accounting.serve.lease import inspect_lease

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
    """Build the command that resumes ``options`` without re-declaring ``--new-epoch``."""
    parts = ["savings", "collect"]
    if options.recompute:
        parts.append("--recompute")
    if options.backfill:
        parts.append("--backfill")
    if options.backfill or options.recompute:
        parts.extend(shlex.quote(value) for value in (options.since, options.until) if value)
    if options.repin_frequency:
        parts.append("--repin-frequency")
    if options.reset_frequency_authority:
        parts.append("--reset-frequency-authority")
    if options.budget.seconds != CycleBudget().seconds:
        parts.extend(("--budget-seconds", str(options.budget.seconds)))
    return command_for(profile, " ".join(parts))


def next_actions(
    profile: str, options: CollectOptions, result: CollectResult
) -> tuple[NextAction, ...]:
    if result.state == "already_running":
        return (NextAction(command_for(profile, "savings status"), "See the running worker."),)
    savings = NextAction(command_for(profile, "savings"), "Read the updated snapshot.")
    if result.state == "partial" and result.report is not None:
        report = result.report
        if report.stopped_by in {"backfill_pending", "backfill_blocked"}:
            historical = result.pending_backfill
            command = (
                resume_command(
                    profile,
                    CollectOptions(
                        backfill=True,
                        since=historical[0],
                        until=historical[1],
                        budget=options.budget,
                    ),
                )
                if historical is not None
                else command_for(profile, "savings status")
            )
            if report.stopped_by == "backfill_blocked":
                return (
                    NextAction(
                        command_for(profile, "savings status --sources"),
                        "Resolve missing, unreadable or unsupported sources first; "
                        "an unchanged retry cannot progress.",
                    ),
                    NextAction(command, "Retry the historical interval after source repair."),
                    savings,
                )
            return (NextAction(command, "Resolve the pending historical backfill."), savings)
        if report.next_action == "restore_missing_sources":
            return (
                NextAction(
                    command_for(profile, "savings collect"),
                    "Restore missing original transcripts first. Unchanged retries cannot "
                    "recover unavailable evidence; measured values are retained.",
                ),
                savings,
            )
        if report.next_action == "increase_cycle_budget":
            required = report.required_budget_seconds
            if required is not None:
                larger = replace(options, budget=replace(options.budget, seconds=required))
                return (
                    NextAction(
                        resume_command(profile, larger),
                        "Increase this pass's budget to the observed minimum; "
                        "the required total runtime is still unknown.",
                    ),
                    savings,
                )
            return (
                NextAction(
                    command_for(profile, "savings collect --help"),
                    "Choose a larger --budget-seconds value.",
                ),
                savings,
            )
        if report.resumable:
            return (
                NextAction(
                    resume_command(profile, options),
                    "Resume saved work; completed roots are not repeated.",
                ),
                savings,
            )
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
        "resumable": report.resumable if report else None,
        "next_action": report.next_action if report else None,
        "configured_budget_seconds": report.configured_budget_seconds if report else None,
        "required_budget_seconds": report.required_budget_seconds if report else None,
        "bytes_read": report.bytes_read if report else None,
        "lag_bytes": report.lag_bytes if report else None,
        "sources": report.sources if report else None,
        "new_evidence": report.new_evidence if report else None,
        "elapsed_ms": report.elapsed_ms if report else None,
        "backfill": plain(result.backfill) if result.backfill is not None else None,
        "pending_backfill": (
            {"since": result.pending_backfill[0], "until": result.pending_backfill[1]}
            if result.pending_backfill is not None
            else None
        ),
        "unavailable_periods": _unavailable(result.unavailable_periods),
        "owner": owner_view(profile, result.progress)
        if result.state == "already_running"
        else None,
    }
    return data


def collect_document(
    profile: str, options: CollectOptions, result: CollectResult
) -> tuple[dict[str, object], dict[str, Any], tuple[NextAction, ...]]:
    """The JSON envelope, the ``data`` it carries, and the next actions."""
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


def _count(value: object) -> str:
    return "unknown" if not isinstance(value, int) else fmt_int(value)


def _size(value: object) -> str:
    return "unknown" if not isinstance(value, int) else fmt_bytes(value)


def _next(doc: Doc, actions: tuple[NextAction, ...]) -> None:
    doc.blank()
    for number, action in enumerate(actions):
        doc.command("Next:" if number == 0 else "Or:", action.command)
        doc.line(safe(action.why), indent=6)


def _backfill(doc: Doc, data: dict[str, Any]) -> None:
    doc.heading(
        "HISTORICAL BACKFILL (not live executions)",
        "Imported history is kept as its own series; it never counts as recent live activity.",
    )
    interval = data["interval"]
    doc.pair("Interval", f"{safe(interval['since'])} to {safe(interval['until'])} (UTC)")
    series = data.get("backfill")
    if not isinstance(series, dict):
        doc.line("No backfill series was recorded by this run.", indent=2)
        return
    if "complete" in series:
        doc.pair("Complete", "yes" if series["complete"] is True else "no; see the next action")
    counts = series.get("counts")
    if isinstance(counts, list):
        for pair in counts:
            if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[1], int):
                doc.pair(safe(pair[0]), fmt_int(pair[1]), indent=4)


def _recompute(doc: Doc, data: dict[str, Any]) -> None:
    periods = data["unavailable_periods"]
    doc.heading("RECOMPUTE", "Uses retained frozen evidence only; no index is rebuilt.")
    if not periods:
        doc.line("Every requested period could be recomputed.", indent=2)
        return
    doc.line(f"{len(periods)} period(s) unavailable; their saved totals are kept:", indent=2)
    for period in periods:
        reason = safe(period["reason"]).replace("_", " ")
        doc.bullet(f"{safe(period['period'])}: {reason}", indent=4)


def render_collect(
    data: dict[str, Any], warnings: tuple[str, ...], actions: tuple[NextAction, ...], width: int
) -> Text:
    doc = Doc(width)
    state = data["state"]
    title = {
        "published": "Collection complete.",
        "partial": "Collection is partial; current measurements are retained.",
    }.get(state, "Collection state")
    doc.line(title, style="bold")
    doc.pair("Profile", safe(data["profile"]))
    doc.pair("Mode", safe(data["mode"]))
    doc.pair("Counter epoch", _count(data["epoch"]))
    doc.pair("Generation", _count(data["generation"]))
    snapshot = data["snapshot_id"]
    doc.pair("Snapshot", safe(snapshot)[:SHORT_ID] if snapshot else "none published")
    doc.pair("Bytes read", _size(data["bytes_read"]))
    doc.pair("Lag remaining", _size(data["lag_bytes"]))
    if state == "partial":
        reason = _STOPPED.get(str(data["stopped_by"]), safe(data["stopped_by"]))
        doc.blank()
        suffix = "resume saved work" if data.get("resumable") else "intervention is required"
        doc.line(f"Stopped by {reason} -- {suffix}.", style="bold")
    if data["new_epoch"]:
        doc.blank()
        doc.line(
            f"New counter epoch {_count(data['epoch'])} declared: live counters restart at zero "
            "for future events; ledger and history totals are kept."
        )
    if data["mode"] == "backfill":
        _backfill(doc, data)
    if data["mode"] == "recompute":
        _recompute(doc, data)
    for warning in warnings:
        doc.line(f"Note: {safe(warning)}")
    _next(doc, actions)
    return doc.text()


def render_running(data: dict[str, Any], actions: tuple[NextAction, ...], width: int) -> Text:
    doc = Doc(width)
    owner = data["owner"]
    doc.line("Another worker is already collecting; nothing was started.", style="bold")
    doc.line("This is normal: its snapshot is not touched and no competing generation was made.")
    doc.heading("RUNNING WORKER")
    pid = owner["pid"]
    doc.pair("Process", f"pid {pid}" if pid is not None else "unknown pid")
    doc.pair("Kind", safe(owner["kind"]) if owner["kind"] else "unknown")
    doc.pair("Phase", safe(owner["phase"]) if owner["phase"] else "unknown (no progress file)")
    done, total = owner["sources_done"], owner["sources_total"]
    sources = "unknown" if done is None or total is None else f"{fmt_int(done)} of {fmt_int(total)}"
    doc.pair("Sources", sources)
    doc.pair("Bytes read", _size(owner["bytes_read"]))
    doc.pair("Lag remaining", _size(owner["lag_bytes"]))
    if owner["updated_at"]:
        doc.pair("Updated", safe(owner["updated_at"]))
    _next(doc, actions)
    return doc.text()


def render_serve_start(
    profile: str,
    url: str,
    port: int,
    token_file: str,
    collect_seconds: int,
    export_seconds: int,
    pid: int,
    width: int,
) -> Text:
    doc = Doc(width)
    doc.line(f"ssgrep savings serve (pid {pid})", style="bold")
    doc.line(f"URL: {safe(url)}")
    doc.line(f"Port: {port}")
    doc.line(f"Profile: {safe(profile)}")
    doc.line(f"Authentication token file: {safe(token_file)}")
    doc.line(f"Collecting every {collect_seconds} s, export check every {export_seconds} s.")
    doc.line("Foreground only: stop with Ctrl-C or SIGTERM.")
    return doc.text()


def render_serve_final(data: dict[str, Any], width: int) -> Text:
    doc = Doc(width)
    doc.line("Worker stopped.", style="bold")
    doc.line(f"Loopback server stopped at {safe(data['url'])}.")
    doc.line(f"Port: {_count(data['port'])}")
    doc.line(f"Authentication token file: {safe(data['token_file'])}")
    doc.pair("Profile", safe(data["profile"]))
    doc.pair("State", safe(data["phase"]))
    doc.pair("Cycles run", _count(data["cycles"]))
    report = data["last_report"]
    if report:
        generation = report.get("generation")
        doc.pair("Last cycle", f"{safe(report.get('status'))}, generation {_count(generation)}")
    if data["last_error"]:
        doc.pair("Last error", safe(data["last_error"]))
    doc.blank()
    doc.line("The worker lease is released.")
    return doc.text()
