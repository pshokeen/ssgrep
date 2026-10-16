"""Human renderers for ``savings setup/status/pause/resume`` (width-aware, plain text)."""

from __future__ import annotations

from rich.text import Text

from ssgrep.accounting.ledger.ledger_contracts import ProfilePolicy
from ssgrep.accounting.serve.status_worker import (
    LagStatus,
    MissingEvidence,
    StoragePressure,
    WorkerStatus,
)
from ssgrep.accounting.views.service import StatusView
from ssgrep.accounting.views.types import command_for
from ssgrep.cli.savings_render import privacy_line
from ssgrep.cli.savings_state_render import render_conditions, render_delivery, split_conditions
from ssgrep.cli.savings_text import Doc, fmt_age, fmt_bytes, fmt_int, safe

WATERMARK_ROWS = 8  # human view lists this many watermarks; --json carries all of them

_FIELDS = (
    "journal: times, outcomes, duration, salted argument hashes, bounded result references/ranks",
    "compact native IDs, locators, digests, workspace/lineage provenance and cost measurements",
    "keyed candidate-frequency metadata and frozen revision/scope pointers",
    "receipts, accepted totals, reducer checkpoints, worker progress, export queues "
    "and temporary files",
    "policy, fingerprint/pseudonym keys, source cursors/scope manifests "
    "and delivery/deletion bookkeeping",
)


def _roots(policy: ProfilePolicy) -> str:
    if policy.roots:
        return ", ".join(safe(r) for r in policy.roots)
    if policy.profile_id == "personal-local":
        return "all working directories on this machine (personal-local)"
    return "none approved"


def render_setup_preview(
    policy: ProfilePolicy, *, width: int, already_enabled: bool, source: str
) -> Text:
    doc = Doc(width)
    doc.line("ssgrep savings setup", style="bold")
    doc.line(f"Profile {safe(policy.profile_id)} | policy source: {safe(source)}")
    doc.heading("WILL ENABLE")
    doc.line(
        "Local measurement only, kept in your ssgrep data directory (owner-only files).", indent=2
    )
    doc.heading("APPROVED ROOTS")
    doc.line(_roots(policy), indent=2)
    doc.heading("FIELDS KEPT")
    for field in _FIELDS:
        doc.bullet(field)
    doc.heading("RETENTION")
    doc.pair("Evidence", f"{policy.evidence_retention_days} days", indent=2)
    doc.pair("Receipts", f"{policy.receipt_retention_days} days", indent=2)
    if policy.evidence_cache_enabled:
        doc.pair(
            "Plaintext cache",
            f"explicit opt-in; {policy.evidence_cache_retention_days} days; "
            f"{fmt_int(policy.evidence_cache_max_bytes)} bytes maximum",
            indent=2,
        )
        doc.line("May retain prompt/query/tool arguments and outputs locally.", indent=2)
        doc.command(
            "Revoke:",
            command_for(policy.profile_id, "savings setup --no-evidence-cache --yes"),
        )
    else:
        doc.line(
            "Plaintext cache OFF: no copied prompts, queries, tool arguments or outputs. "
            "Replay reads authorized originals; missing/changed sources limit replay, "
            "not accepted totals.",
            indent=2,
        )
    doc.line(
        "Queues: 7 days / 100 MiB; transient files inherit their data class limits. "
        "Native transcripts and the retrieval index are separate and are never purged here.",
        indent=2,
    )
    doc.line(
        "Obsolete copied evidence is removed checkpoint-first on the next successful collection; "
        "previous accepted compact totals survive missing originals.",
        indent=2,
    )
    doc.heading("NOT DONE")
    doc.line(
        "No export is enabled, nothing is sent, and no background service is installed.", indent=2
    )
    if policy.destinations:
        names = ", ".join(safe(d.destination_id) for d in policy.destinations)
        doc.line(f"Destinations permitted by policy (all still disabled): {names}", indent=2)
    if already_enabled:
        doc.blank()
        doc.line("Measurement is already enabled for this profile; confirming re-saves the policy.")
    return doc.text()


def render_setup_result(policy: ProfilePolicy, width: int) -> Text:
    doc = Doc(width)
    doc.line("Local measurement is enabled.", style="bold")
    doc.line(f"Profile {safe(policy.profile_id)}. Export remains off; nothing is sent.")
    doc.blank()
    doc.command("Next:", command_for(policy.profile_id, "savings collect"))
    doc.command("Or: review state", command_for(policy.profile_id, "savings status"))
    return doc.text()


def _words(token: object) -> str:
    return safe(token).replace("_", " ")


def _bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    exact = f"{fmt_int(value)} B"
    return exact if value < 1024 else f"{exact} ({fmt_bytes(value)})"


def _count(value: int | None) -> str:
    return "unknown" if value is None else fmt_int(value)


def _render_worker(doc: Doc, worker: WorkerStatus) -> None:
    owner = safe(worker.kind) if worker.kind else "worker"
    pid = f", pid {worker.pid}" if worker.pid is not None else ""
    if worker.state == "running":
        doc.pair("Worker", f"running ({owner}{pid})", indent=2)
    elif worker.state == "stale_lease":
        doc.pair(
            "Worker",
            f"stale lease: {owner}{pid} is not running; abandoned, the next collect or serve "
            "takes it over",
            indent=2,
        )
    elif worker.state == "failed_last_cycle":
        error = (
            worker.progress.last_cycle.error
            if worker.progress and worker.progress.last_cycle
            else None
        )
        doc.pair("Worker", f"last cycle failed ({_words(error or 'error_not_recorded')})", indent=2)
    elif worker.state == "idle":
        doc.pair("Worker", "idle (no worker holds the lease; last cycle published)", indent=2)
    else:
        label = "never run" if worker.state == "never_run" else "unknown"
        doc.pair("Worker", f"{label} ({_words(worker.reason or 'not_recorded')})", indent=2)
    progress = worker.progress
    if progress is None:
        return
    if worker.state == "running":
        doc.pair(
            "Phase",
            f"{_words(progress.phase or 'unknown')}, mode {_words(progress.mode or 'unknown')}",
            indent=4,
        )
        done, total = progress.sources_done, progress.sources_total
        doc.pair(
            "Cycle",
            f"{_count(done)} of {_count(total)} sources, {_bytes(progress.bytes_read)} read",
            indent=4,
        )
    if progress.cycles is not None:
        doc.pair("Cycles", fmt_int(progress.cycles), indent=4)
    if progress.updated_at:
        doc.pair("Updated", safe(progress.updated_at), indent=4)


def _render_lag(doc: Doc, lag: LagStatus) -> None:
    doc.heading("LAG")
    if lag.lag_bytes is None:
        doc.pair("Lag", f"unknown ({_words(lag.reason or 'lag_not_recorded')})", indent=2)
        return
    behind = "" if lag.sources_behind is None else f"; {fmt_int(lag.sources_behind)} sources behind"
    doc.pair("Lag", f"{_bytes(lag.lag_bytes)} unread{behind}", indent=2)
    if lag.stopped_by and lag.stopped_by != "complete":
        doc.pair(
            "Last cycle", f"stopped by {_words(lag.stopped_by)}; run again to resume", indent=2
        )
    for row in lag.by_runtime:
        doc.pair(
            safe(row.runtime),
            f"{_bytes(row.lag_bytes)} behind of {_bytes(row.total_bytes)}; "
            f"{_count(row.sources)} sources",
            indent=4,
        )
    if lag.watermark_count is None:
        doc.pair("Watermarks", "unknown (not recorded)", indent=2)
        return
    doc.pair("Watermarks", fmt_int(lag.watermark_count), indent=2)
    for mark in (lag.watermarks or ())[:WATERMARK_ROWS]:
        doc.bullet(
            f"{mark.source} {safe(mark.runtime)} offset "
            f"{fmt_int(mark.offset)}/{fmt_int(mark.size)}, lag {fmt_int(mark.lag_bytes)}, "
            f"tail {fmt_int(mark.pending_tail_bytes)}, {safe(mark.status)}",
            indent=4,
        )
    hidden = lag.watermark_count - min(len(lag.watermarks or ()), WATERMARK_ROWS)
    if hidden > 0:
        doc.line(f"and {fmt_int(hidden)} more (see --json)", indent=4)


def _render_missing(doc: Doc, missing: MissingEvidence) -> None:
    doc.heading("MISSING DATA")
    if missing.missing_sources is None:
        doc.pair(
            "Sources", f"unknown ({_words(missing.reason or 'coverage_not_recorded')})", indent=2
        )
    else:
        doc.pair("Missing sources", fmt_int(missing.missing_sources), indent=2)
    if missing.gaps is None:
        doc.pair("Gaps", "unknown", indent=2)
    elif not missing.gaps:
        doc.pair("Gaps", "none recorded", indent=2)
    for reason, count in missing.gaps or ():
        doc.pair(_words(reason), fmt_int(count), indent=4)
    journal = (
        ("incomplete tails", missing.incomplete_tails),
        ("quarantined", missing.quarantined),
        ("conflicts", missing.conflicts),
    )
    doc.pair("Journal", ", ".join(f"{name} {_count(value)}" for name, value in journal), indent=2)
    expired = missing.evidence_expired_before
    doc.pair("Expired before", safe(expired) if expired else "nothing expired", indent=2)
    doc.pair("Frozen roots", _count(missing.frozen_roots), indent=2)
    doc.pair(
        "Completeness",
        f"{missing.capture_completeness}: a finished snapshot does not prove "
        "everything was captured",
        indent=2,
    )


def _render_storage(doc: Doc, storage: StoragePressure) -> None:
    doc.heading("STORAGE")
    level = _words(storage.level) if storage.level else "unknown"
    detail = f" ({_words(storage.level_reason)})" if storage.level_reason else ""
    doc.pair("Pressure", f"{level}{detail}", indent=2)
    doc.pair("Free space", _bytes(storage.free_bytes), indent=2)
    doc.pair("Accounting data", _bytes(storage.total_bytes), indent=2)
    for part in storage.components:
        shown = _bytes(part.bytes) if part.bytes is not None else f"unknown ({_words(part.reason)})"
        doc.pair(safe(part.name), shown, indent=4)


def _collection_freshness(view: StatusView) -> str:
    if view.collected_at is None:
        return "no completed snapshot yet"
    state = "STALE" if view.stale else "fresh"
    after = f"stale after {view.stale_after_seconds} s"
    return f"{state}, last complete {fmt_age(view.collection_age_seconds)} ({after})"


def _delivery_freshness(view: StatusView) -> str:
    if not view.delivery_lines:
        return "no destination configured"
    if view.delivery_age_seconds is None:
        return "nothing acknowledged yet (acknowledged = accepted by the collector)"
    return f"last collector acceptance {fmt_age(view.delivery_age_seconds)}"


def render_status(
    view: StatusView, width: int, *, source_census: dict[str, object] | None = None
) -> Text:
    doc = Doc(width)
    doc.line("ssgrep savings status", style="bold")
    policy = view.policy
    capture = "PAUSED" if policy.paused else ("enabled" if policy.collection_enabled else "off")
    doc.heading("CAPTURE")
    doc.pair("Capture", capture, indent=2)
    if policy.paused:
        doc.pair("Paused since", safe(policy.paused_at), indent=2)
    doc.pair("Roots", _roots(policy), indent=2)
    doc.pair("Policy", f"v{policy.policy_version} ({safe(policy.policy_source)})", indent=2)
    cache = (
        (
            f"opt-in, {policy.evidence_cache_retention_days}d / "
            f"{fmt_bytes(policy.evidence_cache_max_bytes)}"
        )
        if policy.evidence_cache_enabled
        else "off (compact references only)"
    )
    doc.pair("Plaintext cache", cache, indent=2)
    if view.policy_error:
        doc.pair("Policy problem", safe(view.policy_error), indent=2)
    _render_worker(doc, view.worker)
    doc.heading("SNAPSHOT")
    if view.collected_at is None:
        doc.line("No completed snapshot.", indent=2)
    else:
        freshness = "STALE" if view.stale else "fresh"
        doc.pair("Last complete", safe(view.collected_at), indent=2)
        doc.pair("Freshness", f"{freshness}, {fmt_age(view.age_seconds)}", indent=2)
        doc.pair("Generation", str(view.generation), indent=2)
        doc.pair("Covers through", safe(view.coverage_through), indent=2)
    _render_lag(doc, view.lag)
    _render_missing(doc, view.missing)
    if source_census is not None:
        doc.heading(
            "SOURCE CENSUS", "Latest collected source watermarks; no separate filesystem scan."
        )
        sources = source_census.get("sources")
        doc.pair("Sources", fmt_int(sources) if isinstance(sources, int) else "unknown")
        behind = source_census.get("sources_behind")
        if isinstance(behind, int):
            doc.pair("Sources behind", fmt_int(behind), indent=2)
        by_runtime = source_census.get("by_runtime")
        if isinstance(by_runtime, list):
            for row in by_runtime:
                if isinstance(row, dict):
                    runtime = row.get("runtime")
                    count = row.get("sources")
                    if isinstance(runtime, str):
                        doc.pair(
                            safe(runtime),
                            fmt_int(count) if isinstance(count, int) else "unknown",
                            indent=2,
                        )
    _render_storage(doc, view.storage)
    if view.gaps:
        doc.heading("PAUSED INTERVALS", "Not imported unless you backfill them.")
        for start, end in view.gaps:
            doc.line(f"{safe(start)} to {safe(end) if end else 'now (still paused)'}", indent=2)
    doc.heading("FRESHNESS", "Collection and delivery age independently.")
    doc.pair("Collection", _collection_freshness(view), indent=2)
    doc.pair("Delivery", _delivery_freshness(view), indent=2)
    first, _ = split_conditions(view.conditions)
    primary = view.next_actions[0].command if view.next_actions else None
    render_conditions(doc, first, "STATE", primary=primary)
    if not view.destinations:
        doc.heading("DESTINATIONS")
        doc.line("None configured. " + privacy_line(()), indent=2)
    render_delivery(doc, view.delivery_lines, frozenset(c.code for c in first))
    if view.warnings:
        from ssgrep.accounting.views.diagnostics import grouped_warnings

        for group in grouped_warnings(view.warnings):
            doc.line(f"Note: {safe(group.code)} ({group.count})", indent=0)
    if view.next_actions:
        doc.blank()
        doc.line(safe(view.next_actions[0].why), indent=6)
        doc.command("Next:", view.next_actions[0].command)
    return doc.text()


def render_lifecycle(title: str, lines: tuple[str, ...], next_command: str, width: int) -> Text:
    doc = Doc(width)
    doc.line(title, style="bold")
    for line in lines:
        doc.line(safe(line))
    doc.blank()
    doc.command("Next:", next_command)
    return doc.text()
