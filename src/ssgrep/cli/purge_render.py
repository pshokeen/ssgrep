"""Human and JSON views of a ``PurgeReport`` (width-aware; every string through ``safe``).

The four tiers are always printed in the same order and never merged: deleted here, queued
deletion requests, accepted by the collector (always 0 at purge time), backend erasure (always
the operator's, never claimed).
"""

from __future__ import annotations

from rich.text import Text

from ssgrep.accounting.purge_model import BACKEND_ERASED, PurgeReport
from ssgrep.accounting.service_types import NextAction, command_for
from ssgrep.cli.accounting_cli import envelope
from ssgrep.cli.savings_text import Doc, fmt_int, safe

_LOCAL_LABELS = (
    ("evidence_records", "evidence records"),
    ("evidence_segments_rewritten", "evidence segments rewritten"),
    ("ledger_items", "ledger evidence items"),
    ("ledger_period_rows", "ledger period rows"),
    ("ledger_receipt_rows", "ledger receipt rows"),
    ("executions", "retained executions"),
    ("invocations", "retained lookup verdicts"),
    ("sealed_days", "sealed days"),
    ("backfill_series", "backfill series"),
    ("receipt_files", "receipt files"),
    ("snapshot_generations", "snapshot generations"),
    ("journal_segments", "journal segments"),
    ("spool_batches", "unsent spool batches"),
    ("delivery_pins_released", "acknowledged snapshot pins released"),
    ("plaintext_cache_files", "plaintext cache files"),
    ("frequency_files", "active/superseded frequency files"),
    ("temporary_files", "temporary derived files"),
    ("keys", "profile fingerprint keys"),
    ("derived_caches", "rebuildable accounting/rarity-maintenance caches"),
    ("archive_metadata_records", "original archive metadata records"),
)
_RETAINED_LABELS = (
    ("unknown_scope_conversations", "conversations with unknown workspace"),
    ("mixed_scope_conversations", "conversations also working outside the workspace"),
    ("unknown_scope_executions", "executions with unknown workspace"),
    ("sealed_days_not_splittable", "sealed days (aggregates cannot be split by workspace)"),
    ("backfill_series_not_splittable", "backfill series (cannot be split by workspace)"),
    ("journal_segments_retained", "journal segments holding newer events"),
    ("in_flight_batches_not_recallable", "in-flight batches that cannot be recalled"),
)
_NOTE_TEXT = {
    "source_transcripts_untouched": "Source transcripts were not touched.",
    "search_index_untouched": "The search index was not touched.",
    "other_profiles_untouched": "Other profiles were not touched.",
    "live_counters_retained": "Live counters hold counts only and were kept.",
    "source_read_offsets_retained": "Source read positions were kept.",
    "shared_frequency_authority_invalidated_requires_explicit_rebuild": (
        "Shared rarity vocabulary, when erased, is invalidated; other workspace totals remain. "
        "Fresh rarity authority requires an explicitly authorized scope rebuild."
    ),
}


def _tombstone_status(status: str, reason: str | None) -> str:
    base = {
        "queued": "queued for the next export cycle",
        "would_queue": "would be queued for the next export cycle",
        "not_permitted": "NOT queued: sending is not permitted",
        "skipped": "NOT queued",
    }.get(status, status)
    return f"{base} ({safe(reason).replace('_', ' ')})" if reason else base


def _title(report: PurgeReport) -> str:
    if report.empty:
        return "Nothing to purge." if report.performed else "Nothing would be purged."
    return (
        "Purge complete." if report.performed else "Purge preview (dry run). Nothing was changed."
    )


def render_report(report: PurgeReport, width: int) -> Text:
    doc = Doc(width)
    verb = "Deleted" if report.performed else "Would delete"
    doc.line(_title(report), style="bold")
    scope = f"data dated before {safe(report.cutoff_day)} (UTC)"
    if report.workspace:
        scope += f" in workspace {safe(report.workspace)}"
    doc.pair("Profile", safe(report.profile))
    doc.pair("Scope", scope)
    doc.heading("1. LOCAL DELETION", f"{verb} on this machine, in accounting data only.")
    shown = [(label, getattr(report.local_deleted, key)) for key, label in _LOCAL_LABELS]
    for label, value in shown:
        if value:
            doc.pair(label, fmt_int(value), indent=4)
    if not any(value for _, value in shown):
        doc.line("nothing to delete locally", indent=4)
    doc.heading("2. DELETION REQUESTS QUEUED", "Delivered by the next export cycle, not now.")
    for line in report.tombstones:
        doc.bullet(
            f"{safe(line.destination)}: {fmt_int(line.records)} record(s), "
            f"{_tombstone_status(line.status, line.reason)}",
            indent=4,
        )
    if not report.tombstones:
        doc.line("none", indent=4)
    else:
        doc.line(
            f"Unsent deletion requests are kept at most {report.pending_retention_days} days.",
            indent=4,
        )
    doc.heading("3. ACCEPTED BY THE COLLECTOR")
    doc.line("0 now. Acceptance is only known after a later delivery cycle.", indent=4)
    doc.heading("4. ERASED FROM THE BACKEND")
    doc.line(f"{BACKEND_ERASED.replace('_', ' ')}: ssgrep cannot erase remote data.", indent=4)
    for owed in report.obligations:
        doc.bullet(
            f"{safe(owed.destination)}: {fmt_int(owed.records)} exported record(s) the operator "
            f"must erase ({safe(owed.reason).replace('_', ' ')}); ids are in --json",
            indent=4,
        )
    for destination in report.non_reversible_destinations:
        doc.bullet(
            f"{safe(destination)}: aggregate metrics already accepted cannot be recalled",
            indent=4,
        )
    kept = [(label, getattr(report.retained, key)) for key, label in _RETAINED_LABELS]
    doc.heading("KEPT")
    for label, value in kept:
        if value:
            doc.pair(label, fmt_int(value), indent=4)
    for note in report.notes:
        doc.line(_NOTE_TEXT.get(note, safe(note)), indent=4)
    if report.worker is not None and report.worker.running:
        doc.heading("WORKER")
        doc.line(
            f"A {safe(report.worker.kind or 'worker')} is running; a real purge needs it stopped.",
            indent=4,
        )
    return doc.text()


def _warnings(report: PurgeReport) -> tuple[str, ...]:
    found = ["backend_erasure_is_operator_responsibility"]
    if report.obligations:
        found.append("operator_must_erase_exported_records")
    if report.non_reversible_destinations:
        found.append("aggregate_metrics_cannot_be_recalled")
    if report.worker is not None and report.worker.running:
        found.append("worker_running")
    return tuple(found)


def report_document(report: PurgeReport, command_profile: str) -> dict[str, object]:
    actions = (
        NextAction(command_for(command_profile, "metrics status"), "Check export and retention."),
        NextAction(command_for(command_profile, "metrics collect"), "Republish the snapshot."),
    )
    return envelope(
        report.state,
        report.to_dict(),
        warnings=_warnings(report),
        next_actions=actions,
        provenance={"command": "metrics purge", "performed": report.performed},
    )
