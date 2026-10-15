"""Human renderers for ``metrics export`` (width-aware plain text; untrusted text via ``safe``).

Request bytes in a preview are printed verbatim by ``preview_payload_lines``: they are
ASCII-only JSON (control characters are escaped by the serializer), so they are neither wrapped
nor re-encoded and can be compared byte for byte.
"""

from __future__ import annotations

from rich.text import Text

from ssgrep.accounting.export_consent import (
    DisableReport,
    EnablePlan,
    target_command,
)
from ssgrep.accounting.export_consent_preview import BatchView, PreviewView
from ssgrep.cli.savings_text import Doc, fmt_bytes, fmt_int, safe

PREVIEW_PRINT_CAP_BYTES = 64 * 1024  # request bytes printed verbatim; the rest is summarized

_STATE_TEXT = {
    "ok": "Candidate ready: this is exactly what a send would put on the wire.",
    "no_snapshot": "No snapshot exists yet; there is nothing to preview.",
    "ineligible_scope": "The current snapshot predates a scope change and can never be exported; "
    "a newer collection is needed.",
    "no_changes": "Every current record was already exported; nothing new would be sent.",
    "policy_blocked": "Policy does not currently allow this destination/signal.",
    "no_records": "There is nothing to export for this signal yet.",
}


def _retention(plan: EnablePlan) -> tuple[tuple[str, str], ...]:
    r = plan.retention
    return (
        ("Evidence", f"{r['evidence_days']} days"),
        ("Receipts", f"{r['receipt_days']} days"),
        ("Unsent spool", f"{r['spool_days']} days or {fmt_bytes(r['spool_bytes'])} per signal"),
    )


def render_enable_preview(plan: EnablePlan, width: int) -> Text:
    doc = Doc(width)
    doc.line("ssgrep metrics export --enable", style="bold")
    doc.line(f"Profile {safe(plan.profile)} | policy source: {safe(plan.policy_source)}")
    doc.heading("WILL AUTHORIZE")
    doc.pair("Destination", safe(plan.destination))
    doc.pair("Signal", plan.signal.value)
    doc.pair("Endpoint", safe(plan.endpoint))
    if plan.proxy:
        doc.pair("Proxy", safe(plan.proxy))
    doc.pair("Credential", safe(plan.credential_ref) if plan.credential_ref else "none configured")
    doc.line("(the reference name only; the secret is never read or shown here)", indent=2)
    doc.pair("Field schema", safe(plan.fields_version))
    doc.pair("Policy version", fmt_int(plan.policy_version))
    doc.heading("FIELDS THAT LEAVE THIS MACHINE")
    for field in plan.fields:
        doc.bullet(field)
    doc.heading("APPROVED ROOTS")
    doc.line(
        ", ".join(safe(r) for r in plan.roots) if plan.roots else safe(plan.roots_note), indent=2
    )
    doc.heading("RETENTION")
    for label, value in _retention(plan):
        doc.pair(label, value)
    doc.heading("READ BEFORE CONFIRMING")
    for notice in plan.notices:
        doc.bullet(notice)
    if plan.already_enabled:
        doc.blank()
        doc.line("This destination/signal is already enabled; confirming changes nothing.")
    return doc.text()


def render_enable_result(plan: EnablePlan, changed: bool, width: int) -> Text:
    doc = Doc(width)
    verb = "enabled" if changed else "was already enabled"
    doc.line(f"Export {verb}: {safe(plan.destination)}/{plan.signal.value}.", style="bold")
    doc.line("Nothing was sent. Enabling is not a connectivity or delivery claim.")
    if plan.paused:
        doc.line("The profile is paused, so nothing is sent until you resume.")
    doc.blank()
    doc.command("Next:", target_command(plan.profile, plan.destination, plan.signal, "--check"))
    doc.command(
        "Or: inspect the exact data",
        target_command(plan.profile, plan.destination, plan.signal, "--preview"),
    )
    return doc.text()


def render_disable(report: DisableReport, profile: str, width: int) -> Text:
    doc = Doc(width)
    label = f"{safe(report.destination)}/{report.signal.value}"
    doc.line(
        f"Export disabled: {label}." if report.was_enabled else f"Export already off: {label}.",
        style="bold",
    )
    doc.line("Sending stops until you explicitly enable it again. Other signals are untouched.")
    doc.heading("KEPT")
    doc.line("Delivery history, the spool, the exported-record registry and tombstones.", indent=2)
    if report.pending_batches is not None:
        doc.pair("Pending batches", fmt_int(report.pending_batches))
    doc.pair("Resend", "queued batches are kept but never sent while disabled")
    note = report.in_flight_nonrecallable or report.in_flight
    if note:
        doc.line(
            f"A request for batch {safe(note)} was already in flight; it cannot be recalled.",
            indent=2,
        )
    else:
        doc.line("An in-flight request, if any, cannot be recalled.", indent=2)
    doc.blank()
    doc.command("Next:", target_command(profile, report.destination, report.signal, "--enable"))
    return doc.text()


def render_preview_header(view: PreviewView, width: int) -> Text:
    doc = Doc(width)
    doc.line("ssgrep metrics export --preview", style="bold")
    doc.line("Local only: no network request, no credential, no state change.")
    doc.heading("DESTINATION")
    doc.pair("Destination", safe(view.destination))
    doc.pair("Signal", view.signal.value)
    doc.pair("Policy version", fmt_int(view.policy_version))
    if view.endpoint_host:
        doc.pair("Endpoint host", safe(view.endpoint_host))
    doc.pair("Field schema", safe(view.fields_version))
    doc.heading("CANDIDATE")
    doc.pair("State", view.state)
    doc.line(_STATE_TEXT.get(view.state, safe(view.state)), indent=2)
    if view.reason:
        doc.pair("Reason", safe(view.reason).replace("_", " "))
    if view.batches:
        doc.pair("Generation", fmt_int(view.generation or 0))
        doc.pair("Batches", fmt_int(len(view.batches)))
        doc.pair("Items", fmt_int(view.total_items))
        doc.pair("Bytes", f"{fmt_int(view.total_bytes)} ({fmt_bytes(view.total_bytes)})")
        if view.ephemeral_key:
            doc.line(
                "Pseudonyms in this preview use a throwaway key; the first real send mints a "
                "persistent key, so pseudonymous ids will differ.",
                indent=2,
            )
    if view.omitted:
        doc.pair("Omitted", f"{fmt_int(view.omitted)} datum(s) failed the export allowlist")
    return doc.text()


def _summary_line(batch: BatchView) -> str:
    return (
        f"{batch.batch_id}: {fmt_int(batch.bytes)} bytes, {fmt_int(batch.items)} items, "
        f"sha256 {batch.sha256}"
    )


def preview_payload_lines(view: PreviewView, cap: int = PREVIEW_PRINT_CAP_BYTES) -> list[str]:
    """Lines to print after the header: each shown request verbatim, the rest summarized."""
    lines: list[str] = []
    printed = 0
    hidden: list[BatchView] = []
    for batch in view.batches:
        if hidden or (printed and printed + batch.bytes > cap):
            hidden.append(batch)
            continue
        printed += batch.bytes
        lines.append(f"-- request {_summary_line(batch)}")
        lines.append(batch.body.decode("ascii"))
    if hidden:
        lines.append(
            f"-- {fmt_int(len(hidden))} more batch(es) not printed (cap {fmt_bytes(cap)} of "
            "request bytes); use --json for every request:"
        )
        lines.extend(f"-- {_summary_line(b)}" for b in hidden)
    return lines


def render_check(
    destination: str,
    signal: str,
    category: str,
    status: int | None,
    width: int,
) -> Text:
    doc = Doc(width)
    label = f"{safe(destination)}/{safe(signal)}"
    doc.line(f"Connectivity and authentication OK for {label}.", style="bold")
    doc.line("An empty request was accepted; no data was sent or acknowledged.")
    if status is not None:
        doc.line(f"HTTP status: {status} ({safe(category)}). Response bodies are never shown.")
    return doc.text()


def render_send(title: str, lines: tuple[str, ...], width: int) -> Text:
    doc = Doc(width)
    doc.line(title, style="bold")
    for line in lines:
        doc.line(safe(line))
    return doc.text()
