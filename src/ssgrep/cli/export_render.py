"""Human renderers for ``savings export`` (width-aware plain text; untrusted text via ``safe``).

Request bytes in a preview are printed verbatim by ``preview_payload_lines``: they are
ASCII-only JSON (control characters are escaped by the serializer), so they are neither wrapped
nor re-encoded and can be compared byte for byte.
"""

from __future__ import annotations

from rich.text import Text

from ssgrep.accounting.export.consent import (
    BatchView,
    EnablePlan,
    PreviewView,
)
from ssgrep.cli.savings_text import Doc, fmt_bytes, fmt_int, safe

PREVIEW_PRINT_CAP_BYTES = 64 * 1024  # request bytes printed verbatim; the rest is summarized

_STATE_TEXT = {
    "ok": "Candidate ready; sending requires current approval and an unpaused profile.",
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
    doc.line("savings setup --export-to", style="bold")
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


def render_preview_header(view: PreviewView, width: int) -> Text:
    doc = Doc(width)
    doc.line("savings export --dry-run", style="bold")
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
        if hidden or printed + batch.bytes > cap:
            hidden.append(batch)
            continue
        printed += batch.bytes
        lines.append(f"-- request {_summary_line(batch)}")
        lines.append(batch.body.decode("ascii"))
    if hidden:
        lines.append(
            f"-- {fmt_int(len(hidden))} more batch(es) not printed (cap {fmt_bytes(cap)} of "
            "request bytes); use root `ssgrep --json savings export DEST --dry-run` "
            "for all requests:"
        )
        lines.extend(f"-- {_summary_line(b)}" for b in hidden)
    return lines
