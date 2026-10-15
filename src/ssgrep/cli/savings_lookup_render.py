"""Bounded per-lookup evidence display, never causal usefulness or productivity."""

from __future__ import annotations

import shlex

from ssgrep.accounting.service_types import command_for
from ssgrep.cli.savings_text import Doc, fmt_quantity, safe


def render_lookups(view, width=80):
    doc = Doc(width)
    doc.heading("LOOKUP EVIDENCE", "Association is not proof of causal usefulness.")
    doc.pair("Profile", safe(view.profile), indent=0)
    if view.snapshot_id is not None:
        doc.pair("Snapshot", f"{safe(view.snapshot_id)} / generation {view.generation}", indent=0)
    if view.reason:
        doc.line("Unavailable: " + safe(view.reason), indent=0)
        doc.line("Next: " + command_for(view.profile, "metrics status"), indent=0)
        return doc.text()
    doc.line(f"Showing {len(view.rows)} of {view.total} accepted lookup rows", indent=0)
    for row in view.rows:
        doc.heading(safe(row.lookup_id))
        doc.pair(
            "When / runtime",
            f"{safe(row.ts_utc or 'time unknown')} / {safe(row.runtime or 'unknown')}",
            indent=0,
        )
        doc.pair("Execution / outcome", f"{safe(row.execution)} / {safe(row.outcome)}", indent=0)
        doc.pair(
            "Exposure / attribution", f"{safe(row.exposure)} / {safe(row.attribution)}", indent=0
        )
        doc.pair("Reuse / eligibility", f"{safe(row.reuse)} / {safe(row.eligibility)}", indent=0)
        doc.pair("Use evidence", safe(row.useful_outcome), indent=0)
        doc.pair("Actor / basis", f"{safe(row.actor)} / {safe(row.actor_basis)}", indent=0)
        doc.pair("Local scope basis", safe(row.scope_basis), indent=0)
        if row.ambiguous_execution_ids:
            doc.line(
                f"Possible execution links: {len(row.ambiguous_execution_ids)} "
                "(not confirmed ownership)",
                indent=0,
            )
        if row.join_rejection_reasons:
            doc.line(
                "Join limits: " + ", ".join(safe(r) for r in row.join_rejection_reasons), indent=0
            )
        doc.pair("Observed visible bytes", fmt_quantity(row.visible_bytes), indent=0)
        doc.pair("Measured service latency", fmt_quantity(row.measured_service_ms), indent=0)
        for name, quantity in row.known_costs:
            doc.pair("Known cost " + safe(name), fmt_quantity(quantity), indent=0)
        if row.missing_evidence:
            doc.line(
                "Missing evidence: " + ", ".join(safe(r) for r in row.missing_evidence), indent=0
            )
        for receipt in row.receipt_ids:
            doc.line(
                "Explain: " + command_for(view.profile, "savings explain " + safe(receipt)),
                indent=0,
            )
    for warning in view.warnings:
        doc.line("Warning: " + safe(warning), indent=0)
    if view.next_cursor:
        arguments = ["savings", "lookups", "--limit", str(view.limit), "--cursor", view.next_cursor]
        for name in ("since", "until", "project", "runtime", "session", "task"):
            value = getattr(view.scope, name)
            if value is not None:
                arguments.extend(("--" + name, value))
        if view.scope.include_children:
            arguments.append("--include-children")
        continuation = command_for(view.profile, " ".join(shlex.quote(a) for a in arguments))
        doc.command("Next:", continuation)
    return doc.text()
