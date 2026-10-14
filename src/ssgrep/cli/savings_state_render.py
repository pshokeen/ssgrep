"""Renderer blocks for named states, ``--by`` groups and continuation commands.

Every untrusted string goes through ``safe`` and ends up in a ``Doc``; nothing is parsed as
markup. A state is spelled in words (never color alone) and always names its cause and the
real command that addresses it, with ``--profile`` preserved by the facade.
"""

from __future__ import annotations

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.service_types import (
    Condition,
    DeliveryLine,
    GroupRow,
    Scope,
    SummaryView,
    command_for,
)
from ssgrep.cli.savings_text import Doc, display_unit, fmt_int, safe

TITLES = {
    "disabled": "Measurement off",
    "enabled_no_snapshot": "No snapshot yet",
    "corrupt": "Snapshot unreadable",
    "unsupported_schema": "Snapshot from another version",
    "paused": "Paused",
    "stale": "Stale",
    "no_reuse": "No traceable reuse",
    "partial": "Partial evidence",
    "negative": "Negative net",
    "index_unavailable": "Index unavailable",
    "evidence_expired": "Evidence expired",
    "credentials_expired": "Credentials expired",
    "blocked": "Export blocked",
    "partial_export": "Partial export",
    "unknown": "Delivery outcome unknown",
    "offline": "Export offline",
    "unreadable": "Delivery state unreadable",
    "sending_paused": "Sending paused",
}


def _sentences(text: str) -> list[str]:
    """Split a cause into sentences so each starts a line (no sentence breaks mid-phrase)."""
    parts = []
    for piece in text.replace(". ", ".\n").split("\n"):
        if piece.strip():
            parts.append(piece.strip())
    return parts


#: States that decide whether the numbers below can be trusted or acted on: shown FIRST.
BLOCKING = frozenset(
    {
        "disabled",
        "enabled_no_snapshot",
        "corrupt",
        "unsupported_schema",
        "paused",
        "stale",
        "sending_paused",
        "credentials_expired",
        "blocked",
        "partial_export",
        "unknown",
        "offline",
        "unreadable",
    }
)


def split_conditions(
    conditions: tuple[Condition, ...],
) -> tuple[tuple[Condition, ...], tuple[Condition, ...]]:
    """(states that come before the numbers, caveats that qualify them)."""
    first = tuple(c for c in conditions if c.code in BLOCKING)
    return first, tuple(c for c in conditions if c.code not in BLOCKING)


def render_conditions(
    doc: Doc,
    conditions: tuple[Condition, ...],
    heading: str = "STATE",
    note: str = "",
    primary: str | None = None,
) -> None:
    """One block per condition; a ``Do:`` equal to the closing ``Next:`` is not repeated."""
    if not conditions:
        return
    doc.heading(heading, note)
    for condition in conditions:
        title = TITLES.get(condition.code, safe(condition.code))
        doc.line(title, indent=2, style="bold")
        for sentence in _sentences(condition.cause):
            doc.line(safe(sentence), indent=4)
        if condition.action is not None and condition.action.command != primary:
            doc.command("Do:", condition.action.command, indent=4)


def render_delivery(
    doc: Doc, lines: tuple[DeliveryLine, ...], explained: frozenset[str] = frozenset()
) -> None:
    """One block per destination/signal; a cause already spelled out in STATE is not repeated."""
    if not lines:
        return
    doc.heading(
        "DELIVERY", "Acknowledged = accepted by the collector; local collection is separate."
    )
    for line in lines:
        doc.line(f"{safe(line.destination_id)}/{safe(line.signal)}", indent=2)
        doc.pair("State", line.state.replace("_", " "), indent=4)
        if line.state in explained:
            doc.line("Cause and next step: see STATE above.", indent=4)
        else:
            doc.line(safe(line.cause), indent=4)
        if line.pending_batches or line.rejected_points or line.unknown_batches:
            doc.pair(
                "Counts",
                f"pending {fmt_int(line.pending_batches)}, rejected points "
                f"{fmt_int(line.rejected_points)}, unknown {fmt_int(line.unknown_batches)}",
                indent=4,
            )
        if line.next_retry_utc:
            doc.pair("Next retry", safe(line.next_retry_utc), indent=4)
        if line.last_acked_at:
            doc.pair("Last accepted", safe(line.last_acked_at), indent=4)


def scope_command(profile: str, scope: Scope, limit: int, cursor: str | None) -> str:
    """The exact ``ssgrep savings`` command that reproduces this selection (and its next page)."""
    parts = ["savings"]
    for flag, value in (
        ("--since", scope.since),
        ("--until", scope.until),
        ("--project", scope.project),
        ("--runtime", scope.runtime),
        ("--session", scope.session),
        ("--task", scope.task),
        ("--by", scope.by),
    ):
        if value:
            parts.append(f"{flag} {_quote(value)}")
    if scope.include_children:
        parts.append("--include-children")
    if limit != 20:
        parts.append(f"--limit {limit}")
    if cursor:
        parts.append(f"--cursor {cursor}")
    return command_for(profile, " ".join(parts))


def _quote(value: str) -> str:
    """Shell-safe single token for a pasteable command (controls are escaped first)."""
    text = safe(value)
    if text and all(c.isalnum() or c in "-_./:+@=,#" for c in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def _net_triplet(cells: tuple[tuple[object, Quantity], ...]) -> str:
    values = ["unavailable" if q.value is None else fmt_int(q.value, signed=True) for _, q in cells]
    unit = next((display_unit(q.unit) for _, q in cells if q.value is not None), "")
    return " / ".join(values) + (f" {unit}" if unit else "")


def _group(doc: Doc, row: GroupRow) -> None:
    doc.line(safe(row.key), indent=2)
    counts = {name: q for name, q in row.measured}
    parts = [
        f"{label} {fmt_int(counts[name].value or 0)}"
        for name, label in (("searches", "searches"), ("shows", "shows"), ("failed", "failed"))
        if name in counts and counts[name].value is not None
    ]
    if parts:
        doc.line(", ".join(parts), indent=4)
    for scenario_row in row.scenarios:
        label = "context tokens" if scenario_row.unit_key == "context_tokens" else "agent work"
        doc.line(
            f"Net {label} (conservative / central / optimistic): {_net_triplet(scenario_row.net)}",
            indent=4,
        )
        reasons = sorted({q.reason for _, q in scenario_row.net if q.value is None})
        if reasons:
            doc.line("Unknown, not zero: " + ", ".join(safe(r) for r in reasons), indent=6)
    if row.receipts:
        doc.line(f"{fmt_int(row.receipts)} receipt(s) in this group", indent=4)


def render_groups(doc: Doc, view: SummaryView) -> None:
    if view.scope.by is None:
        return
    doc.heading(
        f"BY {view.scope.by.upper()}",
        "Rows are estimates per group; groups are not added to the totals above.",
    )
    if not view.groups:
        doc.line("No rows match this selection.", indent=2)
    for row in view.groups:
        _group(doc, row)
    total = view.group_total or 0
    if total:
        doc.line(f"{fmt_int(len(view.groups))} of {fmt_int(total)} rows on this page.", indent=2)
    if view.next_cursor:
        doc.command(
            "More:",
            scope_command(view.profile, view.scope, view.limit, view.next_cursor),
            indent=2,
        )
