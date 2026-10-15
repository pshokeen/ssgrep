"""Human renderer for ``savings`` (summary, explain, history).

Pure functions return ``rich.text.Text`` for a given width so the actual
renderer can be exercised at 40/80/120 columns; ``emit`` prints it. Section
order is fixed: header, status, MEASURED, ESTIMATED, EVIDENCE, privacy, one
Next action. Unknown quantities say ``unavailable (reason)``; signs are part
of the text so no meaning depends on color.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.text import Text

from ssgrep.accounting.contracts import Quantity, Scenario
from ssgrep.accounting.ledger_contracts import SnapshotState
from ssgrep.accounting.service_types import (
    MEASURED_ORDER,
    HistoryView,
    NextAction,
    ScenarioRow,
    SummaryView,
    command_for,
)
from ssgrep.cli.savings_receipt_render import render_receipt
from ssgrep.cli.savings_state_render import (
    render_compact_conditions,
    render_conditions,
    render_delivery,
    render_groups,
    render_warning_summary,
    split_conditions,
)
from ssgrep.cli.savings_text import (
    Doc,
    display_unit,
    fmt_age,
    fmt_bytes,
    fmt_estimate_net,
    fmt_int,
    fmt_quantity,
    render_estimate_bounds,
    render_estimate_details,
    safe,
)

_MEASURED_LABELS = {
    "searches": "Searches",
    "shows": "Shows",
    "failed": "Failed",
    "visible_bytes": "Visible data",
}
_UNIT_LABELS = {"context_tokens": "Context tokens", "work_seconds": "Agent work"}
_SCENARIO_LABELS = {
    Scenario.CONSERVATIVE: "Conservative",
    Scenario.CENTRAL: "Central",
    Scenario.OPTIMISTIC: "Optimistic",
}
_STATE_HEADLINE = {
    SnapshotState.DISABLED: "Measurement is off",
    SnapshotState.NO_SNAPSHOT: "Enabled, no snapshot yet",
    SnapshotState.FRESH: "Fresh",
    SnapshotState.STALE: "STALE",
    SnapshotState.PAUSED: "PAUSED",
    SnapshotState.CORRUPT: "Snapshot unreadable",
    SnapshotState.UNSUPPORTED_SCHEMA: "Snapshot from another version",
}


def emit(text: Text) -> None:
    """Print plain output on pipes; ``NO_COLOR`` also drops bold/dim on terminals.

    Layout is already complete. A narrower detected terminal must not inject newlines
    into shell-quoted arguments or detach their continuation backslashes.

    Markup is never parsed, so untrusted text cannot style or control output.
    """
    if os.environ.get("NO_COLOR"):
        text = Text(text.plain)
    Console(force_terminal=sys.stdout.isatty(), highlight=False, emoji=False, markup=False).print(
        text, soft_wrap=True
    )
    sys.stdout.flush()


def privacy_line(exports: tuple[tuple[str, str], ...], managed_note: str = "") -> str:
    if not exports:
        return "Local only. Nothing is sent anywhere; export is off." + managed_note
    names = ", ".join(f"{safe(d)}/{safe(s)}" for d, s in exports)
    return f"Export enabled for {names}. Local files stay on this machine." + managed_note


def _scope_text(view: SummaryView) -> str:
    scope = view.scope
    parts = []
    for label, value in (
        ("project", scope.project),
        ("runtime", scope.runtime),
        ("session", scope.session),
        ("task", scope.task),
    ):
        parts.append(f"{label} {safe(value)}" if value else f"all {label}s")
    return ", ".join(parts)


def _interval_text(view: SummaryView) -> str:
    scope = view.scope
    if scope.since or scope.until:
        return f"{scope.since or 'beginning'} to {scope.until or 'now'} (UTC, end exclusive)"
    return "Last 7 days (UTC)"


def _freshness(view: SummaryView) -> str:
    return f"collected {fmt_age(view.age_seconds)} ({safe(view.collected_at)})"


def _words(reason: str, fallback: str) -> str:
    return fallback if reason.endswith("_for_some_executions") else safe(reason).replace("_", " ")


_SHOWN_EXTRA = ("notes",)
_TABLE_MIN_WIDTH = 78  # three scenario columns need 20 + 3 x 19 characters


def _measured(doc: Doc, view: SummaryView) -> None:
    doc.heading("MEASURED", "Counted from your records; not modeled.")
    values = dict(view.measured)
    for name in (*MEASURED_ORDER, *(n for n in _SHOWN_EXTRA if n in values)):
        quantity = values.get(name)
        if quantity is None:
            continue
        label = _MEASURED_LABELS.get(name, safe(name).capitalize())
        text = (
            str(quantity.value)
            if quantity.value is not None and quantity.unit == "count"
            else fmt_quantity(quantity)
        )
        known = values.get(f"{name}_known")
        if quantity.value is None and known is not None and known.value is not None:
            text = (
                f"unavailable: {fmt_quantity(known)} known, "
                f"but {_words(quantity.reason, 'some executions lack it')}"
            )
        doc.pair(label, text)


def _incomplete(quantity: Quantity) -> bool:
    return quantity.value is None and quantity.reason.startswith("incomplete:")


def _scenario_table(doc: Doc, rows: tuple[ScenarioRow, ...]) -> None:
    for row in rows:
        label = _UNIT_LABELS.get(row.unit_key, safe(row.unit_key))
        estimates = {estimate.scenario: estimate for estimate in row.estimates}
        marks = {
            s: not estimates[s].complete if s in estimates else _incomplete(q) for s, q in row.net
        }
        shared_debit = (
            doc.compact and bool(row.debit) and all(q == row.debit[0][1] for _, q in row.debit)
        )
        if doc.narrow or doc.width < _TABLE_MIN_WIDTH:
            doc.line(f"{label} (estimated)", indent=2)
            for (scenario, net), (_, credit), (_, debit) in zip(
                row.net, row.credit, row.debit, strict=True
            ):
                star = "*" if marks[scenario] else ""
                shown = fmt_estimate_net(net, estimates.get(scenario))
                doc.line(
                    f"{_SCENARIO_LABELS[scenario]}: {'' if marks[scenario] else 'net '}{shown}",
                    indent=4,
                )
                doc.line(
                    f"credit {fmt_quantity(credit, signed=True)}{star}"
                    + ("" if shared_debit else f"; cost {fmt_quantity(debit)}{star}"),
                    indent=6,
                )
            if shared_debit:
                doc.line(f"All scenarios cost {fmt_quantity(row.debit[0][1])}", indent=4)
            render_estimate_bounds(doc, row.estimates)
            continue
        width = max((doc.width - 20) // 3, 14)
        header = " " * 20 + "".join(f"{_SCENARIO_LABELS[s]:>{width}}" for s, _ in row.net)
        if doc.compact:
            doc.line(f"{label} ({' / '.join(_SCENARIO_LABELS[s] for s, _ in row.net)})", indent=2)
        else:
            doc.line(f"{label} (estimated)", indent=2)
            doc.line(header.rstrip(), style="dim")
        for name, cells, signed in (
            ("Net", row.net, True),
            ("  credit", row.credit, True),
            ("  cost", row.debit, False),
        ):
            values = "".join(
                format(
                    (
                        fmt_estimate_net(q, estimates.get(s))
                        if name == "Net"
                        else _cell(q, signed) + ("*" if marks[s] else "")
                    ),
                    f">{width}",
                )
                for s, q in cells
            )
            doc.line(
                f"    {('Net/subtotal*' if name == 'Net' and any(marks.values()) else name):<16}"
                f"{values}"
            )
        render_estimate_bounds(doc, row.estimates)


def _cell(quantity: Quantity, signed: bool) -> str:
    if quantity.value is None:
        return "unavailable"
    return f"{fmt_int(quantity.value, signed=signed)} {display_unit(quantity.unit)}"


def _unavailable_reasons(rows: tuple[ScenarioRow, ...]) -> list[str]:
    reasons = {
        q.reason
        for row in rows
        for group in (row.credit, row.debit, row.net)
        for _, q in group
        if q.value is None
    }
    return sorted(reasons)


def _estimated(doc: Doc, view: SummaryView) -> None:
    doc.heading("ESTIMATED NET BENEFIT", "Modeled scenarios, not causal proof.")
    _scenario_table(doc, view.scenarios)
    doc.line("Net = credit minus cost. Minus sign = costs exceed credit.", indent=2, style="dim")
    reasons = _unavailable_reasons(view.scenarios)
    if any(not estimate.complete for row in view.scenarios for estimate in row.estimates) or any(
        _incomplete(q) for row in view.scenarios for _, q in row.net
    ):
        doc.line(
            "* Known net subtotal; credit/cost known parts only. Complete net unavailable.",
            indent=2,
        )
    if reasons:
        doc.line("Unknown, not zero: " + ", ".join(safe(r) for r in reasons), indent=2)


def _count(value: int | None) -> str:
    return "unknown" if value is None else fmt_int(value)


def _evidence(doc: Doc, view: SummaryView) -> None:
    doc.heading("EVIDENCE")
    cov = view.coverage
    if view.partial:
        doc.line("Partial: estimates cover only the subsets named below.", indent=2)
    doc.pair(
        "Attributed",
        f"{_count(cov.get('attributed_lookups'))}/{_count(cov.get('recorded_lookups'))} "
        "recorded lookups",
    )
    doc.pair(
        "Source costs",
        f"known for {_count(cov.get('source_costs_available'))}/"
        f"{_count(cov.get('candidate_receipts'))} candidate reuse receipts",
    )
    gaps = f"{_count(cov.get('ambiguous_joins'))} ambiguous joins; "
    gaps += f"{_count(cov.get('unavailable_sources'))} source records unavailable"
    doc.pair("Gaps", gaps)
    if view.exclusions:
        listed = "; ".join(f"{fmt_int(count)} {safe(reason)}" for reason, count in view.exclusions)
        doc.pair("Excluded", listed)
    doc.pair(
        "Snapshot",
        f"{safe(view.snapshot_id)}, generation {view.generation}, "
        f"methodology {safe(view.methodology_id)}",
    )


def _footer(doc: Doc, view: SummaryView) -> None:
    if doc.compact:
        doc.line("PRIVACY: " + privacy_line(view.exports), style="bold")
    else:
        doc.heading("PRIVACY")
        doc.line(privacy_line(view.exports), indent=2)
    render_warning_summary(doc, view)
    if view.next_actions:
        action = view.next_actions[0]
        if view.replay_unavailable and not view.paused:
            action = NextAction(
                command_for(view.profile, "metrics status --json"),
                "Inspect replay state before approved-source recompute.",
            )
        doc.blank()
        if not doc.compact:
            doc.line(safe(action.why), indent=6)
        doc.command("Next:", action.command)


def _compact_values(doc: Doc, view: SummaryView) -> None:
    doc.line("MEASURED (counted records)", style="bold")
    measured = dict(view.measured)
    counts = []
    for name in ("searches", "shows", "failed", "notes"):
        if name in measured:
            quantity = measured[name]
            counts.append(
                f"{_MEASURED_LABELS.get(name, name.capitalize())} "
                f"{quantity.value if quantity.value is not None else 'unavailable'}"
            )
    if "visible_bytes" in measured:
        counts.append(f"Visible data: {fmt_quantity(measured['visible_bytes'])}")
    doc.line("; ".join(counts), indent=2)
    doc.line("ESTIMATED NET BENEFIT (not causal proof)", style="bold")
    _scenario_table(doc, view.scenarios)
    doc.line("Net = credit minus cost; negative net = costs exceed credit.", indent=2)
    reasons = _unavailable_reasons(view.scenarios)
    if reasons:
        doc.line("Unknown, not zero: " + ", ".join(safe(reason) for reason in reasons), indent=2)
    if any(not estimate.complete for row in view.scenarios for estimate in row.estimates) or any(
        _incomplete(q) for row in view.scenarios for _, q in row.net
    ):
        doc.line(
            "* Known net subtotal; credit/cost known parts only. Complete net unavailable.",
            indent=2,
        )
    doc.line("EVIDENCE", style="bold")
    coverage = view.coverage
    doc.line(
        f"Attributed {_count(coverage.get('attributed_lookups'))}"
        f"/{_count(coverage.get('recorded_lookups'))} lookups; "
        f"source costs {_count(coverage.get('source_costs_available'))}"
        f"/{_count(coverage.get('candidate_receipts'))} receipts.",
        indent=2,
    )
    doc.line(
        f"Gaps: {_count(coverage.get('ambiguous_joins'))} ambiguous joins; "
        f"{_count(coverage.get('unavailable_sources'))} unavailable sources.",
        indent=2,
    )
    if view.exclusions:
        doc.line(
            "Excluded: "
            + "; ".join(f"{fmt_int(count)} {safe(reason)}" for reason, count in view.exclusions),
            indent=2,
        )
    doc.line(
        f"Snapshot {safe(view.snapshot_id)}, generation {view.generation}; "
        f"methodology {safe(view.methodology_id)}.",
        indent=2,
    )
    if view.replay_unavailable:
        doc.line(
            "Non-replayable (snapshot): "
            f"{_count(coverage.get('replay_unavailable_sessions'))} sessions; "
            f"{_count(coverage.get('frequency_revision_unavailable'))} frequency revisions.",
            indent=2,
        )
        doc.command(
            "Recovery: approved native sources only",
            command_for(view.profile, "metrics collect --recompute"),
        )


def render_summary(view: SummaryView, width: int = 80) -> Text:
    doc = Doc(width, compact=view.has_values and view.scope.by is None)
    doc.line("ssgrep savings", style="bold")
    doc.line(f"Profile {safe(view.profile)} | {_interval_text(view)}")
    doc.line(f"Scope: {_scope_text(view)}")
    headline = _STATE_HEADLINE[view.state]
    if view.state is SnapshotState.PAUSED and view.stale:
        headline = "PAUSED (snapshot also stale)"
    if (view.partial or view.replay_unavailable) and view.has_values:
        headline += ", partial evidence"
    status = headline + (f"; {_freshness(view)}" if view.collected_at else "")
    if doc.compact:
        doc.line(f"Status: {status}")
    else:
        doc.pair("Status", status, indent=0)
    first, caveats = split_conditions(view.conditions)
    if doc.compact:
        render_compact_conditions(doc, view)
        _compact_values(doc, view)
        _footer(doc, view)
        return doc.text()
    primary = view.next_actions[0].command if view.next_actions else None
    render_conditions(doc, first, "STATE", primary=primary)
    if not view.has_values:
        render_conditions(doc, caveats, "NOTES", primary=primary)
        render_delivery(doc, view.delivery, frozenset(c.code for c in first))
        _footer(doc, view)
        return doc.text()
    _measured(doc, view)
    _estimated(doc, view)
    render_conditions(
        doc, caveats, "READ WITH CARE", "What limits the numbers above.", primary=primary
    )
    render_groups(doc, view)
    _evidence(doc, view)
    render_delivery(doc, view.delivery, frozenset(c.code for c in first))
    _footer(doc, view)
    return doc.text()


def history_more(view: HistoryView) -> str:
    base = command_for(view.profile, "savings history")
    scope = f" --since {view.since}" if view.since else ""
    scope += f" --until {view.until}" if view.until else ""
    page = f" --limit {view.limit}" if view.limit != 20 else ""
    head, _, tail = base.partition(" --profile ")
    profile = f" --profile {tail}" if tail else ""
    return f"{head}{scope}{page} --cursor {view.next_cursor}{profile}"


def render_history(view: HistoryView, width: int = 80) -> Text:
    doc = Doc(width)
    doc.line("ssgrep savings history", style="bold")
    doc.line(f"Profile {safe(view.profile)} | closed UTC periods and their revisions")
    if not view.entries:
        doc.blank()
        doc.line(f"No closed periods to list ({safe(view.reason or 'empty')}).")
    for entry in view.entries:
        marker = "current" if entry.current else "superseded (replaced by a later revision)"
        doc.heading(f"{safe(entry.period)}  rev {entry.revision}  {safe(entry.status)}")
        doc.pair("Net", fmt_quantity(entry.net, signed=True), indent=2)
        render_estimate_details(doc, entry.scenarios)
        doc.pair("Applies", marker, indent=2)
        if entry.namespace:
            doc.pair("Namespace", safe(entry.namespace), indent=2)
        if entry.revision_name:
            doc.pair("Revision name", safe(entry.revision_name), indent=2)
        if entry.kind == "adjustment":
            doc.pair("Kind", "adjustment: late evidence replaced the earlier revision", indent=2)
        if not entry.recomputable:
            doc.pair("Evidence", "expired: total kept, cannot be recomputed from source", indent=2)
    doc.heading("CURRENT TOTAL", "Applicable revisions only; replaced revisions are not added.")
    doc.line(fmt_quantity(view.current_total, signed=True), indent=2)
    render_estimate_details(doc, view.scenarios)
    if view.next_cursor:
        doc.blank()
        doc.command("More:", history_more(view))
    for action in view.next_actions[:1]:
        doc.blank()
        doc.command("Next:", action.command)
    return doc.text()


__all__ = [
    "emit",
    "fmt_bytes",
    "privacy_line",
    "render_history",
    "render_receipt",
    "render_summary",
]
