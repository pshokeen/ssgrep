"""Named experience states of a summary: each carries its cause and a REAL next command.

Pure functions over an already-built ``SummaryView`` plus delivery lines; no I/O and no clocks.
Conditions are additive facts (a view can be stale AND partial AND negative); the primary
``next_action`` of the view stays the single most applicable step, the conditions explain the
rest. Unknown quantities are never converted to zero here.
"""

from __future__ import annotations

from ssgrep.accounting.ledger_contracts import SnapshotState
from ssgrep.accounting.service_types import (
    Condition,
    DeliveryLine,
    NextAction,
    SummaryView,
    command_for,
)

STALE_LIMIT_NOTE = "three collection intervals (180 s)"
_DELIVERY_PROBLEMS = (
    "credentials_expired",
    "blocked",
    "partial_export",
    "unknown",
    "offline",
    "unreadable",
)


def _act(profile: str, text: str, why: str) -> NextAction:
    return NextAction(command_for(profile, text), why)


def _age_text(seconds: int) -> str:
    """Format a stale snapshot's known age (strictly above the 180-second limit)."""
    if seconds < 7200:
        return f"{seconds // 60} min"
    if seconds < 172800:
        return f"{seconds // 3600} h"
    return f"{seconds // 86400} d"


def _partial_cause(view: SummaryView) -> str:
    cov = view.coverage
    known = []
    recorded, attributed = cov.get("recorded_lookups"), cov.get("attributed_lookups")
    if recorded is not None and attributed is not None:
        known.append(f"{attributed} of {recorded} recorded lookups are attributed")
    costs, receipts = cov.get("source_costs_available"), cov.get("candidate_receipts")
    if costs is not None and receipts is not None and receipts:
        known.append(f"source costs are known for {costs} of {receipts} candidate receipts")
    for name, label in (
        ("ambiguous_joins", "ambiguous joins"),
        ("unavailable_sources", "unavailable source records"),
    ):
        value = cov.get(name)
        if value:
            known.append(f"{value} {label}")
    detail = "; ".join(known) if known else "some evidence was not fully read"
    return f"Estimates cover only part of the evidence: {detail}."


def summary_conditions(
    view: SummaryView,
    delivery: tuple[DeliveryLine, ...],
    sending_paused_reason: str | None,
) -> tuple[Condition, ...]:
    p = view.profile
    out: list[Condition] = []
    state = view.state
    if state is SnapshotState.DISABLED:
        out.append(
            Condition(
                "disabled",
                "Measurement is off; nothing is collected or exported, "
                "and reading does not turn it on.",
                _act(p, "metrics setup", "Preview what would be kept, then enable."),
            )
        )
    elif state is SnapshotState.NO_SNAPSHOT:
        out.append(
            Condition(
                "enabled_no_snapshot",
                "Capture is enabled but no collection has completed, so there are no numbers yet.",
                _act(p, "metrics collect", "Run the first collection."),
            )
        )
    elif state in (SnapshotState.CORRUPT, SnapshotState.UNSUPPORTED_SCHEMA):
        why = (view.reason or state.value).split(":")[0]
        out.append(
            Condition(
                state.value,
                f"The latest snapshot cannot be read ({why}), so no "
                "numbers are shown. Nothing was changed or rewritten.",
                _act(p, "metrics status", "Inspect state without changing it."),
            )
        )
    if view.paused:
        since = view.paused_at or "an unknown time"
        out.append(
            Condition(
                "paused",
                f"Paused since {since}: capture, scheduled collection and sending are stopped; "
                "retained values stay readable and the paused interval is not imported.",
                _act(p, "metrics resume", "Resume capture, collection and sending."),
            )
        )
    if view.stale and view.collected_at:
        assert view.age_seconds is not None
        out.append(
            Condition(
                "stale",
                f"Last good snapshot was collected {_age_text(view.age_seconds)} ago "
                f"({view.collected_at}), older than {STALE_LIMIT_NOTE}. "
                "These are the last good values, not new ones.",
                None if view.paused else _act(p, "metrics collect", "Collect again to refresh."),
            )
        )
    if view.no_reuse:
        reasons = ", ".join(f"{n} {r}" for r, n in view.exclusions) or "none recorded"
        out.append(
            Condition(
                "no_reuse",
                "No traceable reuse was found: usage is measured, so no savings are claimed. "
                f"Excluded: {reasons}.",
                _act(p, "metrics status", "See coverage and what was read."),
            )
        )
    elif view.partial and view.has_values:
        out.append(
            Condition(
                "partial",
                _partial_cause(view),
                _act(p, f"savings explain {view.receipt_id}", "Inspect a real receipt.")
                if view.receipt_id
                else _act(p, "metrics status", "See coverage detail."),
            )
        )
    if view.has_negative:
        out.append(
            Condition(
                "negative",
                "In a negative scenario, retrieval cost exceeded the estimated credit "
                "(minus sign = net cost, not a benefit).",
                _act(p, f"savings explain {view.receipt_id}", "See what drove the cost.")
                if view.receipt_id
                else _act(p, "metrics status", "See coverage detail."),
            )
        )
    if view.index == "unavailable" and view.has_values:
        out.append(
            Condition(
                "index_unavailable",
                "The search index is unavailable: usage is still measured, but valuation of "
                "source content may be missing. Nothing is indexed automatically.",
                NextAction("ssgrep index", "Build the search index yourself if you want it."),
            )
        )
    expired = view.evidence_expired_before
    if expired or view.frozen_roots:
        boundary = f" before {expired}" if expired else ""
        out.append(
            Condition(
                "evidence_expired",
                f"Detailed evidence{boundary} is past retention ({view.frozen_roots or 0} "
                "conversation root(s) frozen): retained totals remain but cannot be recomputed.",
                _act(p, "savings history", "See which periods are not recomputable."),
            )
        )
    for line in delivery:
        if line.state in _DELIVERY_PROBLEMS:
            out.append(
                Condition(
                    line.state, f"{line.destination_id}/{line.signal}: {line.cause}", line.action
                )
            )
    if sending_paused_reason and not view.paused:
        out.append(
            Condition(
                "sending_paused",
                f"Sending is paused until the policy is corrected ({sending_paused_reason}).",
                NextAction(command_for(p, "metrics setup --policy <file>"), "Correct the policy."),
            )
        )
    return tuple(out)


def merged_actions(
    primary: tuple[NextAction, ...], conditions: tuple[Condition, ...]
) -> tuple[NextAction, ...]:
    """Primary action first, then each condition's distinct command (placeholders skipped)."""
    seen: set[str] = set()
    out: list[NextAction] = []
    candidates = [*primary, *(c.action for c in conditions if c.action is not None)]
    for action in candidates:
        if action.command in seen or "<" in action.command:
            continue
        seen.add(action.command)
        out.append(action)
    return tuple(out)
