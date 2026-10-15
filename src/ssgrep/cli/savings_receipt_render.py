"""Human renderer for ``savings explain``: one receipt, in the order a reader asks questions.

What is this, what was credited per scenario and why, what it cost, what was NOT credited
(plain-language reasons), how reliable is the evidence, then where to go next. Wording is
deliberately about association: a receipt records that a later action referenced something an
earlier source first showed, never that the lookup caused it. Untrusted text is escaped by
``safe`` and never parsed as markup.
"""

from __future__ import annotations

import datetime as _dt

from rich.text import Text

from ssgrep.accounting.service_types import NextAction, ReceiptView, command_for
from ssgrep.accounting.timeunits import NS_PER_SECOND
from ssgrep.cli.savings_text import Doc, fmt_int, fmt_quantity, render_estimate_details, safe

_SCENARIO_GLOSS = {
    "conservative": "only the single source step that was directly acted on",
    "central": "adds the discovery span that led to that step",
    "optimistic": "adds the whole surrounding episode (only when acted on, relayed or reported)",
}
_REASONS = {
    "contradicted": "a later interpreted failure contradicted this reuse, so it earns no credit",
    "user_supplied": "the user supplied this fact in a prompt, so no lookup is credited for it",
    "not_novel": "it was already visible in earlier context (prior known), so it is not credited",
    "rarity_unknown": "its rarity could not be rated, so no credit is claimed",
    "rarity_below_threshold": "it is too common to count as a discovery",
    "provenance_cycle": "its source chain loops back on itself, so it is not credited",
    "unavailable": "the source record cannot be read, so its cost is unknown (not zero)",
    "out_of_scope": "the source lies outside the approved roots, so its cost is unknown",
    "digest_conflict": "the source changed since it was recorded, so it earns no credit",
    "self_later_work": "it is the consumer's own later work, never its own source",
    "cycle": "its provenance forms a cycle, so it earns no credit",
    "prior_known": "it was already known before the lookup, so it is not credited",
    "source_known_before_witness": "the source was known before the witnessed lookup, so no credit",
    "source_user_supplied": "the source text came from the user, so no lookup is credited",
    "source_unresolved": "the referenced source could not be found, so its cost is unknown",
    "witness_not_found": "the witnessing record is missing, so its cost is unknown",
    "source_not_observed": "the source was never observed in a retained record",
    "source_postdates_exposure": "the source is newer than the lookup that exposed it",
    "source_digest_drift": "the source changed after it was recorded, so it earns no credit",
    "consumer_own_later_work": "it is the consumer's own later work, never its own source",
}


def _why(code: str) -> str:
    return _REASONS.get(code, "recorded as an exclusion with this reason")


def _split(entry: str) -> tuple[str, str, str]:
    """``kind:subject:reason`` -> (kind, subject, reason); two-part entries have no subject."""
    parts = entry.split(":")
    if parts[0] not in ("candidate", "atom", "missing"):
        return "", "", entry
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], ":".join(parts[1:-1]), parts[-1]


def _credit(raw: int, unit: str | None, raw_unit: str) -> tuple[int | None, str]:
    name, _, method = (unit or "").partition(":")
    if method == "chars_div4_v1":
        return raw // 4, "tokens" if name == "context_tokens" else name
    if method == "capped_gap_v1" and name == "work_seconds":
        return (raw // NS_PER_SECOND if raw_unit == "nanoseconds" else None), "s"
    return raw, "s" if name == "work_seconds" else (name or "units")


def _days(period: str) -> tuple[str, str] | None:
    try:
        day = _dt.date.fromisoformat(period)
    except ValueError:
        return None
    return day.isoformat(), (day + _dt.timedelta(days=1)).isoformat()


def _identity(doc: Doc, view: ReceiptView) -> None:
    doc.pair("Revision", str(view.revision), indent=0)
    doc.pair("Period (UTC)", safe(view.period) if view.period else "unknown", indent=0)
    doc.pair("Conversation", safe(view.root), indent=0)
    doc.pair(
        "Task turn",
        safe(view.task_turn) if view.task_turn else "unknown (task_unknown)",
        indent=0,
    )
    doc.pair("Candidate kind", safe(view.candidate_type), indent=0)
    doc.pair("Associated exposures", str(len(view.exposure_ids)), indent=0)
    evidence = "available" if view.evidence_available else "EXPIRED (retention boundary passed)"
    doc.pair("Evidence", evidence, indent=0)
    doc.pair("Eligibility", safe(view.eligibility), indent=0)
    for reason in view.eligibility_reasons:
        doc.line("Eligibility evidence: " + safe(reason), indent=2)


def _reuse_section(doc: Doc, view: ReceiptView) -> None:
    doc.heading("HOW THE REUSE WAS ESTABLISHED", "Association between a lookup and a later action.")
    if len(view.exposure_ids) > 1:
        doc.line(
            "Flags describe evidence in at least one associated exposure, not every lookup.",
            indent=2,
        )
    if view.reuse is None:
        doc.line("Reuse evidence is not recorded in this receipt.", indent=2)
        return
    facts = dict(view.reuse)
    labels = {
        "class": "Reuse class",
        "relay": "Relayed from a child session",
        "report": "Report reference",
        "action_ok": "Acted-on step succeeded",
        "novel": "Novel in available context",
        "user_supplied": "Observed user-supplied source",
    }
    for key in ("class", "relay", "report", "action_ok", "novel", "user_supplied"):
        if key in facts:
            doc.pair(labels[key], safe(facts[key]).replace("_", " "), indent=2)


def _credit_section(doc: Doc, view: ReceiptView) -> None:
    doc.heading(
        "CREDIT BY SCENARIO",
        "Estimates of source work that a later action reused; association, not proof of cause.",
    )
    atoms = dict(view.allocations)
    raw = dict(view.credit_raw)
    if (
        view.eligibility == "unavailable"
        or not view.scenarios
        or any(s.credit_missing for s in view.scenarios)
    ):
        doc.line(
            "Credit coverage is unavailable; a known subtotal is not complete zero credit.",
            indent=2,
        )
        for scenario, value in view.credit_raw:
            amount, unit = _credit(value, view.unit, view.raw_unit)
            rendered = (
                f"{fmt_int(amount)} {unit}"
                if amount is not None
                else "unavailable (raw_time_precision_unavailable)"
            )
            doc.pair(safe(scenario).capitalize() + " known credit subtotal", rendered, indent=2)
        for scenario, chosen in view.allocations:
            doc.pair(
                safe(scenario).capitalize() + " allocated atoms",
                ", ".join(safe(a) for a in chosen) or "none recorded",
                indent=2,
            )
        if view.scenarios:
            render_estimate_details(doc, view.scenarios)
        return
    for scenario in ("conservative", "central", "optimistic"):
        if scenario not in atoms and scenario not in raw:
            doc.line(f"{scenario.capitalize()}: not computed", indent=2)
            continue
        value, unit = _credit(raw.get(scenario, 0), view.unit, view.raw_unit)
        if value is None:
            doc.line(
                f"{scenario.capitalize()}: raw allocation unavailable "
                "(raw_time_precision_unavailable)",
                indent=2,
            )
            continue
        if scenario in raw and not atoms.get(scenario):
            doc.line(f"{scenario.capitalize()}: no credit", indent=2, style="bold")
            doc.line("no source atom qualified in this scenario (see NOT CREDITED)", indent=4)
            continue
        credit = (
            f"+{fmt_int(value)} {unit}" if scenario in raw else "unavailable (credit_not_recorded)"
        )
        doc.line(f"{scenario.capitalize()}: {credit}", indent=2, style="bold")
        doc.line(_SCENARIO_GLOSS[scenario], indent=4)
        chosen = atoms.get(scenario, ())
        doc.line(
            f"{fmt_int(len(chosen))} source atom(s): "
            + (", ".join(safe(a) for a in chosen) or "none"),
            indent=4,
        )
    doc.line(
        "Overlap removed within this task: an atom already credited to an earlier candidate "
        "of the same task is not counted twice. Other tasks are credited separately.",
        indent=2,
        style="dim",
    )
    render_estimate_details(doc, view.scenarios)


def _cost_section(doc: Doc, view: ReceiptView) -> None:
    doc.heading("COST OF THE LOOKUP", "What the lookup response added to the consumer's context.")
    if not view.costs:
        doc.line("No compatible cost was recorded: unknown, not zero.", indent=2)
        return
    for name, quantity in view.costs:
        doc.pair(safe(name), fmt_quantity(quantity), indent=2)
    doc.line(
        "Individual costs are not added across units, methods or rounded components. "
        "Compatible totals and bounds appear in the typed estimates.",
        indent=2,
    )
    for name, reason in view.missing_costs:
        doc.line(f"Missing: {safe(name)} ({safe(reason)})", indent=2)


def _exclusion_section(doc: Doc, view: ReceiptView) -> None:
    doc.heading("NOT CREDITED", "Plain-language reasons; none of these is counted as savings.")
    if not view.exclusions:
        doc.line("Nothing was excluded from this receipt.", indent=2)
    for entry in view.exclusions:
        kind, subject, code = _split(entry)
        if subject.startswith("unresolved:"):  # placeholder atom: the reason is inside its id
            code = subject.split(":")[1]
            subject = ""
        scope = {
            "candidate": "Candidate gate (some exposures)"
            if len(view.exposure_ids) > 1
            else "Whole candidate",
            "atom": "Source atom",
            "missing": "Unknown source cost",
        }
        label = scope.get(kind, "Recorded")
        where = f" {safe(subject)}" if subject else ""
        doc.line(f"{label}{where}: {safe(code)}", indent=2)
        doc.line(_why(code.split(":")[0]), indent=4)


def _provenance(doc: Doc, view: ReceiptView) -> None:
    doc.heading("PROVENANCE", "What must match for two estimates to be comparable.")
    doc.pair("Methodology", safe(view.methodology), indent=2)
    for name, value in view.identities:
        doc.pair(safe(name).capitalize(), safe(value) if value else "not pinned", indent=2)
    if view.unit:
        doc.pair("Unit", safe(view.unit), indent=2)
    doc.pair("Schema", str(view.schema), indent=2)
    if view.execution_ids:
        doc.pair("Executions", fmt_int(len(view.execution_ids)), indent=2)


def _evidence_section(doc: Doc, view: ReceiptView) -> None:
    doc.heading("EVIDENCE AVAILABILITY")
    if view.evidence_available:
        doc.line("Detailed evidence is retained, so this receipt can be recomputed.", indent=2)
        return
    doc.line("Detailed evidence has expired.", indent=2)
    doc.line("Retained totals above are kept; they cannot be recomputed from source.", indent=2)
    doc.line("Full recomputation is unavailable; nothing was reconstructed.", indent=2)
    if view.note:
        doc.line(safe(view.note), indent=2, style="dim")


def next_actions(view: ReceiptView, profile: str) -> tuple[NextAction, ...]:
    actions = []
    span = _days(view.period) if view.period else None
    if span:
        actions.append(
            NextAction(
                command_for(profile, f"savings history --since {span[0]} --until {span[1]}"),
                "See the closed period this receipt belongs to.",
            )
        )
    actions.append(NextAction(command_for(profile, "savings"), "Return to the summary."))
    return tuple(actions)


def render_receipt(
    view: ReceiptView,
    profile_hint: str = "",
    width: int = 80,
    include_content: bool = False,
) -> Text:
    profile = profile_hint or "personal-local"
    doc = Doc(width)
    doc.line(f"ssgrep savings explain {safe(view.receipt_id)}", style="bold")
    _identity(doc, view)
    _reuse_section(doc, view)
    _credit_section(doc, view)
    _cost_section(doc, view)
    _exclusion_section(doc, view)
    _evidence_section(doc, view)
    _provenance(doc, view)
    if view.content is not None:
        doc.heading("CONTENT", "Local only; shown because --include-content was given.")
        doc.line(safe(view.content), indent=2)
    elif include_content:
        doc.heading("CONTENT")
        reason = (
            "evidence has expired, so no content can be shown"
            if not view.evidence_available
            else "no private content is stored in this receipt (content stays in your transcripts)"
        )
        doc.line(f"Nothing to show: {reason}.", indent=2)
    else:
        doc.blank()
        doc.line("Private content is hidden. Add --include-content to show it locally.")
    actions = next_actions(view, profile)
    doc.blank()
    doc.line(actions[0].why, indent=6)
    doc.command("Next:", actions[0].command)
    return doc.text()
