"""Receipt and revision-history reads for the savings facade (read-only).

See ``service.py`` for the on-disk layout and the no-side-effects contract. A receipt or
history document written by a NEWER schema is reported as unavailable and never rewritten;
older documents retain known subtotals but never acquire complete v2 nets through defaults.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ssgrep.accounting import plainfile
from ssgrep.accounting.contracts import (
    DEFAULT_LIMIT,
    PERSONAL_PROFILE,
    SCHEMA_VERSION,
    Quantity,
    digest_of,
)
from ssgrep.accounting.json_io import MAX_SNAPSHOT_BYTES, load_json
from ssgrep.accounting.policy_rules import PolicyError, check_profile_id
from ssgrep.accounting.scope import interval
from ssgrep.accounting.service_cursor import issue, redeem
from ssgrep.accounting.service_estimates import LEGACY_REASON, parse_scenarios, sum_scenarios
from ssgrep.accounting.service_types import (
    HistoryEntry,
    HistoryView,
    NextAction,
    ReceiptView,
    command_for,
    parse_quantity,
)

MAX_RECEIPT_BYTES = 1024 * 1024
RECEIPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_EXPIRED_NOTE = (
    "evidence_expired: detailed evidence is past retention; "
    "totals are kept, recomputation is unavailable"
)
_READ_ERRORS = (plainfile.UnsafePathError, ValueError, OSError, KeyError, TypeError, AttributeError)


def _quantities(raw: Any, default_unit: str) -> tuple[tuple[str, Quantity], ...]:
    pairs = raw if isinstance(raw, list) else []
    return tuple(
        (str(p[0]), parse_quantity(p[1], default_unit, "cost_missing"))
        for p in pairs
        if isinstance(p, list) and len(p) == 2
    )


def _pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    pairs = raw if isinstance(raw, list) else []
    return tuple((str(p[0]), str(p[1])) for p in pairs if isinstance(p, list) and len(p) == 2)


def _int_pairs(raw: Any) -> tuple[tuple[str, int], ...]:
    pairs = raw if isinstance(raw, list) else []
    return tuple(
        (str(p[0]), p[1])
        for p in pairs
        if isinstance(p, list)
        and len(p) == 2
        and isinstance(p[1], int)
        and not isinstance(p[1], bool)
    )


def _reuse(raw: Any) -> tuple[tuple[str, str], ...] | None:
    """How the reuse was established (optional, bounded vocabulary); None when not recorded."""
    if not isinstance(raw, dict):
        return None
    out = []
    for key in sorted(raw):
        value = raw[key]
        if isinstance(value, bool):
            out.append((str(key)[:24], "yes" if value else "no"))
        elif isinstance(value, str):
            out.append((str(key)[:24], value[:48]))
    return tuple(out)


def _schema_of(raw: dict[str, Any]) -> int:
    version = raw.get("schema", SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("schema must be an integer")
    return version


def _unreadable(profile: str, why: str) -> PolicyError:
    return PolicyError(
        "receipt_unreadable",
        f"That receipt {why}; nothing was changed.",
        command_for(profile, "metrics status"),
        1,
    )


def read_receipt(
    profile: str, receipt_id: str, include_content: bool = False
) -> ReceiptView | None:
    """The stored receipt or None when unknown; ids that could be paths are unknown.

    Raises ``PolicyError`` (never rewrites anything) when the receipt exists but is written
    by a newer schema (``receipt_unsupported_schema``, exit 3) or is damaged
    (``receipt_unreadable``, exit 1).
    """
    check_profile_id(profile)
    if not RECEIPT_ID_RE.fullmatch(receipt_id):
        return None
    try:
        raw = load_json(profile, "receipts", f"{receipt_id}.json", limit=MAX_RECEIPT_BYTES)
    except _READ_ERRORS:
        raise _unreadable(profile, "file exists but cannot be read safely") from None
    if raw is None:
        return None
    try:
        if not isinstance(raw, dict) or raw.get("receipt_id") != receipt_id:
            raise ValueError("receipt id mismatch")
        schema = _schema_of(raw)
    except _READ_ERRORS:
        raise _unreadable(profile, "file is malformed") from None
    if schema > SCHEMA_VERSION:
        raise PolicyError(
            "receipt_unsupported_schema",
            f"That receipt was written by a newer schema ({schema}); this ssgrep reads schema "
            f"{SCHEMA_VERSION}. It is unavailable here and was not modified. Upgrade ssgrep.",
            command_for(profile, "metrics status"),
            3,
        )
    try:
        available = bool(raw.get("evidence_available", True))
        identities = raw.get("identities")
        scenarios = parse_scenarios(raw.get("scenarios"), current=raw.get("format") == "receipt_v2")
        current_format = raw.get("format") == "receipt_v2" and bool(scenarios)
        return ReceiptView(
            receipt_id=receipt_id,
            revision=int(raw["revision"]),
            root=str(raw["root"]),
            task_turn=raw.get("task_turn"),
            candidate_type=str(raw.get("candidate_type", "unknown")),
            atoms=tuple(str(a) for a in raw.get("atoms", [])),
            allocations=tuple((str(k), tuple(map(str, v))) for k, v in raw.get("allocations", [])),
            exclusions=tuple(str(e) for e in raw.get("exclusions", [])),
            costs=_quantities(raw.get("costs"), "tokens"),
            methodology=str(raw.get("methodology", "")),
            evidence_available=available,
            execution_ids=tuple(str(e) for e in raw.get("execution_ids", [])),
            exposure_ids=tuple(str(e) for e in raw.get("exposure_ids", [])),
            content=str(raw["content"])
            if include_content and available and "content" in raw
            else None,
            note=None if available else _EXPIRED_NOTE,
            schema=schema,
            unit=str(raw["unit"]) if isinstance(raw.get("unit"), str) else None,
            period=str(raw["period"]) if isinstance(raw.get("period"), str) else None,
            missing_costs=_pairs(raw.get("missing_costs")),
            credit_raw=_int_pairs(raw.get("credit_raw")),
            raw_unit=raw.get("raw_unit") if isinstance(raw.get("raw_unit"), str) else "",
            debit_raw=raw.get("debit_raw") if type(raw.get("debit_raw")) is int else None,
            debit_raw_min=raw.get("debit_raw_min")
            if type(raw.get("debit_raw_min")) is int
            else None,
            debit_raw_max=raw.get("debit_raw_max")
            if type(raw.get("debit_raw_max")) is int
            else None,
            identities=tuple(sorted((str(k), str(v)) for k, v in identities.items()))
            if isinstance(identities, dict)
            else (),
            atom_digests=_pairs(raw.get("atom_digests")),
            reuse=_reuse(raw.get("reuse")),
            eligibility=str(raw.get("eligibility", "unavailable"))
            if current_format
            else "unavailable",
            eligibility_reasons=tuple(str(r) for r in raw.get("eligibility_reasons", []))
            if current_format
            else (LEGACY_REASON,),
            provenance=_pairs(raw.get("provenance")),
            scenarios=scenarios,
        )
    except _READ_ERRORS:
        raise _unreadable(profile, "file is malformed") from None


def _sum_current(entries: list[HistoryEntry]) -> Quantity:
    scenarios = sum_scenarios(e.scenarios for e in entries if e.current)
    central = next((s for s in scenarios if s.scenario.value == "central"), None)
    return (
        central.net
        if central is not None
        else Quantity.unavailable("tokens", "no_current_estimate_metadata")
    )


def _history_entries(raw: Any) -> list[HistoryEntry]:
    from ssgrep.accounting.published_documents import FORMAT, ORIGINAL_REASON

    rows = sorted(
        (row for row in raw if isinstance(row, dict)),
        key=lambda row: (str(row.get("period")), int(row.get("revision", 1))),
    )
    latest: dict[str, int] = {}
    for row in rows:
        if row.get("current", True) and row.get("published_format") != FORMAT:
            period = str(row["period"])
            latest[period] = max(latest.get(period, 0), int(row["revision"]))
    entries = []
    for row in rows:
        original = row.get("published_format") == FORMAT
        scenarios = (
            ()
            if original
            else parse_scenarios(row.get("scenarios"), current=row.get("format") == "history_v2")
        )
        if original:
            if row.get("current") is not False or not row.get("namespace"):
                raise ValueError("Original publication is not bound to an inactive namespace.")
            net = parse_quantity(row.get("net"), "tokens", ORIGINAL_REASON)
            net = replace(
                net,
                reason=ORIGINAL_REASON
                if net.value is not None
                else ORIGINAL_REASON + ":" + net.reason,
            )
        else:
            net = next(
                (estimate.net for estimate in scenarios if estimate.scenario.value == "central"),
                Quantity.unavailable("tokens", LEGACY_REASON),
            )
        entries.append(
            HistoryEntry(
                period=str(row["period"]),
                revision=int(row["revision"]),
                status=str(row.get("status", "closed")),
                current=not original
                and bool(row.get("current", True))
                and int(row["revision"]) == latest.get(str(row["period"])),
                supersedes=row.get("supersedes"),
                net=net,
                recomputable=False if original else bool(row.get("recomputable", True)),
                kind="original_published" if original else str(row.get("kind", "original")),
                scenarios=scenarios,
                namespace=str(row.get("namespace", "")),
                revision_name=str(row.get("revision_name", "")),
            )
        )
    return entries


def _unavailable(profile: str, reason: str) -> HistoryView:
    return HistoryView(
        profile,
        (),
        Quantity.unavailable("tokens", reason),
        None,
        reason=reason,
        next_actions=(NextAction(command_for(profile, "metrics status"), "Inspect state."),),
    )


def read_history(
    profile: str = PERSONAL_PROFILE,
    since: str | None = None,
    until: str | None = None,
    limit: int = DEFAULT_LIMIT,
    cursor: str | None = None,
) -> HistoryView:
    """Closed periods with revisions; totals count only the applicable revision.

    The cursor is bound to the profile, interval, limit and a digest of the complete stored
    period list, so it is rejected (exit 2) if the data or selection changed.
    """
    check_profile_id(profile)
    start, end = interval(since, until)
    try:
        raw = load_json(profile, "history", "periods.json", limit=MAX_SNAPSHOT_BYTES)
        from ssgrep.accounting.published_documents import merge_stored_history

        raw, published_warnings = merge_stored_history(profile, raw)
        if isinstance(raw, list):
            newer = [r for r in raw if isinstance(r, dict) and _schema_of(r) > SCHEMA_VERSION]
            if newer:
                return _unavailable(profile, "history_unsupported_schema")
        entries = _history_entries(raw) if isinstance(raw, list) else []
    except _READ_ERRORS:
        return _unavailable(profile, "history_unreadable")
    shown = [
        e
        for e in entries
        if (not start or e.period >= start[:10]) and (not end or e.period < end[:10])
    ]
    binding = "|".join(
        [
            profile,
            str(start),
            str(end),
            str(limit),
            digest_of(raw),
        ]
    )
    offset = redeem("h", cursor, binding, command_for(profile, "savings history"))
    page = shown[offset : offset + limit]
    more = offset + limit < len(shown)
    return HistoryView(
        profile=profile,
        entries=tuple(page),
        current_total=_sum_current(shown),
        next_cursor=issue("h", offset + limit, binding) if more else None,
        reason=published_warnings[0]
        if published_warnings
        else (None if shown else "no_closed_periods"),
        next_actions=(NextAction(command_for(profile, "savings"), "Return to the summary."),),
        since=start,
        until=end,
        limit=limit,
        scenarios=sum_scenarios(e.scenarios for e in shown if e.current),
    )
