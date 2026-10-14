"""Receipt and revision-history reads for the savings facade (read-only).

See ``service.py`` for the on-disk layout and the no-side-effects contract. A receipt or
history document written by a NEWER schema is reported as unavailable and never rewritten;
older documents (no ``schema`` field) are read as schema 1.
"""

from __future__ import annotations

import re
from typing import Any

from ssgrep.accounting import plainfile
from ssgrep.accounting.contracts import (
    PERSONAL_PROFILE,
    SCHEMA_VERSION,
    Basis,
    Quantity,
    digest_of,
)
from ssgrep.accounting.json_io import MAX_SNAPSHOT_BYTES, load_json
from ssgrep.accounting.policy_rules import PolicyError, check_profile_id
from ssgrep.accounting.scope import DEFAULT_LIMIT, interval
from ssgrep.accounting.service_cursor import issue, redeem
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
            content=str(raw["content"])
            if include_content and available and "content" in raw
            else None,
            note=None if available else _EXPIRED_NOTE,
            schema=schema,
            unit=str(raw["unit"]) if isinstance(raw.get("unit"), str) else None,
            period=str(raw["period"]) if isinstance(raw.get("period"), str) else None,
            missing_costs=_pairs(raw.get("missing_costs")),
            credit_raw=_int_pairs(raw.get("credit_raw")),
            identities=tuple(sorted((str(k), str(v)) for k, v in identities.items()))
            if isinstance(identities, dict)
            else (),
            atom_digests=_pairs(raw.get("atom_digests")),
            reuse=_reuse(raw.get("reuse")),
        )
    except _READ_ERRORS:
        raise _unreadable(profile, "file is malformed") from None


def _sum_current(entries: list[HistoryEntry]) -> Quantity:
    current = [e.net for e in entries if e.current]
    if not current:
        return Quantity.unavailable("tokens", "no_closed_periods")
    units = {q.unit for q in current}
    if any(q.value is None for q in current) or len(units) != 1:
        return Quantity.unavailable(
            next(iter(units)) if len(units) == 1 else "mixed", "incomplete_or_mixed_units"
        )
    return Quantity(
        sum(q.value or 0 for q in current), units.pop(), Basis.ESTIMATED, "sum_of_current_revisions"
    )


def _history_entries(raw: Any) -> list[HistoryEntry]:
    rows = sorted(
        (r for r in raw if isinstance(r, dict)),
        key=lambda r: (str(r.get("period")), int(r.get("revision", 1))),
    )
    latest: dict[str, int] = {}
    for row in rows:
        latest[str(row["period"])] = max(latest.get(str(row["period"]), 0), int(row["revision"]))
    return [
        HistoryEntry(
            period=str(row["period"]),
            revision=int(row["revision"]),
            status=str(row.get("status", "closed")),
            current=int(row["revision"]) == latest[str(row["period"])],
            supersedes=row.get("supersedes"),
            net=parse_quantity(row.get("net"), "tokens", "net_missing"),
            recomputable=bool(row.get("recomputable", True)),
            kind=str(row.get("kind", "original")),
        )
        for row in rows
    ]


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
            digest_of([[e.period, e.revision, e.status, e.net.value] for e in entries]),
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
        reason=None if shown else "no_closed_periods",
        next_actions=(NextAction(command_for(profile, "savings"), "Return to the summary."),),
        since=start,
        until=end,
        limit=limit,
    )
