"""Receipt and revision-history reads for the savings facade (read-only).

See ``service.py`` for the on-disk layout and the no-side-effects contract. A receipt or
history document written by a NEWER schema is reported as unavailable and never rewritten;
older documents retain known subtotals but never acquire complete v2 nets through defaults.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from ssgrep.accounting.ledger.contracts import (
    DEFAULT_LIMIT,
    PERSONAL_PROFILE,
    SCHEMA_VERSION,
    Basis,
    Quantity,
    Scenario,
    TaskId,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_contracts import Receipt
from ssgrep.accounting.ledger.valuation import (
    Allocation,
    Leaf,
    ScenarioLeaf,
    rollup,
    value_evidence,
)
from ssgrep.accounting.ledger.valuation_inputs import (
    SCENARIOS,
    Candidate,
    Debit,
    Eligibility,
    EvidenceBundle,
    Methodology,
    eligibility_for,
)
from ssgrep.accounting.privacy import storage as plainfile
from ssgrep.accounting.privacy.policy import PolicyError, check_profile_id
from ssgrep.accounting.privacy.storage import MAX_SNAPSHOT_BYTES, load_json
from ssgrep.accounting.views.types import (
    HistoryEntry,
    HistoryView,
    NextAction,
    ReceiptView,
    command_for,
    interval,
    issue,
    parse_quantity,
    redeem,
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
        command_for(profile, "savings status"),
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
    from ssgrep.accounting.views.history import LEGACY_REASON, parse_scenarios

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
            command_for(profile, "savings status"),
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
    from ssgrep.accounting.views.history import sum_scenarios

    scenarios = sum_scenarios(e.scenarios for e in entries if e.current)
    central = next((s for s in scenarios if s.scenario.value == "central"), None)
    return (
        central.net
        if central is not None
        else Quantity.unavailable("tokens", "no_current_estimate_metadata")
    )


def _history_entries(raw: Any) -> list[HistoryEntry]:
    from ssgrep.accounting.privacy.published import FORMAT, ORIGINAL_REASON
    from ssgrep.accounting.views.history import LEGACY_REASON, parse_scenarios

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
        next_actions=(NextAction(command_for(profile, "savings status"), "Inspect state."),),
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
    from ssgrep.accounting.views.history import sum_scenarios

    check_profile_id(profile)
    start, end = interval(since, until)
    try:
        raw = load_json(profile, "history", "periods.json", limit=MAX_SNAPSHOT_BYTES)
        from ssgrep.accounting.privacy.published import merge_stored_history

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


RECEIPT_FORMAT = "receipt_v2"
EXPIRED_REASON = "evidence_expired"


@dataclass(frozen=True)
class ReceiptRecord:
    """One revision of one receipt plus the identities explaining its net."""

    receipt: Receipt
    atom_digests: tuple[tuple[str, str], ...]  # (atom id, content digest)
    witness_locators: tuple[str, ...]
    action_locators: tuple[str, ...]
    missing_costs: tuple[tuple[str, str], ...]  # (cost name, reason)
    credit_raw: tuple[tuple[str, int], ...]  # per scenario, undivided, this receipt's share
    frequency_identity: str
    tokenizer_identity: str
    pricing_identity: str
    unit: str
    period: str  # UTC day of the candidate
    content: str | None = None  # optional, local only, never canonical
    raw_unit: str = ""  # explicit raw allocation/debit scale; never infer legacy time precision
    debit_raw: int | None = None
    debit_raw_min: int | None = None
    debit_raw_max: int | None = None

    def canonical_row(self) -> dict[str, object]:
        """The deterministic ledger content: no clocks, no private content."""
        row = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "content"}
        return {"format": RECEIPT_FORMAT, **row}

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.canonical_row())

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_row())

    @property
    def receipt_id(self) -> str:
        return self.receipt.receipt_id

    def with_content(self, content: str | None) -> ReceiptRecord:
        return replace(self, content=content)

    def mark_expired(self) -> ReceiptRecord:
        """Retain the closed total but state evidence is gone and recomputation is not."""
        receipt = replace(self.receipt, evidence_available=False)
        missing = tuple(sorted(set(self.missing_costs) | {("recomputation", EXPIRED_REASON)}))
        return replace(self, receipt=receipt, missing_costs=missing, content=None)


def receipt_id_for(task: TaskId, candidate_id: str, methodology: Methodology) -> str:
    """Stable id: the same candidate in the same task and methodology never changes id."""
    return digest_of([task.root, task.turn, task.branch, candidate_id, methodology.hash])[:24]


def _all_atoms(candidate: Candidate) -> tuple[tuple[str, str], ...]:
    atoms = (*candidate.immediate, *candidate.discovery, *candidate.episode)
    return tuple(sorted({(a.atom_id, a.atom.digest) for a in atoms}))


def _costs(
    candidate: Candidate, debits: Iterable[Debit]
) -> tuple[tuple[tuple[str, Quantity], ...], tuple[tuple[str, str], ...]]:
    costs: list[tuple[str, Quantity]] = []
    missing: list[tuple[str, str]] = []
    groups: dict[str, dict[str, Debit]] = {}
    for debit in debits:
        if debit.exposure_id is not None and debit.exposure_id in candidate.exposure_ids:
            groups.setdefault(debit.debit_id, {})[digest_of(debit)] = debit
    for _, variants in sorted(groups.items()):
        debit = variants[min(variants)]
        name = f"{debit.kind}:{debit.debit_id}"
        amount = (
            debit.amount
            if len(variants) == 1
            else Quantity.unavailable(debit.unit.unit, "conflicting_duplicate_debit")
        )
        costs.append((name, amount))
        if amount.basis is Basis.UNAVAILABLE:
            missing.append((name, amount.reason))
    return tuple(costs), tuple(missing)


def _scenario_estimates(candidate, by_scenario, debits, methodology):
    """Receipt credit is its allocated share, not the candidate's full source union."""
    unit = methodology.units[0]
    costs = tuple(d for d in debits if d.unit == unit and d.exposure_id in candidate.exposure_ids)
    valued = value_evidence(EvidenceBundle(debits=costs), methodology)
    leaves = [leaf for leaf in valued.leaves if leaf.unit == unit]
    scenarios = tuple(
        ScenarioLeaf(
            s,
            by_scenario[s].credit_raw,
            len(by_scenario[s].atoms),
            by_scenario[s].missing,
            ((candidate.candidate_id, eligibility_for(candidate, (by_scenario[s],)).value),),
        )
        if s in by_scenario
        else ScenarioLeaf(
            s,
            0,
            0,
            ((f"candidate:{candidate.candidate_id}", "allocation_unavailable"),),
            ((candidate.candidate_id, Eligibility.UNAVAILABLE.value),),
        )
        for s in SCENARIOS
    )
    maximum = (
        None
        if any(leaf.debit_max is None for leaf in leaves)
        else sum(leaf.debit_max for leaf in leaves if leaf.debit_max is not None)
    )
    leaf = Leaf(
        candidate.task,
        candidate.day,
        unit,
        methodology.hash,
        scenarios,
        sum(leaf.debit_raw for leaf in leaves),
        sum(leaf.debit_known for leaf in leaves),
        tuple(sorted({m for leaf in leaves for m in leaf.debit_missing})),
        sum(leaf.debit_min for leaf in leaves),
        maximum,
        ((candidate.candidate_id, eligibility_for(candidate, by_scenario.values()).value),),
        tuple(sorted(set(candidate.provenance) | {p for leaf in leaves for p in leaf.provenance})),
        tuple(sorted({a for leaf in leaves for a in leaf.assumptions})),
        "ns" if unit.unit == "work_seconds" and unit.method == "capped_gap_v1" else "",
    )
    return tuple(e.as_scenario_estimate() for e in rollup((leaf,), unit)), leaf


def _group_allocations(rows: Iterable[Allocation]) -> dict[Scenario, Allocation]:
    groups: dict[Scenario, dict[str, Allocation]] = {}
    for row in rows:
        groups.setdefault(row.scenario, {})[digest_of(row)] = row
    result = {}
    for scenario, unique in groups.items():
        members = list(unique.values())
        reasons = {a.ineligible for a in members if a.ineligible}
        result[scenario] = replace(
            members[0],
            ineligible=",".join(sorted(reasons)) if all(a.ineligible for a in members) else None,
            eligible=tuple(sorted({v for a in members for v in a.eligible})),
            atoms=tuple(sorted({v for a in members for v in a.atoms})),
            credit_raw=sum(a.credit_raw for a in members),
            missing=tuple(sorted({v for a in members for v in a.missing})),
            excluded=tuple(sorted({v for a in members for v in a.excluded})),
        )
    return result


def _group_candidate(group: list[Candidate]) -> Candidate:
    rank = {Eligibility.SUPPORTED: 0, Eligibility.UNAVAILABLE: 1, Eligibility.KNOWN_REJECTED: 2}
    members = sorted(group, key=lambda c: (rank[c.eligibility], c.day, digest_of(c)))
    updates = {
        name: tuple(sorted({v for c in members for v in getattr(c, name)}, key=digest_of))
        for name in (
            "exposure_ids",
            "execution_ids",
            "witness_locators",
            "action_locators",
            "immediate",
            "discovery",
            "episode",
        )
    }
    provenance = {p for c in members for p in c.provenance}
    provenance.update(
        (f"eligibility:{exposure}", c.eligibility.value)
        for c in members
        for exposure in c.exposure_ids
    )
    return replace(
        members[0],
        **updates,
        day=min(c.day for c in members),
        provenance=tuple(sorted(provenance)),
        eligibility_reasons=tuple(sorted({r for c in members for r in c.eligibility_reasons})),
    )


def build_receipt(
    candidate: Candidate,
    allocations: Iterable[Allocation],
    debits: Iterable[Debit],
    methodology: Methodology,
    revision: int = 1,
    evidence_available: bool = True,
) -> ReceiptRecord:
    """Explain one candidate: allocations per scenario, exclusions, costs, identities.

    Allocations are explained in the methodology's first unit (receipts are per unit).
    """
    unit = methodology.units[0]
    debits = tuple(debits)
    mine = [
        a
        for a in allocations
        if a.candidate_id == candidate.candidate_id and a.task == candidate.task and a.unit == unit
    ]
    by_scenario = _group_allocations(mine)
    alloc = tuple((s.value, by_scenario[s].atoms) for s in SCENARIOS if s in by_scenario)
    eligible = sorted({atom for a in mine for atom in a.eligible})
    exclusions: set[str] = set()
    for a in mine:
        if a.ineligible:
            exclusions.add(f"candidate:{a.ineligible}")
        exclusions.update(f"atom:{atom}:{reason}" for atom, reason in a.excluded)
        exclusions.update(f"missing:{atom}:{reason}" for atom, reason in a.missing)
    costs, missing_costs = _costs(candidate, debits)
    digests = _all_atoms(candidate)
    scenarios, raw_leaf = _scenario_estimates(candidate, by_scenario, debits, methodology)
    eligibility = eligibility_for(candidate, by_scenario.values())
    if any(s.credit_missing for s in scenarios) and eligibility is Eligibility.SUPPORTED:
        eligibility = Eligibility.UNAVAILABLE
    receipt = Receipt(
        receipt_id=receipt_id_for(candidate.task, candidate.candidate_id, methodology),
        revision=revision,
        root=candidate.task.root,
        task_turn=candidate.task.turn,
        execution_ids=tuple(sorted(set(candidate.execution_ids))),
        exposure_ids=tuple(sorted(set(candidate.exposure_ids))),
        candidate_fingerprint=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
        atoms=tuple(eligible),
        allocations=alloc,
        exclusions=tuple(sorted(exclusions)),
        costs=costs,
        methodology=methodology.hash,
        evidence_available=evidence_available,
        schema=SCHEMA_VERSION,
        eligibility=eligibility.value,
        eligibility_reasons=tuple(
            sorted(
                set(candidate.eligibility_reasons)
                | {reason for s in scenarios for _, reason in s.credit_missing}
            )
        ),
        provenance=tuple(
            sorted(set(candidate.provenance) | {p for s in scenarios for p in s.provenance})
        ),
        scenarios=scenarios,
    )
    return ReceiptRecord(
        receipt=receipt,
        atom_digests=digests,
        witness_locators=tuple(sorted(set(candidate.witness_locators))),
        action_locators=tuple(sorted(set(candidate.action_locators))),
        missing_costs=tuple(sorted(missing_costs)),
        credit_raw=tuple(
            (s.value, by_scenario[s].credit_raw) for s in SCENARIOS if s in by_scenario
        ),
        frequency_identity=methodology.frequency_view,
        tokenizer_identity=methodology.tokenizer,
        pricing_identity=methodology.pricing,
        unit=f"{unit.unit}:{unit.method}",
        period=candidate.day,
        raw_unit=(
            "characters"
            if unit.method == "chars_div4_v1"
            else "nanoseconds"
            if unit.unit == "work_seconds" and unit.method == "capped_gap_v1"
            else unit.unit
        ),
        debit_raw=raw_leaf.debit_raw,
        debit_raw_min=raw_leaf.debit_min,
        debit_raw_max=raw_leaf.debit_max,
    )


def build_receipts(
    candidates: Iterable[Candidate],
    allocations: Iterable[Allocation],
    debits: Iterable[Debit],
    methodology: Methodology,
) -> Mapping[str, ReceiptRecord]:
    """Revision-1 receipts keyed by id (revisions are assigned by the replay layer)."""
    allocs = list(allocations)
    debit_list = list(debits)
    out: dict[str, ReceiptRecord] = {}
    groups: dict[str, list[Candidate]] = {}
    for cand in candidates:
        groups.setdefault(receipt_id_for(cand.task, cand.candidate_id, methodology), []).append(
            cand
        )
    for receipt_id, members in sorted(groups.items()):
        out[receipt_id] = build_receipt(_group_candidate(members), allocs, debit_list, methodology)
    return out


def default_view(record: ReceiptRecord) -> dict[str, object]:
    """What a default local explanation may show: the canonical row, never content."""
    return record.canonical_row()
