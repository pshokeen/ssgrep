"""Canonical receipt records (design D7, ``savings-valuation`` receipts requirement).

A :class:`ReceiptRecord` wraps the frozen :class:`Receipt` with the identities the
spec demands but the frozen contract has no field for (atom digests, witness/action
locators, missing-cost reasons, frequency/tokenizer/pricing identities). Canonical
bytes exclude collection clocks and optional private content, so identical evidence
always yields identical bytes and digests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace

from ssgrep.accounting.contracts import (
    SCHEMA_VERSION,
    Basis,
    Quantity,
    Scenario,
    TaskId,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger_contracts import Receipt
from ssgrep.accounting.valuation import Allocation, Leaf, ScenarioLeaf, rollup, value_evidence
from ssgrep.accounting.valuation_credit import SCENARIOS, eligibility_for
from ssgrep.accounting.valuation_inputs import (
    Candidate,
    Debit,
    Eligibility,
    EvidenceBundle,
    Methodology,
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
