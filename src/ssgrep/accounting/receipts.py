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
from ssgrep.accounting.valuation import Allocation
from ssgrep.accounting.valuation_credit import SCENARIOS
from ssgrep.accounting.valuation_inputs import Candidate, Debit, Methodology

RECEIPT_FORMAT = "receipt_v1"
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


def _all_atoms(candidate: Candidate) -> dict[str, str]:
    atoms = (*candidate.immediate, *candidate.discovery, *candidate.episode)
    return {a.atom_id: a.atom.digest for a in atoms}


def _costs(
    candidate: Candidate, debits: Iterable[Debit]
) -> tuple[tuple[tuple[str, Quantity], ...], tuple[tuple[str, str], ...]]:
    mine = sorted(
        (
            d
            for d in debits
            if d.exposure_id is not None and d.exposure_id in candidate.exposure_ids
        ),
        key=lambda d: d.debit_id,
    )
    costs: list[tuple[str, Quantity]] = []
    missing: list[tuple[str, str]] = []
    for debit in mine:
        name = f"{debit.kind}:{debit.debit_id}"
        costs.append((name, debit.amount))
        if debit.amount.basis is Basis.UNAVAILABLE:
            missing.append((name, debit.amount.reason))
    return tuple(costs), tuple(missing)


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
    mine = [
        a
        for a in allocations
        if a.candidate_id == candidate.candidate_id and a.task == candidate.task and a.unit == unit
    ]
    by_scenario: dict[Scenario, Allocation] = {a.scenario: a for a in mine}
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
    )
    return ReceiptRecord(
        receipt=receipt,
        atom_digests=tuple(sorted(digests.items())),
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
    for cand in candidates:
        record = build_receipt(cand, allocs, debit_list, methodology)
        out[record.receipt_id] = record
    return dict(sorted(out.items()))


def default_view(record: ReceiptRecord) -> dict[str, object]:
    """What a default local explanation may show: the canonical row, never content."""
    return record.canonical_row()
