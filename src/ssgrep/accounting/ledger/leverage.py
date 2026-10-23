"""Note investment/leverage and overhead views (design D6, last paragraph).

Leverage is reported SEPARATELY from net savings: it is never an input to
``valuation.rollup``. A note's leverage is the deduplicated credit of the
consumers that reused it, against its observed creation cost, with an explicit
payback-coverage denominator. Creation cost that is partly unavailable never
yields a complete payback claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

from ssgrep.accounting.ledger.contracts import Basis, Quantity, Scenario
from ssgrep.accounting.ledger.units import UnitSpec, sum_quantities
from ssgrep.accounting.ledger.valuation import Valuation, to_unit_value, value_evidence
from ssgrep.accounting.ledger.valuation_inputs import Candidate, EvidenceBundle, Methodology

LEVERAGE_VERSION = "note_leverage_v1"


class Purpose(StrEnum):
    PROTOCOL = "protocol"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NoteInvestment:
    """One observed cost of creating (or indexing) a note, in a comparable unit."""

    note_id: str
    kind: str  # creation | index
    cost: Quantity


@dataclass(frozen=True)
class ScenarioLeverage:
    scenario: Scenario
    credit: Quantity  # deduplicated subsequent consumer credit (known subtotal)
    consumers: int
    credit_missing: int


@dataclass(frozen=True)
class NoteLeverage:
    note_id: str
    unit: UnitSpec
    scenarios: tuple[ScenarioLeverage, ...]
    known_cost: int | None  # None when no cost component is known
    cost_components: int  # payback-coverage denominator
    known_components: int  # payback-coverage numerator
    missing_reasons: tuple[str, ...]

    @property
    def payback_complete(self) -> bool:
        return self.known_components == self.cost_components and self.cost_components > 0

    def payback_permille(self, scenario: Scenario) -> int | None:
        """credit/cost * 1000 only when cost AND credit are complete; else unknown."""
        row = next(s for s in self.scenarios if s.scenario is scenario)
        if not self.payback_complete or row.credit_missing or not self.known_cost:
            return None
        return (row.credit.value or 0) * 1000 // self.known_cost


def note_leverage(
    note_id: str,
    note_origins: frozenset[str],
    consumers: Iterable[Candidate],
    investments: Iterable[NoteInvestment],
    methodology: Methodology,
    unit: UnitSpec,
) -> NoteLeverage:
    """Credit of consumers' reuse of THIS note's atoms only, unioned per consumer task."""
    scoped: list[Candidate] = []
    for cand in consumers:
        narrowed = replace(
            cand,
            immediate=tuple(a for a in cand.immediate if a.atom.origin_event in note_origins),
            discovery=tuple(a for a in cand.discovery if a.atom.origin_event in note_origins),
            episode=tuple(a for a in cand.episode if a.atom.origin_event in note_origins),
        )
        scoped.append(narrowed)
    valuation: Valuation = value_evidence(
        EvidenceBundle(candidates=tuple(scoped)), replace(methodology, units=(unit,))
    )
    rows: list[ScenarioLeverage] = []
    for scenario in (Scenario.CONSERVATIVE, Scenario.CENTRAL, Scenario.OPTIMISTIC):
        leaves = [
            (leaf, s) for leaf in valuation.leaves for s in leaf.scenarios if s.scenario is scenario
        ]
        total = sum(s.credit_raw for _, s in leaves)
        tasks = {
            (leaf.task.root, leaf.task.turn, leaf.task.branch) for leaf, s in leaves if s.credit_raw
        }
        missing = len({m for _, s in leaves for m in s.missing})
        rows.append(
            ScenarioLeverage(
                scenario,
                Quantity(
                    to_unit_value(total, unit),
                    unit.unit,
                    Basis.ESTIMATED,
                    f"{LEVERAGE_VERSION}:{methodology.hash}",
                ),
                len(tasks),
                missing,
            )
        )
    mine = [i for i in investments if i.note_id == note_id]
    known = [i.cost for i in mine if i.cost.value is not None]
    summary = sum_quantities([i.cost for i in mine]) if mine else None
    return NoteLeverage(
        note_id,
        unit,
        tuple(rows),
        summary.known if summary and known else None,
        len(mine),
        len(known),
        summary.reasons if summary else ("note_creation_cost_unobserved",),
    )


@dataclass(frozen=True)
class OverheadEvent:
    """One overhead cost with only the evidence that tags its purpose."""

    event_id: str
    kind: str  # protocol | index | collector
    cost: Quantity
    explicit_tag: str | None = None  # from explicit invocation metadata
    sequence_evidence: bool = False  # deterministic sequence evidence (e.g. handshake)


@dataclass(frozen=True)
class OverheadView:
    kind: str
    purpose: Purpose
    known: int
    events: int
    missing: int
    unit: str


def purpose_of(event: OverheadEvent) -> Purpose:
    """Protocol only from explicit metadata or deterministic sequence; else unknown."""
    if event.kind == "protocol" and (event.explicit_tag == "protocol" or event.sequence_evidence):
        return Purpose.PROTOCOL
    return Purpose.UNKNOWN


def overhead_views(events: Iterable[OverheadEvent]) -> tuple[OverheadView, ...]:
    """Per (kind, purpose) known subtotals; overhead is separate from lookup benefit.

    ``purpose=unknown`` entries stay in the debit: tagging only ever labels cost, it
    never removes it from a net.
    """
    groups: dict[tuple[str, Purpose, str], list[OverheadEvent]] = {}
    for event in sorted(events, key=lambda e: e.event_id):
        groups.setdefault((event.kind, purpose_of(event), event.cost.unit), []).append(event)
    views = []
    for (kind, purpose, unit), group in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1].value, kv[0][2])
    ):
        known = sum(e.cost.value for e in group if e.cost.value is not None)
        missing = sum(1 for e in group if e.cost.value is None)
        views.append(OverheadView(kind, purpose, known, len(group), missing, unit))
    return tuple(views)
