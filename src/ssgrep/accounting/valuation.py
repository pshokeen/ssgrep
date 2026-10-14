"""Overlap-free, nested, identically-debited valuation (design D6; pure, no I/O).

Per task/scenario/unit, eligible atoms are unioned BEFORE summing; every scenario
subtracts the same compatible debit set; unknown costs keep a quantified subtotal
and suppress the complete net; incomparable units/methodologies are refused.
Leaves (one per task/day/unit) are the only thing rollups ever add.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ssgrep.accounting.contracts import Basis, Quantity, Scenario, TaskId, digest_of
from ssgrep.accounting.ledger_contracts import ScenarioEstimate
from ssgrep.accounting.units import CHARS_DIV4_V1, UnitMismatchError, UnitSpec
from ssgrep.accounting.valuation_credit import (
    SCENARIOS,
    Footprint,
    InvariantError,
    ScenarioCredit,
    assert_nested,
    credit_scenario,
    footprint_size,
    intersect_footprint,
    merge_footprints,
)
from ssgrep.accounting.valuation_inputs import (
    Candidate,
    Debit,
    EvidenceBundle,
    Methodology,
)

ESTIMATED_METHOD_PREFIX = "estimate:"


@dataclass(frozen=True)
class ScenarioLeaf:
    scenario: Scenario
    credit_raw: int  # chars / tokens / seconds, summed BEFORE any unit division
    known_atoms: int
    missing: tuple[tuple[str, str], ...]  # (atom id, reason): unknown source cost


@dataclass(frozen=True)
class Leaf:
    """One unique contribution: a task (or task_unknown root), UTC day, and unit."""

    task: TaskId
    day: str
    unit: UnitSpec
    methodology: str
    scenarios: tuple[ScenarioLeaf, ...]
    debit_raw: int
    debit_known: int
    debit_missing: tuple[tuple[str, str], ...]  # unknown or partial debits

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        t = self.task
        return (t.root, t.turn or "", t.branch, self.day, self.unit.unit, self.unit.method)


@dataclass(frozen=True)
class UnitEstimate:
    """One scenario of a rollup: known subtotals, coverage, and a guarded net."""

    scenario: Scenario
    unit: UnitSpec
    credit: Quantity  # known subtotal (never presented as complete when not)
    debit: Quantity
    net: Quantity  # unavailable whenever credit or debit is incomplete
    known_net: int  # credit - debit over KNOWN parts; negative values preserved
    credit_missing: tuple[tuple[str, str], ...]
    debit_missing: tuple[tuple[str, str], ...]
    known_atoms: int
    known_debits: int

    @property
    def complete(self) -> bool:
        return not self.credit_missing and not self.debit_missing

    def as_scenario_estimate(self) -> ScenarioEstimate:
        return ScenarioEstimate(self.scenario, self.credit, self.debit, self.net)


@dataclass(frozen=True)
class Allocation:
    """Atoms one candidate is the first to claim, per scenario (receipt input)."""

    task: TaskId
    candidate_id: str
    day: str
    unit: UnitSpec
    scenario: Scenario
    ineligible: str | None
    eligible: tuple[str, ...]
    atoms: tuple[str, ...]
    credit_raw: int
    missing: tuple[tuple[str, str], ...]
    excluded: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Valuation:
    methodology: str
    leaves: tuple[Leaf, ...]
    allocations: tuple[Allocation, ...]


def to_unit_value(raw: int, unit: UnitSpec) -> int:
    return raw // 4 if unit.method == CHARS_DIV4_V1 else raw


def _quantity(raw: int, unit: UnitSpec, label: str) -> Quantity:
    return Quantity(
        to_unit_value(raw, unit), unit.unit, Basis.ESTIMATED, f"{ESTIMATED_METHOD_PREFIX}{label}"
    )


def debit_raw(debit: Debit, unit: UnitSpec) -> tuple[int | None, str]:
    """Raw debit in ``unit``; unknown carries its reason (never zero)."""
    if debit.unit != unit:
        raise UnitMismatchError(f"debit {debit.debit_id} is {debit.unit}, not {unit}")
    if unit.method == CHARS_DIV4_V1:
        if debit.chars is not None:
            return debit.chars, ""
        return None, debit.amount.reason or "debit_chars_unavailable"
    if debit.amount.value is None:
        return None, debit.amount.reason
    return debit.amount.value, ""


def _unique_debits(debits: Iterable[Debit]) -> list[Debit]:
    """One debit per id; conflicting duplicates become unknown, never double counted."""
    by_id: dict[str, list[Debit]] = {}
    for debit in debits:
        by_id.setdefault(debit.debit_id, []).append(debit)
    out: list[Debit] = []
    for _, group in sorted(by_id.items()):
        variants = {digest_of(d): d for d in group}
        if len(variants) == 1:
            out.append(next(iter(variants.values())))
        else:
            first = variants[min(variants)]
            conflict = Quantity.unavailable(first.unit.unit, "conflicting_duplicate_debit")
            out.append(
                Debit(first.debit_id, first.task, first.day, first.kind, first.unit, conflict)
            )
    return out


def _unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return [c for _, c in sorted({digest_of(c): c for c in candidates}.items())]


def _task_key(task: TaskId) -> tuple[str, str, str]:
    return (task.root, task.turn or "", task.branch)


@dataclass(frozen=True)
class _TaskCredit:
    """Per scenario/day raw credit of one task, plus the whole-task scenario credits."""

    per_day: dict[Scenario, dict[str, tuple[int, int, tuple[tuple[str, str], ...]]]]
    whole: dict[Scenario, ScenarioCredit]
    days: list[str]


def _task_credit(cands: list[Candidate], unit: UnitSpec, base: Footprint) -> _TaskCredit:
    """Credit one task: union across ALL its days first, then attribute fresh coverage
    to the first day that reaches it, so day leaves partition the task total."""
    whole = {s: credit_scenario(cands, s, unit, base) for s in SCENARIOS}
    assert_nested({s: whole[s].atom_ids for s in SCENARIOS})
    sizes = [footprint_size(whole[s].footprint) for s in SCENARIOS]
    if sizes != sorted(sizes):
        raise InvariantError("scenario credit is not nested")
    days = sorted({c.day for c in cands})
    claimed = dict(base)
    regions: dict[str, Footprint] = {}
    for day in days:
        todays = [c for c in cands if c.day == day]
        fresh = credit_scenario(todays, Scenario.OPTIMISTIC, unit, claimed).footprint
        regions[day] = fresh
        claimed = merge_footprints([claimed, fresh])
    per_day: dict[Scenario, dict[str, tuple[int, int, tuple[tuple[str, str], ...]]]] = {}
    for scenario in SCENARIOS:
        seen_atoms: set[str] = set()
        seen_missing: set[tuple[str, str]] = set()
        per_day[scenario] = {}
        for day in days:
            part = credit_scenario([c for c in cands if c.day == day], scenario, unit)
            new_atoms = set(part.atom_ids) - seen_atoms
            seen_atoms |= new_atoms
            missing = {m for e in part.entries for m in e.missing} - seen_missing
            seen_missing |= missing
            raw = footprint_size(intersect_footprint(whole[scenario].footprint, regions[day]))
            per_day[scenario][day] = (raw, len(new_atoms), tuple(sorted(missing)))
    return _TaskCredit(per_day, whole, days)


def _allocations(task: TaskId, unit: UnitSpec, whole: dict) -> list[Allocation]:
    return [
        Allocation(
            task,
            e.candidate_id,
            e.day,
            unit,
            s,
            e.ineligible,
            e.eligible,
            e.atoms,
            e.credit_raw,
            e.missing,
            e.excluded,
        )
        for s in SCENARIOS
        for e in whole[s].entries
    ]


def value_evidence(bundle: EvidenceBundle, methodology: Methodology) -> Valuation:
    """Reduce frozen evidence to unique leaves; independent of input order."""
    units = set(methodology.units)
    candidates = _unique_candidates(bundle.candidates)
    debits = _unique_debits(bundle.debits)
    for debit in debits:
        if debit.unit not in units:
            raise UnitMismatchError(f"debit {debit.debit_id} unit {debit.unit} not in methodology")
    groups: dict[tuple[str, str, str], list[Candidate]] = {}
    for cand in candidates:
        groups.setdefault(_task_key(cand.task), []).append(cand)
    leaves: list[Leaf] = []
    allocations: list[Allocation] = []
    mhash = methodology.hash
    for unit in sorted(
        units, key=lambda u: (u.unit, u.method, u.tokenizer or "", u.currency or "")
    ):
        credit: dict[tuple[str, str, str], _TaskCredit] = {}
        resolved_base: dict[str, Footprint] = {}
        for key in sorted(k for k in groups if k[1] != ""):  # resolved tasks first
            credit[key] = _task_credit(groups[key], unit, {})
            fp = credit[key].whole[Scenario.OPTIMISTIC].footprint
            resolved_base[key[0]] = merge_footprints([resolved_base.get(key[0], {}), fp])
        for key in sorted(k for k in groups if k[1] == ""):  # task_unknown: dedupe at root
            base = resolved_base.get(key[0], {})
            credit[key] = _task_credit(groups[key], unit, base)
        debit_days: dict[tuple[tuple[str, str, str], str], list[Debit]] = {}
        for debit in debits:
            if debit.unit == unit:
                debit_days.setdefault((_task_key(debit.task), debit.day), []).append(debit)
        keys = {(k, d) for k, tc in credit.items() for d in tc.days}
        keys |= set(debit_days)
        for key, day in sorted(keys):
            task = TaskId(key[0], key[1] or None, key[2])
            tc = credit.get(key)
            per = tc.per_day if tc else {}
            scen = tuple(ScenarioLeaf(s, *per.get(s, {}).get(day, (0, 0, ()))) for s in SCENARIOS)
            known, known_n, missing = 0, 0, []
            for d in debit_days.get((key, day), ()):
                raw, reason = debit_raw(d, unit)
                if raw is None:
                    missing.append((d.debit_id, reason))
                    continue
                known += raw
                known_n += 1
            leaves.append(
                Leaf(task, day, unit, mhash, scen, known, known_n, tuple(sorted(missing)))
            )
        for key, tc in credit.items():
            task = TaskId(key[0], key[1] or None, key[2])
            allocations.extend(_allocations(task, unit, tc.whole))
    return Valuation(mhash, tuple(leaves), tuple(allocations))


def rollup(leaves: Iterable[Leaf], unit: UnitSpec) -> tuple[UnitEstimate, ...]:
    """Sum UNIQUE leaves of one unit/methodology; refuse anything incomparable.

    Identical duplicate leaves count once; conflicting leaves with one key are an
    error. Task and root totals are never inputs, so they cannot be added together.
    """
    unique: dict[tuple, Leaf] = {}
    for leaf in leaves:
        if leaf.unit != unit:
            raise UnitMismatchError(f"leaf unit {leaf.unit} differs from {unit}")
        held = unique.setdefault(leaf.key, leaf)
        if held != leaf:
            raise ValueError(f"conflicting leaves for {leaf.key}")
    methodologies = {leaf.methodology for leaf in unique.values()}
    if len(methodologies) > 1:
        raise UnitMismatchError("leaves from different methodology namespaces")
    label = next(iter(methodologies), "")
    debit_total = sum(leaf.debit_raw for leaf in unique.values())
    debit_known = sum(leaf.debit_known for leaf in unique.values())
    debit_missing = tuple(sorted({m for leaf in unique.values() for m in leaf.debit_missing}))
    result: list[UnitEstimate] = []
    for scenario in SCENARIOS:
        rows = [s for leaf in unique.values() for s in leaf.scenarios if s.scenario is scenario]
        credit_total = sum(r.credit_raw for r in rows)
        missing = tuple(sorted({m for r in rows for m in r.missing}))
        known_net = to_unit_value(credit_total, unit) - to_unit_value(debit_total, unit)
        if missing or debit_missing:
            reasons = sorted({r for _, r in missing + debit_missing})
            net = Quantity.unavailable(unit.unit, "incomplete:" + ",".join(reasons))
        else:
            net = Quantity(
                known_net, unit.unit, Basis.ESTIMATED, f"{ESTIMATED_METHOD_PREFIX}{label}"
            )
        result.append(
            UnitEstimate(
                scenario,
                unit,
                _quantity(credit_total, unit, label),
                _quantity(debit_total, unit, label),
                net,
                known_net,
                missing,
                debit_missing,
                sum(r.known_atoms for r in rows),
                debit_known,
            )
        )
    _assert_nested_credits(result)
    return tuple(result)


def _assert_nested_credits(estimates: list[UnitEstimate]) -> None:
    credits = [e.credit.value or 0 for e in estimates]
    if credits != sorted(credits):
        raise InvariantError("scenario credits are not nested")
