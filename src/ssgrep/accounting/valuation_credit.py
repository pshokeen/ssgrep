"""Nested scenario atom selection and overlap-free credit footprints (design D6).

``scenario_atoms`` builds the conservative <= central <= optimistic atom sets
*cumulatively*, so the subset relation holds by construction even when immediate
step, discovery span, and episode boundaries differ. A :class:`Footprint` is the
union of what the atoms cover in one unit, so a shared source step is credited
once however many facts, references, or result cards expose it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ssgrep.accounting.contracts import Reuse, Scenario
from ssgrep.accounting.timeunits import merge_intervals, subtract_intervals
from ssgrep.accounting.units import CHARS_DIV4_V1, CONTEXT_TOKENS, WORK_SECONDS, UnitSpec
from ssgrep.accounting.valuation_inputs import (
    UNKNOWN_COST_STATUSES,
    Candidate,
    ValuationAtom,
)

SCENARIOS = (Scenario.CONSERVATIVE, Scenario.CENTRAL, Scenario.OPTIMISTIC)
_ACTED = frozenset({Reuse.ACTION_REFERENCE, Reuse.VALIDATED})
_NO_CREDIT_REASON = (
    (lambda c: c.reuse is Reuse.CONTRADICTED, "contradicted"),
    (lambda c: c.user_supplied, "user_supplied"),
    (lambda c: not c.novel, "not_novel"),
)


class InvariantError(AssertionError):
    """A scenario nesting invariant was violated; never expected, never swallowed."""


def exclusion_reason(candidate: Candidate) -> str | None:
    """Why a candidate earns no discovery credit in ANY scenario, else None."""
    for test, reason in _NO_CREDIT_REASON:
        if test(candidate):
            return reason
    return None


def scenario_atoms(candidate: Candidate) -> dict[Scenario, tuple[ValuationAtom, ...]]:
    """Cumulative atoms per scenario for one candidate (empty when ineligible)."""
    empty: tuple[ValuationAtom, ...] = ()
    if exclusion_reason(candidate) is not None:
        return dict.fromkeys(SCENARIOS, empty)
    referenced = candidate.reuse in _ACTED
    conservative = candidate.immediate if referenced and candidate.action_outcome_ok else empty
    central = conservative + (candidate.discovery if referenced else empty)
    broad = referenced or candidate.explicit_relay or candidate.report_reference
    optimistic = central + (candidate.episode if broad else empty)
    return {
        Scenario.CONSERVATIVE: conservative,
        Scenario.CENTRAL: central,
        Scenario.OPTIMISTIC: optimistic,
    }


def assert_nested(sets: dict[Scenario, Iterable[str]]) -> None:
    """Raise unless conservative <= central <= optimistic (as atom-id sets)."""
    low, mid, high = (frozenset(sets[s]) for s in SCENARIOS)
    if not (low <= mid <= high):
        raise InvariantError("scenario atom sets are not nested")


@dataclass(frozen=True)
class AtomSelection:
    """Verified atoms of one scenario, with what could not be credited and why."""

    usable: tuple[ValuationAtom, ...]
    missing: tuple[tuple[str, str], ...]  # unknown source cost: (atom_id, reason)
    excluded: tuple[tuple[str, str], ...]  # invalid provenance: (atom_id, reason)

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(sorted({a.atom_id for a in self.usable}))


def select(atoms: Iterable[ValuationAtom]) -> AtomSelection:
    usable: dict[tuple, ValuationAtom] = {}
    missing: set[tuple[str, str]] = set()
    excluded: set[tuple[str, str]] = set()
    for atom in atoms:
        if atom.usable:
            usable[(atom.atom_id, atom.start, atom.char_end, atom.t_start, atom.t_end)] = atom
        elif atom.status in UNKNOWN_COST_STATUSES:
            missing.add((atom.atom_id, atom.status.value))
        else:
            excluded.add((atom.atom_id, atom.status.value))
    ordered = tuple(usable[k] for k in sorted(usable, key=repr))
    return AtomSelection(ordered, tuple(sorted(missing)), tuple(sorted(excluded)))


Footprint = dict[tuple[str, str], tuple[tuple[int, int], ...]]


def footprint(
    atoms: Iterable[ValuationAtom], unit: UnitSpec
) -> tuple[Footprint, tuple[tuple[str, str], ...]]:
    """Union of covered ranges per source key, plus atoms whose cost is unknown here."""
    spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    unknown: set[tuple[str, str]] = set()
    for atom in atoms:
        key = (atom.atom.origin_event, atom.atom.digest)
        if unit.unit == CONTEXT_TOKENS and unit.method == CHARS_DIV4_V1:
            spans.setdefault(key, []).append((atom.start, atom.char_end))
        elif unit.unit == CONTEXT_TOKENS and unit.tokenizer:
            if atom.tokens is None:
                unknown.add((atom.atom_id, "exact_tokens_unavailable"))
            else:
                spans.setdefault(("atom", atom.atom_id), []).append((0, atom.tokens))
        elif unit.unit == WORK_SECONDS:
            if atom.t_start is not None and atom.t_end is not None:
                spans.setdefault(key, []).append((atom.t_start, atom.t_end))
            elif atom.atom.seconds is not None:
                spans.setdefault(("atom", atom.atom_id), []).append((0, atom.atom.seconds))
            else:
                unknown.add((atom.atom_id, "source_seconds_unavailable"))
        else:
            raise ValueError(f"unsupported credit unit {unit}")
    merged = {k: merge_intervals(v) for k, v in spans.items()}
    return merged, tuple(sorted(unknown))


def merge_footprints(parts: Iterable[Footprint]) -> Footprint:
    combined: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for part in parts:
        for key, intervals in part.items():
            combined.setdefault(key, []).extend(intervals)
    return {k: merge_intervals(v) for k, v in combined.items()}


def subtract_footprint(base: Footprint, removed: Footprint) -> Footprint:
    out: Footprint = {}
    for key, intervals in base.items():
        rest = subtract_intervals(intervals, removed.get(key, ()))
        if rest:
            out[key] = rest
    return out


def footprint_size(fp: Footprint) -> int:
    """Characters, tokens, or seconds covered (raw, before any unit division)."""
    return sum(end - start for intervals in fp.values() for start, end in intervals)


@dataclass(frozen=True)
class CandidateCredit:
    """What one candidate first claims in one scenario, and what it could not credit."""

    candidate_id: str
    day: str
    ineligible: str | None  # reason it earns no discovery credit in any scenario
    eligible: tuple[str, ...]  # every usable atom of this candidate in the scenario
    atoms: tuple[str, ...]  # atoms this candidate is the first to claim (allocation)
    credit_raw: int  # fresh coverage this candidate adds (chars/tokens/seconds, undivided)
    missing: tuple[tuple[str, str], ...]
    excluded: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ScenarioCredit:
    """Overlap-free credit of one scenario for one set of candidates in one unit."""

    footprint: Footprint  # fresh coverage only (base already removed)
    entries: tuple[CandidateCredit, ...]
    atom_ids: tuple[str, ...]  # every usable atom of the scenario (nesting check)


def intersect_footprint(a: Footprint, b: Footprint) -> Footprint:
    return subtract_footprint(a, subtract_footprint(a, b))


def credit_scenario(
    candidates: Iterable[Candidate],
    scenario: Scenario,
    unit: UnitSpec,
    base: Footprint | None = None,
) -> ScenarioCredit:
    """Union atoms across candidates BEFORE summing; ``base`` is coverage already
    credited elsewhere (task_unknown dedupe) and contributes nothing here."""
    base_fp: Footprint = base or {}
    claimed = dict(base_fp)
    entries: list[CandidateCredit] = []
    all_ids: set[str] = set()
    for candidate in sorted(candidates, key=lambda c: (c.day, c.candidate_id)):
        selection = select(scenario_atoms(candidate)[scenario])
        fp, unknown = footprint(selection.usable, unit)
        first = {
            atom.atom_id
            for atom in selection.usable
            if subtract_footprint(footprint([atom], unit)[0], claimed)
        }
        entries.append(
            CandidateCredit(
                candidate.candidate_id,
                candidate.day,
                exclusion_reason(candidate),
                selection.atom_ids,
                tuple(sorted(first)),
                footprint_size(subtract_footprint(fp, claimed)),
                tuple(sorted(set(selection.missing) | set(unknown))),
                selection.excluded,
            )
        )
        all_ids.update(selection.atom_ids)
        claimed = merge_footprints([claimed, fp])
    return ScenarioCredit(
        subtract_footprint(claimed, base_fp), tuple(entries), tuple(sorted(all_ids))
    )
