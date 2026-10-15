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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssgrep.accounting.valuation import Allocation

from ssgrep.accounting.contracts import Reuse, Scenario
from ssgrep.accounting.frequency import THRESHOLD_MILLI
from ssgrep.accounting.timeunits import (
    NS_PER_MS,
    NS_PER_SECOND,
    merge_intervals,
    subtract_intervals,
)
from ssgrep.accounting.units import CHARS_DIV4_V1, CONTEXT_TOKENS, WORK_SECONDS, UnitSpec
from ssgrep.accounting.valuation_inputs import (
    UNKNOWN_COST_STATUSES,
    Candidate,
    Eligibility,
    ValuationAtom,
)

SCENARIOS = (Scenario.CONSERVATIVE, Scenario.CENTRAL, Scenario.OPTIMISTIC)
_ACTED = frozenset({Reuse.ACTION_REFERENCE, Reuse.VALIDATED})
_NO_CREDIT_REASON = (
    (lambda c: c.reuse is Reuse.CONTRADICTED, "contradicted"),
    (lambda c: c.user_supplied, "user_supplied"),
    (lambda c: not c.novel, "not_novel"),
    (
        lambda c: c.reuse not in _ACTED and not c.explicit_relay and not c.report_reference,
        "no_supported_reuse",
    ),
)


class InvariantError(AssertionError):
    """A scenario nesting invariant was violated; never expected, never swallowed."""


def exclusion_reason(candidate: Candidate) -> str | None:
    """Why a candidate earns no discovery credit in ANY scenario, else None."""
    if candidate.eligibility is Eligibility.KNOWN_REJECTED:
        return ",".join(candidate.eligibility_reasons) or "known_rejected"
    for test, reason in _NO_CREDIT_REASON:
        if test(candidate):
            return reason
    return None


def scenario_atoms(candidate: Candidate) -> dict[Scenario, tuple[ValuationAtom, ...]]:
    """Cumulative atoms per scenario for one candidate (empty when ineligible)."""
    empty: tuple[ValuationAtom, ...] = ()
    if exclusion_reason(candidate) is not None:
        return dict.fromkeys(SCENARIOS, empty)
    if candidate.eligibility is Eligibility.UNAVAILABLE:
        return dict.fromkeys(SCENARIOS, empty)
    referenced = candidate.reuse in _ACTED
    conservative = candidate.immediate if referenced and candidate.action_outcome_ok else empty
    central = conservative + (candidate.discovery if referenced else empty)
    broad = referenced or candidate.explicit_relay or candidate.report_reference
    optimistic = central + (candidate.episode if broad else empty)
    selected = {
        Scenario.CONSERVATIVE: conservative,
        Scenario.CENTRAL: central,
        Scenario.OPTIMISTIC: optimistic,
    }
    return {s: atoms if rarity_proven(candidate, s) else empty for s, atoms in selected.items()}


def rarity_proven(candidate: Candidate, scenario: Scenario) -> bool:
    lower = candidate.rarity_lower_bound_milli
    threshold = THRESHOLD_MILLI[scenario]
    return lower is None or lower >= threshold


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
            usable[
                (
                    atom.atom_id,
                    atom.start,
                    atom.char_end,
                    atom.t_start,
                    atom.t_end,
                    atom.t_start_ms,
                    atom.t_end_ms,
                    atom.work_method,
                    atom.actor,
                )
            ] = atom
        elif atom.status in UNKNOWN_COST_STATUSES:
            missing.add((atom.atom_id, atom.cost_reason or atom.status.value))
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
            measured = (
                atom.atom.seconds is not None
                or (atom.t_start is not None and atom.t_end is not None)
                or (atom.t_start_ms is not None and atom.t_end_ms is not None)
            )
            if measured and atom.work_method != unit.method:
                unknown.add((atom.atom_id, "source_work_method_incompatible"))
                continue
            scale = NS_PER_SECOND if unit.method == "capped_gap_v1" else 1
            work_key = ("actor", atom.actor) if atom.actor else key
            has_ms = atom.t_start_ms is not None or atom.t_end_ms is not None
            if unit.method == "capped_gap_v1" and has_ms:
                if atom.t_start_ms is None or atom.t_end_ms is None:
                    unknown.add((atom.atom_id, atom.cost_reason or "source_seconds_unavailable"))
                elif atom.t_end_ms < atom.t_start_ms:
                    unknown.add((atom.atom_id, "source_clock_reversed"))
                else:
                    # Exact dimensional conversion only; native observations remain ms.
                    spans.setdefault(work_key, []).append(
                        (atom.t_start_ms * NS_PER_MS, atom.t_end_ms * NS_PER_MS)
                    )
            elif atom.t_start is not None and atom.t_end is not None:
                if atom.t_end < atom.t_start:
                    unknown.add((atom.atom_id, "source_clock_reversed"))
                else:
                    spans.setdefault(work_key, []).append(
                        (atom.t_start * scale, atom.t_end * scale)
                    )
            elif atom.atom.seconds is not None:
                spans.setdefault(("atom", atom.atom_id), []).append((0, atom.atom.seconds * scale))
            else:
                unknown.add((atom.atom_id, atom.cost_reason or "source_seconds_unavailable"))
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
    footprint: Footprint  # owned ranges, retained by this candidate in broader scenarios


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
    previous: ScenarioCredit | None = None,
) -> ScenarioCredit:
    """Union before summing, retaining narrower-scenario ownership per candidate/day.

    ``base`` was credited elsewhere. ``previous`` reserves only actual lower-band
    allocations: broader candidates may claim new ranges but cannot steal those
    shares. Every inherited range must still exist in its owner's selected atoms.
    """
    base_fp: Footprint = base or {}
    claimed = merge_footprints([base_fp, previous.footprint if previous else {}])
    remaining: dict[tuple[str, str], Footprint] = {}
    for entry in previous.entries if previous else ():
        key = (entry.candidate_id, entry.day)
        remaining[key] = merge_footprints([remaining.get(key, {}), entry.footprint])
    entries: list[CandidateCredit] = []
    all_ids: set[str] = set()
    for candidate in sorted(candidates, key=lambda c: (c.day, c.candidate_id)):
        selection = select(scenario_atoms(candidate)[scenario])
        fp, unknown = footprint(selection.usable, unit)
        key = (candidate.candidate_id, candidate.day)
        inherited = intersect_footprint(fp, remaining.get(key, {}))
        remaining[key] = subtract_footprint(remaining.get(key, {}), inherited)
        allocated = merge_footprints([inherited, subtract_footprint(fp, claimed)])
        first = {
            atom.atom_id
            for atom in selection.usable
            if intersect_footprint(footprint([atom], unit)[0], allocated)
        }
        gate_missing = (
            tuple(
                (f"candidate:{candidate.candidate_id}", reason)
                for reason in candidate.eligibility_reasons or ("eligibility_unavailable",)
            )
            if candidate.eligibility is Eligibility.UNAVAILABLE
            and exclusion_reason(candidate) is None
            else ()
        )
        if not rarity_proven(candidate, scenario) and exclusion_reason(candidate) is None:
            gate_missing += ((f"candidate:{candidate.candidate_id}", "rarity_threshold_unproven"),)
        entries.append(
            CandidateCredit(
                candidate.candidate_id,
                candidate.day,
                exclusion_reason(candidate),
                selection.atom_ids,
                tuple(sorted(first)),
                footprint_size(allocated),
                tuple(sorted(set(selection.missing) | set(unknown) | set(gate_missing))),
                selection.excluded,
                allocated,
            )
        )
        all_ids.update(selection.atom_ids)
        claimed = merge_footprints([claimed, fp])
    if any(remaining.values()):
        raise InvariantError("broader scenario lost inherited candidate coverage")
    return ScenarioCredit(
        subtract_footprint(claimed, base_fp), tuple(entries), tuple(sorted(all_ids))
    )


def eligibility_for(
    candidate: Candidate, entries: Iterable[CandidateCredit | Allocation]
) -> Eligibility:
    rows = [e for e in entries if e.candidate_id == candidate.candidate_id]
    if exclusion_reason(candidate):
        return Eligibility.KNOWN_REJECTED
    if any(e.missing for e in rows):
        return Eligibility.UNAVAILABLE
    if any(e.excluded for e in rows) and not any(e.eligible for e in rows):
        return Eligibility.KNOWN_REJECTED
    return candidate.eligibility
