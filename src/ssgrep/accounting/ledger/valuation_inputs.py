"""Typed inputs and credit-footprint helpers for the valuation core."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ssgrep.accounting.ledger.contracts import (
    THRESHOLD_MILLI,
    Quantity,
    Reuse,
    Scenario,
    TaskId,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_contracts import SourceAtom
from ssgrep.accounting.ledger.units import (
    CHARS_DIV4_V1,
    CONTEXT_TOKENS,
    NS_PER_MS,
    NS_PER_SECOND,
    WORK_SECONDS,
    UnitSpec,
    merge_intervals,
    subtract_intervals,
)

CURRENT_VALUATION_VERSION = "honest_bounds_ns_v4"
RETIRED_VALUATION_VERSION = "honest_bounds_ns_v3"
READABLE_VALUATION_VERSIONS = frozenset((RETIRED_VALUATION_VERSION, CURRENT_VALUATION_VERSION))
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
if TYPE_CHECKING:
    from ssgrep.accounting.ledger.valuation import Allocation
# Consolidated from valuation_inputs.py; public data structures and helpers retained.


class Eligibility(StrEnum):
    """A known rejection is zero credit; unavailable evidence is never zero."""

    SUPPORTED = "supported"
    KNOWN_REJECTED = "known_rejected"
    UNAVAILABLE = "unavailable"


class AtomStatus(StrEnum):
    """Result of verifying one source atom against its resolved content."""

    OK = "ok"
    UNAVAILABLE = "unavailable"  # source cannot be read: cost unknown, not zero
    DIGEST_CONFLICT = "digest_conflict"  # resolved content differs from recorded digest
    SELF_LATER_WORK = "self_later_work"  # consumer's own later work is not a source
    CYCLE = "cycle"  # provenance cycle
    OUT_OF_SCOPE = "out_of_scope"  # outside the profile's approved workspaces


# Statuses whose credit is unknown (coverage gap) rather than invalid (excluded).
UNKNOWN_COST_STATUSES = frozenset({AtomStatus.UNAVAILABLE, AtomStatus.OUT_OF_SCOPE})


@dataclass(frozen=True)
class ValuationAtom:
    """A verified source span: ``start``/``end`` are char offsets in its origin event.

    Overlapping atoms of one ``(origin_event, digest)`` are unioned by offset, so an
    enclosing episode that contains an immediate step never double counts characters.
    ``t_start``/``t_end`` (epoch seconds) do the same for work-seconds credit.
    ``tokens`` serves an explicitly pinned exact tokenizer unit at atom granularity.
    """

    atom: SourceAtom
    start: int = 0
    end: int | None = None
    t_start: int | None = None
    t_end: int | None = None
    tokens: int | None = None  # exact-tokenizer count of the whole atom (atom granularity)
    status: AtomStatus = AtomStatus.OK
    work_method: str = "capped_gap_v1"
    cost_reason: str = ""
    actor: str = ""  # source runtime/session identity, never a consumer actor
    t_start_ms: int | None = None  # exact capped-gap endpoints, preferred to seconds
    t_end_ms: int | None = None

    @property
    def atom_id(self) -> str:
        return self.atom.atom_id

    @property
    def char_end(self) -> int:
        return self.start + self.atom.chars if self.end is None else self.end

    @property
    def usable(self) -> bool:
        return self.status is AtomStatus.OK


@dataclass(frozen=True)
class Candidate:
    """One candidate reuse of a source for one consumer task.

    ``immediate``/``discovery``/``episode`` are the three provenance views (design
    D6); which of them earn credit in which scenario is decided by the valuation
    core from the evidence flags below, never by the caller.
    """

    candidate_id: str  # candidate fingerprint
    candidate_type: str  # identifier | path | flag | error_code
    task: TaskId
    day: str  # UTC YYYY-MM-DD of the exposure/action
    reuse: Reuse
    novel: bool
    user_supplied: bool
    action_outcome_ok: bool
    immediate: tuple[ValuationAtom, ...] = ()
    discovery: tuple[ValuationAtom, ...] = ()
    episode: tuple[ValuationAtom, ...] = ()
    explicit_relay: bool = False
    report_reference: bool = False
    exposure_ids: tuple[str, ...] = ()
    execution_ids: tuple[str, ...] = ()
    witness_locators: tuple[str, ...] = ()
    action_locators: tuple[str, ...] = ()
    accepted_day: str = ""  # UTC day the evidence was accepted; "" means the event day
    eligibility: Eligibility = Eligibility.SUPPORTED
    eligibility_reasons: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()
    rarity_lower_bound_milli: int | None = (
        None  # superset df proves only thresholds at/below this bound
    )

    @property
    def accepted(self) -> str:
        return max(self.day, self.accepted_day)


@dataclass(frozen=True)
class Debit:
    """One supported consumer cost (an exposed response, generation, remediation).

    ``amount`` is unavailable with a reason when the cost cannot be estimated.
    ``partial`` describes exposure coverage, not measurement availability: an
    exactly observed truncated response still has a known debit. Inseparable
    mixed output uses an unavailable amount. Distinct exposures retain distinct
    ``debit_id`` values.
    """

    debit_id: str
    task: TaskId
    day: str
    kind: str  # exposure | generation | remediation
    unit: UnitSpec
    amount: Quantity
    partial: bool = False
    exposure_id: str | None = None
    chars: int | None = None  # observed characters; keeps sum-before-divide exact
    accepted_day: str = ""  # UTC day the evidence was accepted; "" means the event day
    raw_min: int = 0  # missing lower bound; chars, capped-gap ns, otherwise units
    raw_max: int | None = None  # None is unbounded, NOT a zero upper bound
    work_ns: int | None = (
        None  # authoritative modeled ns; never reconstructed from rounded seconds/ms
    )
    mapping_version: str = ""
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.raw_min < 0 or (self.raw_max is not None and self.raw_max < self.raw_min):
            raise ValueError("invalid nonnegative debit bounds")

    @property
    def accepted(self) -> str:
        return max(self.day, self.accepted_day)


@dataclass(frozen=True)
class Methodology:
    """Everything that must match for estimates to be summed (a version namespace)."""

    label: str
    units: tuple[UnitSpec, ...]
    thresholds: tuple[tuple[str, int], ...] = (
        ("conservative", 8),
        ("central", 6),
        ("optimistic", 4),
    )
    frequency_view: str = ""
    tokenizer: str = ""
    pricing: str = ""
    calibrated: bool = False
    valuation_version: str = CURRENT_VALUATION_VERSION

    @property
    def hash(self) -> str:
        return digest_of(self)[:32]


@dataclass(frozen=True)
class EvidenceBundle:
    """Retained frozen evidence for reduction; set semantics, order irrelevant."""

    candidates: tuple[Candidate, ...] = ()
    debits: tuple[Debit, ...] = ()

    def accepted_through(self, day: str) -> EvidenceBundle:
        """The evidence a collector holding everything accepted up to ``day`` has."""
        return EvidenceBundle(
            tuple(c for c in self.candidates if c.accepted <= day),
            tuple(d for d in self.debits if d.accepted <= day),
        )

    @property
    def last_accepted(self) -> str:
        days = [c.accepted for c in self.candidates] + [d.accepted for d in self.debits]
        return max(days, default="")


# Consolidated from valuation_credit.py; public data structures and helpers retained.


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
