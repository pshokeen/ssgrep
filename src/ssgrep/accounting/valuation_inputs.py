"""Typed, file-free inputs to the valuation core (design D6/D7).

Everything here is plain frozen data: no clocks, no I/O. Callers (reconciliation,
evidence, source resolution) construct these from verified evidence; the
valuation core only does integer arithmetic over them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ssgrep.accounting.contracts import Quantity, Reuse, TaskId, digest_of
from ssgrep.accounting.ledger_contracts import SourceAtom
from ssgrep.accounting.units import UnitSpec


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
    valuation_version: str = "honest_bounds_ns_v3"

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
