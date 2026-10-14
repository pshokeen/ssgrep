"""Matched with/without-history comparison: plan and report logic only (no live runs).

The only comparison the lab allows is B's identical L3/L4 prompt on independent identical
database clones WITH versus WITHOUT history access. The order of executions is predeclared
and derived by hashing a seed, every attempt is retained, correctness is reported before any
measured cost, and the output is purely descriptive: no inference, no interval, no
causal claim. Comparing session A (L1/L2) against session B (L3/L4) is refused outright
because those sessions solve tasks of different difficulty.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

from .gates import PASS, STATUSES as GATE_STATUSES

WITH_HISTORY = "with_history"
WITHOUT_HISTORY = "without_history"
ARMS = (WITH_HISTORY, WITHOUT_HISTORY)

BASES = ("observed", "reported", "estimated", "unavailable")
ATTEMPT_STATUSES = ("completed", "failed", "interrupted", "unknown")

CALIBRATION_REASON = (
    "repeated clones of one airline task family cannot satisfy the 50-receipt calibration gate; "
    "the method/arithmetic profile is reported separately"
)
_NO_PAIR = "no_eligible_pair: no repetition has both arms completed and correct"
FORBIDDEN_MEASURE_FRAGMENT = "a_vs_b"
_UNSAFE_AB_MESSAGE = (
    "session A (L1/L2) and session B (L3/L4) differ in task difficulty, so a cost difference "
    "between them is never savings; only identical B tasks on identical clones with versus "
    "without history access may be compared (build_plan/compare)"
)


class ForbiddenComparison(RuntimeError):
    """A comparison the lab does not allow (A versus B, or unknown arm labels)."""


class PlanError(ValueError):
    """The matched plan is malformed or does not re-derive from its declared inputs."""


@dataclass(frozen=True)
class Measurement:
    """One integer measurement. An unknown value is ``None`` and must say why."""

    value: int | None
    unit: str
    basis: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and (
            isinstance(self.value, bool) or not isinstance(self.value, int)
        ):
            raise TypeError(
                f"measurement value must be an int or None, got {type(self.value).__name__}"
            )
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("measurement unit must be a non-empty string")
        if self.basis not in BASES:
            raise ValueError(f"unknown measurement basis {self.basis!r}")
        if self.value is None and not self.reason:
            raise ValueError("an unknown measurement value requires a reason")
        if self.basis == "unavailable" and self.value is not None:
            raise ValueError("basis 'unavailable' cannot carry a value")

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "basis": self.basis, "reason": self.reason}


@dataclass(frozen=True)
class Attempt:
    arm: str
    repetition: int
    run_id: str
    clone_baseline_digest: str
    runtime: str
    model: str
    prompt_hashes: Mapping[str, str]
    correctness: Mapping[str, str]
    measured: Mapping[str, Measurement]
    status: str
    started_utc: str

    def __post_init__(self) -> None:
        if self.status not in ATTEMPT_STATUSES:
            raise ValueError(f"unknown attempt status {self.status!r}")
        for gate, outcome in self.correctness.items():
            if outcome not in GATE_STATUSES:
                raise ValueError(f"gate {gate!r} has unknown outcome {outcome!r}")
        for name, measurement in self.measured.items():
            if not isinstance(measurement, Measurement):
                raise TypeError(f"measure {name!r} must be a Measurement")
        object.__setattr__(self, "prompt_hashes", MappingProxyType(dict(self.prompt_hashes)))
        object.__setattr__(self, "correctness", MappingProxyType(dict(self.correctness)))
        object.__setattr__(self, "measured", MappingProxyType(dict(self.measured)))


@dataclass(frozen=True)
class MatchedPlan:
    repetitions: int
    slots: tuple[tuple[str, int], ...]
    baseline_digest: str
    runtime: str
    model: str
    prompt_hashes: Mapping[str, str]
    order_seed: str
    plan_digest: str
    counts_toward_calibration_gate: bool = False
    calibration_reason: str = CALIBRATION_REASON

    def __post_init__(self) -> None:
        if self.counts_toward_calibration_gate:
            raise PlanError("a matched clone plan never counts toward the calibration gate")
        object.__setattr__(self, "slots", tuple((arm, rep) for arm, rep in self.slots))
        object.__setattr__(self, "prompt_hashes", MappingProxyType(dict(self.prompt_hashes)))

    def as_dict(self) -> dict[str, Any]:
        out = _plan_payload(self)
        out["plan_digest"] = self.plan_digest
        return out


def _plan_payload(plan: MatchedPlan) -> dict[str, Any]:
    return {
        "repetitions": plan.repetitions,
        "slots": [[arm, rep] for arm, rep in plan.slots],
        "baseline_digest": plan.baseline_digest,
        "runtime": plan.runtime,
        "model": plan.model,
        "prompt_hashes": dict(plan.prompt_hashes),
        "order_seed": plan.order_seed,
        "counts_toward_calibration_gate": plan.counts_toward_calibration_gate,
        "calibration_reason": plan.calibration_reason,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _first_arm(order_seed: str, block: int) -> tuple[str, str]:
    """Hash-derived first/second arm for a block of two repetitions."""
    digest = hashlib.sha256(_canonical(["matched-order-v1", order_seed, block])).digest()
    first = ARMS[digest[0] & 1]
    return first, ARMS[1 - (digest[0] & 1)]


def _derive_slots(repetitions: int, order_seed: str) -> tuple[tuple[str, int], ...]:
    """Interleaved ABBA blocks over repetition pairs; each block's lead arm comes from a hash.

    Within a block of repetitions (r, r+1) the order is first(r), second(r), second(r+1),
    first(r+1), so for even repetition counts each arm leads exactly half the repetitions.
    """
    slots: list[tuple[str, int]] = []
    for block in range((repetitions + 1) // 2):
        first, second = _first_arm(order_seed, block)
        r0 = 2 * block + 1
        slots += [(first, r0), (second, r0)]
        if r0 + 1 <= repetitions:
            slots += [(second, r0 + 1), (first, r0 + 1)]
    return tuple(slots)


def _check_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{name} must be a non-empty string")


def build_plan(
    *,
    repetitions: int,
    baseline_digest: str,
    runtime: str,
    model: str,
    prompt_hashes: Mapping[str, str],
    order_seed: str,
) -> MatchedPlan:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise PlanError("repetitions must be an integer >= 1")
    for name, value in (
        ("baseline_digest", baseline_digest),
        ("runtime", runtime),
        ("model", model),
        ("order_seed", order_seed),
    ):
        _check_text(name, value)
    if not prompt_hashes:
        raise PlanError("prompt_hashes must not be empty")
    for key, value in prompt_hashes.items():
        _check_text("prompt hash name", key)
        _check_text(f"prompt hash {key!r}", value)
    unsigned = MatchedPlan(
        repetitions=repetitions,
        slots=_derive_slots(repetitions, order_seed),
        baseline_digest=baseline_digest,
        runtime=runtime,
        model=model,
        prompt_hashes=prompt_hashes,
        order_seed=order_seed,
        plan_digest="",
    )
    return _with_digest(unsigned)


def _with_digest(plan: MatchedPlan) -> MatchedPlan:
    digest = hashlib.sha256(_canonical(_plan_payload(plan))).hexdigest()
    return MatchedPlan(
        repetitions=plan.repetitions,
        slots=plan.slots,
        baseline_digest=plan.baseline_digest,
        runtime=plan.runtime,
        model=plan.model,
        prompt_hashes=plan.prompt_hashes,
        order_seed=plan.order_seed,
        plan_digest=digest,
    )


def verify_plan(plan: MatchedPlan) -> list[str]:
    """Re-derive the plan from its declared inputs; return mismatches (empty = consistent)."""
    try:
        rebuilt = build_plan(
            repetitions=plan.repetitions,
            baseline_digest=plan.baseline_digest,
            runtime=plan.runtime,
            model=plan.model,
            prompt_hashes=plan.prompt_hashes,
            order_seed=plan.order_seed,
        )
    except PlanError as exc:
        return [f"plan_mismatch: declared inputs are invalid ({exc})"]
    problems = []
    if rebuilt.slots != plan.slots:
        problems.append("plan_mismatch: slots do not re-derive from repetitions and order_seed")
    if hashlib.sha256(_canonical(_plan_payload(plan))).hexdigest() != plan.plan_digest:
        problems.append("plan_mismatch: plan_digest does not match the canonical plan contents")
    return problems


def _parse_started(attempt: Attempt) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(attempt.started_utc)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _slot(attempt: Attempt) -> tuple[str, int]:
    return (attempt.arm, attempt.repetition)


def validate_attempts(
    plan: MatchedPlan, attempts: list[Attempt] | tuple[Attempt, ...]
) -> list[str]:
    """Protocol defects; an empty list means the attempts satisfy the predeclared plan."""
    problems = verify_plan(plan)
    slots = set(plan.slots)
    by_slot: dict[tuple[str, int], list[Attempt]] = {}
    for attempt in attempts:
        slot = _slot(attempt)
        if slot not in slots:
            problems.append(
                f"not_in_plan: attempt {attempt.run_id!r} claims slot {slot!r}, which the plan "
                "does not contain"
            )
            continue
        by_slot.setdefault(slot, []).append(attempt)
    for slot in plan.slots:
        found = by_slot.get(slot, [])
        if len(found) > 1:
            ids = ", ".join(repr(a.run_id) for a in found)
            problems.append(f"duplicate_slot: {slot!r} has {len(found)} attempts ({ids})")
        elif not found:
            problems.append(f"incomplete: slot {slot!r} has no attempt")
    run_ids: dict[str, int] = {}
    for attempt in attempts:
        run_ids[attempt.run_id] = run_ids.get(attempt.run_id, 0) + 1
        if attempt.clone_baseline_digest != plan.baseline_digest:
            problems.append(
                f"baseline_digest_mismatch: attempt {attempt.run_id!r} ran on a clone with a "
                "different baseline digest than the plan"
            )
        if attempt.runtime != plan.runtime:
            problems.append(
                f"runtime_mismatch: attempt {attempt.run_id!r} used {attempt.runtime!r}, "
                f"plan pins {plan.runtime!r}"
            )
        if attempt.model != plan.model:
            problems.append(
                f"model_mismatch: attempt {attempt.run_id!r} used {attempt.model!r}, "
                f"plan pins {plan.model!r}"
            )
        if dict(attempt.prompt_hashes) != dict(plan.prompt_hashes):
            problems.append(
                f"prompt_hashes_mismatch: attempt {attempt.run_id!r} did not use the plan's "
                "identical prompts"
            )
    for run_id, count in run_ids.items():
        if count > 1:
            problems.append(f"duplicate_run_id: {run_id!r} appears in {count} attempts")
    problems += _order_problems(plan, by_slot)
    return problems


def _order_problems(plan: MatchedPlan, by_slot: dict[tuple[str, int], list[Attempt]]) -> list[str]:
    problems: list[str] = []
    timed: list[tuple[datetime, tuple[str, int]]] = []
    for slot in plan.slots:
        found = by_slot.get(slot, [])
        if len(found) != 1:
            continue
        started = _parse_started(found[0])
        if started is None:
            problems.append(
                f"invalid_started_utc: slot {slot!r} has no timezone-aware ISO-8601 start time"
            )
            continue
        timed.append((started, slot))
    starts = [t for t, _ in timed]
    if len(set(starts)) != len(starts):
        problems.append("ambiguous_order: two attempts share an identical start time")
        return problems
    executed = [slot for _, slot in sorted(timed)]
    planned = [slot for slot in plan.slots if slot in {s for _, s in timed}]
    if executed != planned:
        problems.append(
            f"out_of_order: attempts started as {executed!r} but the plan declared {planned!r}"
        )
    return problems


# --------------------------------------------------------------------------- report


@dataclass(frozen=True)
class ComparisonReport:
    plan: Mapping[str, Any]
    plan_problems: tuple[str, ...]
    correctness: Mapping[str, Any]
    attempts: tuple[Mapping[str, Any], ...]
    measures: Mapping[str, Any]
    pairs_planned: int
    pairs_compared: int
    notes: tuple[str, ...] = field(default_factory=tuple)
    statistical_claim: str = "none"
    descriptive_only: bool = True
    counts_toward_calibration_gate: bool = False
    calibration_reason: str = CALIBRATION_REASON
    method_arithmetic_profile_separate: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Correctness first, then every attempt, then measures, then protocol facts."""
        return json.loads(
            json.dumps(
                {
                    "correctness": self.correctness,
                    "attempts": list(self.attempts),
                    "measures": self.measures,
                    "plan": self.plan,
                    "plan_problems": list(self.plan_problems),
                    "plan_complete": not self.plan_problems,
                    "pairs_planned": self.pairs_planned,
                    "pairs_compared": self.pairs_compared,
                    "notes": list(self.notes),
                    "statistical_claim": self.statistical_claim,
                    "descriptive_only": self.descriptive_only,
                    "counts_toward_calibration_gate": self.counts_toward_calibration_gate,
                    "calibration_reason": self.calibration_reason,
                    "method_arithmetic_profile_separate": self.method_arithmetic_profile_separate,
                },
                default=dict,
            )
        )


def savings_from_sessions(*args: object, **kwargs: object) -> None:
    """Always refused: A(L1/L2) and B(L3/L4) cost differences are never savings."""
    raise ForbiddenComparison(_UNSAFE_AB_MESSAGE)


def _refuse_forbidden(attempts: list[Attempt] | tuple[Attempt, ...]) -> None:
    for attempt in attempts:
        if attempt.arm not in ARMS:
            raise ForbiddenComparison(
                f"attempt {attempt.run_id!r} carries session label {attempt.arm!r}, outside "
                f"{ARMS!r}. {_UNSAFE_AB_MESSAGE}"
            )
        for name in attempt.measured:
            if FORBIDDEN_MEASURE_FRAGMENT in name.lower():
                raise ForbiddenComparison(
                    f"measure {name!r} on attempt {attempt.run_id!r} names an A-versus-B "
                    f"comparison. {_UNSAFE_AB_MESSAGE}"
                )


def _correctness_section(
    attempts: list[Attempt] | tuple[Attempt, ...], gates: list[str]
) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for arm in ARMS:
        mine = [a for a in attempts if a.arm == arm]
        per_gate: dict[str, Any] = {}
        for gate in gates:
            counts = {"PASS": 0, "FAIL": 0, "other": 0}
            other: dict[str, int] = {}
            for attempt in mine:
                outcome = attempt.correctness.get(gate, "ABSENT")
                if outcome == "PASS":
                    counts["PASS"] += 1
                elif outcome == "FAIL":
                    counts["FAIL"] += 1
                else:
                    counts["other"] += 1
                    other[outcome] = other.get(outcome, 0) + 1
            per_gate[gate] = {**counts, "other_outcomes": dict(sorted(other.items()))}
        section[arm] = {"attempts": len(mine), "gates": per_gate}
    return section


def _exclusion_reason(attempt: Attempt, gates: list[str]) -> str | None:
    if attempt.status != "completed":
        return f"attempt_status_{attempt.status}: a cost is not comparable without a completed run"
    if not gates:
        return "no_correctness_gates: correctness was not assessed"
    for gate in gates:
        outcome = attempt.correctness.get(gate, "ABSENT")
        if outcome != PASS:
            return (
                f"correctness_not_pass: gate {gate!r} is {outcome}; the cost of an incorrect or "
                "unverified solution is not comparable"
            )
    return None


def _measure_section(
    names: list[str], by_rep: dict[int, dict[str, Attempt]], reps: list[int]
) -> dict[str, Any]:
    comparable: dict[str, Any] = {}
    not_comparable: dict[str, Any] = {}
    for name in names:
        reason: str | None = None
        units: set[str] = set()
        for rep in reps:
            for arm in ARMS:
                attempt = by_rep[rep][arm]
                measurement = attempt.measured.get(name)
                if measurement is None:
                    reason = f"measure_absent: {arm} repetition {rep} did not record it"
                elif measurement.value is None and reason is None:
                    reason = (
                        f"value_unknown: {arm} repetition {rep} has no value ({measurement.reason})"
                    )
                if measurement is not None:
                    units.add(measurement.unit)
            if reason is not None:
                break
        if reason is not None:
            not_comparable[name] = {"reason": reason}
            continue
        if len(units) != 1:
            not_comparable[name] = {
                "reason": f"unit_mismatch: attempts record {sorted(units)!r}",
            }
            continue
        totals = dict.fromkeys(ARMS, 0)
        bases: dict[str, set[str]] = {arm: set() for arm in ARMS}
        per_pair = []
        for rep in reps:
            values = {}
            for arm in ARMS:
                measurement = by_rep[rep][arm].measured[name]
                assert measurement.value is not None
                values[arm] = measurement.value
                totals[arm] += measurement.value
                bases[arm].add(measurement.basis)
            per_pair.append(
                {
                    "repetition": rep,
                    WITH_HISTORY: values[WITH_HISTORY],
                    WITHOUT_HISTORY: values[WITHOUT_HISTORY],
                    "difference": values[WITH_HISTORY] - values[WITHOUT_HISTORY],
                }
            )
        comparable[name] = {
            "unit": next(iter(units)),
            "pairs": len(reps),
            "with_history_total": totals[WITH_HISTORY],
            "without_history_total": totals[WITHOUT_HISTORY],
            "difference_with_history_minus_without_history": totals[WITH_HISTORY]
            - totals[WITHOUT_HISTORY],
            "per_pair": per_pair,
            "bases": {arm: sorted(bases[arm]) for arm in ARMS},
        }
    return {"comparable": comparable, "not_comparable": not_comparable}


def compare(plan: MatchedPlan, attempts: list[Attempt] | tuple[Attempt, ...]) -> ComparisonReport:
    _refuse_forbidden(attempts)
    problems = validate_attempts(plan, attempts)
    gates = sorted({gate for a in attempts for gate in a.correctness})
    correctness = _correctness_section(attempts, gates)

    reasons: list[str | None] = [_exclusion_reason(a, gates) for a in attempts]
    names = sorted({name for a in attempts for name in a.measured})
    pairs_planned = plan.repetitions
    notes: list[str] = []

    eligible_by_rep: dict[int, dict[str, Attempt]] = {}
    if problems:
        reasons = [r or "plan_not_satisfied: see plan_problems" for r in reasons]
        measures: dict[str, Any] = {
            "comparable": {},
            "not_comparable": {
                name: {"reason": "plan_not_satisfied: see plan_problems"} for name in names
            },
        }
        notes.append("plan_not_satisfied")
    else:
        for attempt, reason in zip(attempts, reasons, strict=True):
            if reason is None:
                eligible_by_rep.setdefault(attempt.repetition, {})[attempt.arm] = attempt
        paired = sorted(rep for rep, arms in eligible_by_rep.items() if len(arms) == len(ARMS))
        reasons = [
            r
            or (
                None
                if a.repetition in paired
                else "pair_partner_excluded: the other arm's attempt of this repetition is excluded"
            )
            for a, r in zip(attempts, reasons, strict=True)
        ]
        if paired:
            measures = _measure_section(names, eligible_by_rep, paired)
        else:
            measures = {
                "comparable": {},
                "not_comparable": {name: {"reason": _NO_PAIR} for name in names},
            }
            notes.append("no_eligible_pair")
        eligible_by_rep = {rep: eligible_by_rep[rep] for rep in paired}
    pairs_compared = len(eligible_by_rep)
    if pairs_compared == 1:
        notes.append("single_pair_descriptive")

    rows = []
    for attempt, reason in zip(attempts, reasons, strict=True):
        rows.append(
            {
                "arm": attempt.arm,
                "repetition": attempt.repetition,
                "run_id": attempt.run_id,
                "status": attempt.status,
                "started_utc": attempt.started_utc,
                "clone_baseline_digest": attempt.clone_baseline_digest,
                "runtime": attempt.runtime,
                "model": attempt.model,
                "prompt_hashes": dict(attempt.prompt_hashes),
                "correctness": dict(attempt.correctness),
                "measured": {n: m.as_dict() for n, m in sorted(attempt.measured.items())},
                "included_in_measure_comparison": reason is None,
                "exclusion_reason": reason,
            }
        )
    return ComparisonReport(
        plan=plan.as_dict(),
        plan_problems=tuple(problems),
        correctness=correctness,
        attempts=tuple(rows),
        measures=measures,
        pairs_planned=pairs_planned,
        pairs_compared=pairs_compared,
        notes=tuple(notes),
    )
