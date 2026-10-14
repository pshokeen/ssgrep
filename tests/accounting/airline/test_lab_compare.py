"""Matched with/without-history comparison: plan, protocol checks, descriptive report."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from eval.accounting_airline import compare as cmp
from eval.accounting_airline.compare import (
    ARMS,
    Attempt,
    ForbiddenComparison,
    Measurement,
    PlanError,
    build_plan,
    compare,
    savings_from_sessions,
    validate_attempts,
)

BASELINE = "sha256:" + "ab" * 32
PROMPTS = {"L3": "h-l3", "L4": "h-l4"}
WH, WO = ARMS


def _plan(repetitions: int = 2, seed: str = "seed-1") -> cmp.MatchedPlan:
    return build_plan(
        repetitions=repetitions,
        baseline_digest=BASELINE,
        runtime="runtime-x",
        model="model-y",
        prompt_hashes=PROMPTS,
        order_seed=seed,
    )


def _ms(value: int | None, unit: str = "ms", basis: str = "observed") -> Measurement:
    if value is None:
        return Measurement(None, unit, "unavailable", "not reported by the runtime")
    return Measurement(value, unit, basis)


def _attempt(
    arm: str,
    rep: int,
    minute: int,
    *,
    tokens: int | None = 100,
    millis: int | None = 5000,
    status: str = "completed",
    gates: dict[str, str] | None = None,
    **overrides: Any,
) -> Attempt:
    fields: dict[str, Any] = {
        "arm": arm,
        "repetition": rep,
        "run_id": f"{arm}-{rep}",
        "clone_baseline_digest": BASELINE,
        "runtime": "runtime-x",
        "model": "model-y",
        "prompt_hashes": PROMPTS,
        "correctness": gates or {"l3_plan": "PASS", "l4_apply": "PASS"},
        "measured": {"tokens": _ms(tokens, "tokens"), "elapsed": _ms(millis, "ms")},
        "status": status,
        "started_utc": f"2026-10-13T10:{minute:02d}:00+00:00",
    }
    fields.update(overrides)
    return Attempt(**fields)


def _conforming(plan: cmp.MatchedPlan, **per_arm: Any) -> list[Attempt]:
    """One attempt per planned slot, started in plan order."""
    out = []
    for minute, (arm, rep) in enumerate(plan.slots):
        out.append(_attempt(arm, rep, minute, **per_arm.get(arm, {})))
    return out


# ------------------------------------------------------------------------- plan


def test_plan_is_deterministic_for_a_seed_and_covers_every_slot_once() -> None:
    a, b = _plan(6, "s"), _plan(6, "s")
    assert a == b
    assert a.slots == b.slots and a.plan_digest == b.plan_digest
    assert sorted(a.slots) == sorted((arm, r) for arm in ARMS for r in range(1, 7))
    assert len(set(a.slots)) == len(a.slots) == 12
    assert a.counts_toward_calibration_gate is False
    assert "50-receipt" in a.calibration_reason
    # seed is hashed, not clock- or random-derived: a second call much later is identical
    assert _plan(6, "s").slots == a.slots


def test_plan_seed_changes_order_and_digest() -> None:
    plans = {seed: _plan(8, seed) for seed in ("a", "b", "c", "d", "e", "f")}
    assert len({p.slots for p in plans.values()}) > 1
    assert len({p.plan_digest for p in plans.values()}) == len(plans)


@pytest.mark.parametrize("repetitions", [2, 4, 6, 10])
@pytest.mark.parametrize("seed", ["x", "y", "z", "seed-17"])
def test_plan_is_counterbalanced_for_even_repetitions(repetitions: int, seed: str) -> None:
    plan = _plan(repetitions, seed)
    leads = {}
    for arm, rep in plan.slots:
        leads.setdefault(rep, arm)  # the arm executed first within each repetition
    assert sorted(leads) == list(range(1, repetitions + 1))
    counts = {arm: sum(1 for lead in leads.values() if lead == arm) for arm in ARMS}
    assert counts[WH] == counts[WO] == repetitions // 2
    # interleaved, not blocked by arm
    assert plan.slots[0][1] == plan.slots[1][1] and plan.slots[0][0] != plan.slots[1][0]


def test_plan_odd_repetitions_still_complete_and_single_rep_plan() -> None:
    plan = _plan(3)
    assert sorted(plan.slots) == sorted((arm, r) for arm in ARMS for r in (1, 2, 3))
    assert len(_plan(1).slots) == 2


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "2"])
def test_plan_rejects_bad_repetitions(bad: Any) -> None:
    with pytest.raises(PlanError, match="repetitions"):
        _plan(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_digest", ""),
        ("runtime", ""),
        ("model", ""),
        ("order_seed", ""),
        ("prompt_hashes", {}),
    ],
)
def test_plan_rejects_unpinned_configuration(field: str, value: Any) -> None:
    kwargs: dict[str, Any] = {
        "repetitions": 2,
        "baseline_digest": BASELINE,
        "runtime": "r",
        "model": "m",
        "prompt_hashes": PROMPTS,
        "order_seed": "s",
    }
    kwargs[field] = value
    with pytest.raises(PlanError):
        build_plan(**kwargs)


def test_plan_can_be_rederived_and_tampering_is_detected() -> None:
    plan = _plan(4, "audit")
    assert cmp.verify_plan(plan) == []
    reordered = dataclasses.replace(plan, slots=tuple(reversed(plan.slots)))
    assert any("slots do not re-derive" in p for p in cmp.verify_plan(reordered))
    assert any("plan_digest" in p for p in cmp.verify_plan(reordered))
    retargeted = dataclasses.replace(plan, model="model-z")
    assert any("plan_digest" in p for p in cmp.verify_plan(retargeted))
    assert not any("slots" in p for p in cmp.verify_plan(retargeted))
    with pytest.raises(PlanError):
        dataclasses.replace(plan, counts_toward_calibration_gate=True)


# ------------------------------------------------------------------ measurement


@pytest.mark.parametrize("value", [1.5, 0.0, float("nan"), "5", True])
def test_measurement_rejects_non_integers(value: Any) -> None:
    with pytest.raises(TypeError, match="int or None"):
        Measurement(value, "ms", "observed")


def test_measurement_accepts_integers_and_requires_reason_for_unknown() -> None:
    assert Measurement(-3, "ms", "reported").value == -3
    assert Measurement(0, "ms", "estimated").as_dict()["basis"] == "estimated"
    with pytest.raises(ValueError, match="requires a reason"):
        Measurement(None, "ms", "unavailable")
    assert Measurement(None, "ms", "unavailable", "runtime omits it").reason
    with pytest.raises(ValueError, match="basis"):
        Measurement(1, "ms", "guessed")
    with pytest.raises(ValueError, match="unavailable"):
        Measurement(1, "ms", "unavailable")


# -------------------------------------------------------------- validate_attempts


def test_conforming_attempts_validate_cleanly() -> None:
    plan = _plan(4, "ok")
    assert validate_attempts(plan, _conforming(plan)) == []


def _problem(plan: cmp.MatchedPlan, attempts: list[Attempt], prefix: str) -> list[str]:
    return [p for p in validate_attempts(plan, attempts) if p.startswith(prefix)]


def test_attempt_outside_plan_slots_is_reported() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    assert _problem(plan, attempts, "not_in_plan") == []
    attempts.append(_attempt(WH, 3, 30))
    assert len(_problem(plan, attempts, "not_in_plan")) == 1


def test_duplicate_slot_is_reported() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    assert _problem(plan, attempts, "duplicate_slot") == []
    dup = dataclasses.replace(attempts[0], run_id="again", started_utc="2026-10-13T11:00:00+00:00")
    problems = _problem(plan, [*attempts, dup], "duplicate_slot")
    assert len(problems) == 1 and "again" in problems[0]


@pytest.mark.parametrize(
    "override,prefix",
    [
        ({"clone_baseline_digest": "sha256:" + "cd" * 32}, "baseline_digest_mismatch"),
        ({"runtime": "other-runtime"}, "runtime_mismatch"),
        ({"model": "other-model"}, "model_mismatch"),
        ({"prompt_hashes": {"L3": "h-l3", "L4": "tampered"}}, "prompt_hashes_mismatch"),
    ],
)
def test_pinned_configuration_drift_is_reported(override: dict[str, Any], prefix: str) -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    assert _problem(plan, attempts, prefix) == []
    attempts[1] = dataclasses.replace(attempts[1], **override)
    assert len(_problem(plan, attempts, prefix)) == 1


def test_out_of_plan_execution_order_is_reported() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    assert _problem(plan, attempts, "out_of_order") == []
    # swap start times of the first two planned slots
    first, second = attempts[0], attempts[1]
    swapped = [
        dataclasses.replace(first, started_utc=second.started_utc),
        dataclasses.replace(second, started_utc=first.started_utc),
        *attempts[2:],
    ]
    problems = _problem(plan, swapped, "out_of_order")
    assert len(problems) == 1 and "plan declared" in problems[0]


def test_missing_slots_are_reported_as_incomplete_not_dropped() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    assert _problem(plan, attempts, "incomplete") == []
    missing = attempts.pop(1)
    problems = _problem(plan, attempts, "incomplete")
    assert len(problems) == 1 and repr((missing.arm, missing.repetition)) in problems[0]
    report = compare(plan, attempts).as_dict()
    assert report["plan_complete"] is False
    assert any(p.startswith("incomplete") for p in report["plan_problems"])


def test_tampered_plan_is_reported_by_validation() -> None:
    plan = _plan(2)
    bad = dataclasses.replace(plan, slots=tuple(reversed(plan.slots)))
    assert any(p.startswith("plan_mismatch") for p in validate_attempts(bad, _conforming(plan)))


def test_naive_timestamps_are_reported() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    attempts[0] = dataclasses.replace(attempts[0], started_utc="2026-10-13 10:00:00")
    assert len(_problem(plan, attempts, "invalid_started_utc")) == 1


# ---------------------------------------------------------------------- compare


def test_report_lists_correctness_before_everything_else() -> None:
    plan = _plan(2)
    report = compare(plan, _conforming(plan)).as_dict()
    keys = list(report)
    assert keys[0] == "correctness"
    assert keys.index("correctness") < keys.index("attempts") < keys.index("measures")
    for arm in ARMS:
        gates = report["correctness"][arm]["gates"]
        assert gates["l3_plan"] == {"PASS": 2, "FAIL": 0, "other": 0, "other_outcomes": {}}
        assert gates["l4_apply"]["PASS"] == 2
    assert report["statistical_claim"] == "none"
    assert report["descriptive_only"] is True
    assert report["counts_toward_calibration_gate"] is False
    assert "50-receipt" in report["calibration_reason"]
    assert report["method_arithmetic_profile_separate"] is True
    assert report["pairs_compared"] == 2 and report["pairs_planned"] == 2
    assert "single_pair_descriptive" not in report["notes"]


def test_single_pair_is_marked_descriptive() -> None:
    plan = _plan(1)
    report = compare(plan, _conforming(plan)).as_dict()
    assert report["pairs_compared"] == 1
    assert "single_pair_descriptive" in report["notes"]


def test_signed_integer_differences_with_units_preserved() -> None:
    plan = _plan(2)
    attempts = _conforming(
        plan,
        with_history={"tokens": 70, "millis": 4000},
        without_history={"tokens": 100, "millis": 3500},
    )
    measures = compare(plan, attempts).as_dict()["measures"]["comparable"]
    tokens = measures["tokens"]
    assert tokens["unit"] == "tokens"
    assert tokens["with_history_total"] == 140 and tokens["without_history_total"] == 200
    assert tokens["difference_with_history_minus_without_history"] == -60
    assert [p["difference"] for p in tokens["per_pair"]] == [-30, -30]
    elapsed = measures["elapsed"]
    assert elapsed["unit"] == "ms"
    assert elapsed["difference_with_history_minus_without_history"] == 1000  # positive twin
    assert isinstance(tokens["with_history_total"], int) and not isinstance(
        tokens["with_history_total"], bool
    )
    assert elapsed["bases"] == {WH: ["observed"], WO: ["observed"]}


def test_failed_attempt_is_retained_and_excluded_from_cost_comparison() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    failed_slot = next(i for i, a in enumerate(attempts) if (a.arm, a.repetition) == (WH, 2))
    attempts[failed_slot] = _attempt(
        WH,
        2,
        failed_slot,
        tokens=5,
        status="failed",
        gates={"l3_plan": "PASS", "l4_apply": "FAIL"},
    )
    report = compare(plan, attempts).as_dict()
    assert len(report["attempts"]) == 4  # nothing dropped
    rows = {(r["arm"], r["repetition"]): r for r in report["attempts"]}
    failed = rows[(WH, 2)]
    assert failed["status"] == "failed"
    assert failed["included_in_measure_comparison"] is False
    assert "attempt_status_failed" in failed["exclusion_reason"]
    assert rows[(WH, 1)]["included_in_measure_comparison"] is True
    assert rows[(WH, 1)]["exclusion_reason"] is None
    # its cost does not leak into the totals; the unmatched partner is excluded too
    assert rows[(WO, 2)]["included_in_measure_comparison"] is False
    assert "pair_partner_excluded" in rows[(WO, 2)]["exclusion_reason"]
    tokens = report["measures"]["comparable"]["tokens"]
    assert tokens["pairs"] == 1 and tokens["with_history_total"] == 100
    assert report["pairs_compared"] == 1 and "single_pair_descriptive" in report["notes"]
    assert report["correctness"][WH]["gates"]["l4_apply"]["FAIL"] == 1


def test_completed_attempt_with_failing_gate_is_excluded_with_reason() -> None:
    plan = _plan(1)
    attempts = [
        _attempt(a, r, m, gates={"l3_plan": "PASS" if a == WO else "FAIL"})
        for m, (a, r) in enumerate(plan.slots)
    ]
    report = compare(plan, attempts).as_dict()
    wh = next(r for r in report["attempts"] if r["arm"] == WH)
    assert "correctness_not_pass" in wh["exclusion_reason"] and "l3_plan" in wh["exclusion_reason"]
    assert report["measures"]["comparable"] == {}
    assert report["measures"]["not_comparable"]["tokens"]["reason"].startswith("no_eligible_pair")
    assert report["pairs_compared"] == 0
    assert report["correctness"][WH]["gates"]["l3_plan"]["FAIL"] == 1
    assert report["correctness"][WO]["gates"]["l3_plan"]["PASS"] == 1


@pytest.mark.parametrize("other", ["BLOCKED", "NOT_RUN"])
def test_blocked_or_not_run_gate_is_counted_as_other_and_excluded(other: str) -> None:
    plan = _plan(1)
    attempts = [
        _attempt(a, r, m, gates={"g": other if a == WH else "PASS"})
        for m, (a, r) in enumerate(plan.slots)
    ]
    report = compare(plan, attempts).as_dict()
    gate = report["correctness"][WH]["gates"]["g"]
    assert gate == {"PASS": 0, "FAIL": 0, "other": 1, "other_outcomes": {other: 1}}
    assert report["pairs_compared"] == 0


def test_unknown_measure_value_is_not_comparable_with_reason() -> None:
    plan = _plan(2)
    attempts = _conforming(plan, with_history={"tokens": None})
    report = compare(plan, attempts).as_dict()
    entry = report["measures"]["not_comparable"]["tokens"]
    assert (
        entry["reason"].startswith("value_unknown")
        and "not reported by the runtime" in entry["reason"]
    )
    assert "tokens" not in report["measures"]["comparable"]
    # positive twin: the other measure, known everywhere, is still compared
    assert "elapsed" in report["measures"]["comparable"]
    # missing entirely is also reported, not treated as zero
    attempts = _conforming(plan)
    attempts[0] = dataclasses.replace(attempts[0], measured={"elapsed": _ms(5000)})
    report = compare(plan, attempts).as_dict()
    assert report["measures"]["not_comparable"]["tokens"]["reason"].startswith("measure_absent")


def test_unit_mismatch_is_refused_for_that_measure() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    attempts[2] = dataclasses.replace(
        attempts[2], measured={"tokens": _ms(100, "ktokens"), "elapsed": _ms(5000)}
    )
    report = compare(plan, attempts).as_dict()
    entry = report["measures"]["not_comparable"]["tokens"]
    assert entry["reason"].startswith("unit_mismatch") and "ktokens" in entry["reason"]
    assert "tokens" not in report["measures"]["comparable"]
    assert "elapsed" in report["measures"]["comparable"]  # positive twin


def test_unsatisfied_plan_blocks_measure_comparison() -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    attempts[0] = dataclasses.replace(attempts[0], model="other-model")
    report = compare(plan, attempts).as_dict()
    assert report["plan_complete"] is False
    assert report["measures"]["comparable"] == {}
    assert all(
        v["reason"].startswith("plan_not_satisfied")
        for v in report["measures"]["not_comparable"].values()
    )
    assert len(report["attempts"]) == 4


def _walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield f"{path}/{key}", key
            yield from _walk(inner, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, inner in enumerate(value):
            yield from _walk(inner, f"{path}[{i}]")
    else:
        yield path, value


def test_no_forbidden_wording_anywhere_in_report() -> None:
    plan = _plan(2)
    failed = _conforming(plan)
    failed[1] = dataclasses.replace(failed[1], status="failed", correctness={"l3_plan": "FAIL"})
    reports = [
        compare(plan, _conforming(plan)).as_dict(),
        compare(plan, failed).as_dict(),
        compare(_plan(1), _conforming(_plan(1))).as_dict(),
        compare(plan, _conforming(plan)[:2]).as_dict(),
    ]
    banned = ("saving", "saved", "significan", "p-value", "p_value", "confidence", "causal")
    checked = 0
    for report in reports:
        for _, item in _walk(report):
            if isinstance(item, str):
                checked += 1
                low = item.lower()
                assert not any(word in low for word in banned), item
            assert not isinstance(item, float)
    assert checked > 50


def test_savings_from_sessions_is_always_refused_naming_difficulty() -> None:
    for args, kwargs in [((), {}), (("a", "b"), {}), ((), {"a_cost": 10, "b_cost": 20})]:
        with pytest.raises(ForbiddenComparison) as info:
            savings_from_sessions(*args, **kwargs)
        message = str(info.value)
        assert "task difficulty" in message
        assert "A" in message and "B" in message and "never savings" in message
    assert issubclass(ForbiddenComparison, RuntimeError)
    assert issubclass(PlanError, ValueError)


@pytest.mark.parametrize("label", ["A", "B", "session-a", "session_b", "", "WITH_HISTORY"])
def test_attempt_with_foreign_session_label_is_refused(label: str) -> None:
    plan = _plan(1)
    attempts = _conforming(plan)
    attempts[0] = dataclasses.replace(attempts[0], arm=label)
    with pytest.raises(ForbiddenComparison, match="task difficulty"):
        compare(plan, attempts)
    # positive twin: the correct labels are accepted
    assert compare(plan, _conforming(plan)).as_dict()["pairs_compared"] == 1


def test_a_vs_b_measure_name_is_refused() -> None:
    plan = _plan(1)
    attempts = _conforming(plan)
    bad = dict(attempts[0].measured, tokens_a_vs_b=_ms(3, "tokens"))
    attempts[0] = dataclasses.replace(attempts[0], measured=bad)
    with pytest.raises(ForbiddenComparison, match="a_vs_b"):
        compare(plan, attempts)
    ok = dict(attempts[0].measured)
    del ok["tokens_a_vs_b"]
    attempts[0] = dataclasses.replace(attempts[0], measured=ok)
    assert "tokens" in compare(plan, attempts).as_dict()["measures"]["comparable"]


def test_attempt_rejects_unknown_status_and_gate_outcome() -> None:
    with pytest.raises(ValueError, match="status"):
        _attempt(WH, 1, 0, status="done")
    with pytest.raises(ValueError, match="outcome"):
        _attempt(WH, 1, 0, gates={"g": "OK"})
    with pytest.raises(TypeError):
        _attempt(WH, 1, 0, measured={"tokens": 5})


@pytest.mark.parametrize("status", ["interrupted", "unknown"])
def test_interrupted_and_unknown_attempts_are_kept_and_excluded(status: str) -> None:
    plan = _plan(2)
    attempts = _conforming(plan)
    attempts[0] = dataclasses.replace(attempts[0], status=status)
    report = compare(plan, attempts).as_dict()
    row = next(r for r in report["attempts"] if r["run_id"] == attempts[0].run_id)
    assert row["status"] == status and row["included_in_measure_comparison"] is False
    assert f"attempt_status_{status}" in row["exclusion_reason"]
    assert len(report["attempts"]) == 4
