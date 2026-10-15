"""Independent raw-input boundaries and deliberately wrong estimate sensitivity."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval.accounting_airline import accounting_oracle as oracle
from eval.accounting_airline.oracle_estimates import bounded_estimate
from eval.accounting_airline.oracle_inputs import IndexStats, accept_lookup_inputs
from eval.accounting_airline.oracle_time import reconcile_timing, utc_nanoseconds
from eval.accounting_airline.roots import LabRoots
from tests.accounting.airline.oracle_fixtures import (
    TOKEN,
    consumer_b,
    frozen,
    lookup,
    search_call,
    search_json,
    source_a,
)
from tests.accounting.airline.test_lab_oracle import savings_doc


class FlatFixtureRoots(LabRoots):
    @property
    def project(self) -> Path:
        return self.root

    @property
    def ssgrep_data(self) -> Path:
        return self.root


def literal_lookups(durations=(333400000, 333400000, 333400000), *, action=False):
    source = source_a()
    consumer = consumer_b(action="use" if action else None, t0=10000)
    calls, events = [], []
    for index, duration in enumerate(durations):
        native = "cs1" if index == 0 else f"extra{index}"
        if index:
            lookup(consumer, native, search_json(TOKEN, 200))
        wall = (1000 + 2 * index) * 1000000000
        call = replace(
            search_call(call_id=f"observed-{index}", native=native),
            wall_ns=wall,
            duration_ns=duration,
        )
        calls.append(call)
        events.append(
            {
                "execution_id": f"execution-{index}",
                "operation": "search",
                "phase": "completed",
                "mono_ns": duration,
                "ts_utc": datetime.fromtimestamp(1000 + 2 * index, UTC).isoformat(),
            }
        )
    return frozen([source, consumer], calls), events


def test_raw_nanosecond_remainders_cross_seconds_once_and_replay_is_idempotent():
    world, events = literal_lookups()
    profile = oracle.arithmetic_profile([("identifier", TOKEN)], events)
    result = oracle.evaluate(world, profile)
    for row in result.time_estimates.values():
        assert (row.credit, row.debit, row.net, row.known_net) == (0, 1, -1, -1)
        assert (row.bounds["net_lower"], row.bounds["net_upper"]) == (-2, -1)
    assert result.timing["known_service_ns"] == 1000200000
    assert result.timing["known_service_ms"] == 1000
    repeated = oracle.evaluate(
        world, oracle.arithmetic_profile([("identifier", TOKEN)], events * 2)
    )
    assert repeated.time_estimates == result.time_estimates
    assert repeated.timing == result.timing
    extra_world, extra_events = literal_lookups((333400000, 333400000, 333400000, 999800000))
    extra = oracle.evaluate(
        extra_world, oracle.arithmetic_profile([("identifier", TOKEN)], extra_events)
    )
    assert extra.timing["known_service_ns"] == 2000000000
    assert all(row.debit == 2 and row.net == -2 for row in extra.time_estimates.values())


def test_source_millisecond_union_composes_exactly_with_raw_nanosecond_debits():
    world, events = literal_lookups(action=True)
    actor, task, native = world.sessions["sa"]
    native = replace(
        native,
        records=tuple(replace(r, ts_ms=1000 + i * 600) for i, r in enumerate(native.records)),
    )
    world = replace(world, sessions={**world.sessions, "sa": (actor, task, native)})
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)], events))
    # Immediate call->result and result->following-call are two 600ms source gaps.
    row = result.time_estimates["conservative"]
    assert (row.credit, row.debit, row.net) == (1, 1, 0)


@pytest.mark.parametrize("maximum,lower", [(1333600000, -2), (None, None)])
def test_bounded_and_unbounded_missing_debits_do_not_overstate_net_lower(maximum, lower):
    row = bounded_estimate(
        "central", 0, False, 1000200000, 1000200000, maximum, 1000000000, ("debit:incomplete",)
    )
    assert row.net is None and row.known_net == -1
    assert row.bounds["net_lower"] == lower and row.bounds["net_upper"] == -1


def test_missing_native_result_and_raw_duration_leave_both_net_lower_bounds_unbounded():
    world, events = literal_lookups((None,))
    actor, task, native = world.sessions["sb"]
    native = replace(native, records=tuple(r for r in native.records if r.kind != "tool_result"))
    world = replace(world, sessions={**world.sessions, "sb": (actor, task, native)})
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)], events))
    for estimates in (result.estimates, result.time_estimates):
        assert all(
            row.net is None
            and row.bounds["debit_upper"] is None
            and row.bounds["net_lower"] is None
            for row in estimates.values()
        )


def test_raw_timing_uses_one_millisecond_conversion_after_summing_and_exact_per_op_cap():
    world, events = literal_lookups((999999, 1))
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)], events))
    assert result.timing["known_service_ns"] == 1000000
    assert result.timing["known_service_ms"] == 1
    world, events = literal_lookups((300000000001,))
    call = replace(world.calls[0], duration_ns=400000000000)
    events[0]["ts_utc"] = datetime.fromtimestamp(1301, UTC).isoformat()
    world = replace(world, calls=(call,))
    from eval.accounting_airline.oracle_scan import scan_frozen

    service_ns, work_ns, missing = reconcile_timing(scan_frozen(world), events)
    assert (service_ns, work_ns, missing) == (300000000001, 300000000000, ())


def test_naive_utc_clock_is_explicitly_unavailable_not_local_timezone_interpreted():
    assert utc_nanoseconds("2026-10-14T00:00:00") == (None, "journal_utc_timezone_missing")
    assert utc_nanoseconds("1970-01-01T00:00:01Z") == (1000000000, "")
    world, events = literal_lookups((333400000,))
    events[0]["ts_utc"] = "1970-01-01T00:16:40"
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)], events))
    assert "journal_utc_timezone_missing" in result.timing["missing"]
    assert result.time_estimates["central"].bounds["net_lower"] is None


def raw_stats(n, generation):
    return IndexStats(
        generation=generation,
        documents=n,
        df=(("mct", 1),),
        corpus_digest=f"digest-{generation}",
        authority_id="lab",
        key_id="key",
        profile_scope="personal-local",
    )


def test_frequency_freezes_at_collection_acceptance_not_wall_lookup_or_later_profile():
    from tests.accounting.airline.oracle_fixtures import base_world

    world, _, _ = base_world()  # Lookup occurred before this independent collection boundary.
    accepted = accept_lookup_inputs(
        {}, world.calls, raw_stats(511, "accepted"), raw_stats(511, "accepted")
    )
    result = oracle.evaluate(world, oracle.production_profile(accepted))
    assert result.estimates["conservative"].credit == 35
    assert dict(result.verdicts[0].provenance)["generation"] == "accepted"
    preserved = accept_lookup_inputs(
        accepted, world.calls, raw_stats(199, "later"), raw_stats(199, "later")
    )
    assert (
        oracle.evaluate(world, oracle.production_profile(preserved)).estimates == result.estimates
    )
    raced = accept_lookup_inputs({}, world.calls, raw_stats(199, "before"), raw_stats(511, "after"))
    unproved = oracle.evaluate(world, oracle.production_profile(raced))
    assert (
        dict(unproved.verdicts[0].provenance)["unavailable_reason"]
        == "frequency_inputs_changed_during_collection"
    )
    assert all(row.credit == 0 and row.net is None for row in unproved.estimates.values())


def test_conflicting_dispatch_statistics_have_typed_provenance_and_no_positive_credit():
    from tests.accounting.airline.oracle_fixtures import base_world

    world, _, _ = base_world()
    first = world.calls[0]
    second = replace(first, call_id="second-dispatch", dispatch_seq=2)
    world = replace(world, calls=(first, second))
    result = oracle.evaluate(
        world,
        oracle.production_profile(
            {first.call_id: raw_stats(199, "g1"), second.call_id: raw_stats(511, "g2")}
        ),
    )
    assert dict(result.verdicts[0].provenance)["unavailable_reason"] == "conflicting_lookup_stats"
    assert all(row.credit == 0 and row.net is None for row in result.estimates.values())


def test_wrong_candidate_snapshot_subtotals_and_bounded_negative_endpoints_are_detected():
    from tests.accounting.airline.oracle_fixtures import base_world

    world, _, _ = base_world()
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)]))
    document = savings_doc(result)
    row = document["data"]["data"]["estimates"]["context_tokens"]["central"]
    # Literal independently justified reference: source156//4=39, visible200//4=50 =>-11.
    assert result.estimates["central"].net == -11
    row["credit"]["value"] = 40
    row["known_net"] = -10
    row["bounds"]["net_lower"]["value"] = -10
    problems = oracle.compare_with_snapshot(result, document)
    assert any("central.credit" in p for p in problems)
    assert any("central.known_net" in p for p in problems)
    assert any("net_lower" in p for p in problems)


def work_document(result, *, missing=False):
    def q(value):
        return {
            "value": value,
            "unit": "work_seconds",
            "reason": "incomplete" if value is None else None,
        }

    document = savings_doc(result)
    bounds = (
        {
            "credit_lower": 0,
            "credit_upper": 0,
            "debit_lower": 0,
            "debit_upper": None,
            "net_lower": None,
            "net_upper": 0,
        }
        if missing
        else {
            "credit_lower": 0,
            "credit_upper": 0,
            "debit_lower": 1,
            "debit_upper": 2,
            "net_lower": -2,
            "net_upper": -1,
        }
    )
    document["data"]["data"]["estimates"]["work_seconds"] = {
        scenario: {
            "credit": q(0),
            "debit": q(0 if missing else 1),
            "net": q(None if missing else -1),
            "known_net": 0 if missing else -1,
            "complete": not missing,
            "credit_missing": [],
            "debit_missing": [["raw", "missing"]] if missing else [],
            "bounds": {
                "assumptions": [
                    "outward_round_lower_floor_upper_ceil",
                    "point_net_uses_floored_component_totals",
                ],
                **{name: q(value) for name, value in bounds.items()},
            },
            "unit": {"unit": "work_seconds", "method": "capped_gap_v1"},
            "provenance": [["lookup_work_mapping", "journal_active_proxy_ns_v2"]],
        }
        for scenario in oracle.SCENARIOS
    }
    return document


def test_snapshot_catches_old_per_call_quantization_and_false_finite_negative_bound():
    world, events = literal_lookups()
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)], events))
    document = work_document(result)
    assert oracle.compare_with_snapshot(result, document) == []
    row = document["data"]["data"]["estimates"]["work_seconds"]["central"]
    row["debit"]["value"] = row["net"]["value"] = row["known_net"] = 0
    row["bounds"]["net_lower"]["value"] = 0
    problems = oracle.compare_with_snapshot(result, document)
    assert any("work_seconds.central.debit" in p for p in problems)
    assert any("work_seconds.central.bounds.net_lower" in p for p in problems)
    document = work_document(result)
    row = document["data"]["data"]["estimates"]["work_seconds"]["central"]
    row["bounds"]["debit_upper"]["value"] = 1
    row["bounds"]["net_lower"]["value"] = -1
    # Correct point=-1 cannot justify the old closed [-1,-1] raw interval.
    problems = oracle.compare_with_snapshot(result, document)
    assert any("work_seconds.central.bounds.debit_upper" in p for p in problems)
    assert any("work_seconds.central.bounds.net_lower" in p for p in problems)
    world, events = literal_lookups((None,))
    actor, task, native = world.sessions["sb"]
    native = replace(native, records=tuple(r for r in native.records if r.kind != "tool_result"))
    world = replace(world, sessions={**world.sessions, "sb": (actor, task, native)})
    result = oracle.evaluate(world, oracle.arithmetic_profile([("identifier", TOKEN)], events))
    document = work_document(result, missing=True)
    assert oracle.compare_with_snapshot(result, document) == []
    document["data"]["data"]["estimates"]["work_seconds"]["central"]["bounds"]["net_lower"][
        "value"
    ] = 0
    assert any(
        "work_seconds.central.bounds.net_lower" in p
        for p in oracle.compare_with_snapshot(result, document)
    )


def test_raw_journal_hmac_join_and_missing_key_cannot_invent_measured_precision(tmp_path):
    import hmac
    import json

    from eval.accounting_airline.oracle_scan import scan_frozen
    from eval.accounting_airline.oracle_time import read_timing_events

    world, events = literal_lookups((333400000,))
    call = replace(world.calls[0], argv=("ssgrep", "search", "--json", " rules "))
    world = replace(world, calls=(call,))
    directory = tmp_path / "accounting/personal-local"
    (directory / "journal").mkdir(parents=True)
    key = directory / "fingerprint.key"
    key.write_bytes(b"k" * 32)
    events[0]["fingerprint"] = hmac.digest(
        b"k" * 32, b'{"op":"search","q":"rules","v":"fp1","w":null}', "sha256"
    ).hex()
    journal = directory / "journal/seg-1.jsonl"
    original = json.dumps(events[0]).encode() + b"\n"
    journal.write_bytes(original)
    raw = read_timing_events(tmp_path, world.calls)
    assert raw[0]["_observer_call_ids"] == [call.call_id]
    assert reconcile_timing(scan_frozen(world), raw) == (333400000, 333400000, ())
    key.unlink()
    unknown = read_timing_events(tmp_path, world.calls)
    assert unknown[0]["_observer_call_ids"] == []
    assert reconcile_timing(scan_frozen(world), unknown)[2] == (
        "no_unique_observed_journal_timing",
    )
    assert journal.read_bytes() == original


def test_report_time_profile_cannot_accept_an_uncollected_lookup(tmp_path):
    from eval.accounting_airline.lab_context import Run
    from tests.accounting.airline.oracle_fixtures import base_world

    world, _, _ = base_world()
    run = object.__new__(Run)
    run.notes = {}
    run.roots = FlatFixtureRoots(tmp_path, "fixture")
    pending = oracle.evaluate(world, run.production_profile(world))
    assert dict(pending.verdicts[0].provenance)["unavailable_reason"] == "lookup_not_collected"
    assert run.notes == {}
    run.notes["oracle_lookup_stats"] = accept_lookup_inputs(
        {}, world.calls, raw_stats(511, "accepted"), raw_stats(511, "accepted")
    )
    accepted = oracle.evaluate(world, run.production_profile(world))
    assert accepted.estimates["conservative"].credit == 35
    assert dict(accepted.verdicts[0].provenance)["generation"] == "accepted"


@pytest.mark.parametrize(
    "credit,debit,divisor,point,expected",
    [
        (3000000000, 1999999999, 1000000000, 2, (1, 2)),
        (0, 1000200000, 1000000000, -1, (-2, -1)),
        (7, 5, 4, 0, (-1, 1)),
        (2000000000, 1000000000, 1000000000, 1, (1, 1)),
    ],
)
def test_outward_integer_bounds_enclose_unquantized_raw_quantity(
    credit,
    debit,
    divisor,
    point,
    expected,
):
    row = bounded_estimate("central", credit, False, debit, debit, debit, divisor, ())
    lower, upper = row.bounds["net_lower"], row.bounds["net_upper"]
    assert row.net == row.known_net == point
    assert (lower, upper) == expected
    assert lower * divisor <= credit - debit <= upper * divisor
    assert row.bounds["credit_lower"] * divisor <= credit <= row.bounds["credit_upper"] * divisor
    assert row.bounds["debit_lower"] * divisor <= debit <= row.bounds["debit_upper"] * divisor


def test_post_collection_cutoff_does_not_assign_unaccepted_future_calls_to_earlier_generation():
    from tests.accounting.airline.oracle_fixtures import base_world

    world, _, _ = base_world()
    late = world.calls[0]
    before, after = raw_stats(511, "old"), raw_stats(511, "old")
    frozen_inputs = accept_lookup_inputs({}, (), before, after, calls_after=(late,))
    row = frozen_inputs[late.call_id]
    assert row.generation == ""
    assert row.unavailable_reason == "lookup_acceptance_during_collection_unproven"
    assert all(
        estimate.net is None and estimate.credit == 0
        for estimate in oracle.evaluate(
            world, oracle.production_profile(frozen_inputs)
        ).estimates.values()
    )
    future = replace(late, call_id="after-cutoff", native_call_id="later")
    next_inputs = accept_lookup_inputs(
        frozen_inputs,
        (late, future),
        raw_stats(199, "next"),
        raw_stats(199, "next"),
        calls_after=(late, future),
    )
    assert next_inputs[future.call_id].generation == "next"
    assert next_inputs[late.call_id].generation == ""
    raced = accept_lookup_inputs(
        {}, (), raw_stats(199, "before"), raw_stats(511, "after"), calls_after=(late,)
    )
    assert raced[late.call_id].unavailable_reason == "lookup_raced_collection"


@pytest.mark.parametrize(
    "unit,method,raw_unit,divisor,credit,debit,points,endpoints",
    [
        (
            "work_seconds",
            "capped_gap_v1",
            "nanoseconds",
            1000000000,
            3000000000,
            1999999999,
            (3, 1, 2),
            (3, 3, 1, 2, 1, 2),
        ),
        ("context_tokens", "chars_div4_v1", "characters", 4, 7, 5, (1, 1, 0), (1, 2, 1, 2, -1, 1)),
    ],
)
def test_receipt_outward_endpoints_use_explicit_raw_inputs_not_rounded_points(
    unit,
    method,
    raw_unit,
    divisor,
    credit,
    debit,
    points,
    endpoints,
):
    import copy
    from types import SimpleNamespace

    from eval.accounting_airline.oracle_receipts import compare_receipt

    expected = bounded_estimate("central", credit, False, debit, debit, debit, divisor, ())
    names = ("credit_lower", "credit_upper", "debit_lower", "debit_upper", "net_lower", "net_upper")
    assert tuple(expected.bounds[name] for name in names) == endpoints

    def q(value):
        return {"value": value, "unit": unit, "reason": None}

    row = {
        "scenario": "central",
        "credit": q(points[0]),
        "debit": q(points[1]),
        "net": q(points[2]),
        "known_net": points[2],
        "complete": True,
        "credit_missing": [],
        "debit_missing": [],
        "bounds": {
            "assumptions": list(expected.bound_assumptions),
            **{name: q(value) for name, value in zip(names, endpoints, strict=True)},
        },
    }
    result = SimpleNamespace(estimates={"central": expected}, time_estimates={"central": expected})
    document: dict[str, Any] = {
        "unit": f"{unit}:{method}",
        "raw_unit": raw_unit,
        "credit_raw": {"central": credit},
        "debit_raw": debit,
        "debit_raw_min": debit,
        "debit_raw_max": debit,
        "scenarios": [row],
    }
    assert compare_receipt(result, document) == []
    wrong = copy.deepcopy(document)
    wrong["scenarios"][0]["bounds"]["debit_upper"]["value"] = points[1]
    assert any("incorrect outward endpoints" in p for p in compare_receipt(result, wrong))
    wrong = copy.deepcopy(document)
    wrong["credit_raw"]["central"] += 1
    assert any(
        "raw allocation exceeds independently observed" in p for p in compare_receipt(result, wrong)
    )
    document.pop("raw_unit")
    assert any("no rounded fallback" in p for p in compare_receipt(result, document))


def test_collect_records_post_cutoff_arrival_as_unproved_not_earlier_positive_stats(
    tmp_path, monkeypatch
):
    from eval.accounting_airline import lab_context
    from eval.accounting_airline.lab_context import Run
    from eval.accounting_airline.lab_product import Collected
    from tests.accounting.airline.oracle_fixtures import base_world

    world, _, _ = base_world()
    state = {"frozen": replace(world, calls=())}

    def collecting(_product, *_flags):
        state["frozen"] = world
        return Collected({}, {}, {}, 0, {})

    monkeypatch.setattr(lab_context, "collect", collecting)
    monkeypatch.setattr(lab_context, "read_index_inputs", lambda *_args: raw_stats(511, "before"))
    run = object.__new__(Run)
    run.roots = FlatFixtureRoots(tmp_path, "fixture")
    run.notes, run.collected = {}, {}
    from eval.accounting_airline.lab_product import Product

    run.product = Product({"SSGREP_DATA_DIR": str(tmp_path)}, tmp_path)
    monkeypatch.setattr(run, "frozen", lambda: state["frozen"])
    run.collect("late")
    result = oracle.evaluate(world, run.production_profile(world))
    assert dict(result.verdicts[0].provenance)["unavailable_reason"] == (
        "lookup_acceptance_during_collection_unproven"
    )
    assert all(row.credit == 0 and row.net is None for row in result.estimates.values())


def test_terminal_billing_record_cannot_close_a_source_work_gap_or_make_net_complete():
    from eval.accounting_airline.observer import NativeRecord

    world, events = literal_lookups(action=True)
    actor, task, native = world.sessions["sa"]
    billing = NativeRecord(
        len(native.records),
        "usage",
        "closing-billing",
        ts_ms=native.records[-1].ts_ms + 10,
        usage={"output": 7},
        cost_nano_usd=123,
    )
    with_billing = replace(
        world,
        sessions={
            **world.sessions,
            "sa": (actor, task, replace(native, records=(*native.records, billing))),
        },
    )
    profile = oracle.arithmetic_profile([("identifier", TOKEN)], events)
    without = oracle.evaluate(world, profile)
    after = oracle.evaluate(with_billing, profile)
    assert after.time_estimates == without.time_estimates
    assert after.time_estimates["optimistic"].net is None
    assert after.time_estimates["optimistic"].bounds["credit_upper"] is None
    assert "credit:source_seconds_unavailable" in after.time_estimates["optimistic"].missing
    assert with_billing.sessions["sa"][2].records[-1].cost_nano_usd == 123
