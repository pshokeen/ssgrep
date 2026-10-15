"""F2: raw journal precision survives caps, uniqueness, bounds and presentation."""

from dataclasses import replace

import pytest

from ssgrep.accounting.contracts import Basis, Phase, Quantity
from ssgrep.accounting.ledger_codec import bundle_from_dict, bundle_to_dict
from ssgrep.accounting.lookup_time import LOOKUP_WORK_ASSUMPTIONS, LookupTiming, journal_timings
from ssgrep.accounting.reduce_cands import WORK
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import Debit, EvidenceBundle, Methodology
from tests.accounting.reduce_builders import event, ts
from tests.accounting.valuation_builders import DAY, TASK, atom, candidate


def timings(durations):
    events = tuple(
        replace(event(f"execution-{i}", ts(12), phases=(Phase.COMPLETED,))[0], mono_ns=duration)
        for i, duration in enumerate(durations)
    )
    rows = tuple(journal_timings(events + events).values())
    return rows


def cost_rows(rows):
    return tuple(
        Debit(
            row.execution_id + ":work",
            TASK,
            DAY,
            "exposure",
            WORK,
            row.work,
            work_ns=row.work_ns,
            mapping_version="journal_active_proxy_ns_v2",
            assumptions=LOOKUP_WORK_ASSUMPTIONS,
        )
        for row in rows
    )


def estimates(debits, candidates=()):
    valued = value_evidence(
        EvidenceBundle(tuple(candidates), tuple(debits)), Methodology("literal-ns-fixture", (WORK,))
    )
    return rollup(valued.leaves, WORK)


def test_three_333400000ns_operations_debit_one_second_not_zero_in_all_bands():
    rows = timings((333_400_000,) * 3)
    assert [row.measured_service_ms.value for row in rows] == [333] * 3
    assert [row.work_ns for row in rows] == [333_400_000] * 3
    linked = LookupTiming(rows, True, "")
    assert linked.measured_service_ns.value == 1_000_200_000
    assert linked.measured_service_ms.value == 1000 and linked.work.value == 1
    costs = cost_rows(rows)
    for row in estimates(costs + costs):
        assert row.credit.value == 0 and row.debit.value == 1 and row.net.value == -1
        assert row.known_debits == 3 and row.complete
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (-2, -1)
    extra = replace(costs[0], debit_id="genuine-extra", work_ns=999_800_000)
    assert all(row.debit.value == 2 and row.net.value == -2 for row in estimates((*costs, extra)))
    assert bundle_from_dict(bundle_to_dict(EvidenceBundle(debits=costs))).debits == costs


def test_nanoseconds_cross_millisecond_boundary_only_after_sum():
    rows = timings((999_999, 1))
    assert [row.measured_service_ms.value for row in rows] == [0, 0]
    assert LookupTiming(rows, True, "").measured_service_ms.value == 1


def test_cap_is_per_operation_in_raw_ns_before_sum_and_seconds_conversion():
    rows = timings((300_000_999_999, 333_400_000, 333_400_000, 333_400_000))
    assert rows[0].measured_service_ns.value == 300_000_999_999
    assert rows[0].work_ns == 300_000_000_000
    assert LookupTiming(rows, True, "").measured_service_ms.value == 301001
    assert all(
        row.debit.value == 301 and row.net.value == -301 for row in estimates(cost_rows(rows))
    )


def test_source_native_ms_actor_unions_scale_exactly_when_composing_ns_debits():
    source = candidate(
        "source",
        immediate=(
            atom("a", 4, origin="a", t_start_ms=0, t_end_ms=600, actor="one"),
            atom("b", 4, origin="b", t_start_ms=400, t_end_ms=1000, actor="one"),
            atom("c", 4, origin="c", t_start_ms=0, t_end_ms=600, actor="two"),
        ),
    )
    for row in estimates(cost_rows(timings((333_400_000,) * 3)), (source,)):
        assert row.credit.value == 1 and row.debit.value == 1 and row.net.value == 0
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (-1, 1)


def test_missing_raw_ns_has_no_finite_lower_even_if_rounded_seconds_are_present():
    costs = cost_rows(timings((333_400_000,) * 3))
    legacy = Debit(
        "legacy-rounded",
        TASK,
        DAY,
        "exposure",
        WORK,
        Quantity(2, WORK.unit, Basis.ESTIMATED, WORK.method),
    )
    for row in estimates((*costs, legacy)):
        assert row.known_net == -1 and row.net.value is None
        assert row.bounds.net_lower.value is None and row.bounds.net_upper.value == -1
        assert row.debit_missing == (("legacy-rounded", "lookup_work_nanoseconds_unavailable"),)


def test_sealed_days_derive_milliseconds_from_raw_ns_across_day_boundaries(tmp_path):
    from ssgrep.accounting.windows import SealedDay, _observations, seal_days
    from tests.accounting.reduce_builders import claude_evidence
    from tests.accounting.test_lookup_time import records, reduce_timing, timed_events

    evidence = claude_evidence(tmp_path, {"-work/consumer.jsonl": records()})
    template = reduce_timing(tmp_path, evidence, timed_events()).executions[0]
    exact = tuple(
        replace(
            template,
            execution_id=f"exact-{i}",
            day=f"2026-10-{11 + i}",
            measured_service_ns=Quantity.observed(value, "work_ns", "monotonic_v1"),
            measured_service_ms=Quantity.observed(0, "work_ms", "monotonic_v1"),
        )
        for i, value in enumerate((999_999, 1))
    )
    sealed, _ = seal_days(exact, before_day="2026-10-14", already={})
    restored = tuple(SealedDay.from_dict(day.to_dict()) for day in sealed.values())
    assert _observations(restored)["measured_service_ms"]["value"] == 1
    raw = restored[0].to_dict()
    raw.pop("measured_service_known_ns")
    raw["measured_service_known_ms"] = 0  # old rounded artifact, not exact ns evidence
    observed = _observations((SealedDay.from_dict(raw),))
    assert observed["measured_service_ms"]["value"] is None
    assert observed["measured_service_ms_known"]["value"] == 0


def test_published_amount_without_raw_ns_requires_explicit_precision_migration():
    raw = bundle_to_dict(EvidenceBundle(debits=cost_rows(timings((333_400_000,)))))
    published_fields = {
        "debit_id",
        "task",
        "day",
        "kind",
        "unit",
        "amount",
        "partial",
        "exposure_id",
        "chars",
        "accepted_day",
    }
    raw["debits"][0] = {
        key: value for key, value in raw["debits"][0].items() if key in published_fields
    }
    with pytest.raises(ValueError, match="legacy_valuation_migration_required"):
        bundle_from_dict(raw)


def test_published_method_without_version_and_unmarked_raw_leaves_are_not_ns():
    from ssgrep.accounting.ledger_codec import ledger_state_from_dict
    from ssgrep.accounting.plain_codec import to_plain
    from ssgrep.accounting.replay import LedgerState
    from ssgrep.accounting.units import UnitMismatchError

    current = Methodology("literal-ns-fixture", (WORK,))
    published = to_plain(LedgerState(current))
    published["methodology"].pop("valuation_version")  # actual 407e3c2 schema
    with pytest.raises(ValueError, match="legacy_valuation_migration_required"):
        ledger_state_from_dict(published)
    valued = value_evidence(EvidenceBundle(debits=cost_rows(timings((333_400_000,) * 3))), current)
    with pytest.raises(UnitMismatchError, match="time_precision_namespace_migration_required"):
        rollup((replace(valued.leaves[0], raw_time_unit=""),), WORK)


def test_closed_company_projection_declares_raw_ns_and_accepts_new_mapping_version():
    from ssgrep.accounting import export_records
    from ssgrep.accounting.export_records_model import Registry
    from ssgrep.accounting.receiver_contract import Key, Outcome, ReceiverState, apply
    from ssgrep.accounting.replay import Ledger

    ledger = Ledger(Methodology("literal-ns-fixture", (WORK,)))
    ledger.ingest(EvidenceBundle(debits=cost_rows(timings((333_400_000,) * 3))))
    projected, omitted = export_records.project_candidates(
        ledger.rows(), key=bytes(range(32)), destination="test-only"
    )
    assert omitted == 0 and len(projected) == 1
    value = projected[0]
    assert value.core["known_costs"]["raw_unit"] == "nanoseconds"
    assert value.core["known_costs"]["debit_raw"] == 1_000_200_000
    for scenario in value.core["scenarios"].values():
        assert scenario["net"]["value"] == -1 and scenario["complete"]
        assert "journal_active_proxy_ns_v2" in scenario["bounds"]["assumptions"]
    planned = export_records.plan_revisions(Registry(), projected)
    key = Key("authenticated-test-tenant", "authenticated-test-device", value.record_id)
    assert apply(ReceiverState(), key, planned[0].body)[1] is Outcome.APPLIED_NEW


def test_outward_bounds_do_not_round_a_nearly_two_second_debit_down_at_lower_net():
    source = candidate(
        "three-seconds", immediate=(atom("source", 4, t_start_ms=0, t_end_ms=3000, actor="one"),)
    )
    rows = estimates(cost_rows(timings((1_999_999_999,))), (source,))
    for row in rows:
        assert row.net.value == row.known_net == 2  # documented whole-unit point model
        assert (row.bounds.credit_lower.value, row.bounds.credit_upper.value) == (3, 3)
        assert (row.bounds.debit_lower.value, row.bounds.debit_upper.value) == (1, 2)
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (1, 2)


def test_explicit_missing_ns_bounds_round_outward_but_unbounded_debit_has_no_lower():
    source = candidate(
        "three-seconds", immediate=(atom("source", 4, t_start_ms=0, t_end_ms=3000, actor="one"),)
    )
    missing = Debit(
        "bounded-missing",
        TASK,
        DAY,
        "exposure",
        WORK,
        Quantity.unavailable(WORK.unit, "missing_duration"),
        raw_min=1,
        raw_max=1_999_999_999,
    )
    bounded = estimates((missing,), (source,))
    assert all(row.net.value is None and row.known_net == 3 for row in bounded)
    assert all(
        (row.bounds.net_lower.value, row.bounds.net_upper.value) == (1, 3) for row in bounded
    )
    unbounded = estimates((replace(missing, raw_max=None),), (source,))
    assert all(
        row.bounds.net_lower.value is None and row.bounds.net_upper.value == 3 for row in unbounded
    )


def test_character_proxy_bounds_enclose_subtoken_raw_endpoints():
    from ssgrep.accounting.valuation_inputs import EvidenceBundle
    from tests.accounting.valuation_builders import CHARS

    source = candidate("fractional-character-proxy", immediate=(atom("source", 3),))
    cost = Debit(
        "one-character",
        TASK,
        DAY,
        "exposure",
        CHARS,
        Quantity(0, CHARS.unit, Basis.ESTIMATED, CHARS.method),
        chars=1,
    )
    value = value_evidence(EvidenceBundle((source,), (cost,)), Methodology("chars", (CHARS,)))
    for row in rollup(value.leaves, CHARS):
        assert row.net.value == 0
        assert (row.bounds.credit_lower.value, row.bounds.credit_upper.value) == (0, 1)
        assert (row.bounds.debit_lower.value, row.bounds.debit_upper.value) == (0, 1)
        assert (row.bounds.net_lower.value, row.bounds.net_upper.value) == (-1, 1)


def test_actual_native_reducer_keeps_three_fractional_lookup_debits_and_copy_replay(tmp_path):
    from ssgrep.accounting.frequency import FrequencyView
    from ssgrep.accounting.reduce import methodology_for, reduce_profile
    from tests.accounting.reduce_builders import (
        KEY,
        MCP,
        PROFILE,
        claude_evidence,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        policy,
        results_json,
    )

    native = [claude_prompt("consumer", "inspect history", uuid="precision-user", ts=ts(12, 9))]
    events = []
    for i in range(3):
        call_id = f"precise-call-{i}"
        native.extend(
            (
                claude_tool_use(
                    "consumer",
                    call_id,
                    MCP,
                    {"query": f"precision-{i}"},
                    message_id=f"precise-message-{i}",
                    ts=ts(12, 10, i),
                ),
                claude_tool_result("consumer", call_id, results_json(), ts=ts(12, 10, i, 2)),
            )
        )
        events.extend(
            replace(item, mono_ns=333_400_000) if item.phase is Phase.COMPLETED else item
            for item in event(f"precise-execution-{i}", ts(12, 10, i, 1), text=f"precision-{i}")
        )
    evidence = claude_evidence(
        tmp_path, {"-work/original.jsonl": native, "-copy/copied.jsonl": native}
    )
    result = reduce_profile(
        PROFILE,
        evidence,
        events,
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )
    copied = reduce_profile(
        PROFILE,
        evidence + evidence,
        events + events,
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )
    assert copied.executions == result.executions and copied.bundles == result.bundles
    costs = [cost for bundle in result.bundles for cost in bundle.debits if cost.unit == WORK]
    assert len(costs) == 3 and {cost.work_ns for cost in costs} == {333_400_000}
    value = value_evidence(
        EvidenceBundle(debits=tuple(costs)), methodology_for(FrequencyView.unknown())
    )
    rows = rollup((leaf for leaf in value.leaves if leaf.unit == WORK), WORK)
    assert all(row.net.value == -1 and row.debit.value == 1 for row in rows)
    assert all((row.bounds.net_lower.value, row.bounds.net_upper.value) == (-2, -1) for row in rows)


def test_public_receipt_keeps_exact_raw_ns_bound_inputs_and_declared_units(tmp_path):
    import os

    from ssgrep.accounting import storage
    from ssgrep.accounting.collector_publish import receipt_document as stored_document
    from ssgrep.accounting.contracts import canonical_bytes
    from ssgrep.accounting.receipts import build_receipt
    from ssgrep.accounting.service_records import read_receipt
    from ssgrep.cli.savings_json import receipt_document as public_document

    source = candidate(
        "three-seconds",
        immediate=(atom("source", 4, t_start_ms=0, t_end_ms=3000, actor="one"),),
        exposure_ids=("lookup",),
    )
    costs = tuple(
        replace(cost, exposure_id="lookup") for cost in cost_rows(timings((1_999_999_999,)))
    )
    method = Methodology("literal-ns-fixture", (WORK,))
    value = value_evidence(EvidenceBundle((source,), costs), method)
    record = build_receipt(source, value.allocations, costs, method)
    stored = stored_document(record)
    assert stored["raw_unit"] == "nanoseconds"
    assert (
        stored["debit_raw"] == stored["debit_raw_min"] == stored["debit_raw_max"] == 1_999_999_999
    )
    assert dict(stored["credit_raw"]) == {
        scenario: 3_000_000_000 for scenario in ("conservative", "central", "optimistic")
    }
    directory = storage.open_profile_dir("personal-local", "receipts")
    try:
        storage.atomic_write(directory, record.receipt_id + ".json", canonical_bytes(stored))
    finally:
        os.close(directory)
    view = read_receipt("personal-local", record.receipt_id)
    assert view is not None
    public = public_document(view, "personal-local")["data"]
    assert isinstance(public, dict)
    assert public["raw_unit"] == "nanoseconds" and public["debit_raw_max"] == 1_999_999_999
    assert public["credit_raw"]["conservative"] == 3_000_000_000
    for scenario in public["scenarios"]:
        assert scenario["net"]["value"] == 2
        assert (
            scenario["bounds"]["net_lower"]["value"],
            scenario["bounds"]["net_upper"]["value"],
        ) == (1, 2)


def test_complete_fractional_point_still_renders_its_distinct_outward_bounds():
    from ssgrep.cli.savings_text import Doc, render_estimate_bounds

    rows = tuple(
        row.as_scenario_estimate() for row in estimates(cost_rows(timings((333_400_000,) * 3)))
    )
    assert all(row.complete for row in rows)
    doc = Doc(80)
    render_estimate_bounds(doc, rows)
    text = " ".join(doc.text().plain.split())
    assert "-2 s" in text and "-1 s" in text
    assert "Lower bound" in text and "Upper bound" in text


def test_legacy_sealed_metadata_cannot_manufacture_zero_latency_from_absent_ns():
    from ssgrep.accounting.windows import SealedDay, _observations

    old = {
        "day": "2026-10-12",
        "counts": {},
        "visible_known": 0,
        "visible_missing": 0,
        "rendered_known": 0,
        "rendered_missing": 0,
    }
    assert _observations((SealedDay.from_dict(old),))["measured_service_ms"]["value"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_daily_partial_source_ownership_keeps_raw_ns_and_receipt_nesting(reverse):
    from ssgrep.accounting.contracts import Reuse, Scenario
    from ssgrep.accounting.receipts import build_receipts

    relay = candidate(
        "relay",
        day="2026-11-01",
        reuse=Reuse.INSPECTED,
        explicit_relay=True,
        immediate=(),
        episode=(atom("relay-span", 7, t_start_ms=0, t_end_ms=1600, actor="shared"),),
    )
    acted = candidate(
        "acted",
        day="2026-11-02",
        immediate=(atom("acted-span", 3, t_start_ms=200, t_end_ms=1300, actor="shared"),),
    )
    candidates = (acted, relay) if reverse else (relay, acted)
    costs = (
        replace(cost_rows(timings((333_400_000,)))[0], day=relay.day, debit_id="early"),
        replace(cost_rows(timings((666_800_000,)))[0], day=acted.day, debit_id="late"),
    )
    method = Methodology("daily-ns-ownership", (WORK,))
    valued = value_evidence(EvidenceBundle(candidates, costs), method)
    leaves = {leaf.day: leaf for leaf in valued.leaves}
    assert [row.credit_raw for row in leaves[relay.day].scenarios] == [0, 0, 500_000_000]
    assert [row.credit_raw for row in leaves[acted.day].scenarios] == [1_100_000_000] * 3
    assert (leaves[relay.day].debit_raw, leaves[acted.day].debit_raw) == (
        333_400_000,
        666_800_000,
    )
    assert [
        (row.credit.value, row.debit.value, row.net.value) for row in rollup(valued.leaves, WORK)
    ] == [(1, 1, 0)] * 3
    receipts = {
        record.receipt.candidate_fingerprint: record
        for record in build_receipts(candidates, valued.allocations, costs, method).values()
    }
    assert dict(receipts["relay"].credit_raw) == {
        "conservative": 0,
        "central": 0,
        "optimistic": 500_000_000,
    }
    assert dict(receipts["acted"].credit_raw) == {
        scenario.value: 1_100_000_000 for scenario in Scenario
    }
    for leaf in leaves.values():
        for row in leaf.scenarios:
            assert row.credit_raw == sum(
                allocation.credit_raw
                for allocation in valued.allocations
                if allocation.day == leaf.day and allocation.scenario is row.scenario
            )
