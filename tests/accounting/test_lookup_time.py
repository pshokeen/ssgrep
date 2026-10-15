"""V08: measured latency, explicitly modeled work, unique debit and honest nets."""

from dataclasses import replace

import pytest

from ssgrep.accounting.contracts import Basis, Phase, Quantity, Reuse, Scenario, TaskId
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.ledger_contracts import SourceAtom
from ssgrep.accounting.lookup_time import (
    LOOKUP_WORK_ASSUMPTIONS,
    LOOKUP_WORK_MAPPING,
    journal_timings,
    link_call_timing,
)
from ssgrep.accounting.reconcile import Request
from ssgrep.accounting.reduce import methodology_for, reduce_profile
from ssgrep.accounting.reduce_cands import CHARS, WORK
from ssgrep.accounting.timeunits import ActorInterval, capped_gap_v1, work_time
from ssgrep.accounting.units import WORK_SECONDS, UnitSpec
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import Candidate, Debit, EvidenceBundle, ValuationAtom
from tests.accounting.reduce_builders import (
    KEY,
    MCP,
    PROFILE,
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    event,
    policy,
    results_json,
    ts,
)
from tests.accounting.test_reduce_evidence import FACT, VIEW, consumer


def timed_events(execution_id="service-1", milliseconds=2000, **kwargs):
    return [
        replace(item, mono_ns=milliseconds * 1_000_000) if item.phase is Phase.COMPLETED else item
        for item in event(execution_id, ts(12), **kwargs)
    ]


def request(capacity=1):
    return Request(
        "lookup-1", 0, "search", "mcp", None, None, None, (0, 3000), capacity, None, None
    )


def linked(events, capacity=1, owned_ids=("service-1",)):
    req = request(capacity)
    return link_call_timing((req,), {req.key: list(owned_ids)}, {}, journal_timings(events))


def test_duplicate_journal_phases_preserve_two_second_latency_once():
    events = timed_events()
    timing = linked(events + events)
    assert timing.complete
    assert len(timing.rows) == 1
    assert timing.measured_service_ms == Quantity(2000, "work_ms", Basis.OBSERVED, "monotonic_v1")
    assert timing.work == Quantity(2, WORK_SECONDS, Basis.ESTIMATED, "capped_gap_v1")
    assert timing.mapping_version == LOOKUP_WORK_MAPPING == "journal_active_proxy_ns_v2"
    assert timing.assumptions == LOOKUP_WORK_ASSUMPTIONS
    assert timing.rows[0].work_ns == 2_000_000_000
    assert timing.rows[0].measured_service_ns.value == 2_000_000_000


@pytest.mark.parametrize("milliseconds", [0, 299000, 300000, 301000, 28801000])
def test_lookup_proxy_caps_each_operation_but_never_changes_measured_latency(milliseconds):
    timing = linked(timed_events(milliseconds=milliseconds))
    assert timing.measured_service_ms.value == milliseconds
    assert timing.work.value == min(milliseconds, 300000) // 1000
    assert timing.work.basis is Basis.ESTIMATED


def test_wall_discontinuity_does_not_rewrite_monotonic_service_duration():
    events = timed_events()
    events = [replace(e, ts_utc=ts(12, 8)) if e.phase is Phase.COMPLETED else e for e in events]
    assert linked(events).measured_service_ms.value == 2000


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "journal_monotonic_duration_missing"),
        ("reversed", "monotonic_clock_reversed"),
        ("conflict", "conflicting_journal_timing"),
    ],
)
def test_unknown_journal_timing_never_becomes_zero(case, reason):
    events = timed_events()
    if case == "missing":
        events = [replace(e, mono_ns=None) if e.phase is Phase.COMPLETED else e for e in events]
    elif case == "reversed":
        events = timed_events(milliseconds=-1)
    else:
        events += timed_events(milliseconds=3000)
    timing = linked(events)
    assert not timing.complete
    assert timing.measured_service_ms.value is None
    assert timing.work.value is None
    assert timing.rows[0].measured_service_ms.reason == reason


def test_unbounded_loop_retains_supported_rows_and_unknown_total():
    timing = linked(timed_events(), capacity=None)
    assert len(timing.rows) == 1 and timing.rows[0].measured_service_ms.value == 2000
    assert not timing.complete
    assert timing.measured_service_ms.value is None and timing.work.value is None
    assert timing.reason == "journal_cardinality_unbounded"


def test_ambiguous_join_never_assigns_journal_time_to_lookup():
    req = request()
    timing = link_call_timing((req,), {}, {req.key: ["service-1"]}, journal_timings(timed_events()))
    assert timing.rows == ()
    assert timing.work.value is None and timing.reason == "ambiguous_journal_timing"


def records():
    return [
        claude_prompt("consumer", "find deployment history", uuid="timing-prompt", ts=ts(12, 9)),
        claude_tool_use(
            "consumer",
            "call-one",
            MCP,
            {"query": "deploy failure"},
            message_id="timing-message-one",
            ts=ts(12),
        ),
        claude_tool_result("consumer", "call-one", results_json(), ts=ts(12, s=2)),
    ]


def reduce_timing(root, evidence, events):
    return reduce_profile(
        PROFILE,
        evidence,
        events,
        policy=policy(),
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
    )


def test_runtime_copy_backfill_replay_preserves_unique_duration_and_work_debit(tmp_path):
    native = records()
    evidence = claude_evidence(
        tmp_path, {"-work/original.jsonl": native, "-work-copy/copied.jsonl": native}
    )
    events = timed_events()
    original = reduce_timing(tmp_path, evidence, events)
    replay = reduce_timing(tmp_path, evidence + evidence, events + events)
    assert len(original.executions) == len(original.invocations) == 1
    assert original.executions[0].measured_service_ms.value == 2000
    assert original.invocations[0].measured_service_ms.value == 2000
    assert replay.executions == original.executions and replay.bundles == original.bundles
    debits = [d for b in replay.bundles for d in b.debits if d.unit == WORK]
    assert len(debits) == 1 and debits[0].debit_id == "service-1:work"
    assert debits[0].amount.value == 2 and debits[0].work_ns == 2_000_000_000
    assert debits[0].mapping_version == LOOKUP_WORK_MAPPING


def test_genuine_repeated_lookup_adds_debit_despite_identical_output(tmp_path):
    native = records() + [
        claude_tool_use(
            "consumer",
            "call-two",
            MCP,
            {"query": "deploy failure"},
            message_id="timing-message-two",
            ts=ts(12, m=1),
        ),
        claude_tool_result("consumer", "call-two", results_json(), ts=ts(12, m=1, s=2)),
    ]
    evidence = claude_evidence(tmp_path, {"-work/consumer.jsonl": native})
    second = [replace(e, ts_utc=ts(12, m=1)) for e in timed_events("service-2")]
    reduction = reduce_timing(tmp_path, evidence, timed_events() + second)
    debits = [d for b in reduction.bundles for d in b.debits if d.unit == WORK]
    assert {d.debit_id for d in debits} == {"service-1:work", "service-2:work"}
    assert sum(d.work_ns for d in debits) == 4_000_000_000


def test_missing_journal_never_uses_later_user_prompt_as_duration(tmp_path):
    evidence = claude_evidence(
        tmp_path,
        {
            "-work/consumer.jsonl": records()
            + [
                claude_prompt("consumer", "a new task tomorrow", uuid="later-prompt", ts=ts(13)),
            ]
        },
    )
    reduction = reduce_timing(tmp_path, evidence, [])
    assert reduction.invocations[0].measured_service_ms.value is None
    debits = [d for b in reduction.bundles for d in b.debits if d.unit == WORK]
    assert len(debits) == 1 and debits[0].amount.value is None


def atom(name, start, end, actor="claude:source"):
    origin = f"{actor}:result:{name}"
    return ValuationAtom(
        SourceAtom(origin, origin, "digest-" + name, 40), 0, 40, start, end, actor=actor
    )


def time_bundle(milliseconds=2000, missing=False):
    task = TaskId("claude:consumer", "turn:one", "main")
    candidate = Candidate(
        "fact-1",
        "identifier",
        task,
        "2026-10-12",
        Reuse.ACTION_REFERENCE,
        True,
        False,
        True,
        immediate=(atom("immediate", 0, 10),),
        discovery=(atom("discovery", 0, 20),),
        episode=(atom("episode", 0, 30),),
    )
    timing = linked(timed_events(milliseconds=milliseconds))
    row = timing.rows[0]
    debit = Debit(
        "service-1:work",
        task,
        "2026-10-12",
        "exposure",
        WORK,
        row.work,
        work_ns=row.work_ns,
        mapping_version=timing.mapping_version,
        assumptions=timing.assumptions,
    )
    if missing:
        debit = replace(
            debit, amount=Quantity.unavailable(WORK_SECONDS, "journal_missing"), work_ns=None
        )
    context = Debit(
        "lookup-1",
        task,
        "2026-10-12",
        "exposure",
        CHARS,
        Quantity(10, "context_tokens", Basis.ESTIMATED, "chars_div4_v1"),
        chars=40,
    )
    return EvidenceBundle((candidate,), (debit, debit, context))


def unit_estimates(leaves, unit):
    return rollup((leaf for leaf in leaves if leaf.unit == unit), unit)


def test_evidenced_source_intervals_support_three_nets_with_one_common_debit():
    bundle = time_bundle()
    valuation = value_evidence(bundle, methodology_for(FrequencyView.unknown()))
    estimates = unit_estimates(valuation.leaves, WORK)
    assert [e.scenario for e in estimates] == list(Scenario)
    assert [e.credit.value for e in estimates] == [10, 20, 30]
    assert [e.net.value for e in estimates] == [8, 18, 28]
    assert [e.debit.value for e in estimates] == [2, 2, 2]
    assert [e.known_debits for e in estimates] == [1, 1, 1]
    token_estimates = unit_estimates(valuation.leaves, CHARS)
    assert all(e.net.value is not None for e in token_estimates)


def test_native_reducer_source_and_journal_produce_three_complete_time_nets(tmp_path):
    # Source clocks declared independently: immediate [5,8), discovery [0,8),
    # episode [0,10). The terminal result is empty, so it invents no final work.
    source = [
        claude_prompt("src1", "why does deployment fail", uuid="source-prompt", ts=ts(1)),
        claude_tool_use(
            "src1",
            "w1",
            "Bash",
            {"command": "cat deploy.log"},
            message_id="source-one",
            ts=ts(1, s=5),
        ),
        claude_tool_result("src1", "w1", FACT, ts=ts(1, s=6)),
        claude_tool_use(
            "src1", "w2", "Bash", {"command": "true"}, message_id="source-two", ts=ts(1, s=8)
        ),
        claude_tool_result("src1", "w2", "", ts=ts(1, s=10)),
    ]
    evidence = claude_evidence(
        tmp_path,
        {
            "-work/src1.jsonl": source,
            "-work/cons.jsonl": consumer(query="deploy failure"),
        },
    )
    reduction = reduce_profile(
        PROFILE, evidence, timed_events(), policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    estimates = unit_estimates(valuation.leaves, WORK)
    assert [e.credit.value for e in estimates] == [3, 8, 10]
    assert [e.net.value for e in estimates] == [1, 6, 8]
    assert [e.debit.value for e in estimates] == [2, 2, 2]
    assert all(e.known_debits == 1 and e.complete for e in estimates)
    assert all(e.net.value is not None for e in unit_estimates(valuation.leaves, CHARS))


def test_unique_subsecond_operation_debits_floor_only_after_summing():
    bundle = time_bundle(milliseconds=700)
    first = bundle.debits[0]
    second = replace(first, debit_id="service-2:work")
    valuation = value_evidence(
        replace(bundle, debits=(first, first, second)), methodology_for(FrequencyView.unknown())
    )
    assert [e.debit.value for e in unit_estimates(valuation.leaves, WORK)] == [1, 1, 1]


def test_partial_timing_keeps_all_three_time_nets_unavailable():
    estimates = unit_estimates(
        value_evidence(time_bundle(missing=True), methodology_for(FrequencyView.unknown())).leaves,
        WORK,
    )
    assert all(e.net.value is None for e in estimates)


def test_negative_time_net_is_not_forced_positive():
    estimates = unit_estimates(
        value_evidence(
            time_bundle(milliseconds=25000), methodology_for(FrequencyView.unknown())
        ).leaves,
        WORK,
    )
    assert [e.net.value for e in estimates] == [-15, -5, 5]


def test_source_overlap_unions_per_actor_while_parallel_actors_sum_work():
    bundle = time_bundle()
    candidate = replace(
        bundle.candidates[0],
        immediate=(
            atom("one", 0, 10, "claude:a"),
            atom("overlap", 0, 10, "claude:a"),
            atom("parallel", 0, 10, "claude:b"),
        ),
        discovery=(),
        episode=(),
    )
    estimates = unit_estimates(
        value_evidence(
            replace(bundle, candidates=(candidate, candidate)),
            methodology_for(FrequencyView.unknown()),
        ).leaves,
        WORK,
    )
    assert estimates[0].credit.value == 20
    intervals = [
        ActorInterval("a", 0, 10000),
        ActorInterval("a", 0, 10000),
        ActorInterval("b", 0, 10000),
    ]
    duration = work_time(intervals)
    assert duration.work_seconds.value == 20 and duration.elapsed_seconds.value == 10
    assert duration.human_time.value is None


def test_capped_gaps_are_estimates_and_missing_clocks_are_unknown():
    assert capped_gap_v1([0, 301000]).quantity.basis is Basis.ESTIMATED
    assert capped_gap_v1([0, None, 3000]).quantity.value is None


def test_unsupported_time_mapping_cannot_enter_supported_debit_set():
    bundle = time_bundle()
    foreign = replace(bundle.debits[0], unit=UnitSpec(WORK_SECONDS, "monotonic_v1"))
    from ssgrep.accounting.units import UnitMismatchError

    with pytest.raises(UnitMismatchError):
        value_evidence(replace(bundle, debits=(foreign,)), methodology_for(FrequencyView.unknown()))


def test_incompatible_source_work_is_unknown_not_measured_time_credit():
    bundle = time_bundle()
    candidate = bundle.candidates[0]
    foreign = replace(candidate.immediate[0], work_method="monotonic_v1")
    candidate = replace(candidate, immediate=(foreign,))
    estimates = unit_estimates(
        value_evidence(
            replace(bundle, candidates=(candidate,)), methodology_for(FrequencyView.unknown())
        ).leaves,
        WORK,
    )
    assert estimates[0].net.value is None
    assert (foreign.atom_id, "source_work_method_incompatible") in estimates[0].credit_missing
