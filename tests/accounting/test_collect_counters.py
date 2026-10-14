"""Live captured counters: once-only counting, labels frozen at capture, epochs, backfill apart."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ssgrep.accounting.contracts import (
    Attribution,
    Basis,
    ExecutionEvent,
    Exposure,
    InvocationClass,
    Operation,
    Outcome,
    Phase,
    Quantity,
    Reuse,
    Transport,
    canonical_bytes,
)
from ssgrep.accounting.counters import (
    BUCKET_BOUNDS_US,
    UNKNOWN_RUNTIME,
    BackfillSeries,
    ExecKey,
    Histogram,
    LiveCounters,
    OpKey,
    backfill_series,
    bucket_index,
    empty_counters,
    observe,
    prune,
    start_epoch,
    view,
)
from ssgrep.accounting.reduce_types import ExecutionRecord, InvocationRecord, summarize
from tests.accounting.conftest import build_event

T0 = "2026-11-01T00:00:00Z"
SEARCH_CLI = OpKey("search", "cli")


def completed(n: int, **overrides: object) -> ExecutionEvent:
    return build_event(execution_id=f"e{n:04d}", phase=Phase.COMPLETED, **overrides)


def rendered(n: int, value: int | None = 100, **overrides: object) -> ExecutionEvent:
    quantity = (
        Quantity.observed(value, "bytes", "mcp_json")
        if value is not None
        else Quantity.unavailable("bytes", "not_recorded")
    )
    return build_event(
        execution_id=f"e{n:04d}",
        phase=Phase.RENDERED,
        outcome=None,
        mono_ns=None,
        rendered_bytes=quantity,
        **overrides,
    )


def started(n: int) -> ExecutionEvent:
    return build_event(
        execution_id=f"e{n:04d}",
        phase=Phase.STARTED,
        outcome=None,
        mono_ns=None,
        rendered_bytes=None,
    )


def mixed_events() -> list[ExecutionEvent]:
    events: list[ExecutionEvent] = []
    for n in range(40):
        operation = (Operation.SEARCH, Operation.SHOW, Operation.NOTE)[n % 3]
        transport = (Transport.CLI, Transport.MCP)[n % 2]
        outcome = (Outcome.OK, Outcome.EMPTY, Outcome.ERROR, None)[n % 4]
        mono = None if n % 7 == 0 else (n * 997_000) % 40_000_000
        events.append(started(n))
        events.append(
            completed(n, operation=operation, transport=transport, outcome=outcome, mono_ns=mono)
        )
        events.append(
            rendered(n, None if n % 5 == 0 else n * 10, operation=operation, transport=transport)
        )
    return events


def fresh() -> LiveCounters:
    return empty_counters(1, T0)


def total(counters: LiveCounters) -> int:
    return sum(count for _, count in counters.executions)


def test_property_any_order_batches_and_repeats_equal_one_pass() -> None:
    events = mixed_events()
    reference = observe(fresh(), events, horizon_day=None)
    assert total(reference) == 40  # only completed events count; 80 other events added nothing
    for seed in range(25):
        rng = random.Random(seed)
        shuffled = events + rng.sample(events, 30)  # duplicates inside the stream
        rng.shuffle(shuffled)
        counters = fresh()
        cut = sorted(rng.sample(range(len(shuffled)), 4))
        batches = [shuffled[a:b] for a, b in zip([0, *cut], [*cut, len(shuffled)], strict=True)]
        previous_total = 0
        for batch in batches:
            counters = observe(counters, batch, horizon_day=None)
            counters = observe(counters, batch, horizon_day=None)  # repeated batch
            assert total(counters) >= previous_total  # never decreases
            previous_total = total(counters)
        assert counters == reference
        assert canonical_bytes(counters.to_dict()) == canonical_bytes(reference.to_dict())


def test_only_completed_phase_increments_executions_and_labels_come_from_capture() -> None:
    counters = observe(
        fresh(),
        [
            started(1),
            rendered(1, 50),
            completed(1, outcome=Outcome.EMPTY, transport=Transport.MCP, operation=Operation.SHOW),
            completed(2, outcome=None),
        ],
        horizon_day=None,
    )
    assert counters.executions == (
        (ExecKey("search", "cli", "unknown", UNKNOWN_RUNTIME), 1),
        (ExecKey("show", "mcp", "empty", UNKNOWN_RUNTIME), 1),
    )
    only_started = observe(fresh(), [started(1)], horizon_day=None)
    assert only_started.executions == () and only_started.counted == ()


def test_runtime_label_never_moves_when_an_attribution_appears_elsewhere() -> None:
    event = completed(1)
    counters = observe(fresh(), [event], horizon_day=None)
    persisted = canonical_bytes(counters.to_dict())
    # Later the reducer attributes the very same execution to a runtime (ledger/gauge side).
    record = ExecutionRecord(
        execution_id=event.execution_id, ts_utc=event.ts_utc, day=event.ts_utc[:10],
        operation="search", transport="cli", outcome=Outcome.OK, outcome_basis="journal",
        attribution=Attribution.EXPLICIT, source="both", runtime="claude", session_id="s1",
        invocation_id="inv1", actor="agent", task=None,
        rendered_bytes=Quantity.observed(812, "bytes", "mcp_json"),
        visible_bytes=Quantity.observed(812, "bytes", "mcp_json"),
        exposure=Exposure.OBSERVED, reuse=Reuse.NONE_OBSERVED, timing_known=True, in_scope=True,
    )  # fmt: skip
    gauges = summarize([record])
    assert gauges["runtime.claude"].value == 1 and "runtime.unknown" not in gauges
    again = observe(counters, [event, rendered(1)], horizon_day=None)
    assert [key.runtime for key, _ in again.executions] == [UNKNOWN_RUNTIME]
    assert [count for _, count in again.executions] == [1]
    assert canonical_bytes(observe(counters, [event], horizon_day=None).to_dict()) == persisted
    assert json.loads(persisted)["executions"] == [
        {
            "operation": "search",
            "transport": "cli",
            "outcome": "ok",
            "runtime": "unknown",
            "count": 1,
        }
    ]


@pytest.mark.parametrize(
    ("duration_us", "index"),
    [
        (0, 0),
        (999, 0),
        (1_000, 0),
        (1_001, 1),
        (5_000, 1),
        (5_001, 2),
        (500_000, 5),
        (500_001, 6),
        (5_000_000, 7),
        (5_000_001, 8),
        (30_000_000, 8),
        (30_000_001, 9),
        (900_000_000, 9),
    ],
)
def test_bucket_boundaries_are_inclusive(duration_us: int, index: int) -> None:
    assert bucket_index(duration_us) == index
    counters = observe(fresh(), [completed(1, mono_ns=duration_us * 1000)], horizon_day=None)
    histogram = dict(counters.durations)[SEARCH_CLI]
    assert histogram.buckets[index] == 1 and sum(histogram.buckets) == 1
    assert histogram.sum_us == duration_us


def test_bounds_are_the_documented_ten_buckets() -> None:
    assert BUCKET_BOUNDS_US == (
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
        5_000_000,
        30_000_000,
    )
    assert len(Histogram().buckets) == len(BUCKET_BOUNDS_US) + 1


def test_histogram_count_sum_exact_and_unknown_duration_kept_apart() -> None:
    durations_us = [1_000, 1_001, 7, 30_000_001, 123_456]
    events = [
        completed(i, mono_ns=us * 1000 + 999) for i, us in enumerate(durations_us)
    ]  # ns remainder floors
    events.append(completed(90, mono_ns=None))
    counters = observe(fresh(), events, horizon_day=None)
    histogram = dict(counters.durations)[SEARCH_CLI]
    assert histogram.count == sum(histogram.buckets) == 5
    assert histogram.sum_us == sum(durations_us)
    assert histogram.buckets[0] == 2 and histogram.buckets[9] == 1
    assert dict(counters.duration_unknown) == {SEARCH_CLI: 1}
    assert total(counters) == 6  # the unknown-duration execution is still an execution
    only_unknown = observe(fresh(), [completed(1, mono_ns=None)], horizon_day=None)
    assert only_unknown.durations == () and dict(only_unknown.duration_unknown) == {SEARCH_CLI: 1}


def test_rendered_known_adds_bytes_unknown_bumps_marker_never_zero() -> None:
    events = [
        rendered(1, 100),
        rendered(2, 250),
        rendered(3, None),
        rendered(4, None, operation=Operation.SHOW),
        build_event(execution_id="e5", phase=Phase.RENDERED, rendered_bytes=None),
    ]
    counters = observe(fresh(), events, horizon_day=None)
    assert dict(counters.rendered_bytes) == {SEARCH_CLI: 350}
    assert dict(counters.rendered_unknown) == {SEARCH_CLI: 2, OpKey("show", "cli"): 1}


def test_each_execution_phase_counts_once_even_with_conflicting_repeat() -> None:
    first = completed(1, mono_ns=2_000_000)
    counters = observe(fresh(), [first], horizon_day=None)
    redelivered = replace(first, mono_ns=9_000_000_000, outcome=Outcome.ERROR)
    assert observe(counters, [redelivered], horizon_day=None) == counters
    assert ("e0001|completed", "2026-11-01") in counters.counted


def test_roundtrip_is_byte_identical_and_restart_counts_nothing_twice() -> None:
    events = mixed_events()
    counters = observe(fresh(), events, horizon_day="2026-10-01")
    data = counters.to_dict()
    restored = LiveCounters.from_dict(json.loads(canonical_bytes(data)))
    assert restored == counters
    assert canonical_bytes(restored.to_dict()) == canonical_bytes(data)
    after_restart = observe(restored, events, horizon_day="2026-10-01")
    assert after_restart == counters and after_restart.epoch == 1
    extra = completed(500)
    grown = observe(restored, [*events, extra], horizon_day=None)
    assert total(grown) == total(counters) + 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(epoch=1.0),
        lambda d: d.update(epoch=0),
        lambda d: d.update(skipped_expired=-1),
        lambda d: d.update(skipped_expired=True),
        lambda d: d.update(epoch_started_at="yesterday"),
        lambda d: d.update(extra=1),
        lambda d: d["executions"][0].update(count=1.5),
        lambda d: d["executions"][0].pop("runtime"),
        lambda d: d["durations"][0]["histogram"].update(count=99),
        lambda d: d["durations"][0]["histogram"]["buckets"].pop(),
        lambda d: d["counted"].append(["no-phase-separator", "2026-11-01"]),
        lambda d: d["counted"].append(["e1|bogus", "2026-11-01"]),
        lambda d: d["counted"].append(["e1|completed", "11/01/2026"]),
        lambda d: d["executions"].append(dict(d["executions"][0])),
        lambda d: d["rendered_bytes"].append(dict(d["rendered_bytes"][0])),
        lambda d: d["durations"].append(dict(d["durations"][0])),
        lambda d: d["counted"].append(list(d["counted"][0])),
        lambda d: d["counted"].append(["e1|completed"]),
    ],
)
def test_from_dict_rejects_garbage(mutate) -> None:
    data = json.loads(canonical_bytes(observe(fresh(), mixed_events(), horizon_day=None).to_dict()))
    mutate(data)
    with pytest.raises(ValueError):
        LiveCounters.from_dict(data)


def test_from_dict_rejects_float_literals_in_json_text() -> None:
    text = canonical_bytes(observe(fresh(), [completed(1)], horizon_day=None).to_dict()).decode()
    assert '"count":1,' in text
    assert LiveCounters.from_dict(json.loads(text)).executions  # the untouched text is valid
    for literal in ("1.5", "1.0", "1e0"):
        with pytest.raises(ValueError):
            LiveCounters.from_dict(json.loads(text.replace('"count":1,', f'"count":{literal},', 1)))
    with pytest.raises(ValueError):
        LiveCounters.from_dict("not an object")


def test_start_epoch_zeroes_series_and_known_events_never_count_again() -> None:
    old = mixed_events()
    previous = observe(fresh(), old, horizon_day=None)
    now = datetime(2026, 11, 5, 12, 30, 15, tzinfo=UTC)
    epoch2 = start_epoch(previous, old, now)
    assert epoch2.epoch == 2 and epoch2.epoch_started_at == "2026-11-05T12:30:15Z"
    assert (epoch2.executions, epoch2.rendered_bytes, epoch2.rendered_unknown) == ((), (), ())
    assert (epoch2.durations, epoch2.duration_unknown, epoch2.skipped_expired) == ((), (), 0)
    assert set(previous.counted) <= set(epoch2.counted)
    replayed = observe(epoch2, old, horizon_day=None)
    assert replayed == epoch2  # captured before the call: never counts in the new epoch
    fresh_event = completed(777, mono_ns=3_000_000)
    after = observe(replayed, [*old, fresh_event], horizon_day=None)
    assert dict(after.executions) == {ExecKey("search", "cli", "ok", "unknown"): 1}
    assert dict(after.durations)[SEARCH_CLI].count == 1
    assert previous.epoch == 1 and total(previous) == 40  # previous value untouched (immutable)


def test_start_epoch_counts_events_known_only_through_the_argument_and_first_epoch() -> None:
    known_only = [completed(1), started(2)]
    epoch2 = start_epoch(fresh(), known_only, datetime(2026, 11, 5, tzinfo=UTC))
    assert {key for key, _ in epoch2.counted} == {"e0001|completed", "e0002|started"}
    first = start_epoch(None, known_only, datetime(2026, 11, 5, tzinfo=UTC))
    assert first.epoch == 1 and first.executions == ()
    assert observe(first, known_only, horizon_day=None).executions == ()
    with pytest.raises(ValueError):
        start_epoch(None, [], datetime(2026, 11, 5))  # naive clock value is ambiguous


def test_horizon_skips_old_events_and_counts_them_as_expired() -> None:
    old = completed(1, ts_utc="2026-09-30T23:59:59Z")
    edge = completed(2, ts_utc="2026-10-01T00:00:00Z")
    new = completed(3, ts_utc="2026-11-01T01:00:00Z")
    counters = observe(fresh(), [old, old, edge, new], horizon_day="2026-10-01")
    assert (
        total(counters) == 2 and counters.skipped_expired == 1
    )  # duplicates in one pass skipped once
    assert {key for key, _ in counters.counted} == {"e0002|completed", "e0003|completed"}
    # Expired events are not remembered: with a looser horizon they still count (once).
    loosened = observe(counters, [old], horizon_day=None)
    assert total(loosened) == 3


def test_prune_drops_only_old_counted_keys_and_keeps_totals() -> None:
    events = [
        completed(1, ts_utc="2026-09-01T00:00:00Z"),
        completed(2, ts_utc="2026-10-15T00:00:00Z"),
        completed(3, ts_utc="2026-11-01T00:00:00Z"),
    ]
    counters = observe(fresh(), events, horizon_day=None)
    pruned = prune(counters, "2026-10-15")
    assert [key for key, _ in pruned.counted] == ["e0002|completed", "e0003|completed"]
    assert (
        replace(pruned, counted=counters.counted) == counters
    )  # everything but `counted` unchanged
    assert prune(pruned, "2026-10-15") == pruned
    assert prune(counters, "2026-01-01") == counters
    # Pruned keys behind the horizon cannot recount when the same horizon guards observe.
    again = observe(pruned, events, horizon_day="2026-10-15")
    assert total(again) == 3 and again.skipped_expired == 1


def test_view_is_integer_only_sorted_with_cumulative_le_buckets() -> None:
    events = [
        completed(1, mono_ns=1_000_000),
        completed(2, mono_ns=1_001_000),
        completed(3, mono_ns=40_000_000_000),
        completed(4, operation=Operation.SHOW, transport=Transport.MCP, mono_ns=None),
        rendered(1, 70),
        rendered(2, None, operation=Operation.NOTE),
    ]
    counters = observe(fresh(), events, horizon_day=None)
    document = view(counters)
    assert json.loads(canonical_bytes(document)) == document
    assert document["epoch"] == 1 and document["epoch_started_at"] == T0
    assert [(r["operation"], r["transport"], r["count"]) for r in document["executions"]] == [
        ("search", "cli", 3),
        ("show", "mcp", 1),
    ]
    assert document["rendered_bytes"] == [{"operation": "search", "transport": "cli", "bytes": 70}]
    assert document["rendered_unknown"] == [{"operation": "note", "transport": "cli", "count": 1}]
    assert document["duration_unknown"] == [{"operation": "show", "transport": "mcp", "count": 1}]
    (series,) = document["duration"]
    assert series["count"] == 3 and series["sum_us"] == 1_000 + 1_001 + 40_000_000
    assert [(b["le_us"], b["count"]) for b in series["buckets"]] == [
        ("1000", 1),
        ("5000", 2),
        ("10000", 2),
        ("50000", 2),
        ("100000", 2),
        ("500000", 2),
        ("1000000", 2),
        ("5000000", 2),
        ("30000000", 2),
        ("+Inf", 3),
    ]
    assert view(observe(fresh(), list(reversed(events)), horizon_day=None)) == document


# --- backfill -----------------------------------------------------------------------------


def invocation(
    n: int,
    ts: str | None,
    operations: tuple[str, ...] = ("search",),
    cls=InvocationClass.TRANSCRIPT_ONLY,
    cardinality: str = "single_call",
) -> InvocationRecord:
    return InvocationRecord(
        invocation_id=f"inv{n}",
        runtime="claude",
        session_id="s",
        call_id=f"c{n}",
        args_hash="h",
        ts_utc=ts,
        operations=operations,
        route="mcp",
        cardinality=cardinality,
        cls=cls,
        attribution=Attribution.UNATTRIBUTED,
        execution_ids=(),
        ambiguous_execution_ids=(),
        outcome=Outcome.UNKNOWN,
        exposure=Exposure.UNAVAILABLE,
        visible_bytes=Quantity(None, "bytes", Basis.UNAVAILABLE, reason="not_recorded"),
        task=None,
        actor="agent",
    )


def test_backfill_counts_half_open_interval_keyed_by_operation_and_class() -> None:
    since, until = "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"
    records = [
        invocation(1, since),  # inclusive start
        invocation(2, "2026-01-31T23:59:59Z", ("show",), InvocationClass.MATCHED),
        invocation(3, until),  # exclusive end
        invocation(4, "2025-12-31T23:59:59Z"),
        invocation(5, None),  # cannot be placed in the interval
        invocation(
            6,
            "2026-01-15T00:00:00Z",
            ("search", "show"),
            InvocationClass.TRANSCRIPT_ONLY,
            "unknown",
        ),
        invocation(7, "2026-01-16T00:00:00Z", (), InvocationClass.MENTION),
        invocation(8, "2026-01-17T00:00:00Z", ("search",), InvocationClass.JOURNAL_GAP),
        invocation(1, since),  # same invocation seen twice
    ]
    series = backfill_series(records, since, until, "2026-11-02T00:00:00Z")
    assert dict(series.counts) == {
        "search:transcript_only": 2,
        "show:matched": 1,
        "show:transcript_only": 1,
        "search:journal_gap": 1,
    }
    assert series.counts == tuple(sorted(series.counts))
    assert series.unknown_cardinality == 1
    assert (series.since, series.until, series.completed_at) == (
        since,
        until,
        "2026-11-02T00:00:00Z",
    )
    assert backfill_series(list(reversed(records)), since, until, "2026-11-02T00:00:00Z") == series
    restored = BackfillSeries.from_dict(json.loads(canonical_bytes(series.to_dict())))
    assert restored == series and canonical_bytes(restored.to_dict()) == canonical_bytes(
        series.to_dict()
    )
    with pytest.raises(ValueError):
        BackfillSeries.from_dict(
            json.loads(canonical_bytes(series.to_dict()).replace(b":2", b":2.5", 1))
        )
    with pytest.raises(ValueError):
        backfill_series(records, until, since, "2026-11-02T00:00:00Z")
    with pytest.raises(ValueError):
        LiveCounters.from_dict(series.to_dict())  # a backfill series is never a live counter


def test_backfill_does_not_touch_live_counters_or_create_a_burst() -> None:
    live = observe(fresh(), [completed(1), completed(2)], horizon_day=None)
    before = canonical_bytes(live.to_dict())
    view_before = canonical_bytes(view(live))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    records = [
        invocation(n, (start + timedelta(seconds=n * 60)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        for n in range(10_000)
    ]
    series = backfill_series(
        records, "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z", "2026-11-02T00:00:00Z"
    )
    assert dict(series.counts) == {"search:transcript_only": 10_000}
    assert not isinstance(series, LiveCounters) and not isinstance(live, BackfillSeries)
    assert canonical_bytes(live.to_dict()) == before and canonical_bytes(view(live)) == view_before
    assert total(live) == 2  # 10,000 imported invocations produced no live executions
    assert total(observe(live, [], horizon_day=None)) == 2
