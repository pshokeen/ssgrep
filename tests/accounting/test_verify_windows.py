"""9.4/9.5: rolling event-time windows and live counters through the real collector.

* Clock-only collections expire an old observation from ``24h`` while ``7d``/``30d``/lifetime and
  the durable live counters keep their supported values (and the snapshot id moves only because
  the 24h window content moved).
* Live counters keep their capture-time ``unknown`` runtime label after attribution, never
  decrease across cycles/restarts, keep the epoch across an ordinary restart, and reset only on a
  declared ``--new-epoch`` (future events only, ledger history kept). Backfill is a separate
  series. Histograms use the fixed ten buckets with count+sum in integer microseconds.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import counters as counters_module
from ssgrep.accounting.collector_types import CollectOptions
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import MCP, claude_prompt, ts

CLOCK = scene.at(hour=9)
DAY = 12
MARK = 1  # the capture-start marker is itself one real (cli search) execution on day 11, 08:00


def later(days: int, hour: int = 12) -> Any:
    return scene.at(DAY, hour) + timedelta(days=days)


def observation(envelope: Any, window: str, name: str) -> int | None:
    return scene.window(envelope, window)["observations"][name]["value"]


def counters(envelope: Any) -> dict[str, Any]:
    return scene.observations(envelope)["counters"]


def total_executions(envelope: Any) -> int:
    return sum(row["count"] for row in counters(envelope)["executions"])


# ---- 9.4 rolling windows ---------------------------------------------------------------------


def test_clock_only_collections_expire_the_old_observation_from_24h_only(tmp_path: Path) -> None:
    vs.enable()
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)  # one search on day 12 at 09:00:11
    first = scene.collect(scene.at(hour=10))
    assert first.state == "published"
    fresh = scene.current()
    assert observation(fresh, "24h", "searches") == 1  # the day-11 marker is already outside 24h
    assert observation(fresh, "lifetime", "searches") == 1 + MARK
    assert total_executions(fresh) == 1 + MARK and counters(fresh)["epoch"] == 1

    # the clock advances 19 hours: still inside 24h (no new evidence, nothing changes)
    scene.collect(later(1, 4))
    assert observation(scene.current(), "24h", "searches") == 1

    # past the 24 h boundary of the 09:00:11 event: ONLY the clock moved, the window changes
    before = vs.evidence_identities()
    result = scene.collect(later(1, 10))
    stale = scene.current()
    assert scene.report(result).new_evidence == 0 and vs.evidence_identities() == before
    assert observation(stale, "24h", "searches") == 0  # expired from the rolling window
    assert scene.window(stale, "24h")["observations"]["searches"]["basis"] == "observed"
    assert observation(stale, "7d", "searches") == 1 + MARK  # longer windows keep the values
    assert observation(stale, "30d", "searches") == 1 + MARK
    assert observation(stale, "lifetime", "searches") == 1 + MARK
    assert total_executions(stale) == 1 + MARK  # durable live counters never decreased
    assert stale.snapshot_id != fresh.snapshot_id  # the 24h window content really moved
    assert stale.generation == fresh.generation + 2

    # 8 days on: 7d drops both, 30d and lifetime keep them
    scene.collect(later(8))
    week = scene.current()
    assert observation(week, "7d", "searches") == 0
    assert observation(week, "30d", "searches") == 1 + MARK
    assert observation(week, "lifetime", "searches") == 1 + MARK
    assert total_executions(week) == 1 + MARK

    # 31 days on: only lifetime and the live counters remain
    scene.collect(later(31))
    month = scene.current()
    assert observation(month, "30d", "searches") == 0
    assert observation(month, "lifetime", "searches") == 1 + MARK
    assert total_executions(month) == 1 + MARK


def test_membership_uses_event_time_not_collection_time(tmp_path: Path) -> None:
    """A late collection of an old event never relabels it as recent."""
    vs.enable()
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)  # event at day 12 09:10:11
    scene.collect(later(3))  # first ever collection happens three days later
    envelope = scene.current()
    assert observation(envelope, "24h", "searches") == 0  # event time, not collected_at
    assert observation(envelope, "7d", "searches") == 1 + MARK
    assert envelope.collected_at.startswith("2026-10-15")


def test_a_window_boundary_event_is_inclusive_at_exactly_the_window_length(tmp_path: Path) -> None:
    vs.enable()
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)  # the journal event is stamped 09:00:11
    edge = scene.at(DAY, 9, 0, 11) + timedelta(days=1)
    scene.collect(edge)
    assert observation(scene.current(), "24h", "searches") == 1  # exactly 24h old: still inside
    scene.collect(edge + timedelta(seconds=1))
    assert observation(scene.current(), "24h", "searches") == 0


def test_partial_retention_is_reported_for_lifetime_when_evidence_has_expired(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)  # credit exists, so the conversation is a ledger root that can freeze
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    clean = scene.window(scene.current(), "lifetime")["coverage"]
    assert "evidence_expired" not in clean["reasons"]
    assert clean["retention"]["evidence_expired_before"] is None
    scene.collect(later(45))  # the 30-day evidence horizon passes over day 12
    after = scene.current()
    life = scene.window(after, "lifetime")["coverage"]
    assert life["partial"] is True and "evidence_expired" in life["reasons"]
    assert life["retention"]["evidence_expired_before"] is not None
    assert life["retention"]["frozen_roots"] >= 1
    assert observation(after, "lifetime", "searches") == 1 + MARK  # supported value, not zeroed
    assert scene.coverage(after)["retention"]["evidence_expired_before"] is not None


# ---- 9.5 live counters -----------------------------------------------------------------------


def test_the_runtime_label_stays_unknown_after_the_execution_is_later_attributed(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    # capture-time: the journal records the execution, the transcript does not exist yet
    scene.journal_events(
        scene.execution("n-0", ts(DAY, 9, 0, 11), text=vs.query(0), transport="mcp")
    )
    scene.collect(scene.at(hour=10))
    early = scene.current()
    mcp = [r for r in counters(early)["executions"] if r["transport"] == "mcp"]
    assert mcp == [
        {
            "operation": "search",
            "transport": "mcp",
            "outcome": "ok",
            "runtime": "unknown",
            "count": 1,
        }
    ]
    assert observation(early, "lifetime", "runtime.unknown") == 1 + MARK
    assert (
        "runtime.claude" not in scene.window(early, "lifetime")["observations"]
    )  # nothing attributed

    # reconciliation later attributes the execution to the claude transcript that called it
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=11))
    attributed = scene.current()
    assert observation(attributed, "lifetime", "runtime.claude") == 1  # the attributed view moved
    assert observation(attributed, "lifetime", "runtime.unknown") == MARK  # only the marker is left
    rows = counters(attributed)["executions"]
    assert [r for r in rows if r["transport"] == "mcp"] == mcp  # the frozen label did not move
    assert sum(r["count"] for r in rows) == 1 + MARK and all(
        r["runtime"] == "unknown" for r in rows
    )


def test_counters_never_decrease_across_cycles_restarts_and_expiry(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    totals: list[int] = []
    hist_counts: list[int] = []
    for n in range(4):
        vs.pair(vs.projects(tmp_path), n, CLOCK)
        scene.collect(scene.at(hour=10 + n))
        envelope = scene.current()
        totals.append(total_executions(envelope))
        hist_counts.append(sum(h["count"] for h in counters(envelope)["duration"]))
    scene.collect(scene.at(hour=20))  # nothing new at all
    totals.append(total_executions(scene.current()))
    scene.collect(later(45))  # clock jumps far ahead: windows and retention horizon move
    totals.append(total_executions(scene.current()))
    assert totals == [n + MARK for n in (1, 2, 3, 4, 4, 4)]
    assert hist_counts == sorted(hist_counts) and hist_counts[-1] == 4 + MARK


def test_an_ordinary_restart_preserves_the_epoch_and_the_counters(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    first = scene.current()
    vs.pair(vs.projects(tmp_path), 1, CLOCK)
    result = scene.collect(
        scene.at(hour=11)
    )  # a fresh process call: a restart as far as state goes
    second = scene.current()
    assert result.epoch == 1 and second.counter_epoch == first.counter_epoch == 1
    assert counters(second)["epoch_started_at"] == counters(first)["epoch_started_at"]
    assert total_executions(second) == 2 + MARK and total_executions(first) == 1 + MARK


def test_new_epoch_zeroes_future_live_counters_and_keeps_every_ledger_total(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(tmp_path), n, CLOCK)
    scene.collect(scene.at(hour=10))
    before = scene.current()
    ledger_before = vs.ledger_digest()
    assert total_executions(before) == 2 + MARK and before.counter_epoch == 1

    declared = scene.collect(scene.at(hour=11), CollectOptions(new_epoch=True))
    after = scene.current()
    assert declared.epoch == 2 and after.counter_epoch == 2
    assert counters(after)["epoch"] == 2
    assert counters(after)["executions"] == []  # live counters start at zero
    assert counters(after)["duration"] == [] and counters(after)["rendered_bytes"] == []
    assert counters(after)["epoch_started_at"] == "2026-10-12T11:00:00Z"
    # history, the ledger and the lifetime window are untouched by the declaration
    assert vs.ledger_digest() == ledger_before
    assert observation(after, "lifetime", "searches") == observation(before, "lifetime", "searches")
    assert observation(after, "lifetime", "searches") == 2 + MARK

    # the events captured before the declaration never count in the new epoch ...
    scene.collect(scene.at(hour=12))
    assert counters(scene.current())["executions"] == []
    # ... only events captured AFTER it do, from zero
    vs.pair(vs.projects(tmp_path), 2, CLOCK)
    scene.journal_events(
        scene.execution("n-late", ts(DAY, 13, 0, 0), text="after the reset", transport="cli")
    )
    scene.collect(scene.at(hour=14))
    final = scene.current()
    assert final.counter_epoch == 2
    assert total_executions(final) == 2  # n-2 (09+2 h) and n-late, counted in epoch 2 once
    assert observation(final, "lifetime", "searches") == 4 + MARK  # ledger history kept: all


def test_a_new_epoch_is_not_inherited_by_the_next_ordinary_restart(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    scene.collect(scene.at(hour=11), CollectOptions(new_epoch=True))
    scene.collect(scene.at(hour=12))
    scene.collect(scene.at(hour=13))
    assert scene.current().counter_epoch == 2  # not 3, not back to 1


def test_backfill_is_a_separate_series_and_never_a_burst_of_live_executions(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    live = scene.current()
    # an old transcript the live reader pre-baselined as history
    old = vs.put(vs.projects(tmp_path), "old", [], scene.at(DAY - 10))
    old.write_text("")  # placeholder replaced below by real old records
    from tests.accounting.reduce_builders import claude_tool_result, claude_tool_use, results_json

    records = [
        claude_prompt("old", "old work", uuid="uo", ts=ts(2, 9)),
        claude_tool_use("old", "o1", MCP, {"query": "old q"}, ts=ts(2, 9, 0, 5), message_id="mo"),
        claude_tool_result("old", "o1", results_json(), ts=ts(2, 9, 0, 8)),
    ]
    vs.put(vs.projects(tmp_path), "old", records, scene.at(DAY - 10))
    backfill = CollectOptions(
        backfill=True, since="2026-10-01T00:00:00Z", until="2026-10-05T00:00:00Z"
    )
    result = scene.collect(scene.at(hour=11), backfill)
    assert result.mode == "backfill" and result.backfill is not None
    after = scene.current()
    series = scene.observations(after)["backfill"]
    assert len(series) == 1 and series[0]["since"] == "2026-10-01T00:00:00Z"
    assert sum(series[0]["counts"].values()) >= 1  # the imported history is counted ...
    assert counters(after)["executions"] == counters(live)["executions"]  # ... but not live
    assert after.counter_epoch == live.counter_epoch  # backfill never starts an epoch
    assert observation(after, "24h", "searches") == observation(live, "24h", "searches")


def test_duration_histograms_use_the_ten_fixed_buckets_with_exact_count_and_sum(
    tmp_path: Path,
) -> None:
    vs.enable()
    vs.start_capture()
    durations = [500, 1_000, 1_001, 5_000, 99_999, 100_000, 30_000_000, 30_000_001, None]
    for index, micros in enumerate(durations):
        scene.journal_events(vs.timed_event(f"h-{index}", ts(DAY, 9, index), micros))
    scene.collect(scene.at(hour=10))
    view = counters(scene.current())
    (series,) = [d for d in view["duration"] if d["transport"] == "mcp"]
    assert [b["le_us"] for b in series["buckets"]] == [
        "1000",
        "5000",
        "10000",
        "50000",
        "100000",
        "500000",
        "1000000",
        "5000000",
        "30000000",
        "+Inf",
    ]
    known = [d for d in durations if d is not None]
    assert series["count"] == len(known) and series["sum_us"] == sum(known)  # exact integers
    cumulative = [b["count"] for b in series["buckets"]]
    assert cumulative == sorted(cumulative) and cumulative[-1] == series["count"]
    assert cumulative == [2, 4, 4, 4, 6, 6, 6, 6, 7, 8]  # inclusive upper bounds (le)
    unknown = [d for d in view["duration_unknown"] if d["transport"] == "mcp"]
    assert [d["count"] for d in unknown] == [1]  # unknown duration kept apart, not 0
    assert counters_module.BUCKET_BOUNDS_US == (
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
    raw = (vs.accounting_dir() / "snapshots" / "current.json").read_text()
    assert "." not in json.dumps(view["duration"])  # no float anywhere in the series
    assert '"sum_us":' in raw


@pytest.mark.parametrize("ordinary_restart", [True, False])
def test_histogram_count_and_sum_survive_restart_and_epoch_boundaries(
    tmp_path: Path, ordinary_restart: bool
) -> None:
    vs.enable()
    vs.start_capture()
    scene.journal_events(vs.timed_event("h-1", ts(DAY, 9, 1), 2_000))
    scene.collect(scene.at(hour=10))
    scene.journal_events(vs.timed_event("h-2", ts(DAY, 9, 2), 3_000))
    options = scene.LIVE if ordinary_restart else CollectOptions(new_epoch=True)
    scene.collect(scene.at(hour=11), options)
    mcp = [d for d in counters(scene.current())["duration"] if d["transport"] == "mcp"]
    series = mcp[0] if mcp else {"count": 0, "sum_us": 0}
    if ordinary_restart:
        assert (series["count"], series["sum_us"]) == (2, 5_000)
    else:
        assert (series["count"], series["sum_us"]) == (0, 0)  # reset for the new epoch
