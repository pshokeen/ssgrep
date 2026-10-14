"""9.2: cooperative cycle budgets; a backlog resumes across cycles and converges to single-shot.

Budgets are real (``CycleBudget(bytes=...)`` and an injected monotonic clock for time); the
default 32 MiB / 10 s values are asserted on the type itself and exercised with a REAL >32 MiB
transcript. Convergence is judged on retained evidence identities, the canonical ledger digest,
and the live counters of an unbudgeted single-shot run of the same scene.
"""

from __future__ import annotations

import json
from pathlib import Path

from ssgrep.accounting.collector_types import (
    DEFAULT_CYCLE_BYTES,
    DEFAULT_CYCLE_SECONDS,
    MIB,
    CollectOptions,
    CycleBudget,
)
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import claude_prompt, ts

NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)
SMALL = 1_100  # about one transcript per cycle: every cycle makes progress, none finishes


def backlog_scene(root: Path, sessions: int = 5) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in range(sessions):
        vs.pair(vs.projects(root), n, CLOCK)


def cycles_until_done(budget: CycleBudget, limit: int = 300) -> list:
    results = []
    for _ in range(limit):
        result = scene.collect(NOW, budget=budget)
        results.append(result)
        if scene.report(result).stopped_by == "complete":
            return results
    raise AssertionError("never completed")


def test_documented_defaults_are_ten_seconds_and_32_mib() -> None:
    budget = CollectOptions().budget
    assert (
        (budget.seconds, budget.bytes)
        == (10, 32 * MIB)
        == (
            DEFAULT_CYCLE_SECONDS,
            DEFAULT_CYCLE_BYTES,
        )
    )


def test_a_small_budget_backlog_resumes_to_exactly_the_single_shot_ledger(tmp_path: Path) -> None:
    backlog_scene(tmp_path)
    single = scene.collect(NOW)
    assert scene.report(single).complete and scene.report(single).stopped_by == "complete"
    reference = vs.facts()
    vs.wipe(tmp_path)

    backlog_scene(tmp_path)
    results = cycles_until_done(CycleBudget(bytes=SMALL))
    reports = [scene.report(r) for r in results]
    assert len(reports) >= 4  # the backlog genuinely needed several cycles
    assert [r.stopped_by for r in reports[:-1]] == ["byte_budget"] * (len(reports) - 1)
    assert all(r.status == "published" and not r.complete for r in reports[:-1])
    assert [r.generation for r in reports] == list(range(1, len(reports) + 1))
    assert reports[-1].complete and reports[-1].lag_bytes == 0

    final = vs.facts()
    assert final.evidence == reference.evidence  # no duplicate and no missing record
    assert len(set(final.evidence)) == len(final.evidence)
    assert final.ledger == reference.ledger  # final ledger == single-shot
    assert final.counters == reference.counters
    assert final.snapshot_id == reference.snapshot_id  # even the snapshot identity agrees
    # the budget bound each cycle (one bounded record may overshoot, never unbounded further work)
    largest = max(
        len(json.dumps(r).encode()) + 1
        for path in vs.projects(tmp_path).rglob("*.jsonl")
        for r in map(json.loads, path.read_text().splitlines())
    )
    assert all(r.bytes_read <= SMALL + largest for r in reports)
    assert sum(r.bytes_read for r in reports) >= scene.report(single).bytes_read  # all consumed


def test_every_partial_generation_states_its_prefix_lag_and_never_claims_completeness(
    tmp_path: Path,
) -> None:
    backlog_scene(tmp_path)
    total = sum(p.stat().st_size for p in vs.projects(tmp_path).rglob("*.jsonl"))
    lags: list[int] = []
    for _ in range(300):
        result = scene.collect(NOW, budget=CycleBudget(bytes=SMALL))
        envelope = scene.current()
        coverage = scene.coverage(envelope)
        watermarks = coverage["watermarks"]
        report = scene.report(result)
        assert coverage["lag_bytes"] == report.lag_bytes == sum(w["lag_bytes"] for w in watermarks)
        assert all(0 <= w["offset"] <= w["size"] for w in watermarks)
        assert coverage["stopped_by"] == report.stopped_by
        if report.stopped_by == "complete":
            assert coverage["complete"] is True and coverage["lag_bytes"] == 0
            break
        assert coverage["complete"] is False  # a lagging generation is never complete
        # A budget can stop before rechecking an already-consumed source. Zero known
        # byte lag does not prove that this discovery/read pass completed.
        assert any("collection_incomplete" in w for w in envelope.warnings)
        assert all(
            w["partial"] for w in [scene.window(envelope, name)["coverage"] for name in ("24h",)]
        )
        assert result.state == "partial"
        lags.append(report.lag_bytes)
    assert lags == sorted(lags, reverse=True) and lags[0] <= total  # lag only ever shrinks
    assert lags[-1] < lags[0]
    assert scene.coverage(scene.current())["complete"] is True


def test_an_incomplete_tail_never_completes_until_its_newline_arrives(tmp_path: Path) -> None:
    backlog_scene(tmp_path, sessions=2)
    target = vs.projects(tmp_path) / "-work" / "cons1.jsonl"
    whole = target.read_bytes()
    last = whole.rstrip(b"\n").rfind(b"\n") + 1
    torn = whole[: last + 40]  # half a record, no newline
    target.write_bytes(torn)
    before = scene.collect(NOW)
    assert scene.report(before).complete is False  # a torn tail is not a complete source
    envelope = scene.current()
    watermark = next(
        w for w in scene.coverage(envelope)["watermarks"] if w["pending_tail_bytes"] > 0
    )
    assert watermark["offset"] + watermark["pending_tail_bytes"] == watermark["size"]
    assert watermark["offset"] == last  # the watermark stops at the last newline
    assert (
        scene.coverage(envelope)["complete"] is False
        and scene.coverage(envelope)["sources"]["complete"]
        < (scene.coverage(envelope)["sources"]["total"])
    )
    for _ in range(3):  # repeating changes nothing: still pending, still not complete
        again = scene.collect(NOW)
        assert not scene.report(again).complete and scene.report(again).new_evidence == 0
    held = {i for i in vs.evidence_identities() if i[1] == "cons1"}
    names = {record_id for _, _, record_id, _ in held}
    assert "call:t1b" in names and "result:t1b" not in names  # only the torn last line is absent

    target.write_bytes(whole)  # the writer finishes the line
    after = scene.collect(NOW)
    assert scene.report(after).complete and scene.report(after).new_evidence > 0
    done = vs.evidence_identities()
    assert len(set(done)) == len(done)
    # the completed tail now yields exactly what an unbudgeted run of the finished file yields
    reference_root = tmp_path / "ref"
    with vs.isolated(vs.env_for(reference_root)):
        backlog_scene(reference_root, sessions=2)
        scene.collect(NOW)
        assert vs.evidence_identities() == done


def test_a_record_larger_than_the_remaining_budget_completes_then_stops(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    big = "x" * 40_000
    records = [
        claude_prompt("big", big, uuid="u-big", ts=ts(12, 9)),
        claude_prompt("big", "second", uuid="u-two", ts=ts(12, 9, 1)),
        claude_prompt("big", "third", uuid="u-three", ts=ts(12, 9, 2)),
    ]
    vs.put(vs.projects(tmp_path), "big", records, CLOCK)
    first = scene.collect(NOW, budget=CycleBudget(bytes=1_000))
    report = scene.report(first)
    assert report.stopped_by == "byte_budget" and report.new_evidence >= 1
    assert report.bytes_read >= 40_000  # the started record finished (cooperative, not hard)
    assert report.lag_bytes > 0  # but nothing further was started: later records still unread
    rest = cycles_until_done(CycleBudget(bytes=1_000))
    assert scene.report(rest[-1]).complete
    ids = [i[2] for i in vs.evidence_identities()]
    assert len(ids) == len(set(ids)) == 3


def test_time_budget_stops_between_sources_and_the_next_cycle_finishes_the_rest(
    tmp_path: Path,
) -> None:
    backlog_scene(tmp_path)
    ticks = iter(range(0, 10_000, 11))  # every clock read advances 11 s: the 10 s target passes
    first = scene.collect(NOW, clock=lambda: float(next(ticks)))
    report = scene.report(first)
    assert report.stopped_by == "time_budget" and not report.complete
    assert report.lag_bytes > 0 and first.state == "partial"
    rest = cycles_until_done(CycleBudget())
    assert scene.report(rest[-1]).complete
    ids = vs.evidence_identities()
    assert len(ids) == len(set(ids))
    reference = tmp_path / "ref"
    with vs.isolated(vs.env_for(reference)):
        backlog_scene(reference)
        scene.collect(NOW)
        expected = vs.facts()
    assert expected.evidence  # the unbudgeted reference really read everything
    assert tuple(ids) == expected.evidence


def test_a_real_backlog_over_32_mib_resumes_under_the_default_budget(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    pad = "p" * 900
    records = [
        claude_prompt("huge", f"{pad}{i}", uuid=f"u{i}", ts=ts(12, 9, i // 3600 % 60, i % 60))
        for i in range(40_000)
    ]
    target = vs.put(vs.projects(tmp_path), "huge", records, CLOCK)
    assert target.stat().st_size > 32 * MIB
    first = scene.collect(NOW)  # default 32 MiB budget
    one = scene.report(first)
    assert one.stopped_by == "byte_budget" and not one.complete and one.lag_bytes > 0
    assert 30 * MIB <= one.bytes_read <= 33 * MIB  # bounded by the default, not by the file
    seen = len(vs.evidence_identities())
    second = scene.collect(NOW)
    two = scene.report(second)
    assert two.complete and two.lag_bytes == 0
    ids = [i[2] for i in vs.evidence_identities()]
    assert len(ids) == len(set(ids)) == 40_000 and seen < 40_000  # all of it, once
    assert one.bytes_read + two.bytes_read >= target.stat().st_size
