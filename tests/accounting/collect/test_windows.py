# ---- Survivor source: test_windows.py ----
"""Event-time windows, sealed-day lifetime aggregates, and per-window estimates."""

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.collect.collector import RetentionCoverage
from ssgrep.accounting.collect.replay import Ledger
from ssgrep.accounting.ledger.contracts import (
    Attribution,
    Basis,
    Exposure,
    Outcome,
    Quantity,
    Reuse,
    Scenario,
    TaskId,
    canonical_bytes,
)
from ssgrep.accounting.ledger.reduce_records import ExecutionRecord
from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
from ssgrep.accounting.views import history as windows, service
from ssgrep.accounting.views.history import (
    SealedDay,
    build_windows,
    in_window,
    seal_days,
    window_start,
)
from ssgrep.accounting.views.types import Scope, parse_quantity
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.ledger.valuation_builders import METHOD, atom, candidate, debit
from tests.accounting.views.snapshot_builders import COLLECTED_AT, enable_policy, write_snapshot

NOW = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
RETENTION = RetentionCoverage()
BYTES = Quantity.observed(100, "bytes")
NO_BYTES = Quantity.unavailable("bytes", "not_recorded")


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def rec(
    ts: str | datetime,
    *,
    eid: str | None = None,
    operation: str = "search",
    outcome: Outcome = Outcome.OK,
    runtime: str | None = "claude",
    visible: Quantity = BYTES,
    rendered: Quantity = BYTES,
) -> ExecutionRecord:
    ts_utc = stamp(ts) if isinstance(ts, datetime) else ts
    return ExecutionRecord(
        execution_id=eid or f"x-{ts_utc}-{operation}-{outcome.value}",
        ts_utc=ts_utc,
        day=ts_utc[:10],
        operation=operation,
        transport="cli",
        outcome=outcome,
        outcome_basis="journal",
        attribution=Attribution.UNATTRIBUTED,
        source="journal",
        runtime=runtime,
        session_id=None,
        invocation_id=None,
        actor="agent",
        task=None,
        rendered_bytes=rendered,
        visible_bytes=visible,
        exposure=Exposure.OBSERVED,
        reuse=Reuse.NONE_OBSERVED,
        timing_known=True,
        in_scope=True,
    )


def build(
    executions,
    *,
    sealed=None,
    rows=None,
    now=NOW,
    retention=RETENTION,
    since="2026-10-01",
    horizon=30,
):
    return build_windows(
        executions=executions,
        sealed=sealed or {},
        rows=rows,
        retention=retention,
        now=now,
        collection_since_day=since,
        evidence_horizon_days=horizon,
    )


def counts(result, name: str, *obs: str) -> list[int | None]:
    return [result[name]["observations"][o]["value"] for o in obs]


# ---- 1. event-time membership ------------------------------------------------------------


@pytest.mark.parametrize(("name", "span"), [("24h", 1), ("7d", 7), ("30d", 30)])
def test_membership_boundaries_are_inclusive_at_start_and_now(name: str, span: int) -> None:
    start = NOW - timedelta(days=span)
    one = timedelta(seconds=1)
    assert window_start(name, NOW) == start
    assert in_window(stamp(start), name, NOW)
    assert not in_window(stamp(start - one), name, NOW)
    assert in_window(stamp(NOW), name, NOW)
    assert not in_window(stamp(NOW + one), name, NOW)

    executions = [
        rec(start - one, operation="search"),  # one second too old
        rec(start, operation="show"),  # exactly at the start
        rec(NOW, operation="note"),  # exactly now
        rec(NOW + one, operation="search", outcome=Outcome.ERROR),  # clock skew: future
    ]
    result = build(executions)
    assert counts(result, name, "searches", "shows", "notes", "failed") == [0, 1, 1, 0]
    # nothing is dropped for good: the future and the old event are both lifetime facts
    assert counts(result, "lifetime", "searches", "shows", "notes", "failed") == [2, 1, 1, 1]


def test_lifetime_window_has_no_start_and_accepts_the_future() -> None:
    assert window_start("lifetime", NOW) is None
    assert in_window("2000-01-01T00:00:00Z", "lifetime", NOW)
    assert in_window(stamp(NOW + timedelta(days=3)), "lifetime", NOW)
    assert not in_window("not-a-time", "7d", NOW)
    with pytest.raises(ValueError, match="unknown window"):
        window_start("1h", NOW)


# ---- 2. clock-only expiry ----------------------------------------------------------------


def test_advancing_only_the_clock_expires_24h_observations() -> None:
    executions = [
        rec(NOW - timedelta(hours=2), operation="search"),
        rec(NOW - timedelta(hours=3), operation="show"),
        rec(NOW - timedelta(hours=30), operation="search", outcome=Outcome.ERROR),
        rec(NOW - timedelta(days=3), operation="search"),
        rec(NOW - timedelta(days=20), operation="show"),
    ]
    names = ("searches", "shows", "failed")
    before = build(executions)
    after = build(executions, now=NOW + timedelta(hours=25))
    assert counts(before, "24h", *names) == [1, 1, 0]
    assert counts(after, "24h", *names) == [0, 0, 0]
    assert counts(before, "7d", *names) == [3, 1, 1]
    assert counts(after, "7d", *names) == [3, 1, 1]
    assert counts(before, "30d", *names) == [3, 2, 1]
    assert counts(after, "30d", *names) == [3, 2, 1]
    assert counts(before, "lifetime", *names) == [3, 2, 1] == counts(after, "lifetime", *names)
    assert before["lifetime"]["observations"] == after["lifetime"]["observations"]


# ---- 3. sealed days: lifetime survives evidence expiry -----------------------------------


def three_days() -> list[ExecutionRecord]:
    return [
        rec("2026-11-01T01:00:00Z", operation="search"),
        rec("2026-11-01T02:00:00Z", operation="show", outcome=Outcome.ERROR),
        rec("2026-11-02T03:00:00Z", operation="search"),
        rec("2026-11-09T09:00:00Z", operation="search"),
    ]


def test_sealing_covers_only_days_before_the_cutoff_and_keeps_counts() -> None:
    sealed, late = seal_days(three_days(), before_day="2026-11-09", already={})
    assert late == 0
    assert sorted(sealed) == ["2026-11-01", "2026-11-02"]
    day = sealed["2026-11-01"]
    assert dict(day.counts)["searches"] == 1 and dict(day.counts)["shows"] == 1
    assert dict(day.counts)["failed"] == 1 and dict(day.counts)["outcome.error"] == 1
    assert dict(day.counts)["runtime.claude"] == 2
    assert (day.visible_known, day.visible_missing) == (200, 0)
    assert (day.rendered_known, day.rendered_missing) == (200, 0)


def test_lifetime_counts_survive_expiry_of_sealed_days_evidence() -> None:
    executions = three_days()
    sealed, _ = seal_days(executions, before_day="2026-11-09", already={})
    retained = [e for e in executions if e.day >= "2026-11-09"]  # older evidence expired
    full = build(executions, sealed=sealed)
    expired = build(retained, sealed=sealed)
    names = ("searches", "shows", "failed", "visible_bytes", "rendered_bytes")
    assert counts(full, "lifetime", *names) == [3, 1, 1, 400, 400]
    assert counts(expired, "lifetime", *names) == [3, 1, 1, 400, 400]
    # without sealing, the same expiry WOULD shrink lifetime: this is what sealing prevents
    assert counts(build(retained), "lifetime", "searches") == [1]
    assert full["lifetime"]["coverage"]["retention"]["sealed_days"] == 2


def test_late_execution_for_a_sealed_day_is_reported_not_added() -> None:
    first = three_days()
    sealed, _ = seal_days(first, before_day="2026-11-09", already={})
    late_one = rec("2026-11-01T23:00:00Z", operation="search", eid="late-1")
    again, late = seal_days([*first, late_one], before_day="2026-11-09", already=sealed)
    assert late == 1
    assert again == sealed  # the sealed day did not absorb it
    result = build([*first, late_one], sealed=again)
    assert counts(result, "lifetime", "searches", "shows") == [3, 1]


def test_seal_days_is_idempotent_and_order_independent() -> None:
    executions = three_days()
    once, late = seal_days(executions, before_day="2026-11-09", already={})
    twice, late_again = seal_days(executions, before_day="2026-11-09", already=once)
    assert (twice, late, late_again) == (once, 0, 0)
    rng = random.Random(7)
    for _ in range(10):
        shuffled = executions[:]
        rng.shuffle(shuffled)
        assert seal_days(shuffled, before_day="2026-11-09", already={}) == (once, 0)
    # a later cutoff only adds the newly closed day
    grown, _ = seal_days(executions, before_day="2026-11-10", already=once)
    assert sorted(grown) == ["2026-11-01", "2026-11-02", "2026-11-09"]
    assert {d: grown[d] for d in once} == once


def test_sealed_day_round_trips_through_json_integers() -> None:
    sealed, _ = seal_days(three_days(), before_day="2026-11-09", already={})
    day = sealed["2026-11-01"]
    raw = json.loads(json.dumps(day.to_dict()))
    assert SealedDay.from_dict(raw) == day
    for bad in (
        None,
        {},
        {**raw, "counts": []},
        {**raw, "visible_known": 1.5},
        {**raw, "visible_known": True},
    ):
        with pytest.raises(ValueError, match="malformed|integer"):
            SealedDay.from_dict(bad)


# ---- 4. unknown bytes are never zero -----------------------------------------------------


def test_bytes_unavailable_when_any_included_execution_lacks_them() -> None:
    executions = [
        rec(NOW - timedelta(hours=1), visible=BYTES, rendered=BYTES),
        rec(
            NOW - timedelta(hours=2),
            operation="show",
            visible=NO_BYTES,
            rendered=Quantity.observed(7, "bytes"),
        ),
    ]
    result = build(executions)
    for name in ("24h", "7d", "30d", "lifetime"):
        obs = result[name]["observations"]
        assert obs["visible_bytes"]["value"] is None
        assert obs["visible_bytes"]["basis"] == "unavailable"
        assert obs["visible_bytes"]["reason"] == "visible_bytes_missing_for_some_executions"
        assert obs["visible_bytes_known"]["value"] == 100
        assert obs["rendered_bytes"]["value"] == 107  # every execution has rendered bytes
        assert parse_quantity(obs["visible_bytes"], "bytes", "x").value is None


def test_unknown_bytes_in_a_sealed_day_make_lifetime_unavailable_but_not_rolling() -> None:
    old = [
        rec("2026-11-01T01:00:00Z", visible=NO_BYTES),
        rec("2026-11-01T02:00:00Z", visible=BYTES),
    ]
    recent = [rec(NOW - timedelta(hours=1), visible=BYTES)]
    sealed, _ = seal_days(old, before_day="2026-11-09", already={})
    assert (sealed["2026-11-01"].visible_known, sealed["2026-11-01"].visible_missing) == (100, 1)
    result = build(recent, sealed=sealed)  # the old evidence has expired; only the day remains
    life = result["lifetime"]["observations"]
    assert (
        life["visible_bytes"]["value"] is None
        and "visible_bytes_missing" in life["visible_bytes"]["reason"]
    )
    assert life["visible_bytes_known"]["value"] == 200
    assert life["rendered_bytes"]["value"] == 300
    assert result["24h"]["observations"]["visible_bytes"]["value"] == 100


def test_empty_input_counts_zero_for_real_counters_only() -> None:
    result = build([])
    for name in ("24h", "lifetime"):
        assert counts(result, name, "searches", "shows", "failed") == [0, 0, 0]


# ---- 5. estimates from current revisions -------------------------------------------------


def ledger_rows():
    t2, t3 = TaskId("root-2", "turn-1", "main"), TaskId("root-3", "turn-1", "main")
    ledger = Ledger(METHOD)
    ledger.ingest(
        EvidenceBundle(
            (
                candidate("c1", day="2026-11-01", immediate=(atom("s1", 400, 0, "ev-1"),)),
                candidate("c2", task=t2, day="2026-11-05", immediate=(atom("s3", 400, 0, "ev-3"),)),
                candidate("c3", task=t3, day="2026-11-08", immediate=(atom("s4", 400, 0, "ev-4"),)),
            ),
            (
                debit("x2", 4000, task=t2, day="2026-11-05"),
                debit("x3", None, task=t3, day="2026-11-08"),
            ),
        )
    )
    ledger.ingest(  # late evidence supersedes 2026-11-01 (revision 2)
        EvidenceBundle(
            (
                candidate(
                    "c1b",
                    day="2026-11-01",
                    accepted_day="2026-11-20",
                    immediate=(atom("s2", 800, 0, "ev-2"),),
                ),
            ),
            (),
        )
    )
    rows = ledger.rows()
    by_period = {(p.period, p.revision) for p in rows.periods}
    assert ("2026-11-01", 1) in by_period and ("2026-11-01", 2) in by_period  # two real revisions
    return rows


def net(result, name: str, scenario: str = "central", unit: str = "context_tokens") -> dict:
    return result[name]["estimates"][unit][scenario]["net"]


def test_only_the_current_revision_contributes_and_negative_nets_survive() -> None:
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    result = build([], rows=ledger_rows(), now=now)
    seven = result["7d"]["estimates"]["context_tokens"]["central"]
    # 11-01 current revision 300 (superseded revision 100 ignored) + 11-05 100 = 400
    assert seven["credit"]["value"] == 400
    assert seven["debit"]["value"] == 1000
    assert seven["net"]["value"] == -600 and seven["net"]["basis"] == "estimated"
    assert parse_quantity(seven["net"], "tokens", "x").value == -600
    assert result["24h"]["estimates"] == {}  # no period day on/after 2026-11-06
    # lifetime also holds 11-08 whose debit is unknown: the net is unavailable WITH its reason
    life = result["lifetime"]["estimates"]["context_tokens"]["central"]
    assert life["net"]["value"] is None
    assert life["net"]["reason"] == "incomplete:downstream_exposure_unobserved"
    assert life["credit"]["value"] == 500 and life["debit"]["value"] == 1000
    assert set(result["lifetime"]["estimates"]["context_tokens"]) == {
        "conservative",
        "central",
        "optimistic",
    }


def test_estimate_membership_is_by_whole_utc_day() -> None:
    rows = ledger_rows()

    def credit(at: datetime, name: str = "7d"):
        entry = build([], rows=rows, now=at)[name]["estimates"].get("context_tokens")
        return None if entry is None else entry["central"]["credit"]["value"]

    # 7d at 2026-11-08T00:00:00 starts 11-01T00:00: period 11-01 (300) is in...
    assert credit(datetime(2026, 11, 8, 0, 0, 0, tzinfo=UTC)) == 300 + 100 + 100
    # ...and still in one second later (start 11-01T00:00:01 is within day 11-01)
    assert credit(datetime(2026, 11, 8, 0, 0, 1, tzinfo=UTC)) == 500
    # a day later the start day is 11-02 and the whole of 11-01 is out
    assert credit(datetime(2026, 11, 9, 0, 0, 0, tzinfo=UTC)) == 200
    # 24h at noon of 11-09 starts 11-08T12:00: day 11-08 counts entirely
    assert credit(datetime(2026, 11, 9, 12, 0, tzinfo=UTC), "24h") == 100


def test_rows_none_gives_no_estimates_but_keeps_observations() -> None:
    result = build([rec(NOW)], rows=None)
    assert all(result[name]["estimates"] == {} for name in windows.WINDOW_NAMES)
    assert counts(result, "7d", "searches") == [1]


def test_unit_keys_use_the_bare_name_for_the_first_unit_and_method_for_others() -> None:
    from ssgrep.accounting.ledger.units import CONTEXT_TOKENS, WORK_SECONDS, UnitSpec
    from ssgrep.accounting.ledger.valuation_inputs import Methodology
    from tests.accounting.ledger.valuation_builders import CHARS

    exact = UnitSpec(CONTEXT_TOKENS, "exact", tokenizer="tok==1")
    seconds = UnitSpec(WORK_SECONDS, "capped_gap_v1")
    ledger = Ledger(Methodology("three-units", (CHARS, exact, seconds)))
    ledger.ingest(
        EvidenceBundle(
            (
                candidate(
                    "c1",
                    day="2026-11-09",
                    immediate=(atom("s1", 400, 0, "ev-1", tokens=50, seconds=30),),
                ),
            ),
            (),
        )
    )
    estimates = build([], rows=ledger.rows(), now=NOW)["lifetime"]["estimates"]
    assert list(estimates) == ["context_tokens", "context_tokens:exact", "work_seconds"]
    assert estimates["context_tokens"]["central"]["credit"]["value"] == 100  # 400 chars // 4
    assert estimates["context_tokens:exact"]["central"]["credit"]["value"] == 50
    assert estimates["work_seconds"]["central"]["credit"]["value"] == 30


def test_same_method_different_tokenizers_keep_separate_estimate_keys() -> None:
    from ssgrep.accounting.ledger.units import CONTEXT_TOKENS, UnitSpec
    from ssgrep.accounting.ledger.valuation_inputs import Methodology
    from tests.accounting.ledger.valuation_builders import CHARS

    first = UnitSpec(CONTEXT_TOKENS, "exact", tokenizer="tok==1")
    second = UnitSpec(CONTEXT_TOKENS, "exact", tokenizer="tok==2")
    ledger = Ledger(Methodology("tokenizer-identities", (CHARS, first, second)))
    ledger.ingest(EvidenceBundle((candidate("one", immediate=(atom("one", 400, tokens=50),)),), ()))
    estimates = build([], rows=ledger.rows())["lifetime"]["estimates"]
    assert set(estimates) == {
        "context_tokens",
        "context_tokens:exact",
        "context_tokens:exact:tok==2:",
    }
    assert estimates["context_tokens"]["central"]["credit"]["value"] == 100
    assert estimates["context_tokens:exact"]["central"]["credit"]["value"] == 50
    assert estimates["context_tokens:exact:tok==2:"]["central"]["credit"]["value"] == 50


def test_naive_window_clock_is_rejected_instead_of_using_machine_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build([], now=NOW.replace(tzinfo=None))
    assert build([], now=NOW)["24h"]["observations"]["searches"]["value"] == 0


def test_periods_after_today_are_outside_rolling_estimates_but_in_lifetime() -> None:
    rows = ledger_rows()  # has a 2026-11-08 period (unknown debit)
    early = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
    result = build([], rows=rows, now=early)
    assert result["30d"]["estimates"]["context_tokens"]["central"]["credit"]["value"] == 400
    assert result["lifetime"]["estimates"]["context_tokens"]["central"]["credit"]["value"] == 500


# ---- 6. coverage -------------------------------------------------------------------------


def test_lifetime_coverage_reports_retention_and_partial_when_anything_expired() -> None:
    sealed, _ = seal_days(three_days(), before_day="2026-11-09", already={})
    clean = build(three_days(), sealed=sealed, since="2026-11-01")["lifetime"]["coverage"]
    assert clean["partial"] is False
    assert clean["retention"] == {
        "evidence_expired_before": None,
        "aggregates_expired_before": None,
        "frozen_roots": 0,
        "complete_since": "2026-11-01",
        "sealed_days": 2,
    }
    cases = [
        RetentionCoverage(evidence_expired_before="2026-10-10"),
        RetentionCoverage(aggregates_expired_before="2025-01-01"),
        RetentionCoverage(frozen_roots=2),
    ]
    for retention in cases:
        coverage = build(three_days(), sealed=sealed, retention=retention, since="2026-11-01")[
            "lifetime"
        ]["coverage"]
        assert coverage["partial"] is True
        assert coverage["retention"]["evidence_expired_before"] == retention.evidence_expired_before
        assert coverage["retention"]["frozen_roots"] == retention.frozen_roots
    # collection began after the earliest retained data: not complete since the data starts
    late_start = build(three_days(), sealed=sealed, since="2026-11-05")["lifetime"]["coverage"]
    assert (
        late_start["partial"] is True and late_start["retention"]["complete_since"] == "2026-11-05"
    )


@pytest.mark.parametrize(
    ("horizon", "partial"),
    [
        (0, {"24h": True, "7d": True, "30d": True}),
        (1, {"24h": False, "7d": True, "30d": True}),
        (6, {"24h": False, "7d": True, "30d": True}),
        (7, {"24h": False, "7d": False, "30d": True}),
        (29, {"24h": False, "7d": False, "30d": True}),
        (30, {"24h": False, "7d": False, "30d": False}),
    ],
)
def test_rolling_windows_are_partial_when_retention_is_shorter_than_the_window(
    horizon: int, partial: dict[str, bool]
) -> None:
    result = build([], horizon=horizon)
    for name, expected in partial.items():
        coverage = result[name]["coverage"]
        assert coverage["partial"] is expected, (horizon, name)
        assert coverage.get("reason") == (windows.SHORT_RETENTION if expected else None)


# ---- 7. shape / snapshot / read facade ---------------------------------------------------


def test_output_is_float_free_json_with_the_documented_markers() -> None:
    sealed, _ = seal_days(three_days(), before_day="2026-11-09", already={})
    result = build(three_days(), sealed=sealed, rows=ledger_rows())
    encoded = canonical_bytes(result)  # raises on any float
    assert json.loads(encoded) == result  # JSON-native: dicts/lists/ints/strings only
    assert set(result) == {"24h", "7d", "30d", "lifetime"}
    for window in result.values():
        assert window["basis"] == "event_time_utc"
        assert window["estimate_granularity"] == "utc_day"
        assert set(window) == {
            "observations",
            "estimates",
            "coverage",
            "basis",
            "estimate_granularity",
        }


def test_snapshot_windows_are_readable_by_the_read_facade(tmp_path: Path) -> None:
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    executions = [rec(now - timedelta(hours=h), operation="search") for h in (1, 2, 3)]
    executions += [rec(now - timedelta(days=2), operation="show", outcome=Outcome.ERROR)]
    result = build(executions, rows=ledger_rows(), now=now)
    data = tmp_path / "data"
    enable_policy()
    write_snapshot(data, windows=result)
    view = service.read_summary(scope=Scope(window="7d"), now=datetime.fromisoformat(COLLECTED_AT))
    measured = dict(view.measured)
    assert measured["searches"].value == 3 and measured["shows"].value == 1
    assert measured["failed"].value == 1
    assert measured["visible_bytes"].value == 400
    assert measured["visible_bytes"].basis is Basis.OBSERVED
    row = {r.unit_key: r for r in view.scenarios}["context_tokens"]
    central = dict(row.net)[Scenario.CENTRAL]
    assert central.value == -600 and central.basis is Basis.ESTIMATED
    assert dict(row.credit)[Scenario.CENTRAL].value == 400
    assert dict(row.debit)[Scenario.OPTIMISTIC].value == 1000
    assert view.partial is False  # 30 days of evidence cover the 7d window
    life = service.read_summary(
        scope=Scope(window="lifetime"), now=datetime.fromisoformat(COLLECTED_AT)
    )
    assert dict(life.measured)["searches"].value == 3


# ---- 8. determinism ----------------------------------------------------------------------


def test_shuffling_executions_never_changes_the_result_bytes() -> None:
    executions = [
        rec(NOW - timedelta(minutes=m), operation=op, outcome=oc, runtime=rt, visible=vis)
        for m, op, oc, rt, vis in [
            (5, "search", Outcome.OK, "claude", BYTES),
            (50, "show", Outcome.ERROR, None, NO_BYTES),
            (60 * 30, "search", Outcome.EMPTY, "codex", BYTES),
            (60 * 24 * 5, "note", Outcome.OK, "pi", BYTES),
            (60 * 24 * 25, "show", Outcome.OK, "claude", BYTES),
            (60 * 24 * 60, "search", Outcome.OK, "omp", BYTES),
        ]
    ]
    sealed, _ = seal_days(executions, before_day="2026-11-09", already={})
    rows = ledger_rows()
    reference = canonical_bytes(build(executions, sealed=sealed, rows=rows))
    rng = random.Random(11)
    for _ in range(20):
        shuffled = executions[:]
        rng.shuffle(shuffled)
        again = build(shuffled, sealed=dict(reversed(list(sealed.items()))), rows=rows)
        assert canonical_bytes(again) == reference


# ---- rolling event-time windows through the real collector -----------------------------------
# Clock-only collections expire an old observation from ``24h`` while ``7d``/``30d``/lifetime and
# the durable live counters keep their supported values (and the snapshot id moves only because
# the 24h window content moved).

CLOCK = scene.at(hour=9)
DAY = 12
MARK = 1  # the capture-start marker is itself one real (cli search) execution on day 11, 08:00


def later(days: int, hour: int = 12) -> Any:
    return scene.at(DAY, hour) + timedelta(days=days)


def observation(envelope: Any, window: str, name: str) -> int | None:
    return scene.window(envelope, window)["observations"][name]["value"]


def total_executions(envelope: Any) -> int:
    return sum(row["count"] for row in scene.observations(envelope)["counters"]["executions"])


def test_clock_only_collections_expire_the_old_observation_from_24h_only(tmp_path: Path) -> None:
    vs.enable()
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)  # one search on day 12 at 09:00:11
    first = scene.collect(scene.at(hour=10))
    assert first.state == "published"
    fresh = scene.current()
    assert observation(fresh, "24h", "searches") == 1  # the day-11 marker is already outside 24h
    assert observation(fresh, "lifetime", "searches") == 1 + MARK
    assert total_executions(fresh) == 1 + MARK
    assert scene.observations(fresh)["counters"]["epoch"] == 1

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


def test_late_execution_is_reported_but_does_not_rewrite_a_sealed_lifetime_day() -> None:
    vs.enable()
    scene.journal_events(scene.execution("original", "2026-10-01T09:00:00Z"))
    scene.collect(scene.at(hour=20))
    assert scene.quantity(scene.current(), "lifetime", "searches") == 1
    scene.journal_events(scene.execution("late", "2026-10-01T10:00:00Z"))
    scene.collect(scene.at(hour=21))
    assert scene.coverage(scene.current())["suppressed_late_executions"] == 1
    assert scene.quantity(scene.current(), "lifetime", "searches") == 1
    assert scene.counter_total(scene.current()) == 2
