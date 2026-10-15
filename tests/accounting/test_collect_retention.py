"""Retention hooks as consumed by a collection cycle: horizons, whole-root expiry, no paths."""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import os
import typing
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ssgrep.accounting import (
    collector_read,
    collector_state,
    collector_steps,
    policy,
    retention_hooks,
)
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_types import CollectOptions, RetentionCoverage
from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.replay import Ledger
from ssgrep.accounting.retention_hooks import (
    ExpiryPlan,
    aggregate_horizon_day,
    apply_expiry,
    coverage_after,
    evidence_horizon_day,
    expire_aggregates,
    expire_journal_events,
    plan_expiry,
    split_evidence,
)
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.conftest import build_event, build_evidence
from tests.accounting.reduce_builders import PROFILE, claude_prompt, ts
from tests.accounting.valuation_builders import METHOD, atom, candidate, debit

NOW = datetime(2026, 12, 10, 12, 30, tzinfo=UTC)
POLICY = ProfilePolicy()  # 30 days of evidence, 400 days of aggregates
EVIDENCE_HORIZON = "2026-11-10"
AGGREGATE_HORIZON = "2025-11-05"


def _ts(day: str, clock: str = "00:00:00") -> str:
    return f"{day}T{clock}Z"


# ---- horizons and the plan ---------------------------------------------------------


def test_default_policy_horizons_are_30_and_400_utc_days_before_now():
    assert POLICY.evidence_retention_days == 30 and POLICY.receipt_retention_days == 400
    assert evidence_horizon_day(NOW, POLICY) == EVIDENCE_HORIZON
    assert aggregate_horizon_day(NOW, POLICY) == AGGREGATE_HORIZON


def test_horizon_uses_the_utc_day_of_now_whatever_its_offset():
    east = datetime(2026, 12, 11, 1, 0, tzinfo=timezone(timedelta(hours=5)))  # 2026-12-10 20:00Z
    assert evidence_horizon_day(east, POLICY) == EVIDENCE_HORIZON
    west = datetime(2026, 12, 10, 22, 0, tzinfo=timezone(timedelta(hours=-5)))  # 12-11 03:00Z
    assert evidence_horizon_day(west, POLICY) == "2026-11-11"


def test_naive_now_is_rejected_so_no_hidden_local_clock_leaks_in():
    with pytest.raises(ValueError, match="timezone-aware"):
        evidence_horizon_day(datetime(2026, 12, 10, 12, 0), POLICY)


def test_evidence_exactly_at_the_horizon_day_is_kept_and_one_day_earlier_expires():
    last = {
        "at-horizon-start": _ts(EVIDENCE_HORIZON, "00:00:00"),
        "at-horizon-end": _ts(EVIDENCE_HORIZON, "23:59:59.999"),
        "day-before-end": _ts("2026-11-09", "23:59:59.999"),
        "long-ago": _ts("2026-01-01"),
    }
    plan = plan_expiry(now=NOW, policy=POLICY, root_last_evidence=last)
    assert plan.expire_roots == ("day-before-end", "long-ago")
    assert plan.keep_roots == ("at-horizon-end", "at-horizon-start")
    assert plan.evidence_horizon_day == EVIDENCE_HORIZON
    assert plan.aggregate_horizon_day == AGGREGATE_HORIZON


def test_managed_shortened_evidence_retention_is_honored():
    managed = ProfilePolicy(managed=True, evidence_retention_days=7)
    assert evidence_horizon_day(NOW, managed) == "2026-12-03"
    last = {"week-old": _ts("2026-12-02"), "five-days": _ts("2026-12-05")}
    shorter = plan_expiry(now=NOW, policy=managed, root_last_evidence=last)
    assert shorter.expire_roots == ("week-old",) and shorter.keep_roots == ("five-days",)
    default = plan_expiry(now=NOW, policy=POLICY, root_last_evidence=last)
    assert default.expire_roots == () and default.keep_roots == ("five-days", "week-old")
    assert shorter.aggregate_horizon_day == AGGREGATE_HORIZON  # evidence knob only


def test_plan_is_sorted_deterministic_and_independent_of_mapping_order():
    items = [(f"r{i}", _ts(f"2026-10-0{i}" if i < 10 else "2026-11-20")) for i in range(1, 12)]
    forward = plan_expiry(now=NOW, policy=POLICY, root_last_evidence=dict(items))
    backward = plan_expiry(now=NOW, policy=POLICY, root_last_evidence=dict(reversed(items)))
    assert forward == backward
    assert list(forward.expire_roots) == sorted(forward.expire_roots) and forward.expire_roots
    assert list(forward.keep_roots) == sorted(forward.keep_roots) and forward.keep_roots
    assert set(forward.expire_roots).isdisjoint(forward.keep_roots)
    assert plan_expiry(now=NOW, policy=POLICY, root_last_evidence={}) == ExpiryPlan(
        EVIDENCE_HORIZON, AGGREGATE_HORIZON, (), ()
    )


def test_a_root_whose_conversation_has_one_recent_record_is_kept_whole():
    """The caller reports the LATEST record of any session of the root: old records alone
    must not decide; here the root's newest evidence is recent so the root is kept."""
    old, recent = _ts("2026-09-01"), _ts("2026-12-09", "23:00:00")
    latest = max(old, recent)
    plan = plan_expiry(now=NOW, policy=POLICY, root_last_evidence={"mixed": latest, "dead": old})
    assert plan.keep_roots == ("mixed",) and plan.expire_roots == ("dead",)


def test_malformed_timestamps_are_rejected_not_silently_expired_or_kept():
    for bad in ("2026-11-01", "yesterday", "2026-11-01T00:00:00+00:00"):
        with pytest.raises(ValueError):
            plan_expiry(now=NOW, policy=POLICY, root_last_evidence={"r": bad})
    recent = {"root": _ts("2026-12-10", "12:30:00")}
    for field in ("evidence_retention_days", "receipt_retention_days"):  # never a future cutoff
        with pytest.raises(ValueError, match="must not be negative"):
            plan_expiry(now=NOW, policy=replace(POLICY, **{field: -1}), root_last_evidence=recent)
    valid = plan_expiry(now=NOW, policy=POLICY, root_last_evidence=recent)
    assert valid.keep_roots == ("root",) and valid.expire_roots == ()


# ---- end-to-end with a real ledger ---------------------------------------------------


def _corpus() -> EvidenceBundle:
    t1 = TaskId("root-1", "turn-1", "main")
    t2 = TaskId("root-2", "turn-1", "main")
    cands = [
        candidate("a", task=t1, day="2026-11-02", accepted_day="2026-11-02"),
        candidate(
            "b",
            task=t1,
            day="2026-11-03",
            accepted_day="2026-11-03",
            immediate=(atom("b1", 600, 100, "ev-2"),),
        ),
        candidate(
            "c",
            task=t2,
            day="2026-11-30",
            accepted_day="2026-11-30",
            immediate=(atom("c1", 800, 0, "ev-3"),),
        ),
    ]
    debits = [
        debit("x1", 640, task=t1, day="2026-11-02", accepted_day="2026-11-02"),
        debit("x2", 400, task=t2, day="2026-11-30", accepted_day="2026-11-30"),
    ]
    return EvidenceBundle(tuple(cands), tuple(debits))


def _closed_totals(ledger: Ledger) -> dict[tuple[str, int], tuple[object, object]]:
    return {(p.period, p.revision): (p.leaves, p.estimates) for p in ledger.rows().periods}


def test_expiry_freezes_the_old_root_keeps_closed_totals_and_rejects_rebuilds():
    ledger = Ledger(METHOD)
    assert ledger.ingest(_corpus()).accepted == 5
    before_rows = ledger.rows()
    before_totals = _closed_totals(ledger)
    assert {r.receipt.root for r in before_rows.receipts} == {"root-1", "root-2"}
    assert all(r.receipt.evidence_available for r in before_rows.receipts)

    plan = plan_expiry(
        now=NOW,
        policy=POLICY,
        root_last_evidence={"root-1": "2026-11-03T08:00:00Z", "root-2": "2026-11-30T08:00:00Z"},
    )
    assert plan.expire_roots == ("root-1",) and plan.keep_roots == ("root-2",)
    assert apply_expiry(ledger, plan) == ["root-1"]

    after_rows = ledger.rows()
    assert _closed_totals(ledger) == before_totals  # no silent downward recompute
    by_root: dict[str, list] = {}
    for record in after_rows.receipts:
        by_root.setdefault(record.receipt.root, []).append(record)
    assert by_root["root-1"] and by_root["root-2"]
    for record in by_root["root-1"]:
        assert record.receipt.evidence_available is False
        assert ("recomputation", "evidence_expired") in record.missing_costs
    for record in by_root["root-2"]:
        assert record.receipt.evidence_available is True
        assert ("recomputation", "evidence_expired") not in record.missing_costs

    rebuilt = EvidenceBundle(
        (candidate("late", task=TaskId("root-1", "turn-9", "main"), day="2026-11-03"),),
        (debit("x9", 4, task=TaskId("root-1", "turn-9", "main")),),
    )
    result = ledger.ingest(rebuilt)
    assert result.accepted == 0
    assert result.rejected == (("late", "evidence_expired_root"), ("x9", "evidence_expired_root"))
    assert _closed_totals(ledger) == before_totals  # rejection never altered a closed period


def test_expiry_of_the_only_root_leaves_every_period_row_byte_identical():
    only = EvidenceBundle(
        tuple(c for c in _corpus().candidates if c.task.root == "root-1"),
        tuple(d for d in _corpus().debits if d.task.root == "root-1"),
    )
    ledger = Ledger(METHOD)
    ledger.ingest(only)
    before = ledger.rows().periods
    plan = plan_expiry(
        now=NOW, policy=POLICY, root_last_evidence={"root-1": "2026-11-03T00:00:00Z"}
    )
    apply_expiry(ledger, plan)
    after = ledger.rows().periods
    assert [(p.period, p.revision, p.leaves, p.estimates) for p in after] == [
        (p.period, p.revision, p.leaves, p.estimates) for p in before
    ]


def test_apply_expiry_ignores_unknown_roots_and_is_idempotent_across_restart():
    ledger = Ledger(METHOD)
    ledger.ingest(_corpus())
    plan = ExpiryPlan(EVIDENCE_HORIZON, AGGREGATE_HORIZON, ("ghost", "root-1"), ())
    assert apply_expiry(ledger, plan) == ["root-1"]
    assert apply_expiry(ledger, plan) == []  # already frozen: nothing more to freeze
    restored = Ledger.restore(ledger.state())
    assert apply_expiry(restored, plan) == []
    assert restored.rows().canonical_bytes() == ledger.rows().canonical_bytes()


def test_apply_expiry_with_an_empty_plan_changes_nothing():
    ledger = Ledger(METHOD)
    ledger.ingest(_corpus())
    before = ledger.rows().canonical_bytes()
    assert (
        apply_expiry(ledger, ExpiryPlan(EVIDENCE_HORIZON, AGGREGATE_HORIZON, (), ("root-1",))) == []
    )
    assert ledger.rows().canonical_bytes() == before


# ---- evidence and journal inputs -----------------------------------------------------


def test_split_evidence_drops_only_records_of_expired_runtime_session_pairs():
    records = [
        build_evidence(runtime="claude", session_id="old", record_id="1"),
        build_evidence(runtime="claude", session_id="old", record_id="2"),  # new tail record
        build_evidence(runtime="codex", session_id="old", record_id="3"),  # same id, other runtime
        build_evidence(runtime="claude", session_id="live", record_id="4"),
    ]
    kept, dropped = split_evidence(records, {("claude", "old")})
    assert dropped == 2
    assert [r.record_id for r in kept] == ["3", "4"]  # order preserved
    everything, none = split_evidence(records, set())
    assert none == 0 and everything == records
    assert split_evidence([], {("claude", "old")}) == ([], 0)


def test_split_evidence_accepts_a_one_shot_iterator():
    stream = iter([build_evidence(session_id="x", record_id="1"), build_evidence(record_id="2")])
    kept, dropped = split_evidence(stream, {("claude", "x")})
    assert (dropped, [r.record_id for r in kept]) == (1, ["2"])


def test_journal_events_at_the_horizon_day_are_kept_and_earlier_days_are_dropped():
    events = [
        build_event(execution_id="e-old", ts_utc="2026-11-09T23:59:59.999Z"),
        build_event(execution_id="e-start", ts_utc="2026-11-10T00:00:00Z"),
        build_event(execution_id="e-end", ts_utc="2026-11-10T23:59:59Z"),
        build_event(execution_id="e-new", ts_utc="2026-12-10T00:00:00Z"),
        build_event(execution_id="e-ancient", ts_utc="2025-01-01T00:00:00Z"),
    ]
    kept, dropped = expire_journal_events(events, EVIDENCE_HORIZON)
    assert dropped == 2
    assert [e.execution_id for e in kept] == ["e-start", "e-end", "e-new"]
    kept_all, none = expire_journal_events(events, "2025-01-01")
    assert none == 0 and len(kept_all) == len(events)


def test_journal_horizon_must_be_a_day_string():
    with pytest.raises(ValueError):
        expire_journal_events([], "2026-11-10T00:00:00Z")


# ---- aggregate expiry and coverage ---------------------------------------------------


def _row(period: str, revision: int = 1) -> dict[str, object]:
    return {"period": period, "revision": revision, "status": "closed", "net": {"value": 3}}


def test_expire_aggregates_removes_only_older_rows_and_reports_exactly_what_went():
    rows = [
        _row("2025-11-03"),
        _row("2025-11-04"),
        _row("2025-11-04", 2),  # adjustment of a removed period leaves with it
        _row(AGGREGATE_HORIZON),
        _row("2026-12-01"),
    ]
    sealed = {"2025-11-04": {"n": 1}, AGGREGATE_HORIZON: {"n": 2}, "2026-12-01": {"n": 3}}
    kept_rows, kept_days, report = expire_aggregates(rows, sealed, AGGREGATE_HORIZON)
    assert [(r["period"], r["revision"]) for r in kept_rows] == [
        (AGGREGATE_HORIZON, 1),
        ("2026-12-01", 1),
    ]
    assert kept_rows[0] == rows[3] and kept_rows[0] is not rows[3]  # whole rows, copied
    assert kept_days == {AGGREGATE_HORIZON: {"n": 2}, "2026-12-01": {"n": 3}}
    assert report.removed_periods == ("2025-11-03", "2025-11-04")
    assert report.removed_history_rows == 3
    assert report.removed_sealed_days == ("2025-11-04",)
    assert report.before_day == AGGREGATE_HORIZON
    assert report.aggregates_expired_before == AGGREGATE_HORIZON
    assert len(rows) == 5 and len(sealed) == 3  # inputs not mutated


def test_expire_aggregates_with_nothing_old_removes_nothing_and_claims_nothing():
    rows = [_row("2026-01-01"), _row("2026-02-01")]
    kept_rows, kept_days, report = expire_aggregates(rows, {"2026-01-01": 1}, AGGREGATE_HORIZON)
    assert kept_rows == rows and kept_days == {"2026-01-01": 1}
    assert (report.removed_periods, report.removed_history_rows, report.removed_sealed_days) == (
        (),
        0,
        (),
    )
    assert report.aggregates_expired_before is None


def test_expire_aggregates_rejects_rows_it_cannot_date():
    with pytest.raises(ValueError):
        expire_aggregates([{"revision": 1}], {}, AGGREGATE_HORIZON)
    with pytest.raises(ValueError):
        expire_aggregates([_row("not-a-day")], {}, AGGREGATE_HORIZON)


def test_coverage_after_advances_horizons_and_counts_total_frozen_roots():
    plan = ExpiryPlan("2026-11-10", AGGREGATE_HORIZON, ("root-1", "root-2"), ())
    fresh = coverage_after(RetentionCoverage(), plan, ["root-1", "root-2"], AGGREGATE_HORIZON, 2)
    assert fresh == RetentionCoverage("2026-11-10", AGGREGATE_HORIZON, 2)
    later_plan = ExpiryPlan("2026-11-20", "2025-11-15", ("root-3",), ())
    second = coverage_after(fresh, later_plan, ["root-3"], "2025-11-15", 3)
    assert second == RetentionCoverage("2026-11-20", "2025-11-15", 3)


def test_coverage_after_never_moves_a_horizon_backwards():
    previous = RetentionCoverage("2026-11-20", "2025-11-15", 4)
    older_plan = ExpiryPlan(
        "2026-11-10", AGGREGATE_HORIZON, ("root-9",), ()
    )  # e.g. shortened policy undone
    result = coverage_after(previous, older_plan, ["root-9"], AGGREGATE_HORIZON, 5)
    assert result.evidence_expired_before == "2026-11-20"
    assert result.aggregates_expired_before == "2025-11-15"
    assert result.frozen_roots == 5


def test_coverage_after_states_no_new_expiry_when_nothing_was_removed():
    previous = RetentionCoverage(None, None, 0)
    plan = ExpiryPlan(EVIDENCE_HORIZON, AGGREGATE_HORIZON, (), ("root-2",))
    assert coverage_after(previous, plan, [], None, 0) == RetentionCoverage(None, None, 0)
    kept = RetentionCoverage("2026-11-01", "2025-10-01", 2)
    assert coverage_after(kept, plan, [], None, 2) == kept


def test_coverage_after_refuses_a_total_below_the_roots_frozen_now():
    plan = ExpiryPlan(EVIDENCE_HORIZON, AGGREGATE_HORIZON, ("a", "b"), ())
    with pytest.raises(ValueError):
        coverage_after(RetentionCoverage(), plan, ["a", "b"], None, 1)


# ---- source transcripts and indexes are out of reach ----------------------------------


def _snapshot(root: Path) -> dict[str, tuple[bytes | None, int]]:
    out: dict[str, tuple[bytes | None, int]] = {}
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        out[str(path.relative_to(root))] = (
            None if path.is_dir() else path.read_bytes(),
            stat.st_mtime_ns,
        )
    return out


def test_every_hook_leaves_transcripts_and_search_indexes_byte_and_mtime_identical(tmp_path):
    transcripts = tmp_path / "claude" / "projects" / "p"
    transcripts.mkdir(parents=True)
    index = tmp_path / "search-index" / "lance"
    index.mkdir(parents=True)
    sentinels = {
        transcripts / "s1.jsonl": b'{"type":"assistant","message":{"usage":{"input_tokens":1}}}\n',
        transcripts / "old.jsonl": b'{"old":true}\n',
        index / "data.lance": b"\x00\x01index-bytes\xff",
        index / "manifest.json": b'{"generation":"gen-1"}',
    }
    for path, content in sentinels.items():
        path.write_bytes(content)
        os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    before = _snapshot(tmp_path)

    ledger = Ledger(METHOD)
    ledger.ingest(_corpus())
    plan = plan_expiry(
        now=NOW, policy=POLICY, root_last_evidence={"root-1": "2026-11-03T08:00:00Z"}
    )
    frozen = apply_expiry(ledger, plan)
    evidence = [build_evidence(session_id="s1", source_id=f"claude:{transcripts / 's1.jsonl'}")]
    kept, dropped = split_evidence(evidence, {("claude", "s1")})
    events = [build_event(ts_utc="2026-01-01T00:00:00Z")]
    expire_journal_events(events, EVIDENCE_HORIZON)
    expire_aggregates([_row("2025-01-01")], {"2025-01-01": 1}, AGGREGATE_HORIZON)
    coverage_after(RetentionCoverage(), plan, frozen, AGGREGATE_HORIZON, len(frozen))

    assert (kept, dropped) == ([], 1) and frozen == ["root-1"]
    assert _snapshot(tmp_path) == before
    for path, content in sentinels.items():
        assert path.read_bytes() == content


def test_no_hook_accepts_or_derives_a_filesystem_path():
    functions = [
        retention_hooks.evidence_horizon_day,
        retention_hooks.aggregate_horizon_day,
        retention_hooks.plan_expiry,
        retention_hooks.apply_expiry,
        retention_hooks.split_evidence,
        retention_hooks.expire_journal_events,
        retention_hooks.expire_aggregates,
        retention_hooks.coverage_after,
    ]
    for fn in functions:
        hints = typing.get_type_hints(fn)
        for name, hint in hints.items():
            assert "Path" not in repr(hint) and "PathLike" not in repr(hint), (fn.__name__, name)
        for name in inspect.signature(fn).parameters:
            assert not any(w in name for w in ("path", "file", "dir", "root_dir", "index")), (
                fn.__name__,
                name,
            )
    tree = ast.parse(inspect.getsource(retention_hooks))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"os", "pathlib", "shutil", "sqlite3", "tempfile", "glob", "io"}
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"open", "remove", "unlink"}
    assert "evidence_expired_before" in inspect.getsource(
        coverage_after
    )  # sanity: real source read


# ---- retention through the real collector ------------------------------------------------------
# Aggregate expiry removes receipt files even if publication was interrupted after commit, and
# widening retention cannot revive aggregates already removed at a committed cutoff.

SCENE_NOW = scene.at(hour=20)
DAY = 12
CLOCK = scene.at(hour=9)


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(root), 0, scene.at(hour=9))
    assert scene.report(scene.collect(SCENE_NOW)).generation == 1
    assert scene.counter_total(scene.current()) == 2


def test_receipt_expiry_does_not_guess_dates_from_corrupt_or_undated_files() -> None:
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-12") == 0
    scene.collect(scene.at(hour=20))
    directory = vs.accounting_dir() / "receipts"
    directory.mkdir(mode=0o700, exist_ok=True)
    payloads = {
        "old.json": b'{"period":"2026-10-11"}',
        "unknown.json": b"{}",
        "noncanonical.json": b'{"period":"20261011"}',
        "invalid.json": b'{"period":"2026-99-99"}',
        "corrupt.json": b"{",
    }
    for name, data in payloads.items():
        path = directory / name
        path.write_bytes(data)
        path.chmod(0o600)
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-12") == 1
    assert not (directory / "old.json").exists()
    for name, data in payloads.items():
        if name != "old.json":
            assert (directory / name).read_bytes() == data


@pytest.mark.parametrize("interrupt", [False, True])
def test_expired_receipts_are_pruned_on_publish_or_committed_restart(
    tmp_path: Path, interrupt: bool
) -> None:
    seeded(tmp_path)
    receipts = vs.accounting_dir() / "receipts"
    original = {p.name: p.read_bytes() for p in receipts.glob("*.json")}
    assert original  # the no-resurrection assertion below has real credit to remove
    later = SCENE_NOW + timedelta(days=401)

    def fail(step: str) -> None:
        if step == "checkpoint_committed":
            raise KeyboardInterrupt

    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            scene.collect(later, on_step=fail)
        assert {p.name: p.read_bytes() for p in receipts.glob("*.json")} == original
        assert scene.current().generation == 1
        assert scene.committed().retention["aggregates_expired_before"] is not None
    scene.collect(later)
    assert list(receipts.glob("*.json")) == []
    retention = scene.coverage(scene.current())["retention"]
    assert retention["aggregates_expired_before"] is not None
    assert retention["evidence_expired_before"] is not None
    assert retention["frozen_roots"] == 0  # the same cycle froze and then aged out the root
    assert scene.counter_total(scene.current()) == 2  # expiry is not a counter epoch reset
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0
    scene.collect(later)
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0
    scene.collect(later, CollectOptions(recompute=True))
    assert list(receipts.glob("*.json")) == []
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0


def test_receipt_pruning_uses_period_not_current_methodology_membership(tmp_path: Path) -> None:
    seeded(tmp_path)
    directory = vs.accounting_dir() / "receipts"
    original = {p.name: p.read_bytes() for p in directory.glob("*.json")}
    sample = json.loads(next(iter(original.values())))
    old = {**sample, "period": "2026-10-10", "methodology": "archived-methodology"}
    old_path = directory / "old-namespace.json"
    old_path.write_text(json.dumps(old))
    old_path.chmod(0o600)
    hidden = directory / ".unfinished.json"
    hidden.write_bytes(b"not a published receipt")
    hidden.chmod(0o600)
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-12") == 1
    assert not old_path.exists()
    assert {
        p.name: p.read_bytes() for p in directory.glob("*.json") if not p.name.startswith(".")
    } == original
    assert hidden.read_bytes() == b"not a published receipt"
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-13") == len(original)
    assert collector_steps.prune_receipt_files(PROFILE, "2026-10-13") == 0


def ledger_periods() -> set[str]:
    checkpoint = scene.committed()
    state = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.ledger is not None
    rows = VersionedLedger.restore_versioned(state.ledger, state.superseded).rows()
    return {row.period for row in rows.periods} | {row.period for row in rows.receipts}


def assert_floor_preserved(floor: str, expected_searches: int) -> None:
    checkpoint = scene.committed()
    assert checkpoint.retention["aggregates_expired_before"] == floor
    state = collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.reduction is not None
    assert all(day >= floor for day in state.sealed)
    assert all(row.day >= floor for row in state.reduction.executions)
    assert all(day >= floor for day in ledger_periods())
    directory = vs.accounting_dir()
    history = json.loads((directory / "history" / "periods.json").read_bytes())
    assert all(row["period"] >= floor for row in history)
    receipts = [json.loads(path.read_bytes()) for path in (directory / "receipts").glob("*.json")]
    assert all(row["period"] >= floor for row in receipts)
    assert scene.quantity(scene.current(), "lifetime", "searches") == expected_searches


def test_widening_retention_and_recompute_never_resurrect_expired_observations_or_ledger(
    tmp_path: Path,
) -> None:
    vs.enable(pin=True)
    configured = policy.load_profile_policy(PROFILE)
    policy.save_profile(replace(configured, evidence_retention_days=30, receipt_retention_days=1))
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    scene.collect(SCENE_NOW)
    assert scene.quantity(scene.current(), "lifetime", "searches") == 2
    assert "2026-10-12" in ledger_periods()
    assert list((vs.accounting_dir() / "receipts").glob("*.json"))
    retained_evidence = vs.evidence_identities()
    assert retained_evidence
    counter_total = scene.counter_total(scene.current())

    later = SCENE_NOW + timedelta(days=10)
    scene.collect(later)
    floor = (later - timedelta(days=1)).date().isoformat()
    assert_floor_preserved(floor, 0)
    assert ledger_periods() == set()
    assert vs.evidence_identities() == retained_evidence  # still inside the 30-day raw horizon

    policy.save_profile(replace(policy.load_profile_policy(PROFILE), receipt_retention_days=400))
    scene.collect(later, CollectOptions(recompute=True))
    assert_floor_preserved(floor, 0)
    assert scene.counter_total(scene.current()) == counter_total
    scene.collect(later)
    assert_floor_preserved(floor, 0)

    # The cutoff is exclusive, not a ban on future data: an execution at its midnight
    # remains eligible while the old retained inputs are still presented to recompute.
    scene.journal_events(scene.execution("at-retention-floor", f"{floor}T00:00:00Z"))
    scene.collect(later, CollectOptions(recompute=True))
    assert_floor_preserved(floor, 1)
    assert scene.counter_total(scene.current()) == counter_total + 1
    scene.collect(later)
    assert_floor_preserved(floor, 1)


def test_partial_retention_is_reported_for_lifetime_when_evidence_has_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vs.enable(pin=True)  # Native exposure retains a root without assuming current-index rarity.
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(scene.at(hour=10))
    clean = scene.window(scene.current(), "lifetime")["coverage"]
    assert "evidence_expired" not in clean["reasons"]
    assert clean["retention"]["evidence_expired_before"] is None
    before_searches = scene.window(scene.current(), "lifetime")["observations"]["searches"]["value"]
    assert (
        before_searches == 2
    )  # Already accepted marker plus native lookup, distinct from durable live counters.
    before_counter = scene.counter_total(scene.current())
    later = scene.at(DAY, 12) + timedelta(days=45)
    scene.collect(later)  # The 30-day horizon passes the already accepted UTC day.
    after = scene.current()
    life = scene.window(after, "lifetime")["coverage"]
    assert life["partial"] is True and "evidence_expired" in life["reasons"]
    assert life["retention"]["evidence_expired_before"] is not None
    assert life["retention"]["frozen_roots"] >= 1
    searches = scene.window(after, "lifetime")["observations"]["searches"]["value"]
    assert (
        searches == before_searches
    )  # Private-evidence expiry cannot erase accepted compact usage.
    assert scene.coverage(after)["retention"]["evidence_expired_before"] is not None
    assert scene.counter_total(after) == before_counter
    assert all(scene.quantity(after, window, "searches") == 0 for window in ("24h", "7d", "30d"))
    from ssgrep.accounting.evidence_replay import EvidenceReplay

    def no_retained_hydration(*args, **kwargs):
        raise AssertionError("Clock-only expiry restart must reuse compact accepted usage")

    monkeypatch.setattr(EvidenceReplay, "hydrate", no_retained_hydration)
    monkeypatch.setattr(EvidenceReplay, "hydrate_many", no_retained_hydration)
    scene.collect(later + timedelta(minutes=1))
    assert scene.quantity(scene.current(), "lifetime", "searches") == before_searches
    assert scene.counter_total(scene.current()) == before_counter
    # Erasure is a separate boundary: a saved pre-expiry execution must never
    # reconstruct a sealed count after an explicit profile-wide date purge.
    from ssgrep.accounting import purge

    purge.execute(
        purge.Selection(PROFILE, later.date().isoformat() + "T00:00:00Z", None), now=later
    )
    scene.collect(later + timedelta(minutes=2))
    assert scene.quantity(scene.current(), "lifetime", "searches") == 0
    assert scene.counter_total(scene.current()) == before_counter


def test_expiry_sealing_inputs_do_not_revive_out_of_scope_or_aggregate_expired_usage():
    from ssgrep.accounting.collector_expiry import sealing_executions
    from tests.accounting.test_collect_windows import rec

    accepted = rec("2026-10-12T09:00:00Z", eid="accepted")
    erased = rec("2026-09-20T09:00:00Z", eid="aggregate-expired")
    out_of_scope = replace(rec("2026-10-13T09:00:00Z", eid="outside"), in_scope=False)
    provisional = rec("2026-11-10T09:00:00Z", eid="not-sealable")
    previous = (accepted, erased, out_of_scope, provisional)

    def seal(previous, current):
        return sealing_executions(
            previous,
            current,
            evidence_before="2026-11-11",
            before_day="2026-11-09",
            aggregate_before="2026-10-01",
            already={},
        )

    assert seal(previous, ()) == (accepted,)
    corrected = replace(accepted, outcome_basis="result_override")
    assert seal(previous, (corrected,)) == (corrected,)
    assert seal((), ()) == ()  # Post-purge state owns the boundary.


# ---- recompute after expiry: unavailable periods are named, live sources never read ------------


def credit_scene(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)


@contextlib.contextmanager
def reader_spy() -> Iterator[list[str]]:
    """Every transcript path the Claude reader is asked to read while the block runs."""
    reads: list[str] = []
    real = collector_read.ClaudeReader.read

    def spy(self, source, state, *args, **kwargs):  # type: ignore[no-untyped-def]
        reads.append(str(source.path))
        return real(self, source, state, *args, **kwargs)

    collector_read.ClaudeReader.read = spy  # type: ignore[method-assign]
    try:
        yield reads
    finally:
        collector_read.ClaudeReader.read = real  # type: ignore[method-assign]


def test_recompute_with_expired_evidence_reports_unavailable_periods_and_keeps_totals(
    tmp_path: Path,
) -> None:
    credit_scene(tmp_path)
    with reader_spy() as reads:  # positive control: the spy really fires on a normal collection
        scene.collect(scene.at(hour=10))
    assert len(reads) >= 4 and all(path.endswith(".jsonl") for path in reads)
    live_total = vs.facts()
    horizon_now = scene.at(DAY) + timedelta(days=45)
    scene.collect(horizon_now)  # retention freezes the day-12 conversations
    frozen = vs.facts()
    assert scene.committed().retention["frozen_roots"] >= 1
    assert not vs.evidence_identities() or len(vs.evidence_identities()) < len(live_total.evidence)

    # the live source CHANGES after retention: recompute must never look at it
    pool = vs.projects(tmp_path)
    marker = "MUTATED-AFTER-RETENTION"
    vs.put(pool, "intruder", [claude_prompt("intr", marker, uuid="ui", ts=ts(12, 9))], CLOCK)
    for path in pool.rglob("src0.jsonl"):
        path.write_text(path.read_text() + "\n")  # a changed live source
    with reader_spy() as reads:
        options = CollectOptions(
            recompute=True, since="2026-10-01T00:00:00Z", until="2026-10-20T00:00:00Z"
        )
        result = scene.collect(horizon_now, options)
    assert reads == []  # no live source was read at all
    assert result.mode == "recompute" and result.state == "published"
    periods = dict(entry.split("=", 1) for entry in result.unavailable_periods)
    assert periods and set(periods.values()) == {"evidence_expired"}
    assert f"2026-10-{DAY}" in periods  # the expired period is named, with its reason
    after = vs.facts()
    assert after.ledger == frozen.ledger  # closed totals preserved, not recomputed downward
    assert not any(marker in repr(e) for e in vs.evidence_identities())
    assert "intruder" not in {e[1] for e in vs.evidence_identities()}
    assert not (tmp_path / "data" / "lancedb").exists()  # never rebuilds the search index
