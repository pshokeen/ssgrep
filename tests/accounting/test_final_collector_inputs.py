"""Admission, damaged sources, and advisory inputs retain their distinct accounting meanings."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import (
    collector_cycle,
    collector_inputs,
    collector_modes,
    collector_read,
    collector_steps,
    fingerprint,
    storage,
)
from ssgrep.accounting.collector_inputs import AdmitRules
from ssgrep.accounting.journal_reader import read_journal
from ssgrep.accounting.reduce_types import Reduction
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.conftest import build_evidence
from tests.accounting.reduce_builders import KEY, PROFILE
from tests.accounting.test_collect_counters import invocation
from tests.accounting.test_guard_collector_recovery import NOW, seeded


def test_expired_conversation_and_retention_rejections_do_not_discard_needed_structure() -> None:
    rules = AdmitRules(
        expired_sessions=frozenset({("claude", "expired")}),
        horizon_day="2026-10-10",
        capture_start="2026-10-11T00:00:00Z",
    )
    old = build_evidence(ts_utc="2026-10-09T23:59:59Z")
    assert rules(old) == "beyond_retention"
    structural = replace(old, kind="session_header")
    assert rules(structural) is None
    assert rules(replace(structural, session_id="expired")) == "expired_conversation"
    assert rules(replace(structural, runtime="codex", session_id="expired")) is None
    assert rules(replace(old, ts_utc="2026-10-12T00:00:00Z")) is None


def test_backfill_retains_undated_structure_but_not_undated_prompt_text() -> None:
    rules = AdmitRules(window=("2026-10-11T00:00:00Z", "2026-10-12T00:00:00Z"))
    undated = build_evidence(ts_utc=None)
    assert rules(undated) == "outside_interval"
    assert rules(replace(undated, kind="lineage")) is None
    assert rules(replace(undated, ts_utc="2026-10-11T00:00:00Z")) is None
    assert rules(replace(undated, ts_utc="2026-10-12T00:00:00Z")) == "outside_interval"


def test_complete_malformed_source_line_advances_watermark_and_counts_gap_once(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    path = vs.projects(tmp_path) / "-work" / "cons0.jsonl"
    with path.open("ab") as handle:
        handle.write(b"not-json\n")
    scene.collect(NOW)
    checkpoint = scene.committed()
    record = next(row for row in checkpoint.sources.values() if row.source_id.endswith(str(path)))
    assert record.offset == path.stat().st_size and record.gaps["malformed_json"] == 1
    assert scene.coverage(scene.current())["gaps"]["malformed_json"] == 1
    assert scene.counter_total(scene.current()) == 2
    scene.collect(NOW)
    again = scene.committed().sources[record.key]
    assert again.offset == record.offset and again.gaps == record.gaps


def test_journal_damage_changes_reduction_signature_without_changing_valid_events(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    before = read_journal(PROFILE)
    original = collector_inputs.journal_signature(before)
    path = next((vs.accounting_dir() / "journal").glob("*.jsonl"))
    with path.open("ab") as handle:
        handle.write(b"not-json\n")
    after = read_journal(PROFILE)
    assert after.events == before.events and after.quarantined
    assert collector_inputs.journal_signature(after) != original
    scene.collect(NOW)
    assert scene.committed().journal_sig == collector_inputs.journal_signature(after)
    assert scene.counter_total(scene.current()) == 2


def test_inaccessible_fingerprint_is_unknown_not_a_newly_minted_key(tmp_path: Path) -> None:
    seeded(tmp_path)
    path = vs.accounting_dir() / fingerprint.KEY_FILE
    before = path.read_bytes()
    fingerprint.forget_keys()
    path.chmod(0o644)
    try:
        assert collector_inputs.load_fingerprint_key(PROFILE) is None
        assert path.read_bytes() == before
    finally:
        path.chmod(0o600)
    assert collector_inputs.load_fingerprint_key(PROFILE) == KEY


def test_disappearing_source_during_prebaseline_does_not_hide_other_sources(tmp_path: Path) -> None:
    vs.enable()
    vs.filler(vs.projects(tmp_path), "gone", scene.at(hour=9), 1)
    vs.filler(vs.projects(tmp_path), "present", scene.at(hour=9), 1)
    discovered = collector_read.discover(collector_read.make_readers())
    vanished = vs.projects(tmp_path) / "-work" / "gone.jsonl"
    vanished.unlink()
    known = {}
    assert collector_modes.pre_baseline(discovered, known, NOW) == 1
    assert len(known) == 1
    record = next(iter(known.values()))
    assert record.source_id.endswith("present.jsonl") and record.baseline == "skipped"
    assert record.offset > 0


@pytest.mark.parametrize("invalid", [True, "not-an-instant"])
def test_invalid_capture_timestamp_is_unknown_not_an_epoch_origin(invalid: object) -> None:
    assert collector_modes.parse_stamp(invalid) is None
    assert collector_modes.parse_stamp("2026-10-12T01:00:00+02:00") == datetime(
        2026, 10, 11, 23, tzinfo=UTC
    )


def test_missing_invocation_cwd_does_not_override_the_earliest_proven_directory() -> None:
    missing = invocation(1, "2026-10-12T09:00:00Z")
    known = replace(invocation(2, "2026-10-12T10:00:00Z"), cwd="/work/first")
    later = replace(invocation(3, "2026-10-12T11:00:00Z"), cwd="/work/later")
    reduction = Reduction(PROFILE, invocations=(missing, later, known))
    assert collector_steps.session_cwds(reduction) == (("claude", "s", "/work/first"),)


def test_compaction_preserves_the_same_evidence_and_ledger_as_uncompacted_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    first = set(vs.evidence_identities())
    vs.pair(vs.projects(tmp_path), 1, scene.at(hour=10))
    monkeypatch.setattr(collector_cycle, "MAX_SEGMENTS", 1)
    scene.collect(NOW)
    actual = vs.facts()
    assert len(scene.committed().segments) == 1
    assert first < set(actual.evidence)
    assert len(actual.evidence) == len(set(actual.evidence))
    reference_env = vs.env_for(tmp_path)
    reference_env["SSGREP_DATA_DIR"] = str(tmp_path / "reference" / "data")
    # Witness/action locators are ledger identity: only the accounting store may differ.
    with vs.isolated(reference_env):
        vs.enable(pin=True)
        vs.start_capture()
        vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
        vs.pair(vs.projects(tmp_path), 1, scene.at(hour=10))
        scene.collect(NOW)
        expected = vs.facts()
    assert actual.evidence == expected.evidence
    assert actual.ledger == expected.ledger
    assert actual.counters == expected.counters


def test_progress_directory_open_failure_is_advisory_after_lease_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    original = storage.open_profile_dir
    failures = []

    def unavailable(profile: str, *parts: str, **options: Any) -> int:
        if parts == ("worker",):
            failures.append(parts)
            raise PermissionError("worker progress unavailable")
        return original(profile, *parts, **options)

    with monkeypatch.context() as patch:

        def fault(step: str) -> None:
            if step == "started":
                patch.setattr(storage, "open_profile_dir", unavailable)

        result = scene.collect(NOW, on_step=fault)
    assert failures and scene.report(result).generation == 2
    assert scene.counter_total(scene.current()) == 2
    assert scene.report(scene.collect(NOW)).generation == 3


def test_inaccessible_index_is_reported_unknown_without_losing_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded(tmp_path)
    index = storage.data_root() / "lancedb"
    index.mkdir(mode=0o700)
    original = Path.stat

    def denied(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == index:
            raise PermissionError("index stat denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", denied)
        scene.collect(NOW)
        assert scene.coverage(scene.current())["index"] == "unknown"
        assert scene.counter_total(scene.current()) == 2
    scene.collect(NOW)
    assert scene.coverage(scene.current())["index"] == "available"
