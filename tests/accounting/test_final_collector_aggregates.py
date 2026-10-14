"""Remaining aggregate boundaries: corrupt series, unit identity, revisions, and late data."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ssgrep.accounting import collector_publish as publish, collector_steps, counters
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_store import SourceRecord
from ssgrep.accounting.contracts import Reuse, TaskId, digest_of
from ssgrep.accounting.replay import Ledger
from ssgrep.accounting.units import CONTEXT_TOKENS, UnitSpec
from ssgrep.accounting.valuation_inputs import EvidenceBundle, Methodology
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_collect_counters import completed, fresh, invocation, mixed_events
from tests.accounting.test_collect_publish import ledger as published_ledger, synthetic_rows
from tests.accounting.test_collect_windows import NOW, build
from tests.accounting.valuation_builders import CHARS, METHOD, atom, candidate, debit


@pytest.mark.parametrize("field", ["operation", "counted"])
def test_corrupt_counter_labels_or_dedupe_container_cannot_be_restored(field: str) -> None:
    valid = counters.observe(fresh(), mixed_events(), horizon_day=None)
    raw = valid.to_dict()
    assert counters.LiveCounters.from_dict(raw) == valid
    if field == "operation":
        raw["executions"][0]["operation"] = ""
    else:
        raw["counted"] = {}
    with pytest.raises(ValueError):
        counters.LiveCounters.from_dict(raw)


def test_counter_epoch_zero_is_rejected_before_any_event_can_be_counted() -> None:
    with pytest.raises(ValueError):
        counters.empty_counters(0, "2026-10-12T00:00:00Z")
    positive = counters.observe(
        counters.empty_counters(1, "2026-10-12T00:00:00Z"), [completed(1)], horizon_day=None
    )
    assert sum(count for _, count in positive.executions) == 1


@pytest.mark.parametrize("field", ["kind", "counts"])
def test_backfill_decoder_does_not_accept_a_live_series_or_unkeyed_counts(field: str) -> None:
    value = counters.backfill_series(
        [invocation(1, "2026-10-12T12:00:00Z")],
        "2026-10-12T00:00:00Z",
        "2026-10-13T00:00:00Z",
        "2026-10-14T00:00:00Z",
    )
    raw = value.to_dict()
    assert counters.BackfillSeries.from_dict(raw) == value
    raw[field] = "live" if field == "kind" else []
    with pytest.raises(ValueError):
        counters.BackfillSeries.from_dict(raw)


def test_naive_window_clock_is_rejected_instead_of_using_machine_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build([], now=NOW.replace(tzinfo=None))
    assert build([], now=NOW)["24h"]["observations"]["searches"]["value"] == 0


def test_same_method_different_tokenizers_keep_separate_estimate_keys() -> None:
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


def test_restored_conflicting_variants_retire_the_stale_copy_when_one_matches() -> None:
    old = VersionedLedger(METHOD)
    old.ingest(EvidenceBundle((candidate("one"),), (debit("lookup", None),)))
    old.ingest(EvidenceBundle((), (debit("lookup", 640),)))
    assert b"conflicting_duplicate_debit" in old.rows().canonical_bytes()
    restored = VersionedLedger.restore_versioned(old.state(), {})
    result = restored.reconcile(
        {"root-1": EvidenceBundle((candidate("one"),), (debit("lookup", 640),))},
        today="2026-11-02",
    )
    assert result.replaced == 1
    assert len(restored.state().debits) == 1
    assert restored.state().debits[0].chars == 640
    assert b"conflicting_duplicate_debit" not in restored.rows().canonical_bytes()


def test_projection_rejects_candidates_and_debits_owned_by_a_different_root() -> None:
    ledger = VersionedLedger(METHOD)
    foreign = TaskId("other-root", "turn", "main")
    result = ledger.reconcile(
        {
            "root-1": EvidenceBundle(
                (candidate("good"), candidate("foreign", task=foreign)),
                (debit("foreign-debit", 400, task=foreign),),
            )
        },
        today="2026-11-02",
    )
    assert set(result.rejected) == {
        ("foreign", "root_mismatch"),
        ("foreign-debit", "root_mismatch"),
    }
    assert [item.candidate_id for item in ledger.state().candidates] == ["good"]
    assert ledger.state().debits == ()


def test_superseded_reuse_does_not_leak_into_the_current_receipt_block() -> None:
    old = candidate("one")
    newer = replace(old, reuse=Reuse.VALIDATED, report_reference=True, accepted_day="2026-11-02")
    assert publish.reuse_blocks([old], METHOD, {digest_of(old)}) == {}
    blocks = publish.reuse_blocks([old, newer], METHOD, {digest_of(old)})
    assert len(blocks) == 1
    block = next(iter(blocks.values()))
    assert block["class"] == Reuse.VALIDATED.value and block["report"] is True


@pytest.mark.parametrize("field", ["class", "action_ok"])
def test_rewriting_receipts_drops_untrusted_stored_reuse_but_keeps_valid_blocks(field: str) -> None:
    ledger = published_ledger()
    rows = ledger.rows()
    blocks = publish.reuse_blocks(ledger.state().candidates, METHOD)
    publish.write_receipts(PROFILE, rows, blocks)
    directory = vs.accounting_dir() / "receipts"
    receipt_id = next(iter(blocks))
    path = directory / f"{receipt_id}.json"
    valid = json.loads(path.read_bytes())
    assert valid["reuse"] == blocks[receipt_id]
    publish.write_receipts(PROFILE, rows)
    assert json.loads(path.read_bytes())["reuse"] == blocks[receipt_id]
    damaged = json.loads(path.read_bytes())
    damaged["reuse"][field] = "bogus" if field == "class" else 1
    path.write_text(json.dumps(damaged))
    publish.write_receipts(PROFILE, rows)
    assert "reuse" not in json.loads(path.read_bytes())


def test_representative_receipt_never_selects_an_unwritable_hostile_id() -> None:
    rows = synthetic_rows(
        ("../escape", 1, {"central": 10000}, True), ("safe", 1, {"central": 10}, True)
    )
    assert publish.representative_receipt(rows) == "safe"
    assert publish.representative_receipt(replace(rows, receipts=rows.receipts[:1])) is None


def test_naive_source_timestamp_cannot_advance_the_published_evidence_watermark() -> None:
    record = SourceRecord("claude", "claude:/private/source", last_ts="2027-01-01T12:00:00")
    assert (
        publish.coverage_through(
            {record.key: record}, "2026-10-12T11:00:00Z", "2026-10-12T12:00:00Z"
        )
        == "2026-10-12T11:00:00Z"
    )


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
