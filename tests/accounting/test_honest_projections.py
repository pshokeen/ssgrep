"""V07 core evidence must survive every local read projection without invented zero."""

from dataclasses import replace

import pytest

from ssgrep.accounting.history_projection import history_documents
from ssgrep.accounting.ledger_codec import (
    bundle_from_dict,
    bundle_to_dict,
    period_row_from_dict,
    period_row_to_dict,
    receipt_record_from_dict,
    receipt_record_to_dict,
)
from ssgrep.accounting.plain_codec import to_plain
from ssgrep.accounting.receipts import build_receipt, build_receipts
from ssgrep.accounting.replay import ArchivedRows, Ledger, LedgerRows
from ssgrep.accounting.service import _rows
from ssgrep.accounting.service_filter import scenario_rows
from ssgrep.accounting.service_records import _history_entries, _sum_current
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import AtomStatus, Eligibility, EvidenceBundle
from ssgrep.cli.savings_json import estimates_data
from tests.accounting.valuation_builders import CHARS, METHOD, atom, candidate, debit


def evidence():
    cand = candidate(
        "unknown",
        immediate=(),
        eligibility=Eligibility.UNAVAILABLE,
        eligibility_reasons=("rarity_unknown",),
        exposure_ids=("lookup",),
    )
    cost = debit("cost", None, exposure_id="lookup")
    bundle = EvidenceBundle((cand,), (cost,))
    return cand, cost, bundle, value_evidence(bundle, METHOD)


def test_legacy_candidate_deserialization_requires_named_migration_not_supported_default():
    _, _, bundle, _ = evidence()
    raw = bundle_to_dict(bundle)
    del raw["candidates"][0]["eligibility"]
    with pytest.raises(ValueError, match="legacy_valuation_migration_required"):
        bundle_from_dict(raw)


def test_legacy_receipt_cannot_silently_gain_empty_complete_scenarios():
    cand, cost, _, valued = evidence()
    record = build_receipt(cand, valued.allocations, (cost,), METHOD)
    raw = receipt_record_to_dict(record)
    del raw["receipt"]["scenarios"]
    with pytest.raises(ValueError, match="legacy_valuation_migration_required"):
        receipt_record_from_dict(raw)


def test_current_receipt_roundtrip_preserves_all_typed_uncertainty():
    cand, cost, _, valued = evidence()
    record = build_receipt(cand, valued.allocations, (cost,), METHOD)
    assert receipt_record_from_dict(receipt_record_to_dict(record)) == record


def test_snapshot_and_filtered_group_json_preserve_same_unknown_bounds_and_units():
    _, _, _, valued = evidence()
    expected = tuple(e.as_scenario_estimate() for e in rollup(valued.leaves, CHARS))
    wire = {"context_tokens": {e.scenario.value: to_plain(e) for e in expected}}
    snap = _rows(wire)[0]
    filtered = scenario_rows(valued.leaves, {CHARS: "context_tokens"})[0]
    assert snap.estimates == filtered.estimates == expected
    for row in (snap, filtered):
        out = estimates_data((row,))["context_tokens"]
        assert isinstance(out, dict)
        for value in out.values():
            assert value["net"]["value"] is None
            assert value["known_net"] == 0 and value["complete"] is False
            assert value["bounds"]["net_lower"]["value"] is None
            assert value["unit"]["method"] == CHARS.method
            assert value["credit_missing"] and value["debit_missing"]


def test_legacy_snapshot_scalar_is_subtotal_only_never_complete_net_or_finite_bound():
    _, _, _, valued = evidence()
    row = rollup(valued.leaves, CHARS)[0].as_scenario_estimate()
    wire = {
        "context_tokens": {
            "conservative": {
                "credit": to_plain(replace(row.credit, value=100)),
                "debit": to_plain(replace(row.debit, value=160)),
                "net": {
                    "value": -60,
                    "unit": CHARS.unit,
                    "basis": "estimated",
                    "method": "old",
                    "reason": "",
                },
            }
        }
    }
    parsed = _rows(wire)[0]
    assert dict(parsed.net)[row.scenario].value is None
    legacy = parsed.estimates[0]
    assert legacy.known_net == -60 and not legacy.complete
    assert legacy.net.reason == "legacy_valuation_uncertainty_unavailable"
    assert legacy.bounds is None and legacy.unit is None


def test_current_period_codec_distinguishes_unit_estimate_from_scenario_estimate():
    _, _, bundle, _ = evidence()
    ledger = Ledger(METHOD)
    ledger.ingest(bundle)
    row = ledger.rows().periods[0]
    assert period_row_from_dict(period_row_to_dict(row)) == row


def test_named_archived_same_day_rows_are_visible_but_never_current_or_summed():
    first = Ledger(METHOD)
    first.ingest(EvidenceBundle(debits=(debit("original", 40),)))
    old = first.rows()
    second = Ledger(METHOD)
    second.ingest(EvidenceBundle(debits=(debit("replacement", 80),)))
    current = second.rows()
    rows = LedgerRows(
        current.namespace,
        current.periods,
        current.receipts,
        (ArchivedRows("explicit-recompute", old.namespace, old.periods, old.receipts),),
    )
    raw = history_documents(rows, CHARS, "2026-11-20")
    entries = _history_entries(raw)
    assert len(entries) == 2
    archived = next(e for e in entries if e.revision_name)
    assert not archived.current and archived.net.value == -10
    assert archived.namespace == old.namespace
    assert _sum_current(entries).value == -20


@pytest.mark.parametrize(
    "counter", ["replay_unavailable_sessions", "frequency_revision_unavailable"]
)
def test_published_compact_totals_with_unavailable_replay_stay_partial(counter):
    from ssgrep.accounting.collector_publish import snapshot_coverage
    from ssgrep.accounting.collector_types import RetentionCoverage
    from ssgrep.accounting.reduce_types import Reduction

    coverage = snapshot_coverage(
        reduction=Reduction("personal-local", coverage={counter: 2}),
        sources={},
        journal={},
        read_gaps={},
        dropped={},
        stopped_by="complete",
        retention=RetentionCoverage(),
        index="available",
        lag_bytes=0,
    )
    assert coverage["partial"] is True
    assert coverage[counter] == 2
    assert coverage["replay"]["available"] is False


def test_repeated_fact_receipt_keeps_first_credit_all_exposures_and_unique_costs():
    first = candidate("same-fact", exposure_ids=("e1",), execution_ids=("n1",))
    repeated = candidate(
        "same-fact",
        immediate=(),
        novel=False,
        eligibility=Eligibility.KNOWN_REJECTED,
        eligibility_reasons=("already_visible",),
        exposure_ids=("e2",),
        execution_ids=("n2",),
    )
    costs = (debit("d1", 160, exposure_id="e1"), debit("d2", 160, exposure_id="e2"))
    valued = value_evidence(EvidenceBundle((first, repeated), costs), METHOD)
    records = build_receipts((first, repeated), valued.allocations, costs, METHOD)
    (record,) = records.values()
    assert record.receipt.exposure_ids == ("e1", "e2")
    assert record.receipt.execution_ids == ("n1", "n2")
    assert dict(record.credit_raw)["conservative"] == 400
    assert all(
        s.credit.value == 100 and s.debit.value == 80 and s.net.value == 20
        for s in record.receipt.scenarios
    )
    assert len(record.receipt.costs) == 2 and record.receipt.eligibility == "supported"
    backwards = build_receipts(
        (repeated, first), reversed(valued.allocations), (*reversed(costs), costs[0]), METHOD
    )
    assert next(iter(backwards.values())).canonical_bytes() == record.canonical_bytes()
    from ssgrep.accounting.collector_publish import reuse_blocks

    block = next(iter(reuse_blocks((repeated, first), METHOD).values()))
    assert block["novel"] is True
    assert reuse_blocks((repeated, first), METHOD) == reuse_blocks((first, repeated), METHOD)


def test_one_invalid_reference_does_not_turn_another_missing_source_into_known_rejection():
    cand = candidate(
        "mixed",
        immediate=(
            atom("invalid", 40, origin="invalid-source", status=AtomStatus.DIGEST_CONFLICT),
            atom("missing", 40, origin="missing-source", status=AtomStatus.UNAVAILABLE),
        ),
    )
    valued = value_evidence(EvidenceBundle((cand,), (debit("cost", 40),)), METHOD)
    rows = rollup(valued.leaves, CHARS)
    assert all(r.net.value is None and dict(r.eligibility)["mixed"] == "unavailable" for r in rows)
    assert all(r.bounds.net_lower.value == -10 and r.bounds.net_upper.value is None for r in rows)


def test_read_only_reduction_expands_committed_root_manifest_without_loading_ledger(tmp_path):
    from ssgrep.accounting import collector_state, collector_store, service_ledger
    from ssgrep.accounting.collector_reduction_store import FORMAT
    from tests.accounting.test_ux_support import collected_scene, fingerprint

    collected_scene(tmp_path, pairs=(0,))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    manifest = collector_store.read_blob(
        "personal-local", checkpoint.blobs[collector_state.REDUCTION]
    )
    assert manifest is not None
    assert manifest["format"] == FORMAT and manifest["roots"]
    before = fingerprint(tmp_path / "data")
    data, reason = service_ledger.load_ledger("personal-local")
    doc, error = service_ledger.load_reduction("personal-local")
    assert data is not None and doc is not None
    assert reason is None and error is None and doc == data.doc
    assert {record.execution_id for record in doc.executions} == {"n-start", "n-0"}
    assert fingerprint(tmp_path / "data") == before


def test_absent_unit_is_unestimated_not_a_missing_band_or_complete_zero():
    row = next(row for row in _rows({}) if row.unit_key == "work_seconds")
    assert all(q.value is None and q.reason == "not_estimated" for _, q in row.net)
    assert all(not estimate.complete and estimate.known_net is None for estimate in row.estimates)
