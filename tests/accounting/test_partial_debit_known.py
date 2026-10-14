"""Known visible debits remain usable when exposure is partial."""

from dataclasses import replace

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.receipts import build_receipt
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.valuation_builders import CHARS, METHOD, candidate, debit


def test_partial_debit_with_exact_visible_text_keeps_known_negative_net():
    visible = debit("show-truncated", 120, partial=True, exposure_id="show-1")
    result = value_evidence(EvidenceBundle((), (visible,)), METHOD)
    estimates = rollup(result.leaves, CHARS)

    assert len(estimates) == 3
    for estimate in estimates:
        assert estimate.credit.value == 0
        assert estimate.debit.value == 30
        assert estimate.net.value == -30
        assert estimate.debit_missing == ()
        assert estimate.known_debits == 1


def test_partial_debit_with_unavailable_visible_length_keeps_net_unknown():
    unknown = replace(
        debit("show-unknown", None, partial=True, exposure_id="show-unknown"),
        amount=Quantity.unavailable(CHARS.unit, "visible_length_unavailable"),
    )
    result = value_evidence(EvidenceBundle((), (unknown,)), METHOD)
    estimates = rollup(result.leaves, CHARS)

    assert all(estimate.net.value is None for estimate in estimates)
    assert all(estimate.debit.value == 0 for estimate in estimates)
    assert all(
        estimate.debit_missing == ((unknown.debit_id, "visible_length_unavailable"),)
        for estimate in estimates
    )


def test_truncated_candidate_absent_from_visible_evidence_earns_no_credit():
    visible = debit("show-truncated", 120, partial=True, exposure_id="show-1")
    hidden = candidate("hidden-fact", immediate=())
    assert not hidden.immediate
    valuation = value_evidence(EvidenceBundle((hidden,), (visible,)), METHOD)

    assert not valuation.leaves[0].scenarios[0].credit_raw
    assert valuation.leaves[0].debit_raw == 120


def test_known_partial_cost_receipt_preserves_partiality_without_missing_cost():
    visible = debit("show-truncated", 120, partial=True, exposure_id="show-1")
    fact = candidate("visible-fact", exposure_ids=("show-1",))
    record = build_receipt(fact, (), (visible,), METHOD)

    assert record.receipt.costs == (("exposure:show-truncated", visible.amount),)
    assert record.missing_costs == ()
    assert record.receipt.exposure_ids == ("show-1",)
