"""Canonical receipt records: stable ids, allocations, exclusions, costs, no clocks."""

from __future__ import annotations

import json
from dataclasses import replace

from ssgrep.accounting.contracts import Reuse, TaskId
from ssgrep.accounting.receipts import (
    EXPIRED_REASON,
    build_receipt,
    build_receipts,
    default_view,
    receipt_id_for,
)
from ssgrep.accounting.units import UnitSpec
from ssgrep.accounting.valuation import value_evidence
from ssgrep.accounting.valuation_inputs import AtomStatus, EvidenceBundle, Methodology
from tests.accounting.valuation_builders import CHARS, METHOD, TASK, atom, candidate, debit


def make(candidates, debits=(), methodology=METHOD):
    valuation = value_evidence(EvidenceBundle(tuple(candidates), tuple(debits)), methodology)
    return build_receipts(candidates, valuation.allocations, debits, methodology)


def test_two_facts_sharing_a_step_allocate_it_once_and_show_the_same_atom_eligible_twice():
    a, b = candidate("fact-a"), candidate("fact-b")
    receipts = make([a, b])
    ra = receipts[receipt_id_for(TASK, "fact-a", METHOD)]
    rb = receipts[receipt_id_for(TASK, "fact-b", METHOD)]
    assert dict(ra.receipt.allocations)["conservative"] == ("step-1",)
    assert dict(rb.receipt.allocations)["conservative"] == ()  # already claimed by fact-a
    assert ra.receipt.atoms == rb.receipt.atoms == ("step-1",)
    assert ra.atom_digests == (("step-1", "d-ev-1"),)
    assert dict(ra.credit_raw)["conservative"] == 400  # chars, undivided
    assert dict(rb.credit_raw)["conservative"] == 0  # credited once, not twice


def test_receipt_carries_lineage_identities_locators_and_methodology():
    cand = candidate(
        "c1",
        execution_ids=("n1-0002", "n1-0001"),
        exposure_ids=("exp-2", "exp-1"),
        witness_locators=("claude:s1:7",),
        action_locators=("claude:s1:19",),
    )
    meth = Methodology(
        "m", (CHARS,), frequency_view="freq-2026-10", tokenizer="", pricing="tariff-1"
    )
    record = make([cand], methodology=meth)[receipt_id_for(TASK, "c1", meth)]
    r = record.receipt
    assert (r.root, r.task_turn, r.revision) == ("root-1", "turn-1", 1)
    assert r.execution_ids == ("n1-0001", "n1-0002") and r.exposure_ids == ("exp-1", "exp-2")
    assert r.candidate_fingerprint == "c1" and r.candidate_type == "path"
    assert r.methodology == meth.hash and r.evidence_available
    assert record.witness_locators == ("claude:s1:7",) and record.action_locators == (
        "claude:s1:19",
    )
    assert (record.frequency_identity, record.pricing_identity) == ("freq-2026-10", "tariff-1")


def test_exclusions_state_why_nothing_was_credited():
    receipts = make(
        [
            candidate("user", user_supplied=True),
            candidate("bad", reuse=Reuse.CONTRADICTED),
            candidate("drift", immediate=(atom("x", 40, status=AtomStatus.DIGEST_CONFLICT),)),
            candidate(
                "gone", immediate=(atom("y", 40, origin="e2", status=AtomStatus.UNAVAILABLE),)
            ),
        ]
    )
    by_fp = {r.receipt.candidate_fingerprint: r.receipt for r in receipts.values()}
    assert by_fp["user"].exclusions == ("candidate:user_supplied",)
    assert by_fp["bad"].exclusions == ("candidate:contradicted",)
    assert by_fp["drift"].exclusions == ("atom:x:digest_conflict",)
    assert by_fp["gone"].exclusions == ("missing:y:unavailable",)
    assert all(dict(r.allocations)["optimistic"] == () for r in by_fp.values())


def test_costs_list_known_and_missing_reasons_for_this_candidates_exposures():
    cand = candidate("c1", exposure_ids=("e1", "e2", "e3"))
    debits = [
        debit("d1", 160, exposure_id="e1"),
        debit("d2", None, exposure_id="e2"),
        debit("d3", 80, exposure_id="e3", partial=True),
        debit("other", 999, exposure_id="unrelated"),
    ]
    record = make([cand], debits)[receipt_id_for(TASK, "c1", METHOD)]
    names = [name for name, _ in record.receipt.costs]
    assert names == ["exposure:d1", "exposure:d2", "exposure:d3"]
    assert record.missing_costs == (("exposure:d2", "downstream_exposure_unobserved"),)


def test_receipt_id_is_stable_and_changes_with_task_candidate_or_methodology():
    base = receipt_id_for(TASK, "c1", METHOD)
    assert base == receipt_id_for(TASK, "c1", METHOD)
    assert base != receipt_id_for(TaskId("root-1", "turn-2", "main"), "c1", METHOD)
    assert base != receipt_id_for(TASK, "c2", METHOD)
    assert base != receipt_id_for(TASK, "c1", Methodology("other", (CHARS,)))
    assert len(base) == 24 and base.isalnum()


def test_canonical_bytes_are_deterministic_clock_free_and_content_free():
    cand = candidate("c1", execution_ids=("n1-0001",))
    first = make([cand])[receipt_id_for(TASK, "c1", METHOD)]
    second = make([cand])[receipt_id_for(TASK, "c1", METHOD)]
    assert first.canonical_bytes() == second.canonical_bytes() and first.digest == second.digest
    text = first.canonical_bytes().decode()
    assert json.loads(text)["format"] == "receipt_v1"
    assert "collected_at" not in text and "ts_utc" not in text
    local = first.with_content("SECRET-PATH /home/user/x.py")
    assert local.canonical_bytes() == first.canonical_bytes()  # content is never canonical
    assert "SECRET" not in json.dumps(default_view(local), default=str)
    assert local.content == "SECRET-PATH /home/user/x.py"


def test_digest_changes_when_any_allocation_or_cost_changes():
    plain = make([candidate("c1")])
    changed = make([candidate("c1", immediate=(atom("step-1", 800),))])
    rid = receipt_id_for(TASK, "c1", METHOD)
    assert plain[rid].digest == plain[rid].digest != changed[rid].digest


def test_expiry_marks_receipt_non_recomputable_and_drops_content():
    record = make([candidate("c1")])[receipt_id_for(TASK, "c1", METHOD)].with_content("secret")
    expired = record.mark_expired()
    assert not expired.receipt.evidence_available and expired.content is None
    assert ("recomputation", EXPIRED_REASON) in expired.missing_costs
    assert expired.digest != record.digest
    assert expired.receipt.allocations == record.receipt.allocations  # total explanation kept


def test_build_receipt_uses_only_allocations_of_its_own_task_and_unit():
    other_task = TaskId("root-9", "turn-1", "main")
    both = [candidate("c1"), candidate("c1", task=other_task)]
    valuation = value_evidence(EvidenceBundle(tuple(both)), METHOD)
    first = build_receipt(both[0], valuation.allocations, [], METHOD)
    second = build_receipt(both[1], valuation.allocations, [], METHOD)
    assert first.receipt.root == "root-1" and second.receipt.root == "root-9"
    assert dict(first.receipt.allocations)["conservative"] == ("step-1",)
    assert dict(second.receipt.allocations)["conservative"] == ("step-1",)  # separate consumer
    secs = Methodology("t", (UnitSpec("work_seconds", "monotonic_v1"),))
    assert build_receipt(both[0], valuation.allocations, [], secs).receipt.allocations == ()
    assert replace(first.receipt, revision=2).revision == 2
