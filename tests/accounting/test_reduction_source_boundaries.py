"""Missing provenance never becomes free work or a fact shared across aliases."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.reduce import methodology_for
from ssgrep.accounting.valuation import rollup, value_evidence
from ssgrep.accounting.valuation_inputs import AtomStatus
from tests.accounting.reduce_builders import (
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    results_json,
    source_session,
    ts,
)
from tests.accounting.test_reduce_evidence import FACT, PATH, VIEW, consumer, reduce_scene
from tests.accounting.valuation_builders import CHARS


@pytest.mark.parametrize(
    "source, reason",
    [
        ([claude_prompt("src1", FACT, ts=ts(1))], "source_not_observed"),
        (
            [
                claude_prompt("src1", "investigate", ts=ts(1)),
                claude_tool_use(
                    "src1", "w1", "Bash", {"command": f"cat {PATH}"}, ts=ts(1, 10, 0, 5)
                ),
                claude_tool_result("src1", "w1", FACT, ts=ts(1, 10, 0, 6)),
            ],
            "source_known_before_witness",
        ),
        (source_session("src1", "no retained witness"), "witness_not_found"),
        (source_session("src1", FACT, when_day=13), "source_postdates_exposure"),
    ],
)
def test_unproven_source_retains_known_consumer_debit_and_unknown_credit(
    tmp_path: Path, source: list[dict], reason: str
) -> None:
    reduction = reduce_scene(tmp_path, consumer(), src=source)
    (record,) = reduction.candidates
    assert reason in {row.reason for row in reduction.exclusions}
    assert record.candidate.immediate
    assert all(atom.status is AtomStatus.UNAVAILABLE for atom in record.candidate.immediate)
    valuation = value_evidence(reduction.bundles[0], methodology_for(VIEW))
    estimates = rollup((leaf for leaf in valuation.leaves if leaf.unit == CHARS), CHARS)
    assert all(estimate.credit.value == 0 for estimate in estimates)
    assert all(estimate.debit.value and estimate.debit.value > 0 for estimate in estimates)
    assert all(estimate.net.value is None for estimate in estimates)
    assert all(estimate.known_net < 0 for estimate in estimates)


def test_legacy_unqualified_ref_resolves_retained_claude_source(tmp_path: Path) -> None:
    body = results_json(card("src1:ep:0", FACT))
    reduction = reduce_scene(tmp_path, consumer(body=body))
    (record,) = reduction.candidates
    assert {atom.atom.origin_event for atom in record.candidate.immediate} == {
        "claude:src1:call:w1",
        "claude:src1:result:w1",
    }
    assert all(atom.usable for atom in record.candidate.immediate)


def test_same_basename_in_another_directory_is_not_a_reference(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(action="vim services/legacy/limits.py"))
    assert reduction.candidates == ()
    assert reduction.coverage["candidates_unreferenced"] == 2
    assert reduction.bundles[0].debits


def test_absolute_action_path_matches_whole_relative_source_path(tmp_path: Path) -> None:
    reduction = reduce_scene(tmp_path, consumer(action=f"vim /work/proj/{PATH}"))
    (record,) = reduction.candidates
    assert record.candidate.action_outcome_ok
    assert record.candidate.immediate
    assert all(atom.usable for atom in record.candidate.immediate)


def test_home_identity_in_visible_response_is_charged_but_never_persisted_as_candidate(
    tmp_path: Path,
) -> None:
    secret_path = "/Users/private_operator/services/internal/credentials.py"
    body = results_json(card("claude:src1:ep:0", secret_path))
    reduction = reduce_scene(
        tmp_path,
        consumer(body=body, action=f"cat {secret_path}"),
        src=source_session("src1", secret_path),
    )
    assert reduction.candidates == ()
    assert reduction.coverage["exclusion.home_path"] == 1
    assert reduction.bundles[0].debits[0].chars == len(body)
    persisted = canonical_bytes(reduction)
    assert b"private_operator" not in persisted
    assert b"credentials.py" not in persisted
