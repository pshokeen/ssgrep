"""Source headers, empty context, malformed locators and incomplete request timing."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.contracts import Attribution
from ssgrep.accounting.reconcile import parse_argv
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.units import WORK_SECONDS, UnitMismatchError, UnitSpec
from ssgrep.accounting.valuation import debit_raw
from tests.accounting.reduce_builders import (
    KEY,
    PROFILE,
    card,
    claude_evidence,
    event,
    omp_call,
    omp_evidence,
    omp_header,
    omp_result,
    omp_user,
    policy,
    results_json,
    ts,
)
from tests.accounting.test_reduce_evidence import FACT, VIEW, consumer, reduce_scene
from tests.accounting.valuation_builders import debit


def test_malformed_card_locator_is_unresolved_despite_visible_action(tmp_path: Path) -> None:
    reduced = reduce_scene(tmp_path, consumer(body=results_json(card("not-a-session-ref", FACT))))
    assert reduced.candidates
    assert {row.reason for row in reduced.exclusions} == {"source_unresolved"}
    assert not any(atom.usable for atom in reduced.candidates[0].candidate.immediate)


def test_source_headers_and_empty_prompt_neither_become_atoms_nor_prior_knowledge(
    tmp_path: Path,
) -> None:
    source = [
        omp_header("source", ts(1)),
        omp_user("prompt", "", ts(1, s=1)),
        omp_call("status", "status", "status", {}, ts(1, s=1)),
        omp_call("call", "witness", "bash", {"command": "cat deploy.log"}, ts(1, s=2)),
        omp_result("result", "witness", FACT, ts(1, s=3)),
    ]
    evidence = omp_evidence(tmp_path, {"project/source.jsonl": source})
    evidence += claude_evidence(
        tmp_path,
        {"project/consumer.jsonl": consumer(body=results_json(card("omp:source:ep:0", FACT)))},
    )
    reduced = reduce_profile(
        PROFILE, evidence, [], policy=policy(), frequency=VIEW, fingerprint_key=KEY
    )
    (record,) = reduced.candidates
    atoms = record.candidate.immediate + record.candidate.discovery + record.candidate.episode
    assert atoms and all(atom.usable for atom in atoms)
    assert {atom.atom.origin_event for atom in atoms} == {
        "omp:source:call:witness",
        "omp:source:result:witness",
    }
    assert reduced.exclusions == ()


@pytest.mark.parametrize(
    "argv, expected",
    [
        (("note", "--title", "repair notes", "--body", "body"), ("note", "repair notes", None)),
        (("search", "--where", "runtime=claude", "repair"), ("search", "repair", "runtime=claude")),
        (("search", "repair", "--where"), ("search", "repair", "")),
    ],
)
def test_cli_option_values_never_become_query_text(argv: tuple[str, ...], expected: tuple) -> None:
    assert parse_argv(argv) == expected


def test_missing_transcript_clock_prevents_time_inferred_join(tmp_path: Path) -> None:
    records = consumer(action=None)
    for record in records:
        record.pop("timestamp", None)
    evidence = claude_evidence(tmp_path, {"project/consumer.jsonl": records})
    reduced = reduce_profile(
        PROFILE,
        evidence,
        event("lookup", ts(), text="why did the deploy fail"),
        policy=policy(),
        frequency=VIEW,
        fingerprint_key=KEY,
    )
    assert reduced.executions[0].attribution is Attribution.UNATTRIBUTED
    assert reduced.executions[0].session_id is None
    assert reduced.invocations[0].execution_ids == ()


def test_public_debit_conversion_rejects_cross_unit_request() -> None:
    with pytest.raises(UnitMismatchError):
        debit_raw(debit("visible", 120), UnitSpec(WORK_SECONDS, "tool_duration_v1"))
