"""Accounting sensitivity cases against the REAL ssgrep binaries and the real embedding model.

One module-scoped run builds one isolated lab root and a genuine per-Scene source index
using the cached real model. Production uses the actual tiny scoped corpus. Positive
arithmetic controls use the separately named private fixture driver with explicit frozen
rarity inputs; they are NOT production-policy credit or exportable savings.
Both lanes execute the real collector and compare with independent raw observer evidence.

Where the product and the oracle genuinely differ the case is a ``DISAGREEMENT`` (never bent):
the tests pin the exact numbers on both sides and the diagnosis so a silent change of either
side fails here.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from eval.accounting_airline import lab_product as lp, sensitivity as sens
from eval.accounting_airline.lab_product import BINARY, real_model_cache
from eval.accounting_airline.sensitivity import (
    BLOCKED,
    CASES,
    DISAGREEMENT,
    FAIL,
    PASS,
    CaseResult,
    Check,
    run_all,
)

pytestmark = pytest.mark.slow

EXPECTED_STATUS = {
    "empty_error_missing_lookups": PASS,
    "tiny_show_truncation": PASS,
    "repeated_lookups": PASS,
    "mentions_not_executions": PASS,
    "shell_loop_two_searches": PASS,
    "prior_known_and_query": PASS,
    "relay_vs_sibling": PASS,
    "duplicate_mutated_partial_histories": PASS,
    "journal_write_loss": PASS,
    "source_expiry": PASS,
    "recollection_replay_no_extra_usage": PASS,
    "omp_trailer_exposure": PASS,
}
EVIDENCE_ID = re.compile(
    r"^(solver-b-[0-9a-f]{10}|[a-z0-9]+:call:standin_\w+|snapshot:[0-9a-f]{16}"
    r"|journal_gap_vs_observed:\d+)$"
)


@pytest.fixture(scope="module")
def emitted() -> list[CaseResult]:
    return []


@pytest.fixture(scope="module")
def results(
    tmp_path_factory: pytest.TempPathFactory, emitted: list[CaseResult]
) -> Iterator[dict[str, CaseResult]]:  # noqa: E501
    if not BINARY.exists():
        pytest.fail(f"the real ssgrep binary is missing: {BINARY}")
    workdir = tmp_path_factory.mktemp("sensitivity")
    rows = run_all(
        workdir,
        model_cache=real_model_cache(),
        emit=emitted.append,
        diagnostics=workdir / "private-diagnostics",
    )
    assert [r.name for r in rows] == list(CASES)
    yield {r.name: r for r in rows}


def test_check_status_rules() -> None:
    clean = Check()
    assert clean.status() == PASS
    clean.disagree("x", "why")
    assert clean.status() == DISAGREEMENT and "DISAGREEMENT x: why" in clean.detail()
    broken = Check()
    broken.eq("a", 1, 2)
    broken.disagree("x", "why")
    assert broken.status() == FAIL  # a violated expectation is never downgraded by a disagreement
    waiting = Check()
    waiting.blocked.append("no relay")
    assert waiting.status() == BLOCKED


def test_scene_collects_original_consumer_files_not_private_copies(tmp_path: Path) -> None:
    from eval.accounting_airline.roots import LabRoots

    roots = LabRoots.create(tmp_path)
    try:
        base = roots.artifacts / "base"
        view = base / "view"
        view.mkdir(parents=True, mode=0o700)
        verifier = roots.verifier / "base-observer"
        verifier.mkdir(mode=0o700)
        fx = sens.Fixture(roots, tmp_path, roots.ssgrep_data, view, verifier, roots.tools, view)
        sc = sens.Scene(fx, "originals", fresh=True)
        path = sc.start_session("original-consumer")
        assert path.parent == sc.view == sc.sessions
        assert path.stat().st_nlink == 1
        assert sc.env("control")["SSGREP_OMP_SESSIONS_DIR"] == str(path.parent)
        sc.sync("original-consumer", path, "original-consumer#0")
        copied = sc.view / "explicit-copy.jsonl"
        tail = b'{"incomplete":'
        copied.write_bytes(path.read_bytes() + tail)
        sc.observe_copy(copied)
        frozen = sc.frozen()
        copy_key = "original-consumer@copy-explicit-copy.jsonl"
        original = frozen.sessions["original-consumer"][2]
        variant = frozen.sessions[copy_key][2]
        assert variant.session_id == original.session_id == "original-consumer"
        assert variant.complete_bytes == original.complete_bytes
        assert variant.partial_tail_bytes == len(tail)
        assert variant.source_aliases == original.source_aliases == ()
    finally:
        roots.remove()


def test_private_diagnostic_keeps_raw_snapshots_and_unavailable_reasons(tmp_path: Path) -> None:
    collected = lp.Collected(
        {"data": {"snapshot_id": "raw-snapshot", "generation": 3}},
        {},
        {"data": {"missing_evidence": {"capture_completeness": "unknown"}}},
        12,
        {
            "ok": True,
            "data": {
                "estimates": {
                    "context_tokens": {
                        "central": {
                            "credit": {"value": 0},
                            "net": {"value": None, "reason": "incomplete:source_unapproved"},
                        }
                    }
                }
            },
        },
    )
    target = tmp_path / "private" / "attempt.json"
    sens.write_diagnostic(target, collected)
    raw = json.loads(target.read_text())
    assert raw["collect"] == collected.collect
    assert raw["savings"] == collected.savings_document
    assert raw["status"] == collected.status
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700


def test_private_case_outcome_preserves_full_exception(tmp_path: Path) -> None:
    row = CaseResult("source_expiry", FAIL, {}, {}, (), "bounded summary")
    cause = RuntimeError("full private cause " + "x" * 2000)
    target = tmp_path / "outcomes" / "source_expiry.json"
    sens.write_case_outcome(target, row, cause)
    raw = json.loads(target.read_text())
    assert raw["result"] == row.as_dict()
    assert raw["exception"] == {"type": "RuntimeError", "message": str(cause)}
    assert target.stat().st_mode & 0o777 == 0o600


def test_recollection_projection_preserves_frequency_authority_and_unknown_values() -> None:
    from eval.accounting_airline.sensitivity_cases import accounting_view

    def collected(transition: str, authority: str = "owned") -> lp.Collected:
        return lp.Collected(
            {},
            {},
            {},
            0,
            {
                "estimates": {
                    "context_tokens": {
                        "central": {
                            "credit": {"value": 0},
                            "net": {"value": None, "reason": "incomplete"},
                            "provenance": [
                                ["frequency_revision", transition],
                                ["frequency", "current"],
                            ],
                        }
                    }
                },
                "measured": {"searches": {"value": 2}},
                "coverage": {
                    "frequency": {
                        "transition": transition,
                        "previous_view_id": "old",
                        "revision_name": "named",
                        "authority_id": authority,
                        "view_id": "current",
                    }
                },
            },
        )

    initial = collected("initial")
    replacement = collected("replacement")
    assert accounting_view(initial) == accounting_view(replacement)
    assert accounting_view(initial) != accounting_view(collected("unchanged", "foreign"))
    assert replacement.coverage["frequency"]["revision_name"] == "named"
    assert replacement.estimates["context_tokens"]["central"]["provenance"][0] == [
        "frequency_revision",
        "replacement",
    ]
    assert accounting_view(initial)["estimates"]["context_tokens"]["central"]["net"] == {
        "value": None,
        "reason": "incomplete",
    }


def test_initial_index_reports_the_missing_offline_model(tmp_path: Path) -> None:
    from ssgrep.indexing.embed import DEFAULT_MODEL_ID

    with pytest.raises(sens.InfrastructureError) as failed:
        sens.Fixture.create(tmp_path, model_cache=tmp_path / "empty-model-cache")
    detail = str(failed.value)
    assert "initial index failed:" in detail
    assert "ModelDownloadError" in detail and DEFAULT_MODEL_ID in detail
    assert "HF_HUB_OFFLINE" in detail
    assert len(detail) < 1400
    assert not list(tmp_path.glob("ssgrep-airline-lab-*"))


def test_index_diagnostic_keeps_the_cause_and_bounds_large_output(tmp_path: Path) -> None:
    launcher = tmp_path / "failing-ssgrep"
    body = {"ok": False, "error": {"type": "IndexFailure", "message": "root-cause " + "x" * 8000}}
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"print({json.dumps(body)!r})\n"
        "print('irrelevant warning ' * 1000, file=sys.stderr)\n"
        "print('Try `ssgrep index --rebuild` to force a full rebuild.', file=sys.stderr)\n"
        "sys.exit(7)\n"
    )
    launcher.chmod(0o700)
    product = lp.Product({"SSGREP_DATA_DIR": str(tmp_path / "data")}, tmp_path, launcher)
    code, elapsed_ms, detail = lp.index(product)
    assert code == 7 and elapsed_ms >= 0
    assert detail.startswith("IndexFailure: root-cause ")
    assert len(detail) <= 1200
    assert "irrelevant warning" not in detail


def test_run_all_emitted_every_row_in_order(
    results: dict[str, CaseResult], emitted: list[CaseResult]
) -> None:  # noqa: E501
    assert [r.name for r in emitted] == list(CASES)
    assert {r.name: r.status for r in emitted} == {k: v.status for k, v in results.items()}


def test_every_case_has_the_expected_status_and_evidence(results: dict[str, CaseResult]) -> None:
    for name in CASES:
        row = results[name]
        assert row.status == EXPECTED_STATUS[name], (name, row.status, row.detail)
        assert row.evidence, name
        bad = [e for e in row.evidence if not EVIDENCE_ID.match(e)]
        assert not bad, (name, bad)  # ids only: never raw transcript or tool text
        assert isinstance(row.expected, dict) and isinstance(row.actual, dict)
        assert row.valuation_profile in (sens.oracle.PRODUCTION, sens.oracle.ARITHMETIC)
        assert row.as_dict()["valuation_profile"] == row.valuation_profile
        for value in (*row.expected.values(), *row.actual.values()):
            assert not isinstance(value, float), (name, value)  # no float expectations
        if row.status == PASS:
            assert row.detail == "" or "DISAGREEMENT" not in row.detail, name


def test_empty_error_and_missing_lookups(results: dict[str, CaseResult]) -> None:
    row = results["empty_error_missing_lookups"]
    assert row.actual["measured.searches"] == row.expected["measured.searches"] == 4
    assert row.actual["measured.shows"] == row.expected["measured.shows"] == 2
    assert row.actual["measured.failed"] == row.expected["measured.failed"] == 3
    assert row.actual["debit"] == row.expected["debit"] > 0
    assert row.actual["journal_rows_added"] == 3  # the usage error added no journal event
    # positive twin lives in the base fixture: control + verifier lookups were journaled ok/empty
    assert "journal records the scope/missing-ref lookups as empty" in row.detail
    assert "DISAGREEMENT" not in row.detail


def test_tiny_show_hides_the_identifier_and_earns_nothing(results: dict[str, CaseResult]) -> None:
    row = results["tiny_show_truncation"]
    assert row.actual["credit"] == row.expected["credit"] == 0
    assert row.actual["debit"] == row.expected["debit"] > 0
    assert row.valuation_profile == sens.oracle.ARITHMETIC
    assert row.actual["twin_credit"] == row.expected["twin_credit"] == 48
    assert (
        row.actual["twin_credits"]
        == row.expected["twin_credits"]
        == {
            "conservative": 26,
            "central": 48,
            "optimistic": 87,
        }
    )
    assert row.actual["net"] == row.expected["net"] == -row.expected["debit"]
    assert row.detail == ""


def test_repeated_lookups_double_debit_not_credit(results: dict[str, CaseResult]) -> None:
    row = results["repeated_lookups"]
    assert row.actual == row.expected
    # the exact per-run values equal visible chars // 4 (asserted in the case); the wall-time
    # trailer digits differ between runs, so "double" is checked as a bound, not equality
    assert 2 * row.actual["debit_double"] > 3 * row.actual["debit_single"]
    assert row.actual["credit_central"] > 0


def test_mentions_are_not_executions(results: dict[str, CaseResult]) -> None:
    row = results["mentions_not_executions"]
    assert row.actual["solver_dispatches"] == 0
    assert row.actual["debit"] in (None, 0)
    assert row.actual["twin_debit"] == row.expected["twin_debit"] > 0  # real lookup IS debited


def test_loop_is_one_invocation_two_executions(results: dict[str, CaseResult]) -> None:
    row = results["shell_loop_two_searches"]
    assert (row.actual["invocations"], row.actual["executions"]) == (1, 2)
    assert row.actual["measured_searches"] == row.expected["measured_searches"] == 4
    assert row.actual["debit_once"] == row.expected["debit_once"] > 0
    assert row.expected["oracle_debit"] == row.actual["debit_once"]
    assert row.actual["product_debit_arith"] == row.expected["oracle_debit"] > 0
    assert row.valuation_profile == sens.oracle.ARITHMETIC
    assert row.actual["separate_lookups"] == row.expected["separate_lookups"] == 2
    assert row.actual["separate_debit"] == row.expected["separate_debit"] > 0
    assert row.actual["separate_credit"] == row.expected["separate_credit"]
    assert row.actual["positive_control_kind"] == "unmatched_two_searches_plus_full_show"
    assert row.actual["positive_control_lookups"] == row.expected["positive_control_lookups"] == 3
    assert row.actual["positive_control_credit"] == row.expected["positive_control_credit"] == 48
    assert row.actual["positive_control_debit"] == row.expected["positive_control_debit"] > 0
    assert row.actual["positive_control_extra_show_chars"] > 0
    assert row.detail == ""


def test_prior_known_and_query_mentions_earn_nothing(results: dict[str, CaseResult]) -> None:
    row = results["prior_known_and_query"]
    assert row.actual == row.expected
    assert row.actual["credit"] == 0 and row.actual["twin_credit"] > 0


def test_relay_credits_only_the_explicit_relay(results: dict[str, CaseResult]) -> None:
    row = results["relay_vs_sibling"]
    assert row.actual["relay_credit"] == row.expected["relay_credit"] > 0
    assert row.actual["sibling_extra_credit"] == row.expected["sibling_extra_credit"] == 0
    assert row.actual["no_payload_credit"] == row.actual["no_header_credit"] == 0
    assert row.detail == ""


def test_duplicate_mutated_and_partial_histories_count_once(results: dict[str, CaseResult]) -> None:
    row = results["duplicate_mutated_partial_histories"]
    assert row.status == PASS, row.detail
    assert row.actual["debit_before"] == row.expected["debit_before"] > 0
    assert row.actual["debit_after_identical"] == row.expected["debit_after_identical"]
    assert row.actual["debit_after_identical"] == row.actual["debit_before"]
    assert (
        row.actual["known_debit_after_conflict"]
        == row.expected["known_debit_after_conflict"]
        < row.actual["debit_before"]
    )
    assert row.actual["executions"] == row.expected["executions"] == 2
    assert row.actual["net_after_conflict"] == row.expected["net_after_conflict"] is None
    assert row.actual["partial_tail_bytes"] == row.expected["partial_tail_bytes"] > 0
    # A same-origin/different-digest variant changes the evidence set: disputed
    # exposure is unavailable, NOT a known zero or another execution.
    assert row.actual["conflicts_min"] >= row.expected["conflicts_min"] == 1


def test_journal_loss_is_a_gap_not_a_fabricated_count(results: dict[str, CaseResult]) -> None:
    row = results["journal_write_loss"]
    assert row.expected["observed_searches"] - row.actual["journal_searches"] == 1
    assert row.actual["journal_gap"] == 1
    assert row.actual["capture_completeness"] == "unknown"


def test_source_expiry_keeps_closed_totals(results: dict[str, CaseResult]) -> None:
    row = results["source_expiry"]
    assert row.status == PASS, row.detail
    assert row.actual["credit"] == row.expected["credit"] > 0
    assert row.actual["debit"] == row.expected["debit"]
    assert row.actual["missing_sources"] == 1 and row.actual["fresh_replay_net"] is None
    assert (
        row.actual["source_adapter_status"]
        == row.expected["source_adapter_status"]
        == "source_missing"
    )
    assert (
        row.actual["optimistic_work_complete"] == row.expected["optimistic_work_complete"] is False
    )
    assert row.actual["optimistic_work_net"] == row.expected["optimistic_work_net"] is None
    assert (
        row.actual["optimistic_work_credit_upper"]
        == row.expected["optimistic_work_credit_upper"]
        is None
    )
    assert (
        row.actual["optimistic_work_terminal_gap"]
        == row.expected["optimistic_work_terminal_gap"]
        is True
    )
    assert "7-day provisional window" in row.detail


def test_recollection_and_replay_add_no_usage(results: dict[str, CaseResult]) -> None:
    row = results["recollection_replay_no_extra_usage"]
    assert (
        row.actual
        == row.expected
        == {
            "digest_equal": True,
            "journal_bytes_unchanged_by_reads": True,
        }
    )


def test_omp_trailer_does_not_change_credit(results: dict[str, CaseResult]) -> None:
    row = results["omp_trailer_exposure"]
    assert row.actual["credit_none"] == row.actual["credit_omp"] > 0
    assert row.valuation_profile == sens.oracle.ARITHMETIC
    assert "trailer chars" in row.detail


def test_unknown_case_is_rejected() -> None:
    with pytest.raises(KeyError):
        sens.run_case(cast(Any, None), "no_such_case")


@pytest.mark.parametrize(
    ("variants", "conflicts"),
    [
        (("clean", "clean", "changed"), 1),
        (("changed", "clean", "clean"), 2),
    ],
)
def test_conflict_drop_count_depends_on_first_admitted_copy(variants, conflicts):
    from ssgrep.accounting.collector_inputs import ClassifyingLedger
    from ssgrep.accounting.readers.values import make_evidence

    ledger = ClassifyingLedger()
    admitted, rejected = 0, []
    for index, value in enumerate(variants):
        evidence = make_evidence(
            runtime="omp",
            source_id=f"omp:/fixture/copy-{index}",
            branch_id="main",
            session_id="one-session",
            record_id="result:one-call",
            locator="L1",
            kind="tool_result",
            ts=None,
            payload={"output": value},
        )
        if ledger.admit(evidence):
            admitted += 1
        else:
            rejected.append(ledger.rejection(evidence))
    assert admitted == 1
    assert rejected.count("conflict") == conflicts
