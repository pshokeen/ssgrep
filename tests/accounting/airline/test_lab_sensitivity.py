"""Accounting sensitivity cases against the REAL ssgrep binaries and the real embedding model.

One module-scoped run builds ONE isolated lab root with ONE real index and executes every case
(``run_all``); each test then asserts what that case measured. Nothing is mocked: the product
side is ``.venv/bin/ssgrep`` (index/search/show/metrics collect/savings/metrics status), the
solver side is the scripted stand-in running real shell commands behind the observer shims, and
the expectation side is the independent oracle + observer evidence.

Where the product and the oracle genuinely differ the case is a ``DISAGREEMENT`` (never bent):
the tests pin the exact numbers on both sides and the diagnosis so a silent change of either
side fails here.
"""

from __future__ import annotations

import json
import re
import shutil
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
    rows = run_all(workdir, model_cache=real_model_cache(), emit=emitted.append)
    assert [r.name for r in rows] == list(CASES)
    yield {r.name: r for r in rows}
    shutil.rmtree(workdir, ignore_errors=True)


def test_case_list_is_frozen_and_ordered() -> None:
    assert CASES[:11] == (
        "empty_error_missing_lookups",
        "tiny_show_truncation",
        "repeated_lookups",
        "mentions_not_executions",
        "shell_loop_two_searches",
        "prior_known_and_query",
        "relay_vs_sibling",
        "duplicate_mutated_partial_histories",
        "journal_write_loss",
        "source_expiry",
        "recollection_replay_no_extra_usage",
    )
    assert set(CASES) == set(EXPECTED_STATUS)


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
    assert row.actual["twin_credit"] == row.expected["twin_credit"] > 0  # positive twin
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
    assert row.actual["debit"] == row.expected["debit"]
    assert row.actual["partial_tail_bytes"] == row.expected["partial_tail_bytes"] > 0
    # This is a lower bound, not an exact drop count. Hash-ordered sources may admit
    # the mutated variant first, in which case both identical clean copies are rejected.
    assert row.actual["conflicts_min"] >= row.expected["conflicts_min"] == 1


def test_journal_loss_is_a_gap_not_a_fabricated_count(results: dict[str, CaseResult]) -> None:
    row = results["journal_write_loss"]
    assert row.expected["observed_searches"] - row.actual["journal_searches"] == 1
    assert row.actual["journal_gap"] == 1
    assert row.actual["capture_completeness"] == "unknown"


def test_source_expiry_keeps_closed_totals(results: dict[str, CaseResult]) -> None:
    row = results["source_expiry"]
    assert row.actual["credit"] == row.expected["credit"] > 0
    assert row.actual["debit"] == row.expected["debit"]
    assert row.actual["missing_sources"] == 1 and row.actual["fresh_replay_net"] is None
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
    assert "structured=False" in row.detail


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
