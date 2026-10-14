"""Entry point behaviour (offline) and the full scripted-standin pipeline (docker + real index)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval.accounting_airline import __main__ as entry
from eval.accounting_airline.gates import BLOCKED, NOT_RUN, PASS


def _live_report(tmp_path: Path) -> dict:
    return json.loads(
        next(tmp_path.glob("ssgrep-airline-run-*/artifacts/live-lane.json")).read_text()
    )


def test_live_without_authorization_is_not_run_and_names_the_missing_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = entry.main(["--mode", "live", "--runtime", "omp", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 3 and "--authorize-live" in out and "[PASS" not in out.split("live.two")[-1]
    verdict = _live_report(tmp_path)["gates"][-1]
    assert verdict["name"] == "live.two_session_lab" and verdict["status"] == NOT_RUN
    assert _live_report(tmp_path)["authorized"] is False


def test_missing_live_authorization_never_reaches_any_external_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    def refused(*_args, **_kwargs):
        pytest.fail("unauthorized live request reached an external process")

    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setattr(subprocess, "Popen", refused)
    code = entry.main(["--mode", "live", "--runtime", "omp", "--run-root-parent", str(tmp_path)])
    assert code == 3
    assert "authorization" in capsys.readouterr().out
    verdict = _live_report(tmp_path)["gates"][-1]
    assert verdict["status"] == NOT_RUN
    assert verdict["status"] != BLOCKED  # Authorization refusal, not a failed live attempt.


def test_standin_mode_is_listed_and_live_requires_runtime_gate(tmp_path: Path) -> None:
    code = entry.main(["--mode", "live", "--run-root-parent", str(tmp_path)])
    assert code == 3
    assert _live_report(tmp_path)["gates"][-1]["status"] == NOT_RUN


@pytest.mark.postgres
def test_standin_pipeline_runs_c0_to_c10_with_real_binaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker unavailable")
    code = entry.main(["--mode", "standin", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    report = json.loads(next(tmp_path.glob("*-report/report.json")).read_text())
    assert report["runtime"] == "scripted-standin" and report["counts_toward_live_gate"] is False
    assert report["gates"]["live_session_gate"]["status"] == NOT_RUN
    assert "scripted-standin" in out
    for cid in (f"C{i}" for i in range(11)):
        assert report["checkpoints"][cid]["domain_status"] in (PASS, BLOCKED)
        assert report["checkpoints"][cid]["accounting_status"] in (PASS, BLOCKED)
    c9 = report["checkpoints"]["C9"]
    assert c9["domain_status"] == PASS
    c9_domain = {item["name"]: item for item in c9["domain"]}
    assert c9_domain["c9.native_source_contents_unchanged"]["status"] == PASS
    assert (
        c9_domain["c9.native_source_contents_unchanged"]["expected"]
        == c9_domain["c9.native_source_contents_unchanged"]["actual"]
    )
    c9_transport = c9["actual_delta"]["c9_transport"]
    assert c9_transport["accepted_metric_points"] > 0
    assert c9_transport["partial_rejected_points"] > 0
    assert c9_transport["partial_snapshot_was_not_acknowledged"] is True
    assert c9_transport["privacy_canary_absent"] is True
    assert c9_transport["metrics_ack_generation"] == c9_transport["snapshot_generation"]
    assert c9_transport["metrics_ack_snapshot_id"] == c9_transport["snapshot_id"]
    assert report["gates"]["export_privacy"]["status"] == PASS
    if report["gates"]["scrape"]["status"] == BLOCKED:
        assert c9["accounting_status"] == BLOCKED
        assert report["gates"]["scrape"]["detail"]
    else:
        assert report["gates"]["scrape"]["status"] == PASS
        assert c9["accounting_status"] == PASS
        samples = c9_transport["scrape_samples"]
        assert len(samples) == c9_transport["sample_count"]
        assert any(
            row["name"] == "ssgrep_executions_total"
            and row["labels"].get("operation") == "search"
            and int(row["value"]) > 0
            for row in samples
        )
        assert c9_transport["sample_count"] > 0
    assert code == 0
    for group in ("valuation_production", "valuation_arithmetic"):
        assert report["gates"][group]["status"] == PASS
    text = json.dumps(report)
    assert "PGPASSWORD" not in text
    assert not list(tmp_path.glob("ssgrep-airline-lab-*[!t]/"))  # lab root torn down
