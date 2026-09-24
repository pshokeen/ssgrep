"""Tests for eval.compare: delta table rendering and gate exit codes.

Synthetic payloads only (no index, no runner): the diff tool is a pure
function of two result JSONs, so the tests exercise ``compare.main`` against
hand-built payloads with an injected regression and an improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import compare


def _payload(
    *,
    ndcg10: float,
    rr10: float,
    r50: float,
    latency_p95_ms: float = 100.0,
    index_size_bytes: int = 2000,
    arm: str = "default",
) -> dict:
    return {
        "result_schema_version": "1.0",
        "dataset_version": "v1",
        "arm": arm,
        "summary": {
            "overall": {"ndcg@10": ndcg10, "rr@10": rr10, "r@50": r50},
            "holdout": {
                "ndcg@10": ndcg10 - 0.02,
                "rr@10": rr10 - 0.02,
                "r@50": r50 - 0.02,
            },
            "class:exact-identifier": {
                "ndcg@10": ndcg10 + 0.01,
                "rr@10": rr10 + 0.01,
                "r@50": r50 + 0.01,
            },
        },
        "latency_p95_ms": latency_p95_ms,
        "index_size_bytes": index_size_bytes,
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_gate_breach_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _payload(ndcg10=0.50, rr10=0.40, r50=0.80)
    regressed = _payload(ndcg10=0.48, rr10=0.40, r50=0.80, index_size_bytes=800)
    base_path = _write(tmp_path, "base.json", base)
    candidate_path = _write(tmp_path, "regressed.json", regressed)

    exit_code = compare.main([str(base_path), str(candidate_path), "--gates"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "-0.0200" in out  # the injected ndcg@10 delta is flagged in the table
    assert "FAIL" in out
    assert "GATES: FAIL" in out


def test_improved_candidate_clean_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _payload(ndcg10=0.50, rr10=0.40, r50=0.80)
    improved = _payload(ndcg10=0.55, rr10=0.45, r50=0.85, latency_p95_ms=90.0, index_size_bytes=800)
    base_path = _write(tmp_path, "base.json", base)
    candidate_path = _write(tmp_path, "improved.json", improved)

    exit_code = compare.main([str(base_path), str(candidate_path), "--gates"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "GATES: PASS" in out
    assert "+0.0500" in out  # ndcg@10 delta 0.55 - 0.50


def test_delta_table_without_gates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _payload(ndcg10=0.50, rr10=0.40, r50=0.80)
    candidate = _payload(ndcg10=0.52, rr10=0.41, r50=0.81)
    base_path = _write(tmp_path, "base.json", base)
    candidate_path = _write(tmp_path, "candidate.json", candidate)

    exit_code = compare.main([str(base_path), str(candidate_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "overall" in out
    assert "holdout" in out
    assert "class:exact-identifier" in out
    assert "+0.0200" in out


def test_tolerance_aware_formatting() -> None:
    assert compare._fmt_delta(0.0) == "  0.0000"
    assert compare._fmt_delta(1e-9) == "  0.0000"
    assert compare._fmt_delta(-0.02) == "-0.0200"
    assert compare._fmt_delta(None) == "    n/a"


def test_missing_slice_renders_n_a(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = _payload(ndcg10=0.50, rr10=0.40, r50=0.80)
    candidate = _payload(ndcg10=0.52, rr10=0.41, r50=0.81)
    del candidate["summary"]["holdout"]
    base_path = _write(tmp_path, "base.json", base)
    candidate_path = _write(tmp_path, "candidate.json", candidate)

    exit_code = compare.main([str(base_path), str(candidate_path)])
    assert exit_code == 0
    assert "n/a" in capsys.readouterr().out


def test_missing_summary_is_usage_error(tmp_path: Path) -> None:
    base_path = _write(tmp_path, "base.json", {"arm": "default"})
    candidate_path = _write(tmp_path, "candidate.json", _payload(ndcg10=0.5, rr10=0.4, r50=0.8))
    exit_code = compare.main([str(base_path), str(candidate_path)])
    assert exit_code == 2
