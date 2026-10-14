"""C9 acceptance through real profile collection, loopback HTTP and OTLP delivery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eval.accounting_airline.lab_c9 import run_c9, source_digests
from eval.accounting_airline.lab_product import Collected, Product, collect, setup_profile
from eval.accounting_airline.roots import LabRoots, runtime_env
from ssgrep.accounting import capture, snapshots
from ssgrep.accounting.contracts import Operation, Outcome
from ssgrep.accounting.journal import close_all


def test_c9_passes_only_after_real_scrape_full_ack_and_partial_ack_are_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = LabRoots.create(tmp_path, "c9-acceptance")
    env = runtime_env(roots, "verifier")
    product = Product(env, roots.project)
    monkeypatch.setenv("HOME", env["HOME"])
    monkeypatch.setenv("SSGREP_DATA_DIR", env["SSGREP_DATA_DIR"])
    setup_profile(product)

    def record_search(text: str) -> None:
        execution = capture.begin(Operation.SEARCH, text=text, cwd=str(roots.project))
        assert execution is not None
        capture.complete(execution, Outcome.OK)

    # Both generations come from actual journal writes and the real collector command.
    source_canary = "synthetic-source-canary-C9"
    record_search(source_canary)
    try:
        previous = collect(product)
        previous_generation = snapshots.current_generation("personal-local")
        assert previous_generation is not None

        def collect_latest() -> Collected:
            record_search("synthetic C9 post-ack evidence")
            return collect(product)

        outcome = run_c9(product, previous, collect_latest, secrets=(source_canary,))
    finally:
        close_all()

    assert outcome.scrape.status == "PASS", outcome.scrape.detail
    assert outcome.status == "PASS"
    assert outcome.export.status == "PASS", outcome.export.detail
    assert snapshots.current_generation("personal-local") == outcome.snapshot_generation
    assert outcome.snapshot_generation is not None
    assert outcome.snapshot_generation > previous_generation
    assert outcome.evidence["collected_ahead_of_acknowledged"] is True
    assert outcome.evidence["sample_count"] > 0
    assert any(
        sample["name"] == "ssgrep_executions_total"
        and sample["labels"].get("operation") == "search"
        and int(sample["value"]) > 0
        for sample in outcome.evidence["scrape_samples"]
    )
    assert outcome.evidence["metrics_ack_generation"] == outcome.snapshot_generation
    assert outcome.evidence["metrics_ack_snapshot_id"] == outcome.snapshot_id
    assert outcome.evidence["logs_consent_was_separate"] is True
    assert outcome.evidence["partial_rejected_points"] > 0
    assert outcome.evidence["partial_snapshot_was_not_acknowledged"] is True
    assert outcome.evidence["privacy_canary_absent"] is True
    assert outcome.evidence["accepted_metric_points"] > 0


def test_missing_real_parser_and_prometheus_prerequisite_blocks_overall_c9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    from tests.accounting import prom_lab

    roots = LabRoots.create(tmp_path, "c9-blocked")
    env = runtime_env(roots, "verifier")
    product = Product(env, roots.project)
    monkeypatch.setenv("HOME", env["HOME"])
    monkeypatch.setenv("SSGREP_DATA_DIR", env["SSGREP_DATA_DIR"])
    setup_profile(product)

    def record_search(text: str) -> None:
        execution = capture.begin(Operation.SEARCH, text=text, cwd=str(roots.project))
        assert execution is not None
        capture.complete(execution, Outcome.OK)

    source_canary = "C9 blocked source canary"
    record_search(source_canary)
    try:
        previous = collect(product)

        def collect_latest() -> Collected:
            record_search("C9 blocked prerequisite successor")
            return collect(product)

        real_import = builtins.__import__

        def without_prometheus_parser(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "prometheus_client.openmetrics.parser":
                raise ImportError("parser intentionally unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_prometheus_parser)
        monkeypatch.setattr(prom_lab, "docker_available", lambda: "prometheus image absent")
        outcome = run_c9(product, previous, collect_latest, secrets=(source_canary,))
    finally:
        close_all()

    assert outcome.scrape.status == "BLOCKED"
    assert outcome.export.status == "PASS", outcome.export.detail
    assert outcome.status == "BLOCKED"

    assert outcome.scrape.detail == "prometheus image absent"


def test_c9_mutated_openmetrics_response_missing_eof_is_rejected() -> None:
    from eval.accounting_airline.lab_c9 import _assert_samples, _CheckFailure, _Reply
    from ssgrep.accounting.metrics_render import CONTENT_TYPE
    from tests.accounting.http_builders import make_snapshot

    reply = _Reply(200, {"content-type": CONTENT_TYPE}, b"# TYPE ssgrep_snapshot_available gauge\n")
    with pytest.raises(_CheckFailure, match="openmetrics_eof_missing"):
        _assert_samples(reply, make_snapshot(1), None)


def test_native_source_digest_detects_same_size_content_mutation(tmp_path: Path) -> None:
    sessions = tmp_path / "runtime-a"
    sessions.mkdir()
    source = sessions / "session.jsonl"
    source.write_bytes(b"source-a")
    before = source_digests({"a": sessions})
    source.write_bytes(b"source-b")
    after = source_digests({"a": sessions})

    assert source.stat().st_size == len(b"source-a")
    assert before != after


def test_preview_gate_rejects_accounting_state_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval.accounting_airline.lab_c9 import _CheckFailure, _preview

    data_dir = tmp_path / "data"
    (data_dir / "accounting").mkdir(parents=True)
    product = Product({"SSGREP_DATA_DIR": str(data_dir)}, tmp_path)

    def mutating_preview(self: Product, *_args: Any, **_kwargs: Any) -> tuple[int, dict[str, Any]]:
        (self.data_dir / "accounting" / "preview-write").write_bytes(b"unexpected")
        return 0, {"data": {"data": {"network": False, "state": "ok"}}}

    monkeypatch.setattr(Product, "json", mutating_preview)
    with pytest.raises(_CheckFailure, match="metrics_preview_changed_accounting_state"):
        _preview(product, "airline-lab-c9", "metrics")
