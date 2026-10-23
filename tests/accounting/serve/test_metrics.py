"""Metric catalog, honest counter/gauge/histogram semantics, bounded dimensions, OpenMetrics text.

The text is parsed with ``prometheus_client``'s strict OpenMetrics parser (installed as a dev
tool of another dependency; NOT a ssgrep dependency) so serialization is judged by a real parser.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import select
import signal
import socket
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from prometheus_client.openmetrics.parser import text_string_to_metric_families

from ssgrep.accounting.ledger.units import UnitSpec
from ssgrep.accounting.serve import metrics
from ssgrep.accounting.serve.metrics_render import Sample, ValidationError, render_openmetrics
from tests.accounting.export.http_builders import (
    COLLECTED_EPOCH,
    make_snapshot,
    synthetic_current_scenarios,
    synthetic_current_window,
    window_with_unknown_optimistic,
)
from tests.accounting.views.test_savings_commands import BINARY, Run, document

HEALTH = [
    metrics.ExportHealth(
        "acme",
        "metrics",
        last_full_ack_epoch=COLLECTED_EPOCH - 60,
        pending_batches=2,
        rejected_points=5,
        lost=(("spool_expired", 7),),
    ),
    metrics.ExportHealth("acme", "logs", pending_batches=0),
]


def parse(samples: list[Sample]) -> dict[str, Any]:
    families = list(text_string_to_metric_families(render_openmetrics(samples).decode()))
    return {family.name: family for family in families}


def value_of(family: Any, name: str, **labels: str) -> float:
    for sample in family.samples:
        if sample.name == name and sample.labels == labels:
            return sample.value
    raise AssertionError(f"no {name} sample with {labels}")


def test_consumer_visible_types_and_numerical_series_are_supported() -> None:
    samples = metrics.all_samples(make_snapshot(3), HEALTH)
    families = parse(samples)
    assert families["ssgrep_executions"].type == "counter"
    assert families["ssgrep_operation_duration_seconds"].type == "histogram"
    assert families["ssgrep_estimated_net_tokens"].type == "gauge"
    assert families["ssgrep_estimate_bound_tokens"].type == "gauge"
    labels = {"scenario": "central", "window": "7d", "method": "chars_div4_v1"}
    assert (
        value_of(families["ssgrep_estimated_net_tokens"], "ssgrep_estimated_net_tokens", **labels)
        == -7000
    )
    assert (
        value_of(
            families["ssgrep_estimate_bound_tokens"],
            "ssgrep_estimate_bound_tokens",
            **labels,
            quantity="net",
            bound="lower",
        )
        == -7000
    )


def test_counters_are_total_suffixed_integer_and_never_negative() -> None:
    samples = metrics.all_samples(make_snapshot(3), HEALTH)
    totals = [s for s in samples if s.kind == "counter"]
    assert totals and all(s.family.endswith("_total") for s in totals)
    assert all(type(s.value) is int and s.value >= 0 for s in totals)
    executions = {
        dict(s.labels)["outcome"]: s.value for s in samples if s.family == "ssgrep_executions_total"
    }
    assert executions == {"ok": 3, "error": 1}
    assert {
        dict(s.labels)["runtime"] for s in samples if s.family == "ssgrep_executions_total"
    } == {"unknown"}
    sized = {s.value for s in samples if s.family == "ssgrep_rendered_bytes_total"}
    assert sized == {3000}


def test_histogram_buckets_are_cumulative_with_inf_and_sum_in_seconds() -> None:
    samples = metrics.all_samples(make_snapshot(3), [])
    family = parse(samples)["ssgrep_operation_duration_seconds"]
    cli = {"operation": "search", "transport": "cli"}
    bounds = [
        s.labels["le"]
        for s in family.samples
        if s.name.endswith("_bucket") and s.labels["operation"] == "search"
    ]
    assert bounds == ["0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1.0", "5.0", "30.0", "+Inf"]
    counts = [
        s.value
        for s in family.samples
        if s.name.endswith("_bucket") and s.labels["operation"] == "search"
    ]
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == 3
    assert counts[2] == 3  # three 2 ms observations land in le=0.005
    assert value_of(family, "ssgrep_operation_duration_seconds_count", **cli) == 3
    assert value_of(family, "ssgrep_operation_duration_seconds_sum", **cli) == pytest.approx(0.006)


def test_savings_gauge_may_be_negative_and_decrease_while_counters_never_do() -> None:
    first = metrics.snapshot_samples(make_snapshot(1))
    revised = metrics.snapshot_samples(make_snapshot(2, extra_searches=2, window=revised_window()))

    def net(samples: list[Sample], scenario: str) -> int:
        (found,) = [
            s.value
            for s in samples
            if s.family == "ssgrep_estimated_net_tokens"
            and dict(s.labels)["scenario"] == scenario
            and dict(s.labels)["window"] == "7d"
        ]
        return found

    assert net(first, "central") == -7000  # negative gauge value is representable
    assert net(revised, "central") < net(first, "central")  # a gauge may decrease
    ok = lambda s: sum(x.value for x in s if x.family == "ssgrep_executions_total")  # noqa: E731
    assert ok(revised) >= ok(first)  # counters only grow within an epoch
    assert all(s.kind != "counter" or s.value >= 0 for s in revised)


def revised_window() -> dict:
    window = synthetic_current_window()
    window["estimates"]["context_tokens"] = synthetic_current_scenarios(
        (400, 1000, 20000), 16000, UnitSpec("context_tokens", "chars_div4_v1")
    )
    return window


def test_unknown_estimate_omits_the_sample_and_reports_coverage_not_zero() -> None:
    envelope = make_snapshot(1, window=window_with_unknown_optimistic())
    samples = metrics.snapshot_samples(envelope)
    nets = {
        dict(s.labels)["scenario"]: s.value
        for s in samples
        if s.family == "ssgrep_estimated_net_tokens"
    }
    assert nets == {"conservative": -12000, "central": -7000}  # optimistic absent, not 0
    coverage = {
        (dict(s.labels)["kind"], dict(s.labels)["status"]): s.value
        for s in samples
        if s.family == "ssgrep_coverage_records"
    }
    assert coverage[("estimates", "missing")] == 2  # optimistic credit and complete net unavailable
    assert coverage[("estimates", "available")] == 12
    known = {
        dict(s.labels)["scenario"]: s.value
        for s in samples
        if s.family == "ssgrep_known_net_tokens"
    }
    assert known == {"conservative": -12000, "central": -7000, "optimistic": 4000}
    upper = [
        s
        for s in samples
        if s.family == "ssgrep_estimate_bound_tokens"
        and dict(s.labels)["quantity"] == "net"
        and dict(s.labels)["bound"] == "upper"
    ]
    assert {dict(s.labels)["scenario"] for s in upper} == {"conservative", "central"}
    assert ("lookups", "recorded") in coverage and coverage[("lookups", "recorded")] == 112


def test_conflicting_duplicate_estimate_identity_is_omitted_not_arbitrarily_selected() -> None:
    from ssgrep.accounting.serve.metric_estimates import estimate_samples

    window = synthetic_current_window()
    original = window["estimates"]["context_tokens"]
    conflict = json.loads(json.dumps(original))
    conflict["central"]["known_net"] += 1
    window["estimates"]["duplicate_context_tokens"] = conflict

    samples, available, missing = estimate_samples(
        {"7d": window},
        lambda family, labels, value: Sample(family, "gauge", tuple(sorted(labels.items())), value),
    )

    methods = {
        dict(sample.labels).get("method")
        for sample in samples
        if sample.family == "ssgrep_estimated_net_work_seconds"
    }
    assert methods == {"capped_gap_v1"}
    assert available == 7 and missing == 7


def test_nonmapping_estimate_is_counted_missing_without_hiding_valid_estimate() -> None:
    from ssgrep.accounting.serve.metric_estimates import estimate_samples

    window = synthetic_current_window()
    samples, available, missing = estimate_samples(
        {
            "7d": {
                "estimates": {
                    "malformed": None,
                    "work_seconds": window["estimates"]["work_seconds"],
                }
            }
        },
        lambda family, labels, value: Sample(family, "gauge", tuple(sorted(labels.items())), value),
    )

    assert any(sample.family == "ssgrep_estimated_net_work_seconds" for sample in samples)
    assert not any(sample.family == "ssgrep_estimated_net_tokens" for sample in samples)
    assert available == 7 and missing == 7


def test_no_sample_has_an_explicit_timestamp_and_freshness_is_a_gauge() -> None:
    text = render_openmetrics(metrics.all_samples(make_snapshot(1), HEALTH)).decode()
    for line in text.splitlines():
        if not line.startswith("#"):
            assert len(line.rsplit(" ", 1)) == 2 and re.fullmatch(
                r"-?\d+(\.\d+)?", line.rsplit(" ", 1)[1]
            )
            assert len(line.split(" ")) == 2  # name{labels} value, nothing after the value
    families = parse(metrics.all_samples(make_snapshot(1), []))
    gauge = families["ssgrep_snapshot_collected_timestamp_seconds"]
    assert value_of(gauge, "ssgrep_snapshot_collected_timestamp_seconds") == COLLECTED_EPOCH


def test_text_ends_with_eof_and_declares_units() -> None:
    text = render_openmetrics(metrics.all_samples(make_snapshot(1), [])).decode()
    assert text.endswith("# EOF\n") and text.count("# EOF") == 1
    assert "# UNIT ssgrep_operation_duration_seconds seconds" in text
    assert "# UNIT ssgrep_rendered_bytes bytes" in text
    assert "# TYPE ssgrep_executions counter" in text and "ssgrep_executions_total{" in text


def test_before_first_snapshot_only_availability_and_exporter_health() -> None:
    samples = metrics.all_samples(None, HEALTH)
    families = parse(samples)
    assert set(families) == {
        "ssgrep_snapshot_available",
        "ssgrep_export_rejected_points",
        "ssgrep_export_recent_lost_records",
        "ssgrep_export_last_full_ack_timestamp_seconds",
        "ssgrep_export_pending_batches",
    }
    assert value_of(families["ssgrep_snapshot_available"], "ssgrep_snapshot_available") == 0
    assert not any(s.family.startswith(("ssgrep_executions", "ssgrep_estimated")) for s in samples)


def test_unknown_export_health_omits_samples_instead_of_zero() -> None:
    samples = metrics.export_samples([metrics.ExportHealth("acme", "logs")])
    assert samples == []
    one = metrics.export_samples([metrics.ExportHealth("acme", "logs", pending_batches=0)])
    assert [(s.family, s.value) for s in one] == [("ssgrep_export_pending_batches", 0)]


def test_info_series_is_single_and_carries_only_schema_and_methodology() -> None:
    samples = metrics.snapshot_samples(make_snapshot(1, methodology_id="m-test-v1"))
    (info,) = [s for s in samples if s.family == "ssgrep_accounting_info"]
    assert dict(info.labels) == {"schema_version": "1", "methodology_id": "m-test-v1"} or set(
        dict(info.labels)
    ) == {"schema_version", "methodology_id"}
    digest = make_snapshot(1).snapshot_id
    assert all(digest not in v for s in samples for _, v in s.labels)


@pytest.mark.parametrize(
    "label",
    [
        "employee",
        "task",
        "task_id",
        "path",
        "ref",
        "error",
        "date",
        "snapshot_id",
        "query",
        "fact",
        "revision",
        "receipt_id",
    ],
)
def test_private_labels_are_rejected_explicitly(label: str) -> None:
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.validate_dimensions("ssgrep_executions_total", ["operation", label])
    assert caught.value.condition == "forbidden_label"


def test_unbounded_but_unlisted_label_and_unknown_family_are_rejected() -> None:
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.validate_dimensions("ssgrep_executions_total", ["operation", "hostname"])
    assert caught.value.condition == "unbounded_label"
    with pytest.raises(metrics.MetricsValidationError):
        metrics.validate_dimensions("ssgrep_custom_total", [])


def test_label_values_must_stay_in_their_bounded_domain() -> None:
    bad = make_snapshot(1)
    rows = bad.observations["counters"]["executions"]  # ty: ignore[not-subscriptable]
    rows[0]["operation"] = "/home/user/secret.py"
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.snapshot_samples(bad)
    assert caught.value.condition == "invalid_label_value"


def test_more_than_5000_series_and_more_than_8_destinations_fail_validation() -> None:
    many = [
        Sample(
            "ssgrep_export_pending_batches",
            "gauge",
            (("destination", f"d{i}"), ("signal", "metrics")),
            1,
        )
        for i in range(5001)
    ]
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.validate_samples(many)
    assert caught.value.condition == "series_limit"
    nine = [metrics.ExportHealth(f"dest{i}", "metrics", pending_batches=0) for i in range(9)]
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.all_samples(make_snapshot(1), nine)
    assert caught.value.condition == "destination_limit"
    eight = [metrics.ExportHealth(f"dest{i}", "metrics", pending_batches=0) for i in range(8)]
    assert metrics.all_samples(make_snapshot(1), eight)


def test_duplicate_series_and_float_values_are_refused() -> None:
    sample = Sample("ssgrep_snapshot_available", "gauge", (), 1)
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.validate_samples([sample, sample])
    assert caught.value.condition == "duplicate_series"
    with pytest.raises(metrics.MetricsValidationError):
        metrics._sample("ssgrep_snapshot_available", {}, 1.5)  # ty: ignore[invalid-argument-type]


def test_malformed_histogram_is_refused_rather_than_exposed() -> None:
    bad = make_snapshot(1)
    hist = bad.observations["counters"]["duration"][0]  # ty: ignore[not-subscriptable]
    hist["buckets"][3]["count"] = 99  # non-monotonic cumulative buckets
    with pytest.raises(metrics.MetricsValidationError) as caught:
        metrics.snapshot_samples(bad)
    assert caught.value.condition == "invalid_histogram"


def test_snapshot_without_counters_exposes_no_execution_samples_not_zeros() -> None:
    samples = metrics.snapshot_samples(make_snapshot(1, with_counters=False))
    assert not any(
        s.family.startswith(("ssgrep_executions", "ssgrep_rendered", "ssgrep_operation"))
        for s in samples
    )
    assert any(s.family == "ssgrep_snapshot_available" and s.value == 1 for s in samples)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("# EOF\n", ""),
        lambda t: t.replace("ssgrep_executions_total", "ssgrep_executions"),
        lambda t: t.replace('le="+Inf"', 'le="99"'),
        lambda t: t + "ssgrep_extra 1\n",
    ],
    ids=["no_eof", "counter_without_total", "no_inf_bucket", "data_after_eof"],
)
def test_the_real_openmetrics_parser_rejects_each_malformed_variant(mutate) -> None:
    text = render_openmetrics(metrics.all_samples(make_snapshot(1), [])).decode()
    assert list(text_string_to_metric_families(text))  # the unmutated text parses
    with pytest.raises(ValueError):
        list(text_string_to_metric_families(mutate(text)))


def test_invalid_collection_timestamp_omits_only_timestamp_not_snapshot_metrics():
    envelope = replace(make_snapshot(1), collected_at="2026-02-30T00:00:00Z")
    samples = metrics.snapshot_samples(envelope)
    by_name = {sample.family: sample for sample in samples}
    assert "ssgrep_snapshot_collected_timestamp_seconds" not in by_name
    assert by_name["ssgrep_snapshot_available"].value == 1
    assert by_name["ssgrep_accounting_info"].value == 1
    valid = metrics.snapshot_samples(make_snapshot(1))
    timestamp = next(s for s in valid if s.family == "ssgrep_snapshot_collected_timestamp_seconds")
    assert timestamp.value == 1791806400


@pytest.mark.parametrize(
    "kind", ["counter", "histogram_bucket", "histogram_count", "histogram_sum"]
)
def test_renderer_rejects_wrong_sample_suffix_before_emitting_text(kind: str):
    with pytest.raises(ValidationError) as caught:
        render_openmetrics([Sample("ssgrep_invalid", kind, (), 1)])
    assert caught.value.condition == "sample_name"


PROFILE = "personal-local"


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _serve(run: Run, port: int, *, json_mode: bool = False) -> subprocess.Popen[str]:
    arguments = [str(BINARY), "savings", "serve", "--port", str(port)]
    if json_mode:
        arguments.insert(1, "--json")
    return subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "HOME": str(run.home),
            "SSGREP_DATA_DIR": str(run.data),
            "NO_COLOR": "1",
            "COLUMNS": "80",
        },
    )


def _read_json_startup(process: subprocess.Popen[str], timeout: float = 30) -> tuple[str, dict]:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            "metrics serve did not emit its startup document before blocking; "
            f"exit={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
        )
    line = process.stdout.readline()
    if not line:
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            f"metrics serve exited before startup (status {process.returncode}); "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )
    return line, json.loads(line)


def _get(port: int, path: str, token: str | None = None) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


def _assert_stopped(port: int) -> None:
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.5)


def _terminate(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    return process.communicate(timeout=60)


def test_json_startup_precedes_blocking_worker_and_rebinds_after_sigterm(run: Run) -> None:
    assert run("savings", "setup", "--yes").returncode == 0
    port = _free_port()
    process = _serve(run, port, json_mode=True)
    try:
        startup_line, startup = _read_json_startup(process)
        token_file = run.data / "accounting" / PROFILE / "endpoint.token"
        token = token_file.read_text().strip()

        assert startup["url"] == f"http://127.0.0.1:{port}"
        assert startup["port"] == port
        assert startup["profile"] == PROFILE
        assert startup["token_file"] == str(token_file)
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert token and token not in startup_line
        assert _get(port, "/healthz")[0] == 200
        assert _get(port, "/metrics")[0] == 401
        status, content_type, body = _get(port, "/metrics", token)
        assert status == 200
        assert content_type == "application/openmetrics-text; version=1.0.0; charset=utf-8"
        assert body.endswith(b"# EOF\n")
        assert b"# TYPE ssgrep_snapshot_available gauge\n" in body

        stdout, stderr = _terminate(process)
        assert process.returncode == 0
        assert stdout == ""  # JSON is exactly one startup document, not a shutdown stream.
        assert stderr == ""
        assert token not in startup_line + stdout + stderr
        _assert_stopped(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)

    # A second real serve proves both the listener and the profile worker lease were released.
    second = _serve(run, port, json_mode=True)
    try:
        second_line, second_startup = _read_json_startup(second)
        assert second_startup.get("port") == port, second_startup
        assert _get(port, "/healthz")[0] == 200
        stdout, stderr = _terminate(second)
        assert second.returncode == 0 and stdout == "" and stderr == ""
        assert token not in second_line + stdout + stderr
        _assert_stopped(port)
    finally:
        if second.poll() is None:
            second.kill()
            second.communicate(timeout=10)


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_occupied_port_has_a_stable_error_and_no_alternate_listener(
    run: Run, json_mode: bool
) -> None:
    assert run("savings", "setup", "--yes").returncode == 0
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        args = ("savings", "serve", "--port", str(port))
        result = run(*(("--json",) if json_mode else ()), *args)

    assert result.returncode == 1
    message = f"Port {port} is already in use; no alternate port was selected."
    if json_mode:
        error = document(result)
        assert error["ok"] is False and error["condition"] == "port_in_use"
        assert error["message"] == message
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        assert message in result.stderr and "Try:" in result.stderr
    token_file = run.data / "accounting" / PROFILE / "endpoint.token"
    token = token_file.read_text().strip()
    assert token and token not in result.stdout + result.stderr


def test_cli_does_not_offer_a_remote_bind_option(run: Run) -> None:
    help_text = run("savings", "serve", "--help").stdout
    assert "--port" in help_text
    assert "9469" in help_text
    assert "--bind" not in help_text

    result = run("--json", "savings", "serve", "--bind", "0.0.0.0")
    assert result.returncode == 2
    assert document(result)["ok"] is False
    assert run.files() == []


def test_unsafe_endpoint_token_file_fails_without_traceback_or_secret_output(run: Run) -> None:
    token = "a" * 64
    token_file = run.data / "accounting" / PROFILE / "endpoint.token"
    token_file.parent.mkdir(parents=True, mode=0o700)
    token_file.write_text(token + "\n")
    token_file.chmod(0o644)

    result = run("--json", "savings", "serve", "--port", str(_free_port()))

    assert result.returncode == 1
    error = document(result)
    assert error["condition"] == "token_file_unsafe"
    assert "Traceback" not in result.stderr
    assert token not in result.stdout + result.stderr
