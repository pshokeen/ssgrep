"""Native/HTTP failure gates exercised with local CLI scripts, never paid turns."""

import json
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.accounting_airline import live_bridge, live_native, runtime_omp
from eval.accounting_airline.gates import FAIL
from eval.accounting_airline.report import Report
from tests.accounting.airline.test_lab_runtime_omp import _env, _spec, make_double


@pytest.mark.parametrize("stream", [1, 2])
def test_native_output_failure_keeps_independent_usage_and_partial_artifact(
    tmp_path, monkeypatch, stream
):
    double = make_double(tmp_path / "bin")
    code = double.read_text().replace(
        'sys.exit(3 if mode == "fail" else 0)',
        f"os.write({stream}, b'x' * 8192); time.sleep(60)",
    )
    double.write_text(code)
    monkeypatch.setattr(runtime_omp, "NATIVE_STDOUT_LIMIT", 4096)
    monkeypatch.setattr(runtime_omp, "NATIVE_STDERR_LIMIT", 2048)
    counter = tmp_path / "counter"
    output = tmp_path / "artifacts"
    result = runtime_omp.run_turn(
        _spec(tmp_path),
        env=_env(FAKE_OMP_COUNTER=str(counter), FAKE_OMP_COST="0.0123"),
        profile=None,
        timeout=5,
        omp=str(double),
        authorized=True,
        output_dir=output,
    )
    assert result.output_limit and result.capture_status == "output_limit"
    assert not result.timed_out and result.returncode != 0
    assert result.native.usage_records == 1 and result.native.cost_nano_usd == 12_300_000
    assert counter.read_text() == "invoked\n"
    assert result.stdout_chars <= 4096 and result.stderr_chars <= 2048
    assert result.to_dict()["output_limit_streams"] == ["stdout" if stream == 1 else "stderr"]
    for name, cap in (("stdout", 4096), ("stderr", 2048)):
        artifact = output / (name + ".txt")
        assert artifact.stat().st_size <= cap and artifact.stat().st_mode & 0o777 == 0o600
    assert result.pid is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)


def test_normal_native_cli_stream_fields_are_unchanged(tmp_path):
    double = make_double(tmp_path / "bin")
    line = '{"type":"done","text":"é"}\n'.encode()
    double.write_text(
        double.read_text().replace(
            'sys.exit(3 if mode == "fail" else 0)',
            f"os.write(1, {line!r}); os.write(2,b'warning\\r\\n'); sys.exit(0)",
        )
    )
    result = runtime_omp.run_turn(
        _spec(tmp_path),
        env=_env(),
        profile=None,
        timeout=5,
        omp=str(double),
        authorized=True,
    )
    assert result.returncode == 0 and not result.output_limit and not result.timed_out
    assert result.stdout_chars == len('{"type":"done","text":"é"}\n')
    assert result.stdout_bytes == len('{"type":"done","text":"é"}\n'.encode())
    assert result.stderr_chars == result.stderr_bytes == 9
    assert result.native.usage_records == 1


def post(bridge, tool="ssgrep", args=None, stdin=""):
    body = json.dumps({"tool": tool, "args": args or ["search", "rules", "--json"], "stdin": stdin})
    request = urllib.request.Request(
        f"http://127.0.0.1:{bridge.server.server_port}/tool",
        body.encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = response.read(live_bridge.MAX_RESPONSE_BYTES + 1)
    assert len(payload) <= live_bridge.MAX_RESPONSE_BYTES
    return json.loads(payload)


def kit(tmp_path):
    return SimpleNamespace(
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        spool=tmp_path / "verifier" / "spool.ndjson",
    )


@pytest.mark.parametrize("fd", [1, 2])
def test_bridge_output_limit_is_explicit_and_artifacts_private(tmp_path, monkeypatch, fd):
    monkeypatch.setattr(live_bridge, "BRIDGE_STDOUT_LIMIT", 128)
    monkeypatch.setattr(live_bridge, "BRIDGE_STDERR_LIMIT", 64)
    monkeypatch.setattr(
        live_bridge, "TRUSTED_SHIM", f"import os,time; os.write({fd}, b'x'*8192); time.sleep(60)"
    )
    with live_bridge.Bridge(kit(tmp_path), "assigned", tmp_path) as bridge:
        data = post(bridge)
        attempts = list(bridge.attempts)
    assert data["returncode"] != 0 and data["stdout"] == ""
    assert data["capture_status"] == "output_limit" and data["output_limit"] is True
    assert "output_limit" in data["stderr"]
    assert len(attempts) == 1 and attempts[0]["output_limit"]
    path = Path(attempts[0]["artifact_dir"])
    assert (path / "stdout.txt").stat().st_size <= 128
    assert (path / "stderr.txt").stat().st_size <= 64
    assert (path / "result.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ChildProcessError):
        os.waitpid(attempts[0]["pid"], os.WNOHANG)


def test_bridge_http_encoded_response_has_independent_hard_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(live_bridge, "MAX_RESPONSE_BYTES", 1024)
    monkeypatch.setattr(live_bridge, "TRUSTED_SHIM", "import os; os.write(1,b'\\x01'*300)")
    with live_bridge.Bridge(kit(tmp_path), "assigned", tmp_path) as bridge:
        data = post(bridge)
        attempts = list(bridge.attempts)
    assert data["capture_status"] == "output_limit" and data["output_limit"]
    assert data["stdout"] == "" and data["returncode"] == 0  # actual child exit, never fabricated
    assert attempts[0]["output_limit_streams"] == ["http_response"]


def test_bridge_normal_json_and_installed_client_preserve_public_fields(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(
        live_bridge,
        "TRUSTED_SHIM",
        "import sys; print('{\"ok\":true}'); sys.stderr.write('warning\\n'); raise SystemExit(3)",
    )
    with live_bridge.Bridge(kit(tmp_path), "assigned", tmp_path) as bridge:
        data = post(bridge)
        bridge.install_clients(tmp_path / "clients")
        done = subprocess.run(
            [str(tmp_path / "clients" / "ssgrep"), "search", "rules", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert (data["returncode"], data["stdout"], data["stderr"]) == (3, '{"ok":true}\n', "warning\n")
    assert (done.returncode, done.stdout, done.stderr) == (3, '{"ok":true}\n', "warning\n")


def test_output_limit_creates_failure_gate_even_with_successful_domain_exit():
    run = SimpleNamespace(report=Report("bounded-test", "live", "omp", True, "omp-live"))
    result = SimpleNamespace(capture_status="output_limit", output_limit_streams=("stdout",))
    detail = live_native.capture_failure(run, "L1", result, [])
    assert detail is not None and "output_limit" in detail
    assert run.report.gate_status("live_session_gate") == FAIL


def test_bridge_limit_creates_failure_gate_even_when_native_stream_did_not_overflow():
    run = SimpleNamespace(report=Report("bounded-test", "live", "omp", True, "omp-live"))
    result = SimpleNamespace(capture_status="completed", output_limit_streams=())
    detail = live_native.capture_failure(run, "L3", result, [{"capture_status": "output_limit"}])
    assert detail is not None and "output_limit" in detail
    assert run.report.gate_status("live_session_gate") == FAIL


def test_installed_client_rejects_limit_even_when_child_exit_is_zero(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(live_bridge, "MAX_RESPONSE_BYTES", 1024)
    monkeypatch.setattr(live_bridge, "TRUSTED_SHIM", "import os; os.write(1,b'\\x01'*300)")
    with live_bridge.Bridge(kit(tmp_path), "assigned", tmp_path) as bridge:
        bridge.install_clients(tmp_path / "clients")
        done = subprocess.run(
            [str(tmp_path / "clients" / "ssgrep"), "search", "rules", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert done.returncode != 0 and not done.stdout and "output_limit" in done.stderr


def test_installed_sql_client_bounds_input_before_launch(tmp_path, monkeypatch):
    import subprocess

    marker = tmp_path / "started"
    monkeypatch.setattr(live_bridge, "TRUSTED_SHIM", f"open({str(marker)!r},'w').write('started')")
    with live_bridge.Bridge(kit(tmp_path), "assigned", tmp_path) as bridge:
        bridge.install_clients(tmp_path / "clients")
        done = subprocess.run(
            [str(tmp_path / "clients" / "sql"), "--role", "reader", "--database", "assigned"],
            input="x" * (live_bridge.MAX_BODY + 1),
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert done.returncode != 0 and "request boundary" in done.stderr and not marker.exists()


def test_bridge_exit_cancels_and_reaps_only_its_inflight_tool(tmp_path, monkeypatch):
    import threading
    import time

    marker = tmp_path / "started"
    monkeypatch.setattr(
        live_bridge,
        "TRUSTED_SHIM",
        f"import time; open({str(marker)!r},'w').write('started'); time.sleep(60)",
    )
    responses = []
    with live_bridge.Bridge(kit(tmp_path), "assigned", tmp_path) as bridge:
        thread = threading.Thread(target=lambda: responses.append(post(bridge)))
        thread.start()
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
    thread.join(timeout=5)
    assert not thread.is_alive() and responses[0]["capture_status"] == "cancelled"
    attempt = bridge.attempts[0]
    with pytest.raises(ChildProcessError):
        os.waitpid(attempt["pid"], os.WNOHANG)
