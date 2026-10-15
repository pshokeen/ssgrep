"""First failure cause, pipe EOF, shim text failures, and failed-attempt billing."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from eval.accounting_airline import bounded_capture, live_metrics, live_run, observer, runtime_omp
from eval.accounting_airline.domain_lane import write_private
from eval.accounting_airline.report import Report
from tests.accounting.airline.test_bridge_nested_capture import fixture_kit
from tests.accounting.airline.test_lab_runtime_omp import _env, _spec, make_double


@pytest.mark.parametrize("cause", ["timeout", "cancelled"])
def test_post_kill_drain_keeps_first_cause_and_actual_byte_counts(tmp_path, monkeypatch, cause):
    ready = tmp_path / "ready"
    code = (
        f"import os,time; os.write(1,b'x'*1025); open({str(ready)!r},'w').close(); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    selector_type = bounded_capture.selectors.DefaultSelector
    release_reads_at = time.monotonic() + 0.3

    class DelayedReads(selector_type):
        def select(self, timeout=None):
            if time.monotonic() < release_reads_at:
                time.sleep(min(timeout or 0.01, 0.02))
                return []
            return super().select(timeout)

    monkeypatch.setattr(bounded_capture.selectors, "DefaultSelector", DelayedReads)
    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set) if cause == "cancelled" else None
    if timer:
        timer.start()
    try:
        result = bounded_capture.capture_process(
            proc.args,
            env={},
            cwd=tmp_path,
            timeout=0.1 if cause == "timeout" else 5,
            stdout_limit=1024,
            stderr_limit=128,
            process=proc,
            cancel_event=stop,
        )
    finally:
        if timer:
            timer.cancel()
            timer.join()
    assert result.status == result.termination_cause == cause
    assert result.stdout_bytes == 1025 and len(result.stdout) == 1024
    assert result.stdout_truncated and result.observed_over_limit_streams == ("stdout",)
    assert not result.output_limit and result.output_limit_streams == ()
    assert not result.capture_complete and result.pipes_drained
    assert result.metadata()["observed_over_limit_streams"] == ["stdout"]
    with pytest.raises(ChildProcessError):
        os.waitpid(proc.pid, os.WNOHANG)


def shim(tmp_path, body, *, timeout=0.5, cancel=False):
    kit = fixture_kit(tmp_path, body)
    repo = Path(__file__).resolve().parents[3]
    code = (
        f"import sys; sys.path.insert(0,{str(repo)!r}); import threading; "
        "from eval.accounting_airline.observer import shim_main; "
        "stop=threading.Event(); "
        + ("timer=threading.Timer(0.5,stop.set); timer.start(); " if cancel else "")
        + (
            "raise SystemExit(shim_main('ssgrep',['search','rules','--json'],"
            f"timeout={timeout!r},cancel_event=stop))"
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        env=kit.env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=15)
        done = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
    return kit, done, observer.read_trace(kit.spool)


@pytest.mark.parametrize("cancel", [False, True])
def test_shim_stopped_invalid_utf8_tail_is_interrupted_masked_and_recorded(tmp_path, cancel):
    raw = b"partial\xe2\x82"
    kit, done, events = shim(
        tmp_path,
        f"import os,time\nos.write(1,{raw!r})\ntime.sleep(60)\n",
        timeout=5 if cancel else 0.5,
        cancel=cancel,
    )
    dispatch, result = [e for e in events if e["kind"] in ("dispatch", "result")]
    cause = "cancelled" if cancel else "timeout"
    assert done.returncode != 0 and done.stdout == b"" and cause.encode() in done.stderr
    assert result["state"] == "interrupted" and result["reason"] == cause
    assert result["capture"]["capture_status"] == cause
    assert result["capture"]["stdout_bytes"] == len(raw)
    assert result["capture"]["capture_complete"] is False
    assert result["stdout_chars"] == 0 and result["blob"] is None
    prefix = kit.spool.parent / "shim-attempts" / dispatch["call_id"] / "tool" / "stdout.txt"
    assert prefix.read_bytes() == raw and prefix.stat().st_mode & 0o777 == 0o600


def test_shim_timeout_after_zero_exit_with_inherited_pipe_is_nonpass(tmp_path):
    body = (
        "import subprocess,sys\n"
        "subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(60)'])\n"
        "print('{\"looks_valid\":true}')\n"
    )
    _kit, done, events = shim(tmp_path, body)
    result = [e for e in events if e["kind"] == "result"][0]
    assert done.returncode != 0 and done.stdout == b""
    assert result["state"] == "interrupted" and result["reason"] == "timeout"
    assert result["capture"]["returncode"] == 0  # Preserve the actual tool leader exit.
    assert not result["capture"]["capture_complete"]


def test_shim_completed_invalid_utf8_is_recorded_before_text_failure(tmp_path):
    _kit, done, events = shim(tmp_path, "import os\nos.write(1,b'\\xff')\n")
    result = [e for e in events if e["kind"] == "result"][0]
    assert done.returncode != 0 and done.stdout == b""
    assert result["state"] == "interrupted" and result["reason"] == "invalid_output_encoding"
    assert result["capture"]["stdout_bytes"] == 1 and result["capture"]["returncode"] == 0
    assert result["capture"]["capture_complete"] is False


def test_verifier_completed_invalid_utf8_is_recorded_not_a_trace_gap(tmp_path):
    recorder = observer.Observer(tmp_path, "encoding-test")
    with pytest.raises(observer.ObserverError, match="invalid_output_encoding"):
        recorder.run(
            actor="control",
            task="encoding",
            tool="python",
            argv=[sys.executable, "-I", "-c", "import os; os.write(1,b'\\xff')"],
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
            cwd=tmp_path,
            timeout=5,
        )
    dispatch, result = observer.read_trace(recorder.trace.path)
    assert dispatch["call_id"] == result["call_id"]
    assert result["state"] == "interrupted" and result["reason"] == "invalid_output_encoding"
    assert result["capture"]["stdout_bytes"] == 1


def test_async_deadline_is_explicitly_owned_and_not_an_implicit_180_seconds(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    capture = bounded_capture.AsyncCapture(proc, timeout=0.2)
    result = capture.finish(timeout=5)
    assert result.returncode is not None
    assert result.status == "timeout" and result.returncode < 0
    assert not result.capture_complete and result.pipes_drained


def test_async_owner_can_defer_launch_deadline_until_explicit_finish(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(0.2); print('normal')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    result = bounded_capture.AsyncCapture(proc, timeout=None).finish(timeout=181)
    assert result.status == "completed" and result.stdout == b"normal\n"
    assert result.capture_complete and result.pipes_drained


def test_live_acceptance_uses_capture_status_for_native_and_bridge_attempts():
    turns = {
        task: {
            "returncode": 0,
            "timed_out": False,
            "output_limit": False,
            "capture_status": "completed",
            "bridge_attempts": [],
        }
        for task in ("L1", "L2", "L3", "L4")
    }
    assert live_run.native_turns_completed(turns)
    turns["L4"]["capture_status"] = "cancelled"
    assert not live_run.native_turns_completed(turns)
    turns["L4"]["capture_status"] = "completed"
    turns["L4"]["bridge_attempts"] = [
        {"returncode": 0, "output_limit": False, "capture_status": "timeout"}
    ]
    assert not live_run.native_turns_completed(turns)


@pytest.mark.parametrize("failure", [False, True])
def test_failed_native_usage_is_persisted_and_reported_with_capture_status(
    tmp_path, monkeypatch, failure
):
    double = make_double(tmp_path / "bin")
    if failure:
        double.write_text(
            double.read_text().replace(
                'sys.exit(3 if mode == "fail" else 0)',
                "os.write(1,b'partial'); time.sleep(60)",
            )
        )
    output = tmp_path / "artifacts"
    result = runtime_omp.run_turn(
        _spec(tmp_path),
        env=_env(FAKE_OMP_COST="0.0123"),
        profile=None,
        timeout=2 if failure else 5,
        omp=str(double),
        authorized=True,
        output_dir=output,
    )
    write_private(output / "result.json", result.to_dict())
    payload = json.loads((output / "result.json").read_text())
    status = "timeout" if failure else "completed"
    assert payload["capture_status"] == status and payload["capture_complete"] is not failure
    assert payload["pipes_drained"] and payload["native"]["usage_records"] == 1
    from eval.accounting_airline.database import BaselineInfo
    from eval.accounting_airline.lab_context import Run
    from eval.accounting_airline.roots import LabRoots

    class BillingRoots(LabRoots):
        @property
        def ssgrep_data(self) -> Path:
            return self.root

    run = object.__new__(Run)
    run.base = BaselineInfo("billing-fixture", {}, {}, hashlib.sha256(b"{}").hexdigest(), {})
    run.notes = {"turns": {"L1": payload}}
    run.report = Report("billing-test", "live", "omp", True, "omp-live")
    run.roots = BillingRoots(tmp_path / "isolated-data", "billing-fixture")
    monkeypatch.setattr(run, "frozen", lambda: observer.Frozen("billing-fixture", (), {}, (), ""))
    live_metrics.fill(run)
    rows = [
        m for m in run.report.metrics.as_dict()["tokens_money"] if m["actor_scope"] == "solver-a"
    ]
    cost = next(m for m in rows if m["name"] == "runtime_reported_cost")
    tokens = next(m for m in rows if m["name"] == "provider_tokens.input")
    assert cost["value"] == 12_300_000 and tokens["value"] == 100
    assert cost["basis"] == tokens["basis"] == "reported"
    assert cost["reason"] == tokens["reason"] == ("capture_status=timeout" if failure else None)


@pytest.mark.parametrize("cause", ["timeout", "invalid_output_encoding"])
def test_bridge_propagates_inner_shim_failure_instead_of_completed_status(
    tmp_path, monkeypatch, cause
):
    from eval.accounting_airline import live_bridge
    from tests.accounting.airline.test_native_capture_limits import post

    body = "import os,time\nos.write(1,b'\\xff')\n" + (
        "time.sleep(60)\n" if cause == "timeout" else ""
    )
    kit = fixture_kit(tmp_path, body)
    monkeypatch.setattr(
        live_bridge,
        "TRUSTED_SHIM",
        live_bridge.TRUSTED_SHIM.replace("group_owner=True", "group_owner=True, timeout=0.5"),
    )
    with live_bridge.Bridge(kit, "assigned", tmp_path) as bridge:
        data = post(bridge)
        attempt = bridge.attempts[0]
    assert data["capture_status"] == cause and data["returncode"] != 0 and data["stdout"] == ""
    assert attempt["capture_status"] == attempt["tool_capture"]["capture_status"] == cause
    assert attempt["capture_complete"] is False
    assert attempt["tool_capture"]["stdout_bytes"] == 1


def test_output_limit_cause_survives_a_later_owner_cancel(tmp_path):
    stop = threading.Event()
    snapshot = []

    def stop_later(captured):
        snapshot.append(captured)
        stop.set()

    result = bounded_capture.capture_process(
        [sys.executable, "-I", "-c", "import os,time; os.write(1,b'x'*8192); time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        cwd=tmp_path,
        timeout=5,
        stdout_limit=1024,
        stderr_limit=128,
        cancel_event=stop,
        on_output_limit=stop_later,
    )
    assert result.status == result.termination_cause == "output_limit"
    assert result.output_limit and not result.timed_out and not result.cancelled
    assert result.stdout_bytes > 1024 and result.stdout_truncated
    assert result.pipes_drained and not result.capture_complete
    assert snapshot[0].output_limit and not snapshot[0].pipes_drained
