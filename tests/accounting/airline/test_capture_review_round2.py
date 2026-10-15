"""Real signal, UTF-8-mode parity, and retained named capture-failure gates."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.accounting_airline import live_bridge, live_native
from eval.accounting_airline.gates import FAIL
from eval.accounting_airline.report import Report
from tests.accounting.airline.test_bridge_nested_capture import fixture_kit
from tests.accounting.airline.test_capture_failure_semantics import shim
from tests.accounting.airline.test_native_capture_limits import post


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGKILL"])
def test_signal_killed_tool_is_typed_interruption_and_shim_masks_partial_answer(
    tmp_path, signal_name
):
    partial = b'{"looks_valid":true}\n'
    body = (
        "import os,signal\n"
        + f"os.write(1,{partial!r})\n"
        + f"os.kill(os.getpid(),signal.{signal_name})\n"
    )
    _kit, done, events = shim(tmp_path, body, timeout=5)
    result = next(e for e in events if e["kind"] == "result")
    assert done.returncode != 0 and done.stdout == b""
    assert result["state"] == "interrupted" and result["reason"] == "interrupted"
    assert result["capture"]["capture_status"] == "interrupted"
    assert result["capture"]["termination_cause"] == "interrupted"
    assert result["capture"]["returncode"] == -getattr(signal, signal_name)
    assert result["capture"]["stdout_bytes"] == len(partial)
    assert not result["capture"]["capture_complete"] and result["capture"]["pipes_drained"]
    assert result["stdout_chars"] == 0 and result["blob"] is None


def test_signal_failure_crosses_bridge_as_interruption_with_named_gate(tmp_path):
    kit = fixture_kit(
        tmp_path,
        "import os,signal\nos.write(1,b'partial-answer')\nos.kill(os.getpid(),signal.SIGTERM)\n",
    )
    with live_bridge.Bridge(kit, "assigned", tmp_path) as bridge:
        data = post(bridge)
        attempts = list(bridge.attempts)
    assert (
        data["capture_status"] == "interrupted" and not data["stdout"] and data["returncode"] != 0
    )
    assert attempts[0]["tool_capture"]["returncode"] == -signal.SIGTERM
    assert_named_failure(tmp_path, attempts, "interrupted")


def assert_named_failure(tmp_path, attempts, cause):
    run = SimpleNamespace(report=Report("named-capture", "live", "omp", True, "omp-live"))
    native = SimpleNamespace(capture_status="completed", output_limit_streams=())
    detail = live_native.capture_failure(run, "L3", native, attempts)
    assert (
        detail is not None
        and cause in detail
        and run.report.gate_status("live_session_gate") == FAIL
    )
    path = tmp_path / "named-report.json"
    run.report.write_private(path)
    entries = json.loads(path.read_text())["gates"]["live_session_gate"]["entries"]
    entry = next(e for e in entries if e["name"] == f"native.L3.bridge.{cause}")
    assert entry["status"] == FAIL and f"bridge capture_status={cause}" in entry["detail"]


@pytest.mark.parametrize(
    "cause", ["timeout", "cancelled", "invalid_output_encoding", "capture_error"]
)
def test_every_failed_bridge_capture_retains_named_acceptance_cause(tmp_path, monkeypatch, cause):
    started = tmp_path / "started"
    body = (
        f"from pathlib import Path\nimport time\nPath({str(started)!r}).write_text('ready')\n"
        "print('partial-answer',flush=True)\ntime.sleep(60)\n"
    )
    if cause == "invalid_output_encoding":
        body = "import os\nos.write(1,b'\\xff')\n"
    kit = fixture_kit(tmp_path, body)
    if cause == "timeout":
        monkeypatch.setattr(
            live_bridge,
            "TRUSTED_SHIM",
            live_bridge.TRUSTED_SHIM.replace("group_owner=True", "group_owner=True, timeout=0.5"),
        )
    if cause == "capture_error":
        monkeypatch.setattr(
            live_bridge,
            "TRUSTED_SHIM",
            "import os,time; from pathlib import Path; "
            "p=Path(os.environ['LAB_CAPTURE_DIR'])/'tool-capture-failed'; "
            "pending=p.with_suffix('.pending'); pending.write_bytes(b'unknown-status'); "
            "pending.replace(p); time.sleep(60)",
        )
    stopper = None
    with live_bridge.Bridge(kit, "assigned", tmp_path) as bridge:
        if cause == "cancelled":

            def cancel_owner():
                deadline = time.monotonic() + 5
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                bridge.stop.set()

            stopper = threading.Thread(target=cancel_owner)
            stopper.start()
        try:
            data = post(bridge)
        finally:
            if stopper:
                stopper.join(timeout=6)
        attempts = list(bridge.attempts)
    assert data["capture_status"] == cause and not data["stdout"] and data["returncode"] != 0
    assert_named_failure(tmp_path, attempts, cause)


def test_normal_exit_one_diagnostic_output_remains_completed(tmp_path):
    kit, done, events = shim(
        tmp_path,
        "import sys\nprint('query diagnostic')\nsys.stderr.write('warn\\n')\nraise SystemExit(1)\n",
    )
    result = next(e for e in events if e["kind"] == "result")
    assert (
        done.returncode == 1 and done.stdout == b"query diagnostic\n" and done.stderr == b"warn\n"
    )
    assert result["state"] == "verified" and result["capture"]["capture_status"] == "completed"
    assert result["capture"]["capture_complete"] and result["capture"]["pipes_drained"]
    run = SimpleNamespace(report=Report("normal-capture", "live", "omp", True, "omp-live"))
    native = SimpleNamespace(capture_status="completed", output_limit_streams=())
    assert live_native.capture_failure(run, "L1", native, [result["capture"]]) is None


def test_utf8_mode_non_utf8_locale_matches_stdlib_across_text_boundaries(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    code = (
        r"""
import contextlib, io, json, locale, os, subprocess, sys
from pathlib import Path
from eval.accounting_airline import bounded_capture, faults, live_bridge, observer
from tests.accounting.airline.test_bridge_nested_capture import fixture_kit
from tests.accounting.airline.test_native_capture_limits import post
root = Path(sys.argv[1])
before = locale.setlocale(locale.LC_CTYPE)
text = "\u00e9\r\n"
cmd = [sys.executable, "-I", "-c", """
        r""""import sys,os; sys.stdout.buffer.write(sys.stdin.buffer.read()); """
        r"""os.write(2,b'\\xc3\\xa9\\r\\n')"]
expected = subprocess.run(cmd, input=text, text=True, capture_output=True, check=False)
bounded = bounded_capture.run_bounded(cmd, input=text, text=True, cwd=root, timeout=5)
recorder = observer.Observer(root / "control", "utf8-mode")
call, controlled = recorder.run(actor="control", task="codec", tool="python", argv=cmd,
                                env=dict(os.environ), cwd=root, timeout=5, stdin=text)
raw = bounded_capture.capture_process(cmd, env=dict(os.environ), cwd=root, timeout=5,
                                     stdout_limit=128, stderr_limit=128, """
        r"""input_bytes=text.encode("utf-8"))
fault_text = faults._captured_text(raw)
kit = fixture_kit(root, """
        r""""import os\nos.write(1,b'\\xc3\\xa9\\r\\n')\nos.write(2,b'\\xc3\\xa9\\r\\n')\n")
kit.env.update({k: os.environ[k] for k in ("LC_ALL", "LANG", "PYTHONUTF8", "PYTHONCOERCECLOCALE")})
shown, errors = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(shown), contextlib.redirect_stderr(errors):
    shim_exit = observer.shim_main("ssgrep", ["search", "rules", "--json"], env=kit.env, timeout=5)
with live_bridge.Bridge(kit, "assigned", root) as bridge:
    response = post(bridge)
print(json.dumps({
    "utf8_mode": sys.flags.utf8_mode, "locale_encoding": locale.getencoding(),
    "locale_unchanged": before == locale.setlocale(locale.LC_CTYPE),
    "expected": [expected.returncode, expected.stdout, expected.stderr],
    "bounded": [bounded.returncode, bounded.stdout, bounded.stderr],
    "control": [controlled.returncode, controlled.stdout, controlled.stderr],
    "faults": list(fault_text), "shim": [shim_exit, shown.getvalue(), errors.getvalue()],
    "bridge": [response["returncode"], response["stdout"], response["stderr"]],
}))
"""
    )
    code = f"import sys; sys.path.insert(0,{str(repo)!r});\n" + code
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "1",
        "PYTHONCOERCECLOCALE": "0",
    }
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", code, str(tmp_path)],
        env=env,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate(timeout=5)
    assert proc.returncode == 0, stderr.decode("utf-8", errors="replace")
    actual = json.loads(stdout)
    assert actual["utf8_mode"] == 1 and actual["locale_encoding"].lower().replace("-", "") != "utf8"
    assert actual["locale_unchanged"]
    assert actual["expected"] == [0, "\u00e9\n", "\u00e9\n"]
    assert (
        actual["bounded"]
        == actual["control"]
        == actual["shim"]
        == actual["bridge"]
        == actual["expected"]
    )
    assert actual["faults"] == actual["expected"][1:]
