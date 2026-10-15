"""Real concurrently running pipes, held stdin and fault-client capture failures."""

import os
import subprocess
import sys

import pytest

from eval.accounting_airline import bounded_capture, faults


def process(tmp_path, code):
    return subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


def test_async_drain_does_not_close_held_stdin_or_deadlock_writer(tmp_path):
    proc = process(
        tmp_path,
        "import os,sys; os.write(1,b'x'*32768); "
        "data=sys.stdin.read(); os.write(2,str(len(data)).encode())",
    )
    capture = bounded_capture.AsyncCapture(proc, timeout=None, stdout_limit=65536, stderr_limit=128)
    proc.stdin.write("input" * 20_000)
    proc.stdin.close()
    proc.stdin = None
    result = capture.finish(timeout=5)
    assert result.returncode == 0 and result.stdout == b"x" * 32768 and result.stderr == b"100000"
    with pytest.raises(ChildProcessError):
        os.waitpid(proc.pid, os.WNOHANG)


@pytest.mark.parametrize("fd", [1, 2])
def test_async_flood_is_killed_before_owner_requests_finish(tmp_path, fd):
    proc = process(tmp_path, f"import os,time; os.write({fd},b'x'*65536); time.sleep(60)")
    capture = bounded_capture.AsyncCapture(proc, timeout=None, stdout_limit=4096, stderr_limit=2048)
    with pytest.raises(bounded_capture.OutputLimitExceeded) as caught:
        capture.finish(timeout=5)
    assert caught.value.capture.output_limit and not caught.value.capture.timed_out
    assert len(caught.value.capture.stdout) <= 4096 and len(caught.value.capture.stderr) <= 2048
    with pytest.raises(ChildProcessError):
        os.waitpid(proc.pid, os.WNOHANG)


def test_async_finish_timeout_kills_reaps_and_keeps_actual_exit(tmp_path):
    proc = process(tmp_path, "import os,time; os.write(1,b'partial'); time.sleep(60)")
    capture = bounded_capture.AsyncCapture(proc, timeout=None)
    result = capture.finish(timeout=0.5)
    assert result.returncode is not None
    assert result.timed_out and result.status == "timeout" and result.returncode < 0
    assert result.stdout == b"partial"
    with pytest.raises(ChildProcessError):
        os.waitpid(proc.pid, os.WNOHANG)


class LocalProcessLab:
    def __init__(self, tmp_path, code):
        self.tmp_path, self.code = tmp_path, code

    def popen(self, _database, **_options):
        return process(self.tmp_path, self.code)


def test_fault_client_overflow_is_explicit_not_domain_response(tmp_path, monkeypatch):
    monkeypatch.setattr(bounded_capture, "DEFAULT_STDOUT_LIMIT", 1024)
    lab = LocalProcessLab(
        tmp_path, "import os,sys,time; sys.stdin.read(); os.write(1,b'x'*8192); time.sleep(60)"
    )
    client = faults._Client(lab, "lab", "local-process-double", "SELECT 1;")
    with pytest.raises(bounded_capture.OutputLimitExceeded):
        client.finish(timeout=5)
    with pytest.raises(ChildProcessError):
        os.waitpid(client.proc.pid, os.WNOHANG)


def test_fault_client_normal_json_response_and_exit_are_preserved(tmp_path):
    lab = LocalProcessLab(
        tmp_path,
        'import sys; sys.stdin.read(); print(\'{"outcome":"accepted"}\'); '
        "sys.stderr.write('warning\\n'); raise SystemExit(3)",
    )
    client = faults._Client(lab, "lab", "local-process-double", "SELECT 1;")
    result = client.finish(timeout=5)
    assert result.returncode == 3 and result.stdout == '{"outcome":"accepted"}\n'
    assert result.stderr == "warning\n" and result.response == {"outcome": "accepted"}
