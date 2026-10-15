"""Verifier/control boundaries fail explicitly after retaining actual process evidence."""

import os
import sys

import pytest

from eval.accounting_airline import observer


@pytest.mark.parametrize("fd", [1, 2])
def test_verifier_output_limit_is_not_a_normal_domain_process_failure(tmp_path, fd):
    recorder = observer.Observer(tmp_path, "bounded-verifier")
    with pytest.raises(observer.ObserverError, match="output_limit"):
        recorder.run(
            actor="control",
            task="boundary",
            tool="python",
            argv=[
                sys.executable,
                "-I",
                "-c",
                f"import os,time; os.write({fd}, b'x'*2_000_000); time.sleep(60)",
            ],
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
            cwd=tmp_path,
            timeout=5,
        )
    dispatch, result = observer.read_trace(recorder.trace.path)
    assert result["call_id"] == dispatch["call_id"] and result["state"] == "interrupted"
    assert result["reason"] == "output_limit" and result["capture"]["output_limit"]
    name = "stdout" if fd == 1 else "stderr"
    assert result["capture"][name + "_bytes"] > observer.MAX_BLOB_BYTES
    path = tmp_path / "control-attempts" / dispatch["call_id"] / (name + ".txt")
    assert path.stat().st_size == observer.MAX_BLOB_BYTES and path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ChildProcessError):
        os.waitpid(result["capture"]["pid"], os.WNOHANG)


def test_verifier_normal_stdout_stderr_stdin_and_exit_remain_exact(tmp_path):
    recorder = observer.Observer(tmp_path, "bounded-verifier")
    call, done = recorder.run(
        actor="control",
        task="normal",
        tool="python",
        argv=[
            sys.executable,
            "-I",
            "-c",
            "import sys; print(sys.stdin.read()); "
            "sys.stderr.write('warn\\r\\n'); raise SystemExit(3)",
        ],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        cwd=tmp_path,
        timeout=5,
        stdin="input",
    )
    assert done is not None and (done.returncode, done.stdout, done.stderr) == (
        3,
        "input\n",
        "warn\n",
    )
    assert (call.state, call.exit_code, call.stdout) == ("verified", 3, "input\n")


def test_verifier_timeout_preserves_public_no_exit_and_retains_prefix(tmp_path):
    recorder = observer.Observer(tmp_path, "bounded-verifier")
    call, done = recorder.run(
        actor="control",
        task="timeout",
        tool="python",
        argv=[sys.executable, "-I", "-c", "import os,time; os.write(1,b'partial'); time.sleep(60)"],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        cwd=tmp_path,
        timeout=0.5,
    )
    assert done is None and call.state == "interrupted" and call.exit_code is None
    result = observer.read_trace(recorder.trace.path)[-1]
    assert result["reason"] == "timeout" and result["capture"]["stdout_bytes"] == 7
    path = tmp_path / "control-attempts" / call.call_id / "stdout.txt"
    assert path.read_bytes() == b"partial" and path.stat().st_mode & 0o777 == 0o600
