"""Real subprocesses exercise native-lab tool and prerequisite capture adapters."""

import os
import subprocess
import sys

import pytest

from eval.accounting_airline import bounded_capture, database, lab_product, manifest, sandbox
from eval.accounting_airline.gates import BLOCKED


def executable(path, source):
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o700)
    return path


def test_bounded_runner_preserves_bytes_text_newlines_stdin_exit_and_timeout(tmp_path):
    done = bounded_capture.run_bounded(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); "
            "sys.stderr.buffer.write(b'warning\\r\\n'); raise SystemExit(3)",
        ],
        cwd=tmp_path,
        input="input\r\n",
        text=True,
        timeout=5,
    )
    assert (done.returncode, done.stdout, done.stderr) == (3, "input\n", "warning\n")
    raw = bounded_capture.run_bounded(
        [sys.executable, "-I", "-c", "import os; os.write(1,b'x\\r\\n')"], cwd=tmp_path, timeout=5
    )
    assert raw.stdout == b"x\r\n"
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        bounded_capture.run_bounded(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"], cwd=tmp_path, timeout=0.5
        )
    assert caught.value.timeout == 0.5


def test_bounded_runner_never_treats_zero_exit_overflow_as_success(tmp_path):
    with pytest.raises(bounded_capture.OutputLimitExceeded) as caught:
        bounded_capture.run_bounded(
            [sys.executable, "-I", "-c", "import os; os.write(1,b'x'*129)"],
            cwd=tmp_path,
            timeout=5,
            stdout_limit=128,
            stderr_limit=64,
        )
    assert caught.value.capture.stdout_bytes == 129 and caught.value.capture.stdout_truncated
    assert "output_limit" in str(caught.value) and "x" * 20 not in str(caught.value)
    with pytest.raises(ChildProcessError):
        os.waitpid(caught.value.capture.pid, os.WNOHANG)


@pytest.mark.parametrize("kind", ["product", "docker", "sandbox"])
def test_public_tool_capture_seams_reject_real_flood(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(bounded_capture, "DEFAULT_STDOUT_LIMIT", 1024)
    monkeypatch.setattr(bounded_capture, "DEFAULT_STDERR_LIMIT", 512)
    binary = executable(
        tmp_path
        / ("docker" if kind == "docker" else "sandbox-exec" if kind == "sandbox" else "tool"),
        "import os,time\nos.write(1,b'x'*8192)\ntime.sleep(60)\n",
    )
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin", "HOME": str(tmp_path)}
    monkeypatch.setenv("PATH", env["PATH"])
    with pytest.raises(bounded_capture.OutputLimitExceeded):
        if kind == "product":
            lab_product.Product(env, tmp_path, binary).run("--test-local-double", timeout=5)
        elif kind == "docker":
            database._docker_raw("--test-local-double", env=env, timeout=5)
        else:
            sandbox.run_sandboxed(
                sandbox.SandboxProfile(
                    text="(version 1)(allow default)", denied=(), limitations=()
                ),
                ["/usr/bin/true"],
                env=env,
                cwd=tmp_path,
                timeout=5,
            )


def test_runtime_manifest_probe_overflow_is_named_blocked_not_pass(tmp_path, monkeypatch):
    executable(
        tmp_path / "omp",
        "import os,sys\n" + "print('omp/0.0.0-test-double') if sys.argv[1]=='--version' "
        "else os.write(1,b'x'*2_000_000)\n",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    gate = manifest._gate_runtime_binary("omp")
    assert gate.status == BLOCKED and "output_limit" in gate.detail
    assert gate.evidence["capture"]["stdout_truncated"] is True


def test_sandbox_probe_overflow_is_named_unavailable_not_pass(tmp_path, monkeypatch):
    executable(tmp_path / "sandbox-exec", "import os\nos.write(1,b'x'*2_000_000)\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    limitation = sandbox.availability()
    assert limitation is not None and "output_limit" in limitation.reason
