"""Real local subprocess boundaries; never invokes an agent or provider."""

import json
import os
import sys

import pytest

from eval.accounting_airline.bounded_capture import capture_process


def capture(tmp_path, code, *, stdout_limit=4096, stderr_limit=2048, timeout=5, stdin=None):
    return capture_process(
        [sys.executable, "-I", "-c", code],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        cwd=tmp_path,
        timeout=timeout,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
        input_bytes=stdin,
        output_dir=tmp_path / "capture",
    )


def assert_reaped(result):
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)
    with pytest.raises(ProcessLookupError):
        os.kill(result.pid, 0)


@pytest.mark.parametrize("stream,limit", [(1, 4096), (2, 2048)])
@pytest.mark.parametrize("extra", [0, 1])
def test_exact_byte_boundary_and_one_over(tmp_path, stream, limit, extra):
    result = capture(tmp_path, f"import os; os.write({stream}, b'x' * {limit + extra})")
    name = "stdout" if stream == 1 else "stderr"
    assert getattr(result, name) == b"x" * limit
    assert getattr(result, name + "_bytes") == limit + extra
    assert getattr(result, name + "_truncated") is bool(extra)
    assert result.output_limit is bool(extra)
    assert result.status == ("output_limit" if extra else "completed")
    assert result.output_limit_streams == ((name,) if extra else ())
    assert (tmp_path / "capture" / (name + ".txt")).read_bytes() == b"x" * limit
    assert_reaped(result)


@pytest.mark.parametrize("streams", [(1,), (2,), (1, 2)])
def test_flooded_streams_are_drained_bounded_and_reaped(tmp_path, streams):
    code = (
        "import os,threading; "
        "def_source = 'def flood(fd):\\n while True: os.write(fd, b\"z\" * 65536)'; "
        "exec(def_source); "
        f"threads=[threading.Thread(target=flood,args=(fd,)) for fd in {streams!r}]; "
        "[t.start() for t in threads]; [t.join() for t in threads]"
    )
    result = capture(tmp_path, code)
    assert result.output_limit and not result.timed_out
    assert result.stdout_bytes + result.stderr_bytes > 0
    assert len(result.stdout) <= 4096 and len(result.stderr) <= 2048
    assert result.status == "output_limit" and result.returncode != 0
    for name, limit in (("stdout", 4096), ("stderr", 2048)):
        artifact = tmp_path / "capture" / (name + ".txt")
        assert artifact.stat().st_size <= limit
        assert artifact.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "capture").stat().st_mode & 0o777 == 0o700
    assert_reaped(result)


def test_large_valid_single_json_line_obeys_aggregate_cap(tmp_path):
    line = json.dumps({"text": "a" * 4096}).encode() + b"\n"
    result = capture(tmp_path, f"import os; os.write(1, {line!r})")
    assert result.output_limit and result.stdout_truncated
    assert len(result.stdout) == 4096
    assert result.stdout_bytes > 4096
    assert_reaped(result)


def test_blocked_stdin_writer_cannot_deadlock_stdout_overflow(tmp_path):
    result = capture(
        tmp_path,
        "import os,time; os.write(1,b'x'*65536); time.sleep(60)",
        stdin=b"input" * 100_000,
    )
    assert result.output_limit and not result.timed_out
    assert_reaped(result)


def test_timeout_retains_partial_artifacts_without_retry(tmp_path):
    result = capture(
        tmp_path,
        "import os,time; os.write(1,b'partial'); os.write(2,b'error'); time.sleep(60)",
        timeout=0.5,
    )
    assert result.timed_out and not result.output_limit
    assert result.status == "timeout"
    assert not result.capture_complete and result.pipes_drained
    assert result.stdout == b"partial" and result.stderr == b"error"
    assert (result.stdout_bytes, result.stderr_bytes) == (7, 5)
    assert not result.stdout_truncated and not result.stderr_truncated
    assert (tmp_path / "capture" / "stdout.txt").read_bytes() == b"partial"
    assert_reaped(result)


def test_timeout_applies_even_after_both_output_pipes_close(tmp_path):
    result = capture(
        tmp_path, "import os,time; os.close(1); os.close(2); time.sleep(60)", timeout=0.5
    )
    assert result.timed_out and result.status == "timeout"
    assert_reaped(result)


def test_timeout_kills_only_owned_group_including_blocked_descendant(tmp_path):
    # SIGTERM is deliberately ignored; cleanup must be a real owned-group SIGKILL.
    pid_file = tmp_path / "child.pid"
    code = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pid=os.fork(); "
        f"open({str(pid_file)!r},'w').write(str(pid)) if pid else None; "
        "time.sleep(60)"
    )
    result = capture(tmp_path, code, timeout=0.5)
    assert result.timed_out and not result.capture_complete and result.pipes_drained
    child = int(pid_file.read_text())
    # A killed orphan can briefly remain a zombie; it must no longer run or hold pipes.
    import subprocess

    state = subprocess.run(
        ["/bin/ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True
    )
    assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
    assert_reaped(result)


def test_normal_json_and_stdin_preserve_exact_bytes(tmp_path):
    line = b'{"native":"ok","text":"\\u00e9"}\n'
    result = capture(
        tmp_path,
        "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); "
        "sys.stderr.buffer.write(b'warning\\r\\n')",
        stdin=line,
    )
    assert result.returncode == 0 and result.status == "completed"
    assert result.stdout == line and result.stderr == b"warning\r\n"
    assert result.stdout_bytes == len(line) and result.stderr_bytes == 9
    assert result.capture_complete and not result.output_limit
    assert_reaped(result)


def test_existing_artifact_is_refused_before_child_start(tmp_path):
    target = tmp_path / "capture"
    target.mkdir()
    (target / "stdout.txt").write_text("prior evidence")
    marker = tmp_path / "started"
    with pytest.raises(FileExistsError):
        capture(tmp_path, f"open({str(marker)!r},'w').write('started')")
    assert not marker.exists()
    assert (target / "stdout.txt").read_text() == "prior evidence"


def test_owner_cancellation_reaps_running_capture(tmp_path):
    import threading

    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set)
    timer.start()
    try:
        result = capture_process(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
            cwd=tmp_path,
            timeout=30,
            stdout_limit=128,
            stderr_limit=128,
            cancel_event=stop,
        )
    finally:
        timer.cancel()
        timer.join()
    assert result.status == "cancelled" and result.cancelled and not result.timed_out
    assert_reaped(result)


def test_overflow_never_kills_unrelated_process_group(tmp_path):
    import subprocess

    unrelated = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        result = capture(tmp_path, "import os,time; os.write(1,b'x'*65536); time.sleep(60)")
        assert result.output_limit and unrelated.poll() is None
        assert_reaped(result)
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_normal_exit_cleans_devnull_descendant_but_preserves_unrelated_group(tmp_path):
    import subprocess

    pid_file = tmp_path / "background.pid"
    descendant = (
        f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)"
    )
    wait_code = f"while not os.path.exists({str(pid_file)!r}): time.sleep(0.01)"
    code = (
        "import subprocess,sys,time,os; "
        f"subprocess.Popen([sys.executable,'-I','-c',{descendant!r}], "
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"exec({wait_code!r}); "
        "print('normal')"
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    result = None
    try:
        result = capture(tmp_path, code)
        assert result.returncode == 0 and result.status == "completed"
        assert result.stdout == b"normal\n" and result.capture_complete
        state = subprocess.run(
            ["/bin/ps", "-o", "stat=", "-p", pid_file.read_text()],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
        assert unrelated.poll() is None
        assert_reaped(result)
    finally:
        if result is not None:
            try:
                os.killpg(result.pid, 9)
            except ProcessLookupError:
                pass
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_normal_nested_capture_never_kills_inherited_owner_group(tmp_path):
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    body = (
        "try:\n"
        " result=capture_process([sys.executable,'-I','-c','print(123)'],"
        "env={'PATH':'/usr/bin:/bin'},"
        f"cwd=Path({str(tmp_path)!r}),timeout=5,stdout_limit=128,stderr_limit=128,start_new_session=False)\n"
        " print(json.dumps({'returncode':result.returncode,'stdout':result.stdout.decode(),"
        "'owner_group_survived':survivor.poll() is None}))\n"
        "finally:\n survivor.kill(); survivor.wait(timeout=5)\n"
    )
    code = (
        f"import sys; sys.path.insert(0,{str(repo)!r}); "
        "import subprocess,os,json; from pathlib import Path; "
        "from eval.accounting_airline.bounded_capture import capture_process; "
        "survivor=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"exec({body!r})"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        start_new_session=True,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {
        "returncode": 0,
        "stdout": "123\n",
        "owner_group_survived": True,
    }
