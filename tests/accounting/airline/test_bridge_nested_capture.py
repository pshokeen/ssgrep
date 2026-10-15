"""End-to-end real shim descendants and honest output-limit supervisor evidence."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.accounting_airline import live_bridge, observer
from tests.accounting.airline.test_native_capture_limits import post


def fixture_kit(tmp_path, tool_body):
    executable = tmp_path / "local-tool"
    executable.write_text(f"#!{sys.executable}\n" + tool_body)
    executable.chmod(0o700)
    private = tmp_path / "verifier"
    private.mkdir(mode=0o700)
    task = private / "task"
    task.write_text("A/L1")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        observer.ENV_SPOOL: str(private / "spool.ndjson"),
        observer.ENV_ACTOR: "solver-a",
        observer.ENV_RUN: "bounded-run",
        observer.ENV_TASK_FILE: str(task),
        observer.ENV_REAL + "SSGREP": str(executable),
    }
    return SimpleNamespace(env=env, spool=private / "spool.ndjson")


@pytest.mark.parametrize("fd", [1, 2])
def test_real_nested_tool_flood_cannot_accumulate_inside_shim(tmp_path, fd):
    kit = fixture_kit(
        tmp_path, "import os,time\n" + f"os.write({fd},b'x' * 2_000_000)\n" + "time.sleep(60)\n"
    )
    with live_bridge.Bridge(kit, "assigned", tmp_path) as bridge:
        data = post(bridge)
        attempts = list(bridge.attempts)
    assert data["output_limit"] and data["capture_status"] == "output_limit"
    assert data["returncode"] != 0 and data["stdout"] == ""
    events = observer.read_trace(kit.spool)
    dispatch, result = [e for e in events if e["kind"] in ("dispatch", "result")]
    assert dispatch["task"] == "A/L1" and result["call_id"] == dispatch["call_id"]
    assert result["state"] == "interrupted" and result["reason"] == "output_limit"
    assert result["exit_code"] != 0
    nested = attempts[0]["tool_capture"]
    name = "stdout" if fd == 1 else "stderr"
    assert nested[name + "_bytes"] == observer.MAX_BLOB_BYTES + 1
    assert nested[name + "_truncated"] is True
    assert nested["returncode"] != 0 and nested["output_limit"]
    with pytest.raises(ChildProcessError):
        os.waitpid(attempts[0]["pid"], os.WNOHANG)
    assert (
        Path(attempts[0]["artifact_dir"], "tool", name + ".txt").stat().st_size
        == observer.MAX_BLOB_BYTES
    )


def test_outer_owned_group_cleans_nested_descendants(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    body = (
        "import os,time\n"
        "pid=os.fork()\n"
        "if not pid:\n"
        f" open({str(pid_file)!r},'w').write(str(os.getpid()))\n"
        " time.sleep(60)\n"
        "while not os.path.exists(" + repr(str(pid_file)) + "): time.sleep(0.01)\n"
        "os.write(1,b'x'*2_000_000)\n"
        "time.sleep(60)\n"
    )
    kit = fixture_kit(tmp_path, body)
    with live_bridge.Bridge(kit, "assigned", tmp_path) as bridge:
        data = post(bridge)
    assert data["output_limit"]
    state = subprocess.run(
        ["/bin/ps", "-o", "stat=", "-p", pid_file.read_text()],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert not state.stdout.strip() or state.stdout.strip().startswith("Z")


def test_real_nested_normal_tool_preserves_output_and_observation(tmp_path):
    kit = fixture_kit(tmp_path, "import sys\nprint('{\"ok\":true}')\nsys.stderr.write('warn\\n')\n")
    with live_bridge.Bridge(kit, "assigned", tmp_path) as bridge:
        data = post(bridge)
    assert (data["returncode"], data["stdout"], data["stderr"]) == (0, '{"ok":true}\n', "warn\n")
    result = [e for e in observer.read_trace(kit.spool) if e["kind"] == "result"][0]
    assert result["state"] == "verified" and result["exit_code"] == 0
