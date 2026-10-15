"""Unsafe accounting policy must not change real CLI/MCP lookup behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = (
    r"""
import importlib
import json
import os
import signal
import sys
from pathlib import Path
from typer.testing import CliRunner
from tests.accounting.verify_scenes import env_for, isolated
from ssgrep.accounting import capture_policy, journal_reader, plainfile, policy
from ssgrep.services import mcp_server

root=Path(sys.argv[1])
surface, kind, operation=sys.argv[2:]
sys.argv=["ssgrep",operation,"isolated-no-index" if operation=="search" else "missing:ep:1"]
with isolated(env_for(root)):
    from usecli import app
    def invoke():
        if surface == "mcp":
            value=(mcp_server.search_sessions("isolated-no-index") if operation == "search"
                   else mcp_server.show_session("missing:ep:1"))
            return json.dumps(value,sort_keys=True)
        result=CliRunner().invoke(app,[operation,"isolated-no-index" if operation == "search"
                                      else "missing:ep:1"],catch_exceptions=False)
        return json.dumps([result.exit_code,result.stdout,result.stderr])

    baseline=invoke()
    policy.enable_collection("personal-local")
    account=root/"data"/"accounting"
    target=account/"policy.json"
    valid=target.read_bytes()
    target.unlink()
    if kind == "fifo":
        os.mkfifo(target,0o600)
    elif kind == "symlink":
        outside=root/"private-not-authority"
        outside.write_text("private canary must not be consumed")
        target.symlink_to(outside)
    else:
        target.write_bytes(valid)
        target.chmod(0o600)

    real_open=os.open
    attempts=0
    swapped=False
    parent_info=account.stat()
    parent_identity=(parent_info.st_dev,parent_info.st_ino)
    def watched_open(path,flags,*args,**kwargs):
        global attempts,swapped
        parent_fd=kwargs.get("dir_fd")
        relative_leaf=(isinstance(path,(str,os.PathLike)) and os.fspath(path)=="policy.json"
                       and parent_fd is not None)
        if relative_leaf:
            info=os.fstat(parent_fd)
            relative_leaf=(info.st_dev,info.st_ino)==parent_identity
        if isinstance(path,(str,os.PathLike)) and (Path(path)==target or relative_leaf):
            attempts+=1
            if kind == "fifo_race" and not swapped:
                swapped=True
                target.unlink()
                os.mkfifo(target,0o600)
        return real_open(path,flags,*args,**kwargs)
    plainfile.os.open=watched_open
    importlib.reload(capture_policy)
    signal.signal(signal.SIGALRM,lambda *_: sys.exit(93))
    signal.alarm(2)
    try:
        actual=invoke()
    finally:
        signal.alarm(0)
    assert actual==baseline,(actual,baseline)
    assert attempts>0,("policy fault was not reached",surface,baseline,str(target))
    assert kind!="fifo_race" or swapped,"replacement race did not occur"
    bad_attempts=attempts
    before={e.execution_id for e in """
    r"""journal_reader.read_journal("personal-local",quarantine=False).events}
    target.unlink()
    target.write_bytes(valid)
    target.chmod(0o600)
    importlib.reload(capture_policy)
    assert invoke()==baseline
    after=journal_reader.read_journal("personal-local",quarantine=False).events
    completed=[e for e in after if e.phase.value=="completed" and e.execution_id not in before]
    assert len(completed)==1
    assert completed[0].transport.value==surface
    assert completed[0].operation.value==operation
    print(json.dumps({"unsafe_open_attempts":bad_attempts,"replacement_occurred":swapped,
                      "valid_capture":completed[0].operation.value,"unchanged_response":True}))
"""
)


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize("kind", ["fifo", "fifo_race", "symlink"])
@pytest.mark.parametrize("operation", ["search", "show"])
def test_unsafe_policy_preserves_real_lookup_response_and_capture(
    tmp_path: Path, surface: str, kind: str, operation: str
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path), surface, kind, operation],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, (completed.returncode, completed.stdout, completed.stderr)
    result = json.loads(completed.stdout)
    assert result["unsafe_open_attempts"] > 0
    assert result["replacement_occurred"] is (kind == "fifo_race")
    assert result["valid_capture"] == operation
    assert result["unchanged_response"] is True
