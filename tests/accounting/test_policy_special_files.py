"""Private readers reject kernel special files before they can block a caller."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = r"""
import json
import os
import signal
import socket
import sys
from pathlib import Path
from ssgrep.accounting import plainfile

base = Path(sys.argv[1])
kind = sys.argv[2]
folder = base / "accounting"
folder.mkdir(mode=0o700)
target = folder / "policy.json"
expected = b'{"collection_enabled": true}'
listener = None
os.chdir(folder)
if kind == "directory":
    target.mkdir(mode=0o700)
elif kind == "socket":
    listener = socket.socket(socket.AF_UNIX)
    listener.bind("policy.json")
elif kind == "symlink":
    (folder / "outside").write_bytes(b"must not be consumed")
    target.symlink_to(folder / "outside")
elif kind == "fifo":
    os.mkfifo(target, 0o600)
else:
    target.write_bytes(expected)
    target.chmod(0o600)

if kind == "fifo_race":
    original_open = plainfile.os.open
    replaced = False
    parent_info = folder.stat()
    parent_identity = (parent_info.st_dev, parent_info.st_ino)
    def swap(path, flags, *args, **kwargs):
        global replaced
        parent_fd = kwargs.get("dir_fd")
        relative_leaf = os.fspath(path) == "policy.json" and parent_fd is not None
        if relative_leaf:
            info = os.fstat(parent_fd)
            relative_leaf = (info.st_dev, info.st_ino) == parent_identity
        if (Path(path) == target or relative_leaf) and not replaced:
            replaced = True
            target.unlink()
            os.mkfifo(target, 0o600)
        return original_open(path, flags, *args, **kwargs)
    plainfile.os.open = swap

signal.signal(signal.SIGALRM, lambda *_: sys.exit(93))
signal.alarm(2)
try:
    try:
        value = plainfile.read_private(base, "accounting", "policy.json", max_bytes=1024)
    except plainfile.UnsafePathError:
        assert kind != "regular"
        rejected = True
    else:
        assert kind == "regular" and value == expected
        rejected = False
finally:
    signal.alarm(0)
    if listener is not None:
        listener.close()
if kind == "fifo_race":
    assert replaced, "the leaf replacement race must have occurred"

if target.is_dir():
    target.rmdir()
else:
    target.unlink()
target.write_bytes(expected)
target.chmod(0o600)
assert plainfile.read_private(base, "accounting", "policy.json", max_bytes=1024) == expected
print(json.dumps({"rejected": rejected, "regular_afterward": expected.decode()}))
"""


@pytest.mark.parametrize("kind", ["fifo", "fifo_race", "socket", "directory", "symlink", "regular"])
def test_private_reader_preserves_availability_and_valid_data(tmp_path: Path, kind: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path), kind],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, (kind, completed.returncode, completed.stderr)
    result = json.loads(completed.stdout)
    assert result["rejected"] is (kind != "regular")
    assert json.loads(result["regular_afterward"]) == {"collection_enabled": True}


def test_cached_policy_permission_revocation_stops_capture_then_recovers() -> None:
    from ssgrep.accounting import datadir, journal, journal_reader, policy
    from ssgrep.accounting.contracts import Phase
    from ssgrep.services import mcp_server

    journal.close_all()
    policy.enable_collection("personal-local")
    response = mcp_server.search_sessions("isolated-policy-permission")

    def completed_ids() -> set[str]:
        return {
            event.execution_id
            for event in journal_reader.read_journal("personal-local", quarantine=False).events
            if event.phase is Phase.COMPLETED
        }

    before = completed_ids()
    assert len(before) == 1
    target = datadir.data_root() / "accounting" / "policy.json"
    original_mtime = target.stat().st_mtime_ns
    target.chmod(0o644)
    assert target.stat().st_mtime_ns == original_mtime
    try:
        assert mcp_server.search_sessions("isolated-policy-permission") == response
        assert completed_ids() == before
    finally:
        target.chmod(0o600)
    assert mcp_server.search_sessions("isolated-policy-permission") == response
    assert len(completed_ids() - before) == 1
