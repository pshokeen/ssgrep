"""A forked retrieval never inherits a locked fingerprint-loading mutex."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = (
    r"""
import json
import os
import threading
from ssgrep.accounting.capture import fingerprint, journal, journal_reader
from ssgrep.accounting.privacy import policy, storage
from ssgrep.accounting.ledger.contracts import Phase
from ssgrep.services import mcp_server

profile = "personal-local"
policy.enable_collection(profile)
response = mcp_server.search_sessions("isolated-fork-control")
assert "error" in response
key = fingerprint.load_key(profile, create=False)
assert key is not None and len(key) == fingerprint.KEY_BYTES
before = {event.execution_id for event in """
    r"""journal_reader.read_journal(profile, quarantine=False).events
          if event.phase is Phase.COMPLETED}
assert len(before) == 1
fingerprint.forget_keys()
assert fingerprint.load_key("fork-cache-control", create=True) is not None
assert fingerprint._cache and fingerprint._signatures
entered = threading.Event()
release = threading.Event()
real_open = storage.open_regular

def paused_open(*args, **kwargs):
    if threading.current_thread().name == "held-key-loader":
        entered.set()
        assert release.wait(10)
    return real_open(*args, **kwargs)

storage.open_regular = paused_open
loaded = []
loader = threading.Thread(target=lambda: loaded.append(fingerprint.load_key(profile, create=False)),
                          name="held-key-loader", daemon=True)
loader.start()
assert entered.wait(10)
assert fingerprint._lock.locked()
pid = os.fork()
if pid == 0:
    os.write(1, b"READY\n")
    try:
        cleared_on_fork = not fingerprint._cache and not fingerprint._signatures
        assert mcp_server.search_sessions("isolated-fork-control") == response
        assert cleared_on_fork
        assert fingerprint.load_key(profile, create=False) == key
    except BaseException:
        os._exit(91)
    os._exit(0)
try:
    waited, status = os.waitpid(pid, 0)
    assert waited == pid and os.waitstatus_to_exitcode(status) == 0
finally:
    release.set()
    loader.join(2)
    storage.open_regular = real_open
assert not loader.is_alive() and loaded == [key]
after = {event.execution_id for event in """
    r"""journal_reader.read_journal(profile, quarantine=False).events
         if event.phase is Phase.COMPLETED}
assert len(after - before) == 1
journal.close_all()
print(json.dumps({"retrieval_unchanged": True, "valid_key": True, "child_captured": True}))
"""
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork is required")
def test_child_retrieval_after_fork_during_key_load_has_an_independent_watchdog(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["SSGREP_DATA_DIR"] = str(tmp_path / "fork-data")
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", PROBE],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    try:
        ready, _, _ = select.select([process.stdout], [], [], 20)
        assert ready, "Fork probe never reached readiness"
        assert os.read(process.stdout.fileno(), len(b"READY\n")) == b"READY\n"
        # Startup/import time is excluded; an independent process enforces the deadline.
        stdout, stderr = process.communicate(timeout=2)
        assert process.returncode == 0, (process.returncode, stderr)
        assert json.loads(stdout) == {
            "retrieval_unchanged": True,
            "valid_key": True,
            "child_captured": True,
        }
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
