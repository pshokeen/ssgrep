"""Regression tests for issue #10: cocoindex's telemetry opt-out.

cocoindex sends a usage-tracking event to https://cocoindex.gateway.scarf.sh
the moment it is imported (and again on later app create/update), so on every
`ssgrep index`, `ssgrep note`, and MCP-server startup reconciliation --
contradicting the "fully local and offline" promise. ``ssgrep.pipeline`` must
set the documented opt-out (``COCOINDEX_DISABLE_USAGE_TRACKING``) before it
makes cocoindex's own first import, and only as a default -- an explicit user
choice must survive.

A subprocess is required: within this test process, ``ssgrep.pipeline`` (and
therefore cocoindex) is almost certainly already imported by other test
modules, so re-importing it here would not re-run the module body that sets
the environment variable.

Routing the subprocess's own proxy env vars at a dead local port (rather than
trusting the ambient network) makes these tests assert on cocoindex's actual
behavior -- whether it attempts the request at all -- instead of only on our
own ``os.environ`` value, and keeps that assertion true whether or not this
machine has real network access: the attempt (when one happens) fails
locally either way, so the *opt-in* case never places a real request against
scarf.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

#: A distinctive prefix picks our line out of stdout, since an explicit
#: opt-*in* to tracking (the second test below) makes cocoindex itself log an
#: outbound-request attempt.
_MARKER = "COCOINDEX_TELEMETRY_ENV="
_PROBE = (
    "import ssgrep.pipeline, os; "
    f"print('{_MARKER}' + str(os.environ.get('COCOINDEX_DISABLE_USAGE_TRACKING')))"
)
#: Nothing listens here: connection attempts fail immediately (ECONNREFUSED),
#: same result with or without real network access.
_DEAD_PROXY = "http://127.0.0.1:9"


def _run_probe(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = dict(env)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        env[key] = _DEAD_PROXY
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    # cocoindex's telemetry client logs through Rust's `tracing`; ask for its
    # own crate at info level so a request attempt is visible either way.
    env["RUST_LOG"] = "cocoindex_core::telemetry=info"
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result


def _env_value(result: subprocess.CompletedProcess[str]) -> str:
    for line in result.stdout.splitlines():
        if line.startswith(_MARKER):
            return line[len(_MARKER) :]
    raise AssertionError(f"probe marker not found in stdout: {result.stdout!r}")


def _telemetry_was_attempted(result: subprocess.CompletedProcess[str]) -> bool:
    return "cocoindex_core::telemetry" in result.stdout + result.stderr


def test_cocoindex_telemetry_is_disabled_by_default() -> None:
    env = dict(os.environ)
    env.pop("COCOINDEX_DISABLE_USAGE_TRACKING", None)
    result = _run_probe(env)
    assert _env_value(result) == "1"
    assert not _telemetry_was_attempted(result)


def test_cocoindex_telemetry_opt_out_respects_an_explicit_user_choice() -> None:
    env = dict(os.environ)
    env["COCOINDEX_DISABLE_USAGE_TRACKING"] = "0"
    result = _run_probe(env)
    assert _env_value(result) == "0"
    assert _telemetry_was_attempted(result)
