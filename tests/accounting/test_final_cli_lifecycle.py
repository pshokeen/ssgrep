"""Accounting CLI consent races and failure cleanup against real files and sockets."""

from __future__ import annotations

import json
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from ssgrep.accounting import http, policy
from ssgrep.accounting.lease import Lease
from ssgrep.accounting.serve_loop import Worker
from ssgrep.cli.accounting_cli import stdin_is_tty
from ssgrep.cli.commands.metrics_serve_command import MetricsServeCommand
from tests.accounting.test_export_consent import DEST, configure, enabled_pairs
from tests.accounting.test_savings_cli import BINARY, Run, document

PROFILE = "personal-local"


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def environment(run: Run) -> dict[str, str]:
    return {**os.environ, "HOME": str(run.home), "SSGREP_DATA_DIR": str(run.data), "NO_COLOR": "1"}


@contextmanager
def prompt(run: Run, *arguments: str):
    """Wait for an actual consent prompt before sending input or changing its policy."""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(BINARY), *arguments],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment(run),
    )
    os.close(slave)
    preview = bytearray()
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 30
        while b"[y/N] " not in preview:
            remaining = deadline - time.monotonic()
            assert remaining > 0, bytes(preview)
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            assert ready, bytes(preview)
            chunk = os.read(process.stdout.fileno(), 65536)
            assert chunk, (process.poll(), bytes(preview))
            preview.extend(chunk)
        yield process, master
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        os.close(master)


def export_arguments() -> tuple[str, ...]:
    return ("metrics", "export", "--enable", "--destination", DEST, "--signal", "metrics")


@pytest.mark.parametrize("command", ["setup", "export"])
@pytest.mark.parametrize("answer", [b"yes\n", b"no\n", b"\x04"], ids=["approve", "refuse", "eof"])
def test_interactive_consent_changes_only_the_approved_state(run: Run, command: str, answer: bytes):
    if command == "export":
        configure()
    before = policy.policy_path().read_bytes() if policy.policy_path().exists() else None
    args = ("metrics", "setup") if command == "setup" else export_arguments()
    with prompt(run, *args) as (process, terminal):
        os.write(terminal, answer)
        stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert b"Traceback" not in stdout + stderr
    approved = answer == b"yes\n"
    if command == "setup":
        assert policy.load_profile_policy().collection_enabled is approved
    else:
        assert enabled_pairs() == (((DEST, "metrics"),) if approved else ())
    if not approved:
        after = policy.policy_path().read_bytes() if policy.policy_path().exists() else None
        assert after == before


@pytest.mark.parametrize("command", ["setup", "export"])
def test_consent_revalidates_policy_changed_while_prompt_is_open(run: Run, command: str):
    configure()
    args = ("metrics", "setup") if command == "setup" else export_arguments()
    with prompt(run, *args) as (process, terminal):
        policy.policy_path().write_text("{")
        os.write(terminal, b"yes\n")
        stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 1
    assert policy.policy_path().read_bytes() == b"{"
    assert b"ssgrep metrics setup" in stderr
    assert b"Traceback" not in stdout + stderr


def test_oversized_setup_policy_is_rejected_before_creating_accounting_state(
    run: Run, tmp_path: Path
):
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * (256 * 1024 + 1))
    result = run("metrics", "setup", "--yes", "--policy", str(source), "--json")
    assert result.returncode == 2
    assert document(result)["condition"] == "invalid_policy"
    assert not run.data.exists()


@pytest.mark.parametrize("port", [-1, 65536])
def test_serve_rejects_out_of_range_ports_without_creating_state(run: Run, port: int):
    result = run("metrics", "serve", "--port", str(port), "--json")
    assert result.returncode == 2
    assert document(result)["condition"] == "invalid_port"
    assert document(result)["command"] == "ssgrep metrics serve"
    assert not run.data.exists()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_second_worker_on_a_different_port_does_not_disturb_the_owner(run: Run):
    port = free_port()
    with Lease.acquire(PROFILE, "collect"):
        result = run("metrics", "serve", "--port", str(port), "--json")
        assert result.returncode == 1
        error = document(result)
        assert error["condition"] == "already_running"
        assert str(os.getpid()) in error["message"]
        assert error["command"] == "ssgrep metrics status"
        assert len(result.stdout.splitlines()) == 1
        with socket.socket() as rebound:
            rebound.bind(("127.0.0.1", port))
    with Lease.acquire(PROFILE, "collect"):
        pass


@pytest.mark.parametrize("occupied", [False, True])
def test_unsafe_worker_storage_has_a_sanitized_startup_error(run: Run, occupied: bool):
    assert run("metrics", "setup", "--yes").returncode == 0
    profile = run.data / "accounting" / PROFILE
    profile.mkdir(mode=0o700, exist_ok=True)
    # A regular file where the private worker directory must be cannot hold a lease.
    (profile / "worker").write_text("KEEP-THIS-CANARY")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
        if occupied:
            listener.listen()
        else:
            listener.close()
        result = run("metrics", "serve", "--port", str(port), "--json")
    assert result.returncode == 1
    error = document(result)
    assert error["condition"] in {"server_start_failed", "serve_failed"}
    assert error["command"] == "ssgrep metrics status"
    assert "KEEP-THIS-CANARY" not in result.stdout + result.stderr
    assert (profile / "worker").read_text() == "KEEP-THIS-CANARY"


@pytest.mark.parametrize(
    "failure", [RuntimeError("PRIVATE-DETAIL"), KeyboardInterrupt(), SystemExit(7)]
)
@pytest.mark.parametrize("stage", ["start", "running"])
def test_serve_failure_releases_listener_lease_and_signal_handlers(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    stage: str,
):
    port = free_port()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def fail_start(self):
        raise failure

    def fail_loop(self, cycle):
        # The real listener and worker lease are both live before the injected failure.
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            pass
        from ssgrep.accounting.lease import AlreadyRunning

        with pytest.raises(AlreadyRunning):
            Lease.acquire(PROFILE, "collect", retry_seconds=0)
        raise failure

    if stage == "start":
        monkeypatch.setattr(http.ServerHandle, "start", fail_start)
    else:
        monkeypatch.setattr(Worker, "_loop", fail_loop)
    command = object.__new__(MetricsServeCommand)
    expected = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else SystemExit
    with pytest.raises(expected) as caught:
        command.handle(port=port, profile=PROFILE)
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == (7 if isinstance(failure, SystemExit) else 1)
    output = capsys.readouterr()
    assert "PRIVATE-DETAIL" not in output.out + output.err
    if isinstance(failure, RuntimeError):
        assert "ssgrep metrics status" in output.err
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
    with socket.socket() as rebound:
        rebound.bind(("127.0.0.1", port))
    with Lease.acquire(PROFILE, "collect"):
        pass


@pytest.mark.parametrize(
    ("expression", "code"),
    [
        ("RuntimeError('PRIVATE-DETAIL')", 1),
        ("KeyboardInterrupt()", 130),
        ("SystemExit(7)", 7),
        ("SystemExit('PRIVATE-DETAIL')", 1),
    ],
)
def test_json_announced_worker_failure_emits_no_second_document(
    run: Run, expression: str, code: int
):
    port = free_port()
    script = f"""
from usecli.cli.core.runtime import execution_context
from ssgrep.accounting.serve_loop import Worker
from ssgrep.cli.commands.metrics_serve_command import MetricsServeCommand
def fail(self, cycle):
    self._publish(phase="idle")
    raise {expression}
Worker._loop = fail
with execution_context(json_mode=True):
    object.__new__(MetricsServeCommand).handle(port={port}, profile={PROFILE!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment(run),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == code
    assert len(result.stdout.splitlines()) == 1
    startup = json.loads(result.stdout)
    assert startup["state"] == "started" and startup["port"] == port
    assert "PRIVATE-DETAIL" not in result.stdout + result.stderr
    assert result.stderr.strip()
    with Lease.acquire(PROFILE, "collect"):
        pass


def test_closed_or_absent_stdin_cannot_grant_interactive_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with (tmp_path / "stdin").open("w+") as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        assert stdin_is_tty() is False
    assert stdin_is_tty() is False
    monkeypatch.setattr(sys, "stdin", None)
    assert stdin_is_tty() is False
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave) as terminal:
            monkeypatch.setattr(sys, "stdin", terminal)
            assert stdin_is_tty() is True
    finally:
        os.close(master)


@pytest.mark.parametrize("json_mode", [False, True])
def test_collect_damaged_checkpoint_is_actionable_and_does_not_replace_it(
    run: Run,
    json_mode: bool,
):
    from ssgrep.accounting import plainfile

    assert run("metrics", "setup", "--yes").returncode == 0
    plainfile.write_private(
        run.data, "accounting", PROFILE, "collector", "checkpoint.json", data=b"{"
    )
    result = run("metrics", "collect", *(("--json",) if json_mode else ()))
    assert result.returncode == 1
    assert (
        run.data / "accounting" / PROFILE / "collector" / "checkpoint.json"
    ).read_bytes() == b"{"
    if json_mode:
        assert document(result)["condition"] == "state_corrupt"
        assert document(result)["command"] == "ssgrep metrics status"
    else:
        assert "ssgrep metrics status" in result.stderr
    with Lease.acquire(PROFILE, "collect"):
        pass


@pytest.mark.parametrize(
    ("kind", "condition"),
    [("key", "preview_unavailable"), ("ledger", "state_corrupt"), ("registry", "registry_invalid")],
)
def test_export_preview_refuses_damaged_private_state_without_repairing_it(
    run: Run,
    kind: str,
    condition: str,
):
    from ssgrep.accounting import plainfile
    from tests.accounting.test_export_consent import publish_snapshot

    configure()
    publish_snapshot()
    parts = {
        "key": ("export", "pseudonym.key"),
        "ledger": ("collector", "checkpoint.json"),
        "registry": ("export", DEST, "records.json"),
    }[kind]
    plainfile.write_private(run.data, "accounting", PROFILE, *parts, data=b"{")
    target = run.data.joinpath("accounting", PROFILE, *parts)
    before = target.stat().st_mtime_ns
    result = run(
        "metrics",
        "export",
        "--preview",
        "--destination",
        DEST,
        "--signal",
        "metrics" if kind == "key" else "logs",
        "--json",
    )
    assert result.returncode == 1
    error = document(result)
    assert error["condition"] == condition
    assert error["command"] == "ssgrep metrics status"
    assert target.read_bytes() == b"{" and target.stat().st_mtime_ns == before
    assert not (run.data / "accounting" / PROFILE / "spool").exists()


def test_serve_listener_creation_error_is_sanitized_and_actionable(
    run: Run,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def denied(*args, **kwargs):
        raise OSError("PRIVATE-SOCKET-DETAIL")

    monkeypatch.setattr(http, "serve", denied)
    with pytest.raises(SystemExit) as caught:
        object.__new__(MetricsServeCommand).handle(port=0, profile=PROFILE)
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert "ssgrep metrics status" in output.err
    assert "PRIVATE-SOCKET-DETAIL" not in output.out + output.err
    with Lease.acquire(PROFILE, "collect"):
        pass
