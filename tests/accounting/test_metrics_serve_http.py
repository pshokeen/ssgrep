"""Real CLI/socket acceptance for the foreground metrics HTTP server."""

from __future__ import annotations

import http.client
import json
import os
import select
import signal
import socket
import stat
import subprocess
from pathlib import Path

import pytest

from tests.accounting.test_savings_cli import BINARY, Run, document

PROFILE = "personal-local"


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _serve(run: Run, port: int, *, json_mode: bool = False) -> subprocess.Popen[str]:
    arguments = [str(BINARY), "metrics", "serve", "--port", str(port)]
    if json_mode:
        arguments.append("--json")
    return subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "HOME": str(run.home),
            "SSGREP_DATA_DIR": str(run.data),
            "NO_COLOR": "1",
            "COLUMNS": "80",
        },
    )


def _read_json_startup(process: subprocess.Popen[str], timeout: float = 30) -> tuple[str, dict]:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            "metrics serve did not emit its startup document before blocking; "
            f"exit={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
        )
    line = process.stdout.readline()
    if not line:
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            f"metrics serve exited before startup (status {process.returncode}); "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )
    return line, json.loads(line)


def _get(port: int, path: str, token: str | None = None) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


def _assert_stopped(port: int) -> None:
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.5)


def _terminate(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    return process.communicate(timeout=60)


def test_json_startup_precedes_blocking_worker_and_rebinds_after_sigterm(run: Run) -> None:
    assert run("metrics", "setup", "--yes").returncode == 0
    port = _free_port()
    process = _serve(run, port, json_mode=True)
    try:
        startup_line, startup = _read_json_startup(process)
        token_file = run.data / "accounting" / PROFILE / "endpoint.token"
        token = token_file.read_text().strip()

        assert startup["url"] == f"http://127.0.0.1:{port}"
        assert startup["port"] == port
        assert startup["profile"] == PROFILE
        assert startup["token_file"] == str(token_file)
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert token and token not in startup_line
        assert _get(port, "/healthz")[0] == 200
        assert _get(port, "/metrics")[0] == 401
        status, content_type, body = _get(port, "/metrics", token)
        assert status == 200
        assert content_type == "application/openmetrics-text; version=1.0.0; charset=utf-8"
        assert body.endswith(b"# EOF\n")
        assert b"# TYPE ssgrep_snapshot_available gauge\n" in body

        stdout, stderr = _terminate(process)
        assert process.returncode == 0
        assert stdout == ""  # JSON is exactly one startup document, not a shutdown stream.
        assert stderr == ""
        assert token not in startup_line + stdout + stderr
        _assert_stopped(port)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)

    # A second real serve proves both the listener and the profile worker lease were released.
    second = _serve(run, port, json_mode=True)
    try:
        second_line, second_startup = _read_json_startup(second)
        assert second_startup.get("port") == port, second_startup
        assert _get(port, "/healthz")[0] == 200
        stdout, stderr = _terminate(second)
        assert second.returncode == 0 and stdout == "" and stderr == ""
        assert token not in second_line + stdout + stderr
        _assert_stopped(port)
    finally:
        if second.poll() is None:
            second.kill()
            second.communicate(timeout=10)


@pytest.mark.parametrize("json_mode", [False, True], ids=["text", "json"])
def test_occupied_port_has_a_stable_error_and_no_alternate_listener(
    run: Run, json_mode: bool
) -> None:
    assert run("metrics", "setup", "--yes").returncode == 0
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        args = ("metrics", "serve", "--port", str(port))
        result = run(*args, *(("--json",) if json_mode else ()))

    assert result.returncode == 1
    message = f"Port {port} is already in use; no alternate port was selected."
    if json_mode:
        error = document(result)
        assert error["ok"] is False and error["condition"] == "port_in_use"
        assert error["message"] == message
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        assert message in result.stderr and "Try:" in result.stderr
    token_file = run.data / "accounting" / PROFILE / "endpoint.token"
    token = token_file.read_text().strip()
    assert token and token not in result.stdout + result.stderr


def test_cli_does_not_offer_a_remote_bind_option(run: Run) -> None:
    help_text = run("metrics", "serve", "--help").stdout
    assert "--port" in help_text
    assert "9469" in help_text
    assert "--bind" not in help_text

    result = run("metrics", "serve", "--bind", "0.0.0.0", "--json")
    assert result.returncode == 2
    assert document(result)["ok"] is False
    assert run.files() == []


def test_unsafe_endpoint_token_file_fails_without_traceback_or_secret_output(run: Run) -> None:
    token = "a" * 64
    token_file = run.data / "accounting" / PROFILE / "endpoint.token"
    token_file.parent.mkdir(parents=True, mode=0o700)
    token_file.write_text(token + "\n")
    token_file.chmod(0o644)

    result = run("metrics", "serve", "--port", str(_free_port()), "--json")

    assert result.returncode == 1
    error = document(result)
    assert error["condition"] == "token_file_unsafe"
    assert "Traceback" not in result.stderr
    assert token not in result.stdout + result.stderr
