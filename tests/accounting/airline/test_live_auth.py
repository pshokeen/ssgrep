"""Exercise the real gateway admission loop with a controlled, non-provider router."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from eval.accounting_airline.live_auth import GATEWAY_SOURCE, AuthGateway
from eval.accounting_airline.roots import LabRoots


def test_gateway_refuses_unverified_provider_before_start(tmp_path):
    roots = LabRoots.create(tmp_path)
    with pytest.raises(ValueError, match="openai-codex"):
        AuthGateway(roots, "unverified/provider")
    assert not (roots.verifier / "auth-gateway").exists()


@pytest.fixture
def controlled_gateway(tmp_path):
    bun = shutil.which("bun")
    if not bun:
        pytest.skip("Bun is required for the native gateway admission-loop test")
    auth = tmp_path / "auth.ts"
    auth.write_text("""
export class SqliteAuthCredentialStore { async upsertAuthCredential() {} }
export class AuthStorage { credentials = { reload: async () => {} }; close() {} }
""")
    router = tmp_path / "router.ts"
    router.write_text("""
export function createAuthGatewayRouter() {
  return { route: async () => new Response('controlled-test-boundary'), close() {} };
}
""")
    catalog = tmp_path / "catalog.ts"
    catalog.write_text("export const getBundledModel = () => ({ id: 'gpt-6-luna' });")
    database = tmp_path / "fake-credential.db"
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE auth_credentials (id INTEGER, provider TEXT, "
            "credential_type TEXT, data TEXT, disabled_cause TEXT)"
        )
        conn.execute(
            "INSERT INTO auth_credentials VALUES(1, 'openai-codex', 'api_key', ?, NULL)",
            (json.dumps({"key": "TEST_DOUBLE_NOT_A_PROVIDER_CREDENTIAL"}),),
        )
    script = tmp_path / "gateway.ts"
    script.write_text(GATEWAY_SOURCE)
    ready, summary = tmp_path / "ready.json", tmp_path / "summary.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "authModule": str(auth),
                "gatewayModule": str(router),
                "catalogModule": str(catalog),
                "credentialStore": str(database),
                "model": "openai-codex/gpt-6-luna",
                "capabilities": {"solver-a": "test-local-a", "solver-b": "test-local-b"},
                "ready": str(ready),
                "summary": str(summary),
            }
        )
    )
    proc = subprocess.Popen(
        [bun, str(script), str(config)],
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "controlled gateway did not initialize"
        yield f"http://127.0.0.1:{json.loads(ready.read_text())['port']}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def request_status(base, path, *, capability="test-local-a", body=None):
    request = urllib.request.Request(
        base + path,
        data=body
        if isinstance(body, bytes)
        else json.dumps(body).encode()
        if body is not None
        else None,
        headers={"Authorization": "Bearer " + capability, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_gateway_scope_and_byte_cap(controlled_gateway):
    base = controlled_gateway
    assert request_status(base, "/healthz") == 200
    assert request_status(base, "/healthz", capability="wrong") == 401
    assert request_status(base, "/v1/credentials/check") == 403
    assert request_status(base, "/v1/pi/stream", body={"modelId": "other-model"}) == 403
    # Bun may reject from Content-Length while urllib is still writing the body,
    # closing the socket before an HTTP 413 can be read. Never drain an oversize
    # request merely to make the client observe a particular status code.
    try:
        assert request_status(base, "/v1/pi/stream", body=b"x" * 2_000_001) == 413
    except (ConnectionResetError, BrokenPipeError):
        pass
    except urllib.error.URLError as exc:
        assert isinstance(exc.reason, (ConnectionResetError, BrokenPipeError))
    assert request_status(base, "/healthz") == 200
    assert request_status(base, "/v1/pi/stream", body={"modelId": "openai-codex/gpt-6-luna"}) == 200


def test_concurrent_requests_cannot_exceed_per_actor_allowance(controlled_gateway):
    def dispatch(_index):
        return request_status(
            controlled_gateway, "/v1/pi/stream", body={"modelId": "openai-codex/gpt-6-luna"}
        )

    with ThreadPoolExecutor(max_workers=80) as pool:
        statuses = list(pool.map(dispatch, range(80)))
    assert statuses.count(200) == 64
    assert statuses.count(429) == 16
    assert (
        request_status(
            controlled_gateway,
            "/v1/pi/stream",
            capability="test-local-b",
            body={"modelId": "openai-codex/gpt-6-luna"},
        )
        == 200
    )
