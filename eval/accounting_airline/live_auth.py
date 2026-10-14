"""Supervised native OMP auth gateway; real credentials remain read-only and in RAM.

Uses the installed OMP gateway implementation, not a provider protocol reimplementation.
Each actor has a separate gateway router/cache and short-lived local capability. Only the
selected model's pi-native stream endpoint is admitted; no credential/status/admin routes.
"""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import BinaryIO

from .domain_lane import write_private
from .roots import LabRoots, runtime_env
from .sandbox import SandboxProfile, account_home, sbpl_quote, wrap_argv

GATEWAY_SOURCE = r"""
import { Database } from "bun:sqlite";
const config = await Bun.file(process.argv[2]).json();
const { AuthStorage, SqliteAuthCredentialStore } = await import(config.authModule);
const { createAuthGatewayRouter } = await import(config.gatewayModule);
const { getBundledModel } = await import(config.catalogModule);
const [provider, id] = config.model.split(/\/(.*)/s);
const model = getBundledModel(provider, id);
if (!model) throw new Error("selected model absent from installed native catalog");
const source = new Database(config.credentialStore, { readonly: true, create: false });
const rows = source.query("SELECT credential_type, data FROM auth_credentials " +
  "WHERE provider = ? AND disabled_cause IS NULL ORDER BY id").all(provider);
source.close();
if (rows.length !== 1) throw new Error("expected one active credential for selected provider");
const data = JSON.parse(rows[0].data);
if (rows[0].credential_type === "oauth" && !(data.expires > Date.now() + 3600000))
  throw new Error("OAuth validity shorter than bounded gateway lifetime; refresh is forbidden");
const memory = new SqliteAuthCredentialStore(new Database(":memory:"));
await memory.upsertAuthCredential(provider, { type: rows[0].credential_type, ...data });
const storage = new AuthStorage(memory, {
  refreshOAuthCredential: async () => {
    throw new Error("credential refresh forbidden in live lab");
  },
});
await storage.credentials.reload();
const routers = new Map();
const counts = {};
for (const actor of Object.keys(config.capabilities)) {
  routers.set(actor, createAuthGatewayRouter({ storage,
    resolveModel: name => name === config.model ? model : undefined,
    listModels: () => [model] }));
  counts[actor] = 0;
}
const started = Date.now();
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, idleTimeout: 255,
  maxRequestBodySize: 2000000,
  async fetch(req) {
    const actor = Object.keys(config.capabilities).find(a =>
      req.headers.get("Authorization") === "Bearer " + config.capabilities[a]);
    if (!actor) return new Response("unauthorized", { status: 401 });
    const path = new URL(req.url).pathname;
    if (req.method === "GET" && path === "/healthz") return Response.json({ ready: true });
    if (req.method === "GET" && path === "/v1/models")
      return routers.get(actor).route(req, actor);
    if (req.method !== "POST" || path !== "/v1/pi/stream")
      return new Response("route forbidden", { status: 403 });
    if (Date.now() - started > 3600000 || counts[actor] >= 64)
      return new Response("bounded gateway exhausted", { status: 429 });
    const text = await req.text();
    if (text.length > 2000000) return new Response("body too large", { status: 413 });
    let body;
    try { body = JSON.parse(text); } catch { return new Response("invalid JSON", { status: 400 }); }
    if (body.modelId !== config.model) return new Response("model forbidden", { status: 403 });
    // Recheck after the await, then reserve synchronously before provider dispatch.
    if (Date.now() - started > 3600000 || counts[actor] >= 64)
      return new Response("bounded gateway exhausted", { status: 429 });
    counts[actor]++;
    const request = new Request(req.url, { method: "POST", headers: req.headers, body: text,
      signal: AbortSignal.any([req.signal, AbortSignal.timeout(610000)]) });
    return routers.get(actor).route(request, actor);
  }
});
await Bun.write(config.ready,
  JSON.stringify({ port: server.port, model: config.model, pid: process.pid }));
const close = async () => {
  await Bun.write(config.summary, JSON.stringify({ model: config.model, requests: counts,
    elapsedMs: Date.now() - started, credentialWrites: false, refreshEnabled: false }));
  for (const router of routers.values()) router.close();
  storage.close(); server.stop(true); process.exit(0);
};
process.on("SIGTERM", close);
setTimeout(close, 3600000);
"""


def catalog_metadata(model: str) -> dict:
    """Read the installed selected model's public settings; never expose account metadata."""
    import hashlib

    provider, model_id = model.split("/", 1)
    path = account_home() / ".bun/install/global/node_modules/@oh-my-pi/pi-catalog/src/models.json"
    payload = path.read_bytes()
    row = json.loads(payload)[provider][model_id]
    keys = (
        "id",
        "name",
        "api",
        "provider",
        "cost",
        "thinking",
        "contextWindow",
        "maxContextWindow",
        "maxTokens",
        "serviceTierCost",
        "omitMaxOutputTokens",
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "model": {key: row[key] for key in keys if key in row},
        "cost_provenance": "Installed runtime catalog USD per million tokens, not provider billing",
    }


class AuthGateway:
    def __init__(self, roots: LabRoots, model: str):
        if not model.startswith("openai-codex/") or len(model.split("/")) != 2:
            raise ValueError(
                "live auth gateway currently verifies only explicit openai-codex models"
            )
        self.roots, self.model = roots, model
        self.capabilities = {actor: secrets.token_hex(32) for actor in ("solver-a", "solver-b")}
        self.directory = roots.verifier / "auth-gateway"
        self.directory.mkdir(mode=0o700)
        self.process: subprocess.Popen[bytes] | None = None
        self.port = 0
        self.log: BinaryIO | None = None

    def __enter__(self) -> AuthGateway:
        packages = account_home() / ".bun/install/global/node_modules/@oh-my-pi"
        modules = {
            "authModule": packages / "pi-ai/src/auth-storage.ts",
            "gatewayModule": packages / "pi-ai/src/auth-gateway/server.ts",
            "catalogModule": packages / "pi-catalog/src/models.ts",
        }
        for path in modules.values():
            if not path.is_file():
                raise RuntimeError(f"installed native gateway module prerequisite missing: {path}")
        config = self.directory / "config.json"
        ready = self.directory / "ready.json"
        script = self.directory / "gateway.ts"
        script.write_text(GATEWAY_SOURCE)
        config_data = {
            **{k: str(v) for k, v in modules.items()},
            "model": self.model,
            "credentialStore": str(account_home() / ".omp/agent/agent.db"),
            "capabilities": self.capabilities,
            "ready": str(ready),
            "summary": str(self.directory / "summary.json"),
        }
        write_private(config, config_data)
        env = runtime_env(self.roots, "control")
        env["TMPDIR"] = str(self.directory)
        profile = SandboxProfile(
            "(version 1)\n(allow default)\n(deny file-write*)\n"
            f"(allow file-write* (subpath {sbpl_quote(self.roots.verifier)}))\n"
            '(allow file-write* (literal "/dev/null"))\n',
            (),
            (),
        )
        command = wrap_argv(profile, [shutil.which("bun") or "bun", str(script), str(config)])
        self.log = (self.directory / "stderr.txt").open("wb")
        self.process = subprocess.Popen(
            command,
            env=env,
            cwd=self.directory,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
        )
        deadline = time.monotonic() + 45
        while not ready.exists() and self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if not ready.exists():
            self.__exit__(None, None, None)
            raise RuntimeError(
                "readonly native auth gateway initialization failed; "
                f"diagnostics={self.directory / 'stderr.txt'}"
            )
        self.port = int(json.loads(ready.read_text())["port"])
        try:
            self.prove_boundary()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def prove_boundary(self) -> None:
        observations = []
        for path, capability, body, expected in (
            ("/healthz", self.capabilities["solver-a"], None, 200),
            ("/healthz", "wrong-local-capability", None, 401),
            ("/v1/credentials/check", self.capabilities["solver-b"], None, 403),
            ("/v1/pi/stream", self.capabilities["solver-b"], {"modelId": "not-authorized"}, 403),
        ):
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}{path}",
                data=json.dumps(body).encode() if body is not None else None,
                headers={
                    "Authorization": f"Bearer {capability}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
            observations.append({"path": path, "status": status, "expected": expected})
        write_private(
            self.roots.artifacts / "auth-boundary.json",
            {
                "probes": observations,
                "real_store": "SQLite readonly open; in-memory credential working state only",
                "actor_router_caches": "distinct",
                "refresh": "disabled",
            },
        )
        if any(o["status"] != o["expected"] for o in observations):
            raise RuntimeError("native auth gateway boundary probe did not pass")

    def configure(self, actor: str, env: dict[str, str]) -> None:
        agent = self.roots.actor_home(actor) / ".omp/agent"
        agent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # models.yml accepts JSON (a YAML subset). This is a generated, run-only
        # transport override; the capability cannot access arbitrary models/routes.
        write_private(
            agent / "models.yml",
            {
                "providers": {
                    "openai-codex": {
                        "baseUrl": f"http://127.0.0.1:{self.port}",
                        "transport": "pi-native",
                        "apiKey": self.capabilities[actor],
                    }
                }
            },
        )
        env["PI_CODING_AGENT_DIR"] = str(agent)
        env["OMP_AGENT_DIR"] = str(agent)

    def __exit__(self, *_args) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()
