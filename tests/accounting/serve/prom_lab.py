"""Run a REAL ``prom/prometheus`` container (loopback-only, run-labelled) against live servers.

Only containers this module creates are ever touched: each carries the label
``ssgrep-lab-run=<run id>`` and is removed by its own id. The API port is published on
``127.0.0.1`` only; Prometheus reaches the host's loopback listener through
the Docker host gateway: the container's ``localhost`` is ALSO mapped to the gateway, so the
scraper naturally sends the approved ``Host: localhost:<port>`` (Prometheus forbids overriding
Host) while the connection falls through the refused in-container 127.0.0.1 to the host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

IMAGE = "prom/prometheus:latest"
LABEL = "ssgrep-lab-run"


def docker_available() -> str | None:
    """None when docker and the image are usable, else the named reason it is not."""
    if shutil.which("docker") is None:
        return "docker binary not found"
    probe = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=60)
    if probe.returncode != 0:
        return f"image {IMAGE} not present (docker pull {IMAGE})"
    return None


def _docker(*args: str, check: bool = True) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def job_config(job: str, port: int, token_name: str | None) -> str:
    auth = (
        f"    authorization:\n      type: Bearer\n      credentials_file: /tmp/{token_name}\n"
        if token_name
        else ""
    )
    return (
        f"  - job_name: {job}\n"
        f"    metrics_path: /metrics\n"
        f"    static_configs:\n      - targets: ['localhost:{port}']\n"
        f"{auth}"
    )


class Prometheus:
    def __init__(self, base: str, container: str) -> None:
        self.base = base
        self.container = container

    def query(self, expression: str) -> list[dict[str, Any]]:
        url = f"{self.base}/api/v1/query?" + urllib.parse.urlencode({"query": expression})
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 (loopback only)
            document = json.load(response)
        assert document["status"] == "success", document
        return document["data"]["result"]

    def scalar(self, expression: str) -> float:
        (row,) = self.query(expression)
        return float(row["value"][1])

    def targets(self) -> dict[str, dict[str, Any]]:
        with urllib.request.urlopen(f"{self.base}/api/v1/targets", timeout=10) as response:  # noqa: S310
            active = json.load(response)["data"]["activeTargets"]
        return {t["labels"]["job"]: t for t in active}

    def wait_scraped(self, jobs: list[str], timeout: float = 45.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                have = {r["metric"]["job"] for r in self.query("up")}
            except OSError:
                have = set()
            if set(jobs) <= have:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Prometheus never scraped {jobs}")


@contextmanager
def prometheus(
    run_id: str, jobs: Sequence[tuple[str, int, str | None]], tokens: dict[str, str]
) -> Iterator[Prometheus]:
    """Start Prometheus scraping ``jobs`` = (name, port, token file name | None); always cleaned."""
    work = Path(tempfile.mkdtemp(prefix="ssgrep-prom-"))
    container = ""
    try:
        for name, value in tokens.items():
            (work / name).write_text(value)
            (work / name).chmod(0o644)  # a throwaway copy of a synthetic token for the container
        config = (
            "global:\n  scrape_interval: 1s\n  scrape_timeout: 1s\nscrape_configs:\n"
            + "".join(job_config(job, port, token) for job, port, token in jobs)
        )
        (work / "prom.yml").write_text(config)
        (work / "prom.yml").chmod(0o644)
        container = _docker(
            "create",
            "--label",
            f"{LABEL}={run_id}",
            "-p",
            "127.0.0.1::9090",
            "--add-host",
            "localhost:host-gateway",
            IMAGE,
            "--config.file=/tmp/prom.yml",
            "--storage.tsdb.path=/tmp/tsdb",
            "--web.listen-address=0.0.0.0:9090",
        )
        _docker("cp", str(work / "prom.yml"), f"{container}:/tmp/prom.yml")
        for name in tokens:
            _docker("cp", str(work / name), f"{container}:/tmp/{name}")
        _docker("start", container)
        port = _docker("port", container, "9090/tcp").splitlines()[0].rsplit(":", 1)[1]
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(f"{base}/-/ready", timeout=2).read()  # noqa: S310
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.3)
        yield Prometheus(base, container)
    finally:
        if container:
            _docker("rm", "-f", container, check=False)
        shutil.rmtree(work, ignore_errors=True)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def leftovers(run_id: str) -> list[str]:
    return _docker("ps", "-a", "-q", "--filter", f"label={LABEL}={run_id}").split()
