"""Run a REAL ``otel/opentelemetry-collector-contrib`` container as a run-owned, loopback-only lab.

Ownership/teardown follows the removed airline lab's run-owned pattern (copied idea, not imported):

* one ``run_id`` per lab run; every container carries ``--label ssgrep-lab-run=<run id>`` and
  ``--label ssgrep-lab=otel``; the name is ``ssgrep-otel-<run id>[-<tag>]``;
* the OTLP port is published ONLY as ``127.0.0.1::4318`` (random host port, read back with
  ``docker port``); never ``--privileged``, never ``--network host``;
* the only host paths mounted are inside ONE temporary directory this run creates: the config
  (read-only) and an output directory (read-write); the container runs as the CURRENT
  ``uid:gid`` so the file exporter's output is readable and no root-owned file appears;
* ``stop``/``restart``/``rm`` first inspect the container's labels and refuse any container that
  does not carry THIS run's labels (so unrelated containers can never be touched); every such
  action is recorded in ``Collector.actions`` for the run evidence.

The collector config is ``docs/examples/otel-collector-reference.yaml`` itself (single source of
truth); enrolled synthetic tokens reach it through the container environment only.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from tests.accounting.lab_receiver import LabCert

IMAGE = "otel/opentelemetry-collector-contrib:0.137.0"
RUN_LABEL = "ssgrep-lab-run"
KIND_LABEL = "ssgrep-lab"
KIND_VALUE = "otel"
INTERNAL_PORT = 4318
REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "docs" / "examples" / "otel-collector-reference.yaml"
)

# Synthetic enrolled credentials and the identity the reference config maps them to.
TOKEN_A = "lab-ingest-token-a-7f3a91c2"
TOKEN_B = "lab-ingest-token-b-52d0e4b8"
IDENTITY = {TOKEN_A: ("tenant-a", "device-a"), TOKEN_B: ("tenant-b", "device-b")}

_TLS_MARKER = (
    "        # tls:\n"
    "        #   cert_file: /certs/server.pem\n"
    "        #   key_file: /certs/server.key\n"
)
_TLS_ENABLED = (
    "        tls:\n          cert_file: /certs/server.pem\n          key_file: /certs/server.key\n"
)


class LabError(RuntimeError):
    """The lab could not do what it was asked (never silently ignored)."""


class OwnershipError(LabError):
    """Refused: the container does not carry this run's ownership labels."""


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def docker(*args: str, timeout: float = 90, check: bool = True) -> str:
    """The only place the lab spawns the docker CLI."""
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
    )
    if check and result.returncode != 0:
        raise LabError(f"docker {args[0]} failed: {result.stderr.strip()[:400]}")
    return result.stdout.strip()


def unavailable_reason() -> str | None:
    """None when docker and the pinned image are usable, else the named reason they are not."""
    if shutil.which("docker") is None:
        return "docker CLI not found on PATH"
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"docker daemon unreachable: {type(error).__name__}"
    if probe.returncode != 0:
        return f"image {IMAGE} is not present locally (the lab never pulls or uses :latest)"
    return None


def labelled(run_id: str) -> list[str]:
    """Full ids of every container (running or not) that carries this run's label."""
    return docker("ps", "-a", "-q", "--no-trunc", "--filter", f"label={RUN_LABEL}={run_id}").split()


def all_container_ids() -> set[str]:
    return set(docker("ps", "-a", "-q", "--no-trunc").split())


def reference_config(*, tls: bool = False) -> str:
    text = REFERENCE_CONFIG.read_text()
    if not tls:
        return text
    if _TLS_MARKER not in text:
        raise LabError("reference config no longer contains the documented TLS block")
    return text.replace(_TLS_MARKER, _TLS_ENABLED)


@dataclass
class Collector:
    """One run-owned collector container plus the temp dir holding its config and output."""

    run_id: str
    tag: str
    work: Path
    container_id: str = ""
    port: int = 0
    tls: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    cert: LabCert | None = None

    # ---- identity ---------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"ssgrep-otel-{self.run_id}-{self.tag}"

    @property
    def base_url(self) -> str:
        return f"{'https' if self.tls else 'http'}://127.0.0.1:{self.port}"

    @property
    def certs_dir(self) -> Path:
        return self.work / "certs"

    @property
    def out_dir(self) -> Path:
        return self.work / "out"

    def labels(self) -> dict[str, str]:
        info = json.loads(docker("inspect", self.container_id or self.name))[0]
        return info.get("Config", {}).get("Labels") or {}

    def _require_owned(self, verb: str) -> str:
        """Inspect labels BEFORE a state-changing docker verb; refuse anything not ours."""
        if not self.container_id:
            raise OwnershipError("no container was started by this lab instance")
        info = json.loads(docker("inspect", self.container_id))[0]
        labels = info.get("Config", {}).get("Labels") or {}
        owned = (
            info.get("Id") == self.container_id
            and labels.get(RUN_LABEL) == self.run_id
            and labels.get(KIND_LABEL) == KIND_VALUE
            and info.get("Name") == "/" + self.name
        )
        self.actions.append(
            {
                "verb": verb,
                "container_id": info.get("Id"),
                "name": info.get("Name"),
                "run_label": labels.get(RUN_LABEL),
                "kind_label": labels.get(KIND_LABEL),
                "owned": owned,
            }
        )
        if not owned:
            raise OwnershipError(f"refusing docker {verb}: labels do not match run {self.run_id}")
        return self.container_id

    # ---- lifecycle --------------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        run_id: str,
        tag: str = "main",
        *,
        tls: bool = False,
        timeout: float = 45,
    ) -> Collector:
        work = Path(tempfile.mkdtemp(prefix="ssgrep-otel-"))
        self = cls(run_id, tag, work, tls=tls)
        try:
            self._create()
            self._wait_ready(timeout)
        except BaseException:
            self.stop()
            raise
        return self

    def _create(self) -> None:
        self.work.chmod(0o700)
        self.out_dir.mkdir(mode=0o700)
        if self.tls:
            # server.pem/server.key; the container runs as THIS uid so the 0600 key is readable
            self.cert = LabCert.generate(self.certs_dir, name="server")
        config = self.work / "config.yaml"
        config.write_text(reference_config(tls=self.tls))
        config.chmod(0o644)
        args = [
            "run", "-d",
            "--name", self.name,
            "--label", f"{RUN_LABEL}={self.run_id}",
            "--label", f"{KIND_LABEL}={KIND_VALUE}",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-p", f"127.0.0.1::{INTERNAL_PORT}",
            "-e", f"INGEST_TOKEN_A={TOKEN_A}",
            "-e", f"INGEST_TOKEN_B={TOKEN_B}",
            "-v", f"{config}:/etc/otelcol-contrib/config.yaml:ro",
            "-v", f"{self.out_dir}:/out:rw",
        ]  # fmt: skip
        if self.tls:
            args += ["-v", f"{self.certs_dir}:/certs:ro"]
        args.append(IMAGE)
        self.container_id = docker(*args)
        self._read_port()

    def _read_port(self) -> None:
        line = docker("port", self.container_id, f"{INTERNAL_PORT}/tcp").splitlines()[0]
        host, _, port = line.rpartition(":")
        if host != "127.0.0.1":
            raise LabError(f"collector published on {host!r}, expected loopback only")
        self.port = int(port)

    def _wait_ready(self, timeout: float) -> None:
        """Ready == the receiver answers HTTP at all (an unauthenticated POST gets 401)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if docker("inspect", self.container_id, "--format", "{{.State.Running}}") != "true":
                raise LabError("collector exited during start:\n" + self.logs()[-1500:])
            if self._answers():
                return
            time.sleep(0.2)
        raise LabError("collector never answered HTTP:\n" + self.logs()[-1500:])

    def _answers(self) -> bool:
        request = urllib.request.Request(  # noqa: S310 (loopback)
            f"{self.base_url}/v1/metrics", data=b"{}", method="POST"
        )
        context = self.cert.client_context() if self.cert is not None else None
        try:
            urllib.request.urlopen(request, timeout=1, context=context)  # noqa: S310
        except urllib.error.HTTPError:
            return True
        except (OSError, TimeoutError):
            return False
        return True

    def restart(self, timeout: float = 45) -> None:
        """``docker restart`` of THIS run's container; the published port is re-read."""
        container = self._require_owned("restart")
        docker("restart", "-t", "2", container)
        self._read_port()
        self._wait_ready(timeout)

    def halt(self) -> None:
        """``docker stop`` of THIS run's container (labels verified first); data dir is kept."""
        docker("stop", "-t", "2", self._require_owned("stop"))

    def resume(self, timeout: float = 45) -> None:
        """``docker start`` of THIS run's container; the re-allocated published port is re-read."""
        docker("start", self._require_owned("start"))
        self._read_port()
        self._wait_ready(timeout)

    def stop(self) -> None:
        """Remove ONLY this run's container (labels verified first) and the temp dir."""
        try:
            if self.container_id and docker(
                "ps", "-a", "-q", "--no-trunc", "--filter", f"id={self.container_id}"
            ):
                container = self._require_owned("rm")
                docker("rm", "-f", container, check=False)
        finally:
            shutil.rmtree(self.work, ignore_errors=True)

    def logs(self) -> str:
        if not self.container_id:
            return ""
        result = subprocess.run(
            ["docker", "logs", self.container_id], capture_output=True, text=True, check=False
        )
        return result.stdout + result.stderr

    def describe(self) -> dict[str, Any]:
        """Facts for the run evidence: image, labels, user, ports, mounts, privilege flags."""
        info = json.loads(docker("inspect", self.container_id))[0]
        image = json.loads(docker("image", "inspect", IMAGE))[0]
        host = info["HostConfig"]
        return {
            "image": IMAGE,
            "image_id": image["Id"],
            "repo_digests": image.get("RepoDigests"),
            "container_id": info["Id"],
            "name": info["Name"],
            "labels": info["Config"]["Labels"],
            "user": info["Config"]["User"],
            "ports": docker("port", self.container_id),
            "port_bindings": host["PortBindings"],
            "privileged": host["Privileged"],
            "network_mode": host["NetworkMode"],
            "mounts": [
                {"source": m["Source"], "destination": m["Destination"], "rw": m["RW"]}
                for m in info["Mounts"]
            ],
        }

    # ---- output -----------------------------------------------------------------------------

    def records(self, signal: str) -> list[dict[str, Any]]:
        """Every JSON line the file exporter wrote for ``metrics`` or ``logs`` (floats exact)."""
        path = self.out_dir / f"{signal}.jsonl"
        if not path.exists():
            return []
        text = path.read_text()
        complete = text[: text.rfind("\n") + 1]  # a line the exporter is still writing is skipped
        return [json.loads(line, parse_float=Decimal) for line in complete.splitlines() if line]

    def count(self, signal: str) -> int:
        return len(self.records(signal))

    def wait_records(
        self, signal: str, count: int, *, since: int = 0, timeout: float = 15
    ) -> list[dict[str, Any]]:
        """Poll until ``count`` documents exist after the first ``since``; return exactly those."""

        def probe() -> list[dict[str, Any]] | None:
            documents = self.records(signal)[since:]
            return documents if len(documents) >= count else None

        return wait_until(probe, f"{count} {signal} documents from the file exporter", timeout)


class Forwarder:
    """A stable loopback TCP port in front of a collector whose published port changes.

    ``docker restart`` re-allocates the random host port; the delivery journey needs ONE
    endpoint across the restart (the destination endpoint is policy). Bytes are relayed
    unchanged, so the collector sees exactly what the client sent; while the collector is down
    the connection is closed (a real connection failure for the client).
    """

    def __init__(self, target: Callable[[], int]) -> None:
        self._target = target
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._streams: list[bytearray] = []
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _relay(self, client: socket.socket) -> None:
        try:
            upstream = socket.create_connection(("127.0.0.1", self._target()), timeout=3)
        except OSError:
            client.close()
            return
        stream = bytearray()
        with self._lock:
            self._streams.append(stream)
        threading.Thread(target=self._pump, args=(upstream, client), daemon=True).start()
        self._pump(client, upstream, stream)

    @staticmethod
    def _pump(source: socket.socket, sink: socket.socket, record: bytearray | None = None) -> None:
        try:
            while data := source.recv(65536):
                if record is not None:
                    record.extend(data)
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (source, sink):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

    def request_bodies(self) -> list[tuple[str, bytes]]:
        """(request path, body bytes) of every complete request that crossed the forwarder."""
        with self._lock:
            streams = [bytes(stream) for stream in self._streams]
        out = []
        for raw in streams:
            head, sep, rest = raw.partition(b"\r\n\r\n")
            if not sep:
                continue
            lines = head.decode("latin-1").split("\r\n")
            length = next(
                (
                    int(x.split(":", 1)[1])
                    for x in lines[1:]
                    if x.lower().startswith("content-length:")
                ),
                -1,
            )
            if length >= 0 and len(rest) >= length:
                out.append((lines[0].split()[1], rest[:length]))
        return out

    def close(self) -> None:
        self._stop.set()
        self._listener.close()


def wait_until(probe: Callable[[], Any], what: str, timeout: float = 15) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = probe()
        if value is not None:
            return value
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(0.1)


# ---- reading the collector's own output ------------------------------------------------------


def attr_map(attributes: list[dict[str, Any]] | None) -> dict[str, Any]:
    """OTLP attribute list -> {key: python value}; ``intValue`` strings become ints."""
    out: dict[str, Any] = {}
    for item in attributes or []:
        ((kind, value),) = item["value"].items()
        out[item["key"]] = int(value) if kind == "intValue" else value
    return out


def metric_points(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one exported metrics document to comparable rows (resource, metric, point)."""
    rows = []
    for resource_metrics in doc.get("resourceMetrics", []):
        resource = attr_map(resource_metrics.get("resource", {}).get("attributes"))
        for scope_metrics in resource_metrics.get("scopeMetrics", []):
            for metric in scope_metrics.get("metrics", []):
                for kind in ("sum", "gauge", "histogram"):
                    body = metric.get(kind)
                    if body is None:
                        continue
                    for point in body.get("dataPoints", []):
                        rows.append(
                            {
                                "name": metric["name"],
                                "unit": metric.get("unit"),
                                "kind": kind,
                                "temporality": body.get("aggregationTemporality"),
                                "monotonic": body.get("isMonotonic"),
                                "attributes": attr_map(point.get("attributes")),
                                "resource": resource,
                                "point": point,
                            }
                        )
    return rows


def log_records(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one exported logs document to rows with resource, attributes and body text."""
    rows = []
    for resource_logs in doc.get("resourceLogs", []):
        resource = attr_map(resource_logs.get("resource", {}).get("attributes"))
        for scope_logs in resource_logs.get("scopeLogs", []):
            for record in scope_logs.get("logRecords", []):
                rows.append(
                    {
                        "resource": resource,
                        "attributes": attr_map(record.get("attributes")),
                        "body": record["body"]["stringValue"],
                        "record": record,
                    }
                )
    return rows
