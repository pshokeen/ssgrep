"""Deterministic loopback lab receiver (and lab proxy) for export transport tests.

Real sockets only: a ``ThreadingHTTPServer`` on ``127.0.0.1`` port 0, optional TLS with
a throwaway self-signed certificate made by the ``openssl`` CLI. Nothing is logged.

API
---
``LabReceiver.start(tls=False, cert=None) -> LabReceiver`` (also a context manager;
``.stop()`` closes the socket and joins the server thread).

* ``.url`` / ``.url_for(path)`` / ``.host`` / ``.port`` / ``.tls``
* ``.queue(*behaviors, path=None)``: FIFO of scripted behaviors, for one request
  path (``path="/v1/metrics"``) or globally (consulted after the path queue). Empty
  queue means ``Respond(200, b"{}")``.
* Behaviors (frozen dataclasses): ``Respond(status, body, headers, delay)``,
  ``Partial(rejected, kind, message, as_string)`` (200 with ``partialSuccess``),
  ``Redirect(location, status)``, ``Hang(seconds)`` (read the request then say
  nothing), ``DropAfterRead()`` (read the request, close without responding),
  ``Truncate(declared, sent)`` (close mid body), ``Reset()`` (TCP RST after the
  request), ``Oversize(nbytes, mode)`` (``content-length``/``chunked``/``stream``),
  ``SlowDrip(interval, body_only)`` (one response byte per ``interval`` seconds).
* Recording (thread safe): ``.requests`` (copy of ``Recorded(method, path, headers,
  body, sha256, tls, status)``; headers are lowercase-keyed), ``.connections``
  (accepted TCP connections even when no request parsed), ``.tls_failures``
  (handshakes that did not complete), ``.bytes_sent`` (bytes written by Oversize and
  SlowDrip), ``.accepted_bodies()`` (bodies answered 200), ``.wait_for_requests``.
* ``LabCert.generate(directory, san=...)``: ``.cert_file`` (also the CA bundle),
  ``.client_context()``.
* ``ProxyLab.start(relay_to=None)``: records CONNECT / absolute-URI requests and
  answers 502 (nothing proxied) unless ``relay_to=(host, port)`` is given, in which
  case CONNECT is answered 200 and bytes are relayed to that address.
"""

from __future__ import annotations

import hashlib
import json
import select
import shutil
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, Self

MAX_LAB_BODY = 16 * 1024 * 1024
CHUNK = 8192


@dataclass(frozen=True)
class Respond:
    status: int = 200
    body: bytes = b"{}"
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


@dataclass(frozen=True)
class Partial:
    rejected: int
    kind: Literal["metrics", "logs"] = "metrics"
    message: str = ""
    as_string: bool = True


@dataclass(frozen=True)
class Redirect:
    location: str
    status: int = 307


@dataclass(frozen=True)
class Hang:
    seconds: float = 30.0


@dataclass(frozen=True)
class DropAfterRead:
    pass


@dataclass(frozen=True)
class Truncate:
    declared: int = 100
    sent: int = 7


@dataclass(frozen=True)
class Reset:
    pass


@dataclass(frozen=True)
class Oversize:
    nbytes: int
    mode: Literal["content-length", "chunked", "stream"] = "content-length"


@dataclass(frozen=True)
class SlowDrip:
    interval: float = 0.3
    body_only: bool = False


Behavior = (
    Respond | Partial | Redirect | Hang | DropAfterRead | Truncate | Reset | Oversize | SlowDrip
)


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    sha256: str
    tls: bool
    status: int | None = None


@dataclass(frozen=True)
class LabCert:
    cert_file: Path
    key_file: Path

    @classmethod
    def generate(
        cls, directory: Path, san: str = "IP:127.0.0.1,DNS:localhost", name: str = "lab"
    ) -> LabCert:
        directory.mkdir(parents=True, exist_ok=True)
        cert, key = directory / f"{name}.pem", directory / f"{name}.key"
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt",
                "ec_paramgen_curve:prime256v1", "-nodes", "-days", "2",
                "-subj", "/CN=lab-receiver", "-keyout", str(key), "-out", str(cert),
                "-addext", f"subjectAltName={san}",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,digitalSignature,keyCertSign",
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
        return cls(cert, key)

    def client_context(self) -> ssl.SSLContext:
        """A verifying client context that trusts exactly this certificate."""
        return ssl.create_default_context(cafile=str(self.cert_file))

    def server_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(self.cert_file), str(self.key_file))
        return context


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    lab: LabReceiver

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        request = super().get_request()
        self.lab._connection_accepted()
        return request

    def handle_error(self, request: object, client_address: object) -> None:
        return None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server
    _dead = False

    def log_message(self, format: str, *args: object) -> None:
        return None

    def setup(self) -> None:
        lab = self.server.lab
        if lab._server_context is not None:
            try:
                self.request = lab._server_context.wrap_socket(self.request, server_side=True)
            except (OSError, ssl.SSLError):
                lab._tls_failed()
                self._dead = True
        super().setup()

    def handle(self) -> None:
        self.close_connection = True
        if self._dead:
            return
        try:
            self.handle_one_request()
        except OSError:
            return

    def _serve(self) -> None:
        lab = self.server.lab
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(min(length, MAX_LAB_BODY)) if length > 0 else b""
        headers = {key.lower(): value for key, value in self.headers.items()}
        record = Recorded(
            self.command, self.path, headers, body, hashlib.sha256(body).hexdigest(),
            lab._server_context is not None,
        )  # fmt: skip
        lab._record(record)
        behavior = lab._next_behavior(self.path, self.command)
        try:
            record.status = self._perform(behavior)
        except OSError:
            return

    do_GET = do_POST = do_PUT = do_CONNECT = _serve  # noqa: N815

    def _head(self, status: int, headers: dict[str, str]) -> None:
        self.send_response_only(status)
        self.send_header("Connection", "close")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()

    def _perform(self, behavior: Behavior) -> int | None:
        lab = self.server.lab
        match behavior:
            case Respond(status, body, headers, delay):
                if delay:
                    lab._stop.wait(delay)
                merged = {"Content-Length": str(len(body)), **headers}
                self._head(status, merged)
                self.wfile.write(body)
                return status
            case Partial(rejected, kind, message, as_string):
                field_name = "rejectedDataPoints" if kind == "metrics" else "rejectedLogRecords"
                partial: dict[str, object] = {field_name: str(rejected) if as_string else rejected}
                if message:
                    partial["errorMessage"] = message
                payload = json.dumps({"partialSuccess": partial}).encode()
                headers = {"Content-Length": str(len(payload)), "Content-Type": "application/json"}
                self._head(200, headers)
                self.wfile.write(payload)
                return 200
            case Redirect(location, status):
                self._head(status, {"Location": location, "Content-Length": "0"})
                return status
            case Hang(seconds):
                lab._stop.wait(seconds)
            case DropAfterRead():
                pass
            case Truncate(declared, sent):
                self._head(200, {"Content-Length": str(declared)})
                self.wfile.write(b"x" * sent)
            case Reset():
                self.request.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            case Oversize(nbytes, mode):
                self._oversize(nbytes, mode)
            case SlowDrip(interval, body_only):
                self._drip(interval, body_only)
        return None

    def _write_counted(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()
        self.server.lab._add_sent(len(data))

    def _oversize(self, nbytes: int, mode: str) -> None:
        lab = self.server.lab
        if mode == "chunked":
            self._head(200, {"Transfer-Encoding": "chunked"})
        elif mode == "content-length":
            self._head(200, {"Content-Length": str(nbytes)})
        else:
            self._head(200, {})
        piece = b"x" * CHUNK
        remaining = nbytes
        while remaining > 0 and not lab._stop.is_set():
            part = piece[: min(CHUNK, remaining)]
            if mode == "chunked":
                self._write_counted(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            else:
                self._write_counted(part)
            remaining -= len(part)
        if mode == "chunked":
            self._write_counted(b"0\r\n\r\n")

    def _drip(self, interval: float, body_only: bool) -> None:
        lab = self.server.lab
        payload = b'{"partialSuccess":{}}'
        head = f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
        if body_only:
            self.wfile.write(head.encode())
            self.wfile.flush()
            data = payload
        else:
            data = head.encode() + payload
        for index in range(len(data)):
            if lab._stop.wait(interval):
                return
            self._write_counted(data[index : index + 1])


class LabReceiver:
    """Scripted loopback receiver; see the module docstring for the API."""

    def __init__(self, *, tls: bool = False, cert: LabCert | None = None) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._requests: list[Recorded] = []
        self._connections = 0
        self._tls_failures = 0
        self._bytes_sent = 0
        self._by_path: dict[str, deque[Behavior]] = {}
        self._global: deque[Behavior] = deque()
        self._default: Behavior = Respond(200, b"{}")
        self._tmp: Path | None = None
        self.cert: LabCert | None = None
        self._server_context: ssl.SSLContext | None = None
        if tls:
            if cert is None:
                self._tmp = Path(tempfile.mkdtemp(prefix="ssgrep-lab-"))
                cert = LabCert.generate(self._tmp)
            self.cert = cert
            self._server_context = cert.server_context()
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.lab = self
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @classmethod
    def start(cls, tls: bool = False, cert: LabCert | None = None, **options: Any) -> Self:
        receiver = cls(tls=tls, cert=cert, **options)
        receiver._thread.start()
        return receiver

    def __enter__(self) -> LabReceiver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    @property
    def host(self) -> str:
        return "127.0.0.1"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def tls(self) -> bool:
        return self._server_context is not None

    @property
    def url(self) -> str:
        return f"{'https' if self.tls else 'http'}://{self.host}:{self.port}"

    def url_for(self, path: str) -> str:
        return self.url + path

    def queue(self, *behaviors: Behavior, path: str | None = None) -> None:
        with self._lock:
            target = self._global if path is None else self._by_path.setdefault(path, deque())
            target.extend(behaviors)

    @property
    def requests(self) -> list[Recorded]:
        with self._lock:
            return list(self._requests)

    @property
    def connections(self) -> int:
        with self._lock:
            return self._connections

    @property
    def tls_failures(self) -> int:
        with self._lock:
            return self._tls_failures

    @property
    def bytes_sent(self) -> int:
        with self._lock:
            return self._bytes_sent

    def accepted_bodies(self) -> list[bytes]:
        return [item.body for item in self.requests if item.status == 200]

    def wait_for_requests(self, count: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.requests) >= count:
                return True
            time.sleep(0.01)
        return len(self.requests) >= count

    def _connection_accepted(self) -> None:
        with self._lock:
            self._connections += 1

    def _tls_failed(self) -> None:
        with self._lock:
            self._tls_failures += 1

    def _add_sent(self, count: int) -> None:
        with self._lock:
            self._bytes_sent += count

    def _record(self, record: Recorded) -> None:
        with self._lock:
            self._requests.append(record)

    def _next_behavior(self, path: str, method: str) -> Behavior:
        with self._lock:
            scripted = self._by_path.get(path)
            if scripted:
                return scripted.popleft()
            if self._global:
                return self._global.popleft()
        return self._default


class _ProxyHandler(_Handler):
    def _perform(self, behavior: Behavior) -> int | None:
        lab = self.server.lab
        relay_to = getattr(lab, "relay_to", None)
        if self.command != "CONNECT" or relay_to is None:
            return super()._perform(behavior)
        upstream = socket.create_connection(relay_to, timeout=5)
        try:
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            self._relay(upstream)
        finally:
            upstream.close()
        return 200

    def _relay(self, upstream: socket.socket) -> None:
        lab = self.server.lab
        client = self.request
        while not lab._stop.is_set():
            ready, _, _ = select.select([client, upstream], [], [], 0.05)
            for source in ready:
                data = source.recv(65536)
                if not data:
                    return
                (upstream if source is client else client).sendall(data)


class ProxyLab(LabReceiver):
    """Records CONNECT / absolute-URI requests; answers 502 unless ``relay_to`` is set."""

    def __init__(
        self,
        *,
        tls: bool = False,
        cert: LabCert | None = None,
        relay_to: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(tls=tls, cert=cert)
        self.relay_to = relay_to
        self._default = Respond(502, b"")
        self._server.RequestHandlerClass = _ProxyHandler

    @property
    def connects(self) -> list[Recorded]:
        return [item for item in self.requests if item.method == "CONNECT"]
