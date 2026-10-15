"""Constrained loopback HTTP surface for one accounting profile (design D9).

Binds ``127.0.0.1`` (and optionally ``::1``) only and refuses every other address. One server
serves exactly one profile with that profile's own bearer token file. All endpoints are GET/HEAD
reads of already published files: a request never collects, indexes, sends, creates storage, or
opens a caller-supplied path. Responses never carry CORS headers. This is a loopback utility, not
an internet server: it speaks HTTP/1.x with ``Connection: close`` over bounded threads.

Request order is fixed so nothing is revealed before authentication: wire limits, body refusal,
method, Host, credential-in-URL, bearer (``/healthz`` excepted), then routing and parameters.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlsplit

from ssgrep.accounting import http_auth, http_views
from ssgrep.accounting.contracts import DEFAULT_PORT
from ssgrep.accounting.export_authority import migration_view
from ssgrep.accounting.http_views import Response, error_response
from ssgrep.accounting.http_wire import (
    ClientGone,
    Request,
    WireError,
    read_request,
    write_response,
)
from ssgrep.accounting.ledger_contracts import (
    DEFAULT_COLLECT_SECONDS,
    DEFAULT_STALE_INTERVALS,
    Signal,
    valid_profile_id,
)

MAX_CONNECTIONS = 16
LOOPBACK_BINDS = ("127.0.0.1", "::1")
ACCEPT_POLL_SECONDS = 0.2
_METHODS = ("GET", "HEAD")
_VIEWS = ("current", "acknowledged")
_SECURITY = (("X-Content-Type-Options", "nosniff"),)


class BindRefused(ValueError):
    """The requested bind address is not loopback (the only addresses this server may use)."""


class PortInUse(OSError):
    """The requested port is occupied; the server never picks another one."""


class WorkerLike(Protocol):
    def run(self) -> Any: ...

    def stop(self) -> None: ...


def check_binds(bind: str | Sequence[str]) -> tuple[str, ...]:
    binds = (bind,) if isinstance(bind, str) else tuple(bind)
    if not binds or len(set(binds)) != len(binds) or any(b not in LOOPBACK_BINDS for b in binds):
        raise BindRefused("only 127.0.0.1 and ::1 may be bound; remote binds are not offered")
    return binds


def _listener(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((host, port))
        sock.listen(MAX_CONNECTIONS)
    except OSError as error:
        sock.close()
        raise PortInUse(error.errno or 0, f"cannot bind {host}:{port}: {error.strerror}") from None
    sock.settimeout(ACCEPT_POLL_SECONDS)
    return sock


class Context:
    """Everything one request needs; read fresh per request so pause/policy changes show up."""

    def __init__(
        self,
        profile: str,
        approved: frozenset[str],
        clock: Callable[[], datetime],
        stale_after: int,
    ) -> None:
        self.profile = profile
        self.approved = approved
        self.clock = clock
        self.stale_after = stale_after


def _policy_view(profile: str) -> tuple[bool, tuple[tuple[str, tuple[str, ...]], ...]]:
    from ssgrep.accounting import policy

    status = policy.policy_status(profile)
    configured = tuple(
        (d.destination_id, tuple(s.value for s in d.signals)) for d in status.profile.destinations
    )
    return status.profile.paused, configured


def _query(target: str, allowed: Sequence[str]) -> dict[str, str] | Response:
    """Parameters of ``target`` or a fixed 400; names outside ``allowed`` are never accepted."""
    parts = urlsplit(target)
    if parts.fragment or "#" in target:
        return error_response(400, "bad_request")
    try:
        pairs = parse_qsl(parts.query, keep_blank_values=True, strict_parsing=bool(parts.query))
    except ValueError:
        return error_response(400, "bad_request")
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names) or any(name not in allowed for name in names):
        return error_response(400, "unsupported_parameter")
    return dict(pairs)


def _snapshot_view(ctx: Context, request: Request, params: dict[str, str]) -> Response:
    view = params.get("view", "current")
    if view not in _VIEWS:
        return error_response(400, "unsupported_parameter")
    if view == "current":
        if "destination" in params or "signal" in params:
            return error_response(400, "unsupported_parameter")
        current, failure = http_views.load_current(ctx.profile)
        if current is None:
            return error_response(503, failure or "snapshot_unavailable", (("Retry-After", "60"),))
        paused, _ = _policy_view(ctx.profile)
        headers = http_views.freshness(current.envelope, ctx.clock(), ctx.stale_after, paused)
        migration = migration_view(ctx.profile)
        headers += http_views.migration_headers(migration)
        view = "current" if migration.state == "current" else "legacy"
        return http_views.snapshot_response(
            current, request.headers.get("if-none-match"), headers, view
        )
    destination = params.get("destination", "")
    signal = params.get("signal", Signal.METRICS.value)
    if not valid_profile_id(destination) or signal not in {s.value for s in Signal}:
        return error_response(400, "unsupported_parameter")
    acked, failure = http_views.acknowledged_current(ctx.profile, destination, signal)
    if acked is None:
        return failure or error_response(500, "internal_error")
    headers = (
        ("X-Ssgrep-Destination", destination),
        ("X-Ssgrep-Signal", signal),
        ("X-Ssgrep-Generation", str(acked.envelope.generation)),
        ("X-Ssgrep-Collected-At", acked.envelope.collected_at),
    )
    headers += http_views.migration_headers(migration_view(ctx.profile))
    return http_views.snapshot_response(
        acked, request.headers.get("if-none-match"), headers, view="acknowledged"
    )


def _metrics_view(ctx: Context) -> Response:
    current, _ = http_views.load_current(ctx.profile)
    migration = migration_view(ctx.profile)
    if current is not None and migration.reason:
        return error_response(
            503,
            migration.reason,
            (
                ("Retry-After", "60"),
                *http_views.migration_headers(migration),
            ),
        )
    paused, configured = _policy_view(ctx.profile)
    rows = http_views.delivery_rows(ctx.profile, configured)
    response = http_views.metrics_response(current, rows)
    if current is None or response.status != 200:
        return response
    headers = http_views.freshness(current.envelope, ctx.clock(), ctx.stale_after, paused)
    return Response(
        response.status,
        (*response.headers, *headers, *http_views.migration_headers(migration)),
        response.body,
    )


def _route(ctx: Context, request: Request) -> Response:
    path = urlsplit(request.target).path
    if path == "/metrics":
        params = _query(request.target, ())
        return params if isinstance(params, Response) else _metrics_view(ctx)
    if path == "/v1/snapshot":
        params = _query(request.target, ("view", "destination", "signal"))
        return params if isinstance(params, Response) else _snapshot_view(ctx, request, params)
    if path == "/v1/export-status":
        params = _query(request.target, ())
        if isinstance(params, Response):
            return params
        _, configured = _policy_view(ctx.profile)
        rows = http_views.delivery_rows(ctx.profile, configured)
        return http_views.export_status_response(rows, ctx.profile)
    return error_response(404, "not_found")


def handle_request(ctx: Context, request: Request) -> Response:
    """The fully ordered decision for one parsed request (pure of the socket)."""
    if request.has_body:
        return error_response(400, "request_body_not_allowed")
    if request.method not in _METHODS:
        return error_response(405, "method_not_allowed", (("Allow", "GET, HEAD"),))
    if not http_auth.host_allowed(request.headers.get("host"), ctx.approved):
        return error_response(421, "host_not_allowed")
    if http_auth.credential_in_url(request.target):
        return error_response(400, "credential_in_url")
    path = urlsplit(request.target).path
    if path == "/healthz":
        if urlsplit(request.target).query:
            return error_response(400, "unsupported_parameter")
        return Response(200, (("Content-Type", "text/plain; charset=utf-8"),), b"ok\n")
    expected = http_auth.load_token_or_none(ctx.profile)
    presented = http_auth.presented_token(request.headers.get("authorization"))
    if not http_auth.token_matches(presented, expected):
        return error_response(401, "unauthorized", (("WWW-Authenticate", "Bearer"),))
    return _route(ctx, request)


class ServerHandle:
    """A running (or startable) loopback server; ``port`` is the real bound port."""

    def __init__(
        self,
        profile: str,
        binds: tuple[str, ...],
        listeners: list[socket.socket],
        ctx: Context,
        worker: WorkerLike | None,
    ) -> None:
        self.profile = profile
        self.binds = binds
        self._listeners = listeners
        self._ctx = ctx
        self._worker = worker
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._started = False

    @property
    def port(self) -> int:
        return int(self._listeners[0].getsockname()[1])

    def start(self) -> ServerHandle:
        if self._started:
            return self
        self._started = True
        for listener in self._listeners:
            thread = threading.Thread(target=self._accept_loop, args=(listener,), daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def stop(self) -> None:
        """Stop accepting, drop open connections, close sockets; idempotent."""
        self._stop.set()
        if self._worker is not None:
            self._worker.stop()
        for thread in self._threads:
            thread.join(timeout=2)
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            _shutdown(client)
        for listener in self._listeners:
            listener.close()

    def run_worker(self) -> Any:
        """Run the composed worker on this thread; the server is stopped when it returns."""
        if self._worker is None:
            raise RuntimeError("no worker was composed with this server")
        self.start()
        try:
            return self._worker.run()
        finally:
            self.stop()

    def __enter__(self) -> ServerHandle:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _accept_loop(self, listener: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                client, _ = listener.accept()
            except (TimeoutError, BlockingIOError):
                continue
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                write_response(
                    client,
                    503,
                    [("Content-Type", http_views.JSON_TYPE), ("Retry-After", "1"), *_SECURITY],
                    b'{"error":"too_many_connections"}\n',
                )
                _shutdown(client)
                continue
            with self._lock:
                self._clients.add(client)
            threading.Thread(target=self._serve_client, args=(client,), daemon=True).start()

    def _serve_client(self, client: socket.socket) -> None:
        try:
            self._answer(client)
        finally:
            _shutdown(client)
            with self._lock:
                self._clients.discard(client)
            self._slots.release()

    def _answer(self, client: socket.socket) -> None:
        head_only = False
        try:
            request = read_request(client)
            head_only = request.method == "HEAD"
            response = handle_request(self._ctx, request)
        except ClientGone:
            return
        except WireError as error:
            response = error_response(error.status, error.code)
        except Exception:
            response = error_response(500, "internal_error")
        headers = [*response.headers, *_SECURITY]
        write_response(
            client,
            response.status,
            headers,
            response.body,
            head_only=head_only,
            declared_length=len(response.body),
        )


def _shutdown(client: socket.socket) -> None:
    try:
        client.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    client.close()


def serve(
    profile: str,
    *,
    port: int = DEFAULT_PORT,
    bind: str | Sequence[str] = "127.0.0.1",
    worker: WorkerLike | None = None,
    collect_seconds: int = DEFAULT_COLLECT_SECONDS,
    clock: Callable[[], datetime] | None = None,
    start: bool = True,
) -> ServerHandle:
    """Bind loopback, ensure the profile's token file, and (by default) start serving.

    Raises ``BindRefused`` for a non-loopback address, ``PortInUse`` when the port is
    taken, ``http_auth.TokenFileError`` for an unsafe token file, and ``ValueError`` for a
    profile id that is not a bounded identifier.
    """
    if not valid_profile_id(profile):
        raise ValueError("profile ids are bounded identifiers, never paths")
    binds = check_binds(bind)
    http_auth.ensure_token(profile)
    listeners: list[socket.socket] = []
    try:
        listeners.append(_listener(binds[0], port))
        bound = int(listeners[0].getsockname()[1])
        listeners.extend(_listener(host, bound) for host in binds[1:])
    except BaseException:
        for listener in listeners:
            listener.close()
        raise
    ctx = Context(
        profile,
        http_auth.approved_hosts(binds, bound),
        clock or (lambda: datetime.now(UTC)),
        collect_seconds * DEFAULT_STALE_INTERVALS,
    )
    handle = ServerHandle(profile, binds, listeners, ctx, worker)
    return handle.start() if start else handle
