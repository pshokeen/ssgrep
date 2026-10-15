"""Bounded OTLP/HTTP transport, proven over real loopback sockets with the lab receiver."""

from __future__ import annotations

import socket
import ssl
import time
from http.client import HTTPConnection

import pytest

from ssgrep.accounting import export_http as http
from ssgrep.accounting.export_http import Outcome, Target, post
from tests.accounting.lab_receiver import (
    DropAfterRead,
    Hang,
    LabCert,
    LabReceiver,
    Oversize,
    ProxyLab,
    Redirect,
    Reset,
    Respond,
    SlowDrip,
    Truncate,
)

CANARY = "CANARY-SECRET-7f3a91"
BODY_CANARY = b'{"canary":"BODY-CANARY-55d2"}'
AUTH = (("Authorization", f"Bearer {CANARY}"),)
QUICK_TIMEOUT = 2.0


@pytest.fixture(scope="module")
def lab_cert(tmp_path_factory: pytest.TempPathFactory) -> LabCert:
    return LabCert.generate(tmp_path_factory.mktemp("lab-cert"))


@pytest.fixture(scope="module")
def other_cert(tmp_path_factory: pytest.TempPathFactory) -> LabCert:
    return LabCert.generate(tmp_path_factory.mktemp("other-cert"), san="DNS:other.example")


@pytest.fixture
def lab():
    with LabReceiver.start() as receiver:
        yield receiver


@pytest.fixture
def tls_lab(lab_cert):
    with LabReceiver.start(tls=True, cert=lab_cert) as receiver:
        yield receiver


def _post(
    url: str,
    body: bytes = b'{"a":1}',
    *,
    headers: tuple[tuple[str, str], ...] = (),
    proxy: str | None = None,
    context: ssl.SSLContext | None = None,
    connect_timeout: float = QUICK_TIMEOUT,
    read_timeout: float = QUICK_TIMEOUT,
    max_response_bytes: int = 64 * 1024,
) -> Outcome:
    return post(
        Target(url, proxy=proxy, headers=headers),
        body,
        context=context,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_response_bytes=max_response_bytes,
    )


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------- request shape


def test_post_sends_exact_headers_and_returns_only_the_200_body(lab):
    lab.queue(Respond(200, b'{"partialSuccess":{}}'))
    outcome = _post(lab.url_for("/v1/metrics"), b'{"x":1}', headers=AUTH)

    assert outcome == Outcome("response", 200, b'{"partialSuccess":{}}', None, "ok")
    (request,) = lab.requests
    assert (request.method, request.path, request.body) == ("POST", "/v1/metrics", b'{"x":1}')
    assert request.headers["content-type"] == "application/json"
    assert request.headers["content-length"] == "7"
    assert request.headers["connection"] == "close"
    assert request.headers["user-agent"] == "ssgrep-export"
    assert request.headers["authorization"] == f"Bearer {CANARY}"
    assert "accept-encoding" not in request.headers
    assert lab.accepted_bodies() == [b'{"x":1}']


def test_oversize_request_is_refused_before_any_connection(lab):
    limit = 1024 * 1024
    refused = _post(lab.url, b"x" * (limit + 1))
    assert (refused.kind, refused.category) == ("transport_refused", "refused_transport")
    assert lab.connections == 0
    # positive twin: exactly the limit is sent
    assert _post(lab.url, b"x" * limit).category == "ok"
    assert lab.connections == 1
    assert len(lab.requests[0].body) == limit


# ---------------------------------------------------------------- transport policy


def test_plain_http_to_non_loopback_never_opens_a_socket(lab):
    for host in (
        "192.0.2.1",
        "10.0.0.1",
        "127.0.0.2",
        "example.invalid",
        "[::2]",
        "localhost.evil",
    ):
        outcome = _post(f"http://{host}:{lab.port}/v1/metrics")
        assert outcome.kind == "transport_refused", host
    assert lab.connections == 0
    # positive twin over the very same receiver
    assert _post(f"http://127.0.0.1:{lab.port}/v1/metrics").category == "ok"
    assert lab.connections == 1


def test_loopback_names_are_allowed_for_plain_http(lab):
    assert _post(f"http://localhost:{lab.port}/v1/metrics").category == "ok"
    closed = _closed_port()
    outcome = _post(f"http://[::1]:{closed}/v1/metrics")
    assert outcome.kind == "network_error"  # permitted by policy; nothing listens


@pytest.mark.parametrize(
    "suffix",
    ["@x", "?token=1", "?", "#frag", "/p?a=b", "/p#f"],
)
def test_userinfo_query_and_fragment_are_refused(lab, suffix):
    url = f"http://127.0.0.1:{lab.port}{suffix}"
    if suffix == "@x":
        url = f"http://user:pw@127.0.0.1:{lab.port}/v1/metrics"
    assert _post(url).kind == "transport_refused"
    assert lab.connections == 0


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1/x",
        "file:///etc/passwd",
        "gopher://127.0.0.1/",
        "127.0.0.1:4318/v1/metrics",
        "http:///v1/metrics",
        "http://127.0.0.1:0/v1/metrics",
        "http://127.0.0.1:99999/v1/metrics",
        "http://127.0.0.1:1/a b",
        "http://127.0.0.1:1/\r\nX: y",
        "",
    ],
)
def test_other_schemes_and_malformed_urls_are_refused(url):
    assert _post(url).kind == "transport_refused"


@pytest.mark.parametrize(
    "proxy",
    [
        "http://user:pw@127.0.0.1:3128",
        "https://127.0.0.1:3128",
        "socks5://127.0.0.1:1080",
        "http://",
        "http://127.0.0.1:3128/path",
        "http://127.0.0.1:3128?x=1",
    ],
)
def test_unsafe_proxy_urls_are_refused(tls_lab, proxy):
    assert _post(tls_lab.url_for("/v1/metrics"), proxy=proxy).kind == "transport_refused"
    assert tls_lab.connections == 0


def test_proxy_with_plain_http_loopback_target_is_refused_not_used(lab):
    with ProxyLab.start() as proxy:
        outcome = _post(lab.url, proxy=f"http://127.0.0.1:{proxy.port}")
        assert outcome.kind == "transport_refused"
        assert proxy.connections == 0 and lab.connections == 0


@pytest.mark.parametrize(
    "headers",
    [
        (("Authorization", "Bearer a\r\nX-Evil: 1"),),
        (("Auth orization", "x"),),
        (("Authorization", "a\x00b"),),
        (("Content-Length", "1"),),
        (("Host", "evil"),),
        (("Authorization", "a"), ("authorization", "b")),
    ],
)
def test_hostile_headers_are_refused_before_connecting(lab, headers):
    assert _post(lab.url, headers=headers).kind == "transport_refused"
    assert lab.connections == 0


# ---------------------------------------------------------------- TLS


def test_https_with_trusted_context_succeeds_and_sends_credentials(tls_lab, lab_cert):
    outcome = _post(tls_lab.url_for("/v1/metrics"), headers=AUTH, context=lab_cert.client_context())
    assert outcome.category == "ok"
    (request,) = tls_lab.requests
    assert request.tls is True
    assert request.headers["authorization"].endswith(CANARY)


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_untrusted_certificate_fails_tls_and_no_request_bytes_reach_the_receiver(tls_lab):
    # default context does not trust the lab's self-signed certificate
    outcome = _post(tls_lab.url_for("/v1/metrics"), BODY_CANARY, headers=AUTH)
    assert (outcome.kind, outcome.category) == ("tls_error", "tls")
    assert _wait_for(lambda: tls_lab.connections >= 1)
    assert tls_lab.requests == []
    assert _wait_for(lambda: tls_lab.tls_failures >= 1)


def test_hostname_mismatch_is_a_tls_error_even_when_the_ca_is_trusted(other_cert):
    with LabReceiver.start(tls=True, cert=other_cert) as receiver:
        outcome = _post(
            receiver.url_for("/v1/metrics"), headers=AUTH, context=other_cert.client_context()
        )
        assert outcome.kind == "tls_error"
        assert receiver.requests == []


def test_plain_http_server_behind_https_url_is_never_downgraded(lab):
    outcome = _post(f"https://127.0.0.1:{lab.port}/v1/metrics", headers=AUTH)
    assert outcome.kind in {"tls_error", "network_error"}
    assert all(request.headers.get("authorization") is None for request in lab.requests)
    assert lab.requests == []


# ---------------------------------------------------------------- redirects


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_refused_and_nothing_is_forwarded(lab, status):
    with LabReceiver.start() as target:
        lab.queue(Redirect(target.url_for("/steal"), status))
        outcome = _post(lab.url, BODY_CANARY, headers=AUTH)
        assert (outcome.kind, outcome.status, outcome.category) == (
            "redirect_refused",
            status,
            "redirect",
        )
        assert len(lab.requests) == 1 and lab.connections == 1
        assert target.requests == [] and target.connections == 0


def test_redirect_from_https_to_http_and_other_host_is_refused(tls_lab, lab_cert):
    with LabReceiver.start() as plain:
        for location in (plain.url_for("/v1/metrics"), "http://192.0.2.1:9/v1/metrics"):
            tls_lab.queue(Redirect(location, 302))
            outcome = _post(
                tls_lab.url_for("/v1/metrics"), BODY_CANARY, headers=AUTH,
                context=lab_cert.client_context(),
            )  # fmt: skip
            assert outcome.kind == "redirect_refused"
        assert len(tls_lab.requests) == 2
        assert plain.requests == [] and plain.connections == 0
    assert outcome.body is None


# ---------------------------------------------------------------- response bounds


def test_read_bounded_counts_bytes_actually_read():
    reads: list[int] = []
    source = iter([b"a" * 10] * 100)

    def read(size: int) -> bytes:
        reads.append(size)
        return next(source, b"")[:size]

    with pytest.raises(Exception) as caught:
        http.read_bounded(read, 25)
    assert type(caught.value).__name__ == "_Oversize"
    assert sum(len(b"a" * 10) for _ in reads[:3]) == 30  # three 10-byte reads crossed 25
    pieces = iter([b"abc", b"de", b""])
    assert http.read_bounded(lambda size: next(pieces), 5) == b"abcde"


@pytest.mark.parametrize("mode", ["content-length", "chunked", "stream"])
def test_oversize_response_ends_early_in_every_mode(lab, mode):
    cap = 4096
    lab.queue(Oversize(64 * 1024 * 1024, mode))
    started = time.monotonic()
    outcome = _post(lab.url, max_response_bytes=cap)
    assert (outcome.kind, outcome.category) == ("oversize_response", "oversize")
    assert outcome.body is None
    assert time.monotonic() - started < 2.0
    # the server cannot have pushed the whole body: the client stopped reading near the cap
    assert lab.bytes_sent < 8 * 1024 * 1024


def test_declared_oversize_content_length_is_refused(lab):
    lab.queue(Respond(200, b"y" * 70_000))
    outcome = _post(lab.url, max_response_bytes=64 * 1024)
    assert outcome.kind == "oversize_response"
    lab.queue(Respond(200, b"y" * 1000))
    assert _post(lab.url, max_response_bytes=64 * 1024).body == b"y" * 1000


def test_body_exactly_at_the_cap_is_accepted_and_one_more_byte_is_not(lab):
    cap = 2048
    lab.queue(Respond(200, b"z" * cap), Respond(200, b"z" * (cap + 1)))
    assert _post(lab.url, max_response_bytes=cap).body == b"z" * cap
    assert _post(lab.url, max_response_bytes=cap).kind == "oversize_response"


def test_non_200_bodies_are_discarded(lab):
    lab.queue(Respond(500, CANARY.encode()), Respond(400, CANARY.encode()))
    for _ in range(2):
        outcome = _post(lab.url)
        assert outcome.body is None
        assert CANARY not in repr(outcome)


def test_non_200_bodies_are_still_size_bounded(lab):
    lab.queue(Oversize(64 * 1024 * 1024, "stream"))
    outcome = _post(lab.url, max_response_bytes=1024)
    assert outcome.kind == "oversize_response"


def test_too_many_response_headers_is_malformed(lab):
    lab.queue(Respond(200, b"{}", {f"X-H{index}": "v" for index in range(150)}))
    outcome = _post(lab.url)
    assert (outcome.kind, outcome.category) == ("malformed_response", "malformed")


def test_overlong_header_line_is_malformed(lab):
    lab.queue(Respond(200, b"{}", {"X-Long": "v" * 70_000}))
    assert _post(lab.url).kind == "malformed_response"


def test_garbage_status_line_is_malformed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def garbage() -> None:
        connection, _ = server.accept()
        connection.recv(65536)
        connection.sendall(b"NOT-HTTP nonsense\r\n\r\n")
        connection.close()

    import threading

    thread = threading.Thread(target=garbage, daemon=True)
    thread.start()
    try:
        outcome = _post(f"http://127.0.0.1:{server.getsockname()[1]}/v1/metrics")
    finally:
        thread.join(2)
        server.close()
    assert outcome.kind == "malformed_response"


# ---------------------------------------------------------------- deadlines and disconnects


def test_receiver_that_never_responds_times_out(lab):
    lab.queue(Hang(30.0))
    started = time.monotonic()
    outcome = _post(lab.url, connect_timeout=0.3, read_timeout=0.4)
    assert (outcome.kind, outcome.category) == ("timeout", "timeout")
    assert time.monotonic() - started < 2.0


def test_slow_drip_is_cut_off_by_the_overall_deadline(lab):
    # every read completes within read_timeout (0.3s < 0.6s), so only the wall clock can stop it
    lab.queue(SlowDrip(interval=0.3))
    started = time.monotonic()
    outcome = _post(lab.url, connect_timeout=0.3, read_timeout=0.6)
    elapsed = time.monotonic() - started
    assert outcome.kind == "timeout"
    assert elapsed < 2.0, elapsed
    assert outcome.body is None


def test_slow_drip_of_the_body_only_is_also_cut_off(lab):
    lab.queue(SlowDrip(interval=0.3, body_only=True))
    started = time.monotonic()
    outcome = _post(lab.url, connect_timeout=0.3, read_timeout=0.6)
    assert outcome.kind == "timeout"
    assert time.monotonic() - started < 2.0


def test_slow_drip_over_tls_is_cut_off(tls_lab, lab_cert):
    tls_lab.queue(SlowDrip(interval=0.3))
    started = time.monotonic()
    outcome = _post(
        tls_lab.url, connect_timeout=0.3, read_timeout=0.6, context=lab_cert.client_context()
    )
    assert outcome.kind == "timeout"
    assert time.monotonic() - started < 2.0


def test_fast_response_within_deadline_is_not_a_timeout(lab):
    lab.queue(Respond(200, b"{}", delay=0.2))
    assert _post(lab.url, connect_timeout=1.0, read_timeout=1.0).category == "ok"


@pytest.mark.parametrize("behavior", [DropAfterRead(), Reset(), Truncate(100, 7)])
def test_disconnects_are_network_errors(lab, behavior):
    lab.queue(behavior)
    outcome = _post(lab.url)
    assert (outcome.kind, outcome.category) == ("network_error", "network")
    assert len(lab.requests) == 1  # the request was fully delivered; only the reply died


def test_connection_refused_is_a_network_error():
    outcome = _post(f"http://127.0.0.1:{_closed_port()}/v1/metrics")
    assert (outcome.kind, outcome.category) == ("network_error", "network")


# ---------------------------------------------------------------- statuses and Retry-After


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (200, "ok"),
        (401, "unauthorized"),
        (403, "forbidden"),
        (429, "rate_limited"),
        (400, "http_4xx"),
        (404, "http_4xx"),
        (413, "http_4xx"),
        (500, "http_5xx"),
        (502, "http_5xx"),
        (503, "http_5xx"),
        (504, "http_5xx"),
    ],
)
def test_status_mapping(lab, status, category):
    lab.queue(Respond(status, b'{"k":1}'))
    outcome = _post(lab.url)
    assert (outcome.kind, outcome.status, outcome.category) == ("response", status, category)
    assert outcome.body == (b'{"k":1}' if status == 200 else None)


@pytest.mark.parametrize("status", [202, 204, 206])
def test_unexpected_success_statuses_are_malformed(lab, status):
    lab.queue(Respond(status, b"" if status == 204 else b"{}"))
    outcome = _post(lab.url)
    assert (outcome.kind, outcome.status, outcome.category) == (
        "malformed_response",
        status,
        "malformed",
    )
    assert outcome.body is None


def test_retry_after_is_parsed_from_the_wire(lab):
    lab.queue(
        Respond(429, b"", {"Retry-After": "7"}),
        Respond(503, b"", {"Retry-After": "999999"}),
        Respond(503, b"", {"Retry-After": "garbage"}),
        Respond(503),
    )
    assert [_post(lab.url).retry_after_seconds for _ in range(4)] == [7, 3600, None, None]


def test_retry_after_http_date_and_clamping():
    now = 1_800_000_000.0
    future = "Fri, 15 Jan 2027 08:01:00 GMT"  # 1_800_000_000 is 2027-01-15 08:00:00 UTC
    assert http.parse_retry_after(future, now=lambda: now) == 60
    assert http.parse_retry_after("Fri, 15 Jan 2027 07:00:00 GMT", now=lambda: now) == 0
    assert http.parse_retry_after("Sat, 16 Jan 2027 08:00:00 GMT", now=lambda: now) == 3600
    assert http.parse_retry_after("0") == 0
    assert http.parse_retry_after("-5") is None
    assert http.parse_retry_after("1.5") is None
    assert http.parse_retry_after("") is None
    assert http.parse_retry_after(None) is None
    assert http.parse_retry_after("99999999999999999999") is None


# ---------------------------------------------------------------- proxies


def test_ambient_proxy_environment_is_never_used(tls_lab, lab_cert, monkeypatch):
    with ProxyLab.start() as proxy:
        proxy_url = f"http://127.0.0.1:{proxy.port}"
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.setenv(name, proxy_url)
        monkeypatch.setenv("NO_PROXY", "")
        outcome = _post(tls_lab.url_for("/v1/metrics"), context=lab_cert.client_context())
        assert outcome.category == "ok"
        assert len(tls_lab.requests) == 1
        assert proxy.connections == 0 and proxy.requests == []


def test_explicit_proxy_receives_connect_without_secrets_and_relays_nothing_when_it_refuses(
    tls_lab, lab_cert
):
    with ProxyLab.start() as proxy:
        outcome = _post(
            tls_lab.url_for("/v1/metrics"), BODY_CANARY, headers=AUTH,
            proxy=f"http://127.0.0.1:{proxy.port}", context=lab_cert.client_context(),
        )  # fmt: skip
        assert outcome.kind == "network_error"
        (connect,) = proxy.requests
        assert connect.method == "CONNECT"
        assert connect.path == f"127.0.0.1:{tls_lab.port}"
        assert "authorization" not in connect.headers
        assert BODY_CANARY not in connect.body and connect.body == b""
        assert tls_lab.connections == 0 and tls_lab.requests == []


def test_explicit_proxy_tunnels_https_end_to_end_with_credentials_inside_the_tunnel(
    tls_lab, lab_cert
):
    with ProxyLab.start(relay_to=(tls_lab.host, tls_lab.port)) as proxy:
        outcome = _post(
            tls_lab.url_for("/v1/metrics"), BODY_CANARY, headers=AUTH,
            proxy=f"http://127.0.0.1:{proxy.port}", context=lab_cert.client_context(),
        )  # fmt: skip
        assert outcome.category == "ok"
        (connect,) = proxy.requests
        assert connect.method == "CONNECT" and "authorization" not in connect.headers
        (request,) = tls_lab.requests
        assert request.body == BODY_CANARY and request.tls is True
        assert request.headers["authorization"] == f"Bearer {CANARY}"
        assert request.headers["host"] == f"127.0.0.1:{tls_lab.port}"


# ---------------------------------------------------------------- secrets


def test_repr_hides_header_values_and_userinfo_free_urls_only():
    target = Target("https://collector.example/v1/metrics", "http://127.0.0.1:3128", AUTH)
    text = repr(target)
    assert CANARY not in text and "Bearer" not in text
    assert "collector.example" in text and "1 redacted" in text
    assert CANARY not in str(target)


def test_repr_never_leaks_url_credentials_that_slipped_in():
    text = repr(Target("https://user:hunter2@collector.example/p?token=abc"))
    assert "hunter2" not in text and "token" not in text


@pytest.mark.parametrize(
    "script",
    [
        [Respond(200, BODY_CANARY)],
        [Respond(401, BODY_CANARY)],
        [Respond(500, BODY_CANARY)],
        [Redirect("http://127.0.0.1:1/" + CANARY, 307)],
        [DropAfterRead()],
        [Reset()],
        [Truncate()],
        [Oversize(1_000_000, "stream")],
        [Hang(10.0)],
    ],
)
def test_outcomes_never_contain_credentials_or_request_body(lab, script):
    lab.queue(*script)
    outcome = _post(lab.url, BODY_CANARY, headers=AUTH, connect_timeout=0.5, read_timeout=0.5,
                    max_response_bytes=4096)  # fmt: skip
    text = repr(outcome) + str(outcome)
    assert CANARY not in text
    assert BODY_CANARY.decode() not in text


def test_refusals_and_failures_do_not_raise_or_leak():
    weird = Target(f"http://user:{CANARY}@evil.example/?k={CANARY}", "x", AUTH)
    outcome = post(weird, BODY_CANARY)
    assert outcome.kind == "transport_refused"
    assert CANARY not in repr(outcome) + repr(weird)
    with LabReceiver.start(tls=True) as receiver:
        failed = _post(receiver.url, BODY_CANARY, headers=AUTH)
    assert failed == Outcome("tls_error", category="tls")


# --- URL, decoder, watchdog and Retry-After boundaries ---------------------------------------


def test_invalid_url_repr_redacts_credentials_and_length_refusal_never_opens_socket() -> None:
    target = http.Target("https://user:SECRET-CANARY@example.test:bad/path?private=1")
    assert "<invalid-url>" in repr(target)
    assert "SECRET-CANARY" not in repr(target) and "private" not in repr(target)
    with LabReceiver.start() as lab:
        oversized = http.Target(lab.url + "/" + "x" * http.MAX_URL_CHARS)
        rejected = http.post(oversized, b"{}")
        assert rejected.kind == "transport_refused" and lab.requests == []
        control = http.post(http.Target(lab.url + "/v1/metrics"), b"{}")
        assert control.kind == "response" and control.status == 200
        assert [r.body for r in lab.requests] == [b"{}"]


@pytest.mark.parametrize("error_type", [ssl.SSLEOFError, ssl.SSLZeroReturnError, ValueError])
def test_response_decoder_failure_discards_exception_text_and_next_request_recovers(
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_response = HTTPConnection.getresponse

    def fail_after_receiving(connection):
        response = real_response(connection)
        response.read()
        response.close()
        raise error_type("PRIVATE-RESPONSE-CANARY")

    with LabReceiver.start() as lab:
        target = http.Target(lab.url + "/v1/metrics")
        with monkeypatch.context() as fault:
            fault.setattr(HTTPConnection, "getresponse", fail_after_receiving)
            failed = http.post(target, b'{"resourceMetrics":[]}')
        assert failed.kind == (
            "malformed_response" if error_type is ValueError else "network_error"
        )
        assert failed.body is None and "PRIVATE-RESPONSE-CANARY" not in repr(failed)
        control = http.post(target, b'{"resourceMetrics":[]}')
        assert control.kind == "response" and control.status == 200 and control.body == b"{}"
        assert [r.body for r in lab.requests] == [b'{"resourceMetrics":[]}'] * 2


def test_deadline_fired_during_connection_setup_prevents_request_even_if_socket_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_start = http._Watchdog.start
    closed_socket = socket.socket()
    closed_socket.close()

    def expire_before_connect(dog, connection):
        real_start(dog, connection)
        dog.sock = closed_socket
        dog._fire()

    with LabReceiver.start() as lab:
        target = http.Target(lab.url + "/v1/metrics")
        with monkeypatch.context() as fault:
            fault.setattr(http._Watchdog, "start", expire_before_connect)
            expired = http.post(target, b"{}", connect_timeout=2, read_timeout=2)
        assert expired.kind == expired.category == "timeout"
        assert expired.body is None and lab.requests == []
        control = http.post(target, b"{}")
        assert control.status == 200 and [r.body for r in lab.requests] == [b"{}"]


def test_retry_after_without_timezone_is_not_interpreted_in_machine_local_time() -> None:
    with LabReceiver.start() as lab:
        target = http.Target(lab.url + "/v1/metrics")
        lab.queue(Respond(503, headers={"Retry-After": "Tue, 13 Oct 2026 12:00:00"}))
        ambiguous = http.post(target, b"{}")
        assert ambiguous.status == 503 and ambiguous.retry_after_seconds is None
        lab.queue(Respond(503, headers={"Retry-After": "17"}))
        bounded = http.post(target, b"{}")
        assert bounded.status == 503 and bounded.retry_after_seconds == 17
        assert http.post(target, b"{}").status == 200
        assert [r.body for r in lab.requests] == [b"{}"] * 3
