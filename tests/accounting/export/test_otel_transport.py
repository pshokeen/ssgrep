"""Bounded OTLP/HTTP transport, proven over real loopback sockets with the lab receiver."""

from __future__ import annotations

import itertools as receiver_itertools
import json as receiver_json
import re as receiver_re
import socket
import ssl
import time
from http.client import HTTPConnection
from pathlib import Path as receiver_Path
from typing import Any as receiver_Any, cast as receiver_cast

import pytest
import pytest as receiver_pytest

import ssgrep.accounting.export.otlp as http
from ssgrep.accounting.export.otlp import Outcome, Target, post
from ssgrep.accounting.export.receiver import (
    Applied as receiver_Applied,
    Key as receiver_Key,
    Outcome as receiver_Outcome,
    Quarantined as receiver_Quarantined,
    ReceiverState as receiver_ReceiverState,
    apply as receiver_apply,
    current_totals as receiver_current_totals,
)
from ssgrep.accounting.ledger.contracts import (
    canonical_bytes as receiver_canonical_bytes,
    digest_of as receiver_digest_of,
)
from tests.accounting.export.lab_receiver import (
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
    deadline = time.monotonic() + 5
    while not lab.accepted_bodies() and time.monotonic() < deadline:
        time.sleep(0.01)
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


# ---- Receiver contract tests (isolated module globals) ----

# Behavior tests for the organization receiver deduplication/tombstone contract.
# Replay the synthetic conformance fixture; assert each rule with exact state values.


receiver_FIXTURE = (
    receiver_Path(__file__).resolve().parents[3] / "docs" / "examples" / "receiver-conformance.json"
)
receiver_RID1 = "0" * 31 + "1"
receiver_RID2 = "0" * 31 + "2"
receiver_METH = "methodology-hash-1"
receiver_UNIT = {"unit": "tokens", "method": "estimate", "tokenizer": None, "currency": None}
receiver_UK = ("tokens", "estimate", None, None)
receiver_KA = receiver_Key("tenant-a", "device-1", receiver_RID1)


def receiver_q(value: int | None, *, unit: str = "tokens") -> dict[str, receiver_Any]:
    if value is None:
        return {
            "value": None,
            "unit": unit,
            "basis": "unavailable",
            "method": "",
            "reason": "no_evidence",
        }
    return {"value": value, "unit": unit, "basis": "estimated", "method": "m", "reason": ""}


def receiver_mk(
    rid: str,
    revision: int,
    *,
    receiver_net: int | None = 100,
    rtype: str | None = None,
    methodology: str = receiver_METH,
    unit: dict[str, receiver_Any] | None = None,
    extra: dict[str, receiver_Any] | None = None,
) -> dict[str, receiver_Any]:
    kind = rtype or ("contribution" if revision == 1 else "adjustment")
    body: dict[str, receiver_Any] = {
        "schema": 1,
        "record_id": rid,
        "revision": revision,
        "supersedes_revision": None if revision == 1 else revision - 1,
        "record_type": kind,
        "task_id": "synthetic-task-1",
        "period": "2026-01-02",
        "unit": dict(unit or receiver_UNIT),
        "scenarios": {},
        "known_costs": {},
        "coverage": {},
        "methodology": methodology,
    }
    if kind != "deletion":
        credit = None if receiver_net is None else max(receiver_net, 0)
        debit = None if receiver_net is None else max(-receiver_net, 0)
        quantity_unit = body["unit"]["unit"]
        assert isinstance(quantity_unit, str)
        body["scenarios"] = {
            "central": {
                "credit": receiver_q(credit, unit=quantity_unit),
                "debit": receiver_q(debit, unit=quantity_unit),
                "net": receiver_q(receiver_net, unit=quantity_unit),
            }
        }
        body["known_costs"] = {"output_tokens": 5}
        body["coverage"] = {"complete": True, "sources": 1}
    body.update(extra or {})
    return body


def receiver_run(
    steps: list[tuple[receiver_Key, dict[str, receiver_Any]]],
    state: receiver_ReceiverState | None = None,
):
    state = state or receiver_ReceiverState()
    outcomes = []
    for key, body in steps:
        state, outcome = receiver_apply(state, key, body)
        outcomes.append(outcome)
    return state, outcomes


def receiver_net(state: receiver_ReceiverState, scenario: str = "central") -> int | None:
    return receiver_current_totals(
        state, methodology=receiver_METH, unit_key=receiver_UK, tenant=receiver_KA.tenant
    )["scenarios"][scenario]["net"]["total"]


# --- executable conformance fixture ---------------------------------------


def receiver__load() -> dict[str, receiver_Any]:
    def no_float(text: str) -> float:
        raise AssertionError(f"float in fixture: {text}")

    return receiver_json.loads(receiver_FIXTURE.read_text(), parse_float=no_float)


def receiver__case_ids() -> list[str]:
    return [c["name"] for c in receiver__load()["cases"]]


def test_fixture_covers_required_scenarios() -> None:
    names = set(receiver__case_ids())
    assert {
        "retry_and_reorder",
        "conflicting_duplicate_quarantine",
        "tombstone_blocks_replay",
        "tenant_isolation",
        "negative_net_preserved",
        "unknown_net_not_zero",
    } <= names


def test_fixture_is_synthetic_only() -> None:
    doc = receiver__load()
    text = receiver_FIXTURE.read_text()
    assert "/Users" not in text and "/home" not in text and "secret" not in text.lower()
    for case in doc["cases"]:
        for step in case["steps"]:
            body = step["body"]
            assert receiver_re.fullmatch(r"0{31}\d", body["record_id"])
            assert receiver_re.fullmatch(r"synthetic-task-\d+", body["task_id"])


@receiver_pytest.mark.parametrize("name", receiver__case_ids())
def test_fixture_case_replays(name: str) -> None:
    doc = receiver__load()
    case = next(c for c in doc["cases"] if c["name"] == name)
    state = receiver_ReceiverState()
    for i, step in enumerate(case["steps"]):
        k = step["key"]
        state, outcome = receiver_apply(
            state, receiver_Key(k["tenant"], k["device"], k["record_id"]), step["body"]
        )
        assert outcome is receiver_Outcome(step["expect"]), f"{name} step {i}"
    current = {f"{k.tenant}/{k.device}/{k.record_id}": a.revision for k, a in state.current.items()}
    assert current == case["expect_current"]
    tombs = {f"{k.tenant}/{k.device}/{k.record_id}": r for k, r in state.tombstones.items()}
    assert tombs == case["expect_tombstones"]
    assert len(state.quarantine) == case.get("expect_quarantined", 0)
    if "expect_net_central_by_tenant" in case:
        with receiver_pytest.raises(ValueError):
            receiver_current_totals(
                state, methodology=doc["methodology"], unit_key=tuple(doc["unit_key"])
            )
        for tenant, expected in case["expect_net_central_by_tenant"].items():
            totals = receiver_current_totals(
                state,
                methodology=doc["methodology"],
                unit_key=tuple(doc["unit_key"]),
                tenant=tenant,
            )
            assert totals["scenarios"]["central"]["net"]["total"] == expected
            assert totals["rows"] == 1
    else:
        tenant = case["steps"][0]["key"]["tenant"]
        totals = receiver_current_totals(
            state,
            methodology=doc["methodology"],
            unit_key=tuple(doc["unit_key"]),
            tenant=tenant,
        )
        assert totals["scenarios"]["central"]["net"]["total"] == case["expect_net_central"]
        if "expect_unknown_central" in case:
            assert (
                totals["scenarios"]["central"]["net"]["unknown"] == case["expect_unknown_central"]
            )


# --- revision rules ---------------------------------------------------------


def test_first_sight_is_applied_new_with_exact_state() -> None:
    body = receiver_mk(receiver_RID1, 1)
    state, outcome = receiver_apply(receiver_ReceiverState(), receiver_KA, body)
    assert outcome is receiver_Outcome.APPLIED_NEW
    assert dict(state.current) == {
        receiver_KA: receiver_Applied(1, receiver_digest_of(body), "contribution", body)
    }
    assert dict(state.tombstones) == {}
    assert state.quarantine == () and state.rejected == ()
    assert receiver_net(state) == 100


def test_higher_revision_replaces_current_row() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=100)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, receiver_net=-40)),
        ]
    )
    assert outcomes == [receiver_Outcome.APPLIED_NEW, receiver_Outcome.APPLIED_UPSERT]
    new = receiver_mk(receiver_RID1, 2, receiver_net=-40)
    assert dict(state.current) == {
        receiver_KA: receiver_Applied(2, receiver_digest_of(new), "adjustment", new)
    }
    assert receiver_net(state) == -40


def test_identical_duplicate_is_noop_and_state_equal() -> None:
    before, _ = receiver_apply(receiver_ReceiverState(), receiver_KA, receiver_mk(receiver_RID1, 3))
    after, outcome = receiver_apply(before, receiver_KA, receiver_mk(receiver_RID1, 3))
    assert outcome is receiver_Outcome.DUPLICATE_NOOP
    assert after == before
    assert receiver_net(after) == 100


def test_retry_then_reordered_older_revision() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 3, receiver_net=300)),
            (receiver_KA, receiver_mk(receiver_RID1, 3, receiver_net=300)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, receiver_net=200)),
        ]
    )
    assert outcomes == [
        receiver_Outcome.APPLIED_NEW,
        receiver_Outcome.DUPLICATE_NOOP,
        receiver_Outcome.REJECTED_OLD_REVISION,
    ]
    assert state.current[receiver_KA].revision == 3
    assert receiver_net(state) == 300  # rev 3 contributes once, rev 2 adds nothing
    assert dict(state.tombstones) == {}


def test_same_revision_different_digest_quarantines_without_replacing() -> None:
    held = receiver_mk(receiver_RID1, 2, receiver_net=100)
    clash = receiver_mk(receiver_RID1, 2, receiver_net=999)
    before, _ = receiver_apply(receiver_ReceiverState(), receiver_KA, held)
    after, outcome = receiver_apply(before, receiver_KA, clash)
    assert outcome is receiver_Outcome.QUARANTINED
    assert after.current == before.current
    assert receiver_net(after) == 100
    assert after.quarantine == (
        receiver_Quarantined(
            receiver_KA,
            2,
            receiver_digest_of(clash),
            receiver_digest_of(held),
            "same_revision_different_digest",
        ),
    )
    again, outcome2 = receiver_apply(after, receiver_KA, clash)
    assert outcome2 is receiver_Outcome.QUARANTINED and again == after  # retries do not grow state


def test_quarantine_records_digests_never_bodies() -> None:
    state, _ = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 2)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, receiver_net=7)),
        ]
    )
    entry = state.quarantine[0]
    assert set(vars(entry)) == {"key", "revision", "digest", "held_digest", "reason"}
    assert "synthetic-task-1" not in repr(state.quarantine)


def test_lower_revision_rejected_and_recorded() -> None:
    older = receiver_mk(receiver_RID1, 1)
    state, outcomes = receiver_run(
        [(receiver_KA, receiver_mk(receiver_RID1, 4)), (receiver_KA, older)]
    )
    assert outcomes[1] is receiver_Outcome.REJECTED_OLD_REVISION
    assert state.current[receiver_KA].revision == 4
    assert [(r.key, r.revision, r.digest) for r in state.rejected] == [
        (receiver_KA, 1, receiver_digest_of(older))
    ]


# --- deletion / tombstones --------------------------------------------------


def test_deletion_removes_row_and_retains_tombstone() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, rtype="deletion")),
        ]
    )
    assert outcomes == [receiver_Outcome.APPLIED_NEW, receiver_Outcome.APPLIED_TOMBSTONE]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {receiver_KA: 2}
    assert receiver_net(state) == 0


def test_tombstone_blocks_replay_of_older_contribution() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, rtype="deletion")),
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, receiver_net=55)),
        ]
    )
    assert outcomes[2:] == [
        receiver_Outcome.REJECTED_TOMBSTONED,
        receiver_Outcome.REJECTED_TOMBSTONED,
    ]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {receiver_KA: 2}
    assert receiver_net(state) == 0


def test_recreation_above_tombstone_is_new_and_tombstone_kept() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, rtype="deletion")),
            (receiver_KA, receiver_mk(receiver_RID1, 3, receiver_net=60)),
        ]
    )
    assert outcomes[2] is receiver_Outcome.APPLIED_NEW
    assert state.current[receiver_KA].revision == 3
    assert dict(state.tombstones) == {receiver_KA: 2}
    assert receiver_net(state) == 60
    replay, outcome = receiver_apply(
        state, receiver_KA, receiver_mk(receiver_RID1, 2, receiver_net=1)
    )
    assert outcome is receiver_Outcome.REJECTED_TOMBSTONED and replay.current == state.current
    stale, outcome = receiver_apply(
        state, receiver_KA, receiver_mk(receiver_RID1, 3, receiver_net=1)
    )
    assert outcome is receiver_Outcome.QUARANTINED and stale.current == state.current


@receiver_pytest.mark.parametrize(("del_rev", "tombs"), [(3, {}), (2, {receiver_KA: 2})])
def test_deletion_not_above_current_revision_cannot_delete_newer_row(
    del_rev: int, tombs: dict[receiver_Key, int]
) -> None:
    before, _ = receiver_run([(receiver_KA, receiver_mk(receiver_RID1, 3))])
    after, outcome = receiver_apply(
        before, receiver_KA, receiver_mk(receiver_RID1, del_rev, rtype="deletion")
    )
    assert outcome is receiver_Outcome.REJECTED_OLD_REVISION
    assert after.current == before.current
    assert dict(after.tombstones) == tombs
    assert receiver_net(after) == 100


def test_deletion_of_unseen_key_still_records_tombstone() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 5, rtype="deletion")),
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
        ]
    )
    assert outcomes == [receiver_Outcome.APPLIED_TOMBSTONE, receiver_Outcome.REJECTED_TOMBSTONED]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {receiver_KA: 5}


def test_repeated_identical_deletion_is_noop() -> None:
    before, _ = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, rtype="deletion")),
        ]
    )
    after, outcome = receiver_apply(
        before, receiver_KA, receiver_mk(receiver_RID1, 2, rtype="deletion")
    )
    assert outcome is receiver_Outcome.DUPLICATE_NOOP
    assert after == before


def test_deletion_below_tombstone_is_tombstoned_and_conflicting_deletion_quarantined() -> None:
    first = receiver_mk(receiver_RID1, 4, rtype="deletion")
    before, _ = receiver_run([(receiver_KA, first)])
    low, outcome = receiver_apply(
        before, receiver_KA, receiver_mk(receiver_RID1, 3, rtype="deletion")
    )
    assert outcome is receiver_Outcome.REJECTED_TOMBSTONED
    assert dict(low.tombstones) == {receiver_KA: 4}
    clash = receiver_mk(receiver_RID1, 4, rtype="deletion", extra={"task_id": "synthetic-task-2"})
    after, outcome = receiver_apply(before, receiver_KA, clash)
    assert outcome is receiver_Outcome.QUARANTINED
    assert dict(after.tombstones) == {receiver_KA: 4}
    assert after.quarantine[0].held_digest == receiver_digest_of(first)


def test_later_deletion_advances_tombstone_and_clears_recreated_row() -> None:
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 2, rtype="deletion")),
            (receiver_KA, receiver_mk(receiver_RID1, 3)),
            (receiver_KA, receiver_mk(receiver_RID1, 4, rtype="deletion")),
        ]
    )
    assert outcomes == [
        receiver_Outcome.APPLIED_TOMBSTONE,
        receiver_Outcome.APPLIED_NEW,
        receiver_Outcome.APPLIED_TOMBSTONE,
    ]
    assert dict(state.current) == {}
    assert dict(state.tombstones) == {receiver_KA: 4}


# --- isolation / authenticated identity --------------------------------------


def test_tenant_and_device_isolation() -> None:
    kb = receiver_Key("tenant-b", "device-1", receiver_RID1)
    kc = receiver_Key("tenant-a", "device-2", receiver_RID1)
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 3, rtype="deletion")),
            (kb, receiver_mk(receiver_RID1, 1)),
            (kc, receiver_mk(receiver_RID1, 1)),
            (receiver_KA, receiver_mk(receiver_RID1, 1)),
        ]
    )
    assert outcomes == [
        receiver_Outcome.APPLIED_TOMBSTONE,
        receiver_Outcome.APPLIED_NEW,
        receiver_Outcome.APPLIED_NEW,
        receiver_Outcome.REJECTED_TOMBSTONED,
    ]
    assert dict(state.tombstones) == {receiver_KA: 3}
    assert set(state.current) == {kb, kc}


def test_tenant_claim_in_body_is_not_accepted_or_used() -> None:
    kb = receiver_Key("tenant-b", "device-1", receiver_RID1)
    for field in ("tenant", "device", "attributes"):
        state, outcome = receiver_apply(
            receiver_ReceiverState(), kb, receiver_mk(receiver_RID1, 1, extra={field: "tenant-a"})
        )
        assert outcome is receiver_Outcome.REJECTED_INVALID
        assert state == receiver_ReceiverState()
    state, _ = receiver_apply(receiver_ReceiverState(), kb, receiver_mk(receiver_RID1, 1))
    assert set(state.current) == {kb}


# --- strict validation --------------------------------------------------------


def receiver__bad_bodies() -> dict[str, dict[str, receiver_Any]]:
    float_net = receiver_mk(receiver_RID1, 1)
    float_net["scenarios"]["central"]["net"]["value"] = 1.5
    unavailable_with_value = receiver_mk(receiver_RID1, 1)
    unavailable_with_value["scenarios"]["central"]["net"]["basis"] = "unavailable"
    bad_unit = receiver_mk(receiver_RID1, 1)
    bad_unit["unit"].pop("currency")
    missing = receiver_mk(receiver_RID1, 1)
    del missing["period"]
    deletion_with_scenarios = receiver_mk(receiver_RID1, 2, rtype="deletion")
    deletion_with_scenarios["scenarios"] = receiver_mk(receiver_RID1, 1)["scenarios"]
    huge = receiver_mk(
        receiver_RID1, 1, extra={"known_costs": {f"k{i}": "x" * 500 for i in range(60)}}
    )
    huge["coverage"] = {f"c{i}": "y" * 500 for i in range(60)}
    huge["scenarios"]["central"]["net"]["method"] = "z" * 100_000
    return {
        "bool_revision": receiver_mk(receiver_RID1, 1, extra={"revision": True}),
        "zero_revision": receiver_mk(receiver_RID1, 0),
        "float_revision": receiver_mk(receiver_RID1, 1, extra={"revision": 1.0}),
        "float_net": float_net,
        "string_revision": receiver_mk(receiver_RID1, 1, extra={"revision": "1"}),
        "unknown_key": receiver_mk(receiver_RID1, 1, extra={"extra": 1}),
        "missing_key": missing,
        "wrong_schema": receiver_mk(receiver_RID1, 1, extra={"schema": 2}),
        "record_id_mismatch": receiver_mk(receiver_RID2, 1),
        "bad_record_type": receiver_mk(receiver_RID1, 1, extra={"record_type": "refund"}),
        "bad_period": receiver_mk(receiver_RID1, 1, extra={"period": "2026-13-40"}),
        "bad_unit": bad_unit,
        "bad_scenario": receiver_mk(
            receiver_RID1,
            1,
            extra={"scenarios": {"best": receiver_mk(receiver_RID1, 1)["scenarios"]["central"]}},
        ),
        "unavailable_inconsistent": unavailable_with_value,
        "supersedes_on_first": receiver_mk(receiver_RID1, 1, extra={"supersedes_revision": 1}),
        "supersedes_not_lower": receiver_mk(receiver_RID1, 3, extra={"supersedes_revision": 3}),
        "deletion_with_scenarios": deletion_with_scenarios,
        "float_known_cost": receiver_mk(receiver_RID1, 1, extra={"known_costs": {"cost": 0.5}}),
        "nested_coverage": receiver_mk(receiver_RID1, 1, extra={"coverage": {"a": {"b": 1}}}),
        "empty_methodology": receiver_mk(receiver_RID1, 1, extra={"methodology": ""}),
        "oversize": huge,
    }


@receiver_pytest.mark.parametrize("name", sorted(receiver__bad_bodies()))
def test_invalid_bodies_rejected_without_state_change(name: str) -> None:
    before, _ = receiver_run([(receiver_KA, receiver_mk(receiver_RID1, 1))])
    after, outcome = receiver_apply(before, receiver_KA, receiver__bad_bodies()[name])
    assert outcome is receiver_Outcome.REJECTED_INVALID
    assert after is before or after == before


@receiver_pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("scenario", "known_net", 99),
        ("scenario", "complete", False),
        ("scenario", "credit_missing", {"missing_credit": 1}),
        ("scenario", "debit_missing", {"Bad-Code": 1}),
        ("scenario", "debit_missing", {"missing_debit": 0}),
        ("bounds", "net_lower", receiver_q(99)),
        ("bounds", "net_upper", receiver_q(100, unit="usd_micros")),
        ("bounds", "credit_upper", receiver_q(99)),
        ("bounds", "debit_lower", receiver_q(-1)),
        ("bounds", "assumptions", ["unrecognized_assumption"]),
    ],
    ids=[
        "known-subtotal-mismatch",
        "incomplete-finite-net",
        "complete-missing-credit",
        "invalid-missing-code",
        "nonpositive-missing-count",
        "bound-arithmetic",
        "bound-unit",
        "bound-order",
        "negative-debit-bound",
        "unknown-bound-assumption",
    ],
)
def test_current_scenario_arithmetic_rejects_invalid_without_mutating_state(
    section: str, field: str, value: receiver_Any
) -> None:
    body = receiver_mk(receiver_RID1, 2)
    scenario: dict[str, receiver_Any] = body["scenarios"]["central"]
    scenario.update(
        {
            "known_net": 100,
            "complete": True,
            "credit_missing": {},
            "debit_missing": {},
            "known_atoms": 1,
            "known_debits": 1,
            "bounds": {
                "credit_lower": receiver_q(100),
                "credit_upper": receiver_q(100),
                "debit_lower": receiver_q(0),
                "debit_upper": receiver_q(0),
                "net_lower": receiver_q(100),
                "net_upper": receiver_q(100),
                "assumptions": ["credit_and_debit_nonnegative"],
            },
        }
    )
    before, _ = receiver_run([(receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=10))])
    accepted, outcome = receiver_apply(before, receiver_KA, body)
    assert outcome is receiver_Outcome.APPLIED_UPSERT
    assert receiver_net(accepted) == 100
    target = scenario if section == "scenario" else scenario["bounds"]
    target[field] = value
    after, outcome = receiver_apply(before, receiver_KA, body)
    assert outcome is receiver_Outcome.REJECTED_INVALID
    assert after == before
    assert receiver_net(after) == 10


def test_non_mapping_body_and_bad_key_are_invalid() -> None:
    empty = receiver_ReceiverState()
    for body in (None, [], "x", 3):
        assert receiver_apply(empty, receiver_KA, receiver_cast(receiver_Any, body)) == (
            empty,
            receiver_Outcome.REJECTED_INVALID,
        )
    for key in (
        receiver_Key("", "d", receiver_RID1),
        receiver_Key("t", "", receiver_RID1),
        receiver_Key("t", "d", ""),
    ):
        assert receiver_apply(empty, key, receiver_mk(receiver_RID1, 1)) == (
            empty,
            receiver_Outcome.REJECTED_INVALID,
        )
    # the receiver requires an authenticated Key instance, not a look-alike, before any mutation
    for bad_key in (None, ("tenant-a", "device-1", receiver_RID1), {"record_id": receiver_RID1}):
        rejected, outcome = receiver_apply(
            empty, receiver_cast(receiver_Key, bad_key), receiver_mk(receiver_RID1, 1)
        )
        assert rejected is empty and outcome is receiver_Outcome.REJECTED_INVALID
    accepted, outcome = receiver_apply(empty, receiver_KA, receiver_mk(receiver_RID1, 1))
    assert outcome is receiver_Outcome.APPLIED_NEW
    assert accepted.current[receiver_KA].revision == 1 and empty.current == {}


def test_body_size_limit_is_64_kib_of_canonical_bytes() -> None:
    def sized(entries: int) -> dict[str, receiver_Any]:
        costs = {f"k{i}": "x" * 1000 for i in range(entries)}
        return receiver_mk(
            receiver_RID1,
            1,
            extra={"known_costs": costs, "coverage": {f"c{i}": "y" * 1000 for i in range(entries)}},
        )

    under, over = sized(31), sized(33)
    assert len(receiver_canonical_bytes(under)) < 64 * 1024 < len(receiver_canonical_bytes(over))
    assert (
        receiver_apply(receiver_ReceiverState(), receiver_KA, under)[1]
        is receiver_Outcome.APPLIED_NEW
    )
    assert receiver_apply(receiver_ReceiverState(), receiver_KA, over) == (
        receiver_ReceiverState(),
        receiver_Outcome.REJECTED_INVALID,
    )


def test_apply_does_not_mutate_inputs_and_state_is_immutable() -> None:
    body = receiver_mk(receiver_RID1, 1)
    snapshot = receiver_json.loads(receiver_json.dumps(body))
    before = receiver_ReceiverState()
    state, _ = receiver_apply(before, receiver_KA, body)
    assert body == snapshot
    assert before == receiver_ReceiverState() and dict(before.current) == {}
    body["scenarios"]["central"]["net"]["value"] = 12345  # caller mutates afterwards
    assert state.current[receiver_KA].body == snapshot
    with receiver_pytest.raises(TypeError):
        state.current[receiver_KA] = state.current[receiver_KA]  # ty: ignore[invalid-assignment]
    with receiver_pytest.raises(TypeError):
        state.tombstones[receiver_KA] = 9  # ty: ignore[invalid-assignment]


def test_rejected_log_is_bounded_to_newest_1000() -> None:
    state, _ = receiver_apply(
        receiver_ReceiverState(), receiver_KA, receiver_mk(receiver_RID1, 2000)
    )
    for rev in range(1, 1051):
        state, outcome = receiver_apply(state, receiver_KA, receiver_mk(receiver_RID1, rev))
        assert outcome is receiver_Outcome.REJECTED_OLD_REVISION
    assert len(state.rejected) == 1000
    assert state.rejected[-1].revision == 1050 and state.rejected[0].revision == 51


def test_quarantine_is_bounded_to_newest_1000() -> None:
    state, _ = receiver_apply(
        receiver_ReceiverState(), receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=0)
    )
    for n in range(1, 1051):
        state, outcome = receiver_apply(
            state, receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=n)
        )
        assert outcome is receiver_Outcome.QUARANTINED
    assert len(state.quarantine) == 1000
    assert state.quarantine[-1].digest == receiver_digest_of(
        receiver_mk(receiver_RID1, 1, receiver_net=1050)
    )
    assert receiver_net(state) == 0


# --- permutation / order insensitivity ----------------------------------------


@receiver_pytest.mark.parametrize(
    "bodies",
    [
        [
            receiver_mk(receiver_RID1, 1, receiver_net=10),
            receiver_mk(receiver_RID1, 2, receiver_net=20),
            receiver_mk(receiver_RID1, 3, rtype="deletion"),
        ],
        [
            receiver_mk(receiver_RID1, 1, receiver_net=10),
            receiver_mk(receiver_RID1, 2, rtype="deletion"),
            receiver_mk(receiver_RID1, 3, receiver_net=30),
        ],
        [
            receiver_mk(receiver_RID1, 1, receiver_net=10),
            receiver_mk(receiver_RID1, 2, receiver_net=20),
            receiver_mk(receiver_RID1, 3, receiver_net=30),
            receiver_mk(receiver_RID1, 4, receiver_net=-5),
        ],
        [
            receiver_mk(receiver_RID1, 1, receiver_net=10),
            receiver_mk(receiver_RID1, 2, rtype="deletion"),
            receiver_mk(receiver_RID1, 4, rtype="deletion"),
            receiver_mk(receiver_RID1, 3, receiver_net=30),
        ],
    ],
    ids=["delete-last", "delete-then-recreate", "four-adjustments", "two-deletions-between"],
)
def test_every_permutation_converges_without_double_counting(
    bodies: list[dict[str, receiver_Any]],
) -> None:
    finals = set()
    for order in receiver_itertools.permutations(bodies):
        state, _ = receiver_run([(receiver_KA, b) for b in order])
        # replaying the whole set again (retry storm) must not change anything
        again, outcomes = receiver_run([(receiver_KA, b) for b in order], state)
        assert again.current == state.current and again.tombstones == state.tombstones
        assert (
            receiver_Outcome.APPLIED_NEW not in outcomes
            and receiver_Outcome.APPLIED_UPSERT not in outcomes
        )
        finals.add(
            (
                tuple(sorted((repr(k), a.revision, a.digest) for k, a in state.current.items())),
                tuple(sorted(state.tombstones.items(), key=repr)),
                receiver_net(state),
            )
        )
    assert len(finals) == 1


# --- totals --------------------------------------------------------------------


def test_authenticated_tenant_totals_exclude_foreign_methodology_and_unit() -> None:
    foreign = receiver_Key("tenant-b", receiver_KA.device, receiver_RID1)
    other_unit = {"unit": "usd_micros", "method": "tariff", "tokenizer": None, "currency": "USD"}
    state, outcomes = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=-10)),
            (
                foreign,
                receiver_mk(
                    receiver_RID1,
                    1,
                    receiver_net=900,
                    methodology="foreign-methodology",
                    unit=other_unit,
                ),
            ),
        ]
    )
    assert outcomes == [receiver_Outcome.APPLIED_NEW, receiver_Outcome.APPLIED_NEW]
    with receiver_pytest.raises(ValueError):
        receiver_current_totals(state, methodology=receiver_METH, unit_key=receiver_UK)
    selected = receiver_current_totals(
        state, methodology=receiver_METH, unit_key=receiver_UK, tenant=receiver_KA.tenant
    )
    assert selected["rows"] == 1
    assert selected["scenarios"]["central"]["net"] == {"total": -10, "known": 1, "unknown": 0}
    absent = receiver_current_totals(
        state, methodology=receiver_METH, unit_key=receiver_UK, tenant="not-enrolled"
    )
    assert absent["rows"] == 0
    assert absent["scenarios"]["central"]["net"] == {"total": 0, "known": 0, "unknown": 0}


def test_totals_sum_current_rows_per_scenario_with_negatives_preserved() -> None:
    state, _ = receiver_run(
        [
            (receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=100)),
            (receiver_KA, receiver_mk(receiver_RID1, 2, receiver_net=-250)),
            (
                receiver_Key(receiver_KA.tenant, "device-2", receiver_RID2),
                receiver_mk(receiver_RID2, 1, receiver_net=100),
            ),
        ]
    )
    totals = receiver_current_totals(
        state, methodology=receiver_METH, unit_key=receiver_UK, tenant=receiver_KA.tenant
    )
    central = totals["scenarios"]["central"]["net"]
    assert central == {"total": -150, "known": 2, "unknown": 0}
    assert totals["scenarios"]["central"]["debit"]["total"] == 250
    assert totals["scenarios"]["central"]["credit"]["total"] == 100
    assert totals["rows"] == 2
    assert totals["methodology"] == receiver_METH
    assert totals["unit"] == receiver_UNIT
    assert all(type(v["total"]) is int for v in totals["scenarios"]["central"].values())


def test_scenario_absent_from_row_counts_as_unknown() -> None:
    state, _ = receiver_run([(receiver_KA, receiver_mk(receiver_RID1, 1))])
    opt = receiver_current_totals(state, methodology=receiver_METH, unit_key=receiver_UK)[
        "scenarios"
    ]["optimistic"]["net"]
    assert opt == {"total": None, "known": 0, "unknown": 1}


def test_unknown_net_is_counted_not_zero() -> None:
    only, _ = receiver_run([(receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=None))])
    assert receiver_current_totals(only, methodology=receiver_METH, unit_key=receiver_UK)[
        "scenarios"
    ]["central"]["net"] == {
        "total": None,
        "known": 0,
        "unknown": 1,
    }
    mixed, _ = receiver_run(
        [
            (
                receiver_Key(receiver_KA.tenant, "device-2", receiver_RID2),
                receiver_mk(receiver_RID2, 1, receiver_net=40),
            )
        ],
        only,
    )
    central = receiver_current_totals(
        mixed, methodology=receiver_METH, unit_key=receiver_UK, tenant=receiver_KA.tenant
    )["scenarios"]["central"]["net"]
    assert central == {"total": 40, "known": 1, "unknown": 1}


def test_empty_state_totals_are_exact_zero() -> None:
    totals = receiver_current_totals(
        receiver_ReceiverState(), methodology=receiver_METH, unit_key=receiver_UK
    )
    assert totals["rows"] == 0
    assert totals["scenarios"]["central"]["net"] == {"total": 0, "known": 0, "unknown": 0}


def test_mixing_methodologies_or_units_raises() -> None:
    other_unit = {"unit": "usd_micros", "method": "tariff", "tokenizer": None, "currency": "USD"}
    rid3 = "0" * 31 + "3"
    base, _ = receiver_run([(receiver_KA, receiver_mk(receiver_RID1, 1, receiver_net=10))])
    assert receiver_net(base) == 10
    by_method, outcomes = receiver_run(
        [
            (
                receiver_Key(receiver_KA.tenant, "device-2", receiver_RID2),
                receiver_mk(receiver_RID2, 1, receiver_net=20, methodology="methodology-hash-2"),
            )
        ],
        base,
    )
    assert outcomes == [receiver_Outcome.APPLIED_NEW]
    by_unit, outcomes = receiver_run(
        [
            (
                receiver_Key(receiver_KA.tenant, "device-2", rid3),
                receiver_mk(rid3, 1, receiver_net=30, unit=other_unit),
            )
        ],
        base,
    )
    assert outcomes == [receiver_Outcome.APPLIED_NEW]
    for mixed in (by_method, by_unit):
        with receiver_pytest.raises(ValueError):
            receiver_current_totals(
                mixed, methodology=receiver_METH, unit_key=receiver_UK, tenant=receiver_KA.tenant
            )
    with receiver_pytest.raises(ValueError):
        receiver_current_totals(base, methodology="methodology-hash-2", unit_key=receiver_UK)
    with receiver_pytest.raises(ValueError):
        receiver_current_totals(
            base, methodology=receiver_METH, unit_key=("usd_micros", "tariff", None, "USD")
        )
    with receiver_pytest.raises(ValueError):
        receiver_current_totals(base, methodology=receiver_METH, unit_key=("tokens", "estimate"))  # ty: ignore[invalid-argument-type]
