"""Synthetic snapshots, delivery files, and a raw-socket client for the metrics/HTTP tests.

Nothing here ships: values are fabricated only to exercise endpoint states. Snapshots are
published through the real ``snapshots.publish`` so the server reads exactly what the
collector would have written.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ssgrep.accounting import (
    counters,
    http as server_http,
    http_auth,
    plainfile,
    policy,
    snapshots,
)
from ssgrep.accounting.contracts import Operation, Outcome, Phase, Quantity, Transport
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope
from tests.accounting.conftest import build_event
from tests.accounting.snapshot_builders import scenarios, window_7d

PROFILE = "personal-local"
COLLECTED_AT = "2026-10-12T12:00:00Z"
COLLECTED_EPOCH = 1_791_806_400  # 2026-10-12T12:00:00Z


def live_counters(extra_searches: int = 0) -> counters.LiveCounters:
    """Epoch-1 counters: 3+extra ok searches over cli (2 ms each), 1 show error over mcp."""
    state = counters.empty_counters(1, "2026-10-01T00:00:00Z")
    events = []
    for index in range(3 + extra_searches):
        events.append(build_event(execution_id=f"s-{index}", mono_ns=2_000_000))
        events.append(
            build_event(
                execution_id=f"s-{index}",
                phase=Phase.RENDERED,
                rendered_bytes=Quantity.observed(1000, "bytes", "cli"),
            )
        )
    events.append(
        build_event(
            execution_id="e-1",
            operation=Operation.SHOW,
            transport=Transport.MCP,
            outcome=Outcome.ERROR,
            mono_ns=40_000_000,
        )
    )
    return counters.observe(state, events, horizon_day=None)


def make_snapshot(
    generation: int,
    *,
    extra_searches: int = 0,
    window: dict[str, Any] | None = None,
    methodology_id: str = "m-test-v1",
    collected_at: str = COLLECTED_AT,
    with_counters: bool = True,
) -> SnapshotEnvelope:
    """A complete valid envelope with live counters, a 7d window, and coverage blocks."""
    observations: dict[str, Any] = {"backfill": []}
    if with_counters:
        observations["counters"] = counters.view(live_counters(extra_searches))
    return snapshots.build_envelope(
        generation=generation,
        counter_epoch=1,
        methodology_id=methodology_id,
        policy_version=1,
        collected_at=collected_at,
        coverage_through=collected_at,
        source_manifest_digest="b" * 16,
        windows={"7d": window if window is not None else window_7d()},
        coverage={
            "collection_scope": policy.collection_scope(policy.load_profile_policy(PROFILE)),
            "recorded_lookups": 112,
            "attributed_lookups": 91,
            "candidate_receipts": 13,
            "source_costs_available": 8,
            "ambiguous_joins": 3,
            "unavailable_sources": 2,
            "attribution": {
                "explicit": 70,
                "inferred_unique": 21,
                "ambiguous": 3,
                "unattributed": 18,
            },
            "reuse_evidence": {
                "none_observed": 90,
                "inspected": 10,
                "action_reference": 8,
                "validated": 3,
                "contradicted": 1,
            },
            "sources": {"total": 9, "complete": 8, "missing": 1, "unreadable": 0},
        },
        observations=observations,
        estimates={"representative_receipt": "rcpt-7f3a91"},
        warnings=(),
    )


def window_with_unknown_optimistic() -> dict[str, Any]:
    """Optimistic net tokens unknown; conservative net negative (a decreasing/negative gauge)."""
    return window_7d(
        estimates={
            "context_tokens": scenarios(
                (4000, 9000, 20000), (16000, 16000, 16000), (-12000, -7000, None), "tokens"
            ),
            "work_seconds": scenarios((60, 300, 700), (120, 120, 120), (-60, 180, 580), "s"),
        }
    )


def publish(profile: str, envelope: SnapshotEnvelope) -> None:
    snapshots.publish(profile, envelope)


def write_delivery(profile: str, destination: str, signal: str, **fields: Any) -> Path:
    """A flat ``delivery/<dest>/<signal>.json`` exactly where the export module writes it."""
    data_dir = Path(os.environ["SSGREP_DATA_DIR"])
    plainfile.write_private(
        data_dir,
        "accounting",
        profile,
        "delivery",
        destination,
        f"{signal}.json",
        data=json.dumps({"destination_id": destination, "signal": signal, **fields}).encode(),
    )
    return data_dir / "accounting" / profile / "delivery" / destination / f"{signal}.json"


def start(profile: str = PROFILE, **kwargs: Any) -> server_http.ServerHandle:
    return server_http.serve(profile, port=0, **kwargs)


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def token_for(profile: str = PROFILE) -> str:
    token = http_auth.load_token(profile)
    assert token is not None
    return token.decode("ascii")


def get(
    handle: server_http.ServerHandle,
    path: str,
    *,
    method: str = "GET",
    token: str | None | bool = True,
    headers: dict[str, str] | None = None,
) -> Reply:
    """A real HTTP request over a real socket; ``token=True`` sends this server's own token."""
    connection = http.client.HTTPConnection("127.0.0.1", handle.port, timeout=10)
    sent = dict(headers or {})
    if token is True:
        sent["Authorization"] = f"Bearer {token_for(handle.profile)}"
    elif isinstance(token, str):
        sent["Authorization"] = f"Bearer {token}"
    try:
        connection.request(method, path, headers=sent)
        response = connection.getresponse()
        body = response.read()
        return Reply(response.status, {k.lower(): v for k, v in response.getheaders()}, body)
    finally:
        connection.close()


def raw_exchange(port: int, payload: bytes, *, read_seconds: float = 3.0) -> bytes:
    """Send ``payload`` verbatim and return every byte the server answers before closing."""
    with socket.create_connection(("127.0.0.1", port), timeout=read_seconds) as sock:
        sock.sendall(payload)
        deadline = time.monotonic() + read_seconds
        chunks = []
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every file and directory under ``root`` (for no-side-effect checks)."""
    state: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        state[str(path.relative_to(root))] = (info.st_size, info.st_mtime_ns)
    return state
