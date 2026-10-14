"""``--check``: one EMPTY OTLP request to prove connectivity and authentication.

A check carries no data and writes NOTHING: no delivery document, spool batch, registry entry,
acknowledgment, scope observation, pseudonym key or lock file. A 401 here therefore never blocks
the destination (only a real send can), and a 200 never acknowledges anything. A paused profile
(or one paused by an invalid managed update) raises ``ExportPaused`` BEFORE any credential read or
network activity; any other policy denial is reported as state ``refused`` with the gate's stable
code, again without touching the network. The credential is resolved at call time and appears in
no result. Even an empty request answered with ``partialSuccess`` is simply ``ok``: it asked for
nothing, so there is nothing to reject.
"""

from __future__ import annotations

from ssgrep.accounting import export_creds, export_http, export_policy, otlp
from ssgrep.accounting.delivery_types import DEFAULT_DEPS, CheckResult, Deps, ExportPaused
from ssgrep.accounting.ledger_contracts import Signal

_KIND_STATES = {
    "network_error": "unreachable",
    "timeout": "timeout",
    "tls_error": "tls_error",
    "redirect_refused": "redirect_refused",
    "oversize_response": "oversize_response",
    "transport_refused": "refused",
    "malformed_response": "malformed",
}
_STATUS_STATES = {200: "ok", 401: "unauthorized", 403: "forbidden", 429: "rate_limited"}


def state_of(outcome: export_http.Outcome) -> str:
    """The enumerated check state for one transport outcome (no text is ever consulted)."""
    if outcome.kind == "response":
        return _STATUS_STATES.get(outcome.status or 0, "http_error")
    return _KIND_STATES.get(outcome.kind, "unreachable")


def check(
    profile: str, destination: str, signal: Signal, *, deps: Deps = DEFAULT_DEPS
) -> CheckResult:
    """Send an empty OTLP request; see the module docstring for the guarantees."""
    sig = Signal(signal)
    gate = export_policy.gate(profile, destination, sig, need_enabled=False, need_unpaused=True)
    if not gate.ok or gate.destination is None:
        if gate.reason == ExportPaused.condition:
            raise ExportPaused
        return CheckResult(destination, sig, "refused", False, None, gate.reason or "refused")
    dest = gate.destination
    try:
        credential = export_creds.resolve(profile, dest.credential_ref, environ=deps.environ)
    except export_creds.CredentialError as error:
        return CheckResult(destination, sig, "refused", False, None, error.condition)
    headers = () if credential is None else ((credential.header_name, credential.value),)
    target = export_http.Target(f"{dest.endpoint.rstrip('/')}/v1/{sig.value}", dest.proxy, headers)
    outcome = deps.post(
        target,
        otlp.empty_request(sig),
        context=deps.context,
        connect_timeout=deps.connect_timeout,
        read_timeout=deps.read_timeout,
    )
    state = state_of(outcome)
    return CheckResult(destination, sig, state, state == "ok", outcome.status, outcome.category)
