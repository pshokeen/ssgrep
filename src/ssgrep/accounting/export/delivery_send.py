"""Recovery, settlement, authorization, and dispatch for one export delivery cycle."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from ssgrep.accounting.export import (
    acks,
    consent as export_policy,
    creds as export_creds,
    ids as export_ids,
    otlp as export_http,
    projection as export_build,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.consent import AUTHORITY_HOLD_REASONS, Gate
from ssgrep.accounting.export.delivery import (
    STAMP_FORMAT,
    Cycle,
    discard,
    enqueue,
    is_current,
    plus_seconds,
    revalidate,
)
from ssgrep.accounting.export.registry import body_pairs, promote_unknown
from ssgrep.accounting.export.spool import Meta
from ssgrep.accounting.ledger.ledger_contracts import Signal
from ssgrep.accounting.privacy import policy, published as tombstones, purge_spool, retention

LIVE = export_spool.LIVE
INTERRUPTED = "send_interrupted"
POSSIBLY = "possibly_delivered"


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=UTC)


def refresh_scope(cycle: Cycle, gate: Gate) -> None:
    """Adopt ``gate`` and recompute today's scope document and its observed change stamp."""
    cycle.gate = gate
    if cycle.key is None:
        cycle.key, _ = export_build.key_for(cycle.profile, create=True)
    installation = export_ids.installation_id(cycle.key, cycle.destination, cycle.profile)
    cycle.scope = export_policy.scope_document(gate.policy, cycle.dest, cycle.signal, installation)
    cycle.scope_state = export_policy.observe_scope(
        cycle.profile, cycle.destination, cycle.signal, cycle.scope, cycle.stamp()
    )


def batches(cycle: Cycle) -> list[Meta]:
    return export_spool.list_batches(cycle.profile, cycle.destination, cycle.signal)


def send_order(metas: list[Meta]) -> list[Meta]:
    """Oldest first: data by (generation, index), then tombstones."""
    return sorted(metas, key=lambda m: (m.kind == "tombstone", m.generation, m.index, m.batch_id))


def is_unresolved(meta: Meta) -> bool:
    """Sent without a definitive answer: still in flight, or earlier doubt never settled."""
    return meta.state == "in_flight" or (meta.state == "queued" and meta.warning == POSSIBLY)


def sync_counts(profile: str, destination: str, signal: Signal) -> None:
    """``pending_batches``/``unknown_batches`` derived from the spool metas (single truth)."""
    metas = export_spool.list_batches(profile, destination, signal)
    pending = sum(1 for m in metas if m.state in LIVE)
    unknown = sum(1 for m in metas if is_unresolved(m))
    current = acks.read_delivery(profile, destination, signal)
    if current is None and not pending:
        return
    if (
        current is not None
        and (current.pending_batches, current.unknown_batches)
        == (
            pending,
            unknown,
        )
        and (pending or current.next_retry_utc is None)
    ):
        return

    def change(d: acks.Delivery) -> acks.Delivery:
        d = acks.set_counts(d, pending, unknown)
        return replace(d, next_retry_utc=None) if not pending else d

    acks.update_delivery(profile, destination, signal, change)


def recover(cycle: Cycle) -> list[str]:
    """Erase privacy-marked bodies, then return other in-flight leftovers to the unknown queue."""
    p, dest, sig = cycle.profile, cycle.destination, cycle.signal
    stamp = cycle.stamp()
    cleared: list[str] = []
    purge_spool.finish_pending(p, dest, sig, stamp)
    for meta in batches(cycle):
        if meta.state != "in_flight":
            continue
        if sig is Signal.LOGS:  # registry first: a crash in between is repaired by re-recovery
            export_registry.mark_unknown(p, dest, meta.batch_id)
        export_spool.update_meta(
            p,
            dest,
            sig,
            meta.batch_id,
            lambda m: replace(
                m,
                state="queued",
                warning=POSSIBLY,
                next_attempt_utc=stamp,
                last_category=m.last_category or INTERRUPTED,
            ),
        )
        cleared.append(meta.batch_id)
    delivery = acks.read_delivery(p, dest, sig)
    if cleared or (delivery is not None and delivery.in_flight is not None):
        acks.update_delivery(p, dest, sig, acks.finish_in_flight)
    if cleared:
        sync_counts(p, dest, sig)
    return cleared


@dataclass(frozen=True)
class Auth:
    """What one request authenticates with; ``error`` is a stable condition, never text."""

    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    fingerprint: str = ""
    error: str | None = None


def auth(cycle: Cycle) -> Auth:
    """Resolve the credential NOW (send time; never cached) and fingerprint what a block needs."""
    assert cycle.key is not None
    secret: str | None = None
    headers: tuple[tuple[str, str], ...] = ()
    error: str | None = None
    try:
        credential = export_creds.resolve(
            cycle.profile, cycle.dest.credential_ref, environ=cycle.deps.environ
        )
        if credential is not None:
            headers, secret = ((credential.header_name, credential.value),), credential.value
    except export_creds.CredentialError as failure:
        error = failure.condition
    except Exception:  # noqa: BLE001 - only a stable condition ever leaves this function
        error = "credential_invalid"
    fingerprint = export_policy.credential_fingerprint(
        cycle.key, cycle.destination, cycle.dest, secret
    )
    return Auth(headers, fingerprint, error)


def enforce_retention(cycle: Cycle) -> None:
    """Spool age/size bounds (explicit loss + registry reset) and acknowledged-snapshot expiry."""
    p, dest, sig = cycle.profile, cycle.destination, cycle.signal
    try:
        ceiling = policy.read_policy_file().ceilings.get(p)
    except Exception:  # noqa: BLE001 - an unreadable file never extends retention
        ceiling = None
    effective = retention.effective_retention(cycle.policy, ceiling)
    moment = cycle.at()

    def on_discard(meta: Meta) -> None:
        if sig is Signal.LOGS:
            export_registry.reset_queued(p, dest, meta.batch_id)

    retention.enforce_spool(p, dest, sig, now=moment, effective=effective, on_discard=on_discard)
    retention.expire_acknowledged_snapshots(p, dest, sig, now=moment, effective=effective)


HOLD_ONLY = frozenset({"export_paused", "policy_unreadable"}) | AUTHORITY_HOLD_REASONS


def denial_state(gate: Gate) -> tuple[str, str]:
    """(CycleResult state, stable reason) for a gate that denies sending right now."""
    if gate.reason in HOLD_ONLY:  # sending is held (never discarded) until policy is corrected
        return "paused", gate.reason
    if gate.reason == "not_enabled":
        return "not_enabled", "not_enabled"
    if gate.reason == "endpoint_invalid":
        return "blocked", "endpoint_invalid"
    return "revoked", gate.reason or "revoked"


def apply_denial(cycle: Cycle, gate: Gate) -> None:
    """Sending is not authorized now: no network, queue held or (when revoked) discarded.

    Pause and an unreadable policy only HOLD (nothing but crash recovery is written); a disabled
    signal keeps its bounded queue under retention; a revoked destination/signal discards the
    queue with an explicit ``policy_contraction`` loss.
    """
    cycle.gate = gate
    tally = cycle.tally
    cleared = recover(cycle)
    if cleared:
        batch_id = cleared[-1]
        acks.update_delivery(
            cycle.profile,
            cycle.destination,
            cycle.signal,
            lambda d: acks.mark_nonrecallable(d, batch_id),
        )
        tally.nonrecallable = batch_id
    if gate.revoked:
        live = [m for m in batches(cycle) if m.state in LIVE]
        if live:
            discard(cycle, live, reproject_ok=False)
    tally.stop = denial_state(gate)
    if gate.reason not in HOLD_ONLY:
        enforce_retention(cycle)
    sync_counts(cycle.profile, cycle.destination, cycle.signal)


"""Settle ONE dispatched request: classify the transport outcome, update every state file.

Write order is chosen so a crash anywhere repairs itself by a SAFE resend, never by a lost
acknowledgment: delivery progress first, registry second, the batch meta last. A meta that is
still ``in_flight`` after a crash is recovered as unknown and resent unless privacy-marked.
Marked bodies are erased after bookkeeping, never retried; ``acked`` always has backing progress.

Nothing here reads, stores or logs a response body (other than parsing a 200 for its rejection
count) or any exception text: every recorded value is an enumerated category or a number.
"""


POSSIBLY_KINDS = frozenset({"network_error", "timeout", "oversize_response", "malformed_response"})
BACKOFF_STOP = ("waiting", "retry_pending")


def _meta(cycle: Cycle, meta: Meta, **changes: object) -> None:
    export_spool.update_meta(
        cycle.profile,
        cycle.destination,
        cycle.signal,
        meta.batch_id,
        lambda m: replace(m, **changes),  # type: ignore[arg-type]
    )


def _delivery(cycle: Cycle, change) -> None:
    acks.update_delivery(cycle.profile, cycle.destination, cycle.signal, change)


def _progress(
    cycle: Cycle,
    meta: Meta,
    kind: str,
    *,
    rejected: int = 0,
    warning: str | None = None,
    identity_unavailable: bool = False,
):
    """The delivery change recording ``kind`` for a DATA batch (tombstones have no generation)."""
    stamp = cycle.stamp()

    def change(d: acks.Delivery) -> acks.Delivery:
        d = acks.finish_in_flight(d)
        if meta.kind == "data":
            d = acks.outcome(
                d,
                generation=meta.generation,
                snapshot_id=meta.snapshot_id,
                index=meta.index,
                total=meta.total,
                kind=kind,
                rejected=rejected,
                warning=warning,
                identity_unavailable=identity_unavailable,
                now=stamp,
            )
        if kind == "acked":
            d = replace(
                d, next_retry_utc=None, last_error_category=None, in_flight_nonrecallable=None
            )
        return d

    return change


def _body(cycle: Cycle, meta: Meta) -> bytes | None:
    try:
        return export_spool.read_body(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)
    except (export_spool.SpoolCorrupt, OSError):
        return None


def _registry(cycle: Cycle, meta: Meta, action: str) -> None:
    """Registry transition for a log batch (metrics have no registry)."""
    if cycle.signal is not Signal.LOGS:
        return
    p, dest, bid = cycle.profile, cycle.destination, meta.batch_id
    if action == "acked":
        export_registry.mark_acked(p, dest, bid)
        body = _body(cycle, meta)
        if body is not None:
            promote_unknown(p, dest, body_pairs(body))
        if meta.kind == "tombstone":
            tombstones.forget_acknowledged_deletions(p, dest)
    elif action == "partial":
        export_registry.mark_partial(p, dest, bid)
    elif action == "quarantined":
        export_registry.mark_quarantined(p, dest, bid)
    elif action == "unknown":
        export_registry.mark_unknown(p, dest, bid)


def accepted(cycle: Cycle, meta: Meta, body: bytes | None) -> None:
    """HTTP 200: full acknowledgment, accepted-with-warning, partial or unparseable."""
    result = acks.parse_response(cycle.signal, body)
    if result.outcome in ("full", "full_with_warning"):
        warning = result.warning_category
        _delivery(cycle, _progress(cycle, meta, "acked", warning=warning))
        _registry(cycle, meta, "acked")
        _meta(
            cycle,
            meta,
            state="acked",
            last_status=200,
            last_category="ok",
            warning=warning,
            rejected=0,
            next_attempt_utc=None,
        )
        cycle.tally.acked += 1
        if warning is not None:
            cycle.tally.warn(warning)
    elif result.outcome == "partial":
        _delivery(
            cycle,
            _progress(cycle, meta, "partial", rejected=result.rejected, identity_unavailable=True),
        )
        _registry(cycle, meta, "partial")
        _meta(
            cycle,
            meta,
            state="partial",
            last_status=200,
            last_category="partial_success",
            warning=None,
            rejected=result.rejected,
            next_attempt_utc=None,
        )
        cycle.tally.partial += 1
        cycle.tally.warn("partial_success")
    else:
        quarantine(cycle, meta, 200, "response_unparseable")


def quarantine(cycle: Cycle, meta: Meta, status: int | None, category: str) -> None:
    """A batch that must not be retried blindly: kept for inspection, never resent."""
    _delivery(cycle, _progress(cycle, meta, "quarantined"))
    _registry(cycle, meta, "quarantined")
    _meta(
        cycle,
        meta,
        state="quarantined",
        last_status=status,
        last_category=category,
        next_attempt_utc=None,
    )
    cycle.tally.quarantined += 1
    cycle.tally.failure = category


def reschedule(cycle: Cycle, meta: Meta, out: export_http.Outcome) -> None:
    """Keep the SAME bytes queued; wait ``backoff_seconds`` (Retry-After is a floor)."""
    seconds = export_http.backoff_seconds(
        max(meta.attempts, 1), out.retry_after_seconds, rand=cycle.deps.rand
    )
    due = plus_seconds(cycle.at(), seconds)
    possibly = out.kind in POSSIBLY_KINDS
    if possibly:
        _registry(cycle, meta, "unknown")
    _delivery(
        cycle,
        lambda d: acks.retry_scheduled(acks.finish_in_flight(d), due, out.category),
    )
    _meta(
        cycle,
        meta,
        state="queued",
        last_status=out.status,
        last_category=out.category,
        next_attempt_utc=due,
        warning=POSSIBLY if possibly else meta.warning,
    )
    tally = cycle.tally
    tally.retry_utc, tally.failure, tally.stop = due, out.category, BACKOFF_STOP


def block(
    cycle: Cycle,
    meta: Meta | None,
    reason: str,
    auth: Auth,
    out: export_http.Outcome | None,
    *,
    category: str | None = None,
) -> None:
    """401/403, an unusable credential or a refused transport: stop until something changes."""
    _delivery(
        cycle,
        lambda d: acks.blocked(
            replace(acks.finish_in_flight(d), last_error_category=category or reason),
            reason,
            auth.fingerprint,
        ),
    )
    if meta is not None and out is not None:
        _meta(
            cycle,
            meta,
            state="queued",
            last_status=out.status,
            last_category=out.category,
            next_attempt_utc=None,
        )
    tally = cycle.tally
    tally.blocked_reason, tally.failure = reason, category or reason
    tally.stop = ("blocked", f"blocked_{reason}")


def settle(cycle: Cycle, meta: Meta, out: export_http.Outcome, auth: Auth) -> None:
    """Dispatch on ``export_http.classify``; the batch ends in exactly one state."""
    verdict = export_http.classify(out)
    if verdict == export_http.ACK_CANDIDATE:
        accepted(cycle, meta, out.body)
    elif verdict == export_http.RETRY:
        reschedule(cycle, meta, out)
    elif verdict == export_http.BLOCKED:
        block(cycle, meta, "unauthorized" if out.status == 401 else "forbidden", auth, out)
    elif verdict == export_http.PERMANENT:
        quarantine(cycle, meta, out.status, out.category)
    elif out.kind == "transport_refused":
        block(cycle, meta, "policy_blocked", auth, out)
    else:  # tls_error | redirect_refused | oversize_response | malformed_response
        reschedule(cycle, meta, out)
    current = export_spool.read_meta(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)
    if current is not None and current.purge_requested:
        _delivery(cycle, lambda d: acks.mark_nonrecallable(d, meta.batch_id))
        cycle.tally.nonrecallable = meta.batch_id
        purge_spool.finish_pending(cycle.profile, cycle.destination, cycle.signal, cycle.stamp())
    sync_counts(cycle.profile, cycle.destination, cycle.signal)
    if not export_policy.gate(cycle.profile, cycle.destination, cycle.signal).ok:
        _delivery(cycle, lambda d: acks.mark_nonrecallable(d, meta.batch_id))
        cycle.tally.nonrecallable = meta.batch_id


"""Send queued batches oldest-first, one request in flight, authorization re-proven per request.

Immediately before EVERY request (not once per cycle) the loop re-reads the current policy
(``export_policy.gate``), recomputes today's scope document and judges the batch about to leave
(``delivery_invalidate.is_current``): a batch authorized when it was queued is never sent merely
for that reason. The meta is persisted ``in_flight`` (attempts + 1) and the delivery document
records the attempt BEFORE the POST; the response is settled by ``delivery_outcome.settle``. The
credential is resolved from its reference for each request and never stored or logged.
"""


WAITING = ("waiting", "retry_pending")


def url_for(cycle: Cycle) -> str:
    return f"{cycle.dest.endpoint.rstrip('/')}/v1/{cycle.signal.value}"


def _next(cycle: Cycle) -> Meta | None:
    queued = [
        m
        for m in batches(cycle)
        if m.state == "queued" and not m.purge_requested and m.generation not in cycle.skip
    ]
    ordered = send_order(queued)
    return ordered[0] if ordered else None


def _due(cycle: Cycle, meta: Meta) -> bool:
    return meta.next_attempt_utc is None or parse_stamp(meta.next_attempt_utc) <= cycle.at()


def _counts(cycle: Cycle) -> tuple[int, int]:
    metas = batches(cycle)
    pending = sum(1 for m in metas if m.state in LIVE)
    return pending, sum(1 for m in metas if is_unresolved(m))


def _begin(cycle: Cycle, meta: Meta) -> Meta:
    """Persist ``in_flight`` + the attempt BEFORE the request is dispatched."""
    stamp = cycle.stamp()

    def begin(m: Meta) -> Meta:
        if m.purge_requested:
            raise FileNotFoundError("batch is pending privacy erasure")
        return replace(
            m,
            state="in_flight",
            attempts=m.attempts + 1,
            last_attempt_utc=stamp,
            next_attempt_utc=None,
        )

    inflight = export_spool.update_meta(
        cycle.profile,
        cycle.destination,
        cycle.signal,
        meta.batch_id,
        begin,
    )

    def change(d: acks.Delivery) -> acks.Delivery:
        if meta.kind == "data":
            d = acks.attempted(d, meta.generation, stamp)
            d = acks.start_in_flight(d, meta.batch_id, meta.generation, stamp)
        else:  # deletion requests belong to no snapshot generation
            d = replace(d, last_attempt_at=stamp)
        return acks.set_counts(d, *_counts(cycle))

    acks.update_delivery(cycle.profile, cycle.destination, cycle.signal, change)
    return inflight


def _dispatch(cycle: Cycle, meta: Meta, credentials) -> None:
    try:
        body = export_spool.read_body(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)
    except export_spool.SpoolCorrupt:
        current = export_spool.read_meta(
            cycle.profile, cycle.destination, cycle.signal, meta.batch_id
        )
        if current is None or current.purge_requested:
            return  # a concurrent privacy purge is not transport corruption
        quarantine(cycle, meta, None, "spool_corrupt")  # bytes no longer match: never sent
        return
    inflight = _begin(cycle, meta)
    deps = cycle.deps
    target = export_http.Target(url_for(cycle), cycle.dest.proxy, credentials.headers)
    out = deps.post(
        target,
        body,
        context=deps.context,
        connect_timeout=deps.connect_timeout,
        read_timeout=deps.read_timeout,
    )
    cycle.tally.sent += 1
    settle(cycle, inflight, out, credentials)


def send_loop(cycle: Cycle) -> None:
    """Dispatch until the queue is empty, a retry/block/denial stops the cycle, or the cap."""
    tally = cycle.tally
    for _ in range(cycle.deps.max_batches):
        meta = _next(cycle)
        if meta is None or tally.stop is not None:
            return
        verdict = export_policy.gate(cycle.profile, cycle.destination, cycle.signal)
        if not verdict.ok:
            apply_denial(cycle, verdict)
            return
        refresh_scope(cycle, verdict)
        if not is_current(cycle, meta):
            revalidate(cycle)  # never send stale bytes; reproject below when still permitted
            if cycle.enqueue:
                enqueue(cycle)
            continue
        if not _due(cycle, meta):
            tally.stop, tally.retry_utc = WAITING, meta.next_attempt_utc
            return
        credentials = auth(cycle)
        if credentials.error is not None:
            block(cycle, None, "credential_invalid", credentials, None, category=credentials.error)
            return
        try:
            _dispatch(cycle, meta, credentials)
        except FileNotFoundError:
            continue  # the batch vanished between listing and sending (retention): re-list
