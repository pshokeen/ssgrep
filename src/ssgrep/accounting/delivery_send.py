"""Send queued batches oldest-first, one request in flight, authorization re-proven per request.

Immediately before EVERY request (not once per cycle) the loop re-reads the current policy
(``export_policy.gate``), recomputes today's scope document and judges the batch about to leave
(``delivery_invalidate.is_current``): a batch authorized when it was queued is never sent merely
for that reason. The meta is persisted ``in_flight`` (attempts + 1) and the delivery document
records the attempt BEFORE the POST; the response is settled by ``delivery_outcome.settle``. The
credential is resolved from its reference for each request and never stored or logged.
"""

from __future__ import annotations

from dataclasses import replace

from ssgrep.accounting import acks, export_http, export_policy, export_spool
from ssgrep.accounting.delivery_enqueue import enqueue
from ssgrep.accounting.delivery_invalidate import apply_denial, is_current, revalidate
from ssgrep.accounting.delivery_outcome import block, quarantine, settle
from ssgrep.accounting.delivery_state import (
    LIVE,
    auth,
    batches,
    is_unresolved,
    parse_stamp,
    refresh_scope,
    send_order,
)
from ssgrep.accounting.delivery_types import Cycle
from ssgrep.accounting.export_spool import Meta

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
