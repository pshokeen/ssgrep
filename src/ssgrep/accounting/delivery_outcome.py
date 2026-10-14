"""Settle ONE dispatched request: classify the transport outcome, update every state file.

Write order is chosen so a crash anywhere repairs itself by a SAFE resend, never by a lost
acknowledgment: delivery progress first, registry second, the batch meta last. A meta that is
still ``in_flight`` after a crash is recovered as unknown and resent unless privacy-marked.
Marked bodies are erased after bookkeeping, never retried; ``acked`` always has backing progress.

Nothing here reads, stores or logs a response body (other than parsing a 200 for its rejection
count) or any exception text: every recorded value is an enumerated category or a number.
"""

from __future__ import annotations

from dataclasses import replace

from ssgrep.accounting import (
    acks,
    export_http,
    export_policy,
    export_records,
    export_retry,
    export_spool,
    purge_spool,
    tombstones,
)
from ssgrep.accounting.delivery_registry import body_pairs, promote_unknown
from ssgrep.accounting.delivery_state import POSSIBLY, Auth, sync_counts
from ssgrep.accounting.delivery_types import Cycle, plus_seconds
from ssgrep.accounting.export_spool import Meta
from ssgrep.accounting.ledger_contracts import Signal

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
        export_records.mark_acked(p, dest, bid)
        body = _body(cycle, meta)
        if body is not None:
            promote_unknown(p, dest, body_pairs(body))
        if meta.kind == "tombstone":
            tombstones.forget_acknowledged_deletions(p, dest)
    elif action == "partial":
        export_records.mark_partial(p, dest, bid)
    elif action == "quarantined":
        export_records.mark_quarantined(p, dest, bid)
    elif action == "unknown":
        export_records.mark_unknown(p, dest, bid)


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
    seconds = export_retry.backoff_seconds(
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
    """Dispatch on ``export_retry.classify``; the batch ends in exactly one state."""
    verdict = export_retry.classify(out)
    if verdict == export_retry.ACK_CANDIDATE:
        accepted(cycle, meta, out.body)
    elif verdict == export_retry.RETRY:
        reschedule(cycle, meta, out)
    elif verdict == export_retry.BLOCKED:
        block(cycle, meta, "unauthorized" if out.status == 401 else "forbidden", auth, out)
    elif verdict == export_retry.PERMANENT:
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
