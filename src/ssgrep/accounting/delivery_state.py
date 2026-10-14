"""Per-cycle state helpers: crash recovery, counts, send-time credentials, retention.

``recover`` runs with the per-destination/signal send lock held, so a meta still ``in_flight``
can only be the leftover of a process that died between dispatch and persisting the response:
its acceptance is UNKNOWN (the receiver may have processed it). Privacy-marked bodies are erased,
never retried. Other batches return to ``queued`` with attempts and the SAME bytes/ids preserved
(at-least-once; the receiver dedupes by record key + revision).

"Possibly delivered" is sticky: a batch whose send ended without a definitive answer (restart in
flight, timeout, disconnect, unreadable response) carries the stable code ``possibly_delivered``
in its meta ``warning`` until a later request resolves it (accepted or rejected). A later 503
does not erase that doubt: the earlier request may well have been processed. ``unknown_batches``
counts exactly those batches, so status keeps saying ``unknown`` until a 200 settles them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from ssgrep.accounting import (
    acks,
    export_build,
    export_creds,
    export_ids,
    export_policy,
    export_records,
    export_spool,
    policy,
    purge_spool,
    retention,
)
from ssgrep.accounting.delivery_types import STAMP_FORMAT, Cycle
from ssgrep.accounting.export_policy import Gate
from ssgrep.accounting.export_spool import Meta
from ssgrep.accounting.ledger_contracts import Signal

LIVE = ("queued", "in_flight")
INTERRUPTED = "send_interrupted"
POSSIBLY = "possibly_delivered"
UNRESOLVED_CATEGORIES = frozenset({"network", "timeout", "oversize", "malformed", INTERRUPTED})


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
            export_records.mark_unknown(p, dest, meta.batch_id)
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
            export_records.reset_queued(p, dest, meta.batch_id)

    retention.enforce_spool(p, dest, sig, now=moment, effective=effective, on_discard=on_discard)
    retention.expire_acknowledged_snapshots(p, dest, sig, now=moment, effective=effective)
