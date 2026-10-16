from __future__ import annotations

import os
import random
import re
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ssgrep.accounting.export import (
    acks,
    consent as export_policy,
    creds as export_creds,
    otlp,
    otlp as export_http,
    projection as export_build,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.consent import (
    AUTHORITY_HOLD_REASONS,
    Gate,
    ScopeState,
)
from ssgrep.accounting.export.spool import Meta
from ssgrep.accounting.ledger.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
    valid_profile_id,
)
from ssgrep.accounting.privacy import (
    policy,
    storage,
)
from ssgrep.accounting.views import snapshots

"""Organization export delivery: one locked cycle per destination/signal (design D10/D11).

Public API: ``run_destination`` (one cycle), ``run_export`` (every destination/signal that is
enabled or still holds queued data; never raises), ``export_callback`` (the zero-argument hook for
``serve_loop.Worker``; never raises), ``check`` (``--check``: an empty OTLP request that writes
nothing); read-only candidate preview lives in ``projection.preview_candidate``.

Cycle order for one destination/signal, exactly one process at a time (non-blocking flock
``export/<destination>/.send-<signal>.lock``; contended -> ``busy`` without any network):

1. gate against the CURRENT policy: pause / disabled / revoked mean NO network; crash leftovers
   (``in_flight``) are returned to the queue as unknown and, when sending is no longer allowed,
   reported ``in_flight_nonrecallable``;
2. observe the data-affecting scope; clear a block only when the credential/endpoint/proxy changed;
3. recover crash leftovers, then retention (age/size bounds with explicit loss);
4. revalidate every queued batch against today's scope (stale bytes are discarded or reprojected,
   never sent) and enqueue the current snapshot when it is new;
5. send oldest first, re-gating and re-checking scope before EVERY request.

Nothing that reaches a result, a state document or ``last_error.json`` is free text: only
enumerated codes, counts and timestamps (``delivery_types.reason_of`` never reads exception text).
"""


"""Types and small pure helpers shared by the delivery modules (no I/O except ``reason_of``).

``Deps`` is the injectable seam (production defaults), ``CycleResult``/``CheckResult`` are the
public verdicts, ``Cycle`` is the mutable context of ONE ``run_destination`` call and ``Tally``
its counters. Every code that can reach a result, a delivery document or ``last_error.json`` is a
stable enumerated string: ``reason_of`` maps an exception to one WITHOUT ever reading its text.
"""


STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
# a dispatched request whose fate the sender cannot know (the receiver may have processed it)
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CLASS = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


class ExportPaused(Exception):
    """Sending/checking was refused because the profile (or an invalid managed update) paused it."""

    condition = "export_paused"

    def __init__(self) -> None:
        super().__init__(self.condition)


class CycleError(Exception):
    """A cycle step failed for a stable, text-free reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Deps:
    """Injectable seams; the defaults are the production behaviour."""

    post: Callable[..., export_http.Outcome] = export_http.post
    rand: Callable[[], float] = random.random
    environ: Mapping[str, str] | None = None
    context: ssl.SSLContext | None = None
    connect_timeout: float = 10.0
    read_timeout: float = 10.0
    max_batches: int = 64


DEFAULT_DEPS = Deps()


@dataclass(frozen=True)
class CycleResult:
    destination: str
    signal: Signal
    state: str  # sent|idle|waiting|paused|not_enabled|blocked|revoked|busy|no_snapshot|failed
    reason: str | None = None
    enqueued_batches: int = 0
    invalidated_batches: int = 0
    sent: int = 0
    acked: int = 0
    partial: int = 0
    quarantined: int = 0
    retry_scheduled_utc: str | None = None
    blocked_reason: str | None = None
    nonrecallable_batch: str | None = None
    warnings: tuple[str, ...] = ()
    error_category: str | None = None  # last transport/refusal category (deviation: extra field)


@dataclass(frozen=True)
class CheckResult:
    destination: str
    signal: Signal
    state: str  # ok|unauthorized|forbidden|rate_limited|unreachable|timeout|tls_error|...
    ok: bool
    status: int | None
    category: str


@dataclass
class Tally:
    enqueued: int = 0
    invalidated: int = 0
    sent: int = 0
    acked: int = 0
    partial: int = 0
    quarantined: int = 0
    retry_utc: str | None = None
    blocked_reason: str | None = None
    nonrecallable: str | None = None
    warnings: list[str] = field(default_factory=list)
    stop: tuple[str, str] | None = None  # (state, reason) that ended the cycle
    failure: str | None = None  # category of the last retry/refusal
    enqueue_error: str | None = None
    idle: str | None = None  # why nothing was enqueued (no_snapshot|ineligible_scope)

    def warn(self, code: str) -> None:
        if code not in self.warnings:
            self.warnings.append(code)


@dataclass
class Cycle:
    """Everything one locked cycle knows; filled in step by step by ``delivery``."""

    profile: str
    destination: str
    signal: Signal
    now: datetime
    fixed_clock: bool
    deps: Deps
    enqueue: bool
    tally: Tally = field(default_factory=Tally)
    gate: Gate | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    scope_state: ScopeState | None = None
    key: bytes | None = None
    skip: set[int] = field(default_factory=set)  # generations whose build failed this cycle

    @property
    def dest(self) -> DestinationPolicy:
        assert self.gate is not None and self.gate.destination is not None
        return self.gate.destination

    @property
    def policy(self) -> ProfilePolicy:
        assert self.gate is not None
        return self.gate.policy

    def at(self) -> datetime:
        """The instant to stamp events with: injected ``now`` verbatim, else the wall clock."""
        return self.now if self.fixed_clock else resolve_now(None)

    def stamp(self) -> str:
        return stamp_of(self.at())


def resolve_now(now: datetime | None) -> datetime:
    """A whole-second UTC instant; naive datetimes are a caller bug."""
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return moment.astimezone(UTC).replace(microsecond=0)


def stamp_of(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(STAMP_FORMAT)


def plus_seconds(moment: datetime, seconds: int) -> str:
    return stamp_of(moment + timedelta(seconds=seconds))


def reason_of(error: BaseException) -> str:
    """A stable code for ``error``; its message text is NEVER read."""
    from ssgrep.accounting.export import acks, otlp, spool as export_spool
    from ssgrep.accounting.privacy import storage

    if isinstance(error, CycleError):
        return error.reason
    condition = getattr(error, "condition", None)
    if isinstance(condition, str) and _CODE.fullmatch(condition):
        return condition
    if isinstance(error, export_spool.StateCorrupt | acks.DeliveryCorrupt):
        return "state_corrupt"
    if isinstance(error, export_spool.SpoolCorrupt):
        return "spool_corrupt"
    if isinstance(error, export_spool.BatchConflict):
        return "batch_conflict"
    if isinstance(error, otlp.ExportTooLarge):
        return "export_too_large"
    if isinstance(error, storage.UnsafeStorageError):
        return "storage_unsafe"
    if isinstance(error, OSError):
        return "storage_error"
    if isinstance(error, ValueError):
        return "invalid_state"
    return "internal_error"


def class_code(error: BaseException) -> str:
    """The exception CLASS name when it is a plain identifier (never the message)."""
    name = type(error).__name__
    return name if _CLASS.fullmatch(name) else "Exception"


__all__ = [
    "CheckResult",
    "CycleResult",
    "Deps",
    "ExportPaused",
    "check",
    "export_callback",
    "run_destination",
    "run_export",
]

LAST_ERROR = "last_error.json"
LOCK_NAME = "send-{signal}"
HOLD_RECOVERY = frozenset({"export_paused", "policy_unreadable"}) | AUTHORITY_HOLD_REASONS


def _result(cycle: Cycle, state: str, reason: str | None = None) -> CycleResult:
    tally = cycle.tally
    return CycleResult(
        destination=cycle.destination,
        signal=cycle.signal,
        state=state,
        reason=reason,
        enqueued_batches=tally.enqueued,
        invalidated_batches=tally.invalidated,
        sent=tally.sent,
        acked=tally.acked,
        partial=tally.partial,
        quarantined=tally.quarantined,
        retry_scheduled_utc=tally.retry_utc,
        blocked_reason=tally.blocked_reason,
        nonrecallable_batch=tally.nonrecallable,
        warnings=tuple(tally.warnings),
        error_category=tally.failure,
    )


def _finish(cycle: Cycle) -> CycleResult:
    tally = cycle.tally
    if tally.stop is not None:
        return _result(cycle, *tally.stop)
    if tally.enqueue_error is not None and not tally.sent:
        return _result(cycle, "failed", tally.enqueue_error)
    if tally.sent:
        return _result(cycle, "sent", tally.enqueue_error)
    if tally.idle == "no_snapshot" and not any(m.state == "queued" for m in _send.batches(cycle)):
        return _result(cycle, "no_snapshot", "no_snapshot")
    return _result(cycle, "idle", tally.idle)


def _has_state(cycle: Cycle) -> tuple[bool, bool]:
    """(anything on disk for this destination/signal, a crash leftover needs recovery)."""
    metas = export_spool.list_batches(cycle.profile, cycle.destination, cycle.signal)
    delivery = acks.read_delivery(cycle.profile, cycle.destination, cycle.signal)
    leftover = any(m.state == "in_flight" for m in metas) or (
        delivery is not None and delivery.in_flight is not None
    )
    return bool(metas) or delivery is not None, leftover


def _denied(cycle: Cycle, gate: export_policy.Gate) -> CycleResult:
    """No send is authorized: write nothing unless a leftover/queue needs handling."""
    cycle.gate = gate
    touches, leftover = _has_state(cycle)
    held = gate.reason in HOLD_RECOVERY
    if not touches or (held and not leftover):
        cycle.tally.stop = _send.denial_state(gate)
        return _finish(cycle)
    with export_spool.locked(
        cycle.profile,
        ("export", cycle.destination),
        LOCK_NAME.format(signal=cycle.signal.value),
        wait=False,
    ) as held_lock:
        if held_lock is None:
            return _result(cycle, "busy", "send_in_progress")
        _send.apply_denial(cycle, gate)
    return _finish(cycle)


def _unblock(cycle: Cycle) -> None:
    """Clear a block only when the credential, endpoint or proxy it depended on changed."""
    delivery = acks.read_delivery(cycle.profile, cycle.destination, cycle.signal)
    if delivery is None or delivery.blocked_reason is None:
        return
    fingerprint = _send.auth(cycle).fingerprint
    after = acks.update_delivery(
        cycle.profile,
        cycle.destination,
        cycle.signal,
        lambda d: acks.unblock_if_changed(d, fingerprint),
    )
    if after.blocked_reason is not None:
        cycle.tally.blocked_reason = after.blocked_reason
        cycle.tally.stop = ("blocked", f"blocked_{after.blocked_reason}")


def _authorized(cycle: Cycle, gate: export_policy.Gate) -> CycleResult:
    _send.refresh_scope(cycle, gate)
    _unblock(cycle)
    _send.recover(cycle)
    _send.enforce_retention(cycle)
    revalidate(cycle)
    if cycle.signal is Signal.LOGS:
        export_registry.require_queued_spool(cycle.profile, cycle.destination)
    if cycle.enqueue:
        enqueue(cycle)
    _send.send_loop(cycle)
    return _finish(cycle)


def run_destination(
    profile: str,
    destination: str,
    signal: Signal,
    *,
    now: datetime | None = None,
    enqueue: bool = True,
    deps: Deps = DEFAULT_DEPS,
) -> CycleResult:
    """One delivery cycle. A failing cycle returns ``failed`` with a stable code, never raises.

    Only programmer errors (malformed ``profile``/``destination`` ids, a naive ``now``) raise.
    """
    sig = Signal(signal)
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise ValueError("profile and destination must be bounded identifiers")
    cycle = Cycle(profile, destination, sig, resolve_now(now), now is not None, deps, enqueue)
    try:
        gate = export_policy.gate(profile, destination, sig)
        if not gate.ok:
            return _denied(cycle, gate)
        with export_spool.locked(
            profile, ("export", destination), LOCK_NAME.format(signal=sig.value), wait=False
        ) as held:
            if held is None:
                return _result(cycle, "busy", "send_in_progress")
            return _authorized(cycle, gate)
    except Exception as error:  # noqa: BLE001 - recorded as a stable code, never as text
        return _result(cycle, "failed", reason_of(error))


def _targets(profile: str) -> list[tuple[str, Signal]]:
    """Enabled pairs plus pairs still holding spooled data (so revocation can discard them)."""
    pairs: set[tuple[str, Signal]] = set(export_spool.iter_spools(profile))
    for destination, signal in policy.policy_status(profile).profile.enabled_exports:
        try:
            if valid_profile_id(destination):
                pairs.add((destination, Signal(signal)))
        except ValueError:
            continue
    return sorted(pairs, key=lambda pair: (pair[0], pair[1].value))


def run_export(
    profile: str, *, now: datetime | None = None, deps: Deps = DEFAULT_DEPS
) -> list[CycleResult]:
    """Every (destination, signal) cycle for ``profile``; NEVER raises."""
    try:
        targets = _targets(profile)
    except Exception:  # noqa: BLE001 - nothing to send to is not a crash
        return []
    results: list[CycleResult] = []
    for destination, sig in targets:
        results.append(run_destination(profile, destination, sig, now=now, deps=deps))
    return results


def _record_error(profile: str, code: str) -> None:
    """Bounded ``export/last_error.json`` (class name or stable code, timestamp; no text)."""
    document = {"schema": 1, "at": stamp_of(resolve_now(None)), "error": code}
    with export_spool.locked(profile, ("export",), "last_error") as dir_fd:
        assert dir_fd is not None
        export_spool.write_json(dir_fd, LAST_ERROR, document)


def _clear_error(profile: str) -> None:
    try:
        dir_fd = storage.open_profile_dir(profile, "export", create=False)
    except OSError:
        return
    try:
        storage.delete_file(dir_fd, LAST_ERROR)
    except OSError:
        pass  # absent (the common case) or unremovable: the next failure rewrites it anyway
    finally:
        os.close(dir_fd)


def export_callback(profile: str, *, deps: Deps = DEFAULT_DEPS) -> Callable[[], None]:
    """The zero-argument export hook for ``serve_loop.Worker``; swallows and records errors."""

    def run() -> None:
        try:
            results = run_export(profile, deps=deps)
            failures = [r for r in results if r.state == "failed" and r.reason is not None]
            if failures:
                _record_error(profile, failures[0].reason or "cycle_failed")
            else:
                _clear_error(profile)
        except Exception as error:  # noqa: BLE001 - the loop must keep running
            try:
                _record_error(profile, class_code(error))
            except Exception:  # noqa: BLE001, S110 - recording is best effort
                pass

    return run


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


"""Spool the current snapshot's batches (durably, before any send) and record the collection.

Order, so that a crash anywhere is repaired by the next cycle (batch writes are idempotent):

1. every batch file + meta is written and fsynced (``export_spool.write_batch``);
2. for logs, ALL batches are registered as ``queued`` in ONE atomic registry write;
3. ``acks.collected`` (+ ``ack_vacuous`` when nothing is to be sent) is persisted.

A generation is built only while it is not acknowledged, has no recorded progress and its batches
are not all on disk. A half-written set (fewer batches than ``total``, or logs not registered)
whose batches were never attempted is deleted and rebuilt: the bytes are deterministic, so the
rebuild is identical unless the ledger changed meanwhile, in which case the NEW state wins. Sending
never starts for a generation whose build failed in this cycle (``Cycle.skip``).

Failures are recorded as stable codes only (``reason_of``): an untrusted snapshot can make the
projection raise, and its text must never reach a result, a document or a log.
"""


def _registered(cycle: Cycle, metas: list[Meta]) -> bool:
    """Whether the registry holds every unattempted log batch as queued under its own id."""
    if cycle.signal is not Signal.LOGS:
        return True
    registry = export_registry.load_registry(cycle.profile, cycle.destination)
    for meta in metas:
        if meta.state != "queued" or meta.attempts:
            continue
        body = export_spool.read_body(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)
        for record_id, revision in export_registry.body_pairs(body):
            entry = registry.entries.get(record_id)
            if entry is None or not any(
                s.batch == meta.batch_id and s.revision == revision for s in entry.queued
            ):
                return False
    return True


def _complete(cycle: Cycle, metas: list[Meta]) -> bool:
    total = metas[0].total
    same = all(m.total == total and m.snapshot_id == metas[0].snapshot_id for m in metas)
    return same and {m.index for m in metas} == set(range(total)) and _registered(cycle, metas)


def _drop(cycle: Cycle, metas: list[Meta]) -> None:
    for meta in metas:
        if cycle.signal is Signal.LOGS:
            export_registry.reset_queued(cycle.profile, cycle.destination, meta.batch_id)
        export_spool.delete_batch(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)


def _pending_build(cycle: Cycle, envelope: SnapshotEnvelope) -> bool:
    """True when ``envelope``'s generation still has to be (re)built."""
    delivery = acks.read_delivery(cycle.profile, cycle.destination, cycle.signal)
    generation = envelope.generation
    if delivery is not None:
        if generation <= (delivery.last_acked_generation or 0):
            return False
        if any(p.generation == generation for p in delivery.progress):
            return False
    existing = [m for m in _send.batches(cycle) if m.kind == "data" and m.generation == generation]
    if not existing:
        return True
    if _complete(cycle, existing):
        acks.update_delivery(  # a crash after the spool write must still record the collection
            cycle.profile, cycle.destination, cycle.signal, lambda d: acks.collected(d, generation)
        )
        return False
    if all(m.state == "queued" and not m.attempts for m in existing):
        _drop(cycle, existing)  # half-built: never sent, deterministic, safe to rebuild
        return True
    return False


def _build(cycle: Cycle, envelope: SnapshotEnvelope) -> export_build.Candidate:
    """Project the snapshot; a projection that rejects it is the stable ``snapshot_invalid``.

    A metrics projection that drops EVERY point of a non-empty snapshot (e.g. an unparseable
    ``collected_at``) exported nothing: that is a failure, never a vacuous acknowledgment.
    """
    assert cycle.key is not None
    try:
        if cycle.signal is Signal.METRICS:
            built = export_build.build_metrics(
                cycle.profile, cycle.dest, cycle.policy, envelope, cycle.key
            )
        else:
            built = export_build.build_logs(
                cycle.profile, cycle.dest, cycle.policy, envelope, cycle.key
            )
    except ValueError as error:
        if reason_of(error) == "invalid_state":  # untrusted snapshot text: code only
            raise CycleError("snapshot_invalid") from None
        raise
    if not built.batches and built.omitted and cycle.signal is Signal.METRICS:
        raise CycleError("snapshot_unexportable")
    return built


def _record(cycle: Cycle, built: export_build.Candidate) -> None:
    """Write, register, then record the collection (vacuous acknowledgment when empty)."""
    stamp = cycle.stamp()
    for batch in built.batches:
        export_spool.write_batch(
            cycle.profile,
            cycle.destination,
            cycle.signal,
            batch,
            created_utc=stamp,
            scope=built.scope,
        )
    if cycle.signal is Signal.LOGS and built.batches:
        export_registry.queue_batches(
            cycle.profile,
            cycle.destination,
            [
                (b.batch_id, members)
                for b, members in zip(built.batches, built.planned, strict=True)
            ],
        )

    def collect(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, built.generation)
        if not built.batches:
            d = acks.ack_vacuous(d, built.generation, built.snapshot_id, stamp)
        return d

    acks.update_delivery(cycle.profile, cycle.destination, cycle.signal, collect)
    _send.sync_counts(cycle.profile, cycle.destination, cycle.signal)
    cycle.tally.enqueued += len(built.batches)


def enqueue(cycle: Cycle) -> None:
    """Build and spool the current snapshot when it needs it; never raises (codes only)."""
    tally = cycle.tally
    envelope = export_build.current_envelope(cycle.profile)
    if envelope is None:
        tally.idle = "no_snapshot"
        return
    generation = envelope.generation
    try:
        if not _pending_build(cycle, envelope):
            return
        if cycle.scope_state is None or not export_policy.snapshot_eligible(
            envelope, cycle.scope_state, cycle.policy.policy_version
        ):
            tally.idle = "ineligible_scope"
            return
        _record(cycle, _build(cycle, envelope))
    except Exception as error:  # noqa: BLE001 - a bad snapshot/state must not stop other sends
        cycle.skip.add(generation)
        tally.enqueue_error = reason_of(error)


"""Pre-dispatch revalidation (design D11): queued bytes are only ever as good as TODAY's policy.

``revalidate`` compares the scope document every queued batch was created under with the scope
document today's policy would authorize (``export_policy.batch_current``; tombstones are judged
only by destination/endpoint/identity/logs permission). A stale batch is NEVER sent:

* its unsent bytes are deleted, its registry ``queued`` entries are reset (logs) and its
  generation's progress is dropped (``acks.reset_generation``);
* when the generation is the current snapshot and that snapshot may still leave the machine under
  the new scope (``snapshot_eligible``), it is REPROJECTED by the enqueue step (same batch ids,
  new bytes);
* otherwise the loss is explicit: progress ``discarded`` for each unsent index (the generation can
  never become fully acknowledged) and a ``policy_contraction`` loss summary.

``apply_denial`` is the same story when sending is not authorized at all right now: pause and a
disabled signal HOLD the bounded queue; a revoked destination/signal discards it
(``policy_contraction``). A leftover ``in_flight`` batch of a dead process is returned to the
queue as unknown and reported as ``in_flight_nonrecallable``: it may already have been delivered.
"""


LOSS = "policy_contraction"


def _remove(cycle: Cycle, meta: Meta) -> None:
    """Registry first (a crash in between is repaired by the next pass), then the files."""
    if cycle.signal is Signal.LOGS:
        export_registry.reset_queued(cycle.profile, cycle.destination, meta.batch_id)
    export_spool.delete_batch(cycle.profile, cycle.destination, cycle.signal, meta.batch_id)


def _reprojectable(cycle: Cycle, generation: int) -> bool:
    """The snapshot of ``generation`` is current and may still leave under today's scope."""
    if not cycle.enqueue or cycle.scope_state is None:
        return False
    envelope = export_build.current_envelope(cycle.profile)
    return (
        envelope is not None
        and envelope.generation == generation
        and export_policy.snapshot_eligible(
            envelope, cycle.scope_state, cycle.policy.policy_version
        )
    )


def _discard_generation(
    cycle: Cycle, generation: int, all_metas: list[Meta], *, reproject_ok: bool
):
    members = [m for m in all_metas if m.kind == "data" and m.generation == generation]
    live = [m for m in members if m.state in _send.LIVE]
    stamp = cycle.stamp()
    again = reproject_ok and _reprojectable(cycle, generation)
    for meta in members if again else live:  # a rebuild replaces settled batches too
        _remove(cycle, meta)

    def change(d: acks.Delivery) -> acks.Delivery:
        if again:
            return acks.reset_generation(d, generation)
        for meta in live:
            d = acks.outcome(
                d,
                generation=generation,
                snapshot_id=meta.snapshot_id,
                index=meta.index,
                total=meta.total,
                kind="discarded",
                now=stamp,
            )
        return acks.record_loss(d, LOSS, len(live), sum(m.items for m in live), stamp)

    acks.update_delivery(cycle.profile, cycle.destination, cycle.signal, change)
    cycle.tally.invalidated += len(live)


def discard(cycle: Cycle, stale: list[Meta], *, reproject_ok: bool) -> None:
    """Delete ``stale`` unsent batches (and their generations' housekeeping) with explicit loss."""
    all_metas = _send.batches(cycle)
    for generation in sorted({m.generation for m in stale if m.kind == "data"}):
        _discard_generation(cycle, generation, all_metas, reproject_ok=reproject_ok)
    tombstones = [m for m in stale if m.kind == "tombstone"]
    if tombstones:
        stamp = cycle.stamp()
        for meta in tombstones:
            _remove(cycle, meta)
        acks.update_delivery(
            cycle.profile,
            cycle.destination,
            cycle.signal,
            lambda d: acks.record_loss(
                d, LOSS, len(tombstones), sum(m.items for m in tombstones), stamp
            ),
        )
        cycle.tally.invalidated += len(tombstones)
    _send.sync_counts(cycle.profile, cycle.destination, cycle.signal)


def is_current(cycle: Cycle, meta: Meta) -> bool:
    """May ``meta``'s immutable bytes still be sent under TODAY's scope? (the one judgement)"""
    if not export_policy.batch_current(meta.scope, cycle.scope, meta.kind):
        return False
    if meta.kind == "tombstone":
        return True
    assert cycle.scope_state is not None  # every caller refreshes the authorized scope first
    try:
        envelope = snapshots.read_generation(cycle.profile, meta.generation)
    except (OSError, ValueError):
        return False
    return (
        envelope is not None
        and envelope.snapshot_id == meta.snapshot_id
        and export_policy.snapshot_eligible(
            envelope, cycle.scope_state, cycle.policy.policy_version
        )
    )


def revalidate(cycle: Cycle) -> None:
    """Invalidate every queued batch whose creation scope is not today's scope."""
    stale = [m for m in _send.batches(cycle) if m.state in _send.LIVE and not is_current(cycle, m)]
    if stale:
        discard(cycle, stale, reproject_ok=True)


# Import after defining the delivery contract; delivery_send imports its Cycle types from here.
from ssgrep.accounting.export import delivery_send as _send  # noqa: E402
