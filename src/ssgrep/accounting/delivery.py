"""Organization export delivery: one locked cycle per destination/signal (design D10/D11).

Public API: ``run_destination`` (one cycle), ``run_export`` (every destination/signal that is
enabled or still holds queued data; never raises), ``export_callback`` (the zero-argument hook for
``serve_loop.Worker``; never raises), ``check`` (``--check``: an empty OTLP request that writes
nothing) and the read-only ``preview_candidate``.

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

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime

from ssgrep.accounting import acks, export_files, export_policy, export_spool, policy, storage
from ssgrep.accounting.delivery_candidate import PreviewCandidate, preview_candidate
from ssgrep.accounting.delivery_check import check
from ssgrep.accounting.delivery_enqueue import enqueue
from ssgrep.accounting.delivery_invalidate import apply_denial, denial_state, revalidate
from ssgrep.accounting.delivery_registry import require_queued_spool
from ssgrep.accounting.delivery_send import send_loop
from ssgrep.accounting.delivery_state import (
    auth,
    batches,
    enforce_retention,
    recover,
    refresh_scope,
)
from ssgrep.accounting.delivery_types import (
    DEFAULT_DEPS,
    CheckResult,
    Cycle,
    CycleResult,
    Deps,
    ExportPaused,
    class_code,
    reason_of,
    resolve_now,
    stamp_of,
)
from ssgrep.accounting.ledger_contracts import Signal, valid_profile_id

__all__ = [
    "CheckResult",
    "CycleResult",
    "Deps",
    "ExportPaused",
    "PreviewCandidate",
    "check",
    "export_callback",
    "preview_candidate",
    "run_destination",
    "run_export",
]

LAST_ERROR = "last_error.json"
LOCK_NAME = "send-{signal}"
HOLD_RECOVERY = frozenset({"export_paused", "policy_unreadable"})


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
    if tally.idle == "no_snapshot" and not any(m.state == "queued" for m in batches(cycle)):
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
        cycle.tally.stop = denial_state(gate)
        return _finish(cycle)
    with export_files.locked(
        cycle.profile,
        ("export", cycle.destination),
        LOCK_NAME.format(signal=cycle.signal.value),
        wait=False,
    ) as held_lock:
        if held_lock is None:
            return _result(cycle, "busy", "send_in_progress")
        apply_denial(cycle, gate)
    return _finish(cycle)


def _unblock(cycle: Cycle) -> None:
    """Clear a block only when the credential, endpoint or proxy it depended on changed."""
    delivery = acks.read_delivery(cycle.profile, cycle.destination, cycle.signal)
    if delivery is None or delivery.blocked_reason is None:
        return
    fingerprint = auth(cycle).fingerprint
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
    refresh_scope(cycle, gate)
    _unblock(cycle)
    recover(cycle)
    enforce_retention(cycle)
    revalidate(cycle)
    if cycle.signal is Signal.LOGS:
        require_queued_spool(cycle.profile, cycle.destination)
    if cycle.enqueue:
        enqueue(cycle)
    send_loop(cycle)
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
        with export_files.locked(
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
    with export_files.locked(profile, ("export",), "last_error") as dir_fd:
        assert dir_fd is not None
        export_files.write_json(dir_fd, LAST_ERROR, document)


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
