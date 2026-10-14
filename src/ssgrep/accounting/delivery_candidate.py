"""Candidate selection shared by ``--preview`` and the enqueue step of a delivery cycle.

``preview_candidate`` is read-only: it observes no scope, mints no pseudonym key, and writes no
registry/spool/delivery state. It goes through the same ``export_build.build_*`` functions as the
enqueue step, so the bytes a preview shows equal the bytes a cycle would spool for the same
snapshot, policy and key (``ephemeral_key`` says when no key exists yet).

States of :class:`PreviewCandidate` (``candidate`` is present for ``ok``, ``no_changes`` and
``no_records``; absent otherwise):

* ``no_snapshot``     no readable current snapshot;
* ``policy_blocked``  the policy gate denies this destination/signal (``destination`` may still
                      name the policy entry); pause and enablement are NOT checked here;
* ``ineligible_scope`` the snapshot predates the last data-affecting scope change or was
                      collected under another policy version: it can never be exported;
* ``ok``              at least one batch would be spooled;
* ``no_changes``      logs only: every current record is already known to the registry;
* ``no_records``      nothing to export at all (no ledger rows, or an empty projection).

Ledger/registry problems propagate as ``RecordsUnavailable`` / ``RegistryCorrupt`` (stable
``condition`` codes, no raw text) rather than being flattened into a state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ssgrep.accounting import export_build, export_ids, export_policy, export_records, snapshots
from ssgrep.accounting.export_policy import Gate, ScopeState
from ssgrep.accounting.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
)

STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
STATES = (
    "ok",
    "no_snapshot",
    "ineligible_scope",
    "no_changes",
    "policy_blocked",
    "no_records",
)


@dataclass(frozen=True)
class PreviewCandidate:
    state: str
    candidate: export_build.Candidate | None
    envelope: SnapshotEnvelope | None
    destination: DestinationPolicy | None
    policy: ProfilePolicy
    ephemeral_key: bool


def current_envelope(profile: str) -> SnapshotEnvelope | None:
    """The verified current snapshot; ``None`` when absent or unreadable. Never writes."""
    try:
        generation = snapshots.current_generation(profile)
        return None if generation is None else snapshots.read_generation(profile, generation)
    except (OSError, ValueError):
        return None


def _simulated_scope(
    profile: str, destination: str, signal: Signal, scope: dict[str, object], stamp: str
) -> ScopeState:
    """The scope state a cycle WOULD act on, without persisting anything."""
    known = export_policy.read_scope(profile, destination, signal)
    return export_policy.next_scope(scope, known, stamp)


def _select(
    profile: str, destination: str, signal: Signal, gate: Gate, stamp: str
) -> PreviewCandidate:
    if not gate.ok or gate.destination is None:
        return PreviewCandidate("policy_blocked", None, None, gate.destination, gate.policy, False)
    dest, current = gate.destination, gate.policy
    envelope = current_envelope(profile)
    if envelope is None:
        return PreviewCandidate("no_snapshot", None, None, dest, current, False)
    key, ephemeral = export_build.key_for(profile, create=False)
    installation = export_ids.installation_id(key, destination, profile)
    scope = export_policy.scope_document(current, dest, signal, installation)
    state = _simulated_scope(profile, destination, signal, scope, stamp)
    if not export_policy.snapshot_eligible(envelope, state, current.policy_version):
        return PreviewCandidate("ineligible_scope", None, envelope, dest, current, ephemeral)
    if signal is Signal.METRICS:
        built = export_build.build_metrics(
            profile, dest, current, envelope, key, ephemeral=ephemeral
        )
    else:
        built = export_build.build_logs(profile, dest, current, envelope, key, ephemeral=ephemeral)
    if built.batches:
        verdict = "ok"
    elif signal is Signal.LOGS and export_records.load_registry(profile, destination).entries:
        verdict = "no_changes"
    else:
        verdict = "no_records"
    return PreviewCandidate(verdict, built, envelope, dest, current, ephemeral)


def preview_candidate(
    profile: str, destination: str, signal: Signal, *, now: datetime | None = None
) -> PreviewCandidate:
    """The exact sanitized candidate a send would spool now. Writes NOTHING (no key, no scope)."""
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    sig = Signal(signal)
    gate = export_policy.gate(profile, destination, sig, need_enabled=False, need_unpaused=False)
    return _select(profile, destination, sig, gate, moment.strftime(STAMP_FORMAT))
