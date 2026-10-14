"""Pre-dispatch revalidation: may THIS destination/signal be sent to, with THESE bytes, NOW?

Authorization is never inherited from the moment a batch was queued (design D11). Immediately
before every dispatch ``gate`` rechecks the profile's current policy (readable, not paused,
sending not held paused by an invalid managed update, destination allowlisted, signal
permitted, logs separately permitted, endpoint/proxy still valid, destination+signal
explicitly enabled) and ``scope_document`` rebuilds the identity of what the policy would
authorize today. A queued batch is sendable only when the scope document it was created under
equals today's; otherwise its bytes are invalid and must be reprojected or discarded.

Snapshots carry the digest of the actual collection authority (canonical workspace roots,
assignments and profile ownership). The first export cannot bless older or unknown authority:
the snapshot provenance must match today's scope. Scope-state corruption requires a collection
after recovery; it is never treated as an unobserved first send. Destination consent and
credential rotation do not change the collection authority.
Deletion requests (tombstones) are judged only by destination/endpoint/identity/logs permission:
narrowing roots must never block a request to delete data that was already exported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from ssgrep.accounting import export_files, policy, storage
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.export_ids import pseudonym
from ssgrep.accounting.export_model import FIELDS_VERSION
from ssgrep.accounting.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    SnapshotEnvelope,
)
from ssgrep.accounting.plain_codec import loads
from ssgrep.accounting.policy_rules import (
    PolicyError,
    check_endpoint,
    check_proxy,
    valid_collection_scope,
)

SCOPE_DIR = "export"
SCOPE_SCHEMA = 1
DATA_KEYS = ("collection_scope", "fields", "policy_version", "logs_permitted")
TOMBSTONE_KEYS = ("endpoint", "proxy", "fields", "logs_permitted", "identity")
BLOCKING_REASONS = frozenset({"policy_unreadable", "unknown_destination", "signal_not_permitted"})


@dataclass(frozen=True)
class Gate:
    """The verdict of one authorization check; ``reason`` is a stable machine code."""

    ok: bool
    reason: str | None
    detail: str
    policy: ProfilePolicy
    destination: DestinationPolicy | None

    @property
    def revoked(self) -> bool:
        """The destination/signal is no longer allowed by policy (not merely disabled/paused)."""
        return self.reason in {"unknown_destination", "signal_not_permitted", "logs_not_permitted"}


def _denied(
    reason: str, detail: str, current: ProfilePolicy, dest: DestinationPolicy | None
) -> Gate:
    return Gate(False, reason, detail, current, dest)


def gate(
    profile: str,
    destination_id: str,
    signal: Signal,
    *,
    need_enabled: bool = True,
    need_unpaused: bool = True,
) -> Gate:
    """Authorize one destination/signal against the CURRENT policy; reads only, never writes."""
    status = policy.policy_status(profile)
    current = status.profile
    if status.error is not None:
        return _denied("policy_unreadable", status.error.condition, current, None)
    dest = next((d for d in current.destinations if d.destination_id == destination_id), None)
    if dest is None:
        return _denied(
            "unknown_destination", "destination is not in the profile policy", current, None
        )
    if signal not in dest.signals:
        return _denied("signal_not_permitted", f"{signal.value} is not permitted", current, dest)
    if signal is Signal.LOGS and not dest.logs_permitted:
        return _denied("logs_not_permitted", "logs need explicit policy permission", current, dest)
    try:
        check_endpoint(dest.endpoint)
        if dest.proxy is not None:
            check_proxy(dest.proxy)
    except PolicyError as error:
        return _denied("endpoint_invalid", str(error), current, dest)
    if need_unpaused:
        try:
            policy.require_not_paused(profile, "export")
        except PolicyError as error:
            return _denied(error.condition, str(error), current, dest)
    if need_enabled and (destination_id, signal.value) not in current.enabled_exports:
        return _denied("not_enabled", "this destination/signal was not enabled", current, dest)
    return Gate(True, None, "", current, dest)


def scope_document(
    current: ProfilePolicy, dest: DestinationPolicy, signal: Signal, installation: str
) -> dict[str, Any]:
    """What a payload built today would be authorized under (plain, comparable JSON)."""
    return {
        "roots": sorted(current.roots),
        "collection_scope": policy.collection_scope(current),
        "endpoint": dest.endpoint,
        "proxy": dest.proxy,
        "fields": FIELDS_VERSION[signal.value],
        "policy_version": current.policy_version,
        "logs_permitted": bool(dest.logs_permitted) if signal is Signal.LOGS else False,
        "identity": installation,
    }


def scope_changes(old: dict[str, Any], new: dict[str, Any]) -> tuple[str, ...]:
    """Names of scope keys whose value differs (values themselves are never reported)."""
    return tuple(sorted(k for k in {*old, *new} if old.get(k) != new.get(k)))


def credential_fingerprint(
    key: bytes, destination: str, dest: DestinationPolicy, secret: str | None
) -> str:
    """Keyed fingerprint of what a block depends on: credential value, endpoint, proxy."""
    return pseudonym(key, "credential", destination, dest.endpoint, dest.proxy or "", secret or "")


# ---- scope tracking ---------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeState:
    digest: str
    changed_at: str | None
    collection_scope: dict[str, Any] | None = None


def _scope_file(signal: Signal) -> str:
    return f"scope-{signal.value}.json"


def _dirs(destination: str) -> tuple[str, ...]:
    return (SCOPE_DIR, destination)


def _parse(raw: Any) -> ScopeState | None:
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "digest", "changed_at", "collection_scope"}
        or type(raw.get("schema")) is not int
        or raw["schema"] != SCOPE_SCHEMA
    ):
        return None
    digest, changed = raw.get("digest"), raw.get("changed_at")
    provenance = raw.get("collection_scope")
    if not valid_collection_scope(provenance):
        return None
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    if any(char not in "0123456789abcdef" for char in digest):
        return None
    if changed is not None:
        try:
            datetime.strptime(changed, "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError):
            return None
    return ScopeState(digest, changed, provenance)


def _read_scope_fd(dir_fd: int, signal: Signal) -> ScopeState | None:
    try:
        data = storage.read_file(dir_fd, _scope_file(signal), export_files.MAX_STATE_BYTES)
        if data is None:
            return None
        return _parse(loads(data)) or ScopeState("", "")
    except (OSError, ValueError):
        return ScopeState("", "")


def read_scope(profile: str, destination: str, signal: Signal) -> ScopeState | None:
    """None only when absent; corrupt state is an invalid sentinel, never first-send authority."""
    try:
        dir_fd = storage.open_profile_dir(profile, *_dirs(destination), create=False)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return ScopeState("", "")
    try:
        return _read_scope_fd(dir_fd, signal)
    finally:
        os.close(dir_fd)


def next_scope(scope: dict[str, Any], known: ScopeState | None, now: str) -> ScopeState:
    """The same fail-closed transition for preview and durable observation."""
    digest = digest_of({k: scope.get(k) for k in DATA_KEYS})
    changed = None if known is None else (now if not known.digest else known.changed_at)
    return ScopeState(digest, changed, scope.get("collection_scope"))


def observe_scope(
    profile: str, destination: str, signal: Signal, scope: dict[str, Any], now: str
) -> ScopeState:
    """Persist collection authority; invalid prior state installs a recollection barrier."""
    with export_files.locked(profile, _dirs(destination), _scope_file(signal)) as dir_fd:
        assert dir_fd is not None
        known = _read_scope_fd(dir_fd, signal)
        state = next_scope(scope, known, now)
        if state == known:
            return state
        export_files.write_json(
            dir_fd,
            _scope_file(signal),
            {
                "schema": SCOPE_SCHEMA,
                "digest": state.digest,
                "changed_at": state.changed_at,
                "collection_scope": state.collection_scope,
            },
        )
        return state


def snapshot_eligible(envelope: SnapshotEnvelope, state: ScopeState, current_version: int) -> bool:
    """Actual collection provenance must match current authority, even on the first send."""
    provenance = envelope.coverage.get("collection_scope")
    if (
        envelope.policy_version != current_version
        or not valid_collection_scope(provenance)
        or not valid_collection_scope(state.collection_scope)
        or provenance != state.collection_scope
    ):
        return False
    return state.changed_at is None or envelope.collected_at > state.changed_at


def batch_current(batch_scope: Any, scope: dict[str, Any], kind: str = "data") -> bool:
    """Whether bytes queued under ``batch_scope`` may still be sent under today's ``scope``."""
    if not isinstance(batch_scope, dict):
        return False
    keys = TOMBSTONE_KEYS if kind == "tombstone" else tuple(scope)
    return all(batch_scope.get(k) == scope.get(k) for k in keys)


def with_enabled(
    current: ProfilePolicy, destination: str, signal: Signal, on: bool
) -> ProfilePolicy:
    """The profile with ``(destination, signal)`` added/removed from ``enabled_exports``."""
    pair = (destination, signal.value)
    kept = tuple(p for p in current.enabled_exports if p != pair)
    return replace(current, enabled_exports=tuple(sorted({*kept, *(({pair}) if on else set())})))
