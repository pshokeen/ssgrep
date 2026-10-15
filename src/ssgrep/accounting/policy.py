"""Profile policy: owner-only persistence, narrowing-only validation, pause state.

One file, ``<data>/accounting/policy.json`` (0600, atomic replace), holds every
profile because the eight-profile cap and overlapping-root checks are
cross-profile. Reads never create anything. Every mutation is validated
against the current file *before* the first byte is written, so a rejected
change leaves the data root untouched. Capture reads ``load_profile_policy``;
a missing, unreadable, or unsafe file yields a disabled profile, never a
permissive default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ssgrep.accounting import plainfile
from ssgrep.accounting.contracts import PERSONAL_PROFILE, digest_of
from ssgrep.accounting.ledger_contracts import (
    EVIDENCE_RETENTION_DAYS,
    MAX_PROFILES,
    RECEIPT_RETENTION_DAYS,
    ProfilePolicy,
)
from ssgrep.accounting.policy_codec import MAX_POLICY_BYTES, dump, parse, profile_from_dict
from ssgrep.accounting.policy_rules import (
    PolicyError,
    PolicyFile,
    canonical_root,
    check_profile_id,
    clamp_to_ceiling,
    validate_profile,
    workspace_owner,
)
from ssgrep.store.paths import data_dir

PauseKind = Literal["collection", "export"]
_POLICY_KEYS = frozenset(
    {
        "profile_id",
        "roots",
        "destinations",
        "evidence_retention_days",
        "receipt_retention_days",
        "policy_version",
        "policy_source",
        "assignments",
        "evidence_cache_enabled",
        "evidence_cache_retention_days",
        "evidence_cache_max_bytes",
    }
)
_MAX_CEILING_DAYS = 3650


def policy_path() -> Path:
    return data_dir() / "accounting" / "policy.json"


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_policy_file() -> PolicyFile:
    """The persisted policy (empty when absent). Raises ``PolicyError`` if unusable."""
    try:
        data = plainfile.read_private(
            data_dir(), "accounting", "policy.json", max_bytes=MAX_POLICY_BYTES
        )
    except (plainfile.UnsafePathError, ValueError, OSError) as error:
        raise PolicyError(
            "policy_corrupt",
            f"Accounting policy cannot be read safely ({error}); sending is paused.",
            "ssgrep metrics setup",
            exit_code=1,
        ) from error
    return PolicyFile() if data is None else parse(data)


def load_profile_policy(profile_id: str = PERSONAL_PROFILE) -> ProfilePolicy:
    """Effective policy for ``profile_id``; disabled when absent or unreadable."""
    default = ProfilePolicy(profile_id=profile_id)
    try:
        file = read_policy_file()
    except PolicyError:
        return default
    return file.profiles.get(profile_id, default)


def effective_ownership(policy: ProfilePolicy, ownership: PolicyFile | None = None) -> PolicyFile:
    """Freeze the ownership map with the explicitly supplied active profile policy."""
    file = read_policy_file() if ownership is None else ownership
    return replace(file, profiles={**file.profiles, policy.profile_id: policy})


def path_in_scope(policy: ProfilePolicy, cwd: str, ownership: PolicyFile | None = None) -> bool:
    """Whether canonical ``cwd`` belongs to this profile, including explicit assignments."""
    try:
        return workspace_owner(effective_ownership(policy, ownership), cwd) == policy.profile_id
    except PolicyError:
        return False


def collection_scope(policy: ProfilePolicy, ownership: PolicyFile | None = None) -> dict[str, Any]:
    """Digest the actual data permissions, independent of export consent and credentials."""
    file = effective_ownership(policy, ownership)
    document = {
        "schema": 1,
        "profile": policy.profile_id,
        "owners": {
            owner: sorted(set(item.roots))
            for owner, item in sorted(file.profiles.items())
            if owner != PERSONAL_PROFILE or owner == policy.profile_id
        },
        "assignments": sorted(file.assignments.items()),
    }
    return {"schema": 1, "profile": policy.profile_id, "digest": digest_of(document)}


@dataclass(frozen=True)
class PolicyStatus:
    profile: ProfilePolicy
    exists: bool
    sending_paused_reason: str | None
    error: PolicyError | None
    gaps: tuple[tuple[str, str | None], ...]
    assignments: tuple[tuple[str, str], ...] = ()


def policy_status(profile_id: str = PERSONAL_PROFILE) -> PolicyStatus:
    """Read-only view of one profile's policy state (never writes)."""
    check_profile_id(profile_id)
    try:
        file = read_policy_file()
    except PolicyError as error:
        return PolicyStatus(ProfilePolicy(profile_id=profile_id), True, error.condition, error, ())
    profile = file.profiles.get(profile_id)
    gaps = tuple(
        (str(g[0]), g[1] if g[1] is None else str(g[1])) for g in file.gaps.get(profile_id, [])
    )
    return PolicyStatus(
        profile or ProfilePolicy(profile_id=profile_id),
        profile is not None,
        file.sending_paused.get(profile_id),
        None,
        gaps,
        tuple(sorted(file.assignments.items())),
    )


def _commit(plan: Callable[[PolicyFile], PolicyFile]) -> PolicyFile:
    """Validate ``plan`` first (no side effects), then persist under a lock.

    A plan that changes nothing writes nothing, so no-op calls never create files.
    """
    current = read_policy_file()
    if plan(current) == current:
        return current
    base = data_dir()
    with plainfile.exclusive_lock(base, "accounting"):
        latest = read_policy_file()
        updated = plan(latest)
        if updated != latest:
            plainfile.write_private(base, "accounting", "policy.json", data=dump(updated))
    return updated


def _ceiling_from_raw(profile_id: str, raw: Any) -> ProfilePolicy:
    if not isinstance(raw, dict) or set(raw) - _POLICY_KEYS:
        raise PolicyError("invalid_policy", "Policy must be an object with only known keys.")
    given = raw.get("profile_id", profile_id)
    if given != profile_id:
        raise PolicyError("invalid_policy", f"Policy is for profile {given!r}, not {profile_id!r}.")
    body = {k: v for k, v in raw.items() if k != "assignments"}
    body.setdefault("evidence_retention_days", EVIDENCE_RETENTION_DAYS)
    body.setdefault("receipt_retention_days", RECEIPT_RETENTION_DAYS)
    body.setdefault("policy_source", "managed")
    body["profile_id"] = profile_id
    body["managed"] = True
    try:
        ceiling = profile_from_dict(body)
    except PolicyError as error:
        raise PolicyError("invalid_policy", str(error)) from error
    for days in (ceiling.evidence_retention_days, ceiling.receipt_retention_days):
        if not 1 <= days <= _MAX_CEILING_DAYS:
            raise PolicyError("invalid_policy", "Retention ceilings must be 1-3650 days.")
    return replace(ceiling, roots=tuple(sorted({canonical_root(r) for r in ceiling.roots})))


def _assignments(raw: Any) -> dict[str, str]:
    mapping = raw.get("assignments", {}) if isinstance(raw, dict) else {}
    if not isinstance(mapping, dict):
        raise PolicyError("invalid_policy", "assignments must map workspace roots to profile ids.")
    return {canonical_root(k): check_profile_id(v) for k, v in mapping.items()}


def plan_enable(
    file: PolicyFile,
    profile_id: str,
    raw_policy: Any = None,
    *,
    evidence_cache: tuple[int, int] | None = None,
) -> tuple[PolicyFile, ProfilePolicy]:
    """The validated file/profile that enabling local measurement would produce."""
    check_profile_id(profile_id)
    ceilings = dict(file.ceilings)
    assignments = dict(file.assignments)
    current = file.profiles.get(profile_id, ProfilePolicy(profile_id=profile_id))
    if raw_policy is not None:
        ceiling = _ceiling_from_raw(profile_id, raw_policy)
        assignments.update(_assignments(raw_policy))
        ceilings[profile_id] = ceiling
        current = clamp_to_ceiling(current, ceiling)
        known = {d.destination_id for d in current.destinations}
        current = replace(
            current,
            destinations=(
                *current.destinations,
                *(d for d in ceiling.destinations if d.destination_id not in known),
            ),
        )
    staged = replace(file, ceilings=ceilings, assignments=assignments)
    if evidence_cache is not None:
        days, size = evidence_cache
        current = replace(
            current,
            evidence_cache_enabled=bool(days or size),
            evidence_cache_retention_days=days,
            evidence_cache_max_bytes=size,
        )
    validated = validate_profile(replace(current, collection_enabled=True), staged)
    sending = {k: v for k, v in file.sending_paused.items() if k != profile_id}
    return replace(
        staged, sending_paused=sending, profiles={**file.profiles, profile_id: validated}
    ), validated


def preview_enable(
    profile_id: str, raw_policy: Any = None, *, evidence_cache: tuple[int, int] | None = None
) -> ProfilePolicy:
    """What setup would approve; validates fully and writes nothing."""
    return plan_enable(read_policy_file(), profile_id, raw_policy, evidence_cache=evidence_cache)[1]


def enable_collection(
    profile_id: str, raw_policy: Any = None, *, evidence_cache: tuple[int, int] | None = None
) -> ProfilePolicy:
    """Persist local measurement as enabled; export stays disabled."""
    result: list[ProfilePolicy] = []

    def plan(file: PolicyFile) -> PolicyFile:
        updated, profile = plan_enable(file, profile_id, raw_policy, evidence_cache=evidence_cache)
        result[:] = [profile]
        return updated

    _commit(plan)
    from ssgrep.accounting.evidence_cache import maintain_cache

    maintain_cache(profile_id, result[0])
    return result[0]


def save_profile(candidate: ProfilePolicy) -> ProfilePolicy:
    """Validate a locally edited profile (narrowing only) and persist it atomically."""
    result: list[ProfilePolicy] = []

    def plan(file: PolicyFile) -> PolicyFile:
        validated = validate_profile(candidate, file)
        result[:] = [validated]
        return replace(file, profiles={**file.profiles, candidate.profile_id: validated})

    _commit(plan)
    from ssgrep.accounting.evidence_cache import maintain_cache

    maintain_cache(candidate.profile_id, result[0])
    return result[0]


def apply_managed_update(profile_id: str, raw_policy: Any) -> tuple[bool, PolicyError | None]:
    """Refresh a managed policy; invalid input keeps the last valid one.

    On rejection the previous policy remains in force, sending is marked
    paused with the rejection condition, and no permissive fallback occurs.
    """
    check_profile_id(profile_id)
    current = read_policy_file()
    enabled = current.profiles.get(
        profile_id, ProfilePolicy(profile_id=profile_id)
    ).collection_enabled
    try:
        plan_enable(current, profile_id, raw_policy)
    except PolicyError as error:
        condition = error.condition
        _commit(lambda f: replace(f, sending_paused={**f.sending_paused, profile_id: condition}))
        return False, error

    def plan(file: PolicyFile) -> PolicyFile:
        updated, profile = plan_enable(file, profile_id, raw_policy)
        profiles = {**updated.profiles, profile_id: replace(profile, collection_enabled=enabled)}
        return replace(updated, profiles=profiles)

    updated = _commit(plan)
    from ssgrep.accounting.evidence_cache import maintain_cache

    maintain_cache(profile_id, updated.profiles[profile_id])
    return True, None


def pause(profile_id: str, now: datetime | None = None) -> ProfilePolicy:
    """Pause capture/collection/sending; non-destructive and idempotent."""
    stamp = _stamp(_now(now))
    result: list[ProfilePolicy] = []

    def plan(file: PolicyFile) -> PolicyFile:
        current = file.profiles.get(profile_id, ProfilePolicy(profile_id=profile_id))
        if current.paused:
            result[:] = [current]
            return file
        if profile_id not in file.profiles and len(file.profiles) >= MAX_PROFILES:
            raise PolicyError("too_many_profiles", f"At most {MAX_PROFILES} profiles are allowed.")
        paused = replace(current, paused=True, paused_at=stamp)
        result[:] = [paused]
        gaps = {**file.gaps, profile_id: [*file.gaps.get(profile_id, []), [stamp, None]]}
        return replace(file, profiles={**file.profiles, profile_id: paused}, gaps=gaps)

    check_profile_id(profile_id)
    _commit(plan)
    return result[0]


def resume(profile_id: str, now: datetime | None = None) -> ProfilePolicy:
    """Clear the user pause; never re-enables exports and never imports the gap."""
    stamp = _stamp(_now(now))
    result: list[ProfilePolicy] = []

    def plan(file: PolicyFile) -> PolicyFile:
        current = file.profiles.get(profile_id, ProfilePolicy(profile_id=profile_id))
        resumed = replace(current, paused=False, paused_at=None)
        result[:] = [resumed]
        if not current.paused:
            return file
        gaps = [list(g) for g in file.gaps.get(profile_id, [])]
        if gaps and gaps[-1][1] is None:
            gaps[-1][1] = stamp
        return replace(
            file,
            profiles={**file.profiles, profile_id: resumed},
            gaps={**file.gaps, profile_id: gaps},
        )

    check_profile_id(profile_id)
    _commit(plan)
    return result[0]


def require_not_paused(profile_id: str, kind: PauseKind) -> None:
    """Raise ``collection_paused``/``export_paused`` (exit 1) while paused.

    Export is also blocked while an invalid managed update holds sending
    paused. Reads and previews never call this.
    """
    status = policy_status(profile_id)
    command = f"ssgrep metrics resume --profile {profile_id}"
    if status.profile.paused:
        label = "Collection" if kind == "collection" else "Export"
        raise PolicyError(
            f"{kind}_paused",
            f"{label} is paused since {status.profile.paused_at}; nothing was collected or sent.",
            command,
            exit_code=1,
        )
    if kind == "export" and status.sending_paused_reason:
        raise PolicyError(
            "export_paused",
            f"Sending is paused until the policy is corrected ({status.sending_paused_reason}).",
            "ssgrep metrics setup --policy <file>",
            exit_code=1,
        )
