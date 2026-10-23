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

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE, SCHEMA_VERSION, digest_of
from ssgrep.accounting.ledger.ledger_contracts import (
    EVIDENCE_RETENTION_DAYS,
    MAX_DESTINATIONS,
    MAX_PROFILES,
    RECEIPT_RETENTION_DAYS,
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    valid_profile_id,
)
from ssgrep.accounting.privacy import storage as plainfile
from ssgrep.accounting.privacy.storage import data_root as data_dir

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
    except plainfile.UnsafePathError as error:
        raise PolicyError(
            "policy_corrupt",
            f"Accounting policy cannot be read safely ({error}); sending is paused.",
            "ssgrep savings setup",
            exit_code=1,
        ) from error
    except (ValueError, OSError) as error:
        # Only the failure class: the raw text can carry the private data-root path.
        raise PolicyError(
            "policy_corrupt",
            f"Accounting policy cannot be read safely ({type(error).__name__}); sending is paused.",
            "ssgrep savings setup",
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
    from ssgrep.accounting.collect.replay_evidence import maintain_cache

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
    from ssgrep.accounting.collect.replay_evidence import maintain_cache

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
    from ssgrep.accounting.collect.replay_evidence import maintain_cache

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
    command = f"ssgrep savings resume --profile {profile_id}"
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
            "ssgrep savings setup --policy <policy-file>",
            exit_code=1,
        )


MAX_ROOTS = 64
MAX_PATH = 4096
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_CREDENTIAL_SCHEMES = ("env:", "file:")


class PolicyError(Exception):
    """A policy/profile problem with a stable condition and recovery command."""

    def __init__(
        self,
        condition: str,
        message: str,
        command: str = "ssgrep savings status",
        exit_code: int = 2,
    ) -> None:
        super().__init__(message)
        self.condition = condition
        self.command = command
        self.exit_code = exit_code


@dataclass(frozen=True)
class PolicyFile:
    """Everything persisted in ``policy.json`` (all profiles, one file)."""

    profiles: dict[str, ProfilePolicy] = field(default_factory=dict)
    ceilings: dict[str, ProfilePolicy] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    sending_paused: dict[str, str] = field(default_factory=dict)
    gaps: dict[str, list[list[str | None]]] = field(default_factory=dict)


def check_profile_id(profile_id: str) -> str:
    if not isinstance(profile_id, str) or not valid_profile_id(profile_id):
        raise PolicyError(
            "invalid_profile",
            "Profile ids are 1-32 characters of a-z, 0-9, '-' or '_' and never paths.",
        )
    return profile_id


def canonical_root(raw: object) -> str:
    """Absolute symlink-resolved root; ``/`` and relative paths are refused."""
    if not isinstance(raw, str) or not raw or "\0" in raw or len(raw) > MAX_PATH:
        raise PolicyError("invalid_policy", "Each root must be a non-empty path string.")
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        raise PolicyError("invalid_policy", f"Root {raw!r} must be an absolute path.")
    real = os.path.realpath(expanded)
    if real == os.sep:
        raise PolicyError("invalid_policy", "The filesystem root cannot be an approved root.")
    return real


def contains(root: str, path: str) -> bool:
    """Component-wise containment of canonical ``path`` within canonical ``root``."""
    return path == root or path.startswith(root + os.sep)


def path_in_roots(roots: tuple[str, ...], path: str) -> bool:
    real = os.path.realpath(path)
    return any(contains(root, real) for root in roots)


def workspace_owner(file: PolicyFile, path: str) -> str | None:
    """Resolve an actual live cwd once against the persisted ownership bindings."""
    return historical_workspace_owner(file, os.path.realpath(path))


def historical_workspace_owner(file: PolicyFile, path: str) -> str | None:
    """Compare recorded canonical paths without following today's filesystem.

    Roots and assignments were canonicalized when approved. An unrecorded alias
    cannot acquire that approval merely because it points into a root today.
    """
    if not os.path.isabs(path) or os.path.normpath(path) != path or "\0" in path:
        return None
    assigned = [
        (len(root), owner) for root, owner in file.assignments.items() if contains(root, path)
    ]
    if assigned:
        owner = max(assigned)[1]
        profile = file.profiles.get(owner)
        if profile is None or not any(contains(root, path) for root in profile.roots):
            return None
        return owner
    matches = [
        (len(root), profile_id)
        for profile_id, profile in file.profiles.items()
        if profile_id != PERSONAL_PROFILE
        for root in profile.roots
        if contains(root, path)
    ]
    if matches:
        longest = max(size for size, _ in matches)
        owners = {owner for size, owner in matches if size == longest}
        return next(iter(owners)) if len(owners) == 1 else None
    personal = file.profiles.get(PERSONAL_PROFILE)
    if personal is not None and (
        not personal.roots or any(contains(root, path) for root in personal.roots)
    ):
        return PERSONAL_PROFILE
    return None


def valid_collection_scope(raw: object) -> bool:
    """Strict local provenance shape; absent/legacy data is never export authority."""
    if not isinstance(raw, dict) or set(raw) != {"schema", "profile", "digest"}:
        return False
    digest = raw.get("digest")
    return (
        type(raw.get("schema")) is int
        and raw["schema"] == 1
        and isinstance(raw.get("profile"), str)
        and valid_profile_id(raw["profile"])
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
    )


def _roots(candidate: ProfilePolicy, others: dict[str, ProfilePolicy], file: PolicyFile) -> tuple:
    if len(candidate.roots) > MAX_ROOTS:
        raise PolicyError("invalid_policy", f"At most {MAX_ROOTS} roots are allowed.")
    roots = tuple(sorted({canonical_root(root) for root in candidate.roots}))
    if candidate.profile_id != PERSONAL_PROFILE and not roots:
        raise PolicyError("invalid_policy", "An organization profile needs approved roots.")
    ceiling = file.ceilings.get(candidate.profile_id)
    if ceiling is not None:
        for root in roots:
            if not any(contains(allowed, root) for allowed in ceiling.roots):
                raise PolicyError(
                    "scope_expansion",
                    f"Root {root} is outside the managed approved roots (symlinks resolved).",
                )
        if not roots:
            roots = ceiling.roots
    if candidate.profile_id == PERSONAL_PROFILE:
        return roots
    for other_id, other in others.items():
        if other_id == candidate.profile_id or other_id == PERSONAL_PROFILE:
            continue
        for mine in roots:
            for theirs in other.roots:
                if not (contains(mine, theirs) or contains(theirs, mine)):
                    continue
                inner = max(mine, theirs, key=len)
                if file.assignments.get(inner) not in (candidate.profile_id, other_id):
                    raise PolicyError(
                        "overlapping_roots",
                        f"{mine} overlaps profile {other_id} ({theirs}); "
                        "assign the workspace to one profile explicitly.",
                    )
    return roots


def check_endpoint(endpoint: object) -> str:
    if not isinstance(endpoint, str) or not endpoint or len(endpoint) > 2048:
        raise PolicyError("invalid_policy", "Destination endpoint must be a URL string.")
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise PolicyError("invalid_policy", "Endpoints must not embed credentials or queries.")
    if parts.scheme == "https" and host:
        return endpoint
    if parts.scheme == "http" and host in _LOOPBACK:
        return endpoint
    raise PolicyError("invalid_policy", "Endpoints must be https (http only for loopback).")


def check_proxy(proxy: object, *, endpoint: str) -> str:
    """An explicit HTTP CONNECT proxy for an HTTPS endpoint, without credentials or routing."""
    if not isinstance(proxy, str) or not proxy or len(proxy) > 2048:
        raise PolicyError("invalid_policy", "Destination proxy must be a URL string.")
    parts = urlsplit(proxy)
    if parts.scheme != "http" or not parts.hostname:
        raise PolicyError("invalid_policy", "Proxies must be explicit http URLs naming a host.")
    if parts.username is not None or parts.password is not None:
        raise PolicyError("invalid_policy", "Proxy URLs must not embed credentials.")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise PolicyError("invalid_policy", "Proxy URLs must be host[:port] only.")
    if urlsplit(endpoint).scheme != "https":
        raise PolicyError("invalid_policy", "Proxies require an https destination endpoint.")
    return proxy


def _destinations(candidate: ProfilePolicy, ceiling: ProfilePolicy | None) -> tuple:
    if len(candidate.destinations) > MAX_DESTINATIONS:
        raise PolicyError("invalid_policy", f"At most {MAX_DESTINATIONS} destinations.")
    seen: set[str] = set()
    for dest in candidate.destinations:
        if not valid_profile_id(dest.destination_id) or dest.destination_id in seen:
            raise PolicyError("invalid_policy", "Destination ids must be unique bounded ids.")
        seen.add(dest.destination_id)
        check_endpoint(dest.endpoint)
        if not dest.signals or any(not isinstance(s, Signal) for s in dest.signals):
            raise PolicyError("invalid_policy", "Destinations need known signals.")
        if Signal.LOGS in dest.signals and not dest.logs_permitted:
            raise PolicyError("invalid_policy", "Logs need explicit logs_permitted policy.")
        if dest.proxy is not None:
            check_proxy(dest.proxy, endpoint=dest.endpoint)
        ref = dest.credential_ref
        if ref is not None and (
            len(ref) > 256 or not ref.startswith(_CREDENTIAL_SCHEMES) or ref.isspace()
        ):
            raise PolicyError("invalid_policy", "credential_ref must be env:NAME or file:PATH.")
        if ceiling is not None:
            _within_destination_ceiling(dest, ceiling)
    return candidate.destinations


def _within_destination_ceiling(dest: DestinationPolicy, ceiling: ProfilePolicy) -> None:
    allowed = {d.destination_id: d for d in ceiling.destinations}.get(dest.destination_id)
    if (
        allowed is None
        or allowed.endpoint != dest.endpoint
        or not set(dest.signals) <= set(allowed.signals)
        or (dest.logs_permitted and not allowed.logs_permitted)
        or dest.proxy != allowed.proxy
    ):
        raise PolicyError(
            "scope_expansion",
            f"Destination {dest.destination_id} exceeds the managed destination allowlist.",
        )


def _retention(candidate: ProfilePolicy, ceiling: ProfilePolicy | None) -> tuple[int, int]:
    evidence_max = ceiling.evidence_retention_days if ceiling else EVIDENCE_RETENTION_DAYS
    receipt_max = ceiling.receipt_retention_days if ceiling else RECEIPT_RETENTION_DAYS
    for name, value, maximum in (
        ("evidence_retention_days", candidate.evidence_retention_days, evidence_max),
        ("receipt_retention_days", candidate.receipt_retention_days, receipt_max),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PolicyError("invalid_policy", f"{name} must be a positive integer.")
        if value > maximum:
            raise PolicyError(
                "retention_widened",
                f"{name}={value} exceeds the approved maximum of {maximum} days.",
            )
    return candidate.evidence_retention_days, candidate.receipt_retention_days


def _cache(candidate: ProfilePolicy, ceiling: ProfilePolicy | None) -> None:
    enabled = candidate.evidence_cache_enabled
    days, size = candidate.evidence_cache_retention_days, candidate.evidence_cache_max_bytes
    if not isinstance(enabled, bool) or any(
        isinstance(v, bool) or not isinstance(v, int) for v in (days, size)
    ):
        raise PolicyError(
            "invalid_policy", "Evidence cache requires explicit boolean consent and integer bounds."
        )
    if not enabled:
        if days != 0 or size != 0:
            raise PolicyError(
                "invalid_policy", "Disabled evidence cache must have zero retention and byte cap."
            )
        return
    if (
        not 1 <= days <= min(30, candidate.evidence_retention_days)
        or not 1 <= size <= 32 * 1024 * 1024
    ):
        raise PolicyError(
            "invalid_policy", "Evidence cache requires 1–30 days and a byte cap of at most 32 MiB."
        )
    if ceiling is not None and (
        not ceiling.evidence_cache_enabled
        or days > ceiling.evidence_cache_retention_days
        or size > ceiling.evidence_cache_max_bytes
    ):
        raise PolicyError("retention_widened", "Evidence cache exceeds managed consent or bounds.")


def _exports(candidate: ProfilePolicy) -> tuple[tuple[str, str], ...]:
    by_id = {d.destination_id: d for d in candidate.destinations}
    for dest_id, signal in candidate.enabled_exports:
        dest = by_id.get(dest_id)
        if dest is None or signal not in {s.value for s in dest.signals}:
            raise PolicyError("invalid_policy", f"Export {dest_id}/{signal} is not permitted.")
    return tuple(sorted(set(candidate.enabled_exports)))


def validate_profile(candidate: ProfilePolicy, file: PolicyFile) -> ProfilePolicy:
    """Canonical, validated copy of ``candidate`` or ``PolicyError`` (no side effects)."""
    check_profile_id(candidate.profile_id)
    if candidate.profile_id not in file.profiles and len(file.profiles) >= MAX_PROFILES:
        raise PolicyError("too_many_profiles", f"At most {MAX_PROFILES} profiles are allowed.")
    ceiling = file.ceilings.get(candidate.profile_id)
    roots = _roots(candidate, file.profiles, file)
    destinations = _destinations(candidate, ceiling)
    evidence_days, receipt_days = _retention(candidate, ceiling)
    _cache(candidate, ceiling)
    return replace(
        candidate,
        roots=roots,
        destinations=destinations,
        evidence_retention_days=evidence_days,
        receipt_retention_days=receipt_days,
        enabled_exports=_exports(candidate),
    )


def clamp_to_ceiling(current: ProfilePolicy, ceiling: ProfilePolicy) -> ProfilePolicy:
    """Narrow ``current`` so it never exceeds a (new) managed ``ceiling``."""
    roots = tuple(r for r in current.roots if any(contains(c, r) for c in ceiling.roots))
    allowed = {d.destination_id: d for d in ceiling.destinations}
    kept = []
    for dest in current.destinations:
        permitted = allowed.get(dest.destination_id)
        if permitted is None or permitted.endpoint != dest.endpoint:
            continue
        signals = tuple(s for s in dest.signals if s in permitted.signals)
        if signals:
            logs = dest.logs_permitted and permitted.logs_permitted
            kept.append(replace(dest, signals=signals, logs_permitted=logs, proxy=permitted.proxy))
    exports = tuple(
        (d, s)
        for d, s in current.enabled_exports
        if any(k.destination_id == d and s in {x.value for x in k.signals} for k in kept)
    )
    return replace(
        current,
        roots=roots or ceiling.roots,
        destinations=tuple(kept),
        enabled_exports=exports,
        evidence_retention_days=min(
            current.evidence_retention_days, ceiling.evidence_retention_days
        ),
        receipt_retention_days=min(current.receipt_retention_days, ceiling.receipt_retention_days),
        evidence_cache_enabled=current.evidence_cache_enabled and ceiling.evidence_cache_enabled,
        evidence_cache_retention_days=min(
            current.evidence_cache_retention_days, ceiling.evidence_cache_retention_days
        )
        if ceiling.evidence_cache_enabled
        else 0,
        evidence_cache_max_bytes=min(
            current.evidence_cache_max_bytes, ceiling.evidence_cache_max_bytes
        )
        if ceiling.evidence_cache_enabled
        else 0,
        managed=True,
        policy_version=ceiling.policy_version,
        policy_source=ceiling.policy_source,
    )


MAX_POLICY_BYTES = 1024 * 1024
MAX_GAPS = 32


def _corrupt(detail: str) -> PolicyError:
    return PolicyError(
        "policy_corrupt",
        f"Accounting policy file is unreadable ({detail}); sending is paused.",
        "ssgrep savings setup",
        exit_code=1,
    )


def _str(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise _corrupt(f"{name} must be a string")
    return value


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _corrupt(f"{name} must be an integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _corrupt(f"{name} must be a boolean")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise _corrupt(f"{name} must be a list")
    return value


def destination_from_dict(raw: Any) -> DestinationPolicy:
    if not isinstance(raw, dict):
        raise _corrupt("destination must be an object")
    try:
        signals = tuple(Signal(s) for s in _list(raw.get("signals", ["metrics"]), "signals"))
    except ValueError as error:
        raise _corrupt("unknown signal") from error
    return DestinationPolicy(
        destination_id=str(_str(raw.get("destination_id"), "destination_id")),
        endpoint=str(_str(raw.get("endpoint"), "endpoint")),
        signals=signals,
        logs_permitted=_bool(raw.get("logs_permitted", False), "logs_permitted"),
        credential_ref=_str(raw.get("credential_ref"), "credential_ref", nullable=True),
        proxy=_str(raw.get("proxy"), "proxy", nullable=True),
    )


def profile_from_dict(raw: Any) -> ProfilePolicy:
    if not isinstance(raw, dict):
        raise _corrupt("profile must be an object")
    exports = []
    for pair in _list(raw.get("enabled_exports", []), "enabled_exports"):
        if not (
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
        ):
            raise _corrupt("enabled_exports entries must be [destination, signal]")
        exports.append((pair[0], pair[1]))
    return ProfilePolicy(
        profile_id=str(_str(raw.get("profile_id"), "profile_id")),
        roots=tuple(str(_str(r, "root")) for r in _list(raw.get("roots", []), "roots")),
        managed=_bool(raw.get("managed", False), "managed"),
        policy_version=_int(raw.get("policy_version", 1), "policy_version"),
        policy_source=str(_str(raw.get("policy_source", "local"), "policy_source")),
        evidence_retention_days=_int(raw.get("evidence_retention_days"), "evidence_retention_days"),
        receipt_retention_days=_int(raw.get("receipt_retention_days"), "receipt_retention_days"),
        destinations=tuple(
            destination_from_dict(d) for d in _list(raw.get("destinations", []), "destinations")
        ),
        collection_enabled=_bool(raw.get("collection_enabled", False), "collection_enabled"),
        paused=_bool(raw.get("paused", False), "paused"),
        paused_at=_str(raw.get("paused_at"), "paused_at", nullable=True),
        enabled_exports=tuple(exports),
        evidence_cache_enabled=_bool(
            raw.get("evidence_cache_enabled", False), "evidence_cache_enabled"
        ),
        evidence_cache_retention_days=_int(
            raw.get("evidence_cache_retention_days", 0), "evidence_cache_retention_days"
        ),
        evidence_cache_max_bytes=_int(
            raw.get("evidence_cache_max_bytes", 0), "evidence_cache_max_bytes"
        ),
    )


def profile_to_dict(profile: ProfilePolicy) -> dict[str, Any]:
    plain = asdict(profile)
    plain["roots"] = list(profile.roots)
    plain["enabled_exports"] = [list(pair) for pair in profile.enabled_exports]
    plain["destinations"] = [
        {**asdict(d), "signals": [s.value for s in d.signals]} for d in profile.destinations
    ]
    return plain


def _profiles(raw: Any, name: str) -> dict[str, ProfilePolicy]:
    if not isinstance(raw, dict):
        raise _corrupt(f"{name} must be an object")
    profiles = {str(k): profile_from_dict(v) for k, v in raw.items()}
    if any(not valid_profile_id(k) or k != v.profile_id for k, v in profiles.items()):
        raise _corrupt(f"{name} keys must match their profile identities")
    return profiles


def _gaps(raw: dict[str, Any]) -> dict[str, list[list[str | None]]]:
    gaps: dict[str, list[list[str | None]]] = {}
    for profile, rows in raw.items():
        if not isinstance(rows, list) or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or (row[1] is not None and not isinstance(row[1], str))
            for row in rows
        ):
            raise _corrupt("gaps must contain [start, end-or-null] pairs")
        gaps[profile] = rows[-MAX_GAPS:]
    return gaps


def parse(data: bytes) -> PolicyFile:
    try:
        raw = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise _corrupt("not valid JSON") from error
    if not isinstance(raw, dict):
        raise _corrupt("top level must be an object")
    if type(raw.get("schema")) is not int or raw["schema"] != SCHEMA_VERSION:
        raise _corrupt("unsupported schema")
    assignments = raw.get("assignments", {})
    paused = raw.get("sending_paused", {})
    gaps = raw.get("gaps", {})
    if not (isinstance(assignments, dict) and isinstance(paused, dict) and isinstance(gaps, dict)):
        raise _corrupt("assignments, sending_paused and gaps must be objects")
    return PolicyFile(
        profiles=_profiles(raw.get("profiles", {}), "profiles"),
        ceilings=_profiles(raw.get("ceilings", {}), "ceilings"),
        assignments={str(k): str(v) for k, v in assignments.items()},
        sending_paused={str(k): str(v) for k, v in paused.items()},
        gaps=_gaps(gaps),
    )


def dump(file: PolicyFile) -> bytes:
    document = {
        "schema": SCHEMA_VERSION,
        "profiles": {k: profile_to_dict(v) for k, v in sorted(file.profiles.items())},
        "ceilings": {k: profile_to_dict(v) for k, v in sorted(file.ceilings.items())},
        "assignments": dict(sorted(file.assignments.items())),
        "sending_paused": dict(sorted(file.sending_paused.items())),
        "gaps": {k: v[-MAX_GAPS:] for k, v in sorted(file.gaps.items())},
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
