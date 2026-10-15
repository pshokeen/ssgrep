"""Pure policy validation: scope, retention, destinations, managed ceilings.

Nothing here touches the filesystem except ``os.path.realpath`` (to resolve
symlinks): validation must be able to reject a candidate before any side
effect. Containment is decided on canonical paths by path component, never by
lexical prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.ledger_contracts import (
    EVIDENCE_RETENTION_DAYS,
    MAX_DESTINATIONS,
    MAX_PROFILES,
    RECEIPT_RETENTION_DAYS,
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    valid_profile_id,
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
        command: str = "ssgrep metrics status",
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
