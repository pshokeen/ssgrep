"""Per-event capture decision from the effective local policy.

Known scope is decided for every event from trusted invocation context, never
from MCP server cwd or client-supplied arguments. Unknown caller scope is retained
locally without fabricated workspace authority. Every cache lookup pins and validates
the containing private directory and policy file with no-follow descriptor opens.
The parsed policy is cached by the leaf's identity and authority signature, so
replacement, permissions or ownership changes are rechecked on the next event.
An absent file stays disabled without importing policy codecs or journal storage.

Rules (all fail closed):

* no policy file, an unreadable/unsafe/corrupt one, or an absent profile means
  capture is off -- never a permissive default;
* the owner of a workspace is the profile with the longest approved root (or
  explicit per-workspace assignment) containing it; otherwise the
  personal-local profile when its scope covers the directory. A workspace owned
  by an organization profile never falls back to a personal-local copy;
* the owning profile must have local collection enabled and must not be paused.
* unknown scope can select one explicitly configured journal or personal-local;
  that selection is not organization authorization and is never broadcast.
"""

from __future__ import annotations

import os
from typing import Any

from ssgrep.accounting import datadir

PERSONAL = "personal-local"
_DISABLED: Any = object()
_cache: tuple[str, tuple[int, ...] | None, Any] = ("", None, _DISABLED)


def _signature(descriptor: int) -> tuple[int, ...]:
    info = os.fstat(descriptor)
    return (
        info.st_dev,
        info.st_ino,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_size,
        info.st_mode,
        info.st_uid,
    )


def _load(descriptor: int) -> Any:
    """Parse the pinned policy using the same lightweight reader and codec as policy."""
    from ssgrep.accounting import plainfile
    from ssgrep.accounting.policy_codec import MAX_POLICY_BYTES, parse
    from ssgrep.accounting.policy_rules import PolicyError

    try:
        return parse(plainfile.read_descriptor(descriptor, max_bytes=MAX_POLICY_BYTES))
    except (plainfile.UnsafePathError, ValueError, OSError, PolicyError):
        return _DISABLED


def policy_file(*, root: datadir.SafeRootSnapshot | None = None) -> Any:
    """Cached parsed policy under one root inode; never create an absent namespace."""
    from ssgrep.accounting import plainfile

    global _cache
    root = datadir.snapshot() if root is None else root
    if root is None:
        _cache = ("", None, _DISABLED)
        return _DISABLED
    path = os.path.join(os.fspath(root.path), "accounting", "policy.json")
    try:
        root_fd = datadir.open_root(root)
        try:
            with plainfile.open_private(root_fd, "accounting", "policy.json") as descriptor:
                if descriptor is not None:
                    signature = _signature(descriptor)
                    if _cache[0] == path and _cache[1] == signature:
                        return _cache[2]
                    parsed = _load(descriptor)
                    if _signature(descriptor) == signature:
                        _cache = (path, signature, parsed)
                    else:
                        _cache = (path, None, _DISABLED)
                    return parsed
        finally:
            os.close(root_fd)
    except (plainfile.UnsafePathError, OSError):
        pass
    _cache = (path, None, _DISABLED)
    return _DISABLED


def capture_profile(
    cwd: str | None,
    *,
    profile: str | None = None,
    root: datadir.SafeRootSnapshot | None = None,
) -> str | None:
    """Select exactly one local journal; selection is not workspace authorization.

    Known cwd must have its policy owner. Unknown caller scope may be retained
    in an explicitly selected enabled profile, or personal-local; it supplies no
    organization scope and must stay unexported until native evidence resolves it.
    Never broadcast one execution among profiles or bypass a known owner.
    """
    if cwd == "":
        return None
    file = policy_file(root=root)
    if file is _DISABLED:
        return None
    from ssgrep.accounting.policy_rules import workspace_owner

    owner = workspace_owner(file, cwd) if cwd else (profile if profile is not None else PERSONAL)
    if owner is None or (cwd and profile is not None and profile != owner):
        return None
    chosen = file.profiles.get(owner)
    if chosen is None or not chosen.collection_enabled or chosen.paused:
        return None
    return owner
