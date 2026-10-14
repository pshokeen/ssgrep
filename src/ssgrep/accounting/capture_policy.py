"""Per-event capture decision from the effective local policy.

The decision is made for *every* event from that event's own working
directory, never from a session header. It costs one ``lstat`` of
``<data>/accounting/policy.json`` when nothing changed: the parsed policy is
cached by the file's identity signature (inode, mtime, size), so a replaced
file (policy writes are atomic renames) is re-read on the next event, and an
absent file stays "disabled" without importing the policy modules at all.

Rules (all fail closed):

* no policy file, an unreadable/unsafe/corrupt one, or an absent profile means
  capture is off -- never a permissive default;
* the owner of a workspace is the profile with the longest approved root (or
  explicit per-workspace assignment) containing it; otherwise the
  personal-local profile when its scope covers the directory. A workspace owned
  by an organization profile never falls back to a personal-local copy;
* the owning profile must have local collection enabled and must not be paused.
"""

from __future__ import annotations

import os
from typing import Any

from ssgrep.accounting import datadir

PERSONAL = "personal-local"
_DISABLED: Any = object()
_cache: tuple[str, tuple[int, int, int] | None, Any] = ("", None, _DISABLED)


def _signature(path: str) -> tuple[int, int, int] | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_ino, info.st_mtime_ns, info.st_size)


def _load() -> Any:
    """Parse the policy file exactly as ``policy.read_policy_file`` does; fail closed.

    ``policy`` itself imports the heavy ``ssgrep.store`` package, so the hot path
    uses the same light primitives (safe reader + codec) it is built from.
    """
    from ssgrep.accounting import plainfile
    from ssgrep.accounting.policy_codec import MAX_POLICY_BYTES, parse
    from ssgrep.accounting.policy_rules import PolicyError

    try:
        data = plainfile.read_private(
            datadir.data_root(), "accounting", "policy.json", max_bytes=MAX_POLICY_BYTES
        )
        return _DISABLED if data is None else parse(data)
    except (plainfile.UnsafePathError, ValueError, OSError, PolicyError):
        return _DISABLED


def policy_file() -> Any:
    """Cached parsed ``PolicyFile`` or the disabled sentinel."""
    global _cache
    path = os.path.join(os.fspath(datadir.data_root()), "accounting", "policy.json")
    signature = _signature(path)
    if _cache[0] == path and _cache[1] == signature:
        return _cache[2]
    parsed = _DISABLED if signature is None else _load()
    _cache = (path, signature, parsed)
    return parsed


def capture_profile(cwd: str | None) -> str | None:
    """The profile whose journal receives an event for ``cwd`` -- or None (no capture).

    ``cwd`` must already be a canonical path (``os.getcwd()`` is; user-supplied
    directories must be ``realpath``-ed by the caller). Unknown scope is never
    captured: there is no event without a workspace.
    """
    if not cwd:
        return None
    file = policy_file()
    if file is _DISABLED:
        return None
    from ssgrep.accounting.policy_rules import workspace_owner

    owner = workspace_owner(file, cwd)
    if owner is None:
        return None
    profile = file.profiles.get(owner)
    if profile is None or not profile.collection_enabled or profile.paused:
        return None
    return owner
