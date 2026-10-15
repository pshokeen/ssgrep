"""Scoped rarity erasure remains denied until explicit fresh-corpus authorization."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.collector_store import dumps, loads
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.frequency import FrequencyView

TOMBSTONE = "frequency-purged.json"
MAX_AUTHORITY_BYTES = 512 * 1024


def _marker(profile: str) -> Mapping[str, Any] | None:
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return None
    try:
        raw = storage.read_file(directory, TOMBSTONE, MAX_AUTHORITY_BYTES)
        if raw is None:
            return None
        try:
            document = loads(raw)
        except (ValueError, TypeError):
            return {}
        return document if isinstance(document, dict) else {}
    finally:
        os.close(directory)


def erased_view(
    profile: str,
    *,
    scope_epoch: int,
    reset_authority: bool,
    repin: bool,
    cached: FrequencyView | None = None,
    repository: Any = None,
    authority: Callable[[str | None], bool] | None = None,
    authority_id: str | None = None,
) -> FrequencyView | None:
    """None means no erased scope; otherwise return authorized stats or typed unknown.

    A reset requires explicit consent, repin, a new ready index generation, and
    full authorization outside erased workspaces. The marker and root exclusions
    remain durable after reset, so future generations cannot silently resurrect
    erased corpus authority. Unknown legacy scope needs explicit whole-profile
    recovery, not a guessed workspace. No index rebuild/edit is performed here.
    """

    def unknown(reason: str = "frequency_scope_erased") -> FrequencyView:
        return FrequencyView.unknown(reason, profile=profile)

    try:
        marker = _marker(profile)
        if marker is None:
            return None
        if marker.get("schema") != 1 or marker.get("pending_purge"):
            return unknown()
        roots = marker.get("erased_workspaces", [])
        if (
            not isinstance(roots, list)
            or len(roots) > 64
            or any(not isinstance(root, str) or not os.path.isabs(root) for root in roots)
        ):
            return unknown()
        if (
            marker.get("erased_scope_unknown") or marker.get("scoped") and not roots
        ) and not marker.get("whole_profile"):
            return unknown("frequency_erased_scope_unknown")
        authorized = (
            marker.get("authorized_scope_epoch") == scope_epoch
            and isinstance(marker.get("authorized_generation"), str)
            and bool(marker.get("authorized_generation"))
        )
        if not authorized and not (reset_authority and repin):
            return unknown()
        from ssgrep.accounting.index_frequency import _authority, read_index_view
        from ssgrep.accounting.policy_rules import contains

        if authority is None:
            authority, authority_id = _authority(profile)
        if not authority_id:
            return unknown("frequency_scope_unapproved")
        denied = tuple(os.path.realpath(root) for root in roots)

        def permitted(cwd: str | None) -> bool:
            if denied and cwd is None:
                return False
            return authority(cwd) and not any(
                contains(root, os.path.realpath(cwd or "")) for root in denied
            )

        identity = digest_of([authority_id, list(denied), scope_epoch, "erased_scope_authority_v1"])
        view = read_index_view(
            profile,
            repository=repository,
            authority=permitted,
            authority_id=identity,
            cached=cached,
        )
        if not view.known:
            return view
        if not authorized:
            if not view.generation or view.generation == marker.get("erased_generation"):
                return unknown("frequency_authority_reset_requires_new_generation")
            updated = {
                **marker,
                "authorized_scope_epoch": scope_epoch,
                "authorized_generation": view.generation,
            }
            directory = storage.open_profile_dir(profile, "collector", create=False)
            try:
                storage.atomic_write(directory, TOMBSTONE, dumps(updated) + b"\n")
            finally:
                os.close(directory)
        return view
    except Exception:
        return unknown()
