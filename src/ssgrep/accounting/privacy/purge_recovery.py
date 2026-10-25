"""Durable exact purge selection, recoverable before or after the checkpoint commit."""

from __future__ import annotations

import os
from typing import Any

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.frequency import MARKER, MAX_MARKER_BYTES, erasure_document
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.purge import PurgeError, PurgeReport, Selection


def _read(profile: str) -> dict[str, Any] | None:
    try:
        fd = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return None
    try:
        raw = storage.read_file(fd, MARKER, MAX_MARKER_BYTES)
        if raw is None:
            return None
        document = store.loads(raw)
        if not isinstance(document, dict) or document.get("schema") != 1:
            raise ValueError("invalid purge intent")
        return document
    except (ValueError, TypeError) as error:
        raise PurgeError(
            "state_corrupt", "Privacy erasure intent is invalid; recover it before collection."
        ) from error
    finally:
        os.close(fd)


def stage_erasure(sel: Selection) -> None:
    """Lease holder: deny old rarity authority and record intent BEFORE any deletion."""
    fd = storage.open_profile_dir(sel.profile, "collector")
    try:
        document = erasure_document(fd, sel)
        document["pending_purge"] = {
            "profile": sel.profile,
            "before": sel.before,
            "workspace": sel.workspace,
        }
        storage.atomic_write(fd, MARKER, store.dumps(document) + b"\n")
    finally:
        os.close(fd)


def complete_erasure(profile: str) -> None:
    """Clear only the pending operation; irreversible scope exclusions remain."""
    document = _read(profile)
    if document is None or not document.get("pending_purge"):
        return
    document.pop("pending_purge")
    fd = storage.open_profile_dir(profile, "collector", create=False)
    try:
        storage.atomic_write(fd, MARKER, store.dumps(document) + b"\n")
    finally:
        os.close(fd)


def _pending_selection(profile: str) -> Selection | None:
    """Validate durable intent once for planning and complete recovery."""
    document = _read(profile)
    pending = document.get("pending_purge") if document is not None else None
    if pending is None:
        return None
    from ssgrep.accounting.views.types import parse_instant

    if not isinstance(pending, dict) or pending.get("profile") != profile:
        raise PurgeError("state_corrupt", "Privacy erasure intent names an invalid profile.")
    workspace = pending.get("workspace")
    if workspace is not None and (
        not isinstance(workspace, str)
        or not os.path.isabs(workspace)
        or workspace == "/"
        or "\0" in workspace
    ):
        raise PurgeError("state_corrupt", "Privacy erasure intent names an invalid workspace.")
    from ssgrep.accounting.privacy.policy import PolicyError

    try:
        if not isinstance(pending.get("before"), str):
            raise ValueError("missing purge cutoff")
        before = parse_instant(pending["before"], "purge intent")
    except (ValueError, PolicyError) as error:
        raise PurgeError(
            "state_corrupt", "Privacy erasure intent has an invalid cutoff."
        ) from error
    return Selection(profile, before, workspace)


def has_pending_erasure(sel: Selection) -> bool:
    return _pending_selection(sel.profile) == sel


def recover_pending_purge(profile: str) -> PurgeReport | None:
    """Lease holder: repeat the complete purge, including checkpoint filtering."""
    pending = _pending_selection(profile)
    if pending is None:
        return None
    from ssgrep.accounting.privacy.purge import execute

    return execute(pending, _lease_held=True)
