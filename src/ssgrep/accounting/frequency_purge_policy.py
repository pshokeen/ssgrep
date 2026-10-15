"""Merge irreversible scope exclusions before erasing shared frequency authority."""

from __future__ import annotations

from typing import Any

from ssgrep.accounting import collector_store as store, storage
from ssgrep.accounting.purge_select import Selection

MARKER = "frequency-purged.json"
MAX_MARKER_BYTES = 1024 * 1024


def erasure_document(fd: int, sel: Selection) -> dict[str, Any]:
    from ssgrep.accounting.collector_frequency import _VIEW_ID, MAX_VIEW_BYTES, view_file_name

    prior: dict[str, Any] = {}
    try:
        raw = storage.read_file(fd, MARKER, MAX_MARKER_BYTES)
        if raw is not None:
            parsed = store.loads(raw)
            if isinstance(parsed, dict) and parsed.get("schema") == 1:
                prior = parsed
            else:
                prior = {"whole_profile": True, "erased_scope_unknown": True}
    except (ValueError, KeyError, TypeError):
        prior = {"whole_profile": True, "erased_scope_unknown": True}
    roots = prior.get("erased_workspaces", [])
    unknown = bool(prior.get("erased_scope_unknown"))
    if not isinstance(roots, list) or not all(
        isinstance(root, str) and root.startswith("/") for root in roots
    ):
        roots, unknown = [], True
    if prior.get("scoped") and not roots:
        unknown = True
    if sel.workspace is not None:
        roots = [*roots, sel.workspace]
    generation = prior.get("authorized_generation") or prior.get("erased_generation")
    try:
        pointer = storage.read_file(fd, "frequency.json", 4096)
        if pointer is not None:
            view_id = store.loads(pointer).get("view_id")
            if isinstance(view_id, str) and _VIEW_ID.fullmatch(view_id) is not None:
                data = storage.read_file(fd, view_file_name(view_id), MAX_VIEW_BYTES)
                view = store.loads(data) if data is not None else {}
                if isinstance(view.get("generation"), str) and view["generation"]:
                    generation = view["generation"]
    except (ValueError, KeyError, TypeError, AttributeError):
        generation = None
    return {
        "schema": 1,
        "reason": "frequency_scope_erased",
        "before": sel.cutoff_day,
        "scoped": sel.workspace is not None,
        "whole_profile": bool(prior.get("whole_profile")) or sel.workspace is None,
        "erased_workspaces": sorted(set(roots)),
        "erased_scope_unknown": unknown,
        "erased_generation": generation,
        "authorized_scope_epoch": None,
        "authorized_generation": None,
        "pending_purge": prior.get("pending_purge"),
    }
