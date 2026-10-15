"""Immutable per-revision rarity evidence with prospective read-only index refresh.

The pointer selects the latest supported generation, not an eternal retained-200
document freeze. Old content-addressed views are never rewritten. The collector
must apply a prospective transition only to new lookup revisions; explicit
recomputation replaces named revisions rather than summing generations.

No automatic retained-plaintext frequency fallback is permitted. Index absence,
damage and unapproved corpus scope produce typed unknown for new work, even if
an older frozen view exists for historical replay.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ssgrep.accounting import storage
from ssgrep.accounting.collector_store import dumps, loads
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.frequency import BASIS_CANDIDATES, BASIS_INDEX, FrequencyView
from ssgrep.accounting.plain_codec import from_plain, to_plain

POINTER = "frequency.json"
VIEW_PREFIX = "frequency-"
MAX_POINTER_BYTES = 4096
MAX_VIEW_BYTES = 256 * 1024 * 1024
_VIEW_ID = re.compile(r"freq-[0-9a-f]{24}")
_LOADED: dict[tuple[str, str, str], tuple[tuple[int, ...], FrequencyView]] = {}

Hook = Callable[[str], None]


def view_file_name(view_id: str) -> str:
    return f"{VIEW_PREFIX}{view_id}.json"


def view_identity(view: FrequencyView) -> str:
    """Identity includes generation, keyed metadata and authorized corpus provenance."""
    return view.identity()


def _corrupt(what: str) -> CollectError:
    return CollectError("state_corrupt", f"The pinned frequency view is unusable ({what}).")


def _verified(view_id: str, data: bytes) -> FrequencyView:
    try:
        view = from_plain(FrequencyView, loads(data))
    except (ValueError, TypeError) as error:
        raise _corrupt("view file is damaged") from error
    if not view.known or view.version_id != view_id:
        raise _corrupt("view file does not match its name")
    if view_identity(view) != view_id:
        raise _corrupt("view content does not match its digest")
    return view


def load_pinned(profile: str) -> FrequencyView | None:
    """Latest immutable supported view, retained independently from live index availability.

    A damaged accounting pin fails closed to preserve historical revision identity.
    This differs from damaged external index statistics, which are typed unknown.
    """
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return None
    try:
        raw = storage.read_file(directory, POINTER, MAX_POINTER_BYTES)
        if raw is None:
            return None
        try:
            pointer = loads(raw)
            view_id = pointer["view_id"]
        except (ValueError, KeyError, TypeError) as error:
            raise _corrupt("pointer is damaged") from error
        if not isinstance(view_id, str) or _VIEW_ID.fullmatch(view_id) is None:
            raise _corrupt("pointer names no valid view")
    except ValueError as error:
        raise _corrupt("a pinned file is too large") from error
    finally:
        os.close(directory)
    return load_view(profile, view_id)


def load_view(profile: str, view_id: str) -> FrequencyView:
    """Read an exact frozen revision view, never substitute the current generation."""
    if _VIEW_ID.fullmatch(view_id) is None:
        raise _corrupt("revision names no valid view")
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError as error:
        raise _corrupt("revision view is missing") from error
    try:
        name = view_file_name(view_id)
        descriptor = storage.open_regular(directory, name)
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if info.st_size > MAX_VIEW_BYTES:
            raise ValueError("view too large")
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        cache_key = (str(storage.data_root()), profile, view_id)
        cached = _LOADED.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        data = storage.read_file(directory, name, MAX_VIEW_BYTES)
    except FileNotFoundError as error:
        raise _corrupt("revision view is missing") from error
    except ValueError as error:
        raise _corrupt("revision view is too large") from error
    finally:
        os.close(directory)
    if data is None:
        raise _corrupt("revision view is missing")
    view = _verified(view_id, data)
    if view.basis == BASIS_INDEX and view.profile_scope != profile:
        raise _corrupt("revision view belongs to another profile")
    if len(_LOADED) >= 2:
        _LOADED.pop(next(iter(_LOADED)))
    _LOADED[cache_key] = (signature, view)
    return view


def pin(profile: str, view: FrequencyView, on_step: Hook | None = None) -> None:
    """Write the immutable view file, then atomically swap the pointer to it."""
    if not view.known or view.basis not in (BASIS_CANDIDATES, BASIS_INDEX):
        raise ValueError("only a supported frequency view with documents can be pinned")
    if view.basis == BASIS_INDEX and view.profile_scope != profile:
        raise ValueError("frequency view belongs to another profile")
    if view_identity(view) != view.version_id:
        raise ValueError("view id does not match its content")
    body = dumps(to_plain(view)) + b"\n"
    name = view_file_name(view.version_id)
    directory = storage.open_profile_dir(profile, "collector")
    try:
        if not storage.write_once(directory, name, body):
            existing = storage.read_file(directory, name, MAX_VIEW_BYTES)
            if existing != body:  # content addressed: a differing file is damage, not a repin
                raise _corrupt("an existing view file differs from its digest")
        if on_step is not None:
            on_step("frequency_view_written")
        pointer = dumps({"view_id": view.version_id}) + b"\n"
        storage.atomic_write(directory, POINTER, pointer)
        if on_step is not None:
            on_step("frequency_pinned")
    finally:
        os.close(directory)


@dataclass(frozen=True)
class FrequencyOutcome:
    """The view this cycle reduces with, and what to say about it.

    ``coverage`` and ``standing_warnings`` are facts of the snapshot (they hold on every cycle
    in this state); ``events`` are things that happened in THIS cycle (pinned, repinned,
    repin refused) and are reported to the caller only.
    """

    view: FrequencyView
    state: str  # pinned | unavailable
    documents: int
    events: tuple[str, ...] = ()
    transition: str = "unchanged"  # initial | unchanged | prospective | replacement | unavailable
    previous_view_id: str | None = None
    revision_name: str | None = None

    def coverage(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "state": self.state,
            "documents": self.documents,
            "transition": self.transition,
            "uncalibrated": not self.view.calibrated,
        }
        if self.view.known:
            out.update(
                view_id=self.view.version_id,
                generation=self.view.generation,
                corpus_digest=self.view.corpus_digest,
                normalization=self.view.normalization,
                document_unit=self.view.document_unit,
                corpus_coverage=self.view.coverage,
                profile_scope=self.view.profile_scope,
                authority_id=self.view.authority_id,
                mapping=self.view.mapping,
            )
        else:
            out["reason"] = self.view.unavailable_reason or "frequency_unavailable"
        if self.previous_view_id is not None:
            out["previous_view_id"] = self.previous_view_id
        if self.revision_name is not None:
            out["revision_name"] = self.revision_name
        return out

    def standing_warnings(self) -> tuple[str, ...]:
        if self.view.known:
            return ()
        return (
            f"{self.view.unavailable_reason or 'frequency_unavailable'}: "
            "source credit stays unknown, not zero",
        )


def resolve(
    profile: str,
    *,
    pinned: FrequencyView | None,
    last_documents: int | None,
    evidence: Callable[[], Iterable[RawEvidence]],
    scan: bool,
    repin: bool,
    on_step: Hook | None = None,
    index_view: Callable[[], FrequencyView] | None = None,
    cached: FrequencyView | None = None,
    reset_authority: bool = False,
    scope_epoch: int = 1,
    authority: Callable[[str | None], bool] | None = None,
    authority_id: str | None = None,
) -> FrequencyOutcome:
    """Refresh compatible current statistics without reinterpreting historical rows.

    ``evidence``, ``scan`` and ``last_documents`` remain accepted by the existing
    collector caller but never authorize a retained-transcript plaintext rescan.
    The index is checked even without new evidence, so later supported candidates
    do not require manual repinning. ``repin`` requests a named replacement revision.
    """
    from ssgrep.accounting.frequency_authority import erased_view

    erased = erased_view(
        profile,
        scope_epoch=scope_epoch,
        reset_authority=reset_authority,
        repin=repin,
        cached=cached or pinned,
        authority=authority,
        authority_id=authority_id,
    )
    if erased is not None:
        view = erased
    elif index_view is None:
        from ssgrep.accounting.index_frequency import read_index_view

        view = read_index_view(
            profile, cached=cached or pinned, authority=authority, authority_id=authority_id
        )
    else:
        view = index_view()
    previous = pinned.version_id if pinned is not None else None
    if not view.known:
        return FrequencyOutcome(
            view, "unavailable", view.doc_count, transition="unavailable", previous_view_id=previous
        )
    if view.profile_scope and view.profile_scope != profile:
        unknown = FrequencyView.unknown("frequency_scope_unapproved", profile=profile)
        return FrequencyOutcome(
            unknown, "unavailable", 0, transition="unavailable", previous_view_id=previous
        )
    if previous == view.version_id:
        return FrequencyOutcome(view, "pinned", view.doc_count)
    pin(profile, view, on_step)
    transition = "replacement" if repin and previous else "prospective" if previous else "initial"
    revision = f"frequency-recompute:{view.version_id}" if transition == "replacement" else None
    note = (
        f"frequency_{transition}: view {view.version_id} from {view.doc_count} indexed chunks; "
        "prior canonical revisions remain immutable and are not summed as new savings"
    )
    return FrequencyOutcome(view, "pinned", view.doc_count, (note,), transition, previous, revision)
