"""The pinned candidate-frequency view (design D5): pinned once, immutable, repinned explicitly.

Rarity needs a frozen view of how common each candidate token is. The collector pins one per
profile from the RETAINED RAW evidence (``reduce_io.corpus_view``: the extractor's own
normalization) once the corpus is large enough to mean anything, and then never changes it:
a view that drifted with corpus growth would move already-published estimates and break
full-vs-incremental replay equality.

* Below ``FREQUENCY_MIN_DOCS`` documents the view stays *unknown*: no candidate earns credit,
  and the snapshot says so (``frequency_insufficient_corpus`` with the count and the floor)
  instead of reporting zero.
* ``collector/frequency-<view_id>.json`` holds the immutable view (``view_id`` is the digest of
  the sorted df table, the document count and the normalization version);
  ``collector/frequency.json`` is the atomically swapped pointer to it.
* Only ``metrics collect --repin-frequency`` pins a different view. The view id is part of the
  methodology hash, so a repin is a NEW methodology namespace; the old namespace's ledger,
  receipts and history are retained and never summed with the new one.
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
from ssgrep.accounting.contracts import RawEvidence, digest_of
from ssgrep.accounting.frequency import BASIS_CANDIDATES, FrequencyView
from ssgrep.accounting.plain_codec import from_plain, to_plain

FREQUENCY_MIN_DOCS = 200
POINTER = "frequency.json"
VIEW_PREFIX = "frequency-"
MAX_POINTER_BYTES = 4096
MAX_VIEW_BYTES = 256 * 1024 * 1024
_VIEW_ID = re.compile(r"freq-[0-9a-f]{24}")
INSUFFICIENT = "frequency_insufficient_corpus"

Hook = Callable[[str], None]


def view_file_name(view_id: str) -> str:
    return f"{VIEW_PREFIX}{view_id}.json"


def view_identity(view: FrequencyView) -> str:
    """The id a view of this content must carry (same recipe as ``frequency.build_view``)."""
    body = {
        "basis": view.basis,
        "norm": view.normalization,
        "n": view.doc_count,
        "df": dict(sorted(view.df.items())),
    }
    return f"freq-{digest_of(body)[:24]}"


def _corrupt(what: str) -> CollectError:
    return CollectError("state_corrupt", f"The pinned frequency view is unusable ({what}).")


def _verified(view_id: str, data: bytes) -> FrequencyView:
    try:
        view = from_plain(FrequencyView, loads(data))
    except (ValueError, TypeError) as error:
        raise _corrupt("view file is damaged") from error
    if view.basis != BASIS_CANDIDATES or view.version_id != view_id:
        raise _corrupt("view file does not match its name")
    if view_identity(view) != view_id:
        raise _corrupt("view content does not match its digest")
    return view


def load_pinned(profile: str) -> FrequencyView | None:
    """The pinned view; None when none was ever pinned. A damaged pin is an error, never unknown.

    Falling back to *unknown* would silently open a new methodology namespace and could pin a
    different view over the damaged one, which immutability forbids.
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
        data = storage.read_file(directory, view_file_name(view_id), MAX_VIEW_BYTES)
    except ValueError as error:
        raise _corrupt("a pinned file is too large") from error
    finally:
        os.close(directory)
    if data is None:
        raise _corrupt("view file is missing")
    return _verified(view_id, data)


def pin(profile: str, view: FrequencyView, on_step: Hook | None = None) -> None:
    """Write the immutable view file, then atomically swap the pointer to it."""
    if view.basis != BASIS_CANDIDATES or view.doc_count < 1:
        raise ValueError("only a candidate-token view with documents can be pinned")
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


def corpus_view(evidence: Iterable[RawEvidence]) -> FrequencyView:
    """The candidate-frequency view of retained raw evidence (the reducer stack loads lazily)."""
    from ssgrep.accounting import reduce_io

    return reduce_io.corpus_view(evidence)


@dataclass(frozen=True)
class FrequencyOutcome:
    """The view this cycle reduces with, and what to say about it.

    ``coverage`` and ``standing_warnings`` are facts of the snapshot (they hold on every cycle
    in this state); ``events`` are things that happened in THIS cycle (pinned, repinned,
    repin refused) and are reported to the caller only.
    """

    view: FrequencyView
    state: str  # pinned | insufficient_corpus
    documents: int  # retained documents at the last scan (a pinned view: its own count)
    events: tuple[str, ...] = ()

    def coverage(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "state": self.state,
            "documents": self.documents,
            "floor": FREQUENCY_MIN_DOCS,
        }
        if self.state == "pinned":
            out["view_id"] = self.view.version_id
        else:
            out["reason"] = INSUFFICIENT
        return out

    def standing_warnings(self) -> tuple[str, ...]:
        if self.state == "pinned":
            return ()
        return (
            f"{INSUFFICIENT}: {self.documents} of {FREQUENCY_MIN_DOCS} documents retained; "
            "source credit stays unknown, not zero",
        )


def _pinned(view: FrequencyView, *events: str) -> FrequencyOutcome:
    return FrequencyOutcome(view, "pinned", view.doc_count, events)


def _thin(documents: int) -> FrequencyOutcome:
    return FrequencyOutcome(FrequencyView.unknown(), "insufficient_corpus", documents)


def resolve(
    profile: str,
    *,
    pinned: FrequencyView | None,
    last_documents: int | None,
    evidence: Callable[[], Iterable[RawEvidence]],
    scan: bool,
    repin: bool,
    on_step: Hook | None = None,
) -> FrequencyOutcome:
    """Decide this cycle's view; pins one when the corpus first reaches the floor (or on repin).

    ``scan`` says retained evidence may have grown (so an unpinned corpus is worth counting
    again); ``evidence`` yields it lazily, and only when a scan is actually needed.
    """
    if pinned is not None and not repin:
        return _pinned(pinned)
    if pinned is None and not scan and not repin and last_documents is not None:
        return _thin(last_documents)
    view = corpus_view(evidence())
    if view.doc_count < FREQUENCY_MIN_DOCS:
        if pinned is not None:  # an explicit repin never trades the pin for a thin corpus
            refused = (
                f"frequency_repin_refused: {view.doc_count} of {FREQUENCY_MIN_DOCS} documents "
                f"retained; keeping pinned view {pinned.version_id}"
            )
            return _pinned(pinned, refused)
        return _thin(view.doc_count)
    if pinned is not None and view.version_id == pinned.version_id:
        return _pinned(pinned)
    pin(profile, view, on_step)
    note = (
        f"frequency_repinned: view {view.version_id} from {view.doc_count} documents is a new "
        "methodology namespace; the earlier one is kept and never summed with it"
        if pinned is not None
        else f"frequency_pinned: view {view.version_id} from {view.doc_count} documents"
    )
    return _pinned(view, note)
