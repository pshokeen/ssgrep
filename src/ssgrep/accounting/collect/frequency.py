"""Collection frequency pins, indexed rarity, and frozen assignment management."""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import islice
from types import MappingProxyType
from typing import Any

from ssgrep.accounting.capture.fingerprint import load_key
from ssgrep.accounting.collect.replay_evidence import NORMALIZATION, Token, tokens_of
from ssgrep.accounting.collect.store import CollectError, dumps, loads
from ssgrep.accounting.ledger.contracts import (
    THRESHOLD_MILLI,
    RawEvidence,
    Scenario,
    digest_of,
)
from ssgrep.accounting.ledger.ledger_contracts import FrequencyAssignment
from ssgrep.accounting.privacy import storage
from ssgrep.accounting.privacy.storage import from_plain, to_plain

# --- accounting/frequency.py ---

BASIS_CANDIDATES = "candidate_tokens_v1"
BASIS_LEXICAL = "lexical_tokens"
BASIS_UNKNOWN = "unknown"
BASIS_INDEX = "keyed_lexical_chunks_v1"
LEXICAL_NORMALIZATION = "lexical_ascii_lower_v1"
LEXICAL_MAPPING = "lexical_superset_lower_bound_v2"
_LEXICAL_COMPONENT = re.compile(r"[a-z0-9_]+")
FRACTION_BITS = 10
MILLI = 1000


def milli_log2(numerator: int, denominator: int) -> int:
    """floor(1000 * log2(numerator / denominator)) for integers with n >= d > 0."""
    if denominator <= 0 or numerator < denominator:
        raise ValueError("expected numerator >= denominator > 0")
    whole = numerator.bit_length() - denominator.bit_length()
    num, den = numerator, denominator
    den <<= whole
    if num < den:
        whole -= 1
        num <<= 1
    fraction = 0
    for _ in range(FRACTION_BITS):
        num, den = num * num, den * den
        fraction <<= 1
        if num >= 2 * den:
            fraction |= 1
            den *= 2
        shift = min(num.bit_length(), den.bit_length()) - 64
        if shift > 0:
            num, den = num >> shift, den >> shift
            den = max(den, 1)
    return whole * MILLI + ((fraction * MILLI) >> FRACTION_BITS)


def lexical_key(key: bytes, text: str) -> str:
    """Domain-separated keyed lexical metadata; neither token nor key is persisted."""
    return hmac.digest(key, b"ssgrep-frequency-lexical-v1\0" + text.encode(), "sha256").hex()


def frequency_key_id(key: bytes) -> str:
    return hmac.digest(key, b"ssgrep-frequency-key-id-v1", "sha256").hex()


@dataclass(frozen=True)
class RarityEvidence:
    value: int | None = None
    kind: str = "unavailable"  # exact | lower_bound | unavailable
    reason: str = ""
    upper_df: int | None = None


@dataclass(frozen=True)
class FrequencyView:
    """Document frequency of normalized candidates over a frozen corpus."""

    version_id: str
    basis: str = BASIS_UNKNOWN
    normalization: str = NORMALIZATION
    doc_count: int = 0
    df: Mapping[str, int] = field(default_factory=dict)
    calibrated: bool = False
    generation: str = ""
    corpus_digest: str = ""
    document_unit: str = ""
    coverage: str = ""
    profile_scope: str = ""
    authority_id: str = ""
    mapping: str = ""
    key_id: str = ""
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "df", MappingProxyType(dict(self.df)))

    @classmethod
    def unknown(cls, reason: str = "frequency_unavailable", *, profile: str = "") -> FrequencyView:
        """Typed unavailable frequency evidence, never a known-zero rarity."""
        return cls(version_id="frequency-unknown", unavailable_reason=reason, profile_scope=profile)

    @property
    def known(self) -> bool:
        if type(self.doc_count) is not int or self.doc_count <= 0 or self.unavailable_reason:
            return False
        if self.basis == BASIS_CANDIDATES:
            return self.normalization == NORMALIZATION
        return (
            self.basis == BASIS_INDEX
            and self.normalization == LEXICAL_NORMALIZATION
            and self.mapping == LEXICAL_MAPPING
            and self.document_unit == "indexed_chunk"
            and self.coverage == "complete_indexed_chunks"
            and bool(
                self.generation
                and self.corpus_digest
                and self.profile_scope
                and self.authority_id
                and self.key_id
            )
        )

    def rarity_evidence(self, token: Token) -> RarityEvidence:
        if not self.known:
            return RarityEvidence(reason=self.unavailable_reason or "frequency_unavailable")
        if self.basis == BASIS_INDEX:
            if (
                token.kind not in ("identifier", "error_code", "path", "flag")
                or not token.text.isascii()
                or not 1 <= len(token.text) <= 160
            ):
                return RarityEvidence(reason="frequency_mapping_unavailable")
            try:
                key = load_key(self.profile_scope, create=False)
            except (OSError, ValueError):
                return RarityEvidence(reason="frequency_key_unavailable")
            if key is None or frequency_key_id(key) != self.key_id:
                return RarityEvidence(reason="frequency_key_unavailable")
            counts = []
            for component in set(_LEXICAL_COMPONENT.findall(token.text.lower())):
                if len(component) < 2:
                    continue
                count = self.df.get(lexical_key(key, component))
                if count is None:
                    continue
                if type(count) is not int or not 1 <= count <= self.doc_count:
                    return RarityEvidence(reason="frequency_df_invalid")
                counts.append(count)
            if not counts:
                return RarityEvidence(reason="frequency_candidate_absent")
            upper_df = min(counts)
            bound = milli_log2(self.doc_count + 1, upper_df + 1)
            reason = "" if bound >= min(THRESHOLD_MILLI.values()) else "rarity_bound_inconclusive"
            return RarityEvidence(bound, "lower_bound", reason, upper_df)
        count = self.df.get(token.key)
        if type(count) is not int or not 1 <= count <= self.doc_count:
            return RarityEvidence(reason="frequency_candidate_absent")
        return RarityEvidence(milli_log2(self.doc_count + 1, count + 1), "exact", upper_df=count)

    def rarity_milli(self, token: Token) -> int | None:
        """Exact rarity or a useful lower bound; an inconclusive bound stays unknown."""
        evidence = self.rarity_evidence(token)
        return evidence.value if not evidence.reason else None

    def identity(self) -> str:
        """Content identity; preserve legacy explicit candidate views without rewriting them."""
        body = {
            "basis": self.basis,
            "norm": self.normalization,
            "n": self.doc_count,
            "df": dict(sorted(self.df.items())),
        }
        if self.generation or self.profile_scope or self.unavailable_reason:
            body["provenance"] = {
                "generation": self.generation,
                "digest": self.corpus_digest,
                "unit": self.document_unit,
                "coverage": self.coverage,
                "profile": self.profile_scope,
                "authority": self.authority_id,
                "mapping": self.mapping,
                "key_id": self.key_id,
                "unavailable": self.unavailable_reason,
                "calibrated": self.calibrated,
            }
        return f"freq-{digest_of(body)[:24]}"


def build_view(documents: Iterable[str], *, basis: str = BASIS_CANDIDATES) -> FrequencyView:
    """Frozen view from document texts using the extractor's own scanner."""
    df: dict[str, int] = {}
    total = 0
    for text in documents:
        total += 1
        for token in tokens_of(text):
            df[token.key] = df.get(token.key, 0) + 1
    ordered = dict(sorted(df.items()))
    version = digest_of({"basis": basis, "norm": NORMALIZATION, "n": total, "df": ordered})
    return FrequencyView(f"freq-{version[:24]}", basis, NORMALIZATION, total, ordered)


def lexical_view(df: Mapping[str, int], doc_count: int) -> FrequencyView:
    """Wrap lexical-token statistics; they use another vocabulary and rate nothing."""
    ordered = dict(sorted(df.items()))
    version = digest_of({"basis": BASIS_LEXICAL, "n": doc_count, "df": ordered})
    return FrequencyView(f"freq-lex-{version[:24]}", BASIS_LEXICAL, "lexical", doc_count, ordered)


def eligible_scenarios(rarity: int | None) -> tuple[Scenario, ...]:
    """Scenarios whose rarity threshold ``rarity`` meets; empty when unknown."""
    if rarity is None:
        return ()
    return tuple(s for s, limit in THRESHOLD_MILLI.items() if rarity >= limit)


# --- collector_frequency.py ---


# --- collector_frequency.py ---
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
    if data is None:  # pragma: no cover - file removed between open/fstat and read
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
    from ssgrep.accounting.collect.frequency_index import read_index_view

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


# --- frequency_assignment.py ---

SCHEMA = 1

MODEL_VERSION = "rarity-log2-frozen-lookup-inputs-v2"


def model_identity(frequency: FrequencyView) -> str:
    """Model identity excludes corpus generation/df/digest/profile/key/input availability."""
    if frequency.basis == BASIS_CANDIDATES:
        vocabulary = (
            frequency.basis,
            frequency.normalization,
            "scanner_exact_df",
            "evidence_record",
        )
    else:
        vocabulary = (
            "keyed_lexical_chunks_v1",
            LEXICAL_NORMALIZATION,
            LEXICAL_MAPPING,
            "indexed_chunk",
        )
    return f"frequency-model-{digest_of([MODEL_VERSION, vocabulary])[:24]}"


class FrequencyAssignments:
    """Freeze recognized lookups, except explicitly provisional pending build inputs.

    A pending build is not an accepted frequency pin: matching generation/scope/key
    completion creates a named late-evidence revision. Actual absence/refusal and
    accepted known inputs stay frozen; explicit recomputation is still required
    to replace them. Missing historical files are never replaced with current df.
    """

    def __init__(
        self,
        current: FrequencyView,
        *,
        assignments: Iterable[FrequencyAssignment] = (),
        loader: Callable[[str], FrequencyView] | None = None,
        revision_name: str = "",
        fingerprint_key_id: str | None = None,
    ) -> None:
        self.current = current
        self.fingerprint_key_id = fingerprint_key_id
        if self.fingerprint_key_id is None and current.profile_scope:
            from ssgrep.accounting.capture.fingerprint import key_id, load_key

            try:
                self.fingerprint_key_id = key_id(load_key(current.profile_scope, create=False))
            except (OSError, ValueError):
                self.fingerprint_key_id = None
        self._records: dict[str, FrequencyAssignment] = {}
        self.completed_pending = False
        for record in assignments:
            if record.invocation_id in self._records:
                raise ValueError("duplicate lookup frequency assignment")
            pending = dict(record.unavailable_provenance)
            if (
                record.unavailable_reason == "frequency_build_pending"
                and current.known
                and record.profile_scope == current.profile_scope
                and record.fingerprint_key_id == self.fingerprint_key_id
                and pending.get("generation") == current.generation
                and pending.get("authority_id") == current.authority_id
                and pending.get("key_id") == current.key_id
            ):
                record = replace(
                    record,
                    view_id=current.version_id,
                    unavailable_reason="",
                    revision_name=f"frequency-build-complete:{current.version_id}",
                    unavailable_provenance=(),
                )
                self.completed_pending = True
            self._records[record.invocation_id] = record
        self.loader = loader
        self.revision_name = revision_name
        self._views = {current.version_id: current}
        self.unavailable: dict[str, str] = {}

    def bind(self, invocation_id: str, execution_ids: Iterable[str]) -> None:
        """A late native join inherits the original accepted journal lookup input."""
        if invocation_id in self._records:
            return
        aliases = [
            self._records[f"journal:{execution_id}"]
            for execution_id in execution_ids
            if f"journal:{execution_id}" in self._records
        ]
        if not aliases:
            self.view_for(invocation_id)
            return
        identities = {
            (
                row.view_id,
                row.unavailable_reason,
                row.profile_scope,
                row.revision_name,
                row.unavailable_provenance,
                row.fingerprint_key_id,
            )
            for row in aliases
        }
        if len(identities) == 1:
            self._records[invocation_id] = replace(aliases[0], invocation_id=invocation_id)
        else:
            self._records[invocation_id] = FrequencyAssignment(
                invocation_id,
                "frequency-unknown",
                "frequency_assignment_ambiguous",
                self.current.profile_scope,
                self.revision_name,
                fingerprint_key_id=self.fingerprint_key_id,
            )

    def view_for(self, invocation_id: str) -> FrequencyView:
        record = self._records.get(invocation_id)
        if record is None:
            record = FrequencyAssignment(
                invocation_id,
                self.current.version_id,
                self.current.unavailable_reason,
                self.current.profile_scope,
                self.revision_name,
                self._unavailable_provenance(),
                self.fingerprint_key_id,
            )
            self._records[invocation_id] = record
        if record.fingerprint_key_id != self.fingerprint_key_id:
            self.unavailable[invocation_id] = "fingerprint_key_revision_unavailable"
            return FrequencyView.unknown(
                "fingerprint_key_revision_unavailable", profile=record.profile_scope
            )
        if record.view_id == "frequency-unknown":
            return replace(
                FrequencyView.unknown(
                    record.unavailable_reason or "frequency_unavailable",
                    profile=record.profile_scope,
                ),
                **dict(record.unavailable_provenance),
            )
        view = self._views.get(record.view_id)
        if view is None:
            try:
                if self.loader is None:
                    raise ValueError("historical frequency loader unavailable")
                view = self.loader(record.view_id)
                if view.version_id != record.view_id:
                    raise ValueError("historical view identity changed")
                self._views[record.view_id] = view
            except Exception:
                self.unavailable[invocation_id] = "frequency_revision_unavailable"
                return FrequencyView.unknown(
                    "frequency_revision_unavailable", profile=record.profile_scope
                )
        if model_identity(view) != model_identity(self.current):
            self.unavailable[invocation_id] = "frequency_model_incompatible"
            return FrequencyView.unknown(
                "frequency_model_incompatible", profile=record.profile_scope
            )
        return view

    def _unavailable_provenance(self) -> tuple[tuple[str, str], ...]:
        if self.current.known:
            return ()
        return tuple(
            (field, getattr(self.current, field))
            for field in (
                "generation",
                "corpus_digest",
                "normalization",
                "document_unit",
                "authority_id",
                "mapping",
                "coverage",
                "key_id",
            )
        )

    def provenance(self, invocation_id: str) -> tuple[tuple[str, str], ...]:
        view = self.view_for(invocation_id)
        record = self._records[invocation_id]
        return tuple(
            sorted(
                {
                    "frequency": record.view_id,
                    "frequency_generation": view.generation,
                    "frequency_corpus_digest": view.corpus_digest,
                    "frequency_normalization": view.normalization,
                    "frequency_mapping": view.mapping,
                    "frequency_document_unit": view.document_unit,
                    "frequency_profile_scope": record.profile_scope,
                    "frequency_authority": view.authority_id,
                    "frequency_revision": record.revision_name,
                    "frequency_unavailable": view.unavailable_reason,
                    "fingerprint_key_id": record.fingerprint_key_id or "",
                }.items()
            )
        )

    def records(
        self, invocation_ids: Iterable[str] | None = None
    ) -> tuple[FrequencyAssignment, ...]:
        if invocation_ids is None:
            keys = sorted(self._records)
        else:
            keys = sorted({key for key in invocation_ids if key in self._records})
        return tuple(self._records[key] for key in keys)

    def recompute(self, name: str) -> None:
        if not name:
            raise ValueError("recomputation must have an explicit revision name")
        self._records = {
            key: replace(
                record,
                view_id=self.current.version_id,
                unavailable_reason=self.current.unavailable_reason,
                profile_scope=self.current.profile_scope,
                revision_name=name,
                unavailable_provenance=self._unavailable_provenance(),
                fingerprint_key_id=self.fingerprint_key_id,
            )
            for key, record in self._records.items()
        }
        self.revision_name = name


# --- frequency authority ---
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
    from ssgrep.accounting.collect.frequency_index import _authority, read_index_view

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
        from ssgrep.accounting.privacy.policy import contains

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


# --- frequency retention ---
_VIEW = re.compile(r"frequency-(freq-[0-9a-f]{24})\.json")


@dataclass(frozen=True)
class ViewPruneOutcome:
    scanned: int = 0
    removed: int = 0
    retained: int = 0


def prune_views(
    profile: str,
    *,
    candidates: Iterable[str],
    referenced: Iterable[str],
    before_day: str,
    limit: int = 32,
) -> ViewPruneOutcome:
    """Reclaim an explicitly scheduled bounded batch of expired orphan views."""
    datetime.strptime(before_day, "%Y-%m-%d")
    if limit < 1 or limit > 128:
        raise ValueError("frequency maintenance batch must be 1..128")
    protected = set(referenced)
    try:
        directory = storage.open_profile_dir(profile, "collector", create=False)
    except FileNotFoundError:
        return ViewPruneOutcome()
    scanned = removed = retained = 0
    try:
        pointer = storage.read_file(directory, POINTER, MAX_POINTER_BYTES)
        if pointer is not None:
            protected.add(loads(pointer)["view_id"])
        for name in islice(candidates, limit):
            scanned += 1
            match = _VIEW.fullmatch(name)
            if match is None or match[1] in protected:
                retained += 1
                continue
            try:
                descriptor = storage.open_regular(directory, name)
            except FileNotFoundError:
                continue
            try:
                info = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if datetime.fromtimestamp(info.st_mtime, UTC).date().isoformat() >= before_day:
                retained += 1
                continue
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (  # pragma: no cover - entry replaced between the two back-to-back stats
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                retained += 1
                continue
            os.unlink(name, dir_fd=directory)
            _LOADED.pop((str(storage.data_root()), profile, match[1]), None)
            removed += 1
        if removed:
            os.fsync(directory)
    finally:
        os.close(directory)
    return ViewPruneOutcome(scanned, removed, retained)


# --- frequency purge policy ---
MARKER = "frequency-purged.json"
MAX_MARKER_BYTES = 1024 * 1024


def erasure_document(fd: int, sel: Any) -> dict[str, Any]:
    prior: dict[str, Any] = {}
    try:
        raw = storage.read_file(fd, MARKER, MAX_MARKER_BYTES)
        if raw is not None:
            parsed = loads(raw)
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
        pointer = storage.read_file(fd, POINTER, 4096)
        if pointer is not None:
            view_id = loads(pointer).get("view_id")
            if isinstance(view_id, str) and _VIEW_ID.fullmatch(view_id) is not None:
                data = storage.read_file(fd, view_file_name(view_id), MAX_VIEW_BYTES)
                view = loads(data) if data is not None else {}
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
