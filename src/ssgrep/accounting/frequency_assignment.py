"""Frozen rarity INPUT generations per lookup, independent from model compatibility."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.frequency import (
    BASIS_CANDIDATES,
    LEXICAL_MAPPING,
    LEXICAL_NORMALIZATION,
    FrequencyView,
)

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


@dataclass(frozen=True)
class FrequencyAssignment:
    invocation_id: str
    view_id: str
    unavailable_reason: str = ""
    profile_scope: str = ""
    revision_name: str = ""
    unavailable_provenance: tuple[tuple[str, str], ...] = ()
    fingerprint_key_id: str | None = None


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
            from ssgrep.accounting.fingerprint import key_id, load_key

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
