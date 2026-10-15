"""Frozen candidate rarity and conservative lexical-superset lower bounds.

An exact candidate's indexed-chunk df cannot exceed the df of any guaranteed
ASCII lexical component (casefolded). The minimum PRESENT component df is thus
a safe upper df, and log-IDF is a LOWER rarity bound, never exact split-token
rarity. Absent components are unknown, never df=0. Bounds below a band's
threshold do not prove rejection; only proven bands can support credit.

Unchanged 8/6/4 thresholds remain uncalibrated: singleton N=199/200 proves 6/4,
while N=511 proves 8. Explicit scanner-normalized source views remain exact.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ssgrep.accounting.contracts import Scenario, digest_of
from ssgrep.accounting.evidence_tokens import NORMALIZATION, Token, tokens_of

BASIS_CANDIDATES = "candidate_tokens_v1"
BASIS_LEXICAL = "lexical_tokens"
BASIS_UNKNOWN = "unknown"
BASIS_INDEX = "keyed_lexical_chunks_v1"
LEXICAL_NORMALIZATION = "lexical_ascii_lower_v1"
LEXICAL_MAPPING = "lexical_superset_lower_bound_v2"
_LEXICAL_COMPONENT = re.compile(r"[a-z0-9_]+")
FRACTION_BITS = 10
MILLI = 1000
UNCALIBRATED = True
# log-IDF thresholds (bits) as integer fixed-point milli-bits; labeled uncalibrated.
THRESHOLD_MILLI: dict[Scenario, int] = {
    Scenario.CONSERVATIVE: 8 * MILLI,
    Scenario.CENTRAL: 6 * MILLI,
    Scenario.OPTIMISTIC: 4 * MILLI,
}


def milli_log2(numerator: int, denominator: int) -> int:
    """floor(1000 * log2(numerator / denominator)) for integers with n >= d > 0."""
    if denominator <= 0 or numerator < denominator:
        raise ValueError("expected numerator >= denominator > 0")
    whole = numerator.bit_length() - denominator.bit_length()
    num, den = numerator, denominator
    den <<= whole
    if num < den:  # the estimate overshot by one bit
        whole -= 1
        num <<= 1
    # now 1 <= num/den < 2; extract fractional bits by repeated squaring
    fraction = 0
    for _ in range(FRACTION_BITS):
        num, den = num * num, den * den
        fraction <<= 1
        if num >= 2 * den:
            fraction |= 1
            den *= 2
        # keep numbers bounded without changing the ratio
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
            from ssgrep.accounting.fingerprint import load_key

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
