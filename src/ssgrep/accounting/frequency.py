"""Frozen, versioned candidate-frequency view and fixed-point log-IDF rarity (D5).

The view is built with the SAME scanner and normalization as extraction
(``evidence_tokens.NORMALIZATION``), so a candidate is looked up under exactly the
key it was extracted with. A token the view has not recorded is *unknown*: absence
is never maximal rarity, and a view built over a different vocabulary (lexical
tokens) cannot rate candidates at all.

Rarity is ``log2((N + 1) / (df + 1))`` in integer milli-bits (1000 = 1 bit) computed
with integer arithmetic only, so it is identical on every platform. The scenario
thresholds 8/6/4 are therefore 8000/6000/4000 milli-bits and are UNCALIBRATED
until a stratified 50-receipt exercise says otherwise (design D5).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ssgrep.accounting.contracts import Scenario, digest_of
from ssgrep.accounting.evidence_tokens import NORMALIZATION, Token, tokens_of

BASIS_CANDIDATES = "candidate_tokens_v1"
BASIS_LEXICAL = "lexical_tokens"
BASIS_UNKNOWN = "unknown"
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


@dataclass(frozen=True)
class FrequencyView:
    """Document frequency of normalized candidates over a frozen corpus."""

    version_id: str
    basis: str = BASIS_UNKNOWN
    normalization: str = NORMALIZATION
    doc_count: int = 0
    df: Mapping[str, int] = field(default_factory=dict)
    calibrated: bool = False

    @classmethod
    def unknown(cls) -> FrequencyView:
        """No frequency evidence: every rarity is unknown and nothing earns credit."""
        return cls(version_id="frequency-unknown")

    @property
    def known(self) -> bool:
        return self.basis == BASIS_CANDIDATES and self.doc_count > 0

    def rarity_milli(self, token: Token) -> int | None:
        """Log-IDF in milli-bits, or None (unknown) when the view cannot rate ``token``."""
        if not self.known or self.normalization != NORMALIZATION:
            return None
        count = self.df.get(token.key)
        if count is None:
            return None  # absent from the frozen view: unknown, never maximal rarity
        return milli_log2(self.doc_count + 1, count + 1)


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
