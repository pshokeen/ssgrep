"""Comparable context-volume units, provider usage, and offline tokenizers.

Pure integer arithmetic only (design D6). ``chars_div4_v1`` sums the characters of
*deduplicated* spans first and divides once, so rounding is consistent. Provider
usage is a separate series and is never added to estimated context tokens.
Tokenizers are loaded only from a locally pinned file; nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ssgrep.accounting.contracts import Basis, Quantity

CHARS_DIV4_V1 = "chars_div4_v1"
CONTEXT_TOKENS = "context_tokens"
WORK_SECONDS = "work_seconds"
PROVIDER_USAGE_V1 = "provider_usage_v1"
PROVIDER_TOKENS = "provider_tokens"
USAGE_CATEGORIES = ("input_uncached", "cache_read", "cache_write", "output", "reasoning")


class UnitMismatchError(ValueError):
    """Raised instead of silently combining incomparable measurements."""


@dataclass(frozen=True)
class UnitSpec:
    """Everything that must match for two quantities to be added or subtracted."""

    unit: str
    method: str
    tokenizer: str | None = None
    currency: str | None = None


def union_length(intervals: Iterable[tuple[int, int]]) -> int:
    """Total length of the union of half-open integer intervals."""
    total = 0
    cur_start = cur_end = -1
    for start, end in sorted(intervals):
        if end < start or start < 0:
            raise ValueError(f"invalid interval ({start}, {end})")
        if start > cur_end:
            total += cur_end - cur_start if cur_end >= 0 else 0
            cur_start, cur_end = start, end
        elif end > cur_end:
            cur_end = end
    return total + (cur_end - cur_start if cur_end >= 0 else 0)


def merged_span_chars(spans: Iterable[tuple[str, str, int, int]]) -> int:
    """Characters of the union of ``(origin_event, digest, start, end)`` spans."""
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for origin, digest, start, end in spans:
        grouped.setdefault((origin, digest), []).append((start, end))
    return sum(union_length(intervals) for intervals in grouped.values())


def chars_div4_v1(total_chars: int) -> int:
    """Deterministic integer estimate: floor(chars / 4); never provider tokens."""
    if total_chars < 0:
        raise ValueError("character count cannot be negative")
    return total_chars // 4


def context_estimate(total_chars: int) -> Quantity:
    value = chars_div4_v1(total_chars)
    return Quantity(value, CONTEXT_TOKENS, Basis.ESTIMATED, CHARS_DIV4_V1)


@dataclass(frozen=True)
class QuantitySum:
    unit: str
    method: str
    known: int
    known_count: int
    missing_count: int
    reasons: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.missing_count == 0


def sum_quantities(quantities: Sequence[Quantity]) -> QuantitySum:
    """Sum known values; refuse different units or methods; keep missing reasons."""
    if not quantities:
        raise ValueError("nothing to sum")
    units = {q.unit for q in quantities}
    methods = {q.method for q in quantities if q.value is not None}
    if len(units) > 1 or len(methods) > 1:
        raise UnitMismatchError(f"mixed units {sorted(units)} / methods {sorted(methods)}")
    known = [q.value for q in quantities if q.value is not None]
    reasons = sorted({q.reason for q in quantities if q.value is None})
    return QuantitySum(
        unit=next(iter(units)),
        method=next(iter(methods), ""),
        known=sum(known),
        known_count=len(known),
        missing_count=len(quantities) - len(known),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class UsageUpdate:
    """One provider usage report; ``stream`` groups repeated/cumulative updates."""

    provider: str
    stream: str
    seq: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cumulative: bool = False
    input_includes_cache: bool = False
    output_includes_reasoning: bool = False

    def values(self) -> tuple[int, int, int, int, int]:
        return (
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_write_tokens,
            self.reasoning_tokens,
        )


@dataclass(frozen=True)
class ProviderUsage:
    """Normalized, non-overlapping categories for one provider."""

    provider: str
    categories: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]

    def quantity(self, category: str) -> Quantity:
        for name, value in self.categories:
            if name == category:
                method = f"{PROVIDER_USAGE_V1}:{self.provider}"
                return Quantity(value, PROVIDER_TOKENS, Basis.REPORTED, method)
        raise KeyError(category)


def _stream_totals(updates: Sequence[UsageUpdate], warnings: list[str]) -> tuple[int, ...]:
    best: dict[int, UsageUpdate] = {}
    for update in updates:  # one report per seq; conflicting repeats keep the larger
        held = best.get(update.seq)
        if held is None or update.values() > held.values():
            best[update.seq] = update
    ordered = [best[seq] for seq in sorted(best)]
    if not ordered[-1].cumulative:
        return ordered[-1].values()  # streaming updates: only the final report counts
    totals = [0] * 5
    previous = (0,) * 5
    for update in ordered:
        current = update.values()
        for i, (now, before) in enumerate(zip(current, previous, strict=True)):
            if now < before:
                warnings.append(f"counter_reset:{update.stream}")
            totals[i] += now - before if now >= before else now
        previous = current
    return tuple(totals)


def _normalize_stream(totals: tuple[int, ...], last: UsageUpdate) -> dict[str, int] | None:
    inp, out, read, write, reasoning = totals
    if last.input_includes_cache:
        inp -= read + write
    if last.output_includes_reasoning:
        out -= reasoning
    if min(inp, out) < 0:
        return None
    return {
        "input_uncached": inp,
        "cache_read": read,
        "cache_write": write,
        "output": out,
        "reasoning": reasoning,
    }


def normalize_usage(updates: Iterable[UsageUpdate]) -> dict[str, ProviderUsage]:
    """Per-provider usage without repeated streaming/cumulative or overlapping counts."""
    streams: dict[tuple[str, str], list[UsageUpdate]] = {}
    for update in updates:
        streams.setdefault((update.provider, update.stream), []).append(update)
    sums: dict[str, dict[str, int]] = {}
    notes: dict[str, list[str]] = {}
    for (provider, stream), group in sorted(streams.items()):
        warnings = notes.setdefault(provider, [])
        sums.setdefault(provider, dict.fromkeys(USAGE_CATEGORIES, 0))
        last = max(group, key=lambda u: u.seq)
        normalized = _normalize_stream(_stream_totals(group, warnings), last)
        if normalized is None:
            warnings.append(f"inconsistent_usage:{stream}")
            continue
        for name, value in normalized.items():
            sums[provider][name] += value
    return {
        provider: ProviderUsage(
            provider,
            tuple((name, sums[provider][name]) for name in USAGE_CATEGORIES),
            tuple(sorted(set(notes[provider]))),
        )
        for provider in sorted(sums)
    }


@dataclass(frozen=True)
class TokenizerPin:
    """Exact offline tokenizer identity: library version plus file digest."""

    library: str
    version: str
    file_path: str
    file_sha256: str

    @property
    def identity(self) -> str:
        return f"{self.library}=={self.version}:sha256:{self.file_sha256}"


@dataclass(frozen=True)
class TokenizerResult:
    identity: str
    reason: str
    counter: Callable[[str], int] | None

    @property
    def available(self) -> bool:
        return self.counter is not None

    def count(self, text: str) -> Quantity:
        if self.counter is None:
            return Quantity.unavailable(CONTEXT_TOKENS, self.reason)
        return Quantity(
            self.counter(text), CONTEXT_TOKENS, Basis.OBSERVED, f"exact:{self.identity}"
        )


def resolve_tokenizer(
    pin: TokenizerPin,
    importer: Callable[[str], Any] = importlib.import_module,
    version_of: Callable[[str], str] = importlib.metadata.version,
) -> TokenizerResult:
    """Load a pinned tokenizer from local files only; never download or fall back."""

    def unavailable(reason: str) -> TokenizerResult:
        return TokenizerResult(pin.identity, reason, None)

    if pin.library != "tokenizers":
        return unavailable(f"unsupported_tokenizer_library:{pin.library}")
    try:
        module = importer(pin.library)
    except ImportError:
        return unavailable(f"tokenizer_library_not_installed:{pin.library}")
    try:
        installed = version_of(pin.library)
    except importlib.metadata.PackageNotFoundError:
        return unavailable(f"tokenizer_library_not_installed:{pin.library}")
    if installed != pin.version:
        return unavailable(f"tokenizer_version_mismatch:{installed}!={pin.version}")
    try:
        payload = Path(pin.file_path).read_bytes()
    except OSError:
        return unavailable("tokenizer_file_missing")
    if hashlib.sha256(payload).hexdigest() != pin.file_sha256:
        return unavailable("tokenizer_file_digest_mismatch")
    try:
        tokenizer = module.Tokenizer.from_file(pin.file_path)
    except Exception as exc:  # third-party loader; surface category, not text
        return unavailable(f"tokenizer_load_failed:{type(exc).__name__}")

    def counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return TokenizerResult(pin.identity, "", counter)
