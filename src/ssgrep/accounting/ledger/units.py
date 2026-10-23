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
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ssgrep.accounting.ledger.contracts import Basis, Quantity, digest_of

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


WORK_MS = "work_ms"
WORK_NS = "work_ns"
NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
MONOTONIC_V1 = "monotonic_v1"
CAPPED_GAP_V1 = "capped_gap_v1"
TRANSCRIPT_INTERVALS_V1 = "transcript_intervals_v1"
GAP_CAP_MS = 300_000
GAP_CAP_NS = GAP_CAP_MS * NS_PER_MS
HUMAN_TIME = "human_seconds"


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Sorted disjoint union of half-open integer intervals (empty ones dropped)."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError(f"invalid interval ({start}, {end})")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def subtract_intervals(
    intervals: Sequence[tuple[int, int]], removed: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """``intervals`` minus ``removed`` (both merged on entry)."""
    out: list[tuple[int, int]] = []
    holes = merge_intervals(removed)
    for start, end in merge_intervals(intervals):
        cursor = start
        for h_start, h_end in holes:
            if h_end <= cursor or h_start >= end:
                continue
            if h_start > cursor:
                out.append((cursor, h_start))
            cursor = max(cursor, h_end)
        if cursor < end:
            out.append((cursor, end))
    return tuple(out)


def measured_duration(start_ns: int, end_ns: int) -> Quantity:
    """One raw monotonic duration in ns; no intermediate quantization."""
    if end_ns < start_ns:
        return Quantity.unavailable(WORK_NS, "monotonic_clock_reversed")
    return Quantity(end_ns - start_ns, WORK_NS, Basis.OBSERVED, MONOTONIC_V1)


def seconds_quantity(total_ms: int, method: str, basis: Basis = Basis.OBSERVED) -> Quantity:
    """Whole seconds from an already-summed millisecond total (floor once)."""
    return Quantity(total_ms // 1000, WORK_SECONDS, basis, method)


def service_milliseconds(duration_ns: Quantity) -> Quantity:
    """Presentation only: the input must retain actual observed raw ns."""
    if duration_ns.value is None:
        return Quantity.unavailable(WORK_MS, duration_ns.reason)
    if (
        duration_ns.unit != WORK_NS
        or duration_ns.basis is not Basis.OBSERVED
        or duration_ns.method != MONOTONIC_V1
    ):
        return Quantity.unavailable(WORK_MS, "nanosecond_precision_unavailable")
    return Quantity(duration_ns.value // NS_PER_MS, WORK_MS, Basis.OBSERVED, duration_ns.method)


def ns_seconds_quantity(total_ns: int, method: str, basis: Basis = Basis.OBSERVED) -> Quantity:
    """Whole seconds from an already-summed nanosecond quantity."""
    return Quantity(total_ns // NS_PER_SECOND, WORK_SECONDS, basis, method)


@dataclass(frozen=True)
class ActorInterval:
    """One observed work interval of one actor; times are wall-clock epoch ms."""

    actor: str
    start_ms: int
    end_ms: int

    @property
    def valid(self) -> bool:
        return self.end_ms >= self.start_ms


@dataclass(frozen=True)
class WorkTime:
    """Work-seconds (per-actor sum) and elapsed (interval union) kept separate."""

    work_ms: int
    elapsed_ms: int
    invalid_intervals: int

    @property
    def work_seconds(self) -> Quantity:
        return seconds_quantity(self.work_ms, TRANSCRIPT_INTERVALS_V1)

    @property
    def elapsed_seconds(self) -> Quantity:
        return seconds_quantity(self.elapsed_ms, TRANSCRIPT_INTERVALS_V1)

    @property
    def human_time(self) -> Quantity:
        """Parallel work is not waiting time; without dependency evidence: unknown."""
        return Quantity.unavailable(HUMAN_TIME, "no_dependency_evidence")


def work_time(intervals: Iterable[ActorInterval]) -> WorkTime:
    """Per-actor union summed (work) vs union over all actors (elapsed)."""
    by_actor: dict[str, list[tuple[int, int]]] = {}
    invalid = 0
    for interval in intervals:
        if not interval.valid:
            invalid += 1
            continue
        by_actor.setdefault(interval.actor, []).append((interval.start_ms, interval.end_ms))
    work = sum(union_length(spans) for spans in by_actor.values())
    elapsed = union_length(span for spans in by_actor.values() for span in spans)
    return WorkTime(work, elapsed, invalid)


@dataclass(frozen=True)
class CappedGap:
    """``capped_gap_v1`` result: known subtotal plus the count of unknown gaps."""

    known_ms: int
    invalid_gaps: int
    capped_gaps: int

    @property
    def quantity(self) -> Quantity:
        if self.invalid_gaps:
            return Quantity.unavailable(WORK_SECONDS, "invalid_timing_in_gaps")
        return seconds_quantity(self.known_ms, CAPPED_GAP_V1, Basis.ESTIMATED)


def capped_gap_v1(timestamps_ms: Sequence[int | None]) -> CappedGap:
    """Cap adjacent gaps; absent/reversed clocks or no bounded pair remain unknown."""
    known = capped = 0
    invalid = int(len(timestamps_ms) < 2)
    for before, after in zip(timestamps_ms, timestamps_ms[1:], strict=False):
        if before is None or after is None or after < before:
            invalid += 1
            continue
        gap = after - before
        if gap > GAP_CAP_MS:
            known += GAP_CAP_MS
            capped += 1
        else:
            known += gap
    return CappedGap(known, invalid, capped)


CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
PER_MILLION = 1_000_000
REPORTED_TRUST = ("provider_exact", "provider_derived", "unverified")


class TariffError(ValueError):
    """The supplied tariff is invalid; nothing is priced from an invalid tariff."""


@dataclass(frozen=True)
class TariffEntry:
    model: str
    category: str
    currency: str
    micros_per_million_tokens: int
    effective_from: str  # inclusive YYYY-MM-DD
    effective_to: str | None = None  # exclusive; None = open ended

    def covers(self, day: str) -> bool:
        return self.effective_from <= day and (self.effective_to is None or day < self.effective_to)


@dataclass(frozen=True)
class Tariff:
    version: str
    entries: tuple[TariffEntry, ...]

    @property
    def identity(self) -> str:
        return f"{self.version}:{digest_of(self)[:16]}"

    def find(self, model: str, category: str, currency: str, day: str) -> TariffEntry | None:
        for entry in self.entries:
            if (entry.model, entry.category, entry.currency) == (model, category, currency):
                if entry.covers(day):
                    return entry
        return None


def _iso(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TariffError(f"{field_name} must be an ISO date")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise TariffError(f"{field_name} must be an ISO date") from None
    return value


def _entry(raw: Any) -> TariffEntry:
    if not isinstance(raw, Mapping):
        raise TariffError("tariff entry must be an object")
    price = raw.get("micros_per_million_tokens")
    if isinstance(price, bool) or not isinstance(price, int) or price < 0:
        raise TariffError("price must be a non-negative integer (no floats)")
    model, category, currency = raw.get("model"), raw.get("category"), raw.get("currency")
    if not isinstance(model, str) or not model:
        raise TariffError("model is required")
    if category not in USAGE_CATEGORIES:
        raise TariffError(f"unknown category {category!r}")
    if not isinstance(currency, str) or not CURRENCY_RE.fullmatch(currency):
        raise TariffError("currency must be a 3-letter upper-case code")
    start = _iso(raw.get("effective_from"), "effective_from")
    end = raw.get("effective_to")
    if end is not None and _iso(end, "effective_to") <= start:
        raise TariffError("effective_to must be after effective_from")
    return TariffEntry(model, category, currency, price, start, end)


def _overlaps(a: TariffEntry, b: TariffEntry) -> bool:
    a_end, b_end = a.effective_to or "9999-12-31", b.effective_to or "9999-12-31"
    return a.effective_from < b_end and b.effective_from < a_end


def validate_tariff(raw: Any) -> Tariff:
    """Validate an explicitly supplied offline tariff; reject anything ambiguous."""
    if not isinstance(raw, Mapping):
        raise TariffError("tariff must be an object")
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        raise TariffError("tariff version is required")
    rows = raw.get("entries")
    if not isinstance(rows, list) or not rows:
        raise TariffError("tariff needs at least one entry")
    entries = tuple(
        sorted(
            (_entry(r) for r in rows),
            key=lambda e: (e.model, e.category, e.currency, e.effective_from),
        )
    )
    for i, first in enumerate(entries):
        for second in entries[i + 1 :]:
            same = (first.model, first.category, first.currency) == (
                second.model,
                second.category,
                second.currency,
            )
            if same and _overlaps(first, second):
                raise TariffError(
                    f"overlapping effective intervals for {first.model}/{first.category}"
                )
    return Tariff(version, entries)


def money_unit(currency: str) -> str:
    return f"micro_{currency}"


def modeled_cost(
    tariff: Tariff | None, usage: ProviderUsage, model: str, day: str, currency: str
) -> Quantity:
    """Tariff price of normalized usage; unavailable (never zero) without a tariff."""
    unit = money_unit(currency)
    if tariff is None:
        return Quantity.unavailable(unit, "no_tariff")
    total = 0
    for category, tokens in usage.categories:
        if tokens == 0:
            continue
        entry = tariff.find(model, category, currency, day)
        if entry is None:
            return Quantity.unavailable(unit, f"no_applicable_tariff:{category}")
        total += tokens * entry.micros_per_million_tokens
    return Quantity(total // PER_MILLION, unit, Basis.ESTIMATED, f"tariff:{tariff.identity}")


def reported_cost(micros: int | None, currency: str, trust: str = "provider_exact") -> Quantity:
    """Provider-reported cost; zero stays a reported zero, missing stays unavailable."""
    if trust not in REPORTED_TRUST:
        raise ValueError(f"unknown trust level {trust!r}")
    unit = money_unit(currency)
    if micros is None:
        return Quantity.unavailable(unit, "provider_cost_missing")
    return Quantity(micros, unit, Basis.REPORTED, f"provider_reported:{trust}")


def sum_money(quantities: Iterable[Quantity]) -> Quantity:
    """Add costs of ONE currency and kind; refuse a mix; any unknown member -> unknown."""
    items = list(quantities)
    if not items:
        raise ValueError("nothing to sum")
    if len({q.unit for q in items}) > 1:
        raise ValueError("cannot combine currencies")
    known = [q for q in items if q.value is not None]
    if len({q.basis for q in known}) > 1 or len({q.method for q in known}) > 1:
        raise ValueError("cannot combine reported and modeled/other-method costs")
    if len(known) != len(items):
        reason = next(q.reason for q in items if q.value is None)
        return Quantity.unavailable(items[0].unit, reason)
    first = known[0]
    return Quantity(sum(q.value or 0 for q in known), first.unit, first.basis, first.method)
