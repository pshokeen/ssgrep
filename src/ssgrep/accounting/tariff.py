"""Optional offline tariff validation and provider-reported cost (design D6 dollars).

Prices are integers: micro-currency units per one million tokens, so no floats are
ever involved. A modeled price exists only when an explicit, versioned tariff has
an entry for the model, category, currency, and effective date; otherwise the
result is *unavailable*, never zero. A provider-reported cost, including zero,
stays a separate reported value.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from ssgrep.accounting.contracts import Basis, Quantity, digest_of
from ssgrep.accounting.units import USAGE_CATEGORIES, ProviderUsage

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
