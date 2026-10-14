"""Representation-independent grading of genuine SQL numbers and rendered source text."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

INTEGER_COLUMNS = frozenset(
    {
        "settled_minor",
        "occupied",
        "live_holds",
        "raw_free",
        "sellable_free",
        "gap_minutes",
        "minimum_minutes",
        "utc_offset_hours",
    }
)


def normalize_tables(tables: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    normalized = {}
    for tag, rows in tables.items():
        converted = []
        for row in rows:
            values = dict(row)
            for key in INTEGER_COLUMNS & row.keys():
                try:
                    value = Decimal(row[key])
                except InvalidOperation as exc:
                    raise ValueError(f"{key}: invalid SQL numeric {row[key]!r}") from exc
                if not value.is_finite() or value != value.to_integral_value():
                    raise ValueError(f"{key}: non-integral SQL numeric {row[key]!r}")
                values[key] = str(int(value))
            converted.append(values)
        normalized[tag] = converted
    return normalized


def rendered_facts(text: str) -> str:
    """Ignore presentational Markdown cell/emphasis separators, never alter fact values."""
    return " ".join(re.sub(r"[|*/`]", " ", text).split())


def missing_facts(text: str, expected: list[str]) -> list[str]:
    normalized = rendered_facts(text)
    return [
        fact
        for fact in expected
        if re.search(
            r"(?<![\w.-])" + re.escape(rendered_facts(fact)) + r"(?=$|\s|[.,;:!?](?:\s|$))",
            normalized,
        )
        is None
    ]
