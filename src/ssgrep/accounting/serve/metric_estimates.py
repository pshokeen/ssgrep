"""Honest window gauges: compatible methods, known subtotals and finite bounds.

Complete nets, known subtotals and interval endpoints are separate series. A
missing endpoint has no sample. Units come from the full compatibility identity,
never a mutable global valuation hash; no source/task/employee IDs become labels.
"""

from collections.abc import Callable, Mapping
from typing import Any

from ssgrep.accounting.ledger.ledger_contracts import WINDOWS
from ssgrep.accounting.serve.metrics_render import Sample

METHODS = frozenset(
    {
        "chars_div4_v1",
        "capped_gap_v1",
        "transcript_intervals_v1",
        "monotonic_v1",
        "provider_usage_v1",
    }
)
SCENARIOS = ("conservative", "central", "optimistic")
BASE = ("scenario", "window", "method")
ADDITIONAL_FAMILIES = (
    ("ssgrep_estimated_credit_work_seconds", BASE),
    ("ssgrep_debit_work_seconds", ("window", "method")),
    *(
        (f"ssgrep_known_{part}_{suffix}", BASE if part != "debit" else ("window", "method"))
        for suffix in ("tokens", "work_seconds")
        for part in ("credit", "debit", "net")
    ),
    *(
        (f"ssgrep_estimate_bound_{suffix}", (*BASE, "quantity", "bound"))
        for suffix in ("tokens", "work_seconds")
    ),
    ("ssgrep_estimate_missing_records", (*BASE, "quantity")),
    ("ssgrep_measured_service_latency_milliseconds", ("window", "method")),
    ("ssgrep_known_service_latency_milliseconds", ("window", "method")),
    ("ssgrep_measured_service_missing_records", ("window",)),
)


def integer(value: object) -> int | None:
    return value if type(value) is int else None


def _value(raw: object, unit: str) -> int | None:
    if not isinstance(raw, dict) or raw.get("unit") != unit:
        return None
    if raw.get("basis") not in ("observed", "estimated", "reported"):
        return None
    return integer(raw.get("value"))


def _identity(block: object) -> tuple[str, str, str | None, str | None] | None:
    if not isinstance(block, dict):
        return None
    identities = []
    for scenario in SCENARIOS:
        row = block.get(scenario)
        unit = row.get("unit") if isinstance(row, dict) else None
        if not isinstance(unit, dict):
            return None
        name, method = unit.get("unit"), unit.get("method")
        if name not in ("context_tokens", "work_seconds") or method not in METHODS:
            return None
        if unit.get("tokenizer") is not None or unit.get("currency") is not None:
            return None  # no silently collapsed tokenizer/currency identities
        identities.append((name, method, None, None))
    return identities[0] if len(set(identities)) == 1 else None


def _unit_samples(block, identity, window, sample):
    name, method, _, _ = identity
    suffix = "tokens" if name == "context_tokens" else "work_seconds"
    out: list[Sample] = []
    missing = 0
    debit_values = [_value(block[s].get("debit"), name) for s in SCENARIOS]
    common_debit = len(set(debit_values)) == 1
    for scenario in SCENARIOS:
        row = block[scenario]
        labels = {"scenario": scenario, "window": window, "method": method}
        compatible = all(
            isinstance(row.get(part), dict) and row[part].get("unit") == name
            for part in ("credit", "debit", "net")
        )
        credit, debit = _value(row.get("credit"), name), _value(row.get("debit"), name)
        subtotal = integer(row.get("known_net")) if compatible else None
        if credit is not None and debit is not None and subtotal != credit - debit:
            subtotal = None
        no_missing = all(
            isinstance(row.get(f"{part}_missing"), list) and not row[f"{part}_missing"]
            for part in ("credit", "debit")
        )
        for part in ("credit", "net"):
            value = _value(row.get(part), name)
            known = subtotal if part == "net" else value
            if known is not None:
                out.append(sample(f"ssgrep_known_{part}_{suffix}", labels, known))
            complete = (
                row.get("complete") is True
                and common_debit
                and compatible
                and no_missing
                and credit is not None
                and debit is not None
                and value == credit - debit
                if part == "net"
                else isinstance(row.get("credit_missing"), list) and not row["credit_missing"]
            )
            if value is not None and complete:
                out.append(sample(f"ssgrep_estimated_{part}_{suffix}", labels, value))
            else:
                missing += 1
        bounds = row.get("bounds")
        if isinstance(bounds, dict):
            for part in ("credit", "debit", "net"):
                for bound in ("lower", "upper"):
                    value = _value(bounds.get(f"{part}_{bound}"), name)
                    if value is not None:
                        out.append(
                            sample(
                                f"ssgrep_estimate_bound_{suffix}",
                                {**labels, "quantity": part, "bound": bound},
                                value,
                            )
                        )
        for part in ("credit", "debit"):
            gaps = row.get(f"{part}_missing")
            if isinstance(gaps, list):
                out.append(
                    sample(
                        "ssgrep_estimate_missing_records", {**labels, "quantity": part}, len(gaps)
                    )
                )
    debit_labels = {"window": window, "method": method}
    value = debit_values[0] if common_debit else None
    if value is not None:
        out.append(sample(f"ssgrep_known_debit_{suffix}", debit_labels, value))
    complete_debit = all(
        isinstance(block[s].get("debit_missing"), list) and not block[s]["debit_missing"]
        for s in SCENARIOS
    )
    if value is not None and complete_debit:
        out.append(sample(f"ssgrep_debit_{suffix}", debit_labels, value))
    else:
        missing += 1
    return out, missing


def _latency(block, window, sample):
    raw = block.get("observations")
    if not isinstance(raw, dict):
        return []
    labels = {"window": window, "method": "monotonic_v1"}
    out = []
    for field, family in (
        ("measured_service_ms", "ssgrep_measured_service_latency_milliseconds"),
        ("measured_service_ms_known", "ssgrep_known_service_latency_milliseconds"),
    ):
        quantity = raw.get(field)
        value = _value(quantity, "work_ms")
        if (
            value is not None
            and value >= 0
            and quantity.get("basis") == "observed"
            and quantity.get("method") == "monotonic_v1"
        ):
            out.append(sample(family, labels, value))
    missing = _value(raw.get("measured_service_missing"), "count")
    if missing is not None and missing >= 0:
        out.append(sample("ssgrep_measured_service_missing_records", {"window": window}, missing))
    return out


def estimate_samples(
    windows: Mapping[str, Any], sample: Callable[..., Sample]
) -> tuple[list[Sample], int, int]:
    out: list[Sample] = []
    available = missing = 0
    for window in WINDOWS:
        block = windows.get(window)
        if not isinstance(block, dict):
            continue
        estimates = block.get("estimates")
        groups: dict[tuple, object] = {}
        conflicted = set()
        for entry in estimates.values() if isinstance(estimates, dict) else ():
            identity = _identity(entry)
            if identity is None:
                missing += 7
                continue
            if identity in groups and groups[identity] != entry:
                conflicted.add(identity)
            else:
                groups[identity] = entry
        for identity, entry in sorted(groups.items()):
            if identity in conflicted:
                missing += 7
                continue
            found, absent = _unit_samples(entry, identity, window, sample)
            out.extend(found)
            available += sum(
                s.family.startswith(("ssgrep_estimated_", "ssgrep_debit_")) for s in found
            )
            missing += absent
        out.extend(_latency(block, window, sample))
    return out, available, missing
