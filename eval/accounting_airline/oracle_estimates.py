"""Independent v2 known subtotals, interval arithmetic and sound net endpoints."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Estimate:
    scenario: str
    credit: int | None
    debit: int | None
    net: int | None
    known_net: int | None
    missing: tuple[str, ...]
    bounds: Mapping[str, int | None]
    complete: bool = True
    bound_assumptions: tuple[str, ...] = (
        "outward_round_lower_floor_upper_ceil",
        "point_net_uses_floored_component_totals",
    )
    raw_credit: int | None = None
    raw_debit: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "credit": self.credit,
            "debit": self.debit,
            "net": self.net,
            "known_net": self.known_net,
            "missing": list(self.missing),
            "bounds": dict(self.bounds),
            "complete": self.complete,
            "bound_assumptions": list(self.bound_assumptions),
            "raw_credit": self.raw_credit,
            "raw_debit": self.raw_debit,
        }


def bounded_estimate(
    scenario: str,
    credit_raw: int,
    credit_missing: bool,
    debit_raw: int,
    debit_min: int,
    debit_max: int | None,
    divisor: int,
    missing: tuple[str, ...],
) -> Estimate:
    # Point estimates intentionally use whole-unit component floors. Bounds instead
    # enclose the unquantized raw quantities: lower floors, upper ceilings.
    cmin, dmin = credit_raw // divisor, debit_min // divisor
    cmax = None if credit_missing else (credit_raw + divisor - 1) // divisor
    dmax = None if debit_max is None else (debit_max + divisor - 1) // divisor
    known = cmin - debit_raw // divisor
    complete = not missing and not credit_missing and debit_max is not None
    return Estimate(
        scenario,
        cmin,
        debit_raw // divisor,
        known if complete else None,
        known,
        missing,
        {
            "credit_lower": cmin,
            "credit_upper": cmax,
            "debit_lower": dmin,
            "debit_upper": dmax,
            "net_lower": None if dmax is None else cmin - dmax,
            "net_upper": None if cmax is None else cmax - dmin,
        },
        complete,
        raw_credit=credit_raw,
        raw_debit=debit_raw,
    )


def interval_work_ms(per_actor: Mapping[str, Sequence[tuple[int, int]]]) -> int:
    """Cap each evidenced positive gap, then union per actor; actors add, not elapsed time."""
    total = 0
    for intervals in per_actor.values():
        end = None
        for start, stop in sorted((a, min(b, a + 300000)) for a, b in intervals if b >= a):
            total += max(0, stop - max(start, end if end is not None else start))
            end = max(stop, end if end is not None else stop)
    return total


def lookup_boundary(record) -> bool:
    """Recognize actual native command words, not quoted narrative mentions."""
    if record.kind != "tool_call":
        return False
    try:
        lexer = shlex.shlex(record.text, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return True  # Unparseable command cannot prove a comparable source gap.
    return any(
        word.rsplit("/", 1)[-1] == "ssgrep"
        and words[i + 1] in ("search", "show")
        and (i == 0 or words[i - 1] in (";", "&&", "||", "|", "(", "do", "exec", "command"))
        for i, word in enumerate(words[:-1])
    )


def source_work(scan, by_task: Mapping[str, Sequence[str]]) -> tuple[int, bool]:
    """Use source-content adjacency; billing/usage metadata cannot close work gaps."""
    total, missing = 0, False
    locations = {
        f"{key}:{r.record_id}": (key, i)
        for key, records in scan.records.items()
        for i, r in enumerate(records)
    }
    for task, atoms in by_task.items():
        intervals: dict[str, list[tuple[int, int]]] = {}
        exposures = [
            scan.records[i.session][i.result].ts_ms
            for i in scan.invocations
            if i.task == task and i.result is not None
        ]
        cutoff = min((stamp for stamp in exposures if stamp is not None), default=None)
        for atom in set(atoms):
            key, index = locations[atom]
            records = scan.records[key]
            following_index = next(
                (
                    j
                    for j in range(index + 1, len(records))
                    if records[j].kind in ("user_turn", "tool_call", "tool_result")
                ),
                None,
            )
            if following_index is None:
                missing = True
                continue
            following = records[following_index]
            if (
                following.kind == "user_turn"
                or lookup_boundary(following)
                or (
                    following.kind == "tool_call"
                    and any(
                        i.session == key and i.index == following_index and i.is_lookup
                        for i in scan.invocations
                    )
                )
            ):
                missing = True
                continue
            start, stop = records[index].ts_ms, following.ts_ms
            if (
                start is None
                or stop is None
                or stop < start
                or (cutoff is not None and stop > cutoff)
            ):
                missing = True
                continue
            intervals.setdefault(key, []).append((start, stop))
        total += interval_work_ms(intervals)
    return total, missing


def observed_timing(scan) -> tuple[int, int, bool]:
    """Independent supervisor service latency is NOT the child's journal service latency.

    Native/controller inputs measure the full dispatch envelope. It is retained as a
    distinct observation; exact journal-vs-observer equality would be a false oracle.
    """
    calls = {d.call_id: d for inv in scan.invocations if inv.is_lookup for d in inv.dispatches}
    calls.update(
        {d.call_id: d for d in scan.unlinked if len(d.argv) > 1 and d.argv[1] in ("search", "show")}
    )
    valid = [
        d.duration_ns for d in calls.values() if type(d.duration_ns) is int and d.duration_ns >= 0
    ]
    return (
        sum(valid) // 1000000,
        sum(min(ns, 300000000000) for ns in valid) // 1000000,
        len(valid) == len(calls),
    )


def compare_estimates(
    result, savings_json: Mapping[str, Any], find, same, *, estimates=None, unit="context_tokens"
) -> list[str]:
    out: list[str] = []
    tokens = (find(savings_json, "estimates") or {}).get(unit)
    for scenario, est in (result.estimates if estimates is None else estimates).items():
        row = tokens.get(scenario) if isinstance(tokens, Mapping) else None
        if not isinstance(row, Mapping):
            out.append(f"{unit}.{scenario}: missing in snapshot")
            continue
        label = f"{unit}.{scenario}"
        for field in ("credit", "debit", "net"):
            same(f"{label}.{field}", getattr(est, field), row.get(field), out)
            quantity = row.get(field)
            if (
                isinstance(quantity, Mapping)
                and quantity.get("value") is None
                and not quantity.get("reason")
            ):
                out.append(f"{label}.{field}: unavailable without reason")
            if isinstance(quantity, Mapping) and quantity.get("unit") != unit:
                out.append(f"{label}.{field}: incomparable quantity unit")
        if type(row.get("known_net")) is not int or row["known_net"] != est.known_net:
            out.append(
                f"{label}.known_net: oracle {est.known_net!r} != snapshot {row.get('known_net')!r}"
            )
        if row.get("complete") is not est.complete:
            out.append(
                f"{label}.complete: oracle {est.complete!r} != snapshot {row.get('complete')!r}"
            )
        for field, expected in est.bounds.items():
            same(f"{label}.bounds.{field}", expected, (row.get("bounds") or {}).get(field), out)
            quantity = (row.get("bounds") or {}).get(field)
            if (
                isinstance(quantity, Mapping)
                and quantity.get("value") is None
                and not quantity.get("reason")
            ):
                out.append(f"{label}.bounds.{field}: unavailable without reason")
            if isinstance(quantity, Mapping) and quantity.get("unit") != unit:
                out.append(f"{label}.bounds.{field}: incomparable quantity unit")
        assumptions = (row.get("bounds") or {}).get("assumptions")
        if (
            not isinstance(assumptions, (list, tuple))
            or any(not isinstance(item, str) for item in assumptions)
            or not set(est.bound_assumptions).issubset(assumptions)
        ):
            out.append(f"{label}.bounds.assumptions: missing outward/point precision methodology")
        for name, expected in (
            ("credit_missing", any(x.startswith("credit:") for x in est.missing)),
            ("debit_missing", any(x.startswith("debit:") for x in est.missing)),
        ):
            actual = row.get(name)
            if not isinstance(actual, (list, tuple)) or bool(actual) != expected:
                out.append(f"{label}.{name}: missing or incompatible coverage")
        spec = row.get("unit")
        expected_method = "capped_gap_v1" if unit == "work_seconds" else "chars_div4_v1"
        if (
            not isinstance(spec, Mapping)
            or spec.get("unit") != unit
            or spec.get("method") != expected_method
        ):
            out.append(f"{label}.unit: incompatible or missing comparison method")
        provenance = {
            tuple(pair)
            for pair in row.get("provenance", ())
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        }
        if unit == "work_seconds" and result.timing.get("compare_snapshot") and result.debits:
            if ("lookup_work_mapping", "journal_active_proxy_ns_v2") not in provenance:
                out.append(f"{label}.provenance: missing versioned active proxy mapping")
        for verdict in result.verdicts:
            # Frequency governs supported adoption, not a leaf containing only
            # measured lookup debits. Unreferenced visible strings leave no
            # candidate contribution whose frozen authority could be compared.
            if verdict.reuse != "action_reference":
                continue
            source = dict(verdict.provenance)
            if not source or source.get("unavailable_reason"):
                continue
            for raw, field in (
                ("generation", "frequency_generation"),
                ("corpus_digest", "frequency_corpus_digest"),
                ("normalization", "frequency_normalization"),
                ("document_unit", "frequency_document_unit"),
                ("profile_scope", "frequency_profile_scope"),
                ("mapping", "frequency_mapping"),
            ):
                if (field, source[raw]) not in provenance:
                    out.append(f"{label}.provenance: missing frozen {field}")
    return out
