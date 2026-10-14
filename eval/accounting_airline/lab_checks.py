"""Small builders that turn grader output into checkpoint assertions and gate entries.

Nothing here decides correctness: each helper receives the violation lists produced by an
independent oracle (domain oracle, accounting oracle, ownership checks) and records them with
the exact expected/actual values and the comparison method used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .checkpoints import BLOCKED, FAIL, PASS, UNKNOWN, Assertion
from .report import GateEntry


def check(
    name: str,
    problems: Sequence[str],
    *,
    expected: object,
    actual: object,
    method: str,
) -> Assertion:
    """PASS iff the independent checker found no violation; FAIL lists the first few."""
    if problems:
        return Assertion(name, FAIL, expected, actual, method, "; ".join(list(problems)[:6]))
    return Assertion(name, PASS, expected, actual, method)


def blocked(name: str, reason: str, *, method: str, expected: object = None) -> Assertion:
    return Assertion(name, BLOCKED, expected, None, method, reason)


def unknown(name: str, reason: str, *, method: str, expected: object = None) -> Assertion:
    return Assertion(name, UNKNOWN, expected, None, method, reason)


def equal(
    name: str, expected: object, actual: object, *, method: str, detail: str = ""
) -> Assertion:
    status = PASS if expected == actual else FAIL
    return Assertion(name, status, expected, actual, method, "" if status == PASS else detail)


def gate_status_of(assertions: Sequence[Assertion]) -> str:
    statuses = {a.status for a in assertions}
    for wanted in (FAIL, BLOCKED, UNKNOWN):
        if wanted in statuses:
            return wanted
    return PASS if statuses else UNKNOWN


def entry_from(assertion: Assertion, artifact: str | None = None) -> GateEntry:
    status = assertion.status
    mapped = status if status in (PASS, FAIL, BLOCKED, UNKNOWN) else UNKNOWN
    return GateEntry(assertion.name, mapped, assertion.detail, artifact)


def entries(assertions: Sequence[Assertion], artifact: str | None = None) -> list[GateEntry]:
    return [entry_from(a, artifact) for a in assertions]


def estimate_values(collected_estimates: Mapping[str, Any]) -> dict[str, Any]:
    """The product's context-token credit/debit/net per scenario as plain integers/None."""
    out: dict[str, Any] = {}
    tokens = collected_estimates.get("context_tokens") or {}
    for scenario in ("conservative", "central", "optimistic"):
        row = tokens.get(scenario) or {}
        out[scenario] = {
            key: (row.get(key) or {}).get("value") for key in ("credit", "debit", "net")
        }
        out[scenario]["reason"] = {
            key: (row.get(key) or {}).get("reason") for key in ("credit", "net")
        }
    return out
