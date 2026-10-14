"""Gate outcomes shared by the airline lab lanes.

A gate is PASS, FAIL, BLOCKED (a prerequisite is missing), or NOT_RUN. A skipped
prerequisite is never reported as PASS and never swallowed into an exception message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"
STATUSES = (PASS, FAIL, BLOCKED, NOT_RUN)


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown gate status {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def from_violations(name: str, violations: list[str], evidence: dict[str, Any]) -> Gate:
    """PASS iff the oracle found no violation; FAIL lists every one."""
    if violations:
        return Gate(name, FAIL, "; ".join(violations), evidence)
    return Gate(name, PASS, "", evidence)


def blocked(name: str, reason: str) -> Gate:
    return Gate(name, BLOCKED, reason)


def not_run(name: str, reason: str) -> Gate:
    return Gate(name, NOT_RUN, reason)
