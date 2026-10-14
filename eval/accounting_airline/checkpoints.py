"""Checkpoint manifests and status logic for the airline lab (validation-lab.md section 5).

A checkpoint carries two independently reported statuses, domain and accounting. Neither is
folded into the other and an empty assertion list is UNKNOWN, never PASS. Statuses combine with
precedence FAIL > BLOCKED > UNKNOWN > PASS. This module only holds manifests and status logic;
the lab runner fills them from real observations.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
UNKNOWN = "UNKNOWN"
NOT_RUN = "NOT_RUN"
STATUSES = (PASS, FAIL, BLOCKED, UNKNOWN)

_PRECEDENCE = (FAIL, BLOCKED, UNKNOWN, PASS)


def combine(statuses: Iterable[str]) -> str:
    """Worst status wins (FAIL > BLOCKED > UNKNOWN > PASS); nothing to combine is UNKNOWN."""
    seen = set()
    for status in statuses:
        if status not in STATUSES:
            raise ValueError(f"unknown checkpoint status {status!r}")
        seen.add(status)
    if not seen:
        return UNKNOWN
    for status in _PRECEDENCE:
        if status in seen:
            return status
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class Assertion:
    """One checked statement.

    ``intentional_negative`` marks an assertion that PROVES an expected failure: a deliberately
    broken variant was fed to the checker. Its status is PASS exactly when the failure was
    detected and FAIL when the broken variant slipped through; the label keeps an expected
    error from being mistaken for a broken lab.
    """

    name: str
    status: str
    expected: object
    actual: object
    method: str
    detail: str = ""
    intentional_negative: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("assertion needs a name")
        if self.status not in STATUSES:
            raise ValueError(f"unknown assertion status {self.status!r}")

    @classmethod
    def negative(
        cls,
        name: str,
        *,
        detected: bool,
        expected: object,
        actual: object,
        method: str,
        detail: str = "",
    ) -> Assertion:
        """Build an intentional-negative assertion from whether the broken variant was caught."""
        return cls(
            name=name,
            status=PASS if detected else FAIL,
            expected=expected,
            actual=actual,
            method=method,
            detail=detail,
            intentional_negative=True,
        )

    @property
    def label(self) -> str:
        if not self.intentional_negative:
            return ""
        if self.status == PASS:
            return "INTENTIONAL NEGATIVE: expected failure detected"
        return f"INTENTIONAL NEGATIVE: expected failure NOT detected ({self.status})"

    def line(self) -> str:
        label = f" [{self.label}]" if self.label else ""
        return f"{self.status} {self.name}{label}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "method": self.method,
            "detail": self.detail,
            "intentional_negative": self.intentional_negative,
            "label": self.label,
        }


@dataclass(frozen=True)
class CheckpointDef:
    cid: str
    name: str
    domain: str
    accounting: str


# validation-lab.md section 5 table.
DEFINITIONS: dict[str, CheckpointDef] = {
    d.cid: d
    for d in (
        CheckpointDef(
            "C0",
            "seed/setup",
            "12 table counts, 82 rows, FK/check validity, logical digest, frozen clock",
            "Local profile enabled, export off, empty/nonexistent ledger state honest; "
            "initial counter/epoch/watermarks recorded",
        ),
        CheckpointDef(
            "C1",
            "A L1 completes",
            "All financial/seat result rows match the independent oracle",
            "A/L1 task, calls, SQL errors/repairs, native usage, known output units and "
            "missing coverage; no reused-source credit fabricated",
        ),
        CheckpointDef(
            "C2",
            "A L2 completes",
            "Affected travellers, 35/45-minute boundaries, DST instants correct",
            "Distinct A/L2 task in same root; candidate witnesses and exclusions retain "
            "actual event IDs",
        ),
        CheckpointDef(
            "C3",
            "A persisted/indexed/restarted",
            "Both actual task sources resolve and verified facts are visible",
            "Source digests/generation, verifier actor separated, "
            "capture→parse→index→search visibility latency and coverage recorded",
        ),
        CheckpointDef(
            "C4",
            "B starts",
            "Same cwd, fresh identity/context, seed unchanged",
            "New consumer root/tasks; no inherited A usage or duplicate historical import; "
            "scope/permission assertions",
        ),
        CheckpointDef(
            "C5",
            "each B search/show",
            "Correct native A source returned/opened, or explicit miss/error",
            "Dispatch/journal/result matching; every real lookup debit, rank/truncation/bytes, "
            "zero adoption before later action; repeat reads cannot add credits",
        ),
        CheckpointDef(
            "C6",
            "B L3 validation",
            "Candidate set and joint priority assignment correct",
            "Exposure→later action evidence, novelty/rarity decisions, unioned atom credit, "
            "equal scenario debit, compatible units, negative/unknowns intact",
        ),
        CheckpointDef(
            "C7",
            "each L4 attempt",
            "Commit/reject/rollback/replay results independently checked",
            "Attempts and failures cost something; DB idempotency doesn't erase tool/lookup "
            "debits; task/ref reuse cannot multiply credit",
        ),
        CheckpointDef(
            "C8",
            "replay/recollection",
            "Business state unchanged by accounting work",
            "Full versus incremental canonical rows/digests agree; duplicate import/collect "
            "does not add usage; restart preserves counters/epoch",
        ),
        CheckpointDef(
            "C9",
            "scrape/export",
            "No SQL or source mutations",
            "Latest collected vs older acknowledged generation, correct OpenMetrics types, "
            "exact sanitized payload, per-signal partial/retry/ack semantics",
        ),
        CheckpointDef(
            "C10",
            "closure",
            "B source persisted; owned resources stopped/retained according to policy",
            "Recompute limits, retention/pauses/deletion scopes, final manifest, per-gate "
            "outcomes and privacy-safe report",
        ),
    )
}


def _cid_index(cid: str) -> int:
    if cid not in DEFINITIONS:
        raise ValueError(f"unknown checkpoint id {cid!r}; expected C0..C10")
    return int(cid[1:])


def _check_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


@dataclass
class Checkpoint:
    cid: str
    name: str
    domain: list[Assertion] = field(default_factory=list)
    accounting: list[Assertion] = field(default_factory=list)
    input_prefixes: dict[str, str] = field(default_factory=dict)
    events_by_actor: dict[str, int] = field(default_factory=dict)
    runtime_ids: dict[str, str] = field(default_factory=dict)
    task_ids: list[str] = field(default_factory=list)
    snapshot_before: str | None = None
    snapshot_after: str | None = None
    expected_delta: dict[str, Any] = field(default_factory=dict)
    actual_delta: dict[str, Any] = field(default_factory=dict)
    comparison_method: str = ""
    coverage_missing: list[str] = field(default_factory=list)
    artifact_path: str | None = None
    next_action: str = ""
    started_utc: str = ""
    elapsed_ms: int = 0
    methodology: str | None = None

    def __post_init__(self) -> None:
        _cid_index(self.cid)
        _check_int("elapsed_ms", self.elapsed_ms)
        for actor, count in self.events_by_actor.items():
            _check_int(f"events_by_actor[{actor!r}]", count)

    @property
    def domain_status(self) -> str:
        return combine(a.status for a in self.domain)

    @property
    def accounting_status(self) -> str:
        return combine(a.status for a in self.accounting)

    @property
    def status(self) -> str:
        return combine((self.domain_status, self.accounting_status))

    def counts(self) -> dict[str, int]:
        """Assertion counts over both lists, keyed PASS/FAIL/BLOCKED/UNKNOWN."""
        counts = dict.fromkeys(STATUSES, 0)
        for a in (*self.domain, *self.accounting):
            counts[a.status] += 1
        return counts

    def assertion_lines(self) -> list[str]:
        return [f"domain: {a.line()}" for a in self.domain] + [
            f"accounting: {a.line()}" for a in self.accounting
        ]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["domain"] = [a.as_dict() for a in self.domain]
        data["accounting"] = [a.as_dict() for a in self.accounting]
        data["domain_status"] = self.domain_status
        data["accounting_status"] = self.accounting_status
        data["status"] = self.status
        data["definition"] = _definition_dict(self.cid)
        return data


def _definition_dict(cid: str) -> dict[str, str]:
    d = DEFINITIONS[cid]
    return {"name": d.name, "domain": d.domain, "accounting": d.accounting}


def _elapsed_text(ms: int) -> str:
    return f"{ms // 1000}.{ms % 1000:03d}s"


def write_private_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON readable only by the owner (0600), also when replacing a file."""
    text = json.dumps(data, indent=2, sort_keys=True, default=repr)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1
            fh.write(text)
            fh.write("\n")
    finally:
        if fd >= 0:
            os.close(fd)


class CheckpointLog:
    def __init__(self) -> None:
        self._items: dict[str, Checkpoint] = {}

    def add(self, cp: Checkpoint) -> None:
        _cid_index(cp.cid)
        if cp.cid in self._items:
            raise ValueError(
                f"checkpoint {cp.cid} already recorded; attempts are never overwritten"
            )
        self._items[cp.cid] = cp

    def get(self, cid: str) -> Checkpoint:
        return self._items[cid]

    def __contains__(self, cid: object) -> bool:
        return cid in self._items

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for cid in sorted(DEFINITIONS, key=_cid_index):
            cp = self._items.get(cid)
            if cp is not None:
                out[cid] = cp.as_dict()
                continue
            out[cid] = {
                "cid": cid,
                "name": DEFINITIONS[cid].name,
                "status": NOT_RUN,
                "domain_status": NOT_RUN,
                "accounting_status": NOT_RUN,
                "definition": _definition_dict(cid),
            }
        return out

    def write(self, path: Path) -> None:
        write_private_json(path, self.as_dict())

    def table(self) -> str:
        header = ("CP", "name", "domain", "accounting", "assertions p/f/b/u", "elapsed", "next")
        rows: list[tuple[str, ...]] = []
        for cid in sorted(DEFINITIONS, key=_cid_index):
            cp = self._items.get(cid)
            if cp is None:
                rows.append((cid, DEFINITIONS[cid].name, NOT_RUN, NOT_RUN, "-", "-", "-"))
                continue
            c = cp.counts()
            rows.append(
                (
                    cid,
                    cp.name,
                    cp.domain_status,
                    cp.accounting_status,
                    f"{c[PASS]}/{c[FAIL]}/{c[BLOCKED]}/{c[UNKNOWN]}",
                    _elapsed_text(cp.elapsed_ms),
                    " ".join(cp.next_action.split()) or "-",
                )
            )
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
        lines = [
            " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)).rstrip(),
            "-+-".join("-" * w for w in widths),
        ]
        lines.extend(
            " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows
        )
        return "\n".join(lines)


def input_prefix(path: Path, complete_bytes: int | None = None) -> str:
    """``sha256hex@bytes`` over a prefix of the file.

    By default the prefix is the newline-complete prefix (everything up to and including the
    last ``\\n``), so a partial tail never enters the digest. An explicit ``complete_bytes``
    must not exceed the file size.
    """
    if complete_bytes is not None and complete_bytes < 0:
        raise ValueError("complete_bytes must be >= 0")
    digest = hashlib.sha256()
    if complete_bytes is not None:
        remaining = complete_bytes
        with open(path, "rb") as fh:
            while remaining:
                chunk = fh.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError(f"{path} is shorter than the requested {complete_bytes} bytes")
                digest.update(chunk)
                remaining -= len(chunk)
        return f"{digest.hexdigest()}@{complete_bytes}"
    complete = 0
    seen = 0
    complete_digest = digest.copy()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            cut = chunk.rfind(b"\n")
            if cut >= 0:
                digest.update(chunk[: cut + 1])
                complete_digest = digest.copy()
                complete = seen + cut + 1
                digest.update(chunk[cut + 1 :])
            else:
                digest.update(chunk)
            seen += len(chunk)
    return f"{complete_digest.hexdigest()}@{complete}"
