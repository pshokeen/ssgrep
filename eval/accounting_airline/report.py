"""Completion report for the airline lab (validation-lab.md sections 5-7).

Principles enforced here, not left to the caller:

* Every gate group is visible on its own line with its own status. There is no overall,
  success, ok, or all-passed aggregate anywhere in any dictionary this module produces.
* Every metric names basis/unit/source/samples/actor scope/window/methodology, or is
  ``unavailable`` with a reason. Floats are refused; quantities are integers or text.
* Production-policy and arithmetic-fixture valuation stay under separate keys and are never
  combined.
* A scripted stand-in run is labeled as such everywhere and can never satisfy the live gate.
* The private report may hold run/session/task/source/receipt ids; the public summary holds
  aggregate statuses and metric counts only.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .checkpoints import (
    BLOCKED,
    FAIL,
    NOT_RUN,
    PASS,
    UNKNOWN,
    Checkpoint,
    CheckpointLog,
    write_private_json,
)

FAMILIES = (
    "fixture_domain",
    "runtime_work",
    "tokens_money",
    "retrieval",
    "persistence_index",
    "attribution",
    "valuation",
    "timing_resources",
    "storage_recovery",
    "export_privacy",
)
BASES = ("observed", "reported", "estimated", "unavailable")

GATE_GROUPS = (
    "seed_integrity",
    "l1",
    "l2",
    "a_native_persistence",
    "restart_retrieval",
    "b_isolation",
    "cross_session_exposure",
    "reuse_evidence",
    "l3",
    "l4_scenarios",
    "accounting_checkpoints",
    "deterministic_replay",
    "scrape",
    "export_privacy",
    "cleanup",
    "valuation_production",
    "valuation_arithmetic",
    "live_session_gate",
)
LIVE_GATE = "live_session_gate"
STANDIN_LABEL = "scripted-standin"
STANDIN_LIVE_REASON = "scripted stand-in never counts toward the live gate"

PRODUCTION_PROFILE = "production-policy"
ARITHMETIC_PROFILE = "airline-lab-arithmetic-v1"
_VALUATION_KEYS = {PRODUCTION_PROFILE: "production", ARITHMETIC_PROFILE: "arithmetic_fixture"}

GATE_STATUSES = (PASS, FAIL, BLOCKED, NOT_RUN, UNKNOWN)
_GATE_PRECEDENCE = (FAIL, BLOCKED, NOT_RUN, UNKNOWN, PASS)
_FORBIDDEN_AGGREGATE_KEYS = frozenset({"overall", "all_passed", "success", "ok"})

REDACTED = "[REDACTED]"


# --------------------------------------------------------------------------- metrics


def _reject_float(name: str, value: object) -> None:
    if isinstance(value, float):
        raise TypeError(f"{name} must not be a float (accounting values are exact ints)")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metric {name} must be a non-empty string")


@dataclass(frozen=True)
class Metric:
    family: str
    name: str
    value: int | str | None
    unit: str
    basis: str
    source: str
    samples: int | None
    actor_scope: str
    window: str
    methodology: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown metric family {self.family!r}")
        if self.basis not in BASES:
            raise ValueError(f"unknown metric basis {self.basis!r}")
        for field_name in ("name", "unit", "source", "actor_scope", "window", "methodology"):
            _require_text(field_name, getattr(self, field_name))
        _reject_float("value", self.value)
        _reject_float("samples", self.samples)
        if isinstance(self.value, bool) or (
            self.value is not None and not isinstance(self.value, int | str)
        ):
            raise TypeError(f"metric value must be int, str or None, got {type(self.value)}")
        if self.samples is not None and (
            isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 0
        ):
            raise TypeError("metric samples must be a non-negative int or None")
        if self.basis == "unavailable":
            if self.value is not None:
                raise ValueError(f"{self.name}: an unavailable metric carries no value")
            _require_text("reason (required when unavailable)", self.reason)
            return
        if self.value is None:
            raise ValueError(f"{self.name}: value None requires basis 'unavailable' and a reason")
        if self.samples is None:
            raise ValueError(f"{self.name}: basis {self.basis!r} requires a sample count")

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.family, self.name, self.actor_scope, self.window, self.methodology)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "basis": self.basis,
            "source": self.source,
            "samples": self.samples,
            "actor_scope": self.actor_scope,
            "window": self.window,
            "methodology": self.methodology,
            "reason": self.reason,
        }


class MetricBook:
    def __init__(self) -> None:
        self._metrics: dict[tuple[str, str, str, str, str], Metric] = {}

    def add(self, metric: Metric) -> None:
        if metric.key in self._metrics:
            raise ValueError(
                f"duplicate metric {metric.family}/{metric.name} for scope "
                f"{metric.actor_scope!r}, window {metric.window!r}, methodology "
                f"{metric.methodology!r}"
            )
        self._metrics[metric.key] = metric

    def __len__(self) -> int:
        return len(self._metrics)

    def family(self, name: str) -> list[Metric]:
        if name not in FAMILIES:
            raise ValueError(f"unknown metric family {name!r}")
        return [m for m in self._metrics.values() if m.family == name]

    def counts(self) -> dict[str, dict[str, int]]:
        """Metric counts by family and basis; every family and basis is listed."""
        out = {fam: dict.fromkeys(BASES, 0) for fam in FAMILIES}
        for m in self._metrics.values():
            out[m.family][m.basis] += 1
        return out

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {fam: [m.as_dict() for m in self.family(fam)] for fam in FAMILIES}

    def require_complete(self, declared: Mapping[str, Sequence[str]]) -> list[str]:
        """Return ``family/name`` for each declared quantity with no metric recorded.

        An ``unavailable`` metric with a reason counts as present: a quantity may be honestly
        missing, but it can never silently vanish.
        """
        present = {(m.family, m.name) for m in self._metrics.values()}
        missing: list[str] = []
        for fam, names in declared.items():
            if fam not in FAMILIES:
                raise ValueError(f"unknown metric family {fam!r}")
            missing.extend(f"{fam}/{n}" for n in names if (fam, n) not in present)
        return missing


# --------------------------------------------------------------------------- gates


@dataclass
class GateEntry:
    name: str
    status: str
    detail: str
    artifact: str | None = None

    def __post_init__(self) -> None:
        if self.status not in GATE_STATUSES:
            raise ValueError(f"unknown gate status {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "artifact": self.artifact,
        }


def _combine_gate(statuses: Iterable[str]) -> str:
    seen = set(statuses)
    if not seen:
        return NOT_RUN
    for status in _GATE_PRECEDENCE:
        if status in seen:
            return status
    raise ValueError(f"unknown gate status in {sorted(seen)}")


def _walk_forbidden_keys(obj: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.lower() in _FORBIDDEN_AGGREGATE_KEYS:
                found.append(here)
            found.extend(_walk_forbidden_keys(value, here))
    elif isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            found.extend(_walk_forbidden_keys(value, f"{path}[{i}]"))
    return found


def _walk_floats(obj: object, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, float):
        found.append(path or "<root>")
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            found.extend(_walk_floats(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list | tuple):
        for i, value in enumerate(obj):
            found.extend(_walk_floats(value, f"{path}[{i}]"))
    return found


@dataclass
class Report:
    run_id: str
    mode: str
    runtime: str
    counts_toward_live_gate: bool
    label: str
    gates: dict[str, list[GateEntry]] = field(init=False)
    metrics: MetricBook = field(init=False)
    checkpoints: CheckpointLog = field(init=False)
    _valuation: dict[str, dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.is_standin and self.counts_toward_live_gate:
            raise ValueError("a scripted stand-in run can never count toward the live gate")
        self.gates = {group: [] for group in GATE_GROUPS}
        self.metrics = MetricBook()
        self.checkpoints = CheckpointLog()
        self._valuation = {}
        if self.is_standin:
            self.gates[LIVE_GATE].append(GateEntry(LIVE_GATE, NOT_RUN, STANDIN_LIVE_REASON))

    @property
    def is_standin(self) -> bool:
        return self.runtime == STANDIN_LABEL

    @property
    def runtime_label(self) -> str:
        return self.label or self.runtime

    def add_gate(self, group: str, entry: GateEntry) -> None:
        if group not in self.gates:
            raise ValueError(f"unknown gate group {group!r}")
        if group == LIVE_GATE and entry.status == PASS and not self.counts_toward_live_gate:
            raise ValueError("this run does not count toward the live gate; it cannot PASS it")
        self.gates[group].append(entry)

    def gate_status(self, group: str) -> str:
        if group not in self.gates:
            raise ValueError(f"unknown gate group {group!r}")
        return _combine_gate(e.status for e in self.gates[group])

    def valuation(self, profile: str, payload: Mapping[str, Any]) -> None:
        if profile not in _VALUATION_KEYS:
            raise ValueError(f"unknown valuation profile {profile!r}")
        label = payload.get("profile")
        if label not in _VALUATION_KEYS:
            raise ValueError(f"valuation payload profile label {label!r} is not a known profile")
        if label != profile:
            raise ValueError(f"valuation payload labeled {label!r} cannot be filed as {profile!r}")
        if profile in self._valuation:
            raise ValueError(f"valuation profile {profile!r} already recorded")
        floats = _walk_floats(payload)
        if floats:
            raise TypeError(f"valuation payload holds floats at {floats}")
        forbidden = _walk_forbidden_keys(payload)
        if forbidden:
            raise ValueError(f"valuation payload holds aggregate-success keys at {forbidden}")
        self._valuation[profile] = dict(payload)

    def _label_fields(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "runtime_label": self.runtime_label,
            "scripted_standin": self.is_standin,
            "mode": self.mode,
            "counts_toward_live_gate": self.counts_toward_live_gate,
        }

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"run_id": self.run_id, **self._label_fields()}
        data["gates"] = {
            group: {"status": self.gate_status(group), "entries": [e.as_dict() for e in entries]}
            for group, entries in self.gates.items()
        }
        data["metrics"] = self.metrics.as_dict()
        data["checkpoints"] = self.checkpoints.as_dict()
        for profile, key in _VALUATION_KEYS.items():
            data[key] = self._valuation.get(profile)
        return data

    def public_summary(self) -> dict[str, Any]:
        """Aggregate statuses and metric counts only; no ids, details, paths or payloads."""
        checkpoints: dict[str, dict[str, str]] = {}
        for cid, item in self.checkpoints.as_dict().items():
            checkpoints[cid] = {
                "domain": item["domain_status"],
                "accounting": item["accounting_status"],
            }
        return {
            **self._label_fields(),
            "gates": {group: self.gate_status(group) for group in self.gates},
            "checkpoints": checkpoints,
            "metric_counts": self.metrics.counts(),
            "valuation_present": {
                key: profile in self._valuation for profile, key in _VALUATION_KEYS.items()
            },
        }

    def write_private(self, path: Path) -> None:
        write_private_json(path, self.as_dict())

    def summary_lines(self) -> list[str]:
        lines = [
            f"mode={self.mode} runtime={self.runtime} "
            f"counts_toward_live_gate={self.counts_toward_live_gate}"
        ]
        lines.extend(f"{group}: {self.gate_status(group)}" for group in self.gates)
        return lines


def gate_table(report: Report) -> str:
    """Every gate group with its own status; nothing is rolled up."""
    header = ("gate", "status", "entries p/f/b/n/u", "note")
    rows: list[tuple[str, ...]] = []
    for group, entries in report.gates.items():
        tally = dict.fromkeys(GATE_STATUSES, 0)
        for e in entries:
            tally[e.status] += 1
        counts = "/".join(str(tally[s]) for s in (PASS, FAIL, BLOCKED, NOT_RUN, UNKNOWN))
        note = next((e.detail for e in entries if e.status != PASS and e.detail), "")
        note = " ".join(note.split())
        if len(note) > 60:
            note = note[:57] + "..."
        rows.append((group, report.gate_status(group), counts, note or "-"))
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = [
        " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)).rstrip(),
        "-+-".join("-" * w for w in widths),
    ]
    lines.extend(
        " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- redaction

_GENERIC_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), REDACTED),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=\-]+"), REDACTED),
    (re.compile(r"PGPASSWORD=\S+"), REDACTED),
    (re.compile(r"(?i)password\s*[=:]\s*\S+"), REDACTED),
    (re.compile(r"(?i)((?:token|key)\W{0,3})[0-9a-f]{32,}"), r"\1" + REDACTED),
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Replace each supplied secret and generic secret shapes with ``[REDACTED]``."""
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    for pattern, replacement in _GENERIC_SECRETS:
        text = pattern.sub(replacement, text)
    return text


def _one_line(text: str) -> str:
    return " ".join(text.split())


def progress_line(
    cp: Checkpoint,
    *,
    elapsed_s: int,
    artifact: str | None,
    next_action: str,
    redactions: Iterable[str] = (),
    ts: str | None = None,
) -> str:
    """One progress line: checkpoint, both statuses, elapsed, artifact, next action.

    Only checkpoint id/name/statuses and the caller-supplied artifact/next fields appear;
    assertion details, expected/actual values and transcript text never do.
    """
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, int):
        raise TypeError("elapsed_s must be an int")
    stamp = ts if ts is not None else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (
        f"[{stamp}] {cp.cid} {cp.name} domain={cp.domain_status} "
        f"accounting={cp.accounting_status} elapsed={elapsed_s}s "
        f"artifact={artifact if artifact else '-'} next={next_action or '-'}"
    )
    return redact(_one_line(line), redactions)


# --------------------------------------------------------------------------- scans


def _check_ids(ids: Iterable[str]) -> list[str]:
    out = list(ids)
    if any(not i for i in out):
        raise ValueError("an empty id would match everything")
    return out


def _digest8(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def assert_no_high_cardinality(openmetrics_text: str, ids: Iterable[str]) -> list[str]:
    """Violations for every run/session/task/source/receipt/query id found in a scrape body.

    Violations name the line and a short digest of the id, never the id itself.
    """
    known = _check_ids(ids)
    violations: list[str] = []
    for lineno, line in enumerate(openmetrics_text.splitlines(), start=1):
        for ident in known:
            start = line.find(ident)
            while start >= 0:
                open_brace = line.rfind("{", 0, start)
                close_brace = line.rfind("}", 0, start)
                in_label = open_brace > close_brace
                if line.startswith("#"):
                    where = "metadata"
                elif in_label:
                    where = "label"
                else:
                    where = "sample"
                violations.append(
                    f"line {lineno}: high-cardinality id sha256:{_digest8(ident)} in {where}"
                )
                start = line.find(ident, start + len(ident))
    return violations


def canary_scan(texts: Mapping[str, str], canaries: Iterable[str]) -> list[str]:
    """``name: canary found`` for each text that contains any canary; canaries are not echoed."""
    needles = _check_ids(canaries)
    return [
        f"{name}: canary found" for name, text in texts.items() if any(c in text for c in needles)
    ]
