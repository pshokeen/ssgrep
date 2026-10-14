"""Read the product journal bytes ONLY to reconcile them with the independent observer.

The observer never consumes these files to produce evidence; this module is the cross-check
(dispatch observed by the verifier vs execution recorded by the product). Reading is
side-effect free: files are opened read-only and never created or repaired.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .observer import Frozen, ObservedCall

PROFILE = "personal-local"


@dataclass(frozen=True)
class JournalView:
    segments: int
    bytes: int
    digest: str
    records: int
    malformed: int
    completed: Mapping[str, int]  # operation -> completed events
    outcomes: Mapping[str, int]  # "<operation>.<outcome>" -> completed events
    started: int
    ids: tuple[str, ...]


def journal_dir(data_dir: Path) -> Path:
    return data_dir / "accounting" / PROFILE / "journal"


def read_journal(data_dir: Path) -> JournalView:
    directory = journal_dir(data_dir)
    files = sorted(directory.glob("seg-*.jsonl")) if directory.is_dir() else []
    digest = hashlib.sha256()
    completed: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    started = records = malformed = size = 0
    ids: list[str] = []
    for path in files:
        data = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + data)
        size += len(data)
        for line in data.split(b"\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            records += 1
            phase = event.get("phase")
            if phase == "started":
                started += 1
            if phase == "completed":
                completed[str(event.get("operation"))] += 1
                outcomes[f"{event.get('operation')}.{event.get('outcome')}"] += 1
                ids.append(str(event.get("execution_id")))
    return JournalView(
        len(files), size, digest.hexdigest(), records, malformed, dict(completed),
        dict(outcomes), started, tuple(ids),
    )  # fmt: skip


def _parsed(call: ObservedCall) -> dict[str, Any] | None:
    for line in reversed(call.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


def expected_outcome(call: ObservedCall) -> str | None:
    """The journal outcome an OBSERVED ssgrep call implies; None = never entered the API.

    Derived from what the verifier saw (exit status + raw output), not from the product:
    exit 2 is argument validation before the API (not an execution); an ``unknown_ref`` show
    and a no-result search are ``empty``; any other failure is ``error``; success is ``ok``.
    """
    if call.state != "verified" or len(call.argv) < 2 or call.tool != "ssgrep":
        return None
    if call.exit_code == 2:
        return None
    body = _parsed(call)
    if call.exit_code in (0, None):
        data = (body or {}).get("data", body or {})
        results = data.get("results") if isinstance(data, dict) else None
        if isinstance(results, list) and not results:
            return "empty"
        return "ok"
    text = call.stdout
    if call.exit_code == 3 or "unknown_ref" in text:
        return "empty"
    return "error"


def observed_expectation(frozen: Frozen) -> dict[str, Any]:
    """Per-operation and per-outcome counts the journal SHOULD hold for what was observed."""
    completed: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    not_journaled = Counter()
    for call in frozen.calls:
        if (
            call.tool != "ssgrep"
            or len(call.argv) < 2
            or call.argv[1] not in ("search", "show", "note")
        ):
            continue
        outcome = expected_outcome(call)
        if outcome is None:
            not_journaled["pre_api" if call.state == "verified" else call.state] += 1
            continue
        completed[call.argv[1]] += 1
        outcomes[f"{call.argv[1]}.{outcome}"] += 1
    return {"completed": dict(completed), "outcomes": dict(outcomes),
            "not_journaled": dict(not_journaled)}  # fmt: skip


def reconcile(frozen: Frozen, view: JournalView) -> list[str]:
    """Differences between observed executions and journaled completions (empty = equal)."""
    want = observed_expectation(frozen)
    problems: list[str] = []
    for key in sorted(set(want["completed"]) | set(view.completed)):
        a, b = want["completed"].get(key, 0), view.completed.get(key, 0)
        if a != b:
            problems.append(f"{key}: observed {a} completed executions, journal holds {b}")
    for key in sorted(set(want["outcomes"]) | set(view.outcomes)):
        a, b = want["outcomes"].get(key, 0), view.outcomes.get(key, 0)
        if a != b:
            problems.append(f"{key}: observed outcome count {a}, journal holds {b}")
    if view.malformed:
        problems.append(f"journal holds {view.malformed} malformed record(s)")
    return problems
