"""Persistence barrier and retrieval checks driven by the VERIFIER actor (15.9, 15.10).

Everything here goes through the real ``ssgrep`` binary as the verifier: lookups are recorded in
the observer trace with ``actor=verifier`` (global executions with real cost, never solver
exposure). Source resolution is judged against independently derived evidence: the native file's
newline-complete prefix digest and the oracle-derived result rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import domain_oracle as oracle
from .lab_product import BINARY, Product
from .observer import NativeSession, Observer, sha
from .roots import LabRoots

LOOKUP_TIMEOUT = 600.0


@dataclass
class Lookup:
    """One observed verifier lookup and its parsed JSON payload."""

    call_id: str
    argv: tuple[str, ...]
    returncode: int | None
    payload: dict[str, Any] | None
    stdout_chars: int
    elapsed_ms: int
    state: str


def verifier_lookup(
    observer: Observer, product: Product, roots: LabRoots, *args: str, task: str | None = None
) -> Lookup:
    """``ssgrep <args> --json`` as the verifier: recorded dispatch, raw output, exit status."""
    import time

    started = time.monotonic()
    shown = ["ssgrep", *args, "--json"]
    call, proc = observer.run(
        actor="verifier", task=task, tool="ssgrep", argv=[str(BINARY), *args, "--json"],
        env=product.env, cwd=roots.project, timeout=LOOKUP_TIMEOUT, shown=shown,
    )  # fmt: skip
    payload: dict[str, Any] | None = None
    if proc is not None:
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.startswith("{"):
                payload = json.loads(line)
                break
    return Lookup(
        call.call_id, tuple(shown), proc.returncode if proc else None, payload, call.stdout_chars,
        int((time.monotonic() - started) * 1000), call.state,
    )  # fmt: skip


def data_of(lookup: Lookup) -> dict[str, Any]:
    body = lookup.payload or {}
    return body.get("data", body) if body.get("ok", True) is not False else {}


def refs_of(lookup: Lookup) -> list[str]:
    results = data_of(lookup).get("results") or []
    return [c["ref"] for c in results if isinstance(c, dict) and isinstance(c.get("ref"), str)]


# ------------------------------------------------------------------ source expectations


def l1_evidence(state: oracle.State) -> list[str]:
    """Strings a faithful A/L1 source must contain, derived from the oracle (never from SQL)."""
    cash = oracle.settled_cash(state)
    rows = [f"{b} {c} {minor}" for (b, c), minor in sorted(cash.items())]
    seats = [f"{r.flight_id} {' '.join(str(x) for x in r.key())}" for r in oracle.seat_rows(state)]
    return rows + seats


def l2_evidence(state: oracle.State) -> list[str]:
    """Strings a faithful A/L2 source must contain (affected people, bad gaps, policy ids)."""
    out = [f"{b} {t}" for b, t in oracle.affected_travellers(state)]
    for c in oracle.connections(state):
        if not c.valid:
            out.append(c.rule_id)
            out.append(f"{c.gap_minutes}")
    out += [r["rule_id"] for r in state["connection_rules"]]
    return out


@dataclass
class Resolution:
    """The judged result of resolving one A task through real search + show."""

    task: str
    query: str
    ref: str | None
    found_in_search: bool
    shown: bool
    problems: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _runtime_identity(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "runtime": card.get("runtime"),
        "project": card.get("project"),
        "source_path": card.get("source_path"),
        "source_absent": card.get("source_absent"),
    }


def _judge_shown(
    detail: Mapping[str, Any],
    card: Mapping[str, Any],
    session: NativeSession,
    project: Path,
    expected_text: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Problems + evidence of one opened ref against independent expectations."""
    problems: list[str] = []
    text = f"{detail.get('prompt_text', '')}\n{detail.get('response_text', '')}"
    ident = _runtime_identity(card)
    evidence: dict[str, Any] = {
        "identity": ident,
        "prompt_truncated": detail.get("prompt_truncated"),
        "response_truncated": detail.get("response_truncated"),
        "shown_text_digest": sha(text),
        "shown_chars": len(text),
    }
    if ident["runtime"] != "omp":
        problems.append(f"runtime {ident['runtime']!r} != 'omp'")
    if os.path.realpath(str(ident["project"])) != os.path.realpath(project):
        problems.append(f"project {ident['project']!r} != {str(project)!r}")
    if ident["source_absent"]:
        problems.append("source reported absent")
    source = Path(str(ident["source_path"]))
    if not source.is_file():
        problems.append("source_path does not exist")
    else:
        data = source.read_bytes()
        got = hashlib.sha256(data[: data.rfind(b"\n") + 1]).hexdigest()
        evidence["source_digest"] = got
        if got != session.file_digest:
            problems.append("source digest differs from A's recorded native digest")
    if detail.get("response_truncated") or detail.get("prompt_truncated"):
        problems.append("shown source is truncated: evidence may be hidden")
    missing = [s for s in expected_text if s not in text]
    if missing:
        problems.append(f"shown source lacks verified evidence: {missing[:4]}")
    return problems, evidence


def resolve_task(
    observer: Observer,
    product: Product,
    roots: LabRoots,
    *,
    task: str,
    query: str,
    session: NativeSession,
    expected_text: Sequence[str],
    project: Path,
) -> tuple[Resolution, Lookup, list[Lookup]]:
    """Search naturally, open each returned ref of A's session, and judge the first faithful one.

    Every opened ref is a real verifier ``show`` (recorded); the verdict is about the ACTUAL
    returned source, its identity and its digest, never about a ref the lab chose in advance.
    """
    searched = verifier_lookup(observer, product, roots, "search", query, task=task)
    prefix = f"omp:{session.session_id}:ep:"
    candidates = [r for r in refs_of(searched) if r.startswith(prefix)]
    res = Resolution(task, query, None, bool(candidates), False)
    opened: list[Lookup] = []
    if searched.returncode != 0:
        res.problems.append(f"search exited {searched.returncode}")
    if not candidates:
        res.problems.append(f"no ref of session {session.session_id} among {refs_of(searched)[:5]}")
        return res, searched, opened
    cards = {c["ref"]: c for c in data_of(searched).get("results", []) if isinstance(c, dict)}
    for ref in candidates[:4]:
        shown = verifier_lookup(observer, product, roots, "show", ref, task=task)
        opened.append(shown)
        problems, evidence = _judge_shown(
            data_of(shown), cards[ref], session, project, expected_text
        )
        if shown.returncode != 0:
            problems.insert(0, f"show exited {shown.returncode}")
        res.ref, res.shown, res.evidence, res.problems = (
            ref,
            shown.returncode == 0,
            evidence,
            problems,
        )
        if not problems:
            break
    return res, searched, opened
