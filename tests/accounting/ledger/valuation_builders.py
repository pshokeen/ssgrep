"""Builders for valuation tests: complete valid candidates, atoms, and debits."""

from __future__ import annotations

from typing import Any

from ssgrep.accounting.ledger.contracts import Basis, Quantity, Reuse, TaskId
from ssgrep.accounting.ledger.ledger_contracts import SourceAtom
from ssgrep.accounting.ledger.units import CHARS_DIV4_V1, CONTEXT_TOKENS, UnitSpec
from ssgrep.accounting.ledger.valuation_inputs import (
    AtomStatus,
    Candidate,
    Debit,
    Methodology,
    ValuationAtom,
)

CHARS = UnitSpec(CONTEXT_TOKENS, CHARS_DIV4_V1)
METHOD = Methodology("test-v1", (CHARS,))
TASK = TaskId("root-1", "turn-1", "main")
DAY = "2026-11-01"


def atom(
    atom_id: str, chars: int, start: int = 0, origin: str = "ev-1", **kw: Any
) -> ValuationAtom:
    """A verified atom of ``origin`` covering [start, start+chars)."""
    digest = kw.pop("digest", f"d-{origin}")
    source = SourceAtom(atom_id, origin, digest, chars, kw.pop("seconds", None))
    return ValuationAtom(source, start=start, **kw)


def candidate(cid: str, **overrides: Any) -> Candidate:
    """An eligible, novel, acted-on candidate with one 400-char immediate step."""
    fields: dict[str, Any] = {
        "candidate_id": cid,
        "candidate_type": "path",
        "task": TASK,
        "day": DAY,
        "reuse": Reuse.ACTION_REFERENCE,
        "novel": True,
        "user_supplied": False,
        "action_outcome_ok": True,
        "immediate": (atom("step-1", 400),),
    }
    fields.update(overrides)
    return Candidate(**fields)


def debit(
    debit_id: str, chars: int | None, task: TaskId = TASK, day: str = DAY, **kw: Any
) -> Debit:
    """One exposed-response debit measured in characters (or unknown when None)."""
    amount = (
        Quantity(chars // 4, CONTEXT_TOKENS, Basis.ESTIMATED, CHARS_DIV4_V1)
        if chars is not None
        else Quantity.unavailable(CONTEXT_TOKENS, "downstream_exposure_unobserved")
    )
    return Debit(debit_id, task, day, kw.pop("kind", "exposure"), CHARS, amount, chars=chars, **kw)


__all__ = ["AtomStatus", "CHARS", "DAY", "METHOD", "TASK", "atom", "candidate", "debit"]
