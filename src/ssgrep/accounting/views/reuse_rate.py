"""Agent-facing reuse verdicts: honest counts straight from the ledger.

One screen, no dashboard: total lookups, operative reuses (``action_reference`` +
``validated`` + ``inspected`` — the verdicts the ledger credits) and the operative
reuse rate over the scope window. Exploratory lookups, contradictions and
unavailable rows are never counted as credit; the rate is operative reuses over
total lookups and nothing more flattering.
"""

from __future__ import annotations

from ssgrep.accounting.ledger.contracts import PERSONAL_PROFILE
from ssgrep.accounting.views.lookups import read_lookups
from ssgrep.accounting.views.types import Scope

#: The only verdicts the ledger treats as credit; mirrors reduce_cands.CREDITED
#: plus INSPECTED, which reduce_cands strengths at 1 (observed, not applied).
OPERATIVE_VERDICTS = ("action_reference", "validated", "inspected")


def _pages(*, profile: str, scope: Scope, limit: int = 100) -> tuple[dict[str, int], int]:
    """Verdict mix and total across every lookup page in scope."""
    mix: dict[str, int] = {}
    walked = 0
    cursor: str | None = None
    while True:
        view = read_lookups(profile, scope, limit=limit, cursor=cursor)
        for row in view.rows:
            verdict = row.reuse if row.reuse_evaluated else "unavailable"
            mix[verdict] = mix.get(verdict, 0) + 1
            walked += 1
        cursor = view.next_cursor
        if cursor is None or not view.rows:
            break
    return mix, walked


def agent_summary(profile: str = PERSONAL_PROFILE, scope: Scope | None = None) -> dict[str, object]:
    """The honest self-correction summary an agent reads in one screen."""
    scope = scope or Scope()
    mix, total = _pages(profile=profile, scope=scope)
    operative = sum(mix.get(verdict, 0) for verdict in OPERATIVE_VERDICTS)
    rate: float | None = (operative / total) if total else None
    return {
        "profile": profile,
        "total_lookups": total,
        "operative_reuses": operative,
        "operative_reuse_rate": rate,
        "verdict_mix": mix,
        "operative_verdicts": list(OPERATIVE_VERDICTS),
    }
