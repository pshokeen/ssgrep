"""Append-only UTC daily period revisions with provisional->sealed semantics (D7).

A period is provisional through the seventh UTC day after it ends; evidence
*accepted* from the eighth day on is late and yields a superseding revision. The
revision a row belongs to depends only on evidence acceptance days (evidence
data), never on the current clock: ``status_at`` derives provisional/sealed for
display. Only current revisions contribute to current totals; superseded rows
are history and are never added again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TypeVar

from ssgrep.accounting.ledger.contracts import SCHEMA_VERSION, Quantity, Scenario, digest_of
from ssgrep.accounting.ledger.ledger_contracts import ScenarioEstimate
from ssgrep.accounting.ledger.units import UnitSpec
from ssgrep.accounting.ledger.valuation import Leaf, rollup

PROVISIONAL_DAYS = 7
T = TypeVar("T")


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def seal_day(period: str) -> str:
    """First UTC day on which accepted evidence for ``period`` counts as late."""
    return _shift(period, 1 + PROVISIONAL_DAYS)


def last_provisional_day(period: str) -> str:
    return _shift(period, PROVISIONAL_DAYS)


def status_at(period: str, today: str) -> str:
    """Display status for a UTC day; never part of canonical ledger content."""
    return "sealed" if today >= seal_day(period) else "provisional"


@dataclass(frozen=True)
class PeriodRow:
    """One immutable revision of one period within one methodology namespace."""

    namespace: str
    period: str
    revision: int
    supersedes: int | None
    kind: str  # original | adjustment
    late: bool  # created by evidence accepted after the seal day
    recomputable: bool
    leaves: tuple[Leaf, ...]
    estimates: tuple[tuple[str, tuple[ScenarioEstimate, ...]], ...]
    schema: int = SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return digest_of(self)


def unit_key(unit: UnitSpec) -> str:
    return f"{unit.unit}:{unit.method}:{unit.tokenizer or ''}:{unit.currency or ''}"


def canonical_leaves(leaves: Iterable[Leaf]) -> tuple[Leaf, ...]:
    return tuple(sorted(leaves, key=lambda leaf: (digest_of(leaf.unit), leaf.key)))


def estimates_of(leaves: Sequence[Leaf]) -> tuple[tuple[str, tuple[ScenarioEstimate, ...]], ...]:
    units = {digest_of(leaf.unit): leaf.unit for leaf in leaves}
    out = []
    for unit in sorted(units.values(), key=unit_key):
        rows = rollup([leaf for leaf in leaves if leaf.unit == unit], unit)
        out.append((unit_key(unit), tuple(e.as_scenario_estimate() for e in rows)))
    return tuple(out)


def revision_points(period: str, accepted_days: Iterable[str]) -> list[str]:
    """Evidence days at which ``period`` may need a revision.

    Everything accepted during the provisional window folds into the first point (the
    last provisional day); each later day with accepted evidence can add a superseding
    revision. A function of evidence only: no clock, no collection horizon.
    """
    seal = seal_day(period)
    return [last_provisional_day(period), *sorted({d for d in accepted_days if d >= seal})]


def revise(
    points: Iterable[str],
    content_at: Callable[[str], T | None],
    digest: Callable[[T], str] = digest_of,
) -> list[tuple[int, str, T]]:
    """Distinct consecutive contents as ``(revision, as_of_day, content)``.

    Empty content before any evidence creates no revision; unchanged content at a
    later point creates none either, so history never carries no-op revisions.
    """
    out: list[tuple[int, str, T]] = []
    previous: str | None = None
    for point in points:
        content = content_at(point)
        if content is None:
            continue
        current = digest(content)
        if current != previous:
            out.append((len(out) + 1, point, content))
            previous = current
    return out


def build_period_rows(
    namespace: str,
    periods: Iterable[str],
    accepted_days: Sequence[str],
    leaves_at: Callable[[str, str], Sequence[Leaf]],
    frozen_roots: frozenset[str] = frozenset(),
) -> list[PeriodRow]:
    """All revisions of every period, derived only from evidence and acceptance days."""
    rows: list[PeriodRow] = []
    for period in sorted(set(periods)):

        def content_at(point: str, period: str = period) -> tuple[Leaf, ...] | None:
            return canonical_leaves(leaves_at(point, period)) or None

        points = revision_points(period, accepted_days)
        for revision, point, leaves in revise(points, content_at):
            rows.append(
                PeriodRow(
                    namespace,
                    period,
                    revision,
                    revision - 1 or None,
                    "original" if revision == 1 else "adjustment",
                    point >= seal_day(period),
                    not any(leaf.task.root in frozen_roots for leaf in leaves),
                    leaves,
                    estimates_of(leaves),
                )
            )
    return rows


def current_rows(rows: Iterable[PeriodRow], namespace: str) -> list[PeriodRow]:
    """The single newest revision per period of ONE namespace (others are ignored)."""
    latest: dict[str, PeriodRow] = {}
    for row in rows:
        if row.namespace == namespace and row.revision > (
            latest[row.period].revision if row.period in latest else 0
        ):
            latest[row.period] = row
    return [latest[p] for p in sorted(latest)]


def current_total(
    rows: Iterable[PeriodRow], namespace: str, unit: UnitSpec
) -> tuple[Quantity | None, dict[Scenario, Quantity]]:
    """Current-revision totals of one namespace/unit; superseded rows never count."""
    leaves = [
        leaf for row in current_rows(rows, namespace) for leaf in row.leaves if leaf.unit == unit
    ]
    if not leaves:
        return None, {}
    nets = {e.scenario: e.net for e in rollup(leaves, unit)}
    return nets[Scenario.CENTRAL], nets


@dataclass(frozen=True)
class HistoryRecord:
    period: str
    revision: int
    status: str
    current: bool
    supersedes: int | None
    kind: str
    recomputable: bool
    net: Quantity


def history(
    rows: Iterable[PeriodRow], namespace: str, unit: UnitSpec, today: str
) -> tuple[HistoryRecord, ...]:
    """Every revision of every period (old and new), each flagged current or not."""
    chosen = sorted(
        (r for r in rows if r.namespace == namespace), key=lambda r: (r.period, r.revision)
    )
    newest = {row.period: row.revision for row in chosen}
    out = []
    for row in chosen:
        net = next(
            (
                e.net
                for key, ests in row.estimates
                if key == unit_key(unit)
                for e in ests
                if e.scenario is Scenario.CENTRAL
            ),
            Quantity.unavailable(unit.unit, "unit_not_in_period"),
        )
        is_current = row.revision == newest[row.period]
        out.append(
            HistoryRecord(
                row.period,
                row.revision,
                status_at(row.period, today) if is_current else "superseded",
                is_current,
                row.supersedes,
                row.kind,
                row.recomputable,
                net,
            )
        )
    return tuple(out)
