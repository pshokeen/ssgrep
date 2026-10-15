"""Actual Resolver path caps each source gap before actor interval union."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting.evidence_tokens import Token
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.sources import Consumer, Resolver
from ssgrep.accounting.timeline import build_timelines
from ssgrep.accounting.units import WORK_SECONDS, UnitSpec
from ssgrep.accounting.valuation_credit import footprint, footprint_size
from tests.accounting.reduce_builders import (
    claude_evidence,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    policy,
)
from tests.accounting.review_corpus import build_corpus
from tests.accounting.test_reduce_evidence import FACT, PATH

BASE = datetime(2026, 4, 15, tzinfo=UTC)
CONSUMER = Consumer("claude:consumer", ("claude", "consumer"), 0, 1_800_000_000_000)
WORK = UnitSpec(WORK_SECONDS, "capped_gap_v1")


def resolved(
    tmp_path: Path, gap: float | None, *, terminal=False, next_turn=False, next_lookup=False
):
    rows = [
        claude_prompt("src", "investigate", uuid="u1", ts=BASE.isoformat()),
        claude_tool_use("src", "w1", "Bash", {"command": "cat log"}, ts=BASE.isoformat()),
        claude_tool_result("src", "w1", FACT, ts=BASE.isoformat()),
    ]
    if not terminal:
        end = (BASE + timedelta(seconds=gap)).isoformat() if gap is not None else None
        row = (
            claude_prompt("src", "unrelated next task", uuid="u2")
            if next_turn
            else claude_tool_use(
                "src",
                "w2",
                "Bash",
                {"command": "ssgrep search unrelated" if next_lookup else "continue"},
            )
        )
        row["timestamp"] = end
        rows.append(row)
    evidence = claude_evidence(tmp_path, {"-work/src.jsonl": rows})
    resolver = Resolver(build_timelines(evidence), Scoper(policy()))
    result = resolver.resolve(Token("path", PATH), ("claude:src:ep:0",), CONSUMER)
    return next(a for a in result.immediate if a.atom.origin_event.endswith("result:w1"))


@pytest.mark.parametrize("gap,expected", [(299, 299), (300, 300), (301, 300), (28801, 300), (0, 0)])
def test_actual_resolver_positive_gap_cap(tmp_path: Path, gap, expected) -> None:
    atom = resolved(tmp_path, gap)
    assert atom.t_end - atom.t_start == expected
    assert atom.work_method == "capped_gap_v1"
    assert atom.actor == "claude:src"
    fp, missing = footprint((atom,), WORK)
    assert footprint_size(fp) == expected * 1_000_000_000
    assert missing == ()


@pytest.mark.parametrize(
    "gap,terminal,next_turn",
    [(-1, False, False), (None, False, False), (1, True, False), (301, False, True)],
)
def test_missing_reversed_terminal_and_task_boundary_gaps_are_unknown(
    tmp_path: Path, gap, terminal, next_turn
) -> None:
    atom = resolved(tmp_path, gap, terminal=terminal, next_turn=next_turn)
    assert atom.usable  # still usable for chars, not fabricated time zero
    assert atom.t_start is None and atom.t_end is None
    fp, missing = footprint((atom,), WORK)
    assert footprint_size(fp) == 0 and missing


def test_new_lookup_boundary_is_not_original_witness_work(tmp_path: Path) -> None:
    atom = resolved(tmp_path, 299, next_lookup=True)
    assert atom.t_start_ms is None and atom.t_end_ms is None
    assert atom.cost_reason == "source_lookup_boundary_gap"


def test_future_completion_after_exposure_is_not_original_saved_work(tmp_path: Path) -> None:
    atom = resolved(tmp_path, 90_000_000)
    assert atom.t_start_ms is None and atom.t_end_ms is None
    assert atom.cost_reason == "source_gap_after_exposure"


def test_cap_precedes_overlap_union_and_distinct_actors_sum(tmp_path: Path) -> None:
    first = resolved(tmp_path, 28801)
    second = replace(
        first,
        atom=replace(first.atom, atom_id="second", origin_event="second"),
        t_start=first.t_start + 200,
        t_end=first.t_start + 500,
        t_start_ms=first.t_start_ms + 200_000,
        t_end_ms=first.t_start_ms + 500_000,
    )
    fp, _ = footprint((first, second, first), WORK)
    assert footprint_size(fp) == 500_000_000_000  # capped overlap union; copies add none
    parallel = replace(first, actor="claude:other")
    fp, _ = footprint((first, parallel), WORK)
    assert footprint_size(fp) == 600_000_000_000


def test_fractional_gaps_preserve_milliseconds_until_union(tmp_path: Path) -> None:
    first = resolved(tmp_path, 0.5)
    second = replace(
        first,
        atom=replace(first.atom, atom_id="second", origin_event="second"),
        t_start_ms=first.t_start_ms + 500,
        t_end_ms=first.t_start_ms + 1000,
    )
    fp, missing = footprint((first, second), WORK)
    assert footprint_size(fp) == 1_000_000_000
    assert missing == ()


def test_frozen_corpus_28801_gap_caps_on_actual_resolver(tmp_path: Path) -> None:
    from ssgrep.accounting.reduce_io import load_evidence

    corpus = build_corpus(tmp_path, setup_at=datetime(2026, 10, 12, tzinfo=UTC))
    evidence = load_evidence(corpus.home.env, only_under=[corpus.home.paths["omp"]]).evidence
    resolver = Resolver(build_timelines(evidence), Scoper(policy()))
    line = resolver.timelines.sessions[("omp", "o-src-gap")]
    call = next(r for r in line.records if r.kind == "tool_call")
    atom = resolver._atom(line, call, CONSUMER, set())
    assert atom is not None and atom.t_end is not None and atom.t_start is not None
    assert atom.t_end - atom.t_start == 300
    assert atom.work_method == "capped_gap_v1"
