"""Conflicting copies of one native event are counted once per dropped record, on any schedule.

Three transcripts hold the same logical tool call; two carry one variant, one carries another.
However the cycles are cut (one shot, or one source per cycle with a restart between), the number
of records dropped as conflicts, the gap block and therefore the snapshot identity must agree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.collector_types import CycleBudget
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import MCP, claude_prompt, claude_tool_use

NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)


def variant(command: str) -> list[dict]:
    return [
        claude_prompt("s1", "hello", uuid="u1", ts="2026-10-12T09:00:00.000Z"),
        claude_tool_use(
            "s1", "t1", MCP, {"query": command}, ts="2026-10-12T09:00:05.000Z", message_id="m1"
        ),
    ]


def build(root: Path) -> None:
    vs.enable()
    vs.start_capture()
    vs.put(vs.projects(root), "one", variant("alpha"), CLOCK)
    vs.put(vs.projects(root), "two", variant("beta"), CLOCK)
    vs.put(vs.projects(root), "three", variant("beta"), CLOCK)


def finish(budget: CycleBudget | None) -> tuple[dict, str, int]:
    cycles = 0
    while True:
        result = scene.collect(NOW, budget=budget)
        cycles += 1
        if scene.report(result).stopped_by == "complete":
            break
        assert cycles < 200
    envelope = scene.current()
    return scene.coverage(envelope)["gaps"], envelope.snapshot_id, cycles


@pytest.mark.parametrize("name", ["a", "eeeee", "g", "hh", "ffffff", "bb"])
def test_dropped_conflicts_do_not_depend_on_how_the_backlog_was_cut(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / name
    with vs.isolated(vs.env_for(root)):
        build(root)
        one_gaps, one_id, one_cycles = finish(None)
        evidence = vs.evidence_identities()
    vs.wipe(root)
    with vs.isolated(vs.env_for(root)):
        build(root)
        cut_gaps, cut_id, cut_cycles = finish(CycleBudget(bytes=1))  # one record per cycle
        assert vs.evidence_identities() == evidence  # same retained variant either way
    assert one_cycles == 1 and cut_cycles >= 3  # the schedules really differ
    assert cut_gaps == one_gaps
    assert cut_id == one_id
    dropped = one_gaps.get("dropped_conflict", 0)
    assert dropped >= 1  # the conflict is reported, never silently swallowed
    assert dropped + one_gaps.get("dropped_duplicate", 0) <= 2
