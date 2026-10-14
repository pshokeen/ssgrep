"""Full vs incremental: identical canonical ledger rows from the REAL reducer on six runtimes.

The corpus is ``readers_fixtures.build_home`` (claude with copies and a conflicting fork, codex,
omp, pi, prime-agent, opencode) plus creditable search/read/act sessions so the ledger really has
rows to compare. Runs of the same final corpus must agree:

* single-shot: one unbudgeted cycle;
* incremental: tiny byte budgets, a restart (fresh state load) between every cycle, and records
  appended to a live transcript while the backlog is still being worked off;
* replay: the retained evidence of the incremental run reduced from scratch by ``reduce_profile``
  into an empty ledger.

Witness/action locators inside the private ledger state name the transcript files, so runs are
compared over the SAME paths (each run wipes its predecessor first).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ssgrep.accounting import collector_store as store, policy
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.collector_types import CycleBudget
from ssgrep.accounting.journal_reader import read_journal
from ssgrep.accounting.reduce import methodology_for, reduce_profile
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.readers_fixtures import SyntheticHome, append_bytes, build_home, jsonl
from tests.accounting.reduce_builders import (
    KEY,
    MCP,
    PROFILE,
    claude_tool_result,
    claude_tool_use,
    results_json,
    ts,
)

NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)
TINY = 2_500  # bytes per cycle: forces many cycles over the corpus
ROUNDS = (1, 2)


def environment(root: Path, home: SyntheticHome) -> dict[str, str]:
    return {**home.env, "SSGREP_DATA_DIR": str(root / "data")}


def prepare(home: SyntheticHome) -> None:
    """Policy, key, pinned view, capture start, and creditable sessions in the claude store."""
    vs.enable(pin=True)
    vs.start_capture(day=1)  # capture began Oct 1 08:00: every fixture record is later
    for n in (0, 1, 2):
        vs.pair(home.paths["claude"], n, CLOCK)


def append_live(home: SyntheticHome, round_: int) -> None:
    """New records land in an already-known transcript between cycles (an active session)."""
    target = home.paths["claude"] / "-work" / "c1.jsonl"
    call = f"tapp{round_}"
    records = [
        claude_tool_use(
            "c1",
            call,
            MCP,
            {"query": f"late query {round_}"},
            ts=ts(12, 15, round_, 0),
            message_id=f"mapp{round_}",
        ),
        claude_tool_result("c1", call, results_json(), ts=ts(12, 15, round_, 3)),
    ]
    append_bytes(target, jsonl(records))
    os.utime(target, (CLOCK.timestamp(), CLOCK.timestamp()))


def single_shot(root: Path) -> vs.Facts:
    home = build_home(root / "hm")
    with vs.isolated(environment(root, home)):
        prepare(home)
        for round_ in ROUNDS:
            append_live(home, round_)
        result = scene.collect(NOW)
        assert result.state == "published" and scene.report(result).complete
        out = vs.facts()
    vs.wipe(root)
    shutil.rmtree(root, ignore_errors=True)  # the synthetic home too: it is rebuilt from scratch
    return out


def run_incremental(home: SyntheticHome) -> int:
    """Many tiny cycles with new data arriving mid-backlog; returns the number of cycles."""
    cycles = 0
    for round_ in ROUNDS:
        for _ in range(2):  # new data lands while the backlog is still unfinished
            scene.collect(NOW, budget=CycleBudget(bytes=TINY))
            cycles += 1
        append_live(home, round_)
    while True:
        result = scene.collect(NOW, budget=CycleBudget(bytes=TINY))
        cycles += 1
        if scene.report(result).stopped_by == "complete":
            return cycles
        assert cycles < 400, "backlog never finished"


def test_incremental_with_restarts_and_live_appends_equals_single_shot(tmp_path: Path) -> None:
    root = tmp_path / "run"
    expected = single_shot(root)
    assert expected.ledger is not None and expected.evidence

    home = build_home(root / "hm")  # the same paths again, rebuilt from scratch
    with vs.isolated(environment(root, home)):
        prepare(home)
        cycles = run_incremental(home)
        got = vs.facts()
    assert cycles >= 6  # a real incremental history, not a one-cycle run
    assert got.evidence == expected.evidence  # same retained records, none doubled, none missing
    assert len(set(got.evidence)) == len(got.evidence)
    assert got.ledger == expected.ledger  # canonical ledger rows are byte-identical
    assert got.counters == expected.counters
    assert got.snapshot_id == expected.snapshot_id  # even the snapshot identity matches


def test_the_corpus_really_spans_all_six_runtimes_and_has_ledger_rows(tmp_path: Path) -> None:
    root = tmp_path / "run"
    home = build_home(root / "hm")
    with vs.isolated(environment(root, home)):
        prepare(home)
        result = scene.collect(NOW)
        assert result.state == "published"
        facts = vs.facts()
        envelope = scene.current()
    runtimes = {e[0] for e in facts.evidence}
    assert runtimes == {"claude", "codex", "omp", "pi", "prime-agent", "opencode"}
    assert set(scene.coverage(envelope)["sources"]["by_runtime"]) == runtimes
    assert facts.ledger is not None
    assert (
        scene.coverage(envelope)["candidate_receipts"] >= 3
    )  # the ledger has real receipts to compare


def test_a_from_scratch_replay_of_the_retained_evidence_equals_the_incremental_ledger(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    home = build_home(root / "hm")
    with vs.isolated(environment(root, home)):
        prepare(home)
        run_incremental(home)
        committed = vs.checkpoint()
        assert committed is not None
        incremental_digest = vs.ledger_digest()
        evidence = list(store.iter_evidence(PROFILE, committed.segments))
        events = read_journal(PROFILE).events
        view = store_view()
        reduction = reduce_profile(
            PROFILE,
            evidence,
            events,
            policy=policy.load_profile_policy(PROFILE),
            frequency=view,
            fingerprint_key=KEY,
        )
    replayed = VersionedLedger(methodology_for(view))
    replayed.reconcile(
        dict(zip(reduction.bundle_roots, reduction.bundles, strict=True)), today="2026-10-12"
    )
    assert reduction.bundles and replayed.rows().receipts  # a real ledger, not two empty ones
    assert replayed.rows().digest == incremental_digest


def store_view():  # type: ignore[no-untyped-def]
    from ssgrep.accounting import collector_frequency as frequency

    view = frequency.load_pinned(PROFILE)
    assert view is not None
    return view
