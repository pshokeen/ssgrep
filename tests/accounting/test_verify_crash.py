"""9.3: SIGKILL a real worker at EVERY fault-hook step of a cycle; readers and recovery stay exact.

A scene runs cycle 1 (generation 1 published). New evidence arrives. A second PROCESS then runs
cycle 2 and kills itself with SIGKILL at one hook step (no ``finally``, no cleanup). Then:

* readers see exactly one COMPLETE generation (the previous or the new one): ``current.json``
  decodes, verifies its own ``snapshot_id``, and equals the bytes of the generation file it names;
* the committed collector state is the previous or the new complete state (ledger digest);
* the next collect (same process type as production: no hand repair) converges to the canonical
  ledger digest, retained evidence, live counters and ``snapshot_id`` of an UNINTERRUPTED run of
  the same scene computed in an independent data root.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import service, snapshots
from ssgrep.accounting.lease import inspect_lease
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE

ROOT = Path(__file__).resolve().parents[2]
NOW1 = scene.at(hour=20)
NOW2 = scene.at(hour=21)
CLOCK = scene.at(hour=9)
BEFORE_COMMIT = (
    "started",
    "evidence_written",
    "state_written",
    "before_stage",
    "gen_tmp_written",
    "gen_linked",
    "before_commit",
)
AFTER_COMMIT = (
    "checkpoint_committed",
    "receipts_written",
    "current_tmp_written",
    "current_renamed",
    "published",
)
CURRENT_SWAPPED = {"current_renamed", "published"}
STEPS = (*BEFORE_COMMIT, *AFTER_COMMIT)


facts = vs.facts
Facts = vs.Facts


def stray_files() -> list[str]:
    """Durable files no committed checkpoint names, plus any leftover temporary."""
    committed = vs.checkpoint()
    assert committed is not None
    named = {s.name for s in committed.segments} | set(committed.blobs.values())
    found = [
        f"{p.parent.name}/{p.name}"
        for sub in ("evidence", "state")
        for p in (vs.accounting_dir() / "collector" / sub).iterdir()
        if p.name not in named
    ]
    found += [
        f"snapshots/{p.name}"
        for p in (vs.accounting_dir() / "snapshots").iterdir()
        if p.name.endswith(".tmp")
    ]
    return sorted(found)


def drive(root: Path, step: str, marks: Path) -> subprocess.CompletedProcess[str]:
    """One cycle in ANOTHER process that SIGKILLs itself at ``step`` (``none``: run to the end)."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.accounting.verify_driver",
            "collect",
            step,
            NOW2.isoformat(),
            "--marks",
            str(marks),
        ],
        env=vs.env_for(root),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def build_cycle_one(root: Path) -> Facts:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)
    first = scene.collect(NOW1)
    assert first.state == "published" and scene.report(first).complete
    return facts()


def add_batch(root: Path, indices: tuple[int, ...]) -> None:
    for n in indices:
        vs.pair(vs.projects(root), n, CLOCK)


wipe = vs.wipe


def reference(root: Path) -> dict[str, Facts]:
    """An UNINTERRUPTED run of the scene on THIS root's paths, then wiped for the crash run.

    Witness/action locators inside the private ledger state name the transcript files, so the
    canonical ledger digest is only comparable between runs over the same paths.
    """
    one = build_cycle_one(root)
    add_batch(root, (2, 3))
    second = scene.collect(NOW2)
    assert second.state == "published" and scene.report(second).complete
    two = facts()
    add_batch(root, (4,))
    third = scene.collect(NOW2)
    assert third.state == "published" and scene.report(third).complete
    out = {"one": one, "two": two, "three": facts()}
    wipe(root)
    return out


def assert_one_complete_generation(old: Facts, new: Facts, step: str) -> Facts:
    """Readers' view right after the crash: exactly the old or the new generation, intact."""
    snap = vs.accounting_dir() / "snapshots"
    raw = (snap / "current.json").read_bytes()
    envelope = snapshots.decode(raw)  # raises on any partial or self-inconsistent document
    named = f"gen-{envelope.generation:010d}-{envelope.snapshot_id}.json"
    assert (snap / named).read_bytes() == raw, "current.json is not the bytes of its generation"
    read, failure, _ = service.read_snapshot(PROFILE)
    assert failure is None and read is not None and read.snapshot_id == envelope.snapshot_id
    if step in CURRENT_SWAPPED:
        assert envelope.generation == old.generation + 1
        assert envelope.snapshot_id == new.snapshot_id  # the NEW generation, complete
    else:
        assert raw == old.current_bytes, "readers must still see the previous generation exactly"
    for path in snap.iterdir():  # no reader-visible partial under a final name
        if not path.name.startswith("."):
            snapshots.decode(path.read_bytes())
    return Facts(
        vs.ledger_digest(),
        tuple(vs.evidence_identities()),
        envelope.snapshot_id,
        "",
        envelope.generation,
        raw,
    )


@pytest.mark.parametrize("variant", ["plain", "late"])
@pytest.mark.parametrize("step", STEPS)
def test_sigkill_at_every_hook_exposes_one_complete_generation_and_recovers_exactly(
    step: str, variant: str, tmp_path: Path
) -> None:
    references = reference(tmp_path)
    ref_old, ref_crashed = references["one"], references["two"]
    ref_new = references["two" if variant == "plain" else "three"]
    old = build_cycle_one(tmp_path)
    assert (old.ledger, old.snapshot_id) == (ref_old.ledger, ref_old.snapshot_id)  # same scene
    add_batch(tmp_path, (2, 3))

    marks = tmp_path / "marks"
    died = drive(tmp_path, step, marks)
    assert died.returncode == -signal.SIGKILL, (died.returncode, died.stderr[-2000:])
    assert (marks / "died-at").read_text() == step  # the fault really struck at this step
    held = inspect_lease(PROFILE)
    assert held is None or not held.alive  # the kernel dropped the lease with the process

    seen = assert_one_complete_generation(old, ref_crashed, step)
    if step in BEFORE_COMMIT:
        assert seen.ledger == old.ledger and seen.evidence == old.evidence  # old state, whole
    else:
        assert (committed := vs.checkpoint()) is not None and committed.generation == 2
        assert seen.ledger == ref_crashed.ledger  # the new state is committed, whole

    if variant == "late":
        add_batch(tmp_path, (4,))  # more evidence lands before the restart
    committed_before = scene.committed().generation
    recovered = scene.collect(NOW2)
    assert recovered.state == "published", recovered  # not already_running: dead lease recovered
    assert scene.report(recovered).complete
    final = facts()
    assert final.generation == committed_before + 1  # one new generation, never a reused number
    assert final.ledger == ref_new.ledger  # canonical rows == an uninterrupted run
    assert final.evidence == ref_new.evidence  # accepted exactly once: no loss, no duplicate
    assert final.snapshot_id == ref_new.snapshot_id
    assert final.counters == ref_new.counters  # live counters: each event once
    again = scene.collect(NOW2)  # and the system is quiescent: a rerun changes nothing durable
    assert again.state == "published"
    assert (facts().ledger, facts().snapshot_id) == (ref_new.ledger, ref_new.snapshot_id)
    assert stray_files() == []  # crash orphans and superseded blobs are all swept by now


def test_the_uninterrupted_driver_matches_the_in_process_reference(tmp_path: Path) -> None:
    """The positive twin of the crash runs: the same driver with no fault finishes identically."""
    references = reference(tmp_path)
    build_cycle_one(tmp_path)
    add_batch(tmp_path, (2, 3))
    done = drive(tmp_path, "none", tmp_path / "marks")
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.splitlines()[-1])["state"] == "published"
    final, ref = facts(), references["two"]
    assert (final.ledger, final.snapshot_id, final.evidence) == (
        ref.ledger,
        ref.snapshot_id,
        ref.evidence,
    )
    assert final.generation == 2


def test_the_scene_really_changes_the_ledger_between_the_cycles(tmp_path: Path) -> None:
    """Without this the old-vs-new distinctions above would be vacuous."""
    references = reference(tmp_path)
    one, two, three = references["one"], references["two"], references["three"]
    assert len({one.ledger, two.ledger, three.ledger}) == 3
    assert len({one.snapshot_id, two.snapshot_id, three.snapshot_id}) == 3
    assert len(one.evidence) < len(two.evidence) < len(three.evidence)


# ---- the frequency steps: the pin happens in cycle 2 -----------------------------------------


def build_unpinned_cycle_one(root: Path) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)
    vs.filler(vs.projects(root), "fill-a", CLOCK, 150)
    first = scene.collect(NOW1)
    assert first.state == "published"
    assert scene.coverage(scene.current())["frequency"]["state"] == "insufficient_corpus"


def add_corpus(root: Path) -> None:
    add_batch(root, (2,))
    vs.filler(vs.projects(root), "fill-b", CLOCK, 100)  # now >= 200 documents: pin on next cycle


def pin_reference(root: Path) -> Facts:
    build_unpinned_cycle_one(root)
    add_corpus(root)
    assert scene.collect(NOW2).state == "published"
    assert scene.coverage(scene.current())["frequency"]["state"] == "pinned"
    out = facts()
    wipe(root)
    return out


@pytest.mark.parametrize("step", ["frequency_view_written", "frequency_pinned"])
def test_sigkill_while_pinning_the_frequency_view_never_leaves_a_half_pinned_view(
    step: str, tmp_path: Path
) -> None:
    expected = pin_reference(tmp_path)
    build_unpinned_cycle_one(tmp_path)
    before = facts()
    add_corpus(tmp_path)
    marks = tmp_path / "marks"
    died = drive(tmp_path, step, marks)
    assert died.returncode == -signal.SIGKILL, died.stderr[-2000:]
    assert (marks / "died-at").read_text() == step
    assert (facts().current_bytes, vs.ledger_digest()) == (before.current_bytes, before.ledger)
    recovered = scene.collect(NOW2)
    assert recovered.state == "published"
    final = facts()
    assert scene.coverage(scene.current())["frequency"] == {
        **scene.coverage(scene.current())["frequency"],
        "state": "pinned",
    }
    assert (final.ledger, final.evidence, final.snapshot_id) == (
        expected.ledger,
        expected.evidence,
        expected.snapshot_id,
    )
    assert final.counters == expected.counters


def snapshot_of(tmp_path: Path) -> dict[str, Any]:  # pragma: no cover - debugging aid
    return json.loads((vs.accounting_dir() / "snapshots" / "current.json").read_text())


def test_a_committed_but_unpublished_generation_is_repaired_before_any_new_work(
    tmp_path: Path,
) -> None:
    """Killed after the commit, before current.json moved; the NEXT cycle itself then fails.

    Recovery must already have made the committed generation current by the time anything new
    can go wrong, so a reader never keeps the superseded generation behind a failing worker.
    """
    old = build_cycle_one(tmp_path)
    add_batch(tmp_path, (2, 3))
    marks = tmp_path / "marks"
    died = drive(tmp_path, "checkpoint_committed", marks)
    assert died.returncode == -signal.SIGKILL, died.stderr[-2000:]
    assert facts().current_bytes == old.current_bytes  # readers still see generation 1
    committed = vs.checkpoint()
    assert committed is not None and committed.generation == 2

    def fail_after_recovery(name: str) -> None:
        if name == "evidence_written":  # recovery ran at the start of the cycle, before this
            raise RuntimeError("synthetic failure after recovery")

    with pytest.raises(Exception, match="Collection failed"):
        scene.collect(NOW2, on_step=fail_after_recovery)
    repaired = snapshots.decode((vs.accounting_dir() / "snapshots" / "current.json").read_bytes())
    assert repaired.generation == 2  # the committed generation is what readers see now
    assert repaired.snapshot_id != old.snapshot_id
