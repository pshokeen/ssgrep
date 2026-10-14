"""9.1: one worker per profile, for real: a second process, a killed holder, a forged record.

The holder is a REAL process running a real cycle that is parked at a hook step (so it genuinely
holds the profile lease with a cycle pending). Every claim is observed from outside: the
``already_running`` answer with the holder's own progress, the reducer spy that proves no second
reduction started, and the snapshot/checkpoint bytes that prove no competing generation appeared.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import snapshots
from ssgrep.accounting.collector import collect_once
from ssgrep.accounting.collector_reduce import Reducer, default_reducer
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.lease import inspect_lease
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_savings_cli import BINARY, document

ROOT = Path(__file__).resolve().parents[2]
NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)


class Holder:
    """A real worker process parked at a hook step with the lease held."""

    def __init__(self, root: Path, step: str, *, now: datetime = NOW) -> None:
        self.hold = root / f"hold-{step}"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.accounting.verify_driver",
                "collect",
                step,
                now.isoformat(),
                "--hold",
                str(self.hold),
            ],
            env=vs.env_for(root),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 120
        while not (self.hold / "reached").exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"holder never parked: {self.process.communicate()}")
            time.sleep(0.02)

    def release(self) -> dict[str, Any]:
        (self.hold / "release").write_text("go")
        out, err = self.process.communicate(timeout=120)
        assert self.process.returncode == 0, err[-2000:]
        return json.loads(out.splitlines()[-1])

    def kill(self) -> None:
        self.process.send_signal(signal.SIGKILL)
        self.process.communicate(timeout=60)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


class CountingReducer:
    """The real reducer behind a spy: how many reductions did THIS process start?"""

    def __init__(self) -> None:
        real = default_reducer()
        self.calls = 0
        self._real = real
        self.reducer = Reducer(self._reduce, real.methodology_for)

    def _reduce(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._real.reduce_profile(*args, **kwargs)


def count(progress: dict[str, object], key: str) -> int:
    """One integer progress field of the running owner (fails loudly if it is not one)."""
    value = progress[key]
    assert isinstance(value, int), (key, value)
    return value


def seeded(root: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)


def disk_state() -> dict[str, bytes]:
    return {
        **{f"snapshots/{k}": v for k, v in vs.tree(vs.accounting_dir() / "snapshots").items()},
        **{
            f"collector/{k}": v
            for k, v in vs.tree(vs.accounting_dir() / "collector").items()
            if k == "checkpoint.json"
        },
    }


def test_a_second_process_gets_already_running_with_progress_and_changes_nothing(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    holder = Holder(tmp_path, "before_stage")
    try:
        before = disk_state()
        spy = CountingReducer()
        answer = collect_once(PROFILE, CollectOptions(), now=NOW, reducer=spy.reducer)
        assert answer.state == "already_running" and answer.report is None
        progress = answer.progress
        assert progress is not None
        assert progress["pid"] == holder.process.pid and progress["kind"] == "collect"
        assert progress["phase"] == "reducing" and progress["mode"] == "live"
        assert count(progress, "sources_total") > 0 and count(progress, "bytes_read") > 0
        owner = inspect_lease(PROFILE)
        assert owner is not None and owner.pid == holder.process.pid and owner.alive
        assert spy.calls == 0  # no duplicate reduction started
        assert disk_state() == before  # no competing generation, checkpoint untouched

        # the real CLI answers the same way: exit 0, state already_running, the holder's numbers
        cli = subprocess.run(
            [str(BINARY), "metrics", "collect", "--json"],
            env=vs.env_for(tmp_path),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert cli.returncode == 0, cli.stderr
        body = document(cli)["data"]
        assert body["state"] == "already_running"
        owner_view = body["data"]["owner"]
        assert owner_view["pid"] == holder.process.pid and owner_view["phase"] == "reducing"
        assert body["data"]["generation"] is None and body["data"]["snapshot_id"] is None
        assert disk_state() == before
    finally:
        holder.close()


def test_progress_of_a_pending_cycle_reports_the_bytes_it_already_read(tmp_path: Path) -> None:
    """Regression: ``bytes_read`` stayed 0 until the cycle ended, so 'current progress' lied."""
    seeded(tmp_path)
    holder = Holder(tmp_path, "evidence_written")  # reading is over, reduction pending
    try:
        answer = scene.collect(NOW)
        assert answer.state == "already_running"
        progress = answer.progress
        assert progress is not None and progress["phase"] == "reducing"
        on_disk = sum(p.stat().st_size for p in (vs.projects(tmp_path)).rglob("*.jsonl"))
        assert (
            0 < count(progress, "bytes_read") <= on_disk
        )  # what the cycle consumed, never invented
        assert count(progress, "sources_done") == count(progress, "sources_total") > 0
    finally:
        holder.close()
    finished = scene.collect(NOW)
    assert finished.state == "published"
    final = json.loads((vs.accounting_dir() / "worker" / "progress.json").read_text())
    assert final["bytes_read"] == scene.report(finished).bytes_read > 0 or final["phase"] == "idle"


def test_progress_of_a_pending_backfill_also_reports_sources_and_bytes(tmp_path: Path) -> None:
    """Regression: a backfill cycle never updated sources_done/bytes_read while it ran."""
    seeded(tmp_path)
    assert scene.collect(NOW).state == "published"
    holder = Holder_backfill(tmp_path)
    try:
        answer = scene.collect(NOW)
        assert answer.state == "already_running"
        progress = answer.progress
        assert progress is not None and progress["mode"] == "backfill"
        assert count(progress, "sources_total") > 0
        assert count(progress, "sources_done") == count(progress, "sources_total")
        assert count(progress, "bytes_read") > 0
    finally:
        holder.close()


def Holder_backfill(root: Path) -> Holder:
    holder = Holder.__new__(Holder)
    holder.hold = root / "hold-backfill"
    holder.process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.accounting.verify_driver",
            "collect",
            "evidence_written",
            NOW.isoformat(),
            "--hold",
            str(holder.hold),
            "--backfill",
            "2026-10-01T00:00:00Z",
            "2026-11-01T00:00:00Z",
        ],
        env=vs.env_for(root),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    while not (holder.hold / "reached").exists():
        if holder.process.poll() is not None or time.monotonic() > deadline:
            raise AssertionError(f"holder never parked: {holder.process.communicate()}")
        time.sleep(0.02)
    return holder


def test_exactly_one_generation_results_from_a_pile_of_concurrent_collects(tmp_path: Path) -> None:
    seeded(tmp_path)
    holder = Holder(tmp_path, "started")
    try:
        racers = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.accounting.verify_driver",
                    "collect",
                    "none",
                    NOW.isoformat(),
                ],
                env=vs.env_for(tmp_path),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(4)
        ]
        states = []
        for racer in racers:
            out, err = racer.communicate(timeout=120)
            assert racer.returncode == 0, err[-2000:]
            states.append(json.loads(out.splitlines()[-1])["state"])
        assert states == ["already_running"] * 4
        assert not (vs.accounting_dir() / "snapshots" / "current.json").exists()  # none published
        done = holder.release()
        assert done["state"] == "published" and done["generation"] == 1
    finally:
        holder.close()
    assert snapshots.generations(PROFILE) == [1]  # one cycle, one generation, nothing competing
    after = scene.collect(NOW)
    assert after.state == "published" and scene.report(after).generation == 2


def test_a_killed_holder_is_recovered_without_waiting_and_without_stealing_a_live_lease(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    assert scene.collect(NOW).state == "published"  # generation 1 exists
    gen_one = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    vs.pair(vs.projects(tmp_path), 2, CLOCK)

    live = Holder(tmp_path, "before_commit")
    try:
        # a stale progress document of a long-dead worker must not be believed over the live one
        dead = subprocess.run([sys.executable, "-c", "pass"], check=True)
        assert dead.returncode == 0
        stale = vs.accounting_dir() / "worker" / "progress.json"
        recorded = json.loads(stale.read_text())
        recorded.update(pid=2_000_000_000, phase="idle")
        stale.write_text(json.dumps(recorded))
        stale.chmod(0o600)
        taken = scene.collect(NOW)
        assert taken.state == "already_running"  # the live holder keeps its lease
        owner = inspect_lease(PROFILE)
        assert owner is not None and owner.pid == live.process.pid
        assert (vs.accounting_dir() / "snapshots" / "current.json").read_bytes() == gen_one
    finally:
        live.kill()  # SIGKILL: no finally blocks, no release, the kernel drops the lock
    abandoned = inspect_lease(PROFILE)
    assert abandoned is not None and abandoned.pid is not None and not abandoned.alive

    spy = CountingReducer()
    recovered = scene.collect(NOW, reducer=spy.reducer)
    assert recovered.state == "published", recovered  # abandoned state is recoverable
    assert spy.calls == 1  # the new evidence is reduced exactly once
    assert scene.report(recovered).generation == 2  # the killed cycle's generation 2 was discarded
    assert scene.current().generation == 2
    progress = json.loads((vs.accounting_dir() / "worker" / "progress.json").read_text())
    assert progress["pid"] == os.getpid() and progress["phase"] == "idle"
    assert progress["last_cycle"]["status"] == "published"


def test_a_live_holder_is_not_displaced_by_a_forged_or_erased_lease_record(tmp_path: Path) -> None:
    seeded(tmp_path)
    holder = Holder(tmp_path, "started")
    lock = vs.accounting_dir() / "worker" / "lease.lock"
    try:
        for forged in (b"", b"{}", b'{"pid": 2000000000, "kind": "collect"}', b"\x00" * 64):
            lock.write_bytes(forged)  # anyone with the file can scribble over the record
            answer = scene.collect(NOW)
            assert answer.state == "already_running", forged
        assert holder.process.poll() is None
    finally:
        holder.close()
    assert scene.collect(NOW).state == "published"  # and the lease is simply free afterwards


def test_a_failed_cycle_releases_the_lease_and_keeps_the_previous_generation(
    tmp_path: Path,
) -> None:
    seeded(tmp_path)
    assert scene.collect(NOW).state == "published"
    before = disk_state()
    vs.pair(vs.projects(tmp_path), 2, CLOCK)

    def boom(name: str) -> None:
        if name == "before_commit":
            raise RuntimeError("synthetic failure")

    with pytest.raises(Exception, match="collection_failed|Collection failed"):
        scene.collect(NOW, on_step=boom)
    held = inspect_lease(PROFILE)
    assert held is None or held.pid is None  # the lease record was released with the failure
    assert (vs.accounting_dir() / "snapshots" / "current.json").read_bytes() == json.loads(
        json.dumps(before["snapshots/current.json"].decode())
    ).encode()
    again = scene.collect(NOW)
    assert again.state == "published" and scene.report(again).generation == 2


def test_utc_helper_matches_scene_clock() -> None:
    assert NOW.tzinfo is UTC
