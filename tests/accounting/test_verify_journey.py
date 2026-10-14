"""9.8: an isolated collect / restart / serve / stop journey with a REAL concurrent reader.

A reader process hammers ``service.read_snapshot`` and the real ``ssgrep savings --json`` command
while another real process sits PARKED inside a cycle (lease held, replacement generation being
produced). The old generation must stay complete, verified and identical the whole time; after
release the reader sees the new generation, and never a mix. The same journey then restarts,
serves, and stops with the real binary.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ssgrep.accounting import service, snapshots
from ssgrep.accounting.lease import inspect_lease
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.snapshots_driver import main as _unused  # noqa: F401  (driver exists)
from tests.accounting.test_savings_cli import BINARY, document
from tests.accounting.test_verify_lease import Holder

ROOT = Path(__file__).resolve().parents[2]
CLOCK = scene.at(hour=9)
NOW = scene.at(hour=20)
PARK = ("evidence_written", "state_written", "before_stage", "gen_linked", "before_commit")


def seeded(root: Path) -> snapshots.SnapshotEnvelope:
    vs.enable(pin=True)
    vs.start_capture()
    for n in (0, 1):
        vs.pair(vs.projects(root), n, CLOCK)
    assert scene.collect(NOW).state == "published"
    envelope = scene.current()
    vs.pair(vs.projects(root), 2, CLOCK)  # new evidence: the next cycle replaces the generation
    return envelope


def savings(root: Path) -> dict:
    done = subprocess.run(
        [str(BINARY), "savings", "--json"],
        env=vs.env_for(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return document(done)["data"]


READER = """
import os, sys, time
from ssgrep.accounting import service, snapshots

ready, stop = sys.argv[1], sys.argv[2]
seen, bad, torn = {}, 0, 0
while not os.path.exists(stop):
    envelope, failure, _why = service.read_snapshot("personal-local")
    if envelope is None:
        bad += 1
        continue
    try:
        ok = snapshots.snapshot_identity(envelope.__dict__) == envelope.snapshot_id
    except ValueError:
        ok = False
    torn += not ok
    identity = (envelope.generation, envelope.snapshot_id)
    seen[identity] = seen.get(identity, 0) + 1
    if not os.path.exists(ready):
        open(ready, "w").close()
print(sorted(seen), bad, torn)
"""


def start_reader(root: Path) -> tuple[subprocess.Popen[str], Path]:
    stop = root / "reader-stop"
    ready = root / "reader-ready"
    process = subprocess.Popen(
        [sys.executable, "-c", READER, str(ready), str(stop)],
        env=vs.env_for(root),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 120
    while not ready.exists():
        assert process.poll() is None and time.monotonic() < deadline, process.communicate()
        time.sleep(0.02)
    return process, stop


def finish_reader(process: subprocess.Popen[str], stop: Path) -> tuple[list, int, int]:
    stop.write_text("stop")
    out, err = process.communicate(timeout=120)
    assert process.returncode == 0, err[-2000:]
    head, bad, torn = out.rsplit(" ", 2)
    return eval(head), int(bad), int(torn)  # noqa: S307 - our own literal list of int/str tuples


def test_old_generation_stays_readable_and_identical_at_every_parked_step(tmp_path: Path) -> None:
    old = seeded(tmp_path)
    old_bytes = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
    reader, stop = start_reader(tmp_path)
    try:
        for step in PARK:
            holder = Holder(tmp_path, step)
            try:
                # the cycle is genuinely pending: lease held by the holder, replacement in flight
                owner = inspect_lease(PROFILE)
                assert owner is not None and owner.alive and owner.pid == holder.process.pid
                for _ in range(3):
                    read, failure, _ = service.read_snapshot(PROFILE)
                    assert failure is None and read is not None
                    assert (read.generation, read.snapshot_id) == (old.generation, old.snapshot_id)
                    shown = savings(tmp_path)
                    assert shown["provenance"]["snapshot_id"] == old.snapshot_id
                assert (
                    vs.accounting_dir() / "snapshots" / "current.json"
                ).read_bytes() == old_bytes
            finally:
                holder.kill()  # SIGKILL: the pending cycle never publishes
            assert scene.collect(NOW).state == "published"  # recovery publishes a clean successor
            old = scene.current()
            old_bytes = (vs.accounting_dir() / "snapshots" / "current.json").read_bytes()
            if step != PARK[-1]:
                vs.pair(vs.projects(tmp_path), 3 + PARK.index(step), CLOCK)  # next cycle's news
    finally:
        seen, bad, torn = finish_reader(reader, stop)
    assert bad == 0 and torn == 0  # the concurrent reader never saw a failure or a torn document
    generations = [g for g, _ in seen]
    assert generations == sorted(generations) and len(set(generations)) == len(seen)
    assert len(seen) >= 2  # it observed real replacement generations as they were published


def test_a_pending_cycle_that_finishes_flips_the_reader_atomically_to_the_new_generation(
    tmp_path: Path,
) -> None:
    old = seeded(tmp_path)
    reader, stop = start_reader(tmp_path)
    holder = Holder(tmp_path, "before_commit")
    try:
        mid, failure, _ = service.read_snapshot(PROFILE)
        assert failure is None and mid is not None and mid.snapshot_id == old.snapshot_id
        done = holder.release()
        assert done["state"] == "published" and done["generation"] == old.generation + 1
        time.sleep(0.3)  # let the concurrent reader observe the new generation
    finally:
        holder.close()
        seen, bad, torn = finish_reader(reader, stop)
    assert bad == 0 and torn == 0
    assert [g for g, _ in seen] == [old.generation, old.generation + 1]  # old, then new: no mix
    final = scene.current()
    assert final.generation == old.generation + 1 and final.snapshot_id != old.snapshot_id
    assert savings(tmp_path)["provenance"]["snapshot_id"] == final.snapshot_id


def test_collect_restart_serve_stop_journey_with_the_real_binary(tmp_path: Path) -> None:
    def cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BINARY), *args],
            env=vs.env_for(tmp_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    assert cli("metrics", "setup", "--yes").returncode == 0
    vs.enable(pin=True)  # fixed fingerprint key + pinned view for a creditable scene
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    first = document(cli("metrics", "collect", "--json"))["data"]
    assert first["state"] == "published" and first["data"]["generation"] == 1

    vs.pair(vs.projects(tmp_path), 1, scene.at(hour=9))  # restart: a brand-new process collects
    second = document(cli("metrics", "collect", "--json"))["data"]
    assert second["data"]["generation"] == 2 and second["data"]["epoch"] == first["data"]["epoch"]

    server = subprocess.Popen(
        [str(BINARY), "metrics", "serve", "--port", "0"],
        env=vs.env_for(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            owner = inspect_lease(PROFILE)
            if owner is not None and owner.pid == server.pid and owner.alive:
                break
            assert server.poll() is None, server.communicate()
            time.sleep(0.05)
        else:
            raise AssertionError("serve never took the lease")
        while snapshots.current_generation(PROFILE) in (None, 2):  # its first cycle publishes 3
            assert time.monotonic() < deadline and server.poll() is None
            time.sleep(0.05)
        generation = snapshots.current_generation(PROFILE)
        assert generation is not None and generation >= 3
        # a manual collect while serve runs gets already_running and changes nothing
        blocked = document(cli("metrics", "collect", "--json"))["data"]
        assert blocked["state"] == "already_running"
        assert blocked["data"]["owner"]["pid"] == server.pid
        assert blocked["data"]["owner"]["kind"] == "serve"
        assert savings(tmp_path)["provenance"]["generation"] >= 3  # readers keep reading throughout
        server.send_signal(signal.SIGTERM)
        out, err = server.communicate(timeout=120)
    finally:
        if server.poll() is None:
            server.kill()
            server.communicate()
    assert server.returncode == 0, err[-2000:]
    assert "Worker stopped." in out
    held = inspect_lease(PROFILE)
    assert held is None or held.pid is None  # the lease record is gone after the clean stop
    after = document(cli("metrics", "collect", "--json"))["data"]
    assert after["state"] == "published"  # and the next collect simply works
    assert snapshots.generations(PROFILE) == list(range(1, after["data"]["generation"] + 1))
    assert os.path.isdir(vs.accounting_dir()) and not json.dumps(after).count("/private/")
