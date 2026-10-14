"""Exclusive per-profile worker lease (task 9.1): real flock, real processes, real files.

The kernel's ``flock`` is the only authority on who holds the lease; the JSON written
into the lock file is a report for humans and ``already_running`` output. These cases
therefore use real subprocesses (including ``SIGKILL`` and descriptor inheritance) and
forge or corrupt the metadata of live and dead holders to prove it never decides.
"""

from __future__ import annotations

import gc
import json
import os
import re
import select
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ssgrep.accounting import lease as lease_module, storage
from ssgrep.accounting.lease import (
    AlreadyRunning,
    Lease,
    LeaseInfo,
    inspect_lease,
    process_start_token,
)

PROFILE = "personal-local"
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
PS_TOKEN = re.compile(r"[A-Z][a-z]{2} [A-Z][a-z]{2} +\d{1,2} \d\d:\d\d:\d\d \d{4}")

HOLDER = textwrap.dedent(
    """
    import os, subprocess, sys, time
    from ssgrep.accounting.lease import Lease

    profile, kind, mode = sys.argv[1:4]
    lease = Lease.acquire(profile, kind)
    extra = "-"
    if mode == "spawn":  # a long-lived helper started the way any worker would start one
        sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
        extra = str(subprocess.Popen(sleeper, close_fds=False).pid)
    elif mode == "fork":  # a forked (never exec'd) child that keeps the open file description
        child = os.fork()
        if child == 0:
            time.sleep(120)
            os._exit(0)
        extra = str(child)
    print("ready", os.getpid(), extra, flush=True)
    if sys.stdin.readline() == "hard\\n":
        os._exit(0)  # no release, no finalizers, no atexit
    """
)

RACER = textwrap.dedent(
    """
    import os, sys, time
    from ssgrep.accounting.lease import AlreadyRunning, Lease

    go, profile = sys.argv[1:3]
    print("ready", flush=True)
    while not os.path.exists(go):
        time.sleep(0.001)
    try:
        lease = Lease.acquire(profile, "collect", retry_seconds=0.5)
    except AlreadyRunning as refused:
        print("lost", refused.info.pid, flush=True)
    else:
        print("won", os.getpid(), flush=True)
        time.sleep(2)
        lease.release()
    """
)

INHERITED_DESCRIPTOR_HOLDER = textwrap.dedent(
    """
    import os, sys
    from ssgrep.accounting.lease import AlreadyRunning, Lease

    profile, descriptor = sys.argv[1], int(sys.argv[2])
    inherited = os.fstat(descriptor)
    try:
        Lease.acquire(profile, "serve", retry_seconds=0).release()
    except AlreadyRunning as refused:
        print("ready", inherited.st_dev, inherited.st_ino, refused.info.pid, flush=True)
    else:
        raise RuntimeError("inherited descriptor did not exclude a fresh acquisition")
    if sys.stdin.readline() != "acquire\\n":
        raise RuntimeError("missing acquire command")
    with Lease.acquire(profile, "serve", retry_seconds=0):
        print("ready", os.getpid(), flush=True)
        sys.stdin.readline()
    """
)


@dataclass
class Holder:
    process: subprocess.Popen[str]
    pid: int
    extra_pid: int | None
    tokens: dict[int, str | None]  # start token per pid, taken while it was alive

    def kill(self, *, reap: bool = True) -> None:
        os.kill(self.pid, signal.SIGKILL)
        if reap:
            self.process.wait(timeout=10)

    def exit(self, *, hard: bool) -> None:
        """The script ends by itself (``os._exit`` when ``hard``) without calling release."""
        assert self.process.stdin is not None
        self.process.stdin.write("hard\n" if hard else "soft\n")
        self.process.stdin.flush()
        assert self.process.wait(timeout=10) == 0


def _ready(process: subprocess.Popen[str]) -> list[str]:
    assert process.stdout is not None and process.stderr is not None
    ready, _, _ = select.select([process.stdout], [], [], 30)
    line = process.stdout.readline() if ready else ""
    if not line.startswith("ready"):
        process.kill()
        pytest.fail(f"child never became ready: {line!r} {process.stderr.read()!r}")
    return str(line).split()


@pytest.fixture
def start_holder() -> Iterator[Callable[..., Holder]]:
    holders: list[Holder] = []

    def start(profile: str = PROFILE, kind: str = "serve", mode: str = "hold") -> Holder:
        process = subprocess.Popen(
            [sys.executable, "-c", HOLDER, profile, kind, mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _, pid, extra = _ready(process)
        pids = [int(pid)] + ([] if extra == "-" else [int(extra)])
        holder = Holder(
            process, int(pid), None if extra == "-" else int(extra),
            {p: process_start_token(p) for p in pids},
        )  # fmt: skip
        holders.append(holder)
        return holder

    yield start
    for holder in holders:
        for pid, token in holder.tokens.items():
            if token is not None and process_start_token(pid) == token:  # never a recycled pid
                with suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
        holder.process.wait(timeout=10)
        for stream in (holder.process.stdin, holder.process.stdout, holder.process.stderr):
            if stream is not None:
                with suppress(OSError):
                    stream.close()


def lease_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "worker" / "lease.lock"


def write_lease_file(tmp_path: Path, content: bytes | dict[str, object]) -> Path:
    """Put raw bytes (or a metadata record) into the lock file with no lock held."""
    os.close(storage.open_profile_dir(PROFILE, "worker"))
    path = lease_path(tmp_path)
    data = content if isinstance(content, bytes) else json.dumps(content).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    return path


def overwrite_in_place(path: Path, data: bytes) -> None:
    """Corrupt the metadata of a lease somebody else is holding (no lock is taken)."""
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, data, 0)
    finally:
        os.close(descriptor)


def dead_owner() -> tuple[int, str]:
    """(pid, start token) of a process that really existed and is now gone and reaped."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    token = process_start_token(child.pid)
    assert token
    child.kill()
    child.wait(timeout=10)
    return child.pid, token


def record(pid: int, token: str | None, kind: str = "serve") -> dict[str, object]:
    return {"pid": pid, "process_start": token, "kind": kind, "acquired_at": "2026-11-01T07:00:00Z"}


def tree(root: Path) -> list[tuple[str, int]]:
    if not root.exists():
        return []
    return sorted((str(p.relative_to(root)), p.lstat().st_mode) for p in root.rglob("*"))


def wait_until(predicate: Callable[[], bool], seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def acquirable() -> bool:
    try:
        Lease.acquire(PROFILE, "collect", retry_seconds=0).release()
    except AlreadyRunning:
        return False
    return True


# --- two real processes ---------------------------------------------------------------


def test_second_process_is_refused_with_owner_info_then_recovers_after_sigkill(
    start_holder: Callable[..., Holder], tmp_path: Path
) -> None:
    holder = start_holder(kind="serve")

    with pytest.raises(AlreadyRunning) as refused:
        Lease.acquire(PROFILE, "collect")
    owner = refused.value.info
    assert (owner.pid, owner.kind, owner.alive) == (holder.pid, "serve", True)
    assert owner.process_start and owner.process_start == process_start_token(holder.pid)
    assert owner.acquired_at and STAMP.fullmatch(owner.acquired_at)
    assert inspect_lease(PROFILE) == owner
    on_disk = json.loads(lease_path(tmp_path).read_text())  # the real bytes, not just the API
    assert on_disk == {
        "pid": holder.pid,
        "process_start": owner.process_start,
        "kind": "serve",
        "acquired_at": owner.acquired_at,
    }

    holder.kill()  # SIGKILL: no release, no cleanup, no atexit

    with Lease.acquire(PROFILE, "collect") as taken:
        assert taken.recovered_from == replace(owner, alive=False)
        mine = inspect_lease(PROFILE)
        assert mine is not None
        assert mine == LeaseInfo(
            os.getpid(), process_start_token(os.getpid()), "collect", mine.acquired_at, alive=True
        )


def test_exit_without_release_or_finalizers_is_recovered_with_the_owner_metadata(
    start_holder: Callable[..., Holder],
) -> None:
    holder = start_holder(kind="serve")
    with pytest.raises(AlreadyRunning):
        Lease.acquire(PROFILE, "collect")
    owner = inspect_lease(PROFILE)
    assert owner is not None and owner.pid == holder.pid

    holder.exit(hard=True)

    with Lease.acquire(PROFILE, "collect") as taken:
        assert taken.recovered_from == replace(owner, alive=False)


def test_orderly_interpreter_exit_frees_the_lease(start_holder: Callable[..., Holder]) -> None:
    holder = start_holder(kind="serve")
    with pytest.raises(AlreadyRunning):
        Lease.acquire(PROFILE, "collect")

    holder.exit(hard=False)

    assert acquirable()


def test_killed_but_unreaped_owner_is_dead_not_running(
    start_holder: Callable[..., Holder],
) -> None:
    holder = start_holder()
    info = inspect_lease(PROFILE)
    assert info is not None and info.alive is True

    holder.kill(reap=False)  # zombie: the pid still exists, the process does not

    assert wait_until(lambda: getattr(inspect_lease(PROFILE), "alive", True) is False)
    assert process_start_token(holder.pid) is None
    with Lease.acquire(PROFILE, "collect") as taken:
        assert taken.recovered_from is not None and taken.recovered_from.pid == holder.pid
    holder.process.wait(timeout=10)


def test_racing_processes_elect_exactly_one_holder(tmp_path: Path) -> None:
    go = tmp_path / "go"
    racers = [
        subprocess.Popen(
            [sys.executable, "-c", RACER, str(go), PROFILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    try:
        for racer in racers:
            assert racer.stdout is not None
            assert racer.stdout.readline().strip() == "ready"
        go.touch()
        results = []
        for racer in racers:
            out, err = racer.communicate(timeout=60)
            assert racer.returncode == 0, err
            results.append(out.split())
    finally:
        for racer in racers:
            racer.kill()
            racer.wait()

    winners = [r for r in results if r[0] == "won"]
    losers = [r for r in results if r[0] == "lost"]
    assert len(winners) == 1 and len(losers) == 5
    assert {r[1] for r in losers} == {winners[0][1]}  # every loser was told who really holds it


def test_failed_attempt_retries_until_a_racing_holder_lets_go() -> None:
    first = Lease.acquire(PROFILE, "collect")
    started = time.monotonic()
    threading.Timer(0.2, first.release).start()

    with pytest.raises(AlreadyRunning):  # no patience: refused at once
        Lease.acquire(PROFILE, "collect", retry_seconds=0)
    patient = Lease.acquire(PROFILE, "collect", retry_seconds=10)  # absorbs the racing release

    assert time.monotonic() - started >= 0.15
    patient.release()


# --- same process ---------------------------------------------------------------------


def test_same_process_second_acquire_refused_and_release_is_idempotent_and_safe() -> None:
    first = Lease.acquire(PROFILE, "collect")
    with pytest.raises(AlreadyRunning) as refused:
        Lease.acquire(PROFILE, "serve")
    assert (refused.value.info.pid, refused.value.info.alive) == (os.getpid(), True)
    assert refused.value.info.kind == "collect"
    assert first.profile == PROFILE and first.kind == "collect" and first.recovered_from is None

    first.release()
    first.release()  # idempotent

    second = Lease.acquire(PROFILE, "serve")
    first.release()  # a stale release must not drop the new holder's lock
    with pytest.raises(AlreadyRunning):
        Lease.acquire(PROFILE, "collect")
    second.release()
    Lease.acquire(PROFILE, "collect").release()


def test_concurrent_releases_close_the_descriptor_exactly_once() -> None:
    for _ in range(50):
        held = Lease.acquire(PROFILE, "collect")
        gate = threading.Barrier(16)
        errors: list[BaseException] = []

        def release(
            held: Lease = held,
            gate: threading.Barrier = gate,
            errors: list[BaseException] = errors,
        ) -> None:
            gate.wait()
            try:
                held.release()
            except BaseException as error:  # noqa: BLE001 - the test reports whatever escaped
                errors.append(error)

        threads = [threading.Thread(target=release) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert errors == []
        assert acquirable()


def test_context_manager_releases_on_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with Lease.acquire(PROFILE, "collect") as held:
            assert isinstance(held, Lease)
            with pytest.raises(AlreadyRunning):
                Lease.acquire(PROFILE, "collect")
            raise RuntimeError("boom")

    assert acquirable()


def test_orderly_release_leaves_nothing_to_recover(tmp_path: Path) -> None:
    Lease.acquire(PROFILE, "collect").release()

    assert lease_path(tmp_path).read_bytes() == b""
    assert inspect_lease(PROFILE) == LeaseInfo(None, None, None, None, alive=False)
    with Lease.acquire(PROFILE, "serve") as again:
        assert again.recovered_from is None


def test_acquired_at_is_the_given_instant_in_utc(tmp_path: Path) -> None:
    moment = datetime(2026, 11, 1, 9, 30, 5, tzinfo=timezone(timedelta(hours=2)))
    with Lease.acquire(PROFILE, "collect", now=moment):
        assert json.loads(lease_path(tmp_path).read_text())["acquired_at"] == "2026-11-01T07:30:05Z"
        info = inspect_lease(PROFILE)
        assert info is not None and info.acquired_at == "2026-11-01T07:30:05Z"
    with Lease.acquire(PROFILE, "collect") as stamped:
        assert stamped.recovered_from is None
        info = inspect_lease(PROFILE)
        assert info is not None and info.acquired_at and STAMP.fullmatch(info.acquired_at)


def test_kind_must_fit_the_record_and_is_checked_before_anything_is_created(
    tmp_path: Path,
) -> None:
    for bad in ("", "k" * 129):
        with pytest.raises(ValueError, match="kind"):
            Lease.acquire(PROFILE, bad)
    assert not (tmp_path / "data").exists()
    with Lease.acquire(PROFILE, "k" * 128):
        info = inspect_lease(PROFILE)
        assert info is not None and info.kind == "k" * 128


def test_dropped_lease_releases_its_flock_and_record() -> None:
    dropped = Lease.acquire(PROFILE, "collect")
    with pytest.raises(AlreadyRunning):
        Lease.acquire(PROFILE, "serve")
    del dropped
    gc.collect()

    with Lease.acquire(PROFILE, "serve") as taken:
        assert taken.recovered_from is None


# --- stale files, forged metadata -------------------------------------------------------


def test_stale_file_of_a_dead_pid_is_recovered_never_reported_running(tmp_path: Path) -> None:
    pid, token = dead_owner()
    write_lease_file(tmp_path, record(pid, token, kind="serve"))

    stale = inspect_lease(PROFILE)
    assert stale == LeaseInfo(pid, token, "serve", "2026-11-01T07:00:00Z", alive=False)
    with Lease.acquire(PROFILE, "collect") as taken:
        assert taken.recovered_from == stale
        current = inspect_lease(PROFILE)
        assert current is not None and current.pid == os.getpid() and current.kind == "collect"


def test_live_pid_with_a_different_start_token_is_pid_reuse_not_a_live_owner(
    tmp_path: Path,
) -> None:
    write_lease_file(tmp_path, record(os.getpid(), "Thu Jan  1 00:00:00 1970"))

    reused = inspect_lease(PROFILE)
    assert reused is not None and reused.pid == os.getpid() and reused.alive is False
    with Lease.acquire(PROFILE, "collect") as taken:
        assert taken.recovered_from == reused


def test_live_pid_without_a_recorded_start_token_cannot_be_proven_alive(tmp_path: Path) -> None:
    write_lease_file(tmp_path, record(os.getpid(), None))

    info = inspect_lease(PROFILE)
    assert info is not None and info.pid == os.getpid() and info.alive is False


def test_flock_not_pid_liveness_decides_a_free_lock_is_taken_even_if_metadata_looks_live(
    tmp_path: Path,
) -> None:
    token = process_start_token(os.getpid())
    write_lease_file(tmp_path, record(os.getpid(), token))
    assert inspect_lease(PROFILE) == LeaseInfo(
        os.getpid(), token, "serve", "2026-11-01T07:00:00Z", alive=True
    )

    with Lease.acquire(PROFILE, "collect") as taken:  # nobody holds the flock
        assert taken.recovered_from is not None and taken.recovered_from.alive is True


def _forged() -> dict[str, Callable[[], bytes]]:
    return {
        "empty": lambda: b"",
        "binary": lambda: b"\xff\xfe\x00 not json {{{ \x01",
        "oversized": lambda: b"x" * 100_000,
        "deeply-nested": lambda: b"[" * 4000,
        "array": lambda: b"[1, 2, 3]",
        "float-pid": lambda: b'{"pid": 1.5, "kind": "serve"}',
        "bool-pid": lambda: b'{"pid": true, "kind": "serve"}',
        "negative-pid": lambda: b'{"pid": -4, "kind": "serve"}',
        "huge-pid": lambda: json.dumps(record(2**40, "x")).encode(),
        "dead-pid": lambda: json.dumps(record(*dead_owner())).encode(),
        "our-own-live-pid": lambda: json.dumps(
            record(os.getpid(), process_start_token(os.getpid()))
        ).encode(),
    }


@pytest.mark.parametrize("name", sorted(_forged()))
def test_live_holder_is_never_taken_over_whatever_its_metadata_says(
    name: str, start_holder: Callable[..., Holder], tmp_path: Path
) -> None:
    holder = start_holder(kind="serve")
    overwrite_in_place(lease_path(tmp_path), _forged()[name]())

    with pytest.raises(AlreadyRunning) as refused:
        Lease.acquire(PROFILE, "collect")
    assert refused.value.info.pid != holder.pid  # the report is garbage, the refusal is not
    assert inspect_lease(PROFILE) is not None
    assert holder.process.poll() is None

    holder.kill()
    with Lease.acquire(PROFILE, "collect") as taken:  # the moment the flock frees, it is ours
        assert (taken.recovered_from is None) == (name == "empty")


def test_dead_recorded_owner_does_not_unlock_a_lease_a_forked_child_still_holds(
    start_holder: Callable[..., Holder],
) -> None:
    holder = start_holder(mode="fork")
    assert holder.extra_pid is not None
    holder.kill()  # the recorded owner dies; its forked child keeps the open file description

    with pytest.raises(AlreadyRunning) as refused:
        Lease.acquire(PROFILE, "collect")
    assert refused.value.info.pid == holder.pid
    assert refused.value.info.alive is False  # recorded owner is dead, the lease is not free

    os.kill(holder.extra_pid, signal.SIGKILL)
    assert wait_until(acquirable)


# --- descriptor inheritance -------------------------------------------------------------


def test_helper_started_by_the_holder_does_not_keep_the_lease_after_a_crash(
    start_holder: Callable[..., Holder],
) -> None:
    holder = start_holder(mode="spawn")
    assert holder.extra_pid is not None
    holder.kill()
    os.kill(holder.extra_pid, 0)  # the helper really outlived the holder

    with Lease.acquire(PROFILE, "collect", retry_seconds=0):
        pass


def test_release_with_a_live_subprocess_leaves_the_lease_immediately_acquirable() -> None:
    held = Lease.acquire(PROFILE, "collect")
    helper = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
        close_fds=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        held.release()  # do not wait for exec/startup to finish before releasing
        with Lease.acquire(PROFILE, "collect", retry_seconds=0):
            _ready(helper)
            assert helper.poll() is None
    finally:
        held.release()
        helper.kill()
        helper.wait(timeout=10)
        for stream in (helper.stdin, helper.stdout, helper.stderr):
            assert stream is not None
            stream.close()


def test_release_unlocks_while_a_duplicate_descriptor_remains_open(tmp_path: Path) -> None:
    held = Lease.acquire(PROFILE, "collect")
    descriptor = held._descriptor
    assert descriptor is not None and not os.get_inheritable(descriptor)
    duplicate = os.dup(descriptor)
    try:
        with pytest.raises(AlreadyRunning):
            Lease.acquire(PROFILE, "serve", retry_seconds=0)
        held.release()

        with pytest.raises(OSError):
            os.fstat(descriptor)  # release must still close its own descriptor
        assert stat.S_ISREG(os.fstat(duplicate).st_mode)  # the shared description lives on
        assert lease_path(tmp_path).read_bytes() == b""
        with Lease.acquire(PROFILE, "serve", retry_seconds=0) as next_owner:
            assert next_owner.recovered_from is None
            held.release()  # a stale release cannot unlock the new open file description
            with pytest.raises(AlreadyRunning):
                Lease.acquire(PROFILE, "collect", retry_seconds=0)
    finally:
        held.release()
        os.close(duplicate)


def test_release_unlocks_a_descriptor_inherited_by_a_live_subprocess(tmp_path: Path) -> None:
    held = Lease.acquire(PROFILE, "collect")
    descriptor = held._descriptor
    assert descriptor is not None
    identity = os.fstat(descriptor)
    helper = subprocess.Popen(
        [sys.executable, "-c", INHERITED_DESCRIPTOR_HOLDER, PROFILE, str(descriptor)],
        pass_fds=(descriptor,),  # deterministic shared-description control, despite CLOEXEC
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, device, inode, owner = _ready(helper)
        assert (int(device), int(inode)) == (identity.st_dev, identity.st_ino)
        assert int(owner) == os.getpid()  # the active parent's lease excluded another process
        held.release()
        assert lease_path(tmp_path).read_bytes() == b""
        with Lease.acquire(PROFILE, "collect", retry_seconds=0) as next_owner:
            assert next_owner.recovered_from is None
            assert helper.poll() is None

        assert helper.stdin is not None
        helper.stdin.write("acquire\n")
        helper.stdin.flush()
        _, owner = _ready(helper)  # another process takes a fresh lease while retaining our fd
        assert int(owner) == helper.pid
        held.release()
        with pytest.raises(AlreadyRunning) as refused:
            Lease.acquire(PROFILE, "collect", retry_seconds=0)
        assert refused.value.info.pid == helper.pid
        helper.stdin.write("exit\n")
        helper.stdin.flush()
        assert helper.wait(timeout=10) == 0
        assert acquirable()
    finally:
        held.release()
        if helper.poll() is None:
            helper.kill()
        helper.wait(timeout=10)
        for stream in (helper.stdin, helper.stdout, helper.stderr):
            assert stream is not None
            stream.close()


def test_release_in_a_forked_copy_cannot_wipe_or_drop_the_parents_lease(tmp_path: Path) -> None:
    held = Lease.acquire(PROFILE, "collect")
    before = lease_path(tmp_path).read_bytes()
    child = os.fork()
    if child == 0:
        try:
            held.release()
            held.__del__()  # the copy's finalizer at child exit must be just as harmless
        except BaseException:  # noqa: BLE001 - report child-side release/finalizer failures
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    assert lease_path(tmp_path).read_bytes() == before
    with pytest.raises(AlreadyRunning) as refused:
        Lease.acquire(PROFILE, "serve")
    assert refused.value.info.pid == os.getpid()
    held.release()
    assert acquirable()


# --- read-only inspection -----------------------------------------------------------------


def test_inspect_never_creates_anything_and_is_none_when_absent(tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert inspect_lease(PROFILE) is None
    assert not data.exists()  # not even the data root

    os.close(storage.open_profile_dir(PROFILE))
    before = tree(data)
    assert inspect_lease(PROFILE) is None
    assert tree(data) == before

    os.close(storage.open_profile_dir(PROFILE, "worker"))
    before = tree(data)
    assert inspect_lease(PROFILE) is None
    assert tree(data) == before


def test_inspect_leaves_an_existing_lease_file_untouched(tmp_path: Path) -> None:
    pid, token = dead_owner()
    path = write_lease_file(tmp_path, record(pid, token))
    before = (path.read_bytes(), path.stat().st_mtime_ns, tree(tmp_path / "data"))

    assert inspect_lease(PROFILE) is not None
    assert inspect_lease(PROFILE) is not None

    assert (path.read_bytes(), path.stat().st_mtime_ns, tree(tmp_path / "data")) == before


def test_inspect_reports_live_and_dead_holders_without_taking_the_flock(
    start_holder: Callable[..., Holder],
) -> None:
    holder = start_holder(kind="serve")
    live = inspect_lease(PROFILE)
    assert live is not None and (live.pid, live.kind, live.alive) == (holder.pid, "serve", True)
    with pytest.raises(AlreadyRunning):  # inspecting did not disturb the holder
        Lease.acquire(PROFILE, "collect")

    holder.kill()
    assert inspect_lease(PROFILE) == replace(live, alive=False)
    assert acquirable()  # and it did not take the lock itself


# --- storage safety -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile", ["../etc", "a/b", "", "..", ".hidden", "UPPER", "x" * 40, "a\x00b", "/abs"]
)
def test_profile_ids_are_identifiers_never_paths(profile: str, tmp_path: Path) -> None:
    with pytest.raises(storage.UnsafeStorageError):
        inspect_lease(profile)
    with pytest.raises(storage.UnsafeStorageError):
        Lease.acquire(profile, "collect")
    assert not (tmp_path / "data").exists()


def test_layout_and_private_modes(tmp_path: Path) -> None:
    root = tmp_path / "data" / "accounting"
    with Lease.acquire(PROFILE, "collect"):
        for directory in (root, root / PROFILE, root / PROFILE / "worker"):
            assert stat.S_IMODE(directory.lstat().st_mode) == 0o700
        assert stat.S_IMODE(lease_path(tmp_path).lstat().st_mode) == 0o600
        assert sorted(p.name for p in (root / PROFILE).iterdir()) == ["worker"]
        assert sorted(p.name for p in (root / PROFILE / "worker").iterdir()) == ["lease.lock"]
        assert not (tmp_path / "data" / "notes").exists()  # never under the indexed notes tree


def test_permissive_existing_lock_file_is_tightened_to_0600(tmp_path: Path) -> None:
    path = write_lease_file(tmp_path, b"")
    os.chmod(path, 0o644)

    with Lease.acquire(PROFILE, "collect"):
        assert stat.S_IMODE(path.lstat().st_mode) == 0o600


def test_symlinked_lock_file_is_refused_and_target_is_never_written(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    os.close(storage.open_profile_dir(PROFILE, "worker"))
    lease_path(tmp_path).symlink_to(victim)

    with pytest.raises(storage.UnsafeStorageError):
        Lease.acquire(PROFILE, "collect")
    with pytest.raises(storage.UnsafeStorageError):
        inspect_lease(PROFILE)
    assert victim.read_text() == "precious"


def test_symlinked_worker_dir_is_refused_and_target_untouched(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    os.close(storage.open_profile_dir(PROFILE))
    (tmp_path / "data" / "accounting" / PROFILE / "worker").symlink_to(victim)

    with pytest.raises(storage.UnsafeStorageError):
        Lease.acquire(PROFILE, "collect")
    with pytest.raises(storage.UnsafeStorageError):
        inspect_lease(PROFILE)
    assert list(victim.iterdir()) == []


def test_non_regular_lock_file_is_refused_without_hanging(tmp_path: Path) -> None:
    os.close(storage.open_profile_dir(PROFILE, "worker"))
    os.mkfifo(lease_path(tmp_path), 0o600)

    with pytest.raises(storage.UnsafeStorageError):
        Lease.acquire(PROFILE, "collect")
    with pytest.raises(storage.UnsafeStorageError):
        inspect_lease(PROFILE)


# --- process_start_token ------------------------------------------------------------------


def test_process_start_token_is_stable_for_a_live_pid_and_none_for_a_missing_one() -> None:
    first = process_start_token(os.getpid())
    assert first and first == process_start_token(os.getpid())

    pid, _ = dead_owner()
    assert process_start_token(pid) is None
    for impossible in (0, -1, 2**31):
        assert process_start_token(impossible) is None


def test_start_token_follows_a_process_for_its_lifetime_and_ends_with_it() -> None:
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        token = process_start_token(helper.pid)
        assert token and token != process_start_token(os.getpid())
        assert token == process_start_token(helper.pid)
    finally:
        helper.kill()
        helper.wait()
    assert process_start_token(helper.pid) is None


def test_ps_fallback_is_used_without_procfs_and_reads_start_time_and_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(lease_module, "_PROC_ROOT", tmp_path / "no-proc-here")

    token = process_start_token(os.getpid())
    assert token is not None and PS_TOKEN.fullmatch(token)
    assert token == process_start_token(os.getpid())
    assert process_start_token(dead_owner()[0]) is None
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert wait_until(lambda: _is_zombie(zombie.pid))  # never poll(): that would reap it
        assert process_start_token(zombie.pid) is None  # exited but not yet reaped
    finally:
        zombie.wait()


def _is_zombie(pid: int) -> bool:
    """True while ``pid`` is exited-but-unreaped: the kernel lists it with state Z."""
    done = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return done.stdout.strip().startswith("Z")


def _fake_stat(proc: Path, pid: int, comm: str, state: str, starttime: int) -> None:
    """A /proc/<pid>/stat with the kernel's layout: fields 4..52 follow ``state``."""
    fields = ["0"] * 49
    fields[22 - 4] = str(starttime)
    (proc / str(pid)).mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) {state} {' '.join(fields)}\n")


@pytest.fixture
def fake_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    proc = tmp_path / "proc"
    (proc / "sys" / "kernel" / "random").mkdir(parents=True)
    (proc / "sys" / "kernel" / "random" / "boot_id").write_text("boot-abc\n")
    monkeypatch.setattr(lease_module, "_PROC_ROOT", proc)
    return proc


def test_proc_start_token_uses_starttime_and_boot_id_and_survives_hostile_comm(
    fake_proc: Path,
) -> None:
    _fake_stat(fake_proc, 4242, "ss grep) (x S 1 2", "S", 987654)
    _fake_stat(fake_proc, 4243, "plain", "R", 987655)

    assert process_start_token(4242) == "boot-abc:987654"
    assert process_start_token(4243) == "boot-abc:987655"
    _fake_stat(fake_proc, 4242, "ss grep) (x S 1 2", "S", 111)  # same pid, newer process
    assert process_start_token(4242) == "boot-abc:111"


def test_proc_zombies_and_missing_pids_are_not_running_and_proc_is_authoritative(
    fake_proc: Path,
) -> None:
    _fake_stat(fake_proc, 4242, "worker", "Z", 5)
    _fake_stat(fake_proc, 4243, "worker", "X", 5)

    assert process_start_token(4242) is None
    assert process_start_token(4243) is None
    assert process_start_token(os.getpid()) is None  # absent from /proc => no such process


def test_pid_reuse_is_detected_through_the_proc_start_time(fake_proc: Path, tmp_path: Path) -> None:
    _fake_stat(fake_proc, 4242, "ssgrep", "S", 987654)
    write_lease_file(tmp_path, record(4242, "boot-abc:987654"))
    same = inspect_lease(PROFILE)
    assert same is not None and same.alive is True

    _fake_stat(fake_proc, 4242, "unrelated", "S", 222)  # the pid now belongs to someone else
    reused = inspect_lease(PROFILE)
    assert reused == replace(same, alive=False)
    with Lease.acquire(PROFILE, "collect") as taken:
        assert taken.recovered_from == reused
