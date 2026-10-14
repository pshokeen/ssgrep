"""Journal identity, concurrency, bounds, and recovery (tasks 3.2, 3.3, 3.7)."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import journal, journal_codec, journal_reader, storage
from ssgrep.accounting.contracts import MAX_EVENT_BYTES, Outcome, Phase
from tests.accounting.conftest import build_event

PROFILE = "personal-local"


@pytest.fixture(autouse=True)
def _fresh_journal_state():
    journal.close_all()
    yield
    journal.close_all()


def _journal_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "journal"


def _segments(tmp_path: Path) -> list[Path]:
    return sorted(_journal_dir(tmp_path).glob("seg-*.jsonl"))


def _append_execution(profile: str = PROFILE) -> str:
    execution_id = journal.next_execution_id()
    for phase in (Phase.STARTED, Phase.COMPLETED):
        journal.append(profile, build_event(execution_id=execution_id, phase=phase))
    return execution_id


WORKER = textwrap.dedent(
    """
    import os, sys, threading
    from ssgrep.accounting import journal
    from ssgrep.accounting.contracts import Phase
    from tests.accounting.conftest import build_event

    def run(count):
        for _ in range(count):
            eid = journal.next_execution_id()
            for phase in (Phase.STARTED, Phase.COMPLETED):
                journal.append("personal-local", build_event(execution_id=eid, phase=phase))

    count = int(sys.argv[1])
    mode = sys.argv[2]
    if mode == "threads":
        run(1)  # open the segment before the threads (and the fork) exist
        pid = os.fork()
        if pid == 0:
            run(count)
            os._exit(0)
        threads = [threading.Thread(target=run, args=(count,)) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        run(count)
        os.waitpid(pid, 0)
    else:
        run(count)
    """
)


def _spawn(count: int, mode: str) -> subprocess.Popen[bytes]:
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root)}
    return subprocess.Popen([sys.executable, "-c", WORKER, str(count), mode], env=env, cwd=root)


def test_execution_ids_are_unique_across_processes_threads_and_a_fork(tmp_path: Path) -> None:
    count = 40
    processes = [_spawn(count, "threads"), _spawn(count, "plain"), _spawn(count, "plain")]
    assert [p.wait(timeout=60) for p in processes] == [0, 0, 0]

    read = journal_reader.read_journal(PROFILE)

    # threads process: 1 warmup + 8 threads*count + count in the main thread + count in the fork
    expected = (1 + 8 * count + count + count) + count + count
    assert len(read.events) == expected * 2  # started + completed each
    ids = {e.execution_id for e in read.events}
    assert len(ids) == expected
    assert read.damage == [] and read.duplicates == 0 and read.conflicts == []
    for execution_id in ids:
        phases = sorted(e.phase for e in read.events if e.execution_id == execution_id)
        assert phases == [Phase.COMPLETED, Phase.STARTED]
    # One private segment per process (parent, its forked child, two others): no shared append file.
    assert len(_segments(tmp_path)) == 4
    assert len({e.execution_id.split("-")[0] for e in read.events}) == 4


def test_forked_child_gets_new_nonce_sequence_and_descriptor(tmp_path: Path) -> None:
    parent_id = _append_execution()
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        try:
            os.close(read_end)
            child_id = _append_execution()
            os.write(write_end, child_id.encode())
        finally:
            os._exit(0)
    os.close(write_end)
    child_id = os.read(read_end, 200).decode()
    os.close(read_end)
    os.waitpid(pid, 0)
    parent_after = _append_execution()

    assert child_id.split("-")[0] != parent_id.split("-")[0]
    assert child_id.endswith("-1")  # sequence restarts with the new nonce
    parent_nonce, parent_seq = parent_id.rsplit("-", 1)
    # The parent's own counter continued; the child's inherited nothing from it.
    assert parent_after == f"{parent_nonce}-{int(parent_seq) + 1}"
    assert len(_segments(tmp_path)) == 2
    read = journal_reader.read_journal(PROFILE)
    assert {e.execution_id for e in read.events} == {parent_id, child_id, parent_after}


def test_segments_and_directory_are_private(tmp_path: Path) -> None:
    _append_execution()
    (segment,) = _segments(tmp_path)
    assert oct(os.stat(segment).st_mode & 0o777) == "0o600"
    assert oct(os.stat(_journal_dir(tmp_path)).st_mode & 0o777) == "0o700"


def test_event_bound_keeps_identity_and_outcome_and_marks_omitted_refs() -> None:
    refs = tuple(f"claude:session-{i:05d}:episode:{'x' * 40}" for i in range(4000))
    event = build_event(refs=refs, ranks_scores=tuple((i, 900) for i in range(4000)))

    line = journal_codec.serialize_event(event)

    assert len(line) <= MAX_EVENT_BYTES and line.endswith(b"\n")
    parsed = journal_codec.parse_event(line[:-1])
    assert parsed.refs_omitted is True
    assert 0 < len(parsed.refs) < len(refs)
    assert parsed.refs == refs[: len(parsed.refs)]
    assert len(parsed.ranks_scores) == len(parsed.refs)
    assert (parsed.execution_id, parsed.phase, parsed.outcome) == (
        event.execution_id,
        event.phase,
        Outcome.OK,
    )
    assert (parsed.fingerprint, parsed.operation, parsed.ts_utc) == (
        event.fingerprint,
        event.operation,
        event.ts_utc,
    )
    # The largest fitting prefix is kept: one more reference would not fit.
    longer = journal_codec._line(journal_codec._trimmed(event, len(parsed.refs) + 1))
    assert len(longer) > MAX_EVENT_BYTES


def test_event_within_bound_is_unmodified() -> None:
    event = build_event()
    parsed = journal_codec.parse_event(journal_codec.serialize_event(event)[:-1])
    assert parsed == event and parsed.refs_omitted is False


def test_oversized_identity_is_rejected_never_silently_truncated(tmp_path: Path) -> None:
    event = build_event(cwd="/" + "d" * (MAX_EVENT_BYTES + 10))
    with pytest.raises(journal_codec.EventTooLarge):
        journal_codec.serialize_event(event)
    with pytest.raises(journal_codec.EventTooLarge):
        journal.append(PROFILE, event)
    assert not _segments(tmp_path) or _segments(tmp_path)[0].stat().st_size == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema=99),
        lambda d: d.pop("execution_id"),
        lambda d: d.update(execution_id="../../evil"),
        lambda d: d.update(phase="bogus"),
        lambda d: d.update(phase="completed", outcome=None),
        lambda d: d.update(limit="ten"),
        lambda d: d.update(refs=[1, 2]),
        lambda d: d.update(ranks_scores=[[1, 0.5]]),
        lambda d: d.update(refs_omitted="yes"),
    ],
)
def test_malformed_but_parseable_records_are_rejected(mutation) -> None:
    document = json.loads(journal_codec.serialize_event(build_event()))
    mutation(document)
    with pytest.raises(journal_codec.MalformedEvent):
        journal_codec.parse_event(json.dumps(document).encode())


def test_complete_records_survive_a_truncated_tail_and_are_not_double_counted(
    tmp_path: Path,
) -> None:
    first = _append_execution()
    second = _append_execution()
    journal.close_all()
    (segment,) = _segments(tmp_path)
    whole = segment.read_bytes()
    segment.write_bytes(whole[:-17])  # the final completed record is now a fragment

    read = journal_reader.read_journal(PROFILE)

    keys = {(e.execution_id, str(e.phase)) for e in read.events}
    assert keys == {
        (first, "started"),
        (first, "completed"),
        (second, "started"),
    }
    (tail,) = read.incomplete_tails
    assert tail.kind == "incomplete_tail" and tail.segment == segment.name
    assert read.quarantined == []
    # Reading again (a restart) reports the same evidence; nothing was admitted or removed.
    again = journal_reader.read_journal(PROFILE)
    assert [(e.execution_id, str(e.phase)) for e in again.events] == [
        (e.execution_id, str(e.phase)) for e in read.events
    ]
    assert segment.read_bytes() == whole[:-17]


def test_short_write_leaves_a_fragment_that_cannot_fuse_with_the_next_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = _append_execution()
    real_write = os.write
    calls = {"n": 0}

    def short_then_fail(fd: int, data: memoryview) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data)[:60])
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "write", short_then_fail)
        broken_id = journal.next_execution_id()
        with pytest.raises(OSError):
            journal.append(PROFILE, build_event(execution_id=broken_id))
    after = _append_execution()  # same process, same segment, next record

    read = journal_reader.read_journal(PROFILE)

    assert {e.execution_id for e in read.events} == {good, after}
    assert broken_id not in {e.execution_id for e in read.events}
    (damage,) = read.quarantined
    assert damage.reason and damage.length == 60 + 1  # fragment bytes + the separating newline
    # The record after the fragment is intact: it did not fuse into the fragment.
    assert {(e.execution_id, str(e.phase)) for e in read.events} >= {
        (after, "started"),
        (after, "completed"),
    }
    # Quarantine metadata (never the raw bytes) exists once, and re-reading adds nothing.
    quarantine = tmp_path / "data" / "accounting" / PROFILE / journal_reader.QUARANTINE_DIR
    names = sorted(p.name for p in quarantine.iterdir())
    assert len(names) == 1
    journal_reader.read_journal(PROFILE)
    assert sorted(p.name for p in quarantine.iterdir()) == names
    record = json.loads((quarantine / names[0]).read_text())
    assert set(record) == {"segment", "offset", "length", "digest", "reason"}


def test_real_short_write_under_file_size_limit_is_recovered_by_a_new_process(
    tmp_path: Path,
) -> None:
    code = textwrap.dedent(
        """
        import resource, sys
        from ssgrep.accounting import journal
        from ssgrep.accounting.contracts import Phase
        from tests.accounting.conftest import build_event
        one = len(journal.serialize_event(build_event(execution_id="x", phase=Phase.STARTED)))
        resource.setrlimit(resource.RLIMIT_FSIZE, (one * 2 + one // 2, one * 2 + one // 2))
        failures = 0
        for i in range(4):
            try:
                journal.append("personal-local", build_event(execution_id=f"x-{i}"))
            except OSError:
                failures += 1
        print(failures)
        """
    )
    root = Path(__file__).resolve().parents[2]
    env: dict[str, str] = {**os.environ, "PYTHONPATH": str(root)}
    # The global write limit also prevents coverage's SQLite database from being saved.
    # Measure journal paths in the unrestricted cases, not this storage-pressure child.
    env.pop("COVERAGE_PROCESS_CONFIG", None)
    env.pop("COVERAGE_PROCESS_START", None)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=root, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) >= 1  # the limit really cut writes short

    follow_up = _append_execution()  # a fresh process-segment, as after a restart
    read = journal_reader.read_journal(PROFILE)

    assert {e.execution_id for e in read.events} >= {follow_up}
    accepted = {e.execution_id for e in read.events if e.execution_id.startswith("x-")}
    assert accepted == {"x-0", "x-1"}  # only the records that fit completely
    assert len(read.incomplete_tails) + len(read.quarantined) == 1


def test_identical_replay_counts_once_and_conflicts_are_deterministic(tmp_path: Path) -> None:
    executed = _append_execution()
    journal.close_all()
    (segment,) = _segments(tmp_path)
    copy = segment.with_name("seg-copy.jsonl")
    copy.write_bytes(segment.read_bytes())
    copy.chmod(0o600)

    replay = journal_reader.read_journal(PROFILE)
    assert len(replay.events) == 2 and replay.duplicates == 2 and replay.conflicts == []
    assert {e.execution_id for e in replay.events} == {executed}

    altered = journal_codec.serialize_event(
        build_event(execution_id=executed, phase=Phase.COMPLETED, outcome=Outcome.ERROR)
    )
    other = segment.with_name("seg-zzz.jsonl")
    other.write_bytes(altered)
    other.chmod(0o600)
    first = journal_reader.read_journal(PROFILE)
    other.rename(segment.with_name("seg-000.jsonl"))
    second = journal_reader.read_journal(PROFILE)

    assert first.conflicts == [(executed, "completed")] == second.conflicts
    assert [e.outcome for e in first.events] == [e.outcome for e in second.events]


def test_oversized_garbage_line_is_skipped_incrementally_and_following_records_survive(
    tmp_path: Path,
) -> None:
    first = _append_execution()
    journal.close_all()
    (segment,) = _segments(tmp_path)
    good_tail = journal_codec.serialize_event(build_event(execution_id="after-1"))
    with open(segment, "ab") as handle:
        handle.write(b"x" * (MAX_EVENT_BYTES * 3) + b"\n" + good_tail)

    read = journal_reader.read_journal(PROFILE)

    assert {e.execution_id for e in read.events} == {first, "after-1"}
    (damage,) = read.quarantined
    assert damage.reason == "line exceeds the event bound"


def test_unsafe_journal_location_is_refused_and_target_untouched(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    root = tmp_path / "data" / "accounting"
    (root / PROFILE).mkdir(parents=True, mode=0o700)
    (root / PROFILE / "journal").symlink_to(victim)

    with pytest.raises(storage.UnsafeStorageError):
        _append_execution()
    # The profile is disabled for this process: later appends do not retry the unsafe target.
    assert journal.append(PROFILE, build_event()) is False
    assert list(victim.iterdir()) == []


def test_reading_absent_journal_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    read = journal_reader.read_journal(PROFILE)
    assert read.events == [] and read.damage == [] and read.segments == 0
    assert not (tmp_path / "data").exists()


def test_replaced_fields_survive_roundtrip() -> None:
    event = replace(build_event(), correlation_id="corr-1", cwd=None, index_generation=None)
    assert journal_codec.parse_event(journal_codec.serialize_event(event)[:-1]) == event
