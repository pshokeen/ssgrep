"""Physical journal expiry is gated by durable acceptance, never writer PID guesses."""

from __future__ import annotations

import errno
import fcntl
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import (
    collector_expiry,
    collector_journal,
    collector_store,
    journal,
    journal_retention,
    storage,
)
from ssgrep.accounting.collector_types import CollectError, RetentionCoverage
from ssgrep.accounting.contracts import Phase
from ssgrep.accounting.journal_codec import MalformedEvent, parse_event, serialize_event
from ssgrep.accounting.journal_reader import JournalRead, read_journal
from tests.accounting.conftest import build_event

PROFILE = "personal-local"
FLOOR = "2026-11-10"
AGGREGATE = "2025-11-05"


@pytest.fixture(autouse=True)
def fresh_writer():
    journal.close_all()
    yield
    journal.close_all()


def segment(name: str, data: bytes) -> Path:
    directory = storage.open_profile_dir(PROFILE, "journal")
    try:
        fd = storage.create_exclusive(directory, name)
        try:
            storage.write_all(fd, data)
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    return storage.data_root() / "accounting" / PROFILE / "journal" / name


def event(identity: str, day: str, **kwargs):
    return build_event(execution_id=identity, ts_utc=f"{day}T12:00:00Z", **kwargs)


def accept(read, *, previous=None, horizon=FLOOR, aggregate=AGGREGATE, generation=1):
    record = journal_retention.prepare_journal_retention(
        read, previous=previous, horizon_day=horizon, aggregate_horizon_day=aggregate
    )
    checkpoint = collector_store.Checkpoint(profile=PROFILE, generation=generation)
    checkpoint.retention = collector_expiry.retention_record(
        RetentionCoverage(),
        (),
        journal=journal_retention.persist_journal_retention(
            PROFILE, record, blobs=checkpoint.blobs
        ),
    )
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    return record


def finish(**kwargs):
    return journal_retention.finish_journal_retention(PROFILE, generation=1, **kwargs)


def test_29_30_31_day_files_expire_only_after_accepted_checkpoint():
    paths = []
    for days, day in ((29, "2026-11-11"), (30, "2026-11-10"), (31, "2026-11-09")):
        paths.append(
            segment(f"seg-v2-boundary-{days}.jsonl", serialize_event(event(f"call-{days}", day)))
        )
    read = read_journal(PROFILE)
    plan = journal_retention.prepare_journal_retention(
        read, previous=None, horizon_day=FLOOR, aggregate_horizon_day=AGGREGATE
    )
    assert finish().status == "no_checkpoint"
    assert all(path.exists() for path in paths)
    checkpoint = collector_store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.retention["journal"] = journal_retention.persist_journal_retention(
        PROFILE, plan, blobs=checkpoint.blobs
    )
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    out = finish()
    assert out.removed_segments == 1
    assert paths[0].exists() and paths[1].exists() and not paths[2].exists()
    assert {e.execution_id for e in read_journal(PROFILE).events} == {"call-29", "call-30"}


def test_started_and_completed_across_segments_preserve_only_old_phase_measurement():
    started = event("cross-midnight", "2026-11-09", phase=Phase.STARTED, outcome=None)
    completed = event("cross-midnight", "2026-11-10", phase=Phase.COMPLETED)
    old = segment("seg-v2-before-midnight.jsonl", serialize_event(started))
    segment("seg-v2-after-midnight.jsonl", serialize_event(completed))
    accept(read_journal(PROFILE))
    finish()
    assert not old.exists()
    events = read_journal(PROFILE).events
    assert {e.phase for e in events} == {Phase.STARTED, Phase.COMPLETED}
    prior = next(e for e in events if e.phase is Phase.STARTED)
    assert prior.mono_ns == started.mono_ns and prior.ts_utc == started.ts_utc
    assert prior.refs == () and prior.cwd is None and prior.fingerprint == ""
    assert prior.index_generation is None and prior.correlation_id is None
    assert next(e for e in events if e.phase is Phase.COMPLETED) == completed


def test_append_race_after_read_defers_source_until_new_checkpoint():
    old = event("old", "2026-11-09")
    path = segment("seg-v2-raced.jsonl", serialize_event(old))
    accept(read_journal(PROFILE))
    late = event("late", "2026-11-11")
    with path.open("ab") as stream:
        stream.write(serialize_event(late))
    assert finish().deferred_segments == 1
    assert path.exists() and serialize_event(late) in path.read_bytes()
    accept(read_journal(PROFILE), previous=journal_retention.load_journal_retention(PROFILE))
    finish()
    assert not path.exists()
    assert read_journal(PROFILE).events == [late]


def test_writer_never_waits_on_compactor_and_reopens_unlinked_descriptor():
    old = event("writer-old", "2026-11-09")
    journal.append(PROFILE, old)
    read = read_journal(PROFILE)
    accept(read)
    active = journal._current().segments[PROFILE]
    original_inode = os.fstat(active.fd).st_ino
    held = storage.open_profile_dir(PROFILE, "journal")
    fd = storage.open_regular(held, read.proofs[0].name)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal.append(PROFILE, event("during-lock", "2026-11-11"))
        assert os.fstat(journal._current().segments[PROFILE].fd).st_ino != original_inode
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
        os.close(held)
    finish()
    journal.append(PROFILE, event("after-expiry", "2026-11-11"))
    assert {e.execution_id for e in read_journal(PROFILE).events} == {"during-lock", "after-expiry"}


def test_active_pid_idle_segment_can_expire_and_next_normal_capture_survives():
    journal.append(PROFILE, event("idle-old", "2026-11-09"))
    fd = journal._current().segments[PROFILE].fd
    accept(read_journal(PROFILE))
    assert finish().removed_segments == 1
    assert os.fstat(fd).st_nlink == 0
    identity = journal.next_execution_id()
    for phase in (Phase.STARTED, Phase.COMPLETED, Phase.RENDERED):
        journal.append(PROFILE, event(identity, "2026-11-11", phase=phase))
    assert {e.phase for e in read_journal(PROFILE).events} == set(Phase)


def test_interrupted_publish_restarts_without_double_read_or_lost_totals(monkeypatch):
    old, recent = event("old", "2026-11-09"), event("recent", "2026-11-11")
    source = segment("seg-v2-mixed.jsonl", serialize_event(old) + serialize_event(recent))
    accept(read_journal(PROFILE))
    real_delete = storage.delete_file

    def crash(directory, name):
        if name == source.name:
            raise OSError(errno.EIO, "interrupted after retained segment publication")
        return real_delete(directory, name)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete_file", crash)
        with pytest.raises(OSError):
            finish()
    assert source.exists()
    during = read_journal(PROFILE)
    assert during.events == [recent] and during.duplicates == 1
    checkpoint = collector_store.load_checkpoint(PROFILE)
    assert checkpoint is not None and checkpoint.generation == 1
    finish()
    assert not source.exists()
    assert read_journal(PROFILE).events == [recent]


def test_late_partial_tail_is_not_deleted_then_newly_completed_record_is_accepted():
    old, recent = event("old", "2026-11-09"), event("late-tail", "2026-11-11")
    line = serialize_event(recent)
    source = segment("seg-v2-partial.jsonl", serialize_event(old) + line[:-19])
    recent_clock = datetime(2026, 11, 11, tzinfo=UTC).timestamp()
    os.utime(source, (recent_clock, recent_clock))
    accept(read_journal(PROFILE))
    assert finish().deferred_segments == 1
    with source.open("ab") as stream:
        stream.write(line[-19:])
    read = read_journal(PROFILE)
    assert read.events == [recent] and not read.incomplete_tails
    accept(read, previous=journal_retention.load_journal_retention(PROFILE))
    finish()
    assert read_journal(PROFILE).events == [recent]


def test_old_partial_and_quarantine_bytes_are_removed_without_raw_reacceptance():
    source = segment("seg-v2-damage.jsonl", b"private-canary-invalid\nprivate-tail")
    old_clock = datetime(2026, 11, 9, tzinfo=UTC).timestamp()
    os.utime(source, (old_clock, old_clock))
    read = read_journal(PROFILE)
    assert len(read.quarantined) == 1 and len(read.incomplete_tails) == 1
    accept(read)
    finish()
    assert not source.exists()
    quarantine = source.parent.parent / "journal-quarantine"
    assert not list(quarantine.iterdir())
    assert not read_journal(PROFILE).events


def test_retired_identity_blocks_copied_and_timestamp_mutated_phase_until_aggregate_floor():
    old = event("already-counted", "2026-11-09")
    segment("seg-v2-original.jsonl", serialize_event(old))
    record = accept(read_journal(PROFILE))
    finish()
    segment(
        "seg-v2-copy.jsonl",
        serialize_event(old) + serialize_event(replace(old, ts_utc="2026-11-11T12:00:00Z")),
    )
    read = read_journal(PROFILE)
    assert read.events == [] and read.retired_events == 2
    newer = journal_retention.prepare_journal_retention(
        read, previous=record, horizon_day="2027-12-15", aggregate_horizon_day="2026-11-10"
    )
    assert newer["retired"] == {}
    assert "already-counted" not in str(record)


def test_compaction_budget_and_live_segment_lock_report_honest_deferral():
    first = segment("seg-v2-one.jsonl", serialize_event(event("one", "2026-11-09")))
    second = segment("seg-v2-two.jsonl", serialize_event(event("two", "2026-11-09")))
    read = read_journal(PROFILE)
    accept(read)
    directory = storage.open_profile_dir(PROFILE, "journal")
    fd = storage.open_regular(directory, first.name)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = finish(segment_budget=1)
        assert out.deferred_segments == 2 and first.exists() and second.exists()
    finally:
        os.close(fd)
        os.close(directory)
    out = finish(byte_budget=1)
    assert out.deferred_segments == 2 and out.bytes_read == 0
    finish()
    assert not first.exists() and not second.exists()


def test_legacy_segment_requires_explicit_stopped_writer_migration():
    legacy = segment("seg-legacy.jsonl", serialize_event(event("legacy", "2026-11-09")))
    accept(read_journal(PROFILE))
    assert finish().legacy_segments == 1 and legacy.exists()
    finish(legacy_writers_stopped=True)
    assert not legacy.exists()


def test_scope_key_metadata_roundtrips_and_old_started_measurements_drop_authority():
    from ssgrep.accounting.journal_codec import minimal_started

    source = event(
        "scoped",
        "2026-11-09",
        phase=Phase.STARTED,
        outcome=None,
        fingerprint_key_id="key-v2",
        scope_basis="trusted_invocation",
    )
    assert parse_event(serialize_event(source)[:-1]) == source
    minimal = minimal_started(source)
    assert minimal.fingerprint_key_id is None and minimal.scope_basis == "unknown_scope"
    assert minimal.mono_ns == source.mono_ns
    with pytest.raises(MalformedEvent, match="scope_basis"):
        parse_event(serialize_event(replace(source, scope_basis="guessed_scope"))[:-1])


def test_accepted_counter_blob_remains_byte_identical_through_expiry_and_restart():
    from ssgrep.accounting.counters import empty_counters, observe

    old, current = event("closed", "2026-11-09"), event("open", "2026-11-11")
    segment("seg-v2-totals.jsonl", serialize_event(old) + serialize_event(current))
    read = read_journal(PROFILE)
    counters = observe(empty_counters(1, "2026-11-01T00:00:00Z"), read.events, horizon_day=None)
    assert sum(n for _, n in counters.executions) == 2
    checkpoint = collector_store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.blobs["counters"] = collector_store.write_blob(
        PROFILE, "counters", counters.to_dict()
    )
    plan = journal_retention.prepare_journal_retention(
        read, previous=None, horizon_day=FLOOR, aggregate_horizon_day=AGGREGATE
    )
    checkpoint.retention["journal"] = journal_retention.persist_journal_retention(
        PROFILE, plan, blobs=checkpoint.blobs
    )
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    finish()
    journal.close_all()
    restored = collector_store.load_checkpoint(PROFILE)
    assert restored is not None
    assert collector_store.read_blob(PROFILE, restored.blobs["counters"]) == counters.to_dict()
    assert read_journal(PROFILE).events == [current]


def test_size_and_utc_day_rotation_keep_every_execution_and_never_reuse_closed_file(monkeypatch):
    clock = {"seconds": datetime(2026, 11, 11, tzinfo=UTC).timestamp()}
    monkeypatch.setattr(journal.time, "time", lambda: clock["seconds"])
    monkeypatch.setattr(journal, "SEGMENT_TARGET_BYTES", 1)
    for n in range(3):
        journal.append(PROFILE, event(f"rotated-{n}", "2026-11-11"))
    journal.close_all()
    journal.append(PROFILE, event("after-close", "2026-11-11"))
    monkeypatch.setattr(journal, "SEGMENT_TARGET_BYTES", 8 * 1024 * 1024)
    clock["seconds"] += 86400
    journal.append(PROFILE, event("next-day", "2026-11-12"))
    read = read_journal(PROFILE)
    assert read.segments == 5 and len(read.events) == 5
    assert {e.execution_id for e in read.events} == {f"rotated-{n}" for n in range(3)} | {
        "after-close",
        "next-day",
    }


def test_idle_retention_does_not_reread_unchanged_current_segments():
    segment("seg-v2-current.jsonl", serialize_event(event("current", "2026-11-11")))
    record = accept(read_journal(PROFILE))
    assert record["proofs"] == []
    assert finish().bytes_read == 0
    assert finish().bytes_read == 0


def test_actual_cli_requires_stopped_writer_ack_and_preserves_other_profile(tmp_path):
    import json

    from tests.accounting.test_savings_cli import Run, document
    from tests.accounting.verify_scenes import env_for

    run = Run(tmp_path / "home", tmp_path / "data")
    environment = env_for(tmp_path)
    run.home.mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy_fixture = tmp_path / "acme.json"
    policy_fixture.write_text(json.dumps({"profile_id": "acme", "roots": [str(workspace)]}))
    setup = run(
        "metrics",
        "setup",
        "--yes",
        "--profile",
        "acme",
        "--policy",
        str(policy_fixture),
        "--json",
        env=environment,
    )
    assert setup.returncode == 0, setup.stderr
    other = segment("seg-untouched-profile.jsonl", serialize_event(event("other", "2025-01-01")))
    old_day = (datetime.now(UTC) - timedelta(days=31)).date().isoformat()
    old = event("legacy-cli", old_day, profile="acme", cwd=str(workspace))
    directory = storage.open_profile_dir("acme", "journal")
    try:
        fd = storage.create_exclusive(directory, "seg-legacy-cli.jsonl")
        try:
            storage.write_all(fd, serialize_event(old))
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    legacy = run.data / "accounting" / "acme" / "journal" / "seg-legacy-cli.jsonl"
    default = run("metrics", "collect", "--profile", "acme", "--json", env=environment)
    assert default.returncode == 0, default.stderr
    body = json.dumps(document(default))
    assert legacy.exists() and other.exists()
    assert "journal_retention_blocked_legacy_writers" in body
    assert "ssgrep metrics collect --profile acme --journal-legacy-writers-stopped" in body
    acknowledged = run(
        "metrics",
        "collect",
        "--profile",
        "acme",
        "--journal-legacy-writers-stopped",
        "--json",
        env=environment,
    )
    assert acknowledged.returncode == 0, acknowledged.stderr
    assert not legacy.exists() and other.exists()
    assert "journal_retention_blocked_legacy_writers" not in json.dumps(document(acknowledged))


def test_publish_interruption_leaves_only_safe_staging_then_restart_cleans_it(monkeypatch):
    from ssgrep.accounting import journal_compact

    old, recent = event("old", "2026-11-09"), event("recent", "2026-11-11")
    source = segment("seg-v2-before-publish.jsonl", serialize_event(old) + serialize_event(recent))
    accept(read_journal(PROFILE))

    def crash(*args, **kwargs):
        raise OSError(errno.EIO, "interrupted retention link")

    with monkeypatch.context() as patch:
        patch.setattr(journal_compact.os, "link", crash)
        with pytest.raises(OSError):
            finish()
    assert source.exists()
    staged = list(source.parent.glob(".retention-*.tmp"))
    assert len(staged) == 1 and staged[0].read_bytes() == serialize_event(recent)
    assert read_journal(PROFILE).events == [recent]
    finish()
    assert not source.exists() and not list(source.parent.glob(".retention-*.tmp"))
    assert read_journal(PROFILE).events == [recent]


@pytest.mark.parametrize(
    "timestamp", ["yesterday", "2026-02-30T00:00:00Z", "2026-11-11T00:00:00+00:00"]
)
def test_invalid_age_cannot_authorize_retention_or_be_reaccepted(timestamp):
    source = segment(
        "seg-v2-invalid-age.jsonl",
        serialize_event(replace(build_event(), ts_utc=timestamp)),
    )
    read = read_journal(PROFILE)
    assert read.events == [] and len(read.quarantined) == 1
    old_clock = datetime(2026, 11, 9, tzinfo=UTC).timestamp()
    os.utime(source, (old_clock, old_clock))
    accept(read_journal(PROFILE))
    finish()
    assert not source.exists()


def test_large_quarantine_metadata_and_orphan_staging_expire_with_bounded_reads():
    import json

    recent = event("current", "2026-11-11")
    segment("seg-v2-current.jsonl", serialize_event(recent))
    quarantine = storage.open_profile_dir(PROFILE, "journal-quarantine")
    try:
        fd = storage.create_exclusive(quarantine, "orphan.json")
        try:
            storage.write_all(
                fd,
                json.dumps(
                    {
                        "segment": "seg-missing.jsonl",
                        "reason": "private-canary" * 2000,
                    }
                ).encode(),
            )
        finally:
            os.close(fd)
    finally:
        os.close(quarantine)
    temp = segment(".retention-abandoned.tmp", serialize_event(recent))
    accept(read_journal(PROFILE))
    out = finish()
    assert out.quarantine_removed == 1 and out.remnants_removed == 1
    assert not temp.exists() and out.bytes_read < 32 * 1024 * 1024
    assert read_journal(PROFILE).events == [recent]


def test_restart_destination_validation_obeys_same_read_byte_budget(monkeypatch):
    old, recent = event("old", "2026-11-09"), event("recent", "2026-11-11")
    source = segment("seg-v2-budget-restart.jsonl", serialize_event(old) + serialize_event(recent))
    accept(read_journal(PROFILE))
    real_delete = storage.delete_file

    def interrupt(directory, name):
        if name == source.name:
            raise OSError(errno.EIO, "source not unlinked")
        return real_delete(directory, name)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete_file", interrupt)
        with pytest.raises(OSError):
            finish()
    budget = source.stat().st_size
    out = finish(byte_budget=budget)
    assert out.status == "pending" and source.exists()
    assert out.bytes_read <= budget
    finish()
    assert not source.exists() and read_journal(PROFILE).events == [recent]


def test_name_collisions_are_bounded_and_next_capture_can_retry():
    state = journal._current()
    for rotation in (state.segment_seq + 1, state.segment_seq + 2):
        segment(
            journal.segment_name(state.nonce, rotation),
            serialize_event(event(f"prior-{rotation}", "2026-11-11")),
        )
    with pytest.raises(FileExistsError):
        journal.append(PROFILE, event("not-written", "2026-11-11"))
    journal.append(PROFILE, event("after-conflicts", "2026-11-11"))
    identities = {e.execution_id for e in read_journal(PROFILE).events}
    assert (
        len(identities) == 3 and "after-conflicts" in identities and "not-written" not in identities
    )


def test_uncommitted_horizon_and_wrong_generation_cannot_authorize_erasure():
    source = segment("seg-v2-uncommitted.jsonl", serialize_event(event("closed", "2026-11-09")))
    previous = accept(read_journal(PROFILE), horizon="2026-11-01")
    uncommitted = journal_retention.prepare_journal_retention(
        read_journal(PROFILE),
        previous=previous,
        horizon_day=FLOOR,
        aggregate_horizon_day=AGGREGATE,
    )
    assert uncommitted["proofs"]
    out = journal_retention.finish_journal_retention(PROFILE, generation=2)
    assert out.status == "generation_mismatch" and source.exists()
    assert read_journal(PROFILE).events[0].execution_id == "closed"
    accept(read_journal(PROFILE), previous=previous, generation=2)
    journal_retention.finish_journal_retention(PROFILE, generation=2)
    assert not source.exists()


@pytest.mark.parametrize("ack", [None, 0, 1, "yes", "false"])
def test_nonboolean_ack_never_authorizes_uncooperative_legacy_deletion(ack):
    source = segment("seg-legacy.jsonl", serialize_event(event("legacy", "2026-11-09")))
    accept(read_journal(PROFILE))
    assert finish(legacy_writers_stopped=ack).status == "blocked_legacy"
    assert source.exists()


def test_live_other_pid_writer_continues_normal_capture_after_accepted_expiry():
    import select
    import subprocess
    import sys
    import textwrap

    from tests.accounting.verify_scenes import env_for

    code = textwrap.dedent(
        """
        import sys
        from ssgrep.accounting import journal
        from ssgrep.accounting.contracts import Phase
        from tests.accounting.conftest import build_event
        journal.append("personal-local", build_event(
            execution_id="child-old", ts_utc="2026-11-09T12:00:00Z"
        ))
        print("ready", flush=True)
        sys.stdin.readline()
        identity = journal.next_execution_id()
        for phase in (Phase.STARTED, Phase.COMPLETED, Phase.RENDERED):
            journal.append("personal-local", build_event(
                execution_id=identity, phase=phase, ts_utc="2026-11-11T12:00:00Z"
            ))
        """
    )
    root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=root,
        env={
            **env_for(storage.data_root().parent),
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None and process.stdin is not None
        assert select.select([process.stdout], [], [], 15)[0]
        assert process.stdout.readline().strip() == "ready"
        assert process.poll() is None
        accept(read_journal(PROFILE))
        assert finish().removed_segments == 1
        _, errors = process.communicate(input=chr(10), timeout=30)
        assert process.returncode == 0, errors
        events = read_journal(PROFILE).events
        assert len(events) == 3 and {e.phase for e in events} == set(Phase)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_real_capture_entry_completion_and_render_survive_idle_writer_expiry():
    from ssgrep.accounting import capture, fingerprint, policy
    from ssgrep.accounting.contracts import Operation, Outcome, Quantity

    capture._unsafe.clear()
    fingerprint.forget_keys()
    now = datetime.now(UTC)
    old_day = (now - timedelta(days=31)).date().isoformat()
    journal.append(PROFILE, event("prior-control", old_day))
    accept(
        read_journal(PROFILE),
        horizon=(now - timedelta(days=30)).date().isoformat(),
        aggregate=(now - timedelta(days=400)).date().isoformat(),
    )
    assert finish().removed_segments == 1
    policy.enable_collection(PROFILE)
    execution = capture.begin(Operation.SEARCH, text="normal retention control")
    assert execution is not None
    capture.complete(execution, Outcome.OK)
    capture.rendered(execution, rendered_bytes=Quantity.observed(812, "bytes", "cli_text"))
    events = read_journal(PROFILE).events
    assert len(events) == 3 and {e.phase for e in events} == set(Phase)
    assert all(e.execution_id != "prior-control" for e in events)


def test_scrubbed_cached_start_still_schedules_original_private_segment_compaction():
    from ssgrep.accounting.journal_codec import minimal_started

    started = event("cached-bridge", "2026-11-09", phase=Phase.STARTED, outcome=None)
    completed = event("cached-bridge", "2026-11-11")
    old = segment("seg-v2-cached-start.jsonl", serialize_event(started))
    fresh = segment("seg-v2-cached-completed.jsonl", serialize_event(completed))
    cached = read_journal(PROFILE)
    cached.events = [minimal_started(started), completed]
    cached.segment_events = {old.name: [minimal_started(started)], fresh.name: [completed]}
    cached.retention_pending = {old.name}
    record = accept(cached)
    assert {proof["name"] for proof in record["proofs"]} == {old.name}
    finish()
    assert not old.exists()
    assert read_journal(PROFILE).events == [minimal_started(started), completed]


def test_safe_native_call_scope_projection_roundtrips_without_trusted_scope_alias():
    projected = event("derived-scope", "2026-11-11", scope_basis="native_call_metadata")
    assert parse_event(serialize_event(projected)[:-1]) == projected


def test_zero_byte_failed_write_remnant_is_reclaimed_and_future_capture_survives(monkeypatch):
    def full_disk(*args):
        raise OSError(errno.ENOSPC, "write failed before any bytes")

    with monkeypatch.context() as patch:
        patch.setattr(journal.os, "write", full_disk)
        with pytest.raises(OSError, match="before any bytes"):
            journal.append(PROFILE, event("failed-empty", "2026-11-11"))
    fd = journal._current().segments[PROFILE].fd
    assert os.fstat(fd).st_size == 0
    accept(read_journal(PROFILE))
    assert finish().removed_segments == 1
    assert os.fstat(fd).st_nlink == 0
    journal.append(PROFILE, event("after-empty", "2026-11-11"))
    assert {e.execution_id for e in read_journal(PROFILE).events} == {"after-empty"}


def accept_cached(read: JournalRead) -> collector_store.Checkpoint:
    record = journal_retention.prepare_journal_retention(
        read, previous=None, horizon_day=FLOOR, aggregate_horizon_day=AGGREGATE
    )
    checkpoint = collector_store.Checkpoint(profile=PROFILE, generation=1)
    checkpoint.retention["journal"] = journal_retention.persist_journal_retention(
        PROFILE, record, blobs=checkpoint.blobs
    )
    collector_journal.save_journal_cache(PROFILE, read, checkpoint.blobs, record)
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    return checkpoint


def test_cached_reader_accepts_compaction_and_new_copies_without_expired_resurrection():
    old = event("cached-expired", "2026-11-09")
    recent = event("cached-retained", "2026-11-11")
    source = segment("seg-v2-cached-mixed.jsonl", serialize_event(old) + serialize_event(recent))
    checkpoint = accept_cached(read_journal(PROFILE))
    assert finish().removed_segments == 1 and not source.exists()
    segment(
        "seg-v2-cached-new-copy.jsonl",
        serialize_event(recent) + serialize_event(replace(old, ts_utc="2026-11-11T12:00:00Z")),
    )
    restored = collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert restored.events == [recent]
    assert restored.duplicates == 1 and restored.retired_events == 1


def test_missing_compaction_replacement_cannot_drop_cached_retained_measurements():
    old = event("mixed-expired", "2026-11-09")
    recent = event("mixed-retained", "2026-11-11")
    source = segment("seg-v2-missing-mixed.jsonl", serialize_event(old) + serialize_event(recent))
    checkpoint = accept_cached(read_journal(PROFILE))
    assert finish().removed_segments == 1
    replacement = next(source.parent.glob("seg-v2-compact-*.jsonl"))
    valid = replacement.read_bytes()
    assert collector_journal.read_cached_journal(PROFILE, checkpoint).events == [recent]
    replacement.unlink()
    with pytest.raises(CollectError) as failed:
        collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert failed.value.condition == "state_corrupt"
    replacement.write_bytes(valid)
    replacement.chmod(0o600)
    assert collector_journal.read_cached_journal(PROFILE, checkpoint).events == [recent]


def test_cached_reader_allows_authorized_expired_only_directory_absence():
    old = event("cached-entirely-expired", "2026-11-09")
    source = segment("seg-v2-cached-expired.jsonl", serialize_event(old))
    checkpoint = accept_cached(read_journal(PROFILE))
    assert finish().removed_segments == 1 and not source.exists()
    source.parent.rmdir()
    restored = collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert restored.events == [] and restored.proofs == []


def test_cached_reader_resumes_recent_pending_tail_without_recounting():
    old = event("cached-tail-expired", "2026-11-09")
    recent = event("cached-tail-retained", "2026-11-11")
    line = serialize_event(recent)
    source = segment("seg-v2-cached-tail.jsonl", serialize_event(old) + line[:-19])
    recent_clock = datetime(2026, 11, 11, tzinfo=UTC).timestamp()
    os.utime(source, (recent_clock, recent_clock))
    checkpoint = accept_cached(read_journal(PROFILE))
    assert finish().reasons == {"recent_damage": 1}
    pending = collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert pending.events == [] and len(pending.incomplete_tails) == 1
    with source.open("ab") as stream:
        stream.write(line[-19:])
    restored = collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert restored.events == [recent]
    assert not restored.incomplete_tails and restored.duplicates == 0


def test_cached_reader_recovers_compaction_crash_after_replacement_publication(monkeypatch):
    old = event("cached-crash-expired", "2026-11-09")
    recent = event("cached-crash-retained", "2026-11-11")
    source = segment("seg-v2-cached-crash.jsonl", serialize_event(old) + serialize_event(recent))
    checkpoint = accept_cached(read_journal(PROFILE))
    real_delete = storage.delete_file

    def interrupted_delete(directory, name):
        if name == source.name:
            raise OSError(errno.EIO, "interrupted after replacement publication")
        return real_delete(directory, name)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "delete_file", interrupted_delete)
        with pytest.raises(OSError, match="replacement publication"):
            finish()
    during = collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert during.events == [recent] and during.duplicates == 1
    assert finish().removed_segments == 1 and not source.exists()
    restored = collector_journal.read_cached_journal(PROFILE, checkpoint)
    assert restored.events == [recent] and restored.duplicates == 0
