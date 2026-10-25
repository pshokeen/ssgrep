"""Native authority failures retain typed gaps rather than private source payloads."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ssgrep.accounting.collect import store
from ssgrep.accounting.readers.identity_migration import prepare
from ssgrep.accounting.readers.native_source import (
    NativeSourceReference,
    NativeSourceStatus as Status,
)
from tests.accounting.ledger.reduce_builders import policy
from tests.accounting.readers.readers_fixtures import (
    opencode_add_message,
    opencode_add_part,
    opencode_connect,
)
from tests.accounting.readers.test_identity import NOW, A, B, legacy_pair, setup
from tests.accounting.readers.test_native_source import REF, old_source
from tests.accounting.readers.test_native_source_locators import digest_tree
from tests.accounting.readers.test_native_source_review_edges import CWD, database_scene


def fresh_source(adapter, reference):
    return replace(
        adapter,
        references={reference.ref: reference},
        bytes_read=0,
        observed_references={},
        _snapshots={},
    )


@pytest.mark.parametrize("field", ["runtime", "session_id", "source_path"])
def test_empty_index_identity_cannot_authorize_native_source(tmp_path, monkeypatch, field):
    from ssgrep.store import LanceStore

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "index"))
    path, reference, adapter = old_source(tmp_path / "native")
    repo = LanceStore()
    try:
        repo.table("episodes", create=True)
        repo.table("metadata", create=True)
        repo.upsert(
            "episodes",
            {
                "episode_id": REF,
                "session_id": reference.session_id,
                "project": "owned",
                "title": "private locator title",
                "source_path": str(path),
                "cwd": reference.cwd,
                "runtime": "claude",
            },
        )
        repo.set_meta("index_state", "ready")
        repo.set_meta("last_index_time", "owned-generation")
        indexed = replace(adapter, references={}, repository=repo)
        control = indexed.resolve(REF)
        assert control.status is Status.OK and control.line.key == ("claude", "src1")
        repo.update("episodes", f"episode_id = '{REF}'", {field: ""})
        denied = replace(indexed, bytes_read=0, observed_references={}, _snapshots={})
        before = digest_tree(tmp_path)
        result = denied.resolve(REF)
        assert result.status is Status.IDENTITY_MISMATCH
        assert result.line is None and result.source_digest == "" and not result.record_digests
        assert result.bytes_read == denied.bytes_read == 0
        assert not denied.observed_references
        assert "private locator title" not in repr(result) and str(path) not in repr(result)
        assert digest_tree(tmp_path) == before
    finally:
        repo.close()


@pytest.mark.parametrize(
    "damage,status",
    [
        ("missing_owner", Status.MISSING),
        ("oversized_owner", Status.BOUNDED),
        ("unsupported_schema", Status.UNSUPPORTED),
        ("pending_part", Status.INCOMPLETE),
        ("malformed_part", Status.INCOMPLETE),
    ],
)
def test_native_session_rejects_incomplete_committed_authority(tmp_path, damage, status):
    database, originals, adapter = database_scene(tmp_path)
    reference = NativeSourceReference("opencode:s:ep:0", "opencode", "s", database)
    control = fresh_source(adapter, reference).resolve(reference.ref)
    assert control.status is Status.OK
    assert {item.native_identity for item in originals} <= dict(control.record_digests).keys()
    writer = opencode_connect(database)
    try:
        if damage == "missing_owner":
            writer.execute("DELETE FROM session WHERE id = 's'")
        elif damage == "oversized_owner":
            writer.execute("UPDATE session SET project_id = ? WHERE id = 's'", ("x" * 16385,))
        elif damage == "unsupported_schema":
            writer.execute("ALTER TABLE part RENAME COLUMN message_id TO unavailable_message_id")
        elif damage == "pending_part":
            data = json.loads(writer.execute("SELECT data FROM part WHERE id = 'p'").fetchone()[0])
            data["state"]["status"] = "running"
            writer.execute("UPDATE part SET data = ? WHERE id = 'p'", (json.dumps(data),))
        else:
            writer.execute("UPDATE part SET data = ? WHERE id = 'p'", ('{"private_token":',))
        assert database.with_name(database.name + "-wal").is_file()
        before = digest_tree(tmp_path)
        denied = fresh_source(adapter, reference)
        result = denied.resolve(reference.ref)
        assert result.status is status and result.line is None
        assert result.source_digest == "" and not result.record_digests
        assert not denied.observed_references
        assert result.bytes_read == denied.bytes_read
        assert denied.bytes_read <= denied.byte_budget
        assert "private_token" not in repr(result) and str(database) not in repr(result)
        assert digest_tree(tmp_path) == before
    finally:
        writer.close()


@pytest.mark.parametrize(
    "damage,status",
    [
        ("empty_text", Status.IDENTITY_MISMATCH),
        ("nontext_part", Status.IDENTITY_MISMATCH),
        ("malformed_part", Status.INCOMPLETE),
        ("oversized_message", Status.BOUNDED),
        ("malformed_message", Status.UNREADABLE),
    ],
)
def test_direct_user_fact_requires_readable_native_user_authority(tmp_path, damage, status):
    database, _, adapter = database_scene(tmp_path)
    reference = NativeSourceReference("opencode:s:ep:0", "opencode", "s", database)
    writer = opencode_connect(database)
    try:
        opencode_add_message(writer, "user-m", "s", {"role": "user"}, at=3)
        opencode_add_part(
            writer, "user-p", "user-m", "s", {"type": "text", "text": "prior private fact"}, at=4
        )
        resolved = fresh_source(adapter, reference).resolve(reference.ref)
        assert resolved.status is Status.OK
        fact = next(record.ev for record in resolved.line.records if record.kind == "user_turn")
        assert fact.payload == {"text": "prior private fact", "cwd": CWD}
        control = adapter.read_record(
            reference, fact.native_identity, fact.digest, locator=fact.locator
        )
        assert control.status is Status.OK and control.evidence == fact
        if damage == "empty_text":
            writer.execute(
                "UPDATE part SET data = ? WHERE id = 'user-p'",
                (json.dumps({"type": "text", "text": ""}),),
            )
        elif damage == "nontext_part":
            writer.execute(
                "UPDATE part SET data = ? WHERE id = 'user-p'",
                (json.dumps({"type": "reasoning", "text": "prior private fact"}),),
            )
        elif damage == "malformed_part":
            writer.execute("UPDATE part SET data = ? WHERE id = 'user-p'", ('{"private_token":',))
        elif damage == "oversized_message":
            writer.execute(
                "UPDATE message SET data = ? WHERE id = 'user-m'",
                (json.dumps({"role": "user", "private": "x" * 8388608}),),
            )
        else:
            writer.execute(
                "UPDATE message SET data = ? WHERE id = 'user-m'", ('{"private_token":',)
            )
        before = digest_tree(tmp_path)
        denied = fresh_source(adapter, reference)
        result = denied.read_record(
            reference, fact.native_identity, fact.digest, locator=fact.locator
        )
        assert result.status is status and result.evidence is None
        assert (
            result.bytes_read == denied.bytes_read and 0 < result.bytes_read <= denied.byte_budget
        )
        assert "prior private fact" not in repr(result) and "private_token" not in repr(result)
        assert str(database) not in repr(result)
        assert digest_tree(tmp_path) == before
    finally:
        writer.close()


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("invalid_json", "source_unreadable"),
        ("no_native_candidate", "source_identity_mismatch"),
    ],
)
def test_migration_gap_reason_survives_checkpoint_restart_without_switching_authority(
    tmp_path, monkeypatch, damage, reason
):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "unavailable.jsonl"
    originals = legacy_pair(path, "legacy-owner")
    valid = path.read_bytes()
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=store.write_segments("personal-local", originals),
        methodology="published-method",
        input_digest="published-input",
    )
    previous = list(checkpoint.segments)
    first_line, remaining = valid.split(b"\n", 1)
    damaged = b'{"private_token":' if damage == "invalid_json" else b"[]"
    path.write_bytes(damaged.ljust(len(first_line), b" ") + b"\n" + remaining)
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy(), max_records=1)
    expected_gaps = {("claude", "legacy-owner"): reason}
    assert not result.complete and result.unavailable_sessions == expected_gaps
    assert checkpoint.segments == previous
    resumed = store.Checkpoint.from_dict(checkpoint.to_dict())
    # The second locator cannot repair the first record's failed authority.
    continued = prepare("personal-local", resumed, NOW, env=env, policy=policy())
    assert not continued.complete and continued.unavailable_sessions == expected_gaps
    assert resumed.segments == previous and resumed.methodology == "published-method"
    assert resumed.input_digest == "published-input"
    assert all(
        item.origin_domain is None
        for item in store.iter_evidence("personal-local", resumed.segments)
    )
    assert "private_token" not in repr(continued)
    path.write_bytes(valid)
    recovered = prepare("personal-local", resumed, NOW, env=env, policy=policy())
    assert recovered.complete
    migrated = list(store.iter_evidence("personal-local", resumed.segments))
    assert len(migrated) == 2
    assert {
        (item.kind, item.origin_domain, item.origin_id.partition(":")[0])
        for item in migrated
        if item.origin_id is not None
    } == {
        ("tool_call", "claude_record_uuid", A),
        ("tool_result", "claude_record_uuid", B),
    }
    assert {item.kind for item in migrated} == {"tool_call", "tool_result"}


@pytest.mark.parametrize("sidecar_kind", ["directory", "symlink", "fifo"])
def test_native_snapshot_rejects_nonregular_wal_without_touching_source(tmp_path, sidecar_kind):
    import os

    database, originals, adapter = database_scene(tmp_path)
    target = next(item for item in originals if item.kind == "tool_result")
    reference = NativeSourceReference("opencode:s:ep:0", "opencode", "s", database)
    control = adapter.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert control.status is Status.OK and control.evidence == target
    sidecar = database.with_name(database.name + "-wal")
    foreign = tmp_path / "foreign-wal"
    foreign.write_bytes(b"private foreign WAL bytes")
    if sidecar_kind == "directory":
        sidecar.mkdir()
    elif sidecar_kind == "symlink":
        sidecar.symlink_to(foreign)
    else:
        os.mkfifo(sidecar)
    before = database.read_bytes()
    sidecar_before = sidecar.lstat()
    denied = fresh_source(adapter, reference)
    result = denied.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert result.status is Status.CHANGED and result.evidence is None
    assert result.bytes_read == denied.bytes_read == 0
    assert database.read_bytes() == before
    assert sidecar.lstat() == sidecar_before
    assert foreign.read_bytes() == b"private foreign WAL bytes"
    assert "private foreign" not in repr(result) and str(database) not in repr(result)


@pytest.mark.parametrize("runtime", ["pi", "omp", "prime-agent"])
def test_direct_pi_header_owner_must_match_requested_native_session(tmp_path, runtime):
    from tests.accounting.readers.test_native_source_review_edges import json_scene

    path, _, _, target, reference, adapter = json_scene(tmp_path, runtime)
    control = adapter.read_record(
        reference, target.native_identity, target.digest, locator=target.locator
    )
    assert control.status is Status.OK and control.evidence == target
    before = path.read_bytes()
    first_line_bytes = len(before.splitlines(keepends=True)[0])
    denied = fresh_source(adapter, reference)
    result = denied.read_record(
        replace(reference, session_id="different-owner"),
        target.native_identity,
        target.digest,
        locator=target.locator,
    )
    assert result.status is Status.IDENTITY_MISMATCH and result.evidence is None
    assert result.bytes_read == denied.bytes_read
    assert first_line_bytes <= result.bytes_read <= int(target.locator.partition(":")[2])
    assert path.read_bytes() == before
