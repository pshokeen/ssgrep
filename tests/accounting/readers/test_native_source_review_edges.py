"""Blind-source review: authority, concurrent append proofs and shared snapshots."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.collect.replay import EvidenceReplay, compact_payload
from ssgrep.accounting.ledger.reduce import readers
from ssgrep.accounting.privacy.policy import PolicyFile
from ssgrep.accounting.readers.native_source import (
    NativeSourceAdapter,
    NativeSourceReference,
    NativeSourceResult,
    NativeSourceStatus as Status,
)
from ssgrep.accounting.views.filter import Scoper
from tests.accounting.ledger.reduce_builders import policy, rare_view
from tests.accounting.readers.readers_fixtures import (
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    codex_call,
    codex_meta,
    codex_output,
    codex_turn,
    opencode_add_message,
    opencode_add_part,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)

CWD = "/work/approved/project"
NOW = 1_800_000_000_000


def authority():
    org = policy("/work/approved", profile="org")
    return Scoper(org, PolicyFile(profiles={"org": org}))


def json_scene(tmp_path, runtime):
    if runtime == "claude":
        rows = [
            claude_prompt("s", "question"),
            claude_tool_use("s", "c", "Bash", {"command": "cat log"}, cwd=CWD),
            claude_tool_result("s", "c", "answer"),
        ]
        rows = [{**row, "cwd": CWD} for row in rows]
    elif runtime == "codex":
        rows = [
            codex_meta("s", CWD),
            codex_turn("t", CWD),
            codex_call("c", "exec_command", {"cmd": "cat log"}),
            codex_output("c", "answer"),
        ]
    else:
        rows = [
            pi_header("s", cwd=CWD),
            pi_tool_call("entry-call", "c", "bash", {"command": "cat log"}),
            pi_tool_result("entry-result", "c", "answer"),
        ]
    path = write_records(tmp_path / "source.jsonl", rows)
    reader = readers({"HOME": str(tmp_path)})[runtime]
    original = reader.read(reader.source_for(path)).evidence
    result = next(item for item in original if item.kind == "tool_result")
    adapter = NativeSourceAdapter({}, {runtime: reader}, (tmp_path,), authority(), NOW)
    reference = NativeSourceReference(f"{runtime}:s:ep:0", runtime, "s", path, cwd=CWD)
    return path, reader, original, result, reference, adapter


def database_scene(tmp_path):
    path = tmp_path / "opencode.db"
    connection = opencode_connect(path)
    connection.execute(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?)", ("s", "p", None, CWD, "s", 1, 1)
    )
    opencode_add_message(connection, "m", "s", opencode_assistant_data(), at=1)
    opencode_add_part(
        connection,
        "p",
        "m",
        "s",
        opencode_tool_part("c", "bash", {"command": "cat log"}, output="answer", start=1, end=2),
        at=2,
    )
    connection.close()
    reader = readers({"HOME": str(tmp_path)})["opencode"]
    original = [
        item
        for item in reader.read(reader.source_for(path)).evidence
        if item.kind in {"tool_call", "tool_result"}
    ]
    adapter = NativeSourceAdapter({}, {"opencode": reader}, (tmp_path,), authority(), NOW)
    return path, original, adapter


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("result_first", [False, True])
def test_org_opencode_result_hydration_does_not_depend_on_group_order(
    tmp_path, batch, result_first
):
    _, originals, adapter = database_scene(tmp_path)
    if result_first:
        originals.reverse()
    compact = [replace(item, payload=compact_payload(item.payload)) for item in originals]
    replay = EvidenceReplay("org", adapter=adapter)
    actual = replay.hydrate_many(compact) if batch else [replay.hydrate(item) for item in compact]
    assert actual == originals
    assert not replay.failures


@pytest.mark.parametrize("runtime", ["claude", "codex", "omp", "pi", "prime-agent"])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rewrite", [False, True])
def test_verified_jsonl_replay_accepts_append_but_not_rewritten_proof(
    tmp_path, monkeypatch, runtime, batch, rewrite
):
    path, reader, _, result, reference, adapter = json_scene(tmp_path, runtime)
    original_decode = reader.decode
    fired = False

    def change_after_decode(source, obj, context, locator):
        nonlocal fired
        decoded = original_decode(source, obj, context, locator)
        if not fired and any(
            item.native_identity == result.native_identity for item in decoded or []
        ):
            fired = True
            data = path.read_bytes()
            if rewrite:
                data = data.replace(b"approved", b"forbiddd")
            path.write_bytes(data + b'{"type":"system","text":"append"}\n')
        return decoded

    monkeypatch.setattr(reader, "decode", change_after_decode)
    if batch:
        from ssgrep.accounting.collect.replay import read_batch

        outcome = read_batch(
            adapter, reference, [replace(result, payload=compact_payload(result.payload))]
        )[0]
    else:
        outcome = adapter.read_record(
            reference, result.native_identity, result.digest, locator=result.locator
        )
    assert fired
    assert outcome.status is (Status.CHANGED if rewrite else Status.OK)
    assert (outcome.evidence is None) is rewrite
    assert adapter.bytes_read <= adapter.byte_budget


def test_singleton_codex_many_uses_bounded_batch_at_200k_offset(tmp_path):
    from tests.accounting.readers.test_codex_native_hydration import scene

    path, _, _, _, adapter = scene(tmp_path)
    rows = [
        codex_meta("native-child", "/work/team-a/project"),
        codex_turn("t", "/work/team-a/project"),
        {"type": "event_msg", "payload": {"type": "agent_reasoning", "text": "x" * 200_000}},
        codex_output("late", "answer"),
    ]
    write_records(path, rows)
    reader = adapter.readers["codex"]
    original = next(
        item for item in reader.read(reader.source_for(path)).evidence if item.kind == "tool_result"
    )
    replay = EvidenceReplay("org-a", adapter=adapter)
    assert replay.hydrate_many([replace(original, payload=compact_payload(original.payload))]) == [
        original
    ]
    assert not replay.failures


def test_codex_session_id_header_matches_collection_fallback(tmp_path):
    path, reader, _, _, _, adapter = json_scene(tmp_path, "codex")
    header = codex_meta("s", CWD)
    header["payload"]["session_id"] = header["payload"].pop("id")
    write_records(path, [header, codex_turn("t", CWD), codex_output("c", "answer")])
    original = next(
        item for item in reader.read(reader.source_for(path)).evidence if item.kind == "tool_result"
    )
    replay = EvidenceReplay("org", adapter=adapter)
    compact = replace(original, payload=compact_payload(original.payload))
    assert replay.hydrate(compact) == original
    assert replay.hydrate_many([compact]) == [original]


def test_compatible_episode_aliases_read_original_bytes_once_and_conflicting_pin_fails(tmp_path):
    path, _, _, _, reference, adapter = json_scene(tmp_path, "claude")
    first = reference.ref
    alias = "claude:s:ep:1"
    bad = "claude:s:ep:2"
    adapter.references = {
        first: reference,
        alias: replace(reference, ref=alias),
        bad: replace(reference, ref=bad, source_digest="0" * 64, source_bytes=path.stat().st_size),
    }
    original = adapter.resolve(first)
    count = adapter.bytes_read
    adapter.byte_budget = count
    same = adapter.resolve(alias)
    assert original.status is Status.OK and same.status is Status.OK
    assert same.line is original.line and adapter.bytes_read == count
    assert adapter.resolve(bad).status is not Status.OK
    adapter.scoper = Scoper(policy("/elsewhere", profile="other"))
    assert adapter.resolve(alias).status is not Status.OK


def test_late_malformed_context_keeps_independently_verified_batch_prefix(tmp_path):
    path, reader, _, _, reference, adapter = json_scene(tmp_path, "codex")
    write_records(
        path,
        [
            codex_meta("s", CWD),
            codex_turn("t", CWD),
            codex_call("c", "exec_command", {"cmd": "cat log"}),
        ],
    )
    with path.open("ab") as stream:
        stream.write(b"not-native-json\n")
    from tests.accounting.readers.readers_fixtures import jsonl

    with path.open("ab") as stream:
        stream.write(jsonl([codex_output("c", "answer")]))
    original = [
        item
        for item in reader.read(reader.source_for(path)).evidence
        if item.kind in {"tool_call", "tool_result"}
    ]
    from ssgrep.accounting.collect.replay import read_batch

    outcomes = read_batch(
        adapter,
        reference,
        [replace(item, payload=compact_payload(item.payload)) for item in original],
    )
    assert outcomes[0].status is Status.OK and outcomes[0].evidence.identity == original[0].identity
    assert outcomes[1].evidence is None and outcomes[1].status in {
        Status.INCOMPLETE,
        Status.UNREADABLE,
    }


@pytest.mark.parametrize("runtime", ["claude", "codex", "omp", "pi", "prime-agent"])
def test_late_io_failure_preserves_verified_rows_for_every_jsonl_runtime(
    tmp_path, monkeypatch, runtime
):
    _, reader, originals, result, reference, adapter = json_scene(tmp_path, runtime)
    original_decode = reader.decode

    def fail_last(source, obj, context, locator):
        if locator == result.locator:
            raise OSError("later native operation unavailable")
        return original_decode(source, obj, context, locator)

    monkeypatch.setattr(reader, "decode", fail_last)
    from ssgrep.accounting.collect.replay import read_batch

    selected = [item for item in originals if item.kind in {"tool_call", "tool_result"}]
    outcomes = read_batch(
        adapter,
        reference,
        [replace(item, payload=compact_payload(item.payload)) for item in selected],
    )
    assert outcomes[0].status is Status.OK
    assert outcomes[0].evidence.identity == selected[0].identity
    assert outcomes[1].status is Status.UNREADABLE and outcomes[1].evidence is None


@pytest.mark.parametrize("checkpoint", [False, True])
def test_opencode_read_only_snapshot_survives_concurrent_append(tmp_path, monkeypatch, checkpoint):
    path, originals, adapter = database_scene(tmp_path)
    reader = adapter.readers["opencode"]
    target = next(item for item in originals if item.kind == "tool_result")
    original_row = reader._row
    fired = False
    # Establish a native WAL before snapshot creation and keep its writer alive.
    # Closing the last writer is a checkpoint, not an append-only WAL operation.
    connection = opencode_connect(path)
    connection.execute("UPDATE session SET time_updated = time_updated + 1 WHERE id = 's'")
    wal = Path(str(path) + "-wal")
    main_before = path.read_bytes()
    wal_before = wal.read_bytes()

    def append_after_row(source, table, row, held, result):
        nonlocal fired
        original_row(source, table, row, held, result)
        if not fired and any(
            item.native_identity == target.native_identity for item in result.evidence
        ):
            fired = True
            opencode_add_message(connection, "later-m", "s", opencode_assistant_data(), at=10)
            opencode_add_part(
                connection,
                "later-p",
                "later-m",
                "s",
                opencode_tool_part(
                    "later-c", "bash", {"command": "later"}, output="later", start=10, end=11
                ),
                at=11,
            )
            appended = wal.read_bytes()
            assert path.read_bytes() == main_before
            assert len(appended) > len(wal_before) and appended.startswith(wal_before)
            if checkpoint:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                assert path.read_bytes() != main_before

    monkeypatch.setattr(reader, "_row", append_after_row)
    replay = EvidenceReplay("org", adapter=adapter)
    try:
        hydrated = replay.hydrate(replace(target, payload=compact_payload(target.payload)))
        assert fired
        if checkpoint:
            assert hydrated.payload.get("_replay_unavailable") == "source_digest_drift"
            assert replay.failures
        else:
            assert hydrated == target
            assert not replay.failures
    finally:
        connection.close()


def test_opencode_modified_old_record_still_rejects_native_replay(tmp_path):
    import json

    path, originals, adapter = database_scene(tmp_path)
    target = next(item for item in originals if item.kind == "tool_result")
    connection = opencode_connect(path)
    try:
        data = json.loads(connection.execute("SELECT data FROM part WHERE id = 'p'").fetchone()[0])
        data["state"]["output"] = "changed original answer"
        connection.execute("UPDATE part SET data = ? WHERE id = 'p'", (json.dumps(data),))
        replay = EvidenceReplay("org", adapter=adapter)
        hydrated = replay.hydrate(replace(target, payload=compact_payload(target.payload)))
        assert hydrated.payload.get("_replay_unavailable") == "source_digest_drift"
    finally:
        connection.close()


def test_opencode_snapshot_race_is_changed_not_unreadable(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from ssgrep.accounting.readers.opencode_storage import SnapshotChanged

    _, originals, adapter = database_scene(tmp_path)

    @contextmanager
    def changed_snapshot(_path, *, descriptor=None):
        raise SnapshotChanged("native snapshot identity changed")
        yield  # contextmanager protocol, no native data accepted

    monkeypatch.setattr(adapter.readers["opencode"], "snapshot", changed_snapshot)
    target = next(item for item in originals if item.kind == "tool_result")
    replay = EvidenceReplay("org", adapter=adapter)
    hydrated = replay.hydrate(replace(target, payload=compact_payload(target.payload)))
    assert hydrated.payload.get("_replay_unavailable") == "source_digest_drift"


def test_direct_opencode_hydration_uses_native_apsw_connection(tmp_path):
    import apsw

    from ssgrep.accounting.readers.opencode_storage import named_row

    path, originals, adapter = database_scene(tmp_path)
    with adapter.readers["opencode"].snapshot(path) as connection:
        assert isinstance(connection, apsw.Connection)
        connection.set_row_trace(named_row)
        row = connection.execute(
            "SELECT id, directory FROM session WHERE id = ?", ("s",)
        ).fetchone()
        assert row == {"id": "s", "directory": CWD}
    target = next(item for item in originals if item.kind == "tool_result")
    replay = EvidenceReplay("org", adapter=adapter)
    assert replay.hydrate(replace(target, payload=compact_payload(target.payload))) == target


@pytest.mark.parametrize("interrupted", [False, True])
def test_direct_opencode_maps_native_apsw_failures(tmp_path, monkeypatch, interrupted):
    from contextlib import contextmanager

    import apsw

    _, originals, adapter = database_scene(tmp_path)

    @contextmanager
    def failed_snapshot(_path, *, descriptor=None):
        error = apsw.InterruptError if interrupted else apsw.SQLError
        raise error("native SQLite operation failed")
        yield  # contextmanager protocol; never fabricated native evidence

    monkeypatch.setattr(adapter.readers["opencode"], "snapshot", failed_snapshot)
    target = next(item for item in originals if item.kind == "tool_result")
    hydrated = EvidenceReplay("org", adapter=adapter).hydrate(
        replace(target, payload=compact_payload(target.payload))
    )
    expected = "source_read_bound" if interrupted else "source_unreadable"
    assert hydrated.payload.get("_replay_unavailable") == expected


def test_direct_opencode_maps_snapshot_unavailable_to_typed_gap(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from ssgrep.accounting.readers.opencode_storage import SnapshotUnavailable

    _, originals, adapter = database_scene(tmp_path)

    @contextmanager
    def unavailable_snapshot(_path, *, descriptor=None):
        raise SnapshotUnavailable("native readonly VFS unavailable")
        yield  # contextmanager protocol; no source evidence accepted

    monkeypatch.setattr(adapter.readers["opencode"], "snapshot", unavailable_snapshot)
    target = next(item for item in originals if item.kind == "tool_result")
    hydrated = EvidenceReplay("org", adapter=adapter).hydrate(
        replace(target, payload=compact_payload(target.payload))
    )
    assert hydrated.payload.get("_replay_unavailable") == "source_unreadable"


def test_direct_opencode_missing_lazy_apsw_is_typed_unavailable(tmp_path, monkeypatch):
    import builtins

    _, originals, adapter = database_scene(tmp_path)
    original_import = builtins.__import__

    def missing_apsw(name, *args, **kwargs):
        if name == "apsw":
            raise ImportError("native APSW dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_apsw)
    target = next(item for item in originals if item.kind == "tool_result")
    hydrated = EvidenceReplay("org", adapter=adapter).hydrate(
        replace(target, payload=compact_payload(target.payload))
    )
    assert hydrated.payload.get("_replay_unavailable") == "source_unreadable"


def test_direct_opencode_snapshot_receives_already_approved_descriptor(tmp_path, monkeypatch):
    import os
    from contextlib import contextmanager

    path, originals, adapter = database_scene(tmp_path)
    reader = adapter.readers["opencode"]
    original_snapshot = reader.snapshot
    seen = []

    @contextmanager
    def bound_snapshot(database, *, descriptor=None):
        assert descriptor is not None
        approved = os.fstat(descriptor)
        expected = path.stat()
        assert (approved.st_dev, approved.st_ino) == (expected.st_dev, expected.st_ino)
        seen.append(descriptor)
        with original_snapshot(database, descriptor=descriptor) as connection:
            yield connection

    monkeypatch.setattr(reader, "snapshot", bound_snapshot)
    target = next(item for item in originals if item.kind == "tool_result")
    hydrated = EvidenceReplay("org", adapter=adapter).hydrate(
        replace(target, payload=compact_payload(target.payload))
    )
    assert hydrated == target
    assert len(seen) == 1


def historical_alias_scene(tmp_path, original_owner):
    """Native cwd strings are not evidence of an alias's historical target."""
    from ssgrep.accounting.privacy import policy as policy_store
    from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader

    roots = {name: tmp_path.resolve() / name for name in ("org-a", "org-b")}
    for root in roots.values():
        root.mkdir()
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(roots[original_owner], target_is_directory=True)
    policies = {
        name: policy_store.enable_collection(name, {"roots": [str(root)]})
        for name, root in roots.items()
    }
    reader = ClaudeReader({"HOME": str(tmp_path)})
    references = {}
    records = {}
    for name, cwd in (("alias", str(alias)), ("canonical", str(roots[original_owner]))):
        rows = [
            claude_prompt(name, "Inspect the retained work"),
            claude_tool_use(name, "lookup", "Bash", {"command": "ssgrep search fixture"}, cwd=cwd),
            claude_tool_result(name, "lookup", "answer"),
        ]
        rows = [{**row, "cwd": cwd} for row in rows]
        path = write_records(tmp_path / ".claude" / "projects" / name / f"{name}.jsonl", rows)
        records[name] = reader.read(reader.source_for(path)).evidence
        references[name] = NativeSourceReference(
            f"claude:{name}:ep:0", "claude", name, path, cwd=cwd
        )
    return roots, alias, policies, reader, references, records


@pytest.mark.parametrize("original_owner", ["org-a", "org-b"])
@pytest.mark.parametrize("boundary", ["resolve", "record", "batch", "reduce", "migration"])
def test_historical_alias_never_acquires_retargeted_owner(tmp_path, original_owner, boundary):
    from ssgrep.accounting.capture import capture_policy
    from ssgrep.accounting.ledger.reduce import reduce_profile
    from ssgrep.accounting.readers.base import evidence_digest
    from ssgrep.accounting.readers.native_identity import migrate_record

    roots, alias, policies, reader, references, records = historical_alias_scene(
        tmp_path, original_owner
    )
    other = "org-b" if original_owner == "org-a" else "org-a"
    assert capture_policy.capture_profile(str(alias)) == original_owner
    alias.unlink()
    alias.symlink_to(roots[other], target_is_directory=True)
    # A genuinely new live operation still follows the live workspace.
    assert capture_policy.capture_profile(str(alias)) == other
    ownership = PolicyFile(profiles=policies)

    for name in ("canonical", "alias"):
        target = original_owner if name == "canonical" else other
        scoper = Scoper(policies[target], ownership)
        reference = references[name]
        adapter = NativeSourceAdapter(
            {reference.ref: reference}, {"claude": reader}, (tmp_path,), scoper, NOW
        )
        witness = next(item for item in records[name] if item.kind == "tool_call")
        if boundary == "resolve":
            result = adapter.resolve(reference.ref)
        elif boundary == "record":
            result = adapter.read_record(
                reference, witness.native_identity, witness.digest, locator=witness.locator
            )
        elif boundary == "batch":
            from ssgrep.accounting.collect.replay import read_batch

            result = read_batch(
                adapter, reference, [replace(witness, payload=compact_payload(witness.payload))]
            )[0]
        elif boundary == "reduce":
            reduced = reduce_profile(
                target, records[name], [], policy=policies[target], frequency=rare_view()
            )
            assert len(reduced.invocations) == (1 if name == "canonical" else 0)
            if name == "alias":
                assert reduced.coverage["invocations_out_of_scope"] == 1
            continue
        else:
            payload = {
                key: value
                for key, value in witness.payload.items()
                if key in {"call_id", "name", "arguments", "message_id", "cwd"}
            }
            legacy = replace(
                witness,
                payload=payload,
                digest=evidence_digest(witness.kind, payload),
                origin_domain=None,
                origin_id=None,
                origin_status="legacy_unverified",
            )
            migrated = migrate_record(legacy, {"claude": reader}, scoper, 16 * 1024 * 1024)
            if name == "canonical":
                assert migrated.evidence is not None
                assert migrated.evidence.native_key == witness.native_key
                assert migrated.evidence.payload["cwd"] == str(roots[original_owner])
            else:
                assert migrated.evidence is None
                assert migrated.reason == Status.SCOPE_UNKNOWN.value
            continue
        assert result.status is (Status.OK if name == "canonical" else Status.SCOPE_UNKNOWN)
        if isinstance(result, NativeSourceResult):
            if name == "canonical":
                assert result.line is not None
                assert result.line.cwd_at(1) == str(roots[original_owner])
            else:
                assert result.line is None
        elif name == "canonical":
            assert result.evidence is not None
            assert result.evidence.native_key == witness.native_key
        else:
            assert result.evidence is None
