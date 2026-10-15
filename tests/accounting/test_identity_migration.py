"""Legacy IDs migrate by verified original locators, never by importing whole histories."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.contracts import RawEvidence
from ssgrep.accounting.identity_migration import prepare
from ssgrep.accounting.readers.values import evidence_digest
from ssgrep.accounting.timeline import build_timelines
from tests.accounting.readers_fixtures import (
    CWD,
    TS,
    claude_tool_result,
    claude_tool_use,
    jsonl,
    write_records,
)
from tests.accounting.reduce_builders import policy

NOW = datetime(2026, 10, 14, tzinfo=UTC)
A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"


def legacy_pair(path: Path, session: str):
    call = {**claude_tool_use(session, "call_0", "Bash", {"command": "ssgrep search x"}), "uuid": A}
    result = {**claude_tool_result(session, "call_0", "x" * 156), "uuid": B}
    # Extra native history must NOT become consumer usage through migration.
    unretained = claude_tool_use(
        session, "never-retained", "Bash", {"command": "ssgrep search old"}
    )
    write_records(path, [call, result, unretained])
    payloads = [
        {
            "call_id": "call_0",
            "name": "Bash",
            "arguments": {"command": "ssgrep search x"},
            "message_id": "msg_1",
            "cwd": CWD,
        },
        {"call_id": "call_0", "output": "x" * 156, "output_bytes": 156, "is_error": None},
    ]
    offset = 0
    evidence = []
    for kind, rid, payload, row in zip(
        ("tool_call", "tool_result"),
        ("call:call_0", "result:call_0"),
        payloads,
        (call, result),
        strict=True,
    ):
        evidence.append(
            RawEvidence(
                "claude",
                f"claude:{path}",
                "main",
                session,
                rid,
                f"offset:{offset}",
                evidence_digest(kind, payload),
                TS,
                kind,
                payload=payload,
                origin_status="legacy_unverified",
            )
        )
        offset += len(jsonl([row]))
    return evidence


def setup(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "claude" / "projects" / "project"
    env = {
        "HOME": str(tmp_path / "home"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "PI_SESSION_DIR": str(tmp_path / "pi"),
        "OMP_SESSIONS_DIR": str(tmp_path / "omp"),
        "PRIME_AGENT_SESSION_DIR": str(tmp_path / "prime"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "SSGREP_OPENCODE_DB": str(tmp_path / "opencode.db"),
    }
    return root, env


def test_verified_legacy_copies_replace_current_evidence_once_and_archive_old_namespace(
    tmp_path, monkeypatch
):
    root, env = setup(tmp_path, monkeypatch)
    evidence = legacy_pair(root / "a.jsonl", "first") + legacy_pair(root / "b.jsonl", "second")
    refs = store.write_segments("personal-local", evidence)
    old_blob = store.write_blob("personal-local", "reduction", {"legacy": "double-counted"})
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=refs,
        blobs={"reduction": old_blob},
        methodology="old-method",
    )
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert result.complete and result.changed
    assert checkpoint.identity_method == "native_identity_v2"
    assert "reduction" not in checkpoint.blobs
    assert old_blob in checkpoint.identity_migration["superseded_blobs"].values()
    retained = list(store.iter_evidence("personal-local", checkpoint.segments))
    from ssgrep.accounting.collector_read import make_readers
    from ssgrep.accounting.evidence_replay import EvidenceReplay
    from ssgrep.accounting.native_source import NativeSourceAdapter
    from ssgrep.accounting.scope_events import Scoper

    readers = make_readers(env)
    adapter = NativeSourceAdapter(
        {}, readers, (root,), Scoper(policy()), int(NOW.timestamp() * 1000)
    )
    replay = EvidenceReplay("personal-local", adapter=adapter)
    hydrated = [replay.hydrate(e) for e in retained]
    view = build_timelines(hydrated)
    rows = [r for line in view.sessions.values() for r in line.records]
    assert len(rows) == 2
    assert sum(len(r.ev.payload.get("output", "")) for r in rows) // 4 == 39
    assert not any(e.payload.get("call_id") == "never-retained" for e in retained)
    assert not prepare("personal-local", checkpoint, NOW, env=env, policy=policy()).changed


def test_missing_original_blocks_migration_without_replacing_authoritative_legacy_totals(
    tmp_path, monkeypatch
):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "gone.jsonl"
    evidence = legacy_pair(path, "first")
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=store.write_segments("personal-local", evidence),
        methodology="old-method",
    )
    previous = list(checkpoint.segments)
    path.unlink()
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert not result.complete
    assert result.unavailable_sessions[("claude", "first")] == "source_missing"
    assert checkpoint.segments == previous and checkpoint.methodology == "old-method"
    assert checkpoint.identity_method != "native_identity_v2"


def test_changed_original_is_not_reinterpreted_under_legacy_digest(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "changed.jsonl"
    evidence = legacy_pair(path, "first")
    checkpoint = store.Checkpoint(
        "personal-local", generation=1, segments=store.write_segments("personal-local", evidence)
    )
    path.write_bytes(path.read_bytes().replace(b"ssgrep search x", b"ssgrep search z"))
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert not result.complete
    assert result.unavailable_sessions[("claude", "first")] == "source_digest_drift"


def test_empty_profile_marks_v2_without_import_or_epoch_rotation(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    checkpoint = store.Checkpoint("personal-local")
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert result.complete and result.changed
    assert checkpoint.identity_method == "native_identity_v2"
    assert checkpoint.identity_migration["status"] == "complete"
    assert not checkpoint.segments and not checkpoint.blobs
    assert result.epoch_reason == "unchanged_journal_identity"


def test_partial_migration_is_resumable_without_mixing_current_ids(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    evidence = legacy_pair(root / "a.jsonl", "first")
    checkpoint = store.Checkpoint(
        "personal-local", generation=1, segments=store.write_segments("personal-local", evidence)
    )
    original = list(checkpoint.segments)
    first = prepare("personal-local", checkpoint, NOW, env=env, policy=policy(), max_records=1)
    assert not first.complete and checkpoint.segments == original
    assert checkpoint.identity_migration["record"] == 1
    resumed = store.Checkpoint.from_dict(checkpoint.to_dict())
    second = prepare("personal-local", resumed, NOW, env=env, policy=policy(), max_records=1)
    assert second.complete and resumed.identity_method == "native_identity_v2"
    assert len(list(store.iter_evidence("personal-local", resumed.segments))) == 2


def test_native_correction_preserves_valid_journal_counter_epoch_and_values(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    evidence = legacy_pair(root / "a.jsonl", "first")
    from ssgrep.accounting.counters import ExecKey, LiveCounters

    counters = LiveCounters(
        3, "2026-10-14T00:00:00Z", executions=((ExecKey("search", "cli", "ok", "claude"), 7),)
    ).to_dict()
    counter_blob = store.write_blob("personal-local", "counters", counters)
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=store.write_segments("personal-local", evidence),
        blobs={"counters": counter_blob},
    )
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert result.complete and result.epoch_reason == "unchanged_journal_identity"
    assert checkpoint.blobs["counters"] == counter_blob
    assert store.read_blob("personal-local", counter_blob) == counters


def test_purge_discards_all_interrupted_migration_recovery_pointers(tmp_path, monkeypatch):
    from ssgrep.accounting.identity_migration import reset_after_purge

    setup(tmp_path, monkeypatch)
    checkpoint = store.Checkpoint(
        "personal-local",
        identity_migration={
            "target": "native_identity_v2",
            "status": "pending",
            "record": 13,
            "staged_segments": [{"name": "discarded"}],
            "compact_legacy_segments": [{"name": "erased"}],
            "superseded_blobs": {"ledger": "erased-ledger"},
            "unavailable": [["claude", "deleted", "missing"]],
        },
    )
    reset_after_purge(checkpoint, [], {})
    assert not checkpoint.segments and not checkpoint.blobs
    assert checkpoint.identity_migration["record"] == 0
    assert checkpoint.identity_migration["staged_segments"] == []
    assert checkpoint.identity_migration["compact_legacy_segments"] == []
    assert "superseded_blobs" not in checkpoint.identity_migration


def test_legacy_pi_eof_cursor_recovers_header_without_usage_import(tmp_path, monkeypatch):
    from ssgrep.accounting.readers.base import SourceState, checkpoint as source_checkpoint
    from ssgrep.accounting.readers.pi import PiReader
    from ssgrep.accounting.readers.source_io import open_source
    from tests.accounting.readers_fixtures import append_bytes, pi_header, pi_tool_call

    root, env = setup(tmp_path, monkeypatch)
    path = write_records(
        tmp_path / "pi" / "wrong-stem.jsonl",
        [pi_header("native-session", parentSessionId="native-parent")],
    )
    reader = PiReader(env)
    source = reader.source_for(path)
    offset = path.stat().st_size
    with open_source(path) as handle:
        legacy_state = source_checkpoint(handle, offset, {})
    record = store.SourceRecord(
        "pi", source.source_id, state=legacy_state.to_dict(), baseline="skipped"
    )
    checkpoint = store.Checkpoint("personal-local", generation=0, sources={record.key: record})
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert result.complete and checkpoint.sources[record.key].offset == offset
    assert not checkpoint.segments
    repaired = SourceState.from_dict(checkpoint.sources[record.key].state)
    assert repaired.extra["session"] == "native-session"
    assert repaired.extra["header"]["parent_session_id"] == "native-parent"
    append_bytes(path, jsonl([pi_tool_call("e", "call_0", "bash", {"command": "ssgrep search x"})]))
    live = reader.read(source, repaired).evidence
    assert len(live) == 1 and live[0].session_id == "native-session"


def test_current_method_without_completion_proof_stays_explicitly_unavailable(
    tmp_path, monkeypatch
):
    root, env = setup(tmp_path, monkeypatch)
    checkpoint = store.Checkpoint("personal-local", identity_method="native_identity_v2")
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert not result.complete and not result.changed
    assert result.warnings == ("identity_migration_proof_unavailable",)
    assert checkpoint.identity_migration == {}


def old_pi_history(path: Path, runtime: str):
    """Literal payload/ID shapes from published407e3c29 PiFamilyReader."""
    from tests.accounting.readers_fixtures import (
        pi_header,
        pi_tool_call,
        pi_tool_result,
        pi_user,
    )

    rows = [
        pi_header("native-session", parentSessionId="parent-session"),
        pi_user("turn-entry", "find the retained symbol"),
        pi_tool_call(
            "call-entry",
            "call_0",
            "bash",
            {"command": "ssgrep search x"},
            usage={"input": 17, "output": 5, "totalTokens": 22},
        ),
        pi_tool_result("result-entry", "call_0", "retained-é"),
        pi_tool_call("unretained-entry", "never-retained", "bash", {"command": "old"}),
    ]
    rows[2]["parentId"] = "turn-entry"
    rows[3]["parentId"] = "call-entry"
    write_records(path, rows)
    shapes = [
        (1, "user_turn", "turn:turn-entry", {"text": "find the retained symbol"}),
        (
            2,
            "tool_call",
            "call:call_0",
            {
                "call_id": "call_0",
                "name": "bash",
                "arguments": {"command": "ssgrep search x"},
                "entry_id": "call-entry",
                "cwd": CWD,
                "occurrence": 0,
            },
        ),
        (
            2,
            "usage",
            "usage:call-entry",
            {
                "tokens": {"input_tokens": 17, "output_tokens": 5, "total_tokens": 22},
                "cost_nano_usd": None,
                "model": "pi-test",
            },
        ),
        (
            3,
            "tool_result",
            "result:call_0",
            {
                "occurrence": 0,
                "call_id": "call_0",
                "tool_name": "any",
                "output": "retained-é",
                "output_bytes": len("retained-é".encode()),
                "is_error": False,
            },
        ),
    ]
    return [
        RawEvidence(
            runtime,
            f"{runtime}:{path}",
            "main",
            "native-session",
            rid,
            f"offset:{len(jsonl(rows[:index]))}",
            evidence_digest(kind, payload),
            TS,
            kind,
            payload=payload,
            origin_status="legacy_unverified",
        )
        for index, kind, rid, payload in shapes
    ]


def migrate_original(item, env, *, budget=16 * 1024 * 1024, scoped_policy=None):
    from ssgrep.accounting.collector_read import make_readers
    from ssgrep.accounting.identity_migration_record import migrate_record
    from ssgrep.accounting.scope_events import Scoper

    return migrate_record(item, make_readers(env), Scoper(scoped_policy or policy()), budget)


def test_pi_family_old_call_result_turn_usage_migrate_without_recount(tmp_path, monkeypatch):
    from ssgrep.accounting.counters import ExecKey, LiveCounters

    _, env = setup(tmp_path, monkeypatch)
    for runtime, directory in (("pi", "pi"), ("omp", "omp"), ("prime-agent", "prime")):
        originals = old_pi_history(tmp_path / directory / "original.jsonl", runtime)
        copies = old_pi_history(tmp_path / directory / "copy.jsonl", runtime)
        migrated = [migrate_original(item, env).evidence for item in originals + copies]
        assert all(item is not None and item.origin_domain == "native-session" for item in migrated)
        view = build_timelines(migrated)
        records = [record for line in view.sessions.values() for record in line.records]
        assert len(records) == 4
        by_kind = {record.ev.kind: record for record in records}
        line = next(iter(view.sessions.values()))
        assert line.result_for(by_kind["tool_call"]) is by_kind["tool_result"]
        assert line.call_for(by_kind["tool_result"]) is by_kind["tool_call"]
        assert by_kind["tool_result"].ev.payload["output_bytes"] == len("retained-é".encode())
        assert by_kind["usage"].ev.payload["tokens"]["total_tokens"] == 22
        assert by_kind["user_turn"].ev.payload["text"] == "find the retained symbol"
        assert all(record.ev.payload["parent_session"] == "parent-session" for record in records)
        counters = LiveCounters(
            3, "2026-10-14T00:00:00Z", executions=((ExecKey("search", "cli", "ok", runtime), 7),)
        ).to_dict()
        blob = store.write_blob("personal-local", "counters", counters)
        checkpoint = store.Checkpoint(
            "personal-local",
            generation=1,
            segments=store.write_segments("personal-local", originals + copies),
            blobs={"counters": blob},
        )
        result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
        assert result.complete
        retained = list(store.iter_evidence("personal-local", checkpoint.segments))
        assert {item.native_key for item in retained} == {item.native_key for item in migrated}
        assert len({item.native_key for item in retained}) == 4
        assert all(item.origin_id != "call:never-retained@unretained-entry:0" for item in retained)
        assert checkpoint.blobs["counters"] == blob
        assert store.read_blob("personal-local", blob) == counters
        assert not prepare("personal-local", checkpoint, NOW, env=env, policy=policy()).changed


def test_claude_old_child_payload_retains_parent_and_bidirectional_pairing(tmp_path, monkeypatch):
    from dataclasses import replace

    root, env = setup(tmp_path, monkeypatch)
    originals = legacy_pair(root / "parent" / "subagents" / "agent-child.jsonl", "parent")
    originals = [
        replace(
            item,
            session_id="parent/child",
            branch_id="child:child",
            payload={**item.payload, "parent_session": "parent"},
            digest=evidence_digest(item.kind, {**item.payload, "parent_session": "parent"}),
        )
        for item in originals
    ]
    migrated = [migrate_original(item, env).evidence for item in originals]
    assert all(item is not None for item in migrated)
    line = next(iter(build_timelines(migrated).sessions.values()))
    call = next(record for record in line.records if record.ev.kind == "tool_call")
    result = line.result_for(call)
    assert result is not None and line.call_for(result) is call
    assert all(
        record.ev.session_id == "parent/child"
        and record.ev.branch_id == "child:child"
        and record.ev.payload["parent_session"] == "parent"
        for record in line.records
    )
    assert result.ev.payload["output"] == "x" * 156


def test_out_of_scope_original_cannot_replace_legacy_provenance(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    originals = legacy_pair(root / "scoped.jsonl", "first")
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=store.write_segments("personal-local", originals),
        methodology="published-method",
        input_digest="published-input",
    )
    before = [item.to_dict() for item in checkpoint.segments]
    result = prepare(
        "personal-local", checkpoint, NOW, env=env, policy=policy(str(tmp_path / "other-project"))
    )
    assert not result.complete and result.bytes_read > 0
    assert result.unavailable_sessions == {("claude", "first"): "source_out_of_scope"}
    assert [item.to_dict() for item in checkpoint.segments] == before
    assert (
        checkpoint.methodology == "published-method"
        and checkpoint.input_digest == "published-input"
    )
    retained = list(store.iter_evidence("personal-local", checkpoint.segments))
    assert [(item.digest, item.locator, item.origin_domain) for item in retained] == [
        (item.digest, item.locator, None) for item in originals
    ]


def test_ambiguous_old_digest_does_not_choose_a_native_call(tmp_path, monkeypatch):
    import json

    _, env = setup(tmp_path, monkeypatch)
    path = tmp_path / "pi" / "ambiguous.jsonl"
    original = next(item for item in old_pi_history(path, "pi") if item.kind == "tool_call")
    rows = [json.loads(raw) for raw in path.read_bytes().splitlines()]
    rows[2]["message"]["content"] *= 2
    write_records(path, rows)
    result = migrate_original(original, env)
    assert result.evidence is None and result.reason == "source_digest_drift"
    assert result.bytes_read == len(jsonl(rows[2:3])) + 2 * len(jsonl(rows[:2]))
    rows[2]["message"]["content"].pop()
    write_records(path, rows)
    verified = migrate_original(original, env)
    assert verified.evidence is not None
    assert verified.evidence.payload["call_id"] == "call_0"


def test_metadata_window_edges_preserve_native_session_without_import(tmp_path, monkeypatch):
    from ssgrep.accounting.native_identity import CONTEXT_BYTES
    from tests.accounting.readers_fixtures import pi_header, pi_user

    _, env = setup(tmp_path, monkeypatch)
    header = jsonl([pi_header("native-session")])
    turn = jsonl([pi_user("edge-turn", "retained turn")])
    for offset in (CONTEXT_BYTES - 1, CONTEXT_BYTES, CONTEXT_BYTES + 1):
        path = tmp_path / "omp" / f"edge-{offset}.jsonl"
        padding = (
            b'{"type":"custom","padding":"'
            + b"x" * (offset - len(header) - len(b'{"type":"custom","padding":""}\n'))
            + b'"}\n'
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header + padding + turn)
        payload = {"text": "retained turn"}
        original = RawEvidence(
            "omp",
            f"omp:{path}",
            "main",
            "native-session",
            "turn:edge-turn",
            f"offset:{offset}",
            evidence_digest("user_turn", payload),
            TS,
            "user_turn",
            payload=payload,
            origin_status="legacy_unverified",
        )
        result = migrate_original(original, env)
        assert result.evidence is not None
        assert result.evidence.origin_domain == "native-session"
        assert result.evidence.payload["text"] == "retained turn"
        assert result.bytes_read == len(turn) + 2 * min(offset, CONTEXT_BYTES)


def test_unterminated_and_oversized_originals_charge_bytes_without_publishing(
    tmp_path, monkeypatch
):
    from ssgrep.accounting.contracts import MAX_RAW_RECORD_BYTES

    root, env = setup(tmp_path, monkeypatch)
    path = root / "bounded.jsonl"
    original = legacy_pair(path, "first")[0]
    valid = path.read_bytes().splitlines(keepends=True)[0]
    for damaged, charged in (
        (valid.rstrip(b"\n"), len(valid) - 1),
        (b"x" * (MAX_RAW_RECORD_BYTES + 1), MAX_RAW_RECORD_BYTES + 1),
    ):
        path.write_bytes(damaged)
        result = migrate_original(original, env)
        assert result.evidence is None and result.reason == "source_read_bound"
        assert result.bytes_read == charged
        assert original.origin_domain is None
    path.write_bytes(valid)
    assert migrate_original(original, env).evidence is not None


def test_small_budget_keeps_legacy_active_until_charged_work_resumes(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "budgeted.jsonl"
    originals = legacy_pair(path, "first")
    checkpoint = store.Checkpoint(
        "personal-local", generation=1, segments=store.write_segments("personal-local", originals)
    )
    previous = list(checkpoint.segments)
    result = prepare(
        "personal-local", checkpoint, NOW, env=env, policy=policy(), byte_budget=1, max_records=1
    )
    assert not result.complete and result.warnings == ("identity_migration_pending",)
    assert result.bytes_read == len(path.read_bytes().splitlines(keepends=True)[0])
    assert checkpoint.segments == previous and checkpoint.identity_method != "native_identity_v2"
    resumed = store.Checkpoint.from_dict(checkpoint.to_dict())
    completed = prepare("personal-local", resumed, NOW, env=env, policy=policy())
    assert completed.complete and completed.bytes_read > 0
    assert len(list(store.iter_evidence("personal-local", resumed.segments))) == 2


def test_unapproved_original_never_acquires_native_provenance(tmp_path, monkeypatch):
    _, env = setup(tmp_path, monkeypatch)
    originals = legacy_pair(tmp_path / "outside-approved-roots" / "original.jsonl", "first")
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=store.write_segments("personal-local", originals),
        methodology="published-method",
        input_digest="published-input",
    )
    previous = list(checkpoint.segments)
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert not result.complete
    assert result.unavailable_sessions == {("claude", "first"): "source_unapproved"}
    assert result.bytes_read == 0
    assert checkpoint.segments == previous
    assert checkpoint.methodology == "published-method"
    assert checkpoint.input_digest == "published-input"
    assert all(
        item.origin_domain is None
        for item in store.iter_evidence("personal-local", checkpoint.segments)
    )


def test_digest_drift_keeps_original_records_and_publication_authority(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "drifted.jsonl"
    originals = legacy_pair(path, "first")
    publication = store.write_blob("personal-local", "reduction", {"legacy_quantity": 39})
    checkpoint = store.Checkpoint(
        "personal-local",
        generation=1,
        segments=store.write_segments("personal-local", originals),
        blobs={"reduction": publication},
        methodology="published-method",
        input_digest="published-input",
    )
    previous = list(checkpoint.segments)
    path.write_bytes(path.read_bytes().replace(b"ssgrep search x", b"ssgrep search z"))
    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())
    assert not result.complete and result.bytes_read > 0
    assert result.unavailable_sessions == {("claude", "first"): "source_digest_drift"}
    assert checkpoint.segments == previous
    assert checkpoint.blobs["reduction"] == publication
    assert store.read_blob("personal-local", publication) == {"legacy_quantity": 39}
    assert checkpoint.methodology == "published-method"
    assert checkpoint.input_digest == "published-input"
    retained = list(store.iter_evidence("personal-local", checkpoint.segments))
    assert [(item.digest, item.locator, item.origin_domain) for item in retained] == [
        (item.digest, item.locator, None) for item in originals
    ]


def test_invalid_offsets_and_insufficient_window_budget_never_guess_origin(tmp_path, monkeypatch):
    from dataclasses import replace

    from ssgrep.accounting.native_identity import CONTEXT_BYTES

    root, env = setup(tmp_path, monkeypatch)
    path = root / "offsets.jsonl"
    original = legacy_pair(path, "first")[0]
    for locator, reason in (
        ("offset:-1", "source_read_bound"),
        ("offset:not-an-integer", "source_identity_mismatch"),
        (f"offset:{path.stat().st_size}", "source_read_bound"),
    ):
        result = migrate_original(replace(original, locator=locator), env)
        assert result.evidence is None and result.reason == reason
    denied = migrate_original(original, env, budget=2 * CONTEXT_BYTES)
    assert denied.evidence is None and denied.reason == "source_read_bound"
    assert denied.bytes_read == 0
    allowed = migrate_original(original, env, budget=2 * CONTEXT_BYTES + 4096)
    assert allowed.evidence is not None and allowed.bytes_read > 0
