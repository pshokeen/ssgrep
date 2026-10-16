"""Native and migrated identity behavior, missing-origin safety, and conflicts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting.collect import store
from ssgrep.accounting.collect.read import baseline_record
from ssgrep.accounting.collect.replay import evidence_from_line, evidence_line
from ssgrep.accounting.ledger.contracts import RawEvidence
from ssgrep.accounting.readers.base import EvidenceLedger, SourceState, evidence_digest
from ssgrep.accounting.readers.identity_migration import prepare
from ssgrep.accounting.readers.jsonl_runtimes import ClaudeReader, CodexReader, total_usage
from ssgrep.accounting.readers.native_identity import native_in_scope
from ssgrep.accounting.readers.pi_family import OmpReader, PiReader, PrimeReader, build_lineage
from ssgrep.accounting.readers.timeline import build_timelines
from ssgrep.accounting.views.filter import Scoper
from tests.accounting.ledger.reduce_builders import policy
from tests.accounting.readers.readers_fixtures import (
    CWD,
    TS,
    append_bytes,
    claude_tool_result,
    claude_tool_use,
    codex_call,
    codex_meta,
    codex_token_count,
    codex_turn,
    jsonl,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    pi_user,
    write_records,
)


def test_reader_type_of_distinguishes_untyped_objects_from_non_objects() -> None:
    from ssgrep.accounting.readers.base import JsonlReader

    assert JsonlReader.type_of({"type": "message"}) == "message"
    assert JsonlReader.type_of({"type": "x" * 70}) == "x" * 64
    assert JsonlReader.type_of({"value": 1}) == "<untyped>"
    assert JsonlReader.type_of([]) == "<non-object>"


def test_native_result_without_explicit_call_link_is_not_guessed_by_timeline() -> None:
    call = RawEvidence(
        "pi",
        "pi:/session.jsonl",
        "main",
        "session",
        "call:one",
        "offset:0",
        "call-digest",
        TS,
        "tool_call",
        payload={"call_id": "one"},
    )
    result = RawEvidence(
        "pi",
        "pi:/session.jsonl",
        "main",
        "session",
        "result:one",
        "offset:1",
        "result-digest",
        TS,
        "tool_result",
        payload={"call_id": "one"},
    )
    timelines = build_timelines([call, result])
    timeline = timelines.sessions[("pi", "session")]
    assert timeline.call_by_result == {}


def test_timeline_timestamp_conversion_rejects_invalid_or_timezone_naive_values():
    from ssgrep.accounting.readers.timeline import day_of, ts_ms

    assert ts_ms("2026-10-14T12:30:45.123Z") == 1_791_981_045_123
    assert day_of(ts_ms("2026-10-14T12:30:45.123Z")) == "2026-10-14"
    assert ts_ms("not-a-timestamp") is None
    assert ts_ms("2026-10-14T12:30:45") is None


def test_timeline_session_key_keeps_runtime_and_session_identity() -> None:
    from ssgrep.accounting.readers.timeline import session_key

    item = RawEvidence(
        "pi",
        "pi:/sessions/one.jsonl",
        "main",
        "same-session",
        "turn:1",
        "offset:0",
        "digest",
        TS,
        "user_turn",
        payload={"text": "hello"},
    )
    assert session_key(item) == ("pi", "same-session")


# --- test_identity_migration.py ---


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
    from ssgrep.accounting.collect.read import make_readers
    from ssgrep.accounting.collect.replay import EvidenceReplay
    from ssgrep.accounting.readers.native_source import NativeSourceAdapter
    from ssgrep.accounting.views.filter import Scoper

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
    from ssgrep.accounting.ledger.ledger_contracts import ExecKey, LiveCounters

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
    from ssgrep.accounting.readers.identity_migration import reset_after_purge

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
    from ssgrep.accounting.readers.base import (
        SourceState,
        checkpoint as source_checkpoint,
        open_source,
    )
    from ssgrep.accounting.readers.pi_family import PiReader
    from tests.accounting.readers.readers_fixtures import append_bytes, pi_header, pi_tool_call

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
    from tests.accounting.readers.readers_fixtures import (
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
    from ssgrep.accounting.collect.read import make_readers
    from ssgrep.accounting.readers.native_identity import migrate_record
    from ssgrep.accounting.views.filter import Scoper

    return migrate_record(item, make_readers(env), Scoper(scoped_policy or policy()), budget)


def test_pi_family_old_call_result_turn_usage_migrate_without_recount(tmp_path, monkeypatch):
    from ssgrep.accounting.ledger.ledger_contracts import ExecKey, LiveCounters

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
    from ssgrep.accounting.readers.native_identity import CONTEXT_BYTES
    from tests.accounting.readers.readers_fixtures import pi_header, pi_user

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
    from ssgrep.accounting.ledger.contracts import MAX_RAW_RECORD_BYTES

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

    from ssgrep.accounting.readers.native_identity import CONTEXT_BYTES

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


# --- test_native_identity.py ---


A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"
C = "33333333-3333-4333-8333-333333333333"
D = "44444444-4444-4444-8444-444444444444"

# Property table: origin, not output equality, determines equivalence.
PROPERTIES = (
    ("copied UUID under another wrapper", True),
    ("new UUID with identical output", False),
    ("reused local tool ID without UUID", False),
    ("same UUID with changed content", "conflict"),
    ("Pi entry in same native session across read modes", True),
    ("Pi entry in another native session", False),
    ("headerless source with another path", False),
)


def claude_pair(session: str, call_uuid: str = A, result_uuid: str = B):
    call = claude_tool_use(session, "call_0", "Bash", {"command": "ssgrep search needle"})
    result = claude_tool_result(session, "call_0", "x" * 156)
    return [{**call, "uuid": call_uuid}, {**result, "uuid": result_uuid}]


def rows(view):
    return [r for line in view.sessions.values() for r in line.records]


def test_claude_native_copy_preserves_ownership_but_merges_origin(tmp_path: Path):
    reader = ClaudeReader()
    original = write_records(tmp_path / "original.jsonl", claude_pair("original"))
    copied = write_records(tmp_path / "resumed.jsonl", claude_pair("resumed"))
    one = reader.read(reader.source_for(original)).evidence
    two = reader.read(reader.source_for(copied)).evidence
    assert one[0].session_id == "original" and two[0].session_id == "resumed"
    assert one[0].source_id != two[0].source_id
    assert [e.native_key for e in one] == [e.native_key for e in two]
    ledger = EvidenceLedger()
    assert len(list(ledger.drain(one + two))) == 2
    for order in (one + two, two + one):
        view = build_timelines(order)
        assert len(rows(view)) == 2 and view.duplicates == 2
        assert (
            sum(len(r.ev.payload["output"]) // 4 for r in rows(view) if r.kind == "tool_result")
            == 39
        )


def test_claude_genuinely_new_uuids_with_same_local_call_and_output_count_twice(tmp_path: Path):
    reader = ClaudeReader()
    path = write_records(tmp_path / "same.jsonl", claude_pair("same") + claude_pair("same", C, D))
    evidence = reader.read(reader.source_for(path)).evidence
    assert len({e.native_key for e in evidence}) == 4
    assert len({e.record_id for e in evidence}) == 4
    view = build_timelines(evidence)
    assert len(rows(view)) == 4 and view.conflicts == 0
    results = [r.ev for r in rows(view) if r.kind == "tool_result"]
    assert len({e.payload["call_record_id"] for e in results}) == 2


def test_claude_uuid_conflict_is_one_origin_not_second_charge(tmp_path: Path):
    reader = ClaudeReader()
    one = write_records(tmp_path / "one.jsonl", claude_pair("one"))
    changed = claude_pair("two")
    changed[1]["message"]["content"][0]["content"] = "mutated"
    two = write_records(tmp_path / "two.jsonl", changed)
    evidence = (
        reader.read(reader.source_for(one)).evidence + reader.read(reader.source_for(two)).evidence
    )
    ledger = EvidenceLedger()
    assert len(list(ledger.drain(evidence))) == 2 and ledger.conflict_count == 1
    view = build_timelines(reversed(evidence))
    assert view.conflicts == 1
    assert next(r for r in rows(view) if r.kind == "tool_result").identity in view.conflicted


def test_claude_local_non_uuid_ids_do_not_merge_across_child_ownership(tmp_path: Path):
    reader = ClaudeReader()
    paths = [
        write_records(
            tmp_path / "subagents" / f"agent-{name}.jsonl",
            claude_pair("parent", "not-a-uuid", "also-local"),
        )
        for name in ("a", "b")
    ]
    evidence = [e for path in paths for e in reader.read(reader.source_for(path)).evidence]
    assert len({e.native_key for e in evidence}) == 4
    assert all(e.origin_status == "session_scoped" for e in evidence)
    assert {e.branch_id for e in evidence} == {"child:a", "child:b"}


@pytest.mark.parametrize("reader_type", [PiReader, OmpReader, PrimeReader])
def test_pi_eof_baseline_restart_and_backfill_share_native_ids(tmp_path: Path, reader_type):
    reader = reader_type()
    path = write_records(
        tmp_path / "filename-not-native.jsonl",
        [
            pi_header("native-session", parentSessionId="parent", cwd="/work/child"),
            pi_tool_call("old-call", "call_0", "bash", {"command": "old"}, usage={"input": 100}),
            pi_tool_result("old-result", "call_0", "old output"),
        ],
    )
    source = reader.source_for(path)
    baseline = baseline_record(reader, source)
    state = SourceState.from_dict(baseline.state)
    assert state.extra["session"] == "native-session"
    assert state.extra["cwd"] == "/work/child"
    assert state.extra["header"]["parent_session_id"] == "parent"
    append_bytes(
        path,
        jsonl(
            [
                pi_tool_call("new-call", "call_0", "bash", {"command": "ssgrep search needle"}),
                pi_tool_result("new-result", "call_0", "x" * 156),
            ]
        ),
    )
    live = reader_type().read(source, SourceState.from_dict(state.to_dict())).evidence
    full = reader.read(source).evidence
    expected = [e for e in full if e.payload.get("entry_id") in {"new-call", "new-result"}]
    assert [e.identity for e in live] == [e.identity for e in expected]
    assert all(e.session_id == "native-session" for e in live)
    assert live[0].payload["cwd"] == "/work/child"
    assert live[1].payload["call_record_id"] == live[0].record_id
    assert not any(e.kind == "usage" for e in live)
    for order in (live + full + full, full + live + full):
        view = build_timelines(order)
        assert view.conflicts == 0
        new_results = [r for r in rows(view) if r.ev.payload.get("entry_id") == "new-result"]
        assert len(new_results) == 1 and len(new_results[0].ev.payload["output"]) // 4 == 39


def test_pi_pending_prebaseline_call_links_without_importing_old_usage(tmp_path: Path):
    reader = PiReader()
    path = write_records(
        tmp_path / "file.jsonl",
        [
            pi_header("native"),
            pi_tool_call(
                "call-entry", "call_0", "bash", {"command": "ssgrep search x"}, usage={"input": 900}
            ),
        ],
    )
    source = reader.source_for(path)
    state = SourceState.from_dict(baseline_record(reader, source).state)
    append_bytes(path, jsonl([pi_tool_result("result-entry", "call_0", "result")]))
    live = reader.read(source, state).evidence
    full = reader.read(source).evidence
    assert len(live) == 1 and live[0].kind == "tool_result"
    full_result = next(item for item in full if item.kind == "tool_result")
    full_call = next(item for item in full if item.kind == "tool_call")
    assert live[0].identity == full_result.identity
    assert live[0].payload["call_record_id"] == full_call.record_id


def test_pi_headerless_entries_are_deterministic_but_not_cross_source_proof(tmp_path: Path):
    reader = PiReader()
    records = [pi_tool_call("entry", "call_0", "bash", {"command": "same"})]
    a = write_records(tmp_path / "a" / "same.jsonl", records)
    b = write_records(tmp_path / "b" / "same.jsonl", records)
    one = reader.read(reader.source_for(a)).evidence[0]
    two = reader.read(reader.source_for(b)).evidence[0]
    assert one.origin_status == two.origin_status == "missing_session_id"
    assert one.native_key != two.native_key
    assert one.identity == reader.read(reader.source_for(a)).evidence[0].identity


def test_pi_missing_entry_ids_use_offset_not_reused_call_id(tmp_path: Path):
    reader = PiReader()
    call = pi_tool_call("entry", "call_0", "bash", {"command": "same"})
    del call["id"]
    path = write_records(tmp_path / "native.jsonl", [pi_header("native"), call, call])
    calls = [e for e in reader.read(reader.source_for(path)).evidence if e.kind == "tool_call"]
    assert len({e.native_key for e in calls}) == 2
    assert all(e.origin_status == "missing_entry_id" for e in calls)


def test_pi_parallel_reused_tool_ids_without_parent_link_remain_ambiguous(tmp_path: Path):
    reader = PiReader()
    path = write_records(
        tmp_path / "native.jsonl",
        [
            pi_header("native"),
            pi_tool_call("a", "call_0", "bash", {"command": "one"}),
            pi_tool_call("b", "call_0", "bash", {"command": "two"}),
            pi_tool_result("r", "call_0", "unknown which"),
        ],
    )
    result = reader.read(reader.source_for(path)).evidence[-1]
    assert result.payload["call_record_id"] is None
    assert result.payload["call_link_status"] == "ambiguous"


def test_timeline_pairing_is_bidirectional_and_rejects_explicit_ambiguity(tmp_path: Path):
    reader = PiReader()
    result = pi_tool_result("r", "call_0", "paired")
    result["parentId"] = "b"
    path = write_records(
        tmp_path / "native.jsonl",
        [
            pi_header("native"),
            pi_tool_call("a", "call_0", "bash", {"command": "one"}),
            pi_tool_call("b", "call_0", "bash", {"command": "two"}),
            result,
        ],
    )
    evidence = reader.read(reader.source_for(path)).evidence
    line = build_timelines(evidence).sessions[("pi", "native")]
    calls = [r for r in line.records if r.kind == "tool_call"]
    exposed = next(r for r in line.records if r.kind == "tool_result")
    assert line.result_for(calls[0]) is None
    assert line.result_for(calls[1]) is exposed
    assert line.call_for(exposed) is calls[1]


def test_copied_claude_ownership_ambiguity_is_not_guessed_from_file_order(tmp_path: Path):
    reader = ClaudeReader()
    one = write_records(tmp_path / "a.jsonl", claude_pair("first"))
    two = write_records(tmp_path / "b.jsonl", claude_pair("second"))
    evidence = (
        reader.read(reader.source_for(one)).evidence + reader.read(reader.source_for(two)).evidence
    )
    forward = build_timelines(evidence)
    backward = build_timelines(reversed(evidence))
    assert (
        forward.ownership_unknown
        == backward.ownership_unknown
        == {e.native_identity for e in evidence}
    )
    assert forward.ownership_claims == backward.ownership_claims
    assert [(r.identity, r.ev.session_id) for r in rows(forward)] == [
        (r.identity, r.ev.session_id) for r in rows(backward)
    ]


def test_provider_usage_copy_and_stream_updates_share_supported_message_domain(tmp_path: Path):
    reader = ClaudeReader()
    message_id = "msg_0123456789abcdefghijkl"
    records = []
    for uuid, tokens in ((A, 10), (C, 20)):
        row = claude_tool_use(
            "first",
            "tool",
            "Bash",
            {"command": "echo"},
            message_id=message_id,
            usage={"input_tokens": tokens},
        )
        row["uuid"] = uuid
        records.append(row)
    original = write_records(tmp_path / "a.jsonl", records)
    copy = write_records(tmp_path / "b.jsonl", [{**r, "sessionId": "second"} for r in records])
    usage = [
        e
        for path in (original, copy)
        for e in reader.read(reader.source_for(path)).evidence
        if e.kind == "usage"
    ]
    assert len({e.native_key for e in usage}) == 1
    ledger = EvidenceLedger()
    list(ledger.drain(usage))
    assert ledger.conflict_count == 0
    [(resolved, updates)] = list(ledger.final_usage())
    assert resolved.payload["tokens"]["input_tokens"] == 20 and updates == 2


@pytest.mark.parametrize("reader_type", [PiReader, OmpReader, PrimeReader])
def test_real_reducer_overlap_debits_39_not_78(tmp_path: Path, reader_type):
    from ssgrep.accounting.collect.frequency import FrequencyView
    from ssgrep.accounting.ledger.reduce import reduce_profile
    from tests.accounting.ledger.reduce_builders import KEY, PROFILE, policy

    reader = reader_type()
    path = write_records(tmp_path / "native.jsonl", [pi_header("native")])
    source = reader.source_for(path)
    baseline = SourceState.from_dict(baseline_record(reader, source).state)
    append_bytes(
        path,
        jsonl(
            [
                pi_tool_call("c", "call_0", "bash", {"command": "ssgrep search needle"}),
                pi_tool_result("r", "call_0", "x" * 156),
            ]
        ),
    )
    live = reader.read(source, baseline).evidence
    full = reader.read(source).evidence
    identities = []
    for evidence in (full, live + full, full + live, full + live + full):
        reduced = reduce_profile(
            PROFILE,
            evidence,
            [],
            policy=policy(),
            frequency=FrequencyView.unknown(),
            fingerprint_key=KEY,
        )
        assert len(reduced.invocations) == 1
        costs = [d for bundle in reduced.bundles for d in bundle.debits]
        assert sum(d.chars or 0 for d in costs) // 4 == 39
        identities.append(reduced.invocations[0].invocation_id)
    assert len(set(identities)) == 1


def test_claude_missing_session_and_uuid_cannot_merge_same_filename_sources(tmp_path: Path):
    reader = ClaudeReader()
    record = claude_tool_use("ignored", "call_0", "Bash", {"command": "same"})
    record.pop("sessionId")
    record.pop("uuid")
    a = write_records(tmp_path / "a" / "same.jsonl", [record])
    b = write_records(tmp_path / "b" / "same.jsonl", [record])
    one = reader.read(reader.source_for(a)).evidence[0]
    two = reader.read(reader.source_for(b)).evidence[0]
    assert one.origin_status == two.origin_status == "missing_session_id"
    assert one.native_key != two.native_key


def test_claude_copied_origin_does_not_merge_workspace_permission_claims(tmp_path: Path):
    reader = ClaudeReader()
    original = claude_pair("same")
    copied = [{**r, "cwd": "/outside/approved"} for r in original]
    a = write_records(tmp_path / "a.jsonl", original)
    b = write_records(tmp_path / "b.jsonl", copied)
    one = reader.read(reader.source_for(a)).evidence
    two = reader.read(reader.source_for(b)).evidence
    assert [e.identity for e in one] == [e.identity for e in two]
    assert all(e.payload["cwd"] == "/outside/approved" for e in two)
    view = build_timelines(one + two)
    assert view.conflicts == 0
    assert view.ownership_unknown == {e.native_identity for e in one}


def test_duplicate_ownership_claim_survives_restart_and_forces_unknown_task(tmp_path: Path):
    from ssgrep.accounting.readers.pi_family import build_lineage

    reader = ClaudeReader()
    a = write_records(tmp_path / "a.jsonl", claude_pair("first"))
    b = write_records(tmp_path / "b.jsonl", claude_pair("second"))
    original = reader.read(reader.source_for(a)).evidence
    copied = reader.read(reader.source_for(b)).evidence
    ledger = EvidenceLedger()
    assert list(ledger.drain(original)) == original
    retained = list(original)
    for item in copied:
        assert not ledger.admit(item)
        claim = ledger.ownership_claim(item)
        assert claim is not None and claim.kind == "ownership_claim"
        assert "output" not in claim.payload and "arguments" not in claim.payload
        retained.append(claim)
    restarted = EvidenceLedger()
    list(restarted.drain(retained))
    assert not restarted.admit(copied[0])
    assert restarted.ownership_claim(copied[0]) is None
    view = build_timelines(retained)
    line = next(iter(view.sessions.values()))
    call = next(r for r in line.records if r.kind == "tool_call")
    lineage = build_lineage(view)
    task = lineage.task_of(line.key, call.seq)
    assert task.turn is None and task.branch == "unknown"
    assert lineage.context(line.key, call.seq) == ([], ["native_origin_ownership_unknown"])


def test_organization_rejects_ambiguous_native_ownership_but_personal_keeps_debit(tmp_path: Path):
    from ssgrep.accounting.readers.native_identity import native_in_scope
    from ssgrep.accounting.views.filter import Scoper
    from tests.accounting.ledger.reduce_builders import policy

    reader = ClaudeReader()
    a = write_records(tmp_path / "a.jsonl", claude_pair("first"))
    b = write_records(tmp_path / "b.jsonl", claude_pair("second"))
    view = build_timelines(
        reader.read(reader.source_for(a)).evidence + reader.read(reader.source_for(b)).evidence
    )
    call = next(r for r in rows(view) if r.kind == "tool_call")
    assert native_in_scope(call, view, Scoper(policy()))
    assert not native_in_scope(call, view, Scoper(policy("/work/demo", profile="org")))


def test_live_only_pi_context_declares_parent_without_imported_header_usage(tmp_path: Path):
    from ssgrep.accounting.readers.pi_family import build_lineage

    reader = PiReader()
    parent = write_records(tmp_path / "parent.jsonl", [pi_header("native-parent")])
    child = write_records(
        tmp_path / "not-child-id.jsonl", [pi_header("native-child", parentSession=str(parent))]
    )
    source = reader.source_for(child)
    baseline = SourceState.from_dict(baseline_record(reader, source).state)
    append_bytes(child, jsonl([pi_tool_call("c", "call_0", "bash", {"command": "echo"})]))
    live = reader.read(source, baseline).evidence
    parent_rows = reader.read(reader.source_for(parent)).evidence
    lineage = build_lineage(build_timelines(parent_rows + live))
    assert lineage.parent_of[("pi", "native-child")] == ("pi", "native-parent")
    assert lineage.task_of(("pi", "native-child"), 0).turn is None


def test_dropped_conflicting_variant_stays_quarantined_after_compact_claim_restart(tmp_path: Path):
    reader = ClaudeReader()
    path = write_records(tmp_path / "a.jsonl", claude_pair("same"))
    original = reader.read(reader.source_for(path)).evidence
    mutated = claude_pair("same")
    mutated[-1]["message"]["content"][0]["content"] = "changed"
    write_records(path, mutated)
    changed = reader.read(reader.source_for(path)).evidence[-1]
    ledger = EvidenceLedger()
    list(ledger.drain(original))
    assert not ledger.admit(changed)
    claim = ledger.ownership_claim(changed)
    assert claim is not None and claim.payload["origin_conflict"] is True
    assert claim.payload["target_digest"] == changed.digest and "output" not in claim.payload

    retained = [evidence_from_line(evidence_line(e)) for e in original + [claim]]
    restarted = EvidenceLedger()
    list(restarted.drain(retained))
    assert restarted.conflict_count == 1
    reversed_seed = EvidenceLedger()
    list(reversed_seed.drain(list(reversed(retained))))
    assert reversed_seed.conflict_count == 1
    assert not restarted.admit(changed)
    assert restarted.ownership_claim(changed) is None
    view = build_timelines(retained)
    assert original[-1].native_identity in view.conflicted and view.conflicts == 1
    line = next(iter(view.sessions.values()))
    call = next(r for r in line.records if r.kind == "tool_call")
    assert line.result_for(call) is None


# --- test_native_missing_identity.py ---


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_header_ids_do_not_merge_equal_filename_headers(tmp_path: Path, missing):
    reader = PiReader()
    header = pi_header("ignored")
    if missing is None:
        header.pop("id")
    else:
        header["id"] = missing
    paths = [write_records(tmp_path / name / "same.jsonl", [header]) for name in ("a", "b")]
    items = [reader.read(reader.source_for(path)).evidence[0] for path in paths]
    assert all(
        e.kind == "session_header" and e.origin_status == "missing_session_id" for e in items
    )
    assert len({e.native_key for e in items}) == 2
    assert all(e.origin_domain == e.source_id for e in items)
    assert build_timelines(items).conflicts == 0


@pytest.mark.parametrize("missing", ["session", "entry"])
def test_missing_native_ids_force_unknown_task_and_nonexportable_org_scope(
    tmp_path: Path, missing: str
):
    reader = PiReader()
    records = [
        pi_header("native"),
        pi_user("turn", "prompt"),
        pi_tool_call("call", "call_0", "bash", {"command": "ssgrep search x"}, cwd="/work/demo"),
    ]
    if missing == "session":
        records.pop(0)
    else:
        records[-1].pop("id")
    path = write_records(tmp_path / "native.jsonl", records)
    view = build_timelines(reader.read(reader.source_for(path)).evidence)
    line = next(iter(view.sessions.values()))
    call = next(r for r in line.records if r.kind == "tool_call")
    lineage = build_lineage(view)
    task = lineage.task_of(line.key, call.seq)
    assert task.turn is None and task.branch == "unknown"
    assert "native-origin-unknown" in task.root
    assert lineage.context(line.key, call.seq) == ([], ["native_identity_unavailable"])
    scoper = Scoper(policy("/work/demo", profile="org"))
    assert scoper.in_scope(call.ev.payload["cwd"])
    assert not native_in_scope(call, view, scoper)


def test_compact_restart_keeps_native_turn_context_and_missing_prior_prompt_coverage(
    tmp_path: Path,
):
    reader = PiReader()
    path = write_records(
        tmp_path / "native.jsonl", [pi_header("native"), pi_user("turn", "private-before-setup")]
    )
    source = reader.source_for(path)
    state = SourceState.from_dict(baseline_record(reader, source).state)
    call = pi_tool_call("call", "call_0", "bash", {"command": "ssgrep search x"})
    call["parentId"] = "turn"
    append_bytes(path, jsonl([call]))
    [live] = reader.read(source, state).evidence
    retained = evidence_from_line(evidence_line(live))
    assert retained.payload["originating_turn_record_id"] == "turn:turn"
    assert retained.payload["task_context_basis"] == "native_parent_chain"
    assert retained.payload["baseline_prior_context_unavailable"] is True
    assert "private-before-setup" not in str(retained.payload)
    view = build_timelines([retained])
    line = next(iter(view.sessions.values()))
    assert build_lineage(view).task_of(line.key, 0).turn == "turn:turn"
    assert "baseline_prior_context_unavailable" in build_lineage(view).context(line.key, 0)[1]


# --- test_conflicts.py ---


def stream(session: str, message: str, outputs: list[int | None], tool: str = "t1") -> list[dict]:
    """One streamed assistant message: every record repeats id, usage grows, same tool block."""
    records = []
    for output in outputs:
        usage = None if output is None else {"input_tokens": 9, "output_tokens": output}
        records.append(
            claude_tool_use(
                session, tool, "Bash", {"command": "ls"}, message_id=message, usage=usage
            )
        )
    return records


def claude(tmp_path: Path) -> ClaudeReader:
    return ClaudeReader({"CLAUDE_CONFIG_DIR": str(tmp_path / "claude")})


def path_for(tmp_path: Path, name: str = "s1") -> Path:
    return tmp_path / "claude" / "projects" / "-p" / f"{name}.jsonl"


def admit_all(reader, paths: list[Path], ledger: EvidenceLedger | None = None):
    ledger = ledger or EvidenceLedger()
    admitted = []
    for path in paths:
        for item in reader.read(reader.source_for(path)).evidence:
            if ledger.admit(item):
                admitted.append(item)
    return ledger, admitted


def test_streaming_usage_2_2_344_resolves_to_final_counted_once(tmp_path: Path) -> None:
    path = write_records(path_for(tmp_path), stream("s1", "msg_1", [2, 2, 344]))
    reader = claude(tmp_path)
    result = reader.read(reader.source_for(path))
    usage = [e for e in result.evidence if e.kind == "usage"]
    assert [e.record_id for e in usage] == ["usage:msg_1"]
    assert usage[0].payload["tokens"] == {"input_tokens": 9, "output_tokens": 344}
    assert usage[0].payload["streaming_updates"] == 3
    # the tool block repeated on every streamed record is still one call
    ledger, admitted = admit_all(reader, [path])
    assert sorted(e.record_id for e in admitted) == ["call:t1", "usage:msg_1"]
    assert ledger.conflict_count == 0


def test_stream_split_across_read_cycles_supersedes_instead_of_conflicting(
    tmp_path: Path,
) -> None:
    records = stream("s1", "msg_1", [2, 7, 204])
    path = write_records(path_for(tmp_path), records[:2])
    reader = claude(tmp_path)
    source = reader.source_for(path)
    ledger = EvidenceLedger()
    first = reader.read(source)
    assert [ledger.admit(e) for e in first.evidence if e.kind == "usage"] == [True]
    append_bytes(path, jsonl(records[2:]))
    second = reader.read(source, first.state)
    assert [ledger.admit(e) for e in second.evidence if e.kind == "usage"] == [False]
    assert ledger.conflict_count == 0 and ledger.superseded == 1
    [(final, updates)] = list(ledger.final_usage())
    assert final.payload["tokens"]["output_tokens"] == 204
    assert updates == 3


def test_older_prefix_copy_read_later_is_not_a_conflict(tmp_path: Path) -> None:
    full = write_records(path_for(tmp_path, "full"), stream("s1", "msg_1", [2, 344]))
    prefix = write_records(path_for(tmp_path, "prefix"), stream("s1", "msg_1", [2]))
    ledger, _ = admit_all(claude(tmp_path), [full, prefix])
    assert ledger.conflict_count == 0
    [(final, _)] = list(ledger.final_usage())
    assert final.payload["tokens"]["output_tokens"] == 344


def test_genuine_decrease_5_then_3_is_the_only_conflict(tmp_path: Path) -> None:
    records = stream("s1", "msg_1", [5, 3]) + stream("s1", "msg_2", [2, 2, 344], tool="t2")
    path = write_records(path_for(tmp_path), records)
    reader = claude(tmp_path)
    ledger, admitted = admit_all(reader, [path])
    [conflict] = ledger.conflicts()
    assert conflict.record_id == "usage:msg_1" and len(conflict.digests) == 2
    assert ledger.conflict_count == 1
    ids = sorted(e.record_id for e in admitted)
    assert ids == ["call:t1", "call:t2", "usage:msg_1", "usage:msg_2"]  # counted once each
    finals = {e.record_id: e.payload["tokens"]["output_tokens"] for e, _ in ledger.final_usage()}
    assert finals == {"usage:msg_1": 5, "usage:msg_2": 344}  # a conflict never lowers the total


def test_mixed_field_movement_is_a_conflict(tmp_path: Path) -> None:
    one = claude_tool_use("s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 5})
    two = claude_tool_use(
        "s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 9, "input_tokens": 1}
    )
    other = write_records(path_for(tmp_path, "b"), [two])
    first = write_records(path_for(tmp_path, "a"), [one, {**one, "uuid": "x"}])
    # field set differs (input_tokens appears) but nothing decreases: still a legal stream
    ledger, _ = admit_all(claude(tmp_path), [first, other])
    assert ledger.conflict_count == 0
    down = claude_tool_use(
        "s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 8, "input_tokens": 2}
    )
    mixed = write_records(path_for(tmp_path, "c"), [down])
    ledger, _ = admit_all(claude(tmp_path), [other, mixed])
    assert ledger.conflict_count == 1  # output went down while input_tokens moved the other way


def test_model_change_for_same_message_id_is_a_conflict(tmp_path: Path) -> None:
    one = claude_tool_use("s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 5})
    two = claude_tool_use("s1", "t1", "Bash", {}, message_id="m", usage={"output_tokens": 9})
    two["message"]["model"] = "another-model"
    path = write_records(path_for(tmp_path), [one, two])
    ledger, _ = admit_all(claude(tmp_path), [path])
    assert [c.record_id for c in ledger.conflicts()] == ["usage:m"]


def test_identical_tool_call_and_result_duplicates_still_dedupe(tmp_path: Path) -> None:
    records = stream("s1", "msg_1", [2, 344])
    a = write_records(path_for(tmp_path, "a"), records)
    b = write_records(path_for(tmp_path, "b"), records)
    ledger, admitted = admit_all(claude(tmp_path), [a, b])
    assert sorted(e.record_id for e in admitted) == ["call:t1", "usage:msg_1"]
    assert ledger.conflict_count == 0 and ledger.duplicates >= 2
    assert {e.kind for e in admitted} == {"tool_call", "usage"}


def test_changed_tool_call_arguments_remain_a_true_conflict(tmp_path: Path) -> None:
    a = write_records(path_for(tmp_path, "a"), [claude_tool_use("s1", "t1", "Bash", {"c": "ls"})])
    b = write_records(path_for(tmp_path, "b"), [claude_tool_use("s1", "t1", "Bash", {"c": "pwd"})])
    ledger, _ = admit_all(claude(tmp_path), [a, b])
    assert [c.record_id for c in ledger.conflicts()] == ["call:t1"]


# ----------------------------------------------------------------------- codex


def codex_reader(tmp_path: Path) -> CodexReader:
    return CodexReader({"CODEX_HOME": str(tmp_path / "codex")})


def rollout(tmp_path: Path, name: str) -> Path:
    return tmp_path / "codex" / "sessions" / f"{name}.jsonl"


def test_codex_fork_with_embedded_parent_meta_keeps_its_own_identity(tmp_path: Path) -> None:
    parent = write_records(
        rollout(tmp_path, "parent"),
        [
            codex_meta("parent-id"),
            codex_turn("t1"),
            codex_call("call_0", "exec_command", {"cmd": "ls"}),
            codex_token_count(100),
            codex_token_count(150),
        ],
    )
    fork_meta = codex_meta("fork-id")
    fork_meta["payload"]["forked_from_id"] = "parent-id"
    fork = write_records(
        rollout(tmp_path, "fork"),
        [
            fork_meta,
            codex_meta("parent-id"),  # embedded parent metadata, not a new owner of what follows
            codex_turn("t9"),
            codex_call("call_0", "exec_command", {"cmd": "ls"}),
            codex_token_count(19),  # the fork's own counter starts over
            codex_token_count(41),
        ],
    )
    reader = codex_reader(tmp_path)
    ledger, admitted = admit_all(reader, [parent, fork])
    assert ledger.conflict_count == 0
    assert {e.session_id for e in admitted} == {"parent-id", "fork-id"}
    fork_rows = [e for e in admitted if e.session_id == "fork-id"]
    assert total_usage(fork_rows)["total_tokens"] == 41  # not mixed into the parent's 150
    assert total_usage([e for e in admitted if e.session_id == "parent-id"])["total_tokens"] == 150


def test_codex_user_turns_with_equal_length_text_do_not_collide(tmp_path: Path) -> None:
    def turn(text: str) -> dict:
        payload = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        return {"timestamp": "t", "type": "response_item", "payload": payload}

    path = write_records(
        rollout(tmp_path, "r"), [codex_meta("s"), turn("fix the bug"), turn("add the test")]
    )
    ledger, admitted = admit_all(codex_reader(tmp_path), [path])
    assert ledger.conflict_count == 0
    assert len([e for e in admitted if e.kind == "user_turn"]) == 2


# ----------------------------------------------------------------------- prime


def prime(tmp_path: Path) -> PrimeReader:
    return PrimeReader({"PRIME_AGENT_SESSION_DIR": str(tmp_path / "prime" / "sessions")})


def test_prime_repeated_call_ids_in_one_session_are_distinct_calls(tmp_path: Path) -> None:
    records = [pi_header("m1")]
    for index in range(3):  # `call_0` reused for three different invocations
        records.append(pi_tool_call(f"e{index}a", "call_0", "ipython", {"code": f"x={index}"}))
        records.append(pi_tool_result(f"e{index}b", "call_0", f"out {index}"))
    one = write_records(tmp_path / "prime" / "sessions" / "m1.jsonl", records)
    copy = write_records(tmp_path / "prime" / "sessions" / "copy.jsonl", records)
    reader = prime(tmp_path)
    ledger, admitted = admit_all(reader, [one, copy])
    calls = [e for e in admitted if e.kind == "tool_call"]
    results = [e for e in admitted if e.kind == "tool_result"]
    assert ledger.conflict_count == 0
    assert len(calls) == 3 and len(results) == 3  # not collapsed, copies still dedupe
    assert [c.payload["arguments"]["code"] for c in calls] == ["x=0", "x=1", "x=2"]
    assert [r.payload["output"] for r in results] == ["out 0", "out 1", "out 2"]
    assert len({c.record_id for c in calls}) == 3 and len({r.record_id for r in results}) == 3
    assert len({c.origin_id for c in calls}) == 3 and len({r.origin_id for r in results}) == 3
    assert {c.origin_domain for c in calls} | {r.origin_domain for r in results} == {"m1"}
    assert all(c.payload["call_id"] == "call_0" for c in calls)
    # each result is bound to exactly the call that preceded it, in order
    assert [r.payload["call_record_id"] for r in results] == [c.record_id for c in calls]
    assert {r.payload["call_link_status"] for r in results} == {"unique_pending"}


def test_prime_repeated_call_ids_survive_incremental_reads(tmp_path: Path) -> None:
    first_half = [pi_header("m1"), pi_tool_call("e1", "call_0", "ipython", {"code": "a"})]
    path = write_records(tmp_path / "prime" / "sessions" / "m1.jsonl", first_half)
    reader = prime(tmp_path)
    source = reader.source_for(path)
    first = reader.read(source)
    append_bytes(path, jsonl([pi_tool_call("e2", "call_0", "ipython", {"code": "b"})]))
    second = reader.read(source, first.state)
    ids = [e.record_id for e in first.evidence + second.evidence if e.kind == "tool_call"]
    assert len(set(ids)) == 2


def test_prime_lineage_status_transitions_are_distinct_events(tmp_path: Path) -> None:
    def row(status: str) -> dict:
        return {"type": "rlm_subagent", "childId": "sub-1", "status": status, "rlmDepth": 1}

    path = write_records(
        tmp_path / "prime" / "session-artifacts" / "m1" / "rlm-subagents.jsonl",
        [row("running"), row("completed"), row("deleted")],
    )
    reader = prime(tmp_path)
    ledger, admitted = admit_all(reader, [path, path])
    assert ledger.conflict_count == 0
    assert [e.payload["status"] for e in admitted] == ["running", "completed", "deleted"]


def test_native_identity_context_is_validated_bounded_and_parent_linked() -> None:
    from ssgrep.accounting.ledger.contracts import RawEvidence
    from ssgrep.accounting.readers.native_identity import (
        _offset,
        fields,
        link_result,
        native_uuid,
        observe,
        remember_call,
    )

    assert native_uuid("11111111-1111-4111-8111-111111111111") == A
    assert native_uuid("11111111-1111-1111-8111-111111111111") is None
    assert native_uuid("not-a-uuid") is None

    context: dict = {}
    observe({"id": "prompt", "message": {"role": "user"}}, context, "offset:0")
    assert fields(context) == {
        "originating_turn_record_id": "turn:prompt",
        "task_context_basis": "native_user_turn",
        "baseline_prior_context_unavailable": False,
    }
    observe({"id": "answer", "parentId": "prompt"}, context, "offset:25")
    assert fields(context)["originating_turn_record_id"] == "turn:prompt"
    assert fields(context)["task_context_basis"] == "native_parent_chain"

    remember_call(context, "call", "call:one", "prompt")
    remember_call(context, "call", "call:two", "answer")
    assert link_result(context, "call", "answer") == ("call:two", "native_parent")
    assert link_result(context, "call") == ("call:one", "unique_pending")

    # More than the bounded context cap invalidates future unique guesses.
    for index in range(257):
        remember_call(context, f"pending-{index}", f"call:{index}", "prompt")
    assert context["call_context_incomplete"] is True
    assert context["pending_calls"] == {}
    assert link_result(context, "never-seen") == (None, "missing_context")

    evidence = RawEvidence(
        "pi",
        "pi:/tmp/session.jsonl",
        "main",
        "session",
        "call",
        "offset:not-an-int",
        "0",
        "2026-10-12T00:00:00Z",
        "tool_call",
    )
    assert _offset(evidence) == -1
    for index in range(257):
        observe({"id": f"entry-{index}"}, context, f"offset:{index}")
    assert "entry-0" not in context["entry_turns"]
    assert context["entry_turns"]["entry-256"] == "turn:prompt"

    from ssgrep.accounting.readers.native_identity import with_origin

    original = RawEvidence(
        "pi",
        "pi:/tmp/session.jsonl",
        "main",
        "session",
        "call",
        "offset:1",
        "unused",
        ts_utc="2026-10-12T00:00:00Z",
        kind="tool_call",
        payload={
            "call_id": "call",
            "cwd": "/private/work",
            "parent_session": "parent",
            "call_record_id": "derived-call",
        },
    )
    confirmed = with_origin(original, "session", "call", "confirmed_uuid")
    assert confirmed.payload == original.payload
    assert confirmed.digest == evidence_digest("tool_call", {"call_id": "call"})


def test_recorded_native_conflict_without_original_keeps_unknown_conflict_time(tmp_path: Path):
    reader = ClaudeReader()
    original_path = write_records(tmp_path / "original.jsonl", claude_pair("same"))
    changed_path = write_records(
        tmp_path / "changed.jsonl",
        [
            {
                **claude_pair("same")[0],
                "message": {
                    "id": "msg_1",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_0",
                            "name": "Bash",
                            "input": {"command": "other"},
                        }
                    ],
                },
            }
        ],
    )
    original_evidence = reader.read(reader.source_for(original_path)).evidence
    assert len(original_evidence) == 2
    original = original_evidence[0]
    unrelated = original_evidence[1]
    changed = reader.read(reader.source_for(changed_path)).evidence[0]
    ledger = EvidenceLedger()
    assert ledger.admit(original)
    assert not ledger.admit(changed)
    claim = ledger.ownership_claim(changed)
    assert claim is not None and claim.payload["origin_conflict"] is True

    view = build_timelines([unrelated, claim])
    assert view.conflicts == 1
    conflict_times = tuple(
        values for timeline in view.sessions.values() for values in timeline.conflict_times.values()
    )
    assert len(conflict_times) == 1
    assert conflict_times[0][0] is None
    assert any(value is not None for value in conflict_times[0])


def _legacy_claude_source(path: Path, rows: list[dict], *, offset: int | None = None):
    """A pre-v2 source record whose committed cursor sits just past ``offset`` bytes."""
    from ssgrep.accounting.readers.base import checkpoint as source_checkpoint, open_source

    write_records(path, rows)
    size = path.stat().st_size if offset is None else offset
    with open_source(path) as handle:
        state = source_checkpoint(handle, size, {})
    return store.SourceRecord(
        "claude", f"claude:{path}", state=state.to_dict(), baseline="skipped", complete=True
    )


def test_source_repair_only_rewrites_verified_claude_pi_family_cursors(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    rows = [claude_tool_use("repair", "call_0", "Bash", {"command": "ssgrep search x"})]
    repairable = _legacy_claude_source(root / "repairable.jsonl", rows)
    truncated = _legacy_claude_source(root / "truncated.jsonl", rows)
    truncated.state["offset"] = truncated.offset + 1_000  # cursor beyond the file
    vanished = _legacy_claude_source(root / "vanished.jsonl", rows)
    (root / "vanished.jsonl").unlink()
    unprefixed = _legacy_claude_source(root / "unprefixed.jsonl", rows)
    unprefixed.source_id = f"other:{root / 'unprefixed.jsonl'}"
    unstarted = _legacy_claude_source(root / "unstarted.jsonl", rows)
    unstarted.state["offset"] = 0
    unsupported = store.SourceRecord(
        "codex",
        f"codex:{root / 'codex.jsonl'}",
        state={"offset": 12, "head_digest": "unchanged"},
        baseline="skipped",
        complete=True,
    )
    records = [repairable, truncated, vanished, unprefixed, unstarted, unsupported]
    checkpoint = store.Checkpoint("personal-local", sources={r.key: r for r in records})
    before = {r.key: (dict(r.state), r.status, r.complete) for r in records}

    result = prepare("personal-local", checkpoint, NOW, env=env, policy=policy())

    assert result.complete and checkpoint.identity_method == "native_identity_v2"
    assert result.source_counts["repaired_sources"] == len(records)
    assert result.bytes_read > 0
    # Only the intact legacy cursor gains native context; its offset is preserved.
    assert repairable.state["extra"]["native_identity_method"] == "native_identity_v2"
    assert repairable.state["offset"] == before[repairable.key][0]["offset"]
    assert repairable.status == "ok" and repairable.complete
    # A lost or shortened committed prefix cannot be re-verified and stays explicit.
    for lost in (truncated, vanished):
        assert (lost.status, lost.complete) == ("unreadable", False)
    # Cursors that carry no committed prefix or a foreign identity are left untouched.
    for ignored in (unprefixed, unstarted, unsupported):
        assert (dict(ignored.state), ignored.status, ignored.complete) == before[ignored.key]


def test_source_repair_resumes_after_the_record_budget_without_repeating_work(
    tmp_path, monkeypatch
):
    root, env = setup(tmp_path, monkeypatch)
    rows = [claude_tool_use("repair", "call_0", "Bash", {"command": "ssgrep search x"})]
    first = _legacy_claude_source(root / "a.jsonl", rows)
    second = _legacy_claude_source(root / "b.jsonl", rows)
    checkpoint = store.Checkpoint("personal-local", sources={r.key: r for r in (first, second)})
    ordered = sorted((first, second), key=lambda r: r.key)

    partial = prepare("personal-local", checkpoint, NOW, env=env, policy=policy(), max_records=1)

    assert not partial.complete and partial.warnings == ("identity_migration_pending",)
    assert checkpoint.identity_migration["source"] == 1
    assert checkpoint.identity_method != "native_identity_v2"
    assert "native_identity_method" in ordered[0].state["extra"]
    assert "native_identity_method" not in ordered[1].state["extra"]

    resumed = store.Checkpoint.from_dict(checkpoint.to_dict())
    final = prepare("personal-local", resumed, NOW, env=env, policy=policy(), max_records=1)
    assert final.complete and resumed.identity_method == "native_identity_v2"
    assert all("native_identity_method" in r.state["extra"] for r in resumed.sources.values())


def test_migration_loads_the_profile_policy_when_none_is_supplied(tmp_path, monkeypatch):
    root, env = setup(tmp_path, monkeypatch)
    record = store.SourceRecord(
        "codex", f"codex:{root / 'x.jsonl'}", state={"offset": 3}, baseline="skipped"
    )
    checkpoint = store.Checkpoint("personal-local", sources={record.key: record})
    result = prepare("personal-local", checkpoint, NOW, env=env)
    assert result.complete and result.changed
    assert checkpoint.identity_method == "native_identity_v2"
    assert checkpoint.identity_migration["status"] == "complete"


def test_archive_expands_compact_root_reduction_to_prove_the_workspace_mapping(
    tmp_path, monkeypatch
):
    from ssgrep.accounting.collect.partitions import write_reduction_document
    from ssgrep.accounting.readers.identity_migration import ARCHIVE, archive_current

    setup(tmp_path, monkeypatch)
    document = {
        "executions": [],
        "invocations": [],
        "session_roots": [["claude", "first", "claude:first"]],
        "session_parents": [],
        "session_cwd": [["claude", "first", "/workspace/alpha"]],
        "invocation_sessions": [],
        "methodology": "legacy",
    }
    reduction = write_reduction_document("personal-local", "reduction", document)
    compact = store.read_blob("personal-local", reduction)
    assert compact is not None and compact["format"] == "root-reduction-v1"
    assert "session_roots" not in compact  # rows live only in the per-root documents
    checkpoint = store.Checkpoint(
        "personal-local", methodology="legacy", blobs={"reduction": reduction}
    )

    previous = archive_current("personal-local", checkpoint, NOW, [])

    assert previous == {"reduction": reduction}
    manifest = store.read_blob("personal-local", checkpoint.blobs[ARCHIVE])
    assert manifest is not None
    assert manifest["session_roots"] == [["claude", "first", "claude:first"]]
    assert manifest["session_cwd"] == [["claude", "first", "/workspace/alpha"]]
    assert manifest["root_mapping_status"] == "available"
    assert set(checkpoint.blobs) == {ARCHIVE}


def _native_copy(
    kind: str,
    record_id: str,
    payload: dict,
    *,
    source: str = "claude:/a.jsonl",
    session: str = "native-session",
    offset: int = 0,
    digest: str | None = None,
) -> RawEvidence:
    return RawEvidence(
        "claude",
        source,
        "main",
        session,
        record_id,
        f"offset:{offset}",
        digest or evidence_digest(kind, payload),
        TS,
        kind,
        payload=payload,
        origin_domain="native-domain",
        origin_id=record_id,
        origin_status="confirmed_uuid",
    )


def _usage_copy(model: str, tokens: dict, *, source: str) -> RawEvidence:
    return _native_copy(
        "usage",
        "usage:msg_1",
        {"message_id": "msg_1", "model": model, "tokens": tokens, "streaming_updates": 1},
        source=source,
    )


def test_streamed_usage_copies_with_different_context_are_a_conflict_not_a_revision():
    low = {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}
    high = {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
    # Positive control: same context and greater counters in another copy is one revised event.
    revised = build_timelines(
        [
            _usage_copy("m1", low, source="claude:/a.jsonl"),
            _usage_copy("m1", high, source="claude:/b.jsonl"),
        ]
    )
    assert revised.conflicts == 0 and not revised.conflicted

    conflicted = build_timelines(
        [
            _usage_copy("m1", low, source="claude:/a.jsonl"),
            _usage_copy("m2", high, source="claude:/b.jsonl"),
        ]
    )
    assert conflicted.conflicts == 1
    assert conflicted.conflicted == {"claude:native-domain:usage:msg_1"}


def _claim_for(held: RawEvidence, **claimant) -> RawEvidence:
    from ssgrep.accounting.readers.native_identity import OwnershipClaims

    moved = replace_evidence(held, **claimant)
    claim = OwnershipClaims().take(moved, b"k")
    assert claim is not None
    return claim


def replace_evidence(item: RawEvidence, *, source=None, session=None, payload=None) -> RawEvidence:
    return replace(
        item,
        source_id=source or item.source_id,
        session_id=session or item.session_id,
        payload=payload if payload is not None else item.payload,
    )


def test_earlier_ownership_claim_replaces_cwd_and_parent_of_the_retained_copy():
    held = _native_copy(
        "tool_call",
        "call:one",
        {"call_id": "one", "cwd": "/old/place", "parent_session": "old-parent"},
        source="claude:/z.jsonl",
    )
    claim = _claim_for(
        held,
        source="claude:/a.jsonl",
        session="other-session",
        payload={"call_id": "one", "cwd": "/new/place", "parent_session": "new-parent"},
    )
    view = build_timelines([held, claim])
    (line,) = view.sessions.values()
    (record,) = line.records
    assert record.ev.source_id == "claude:/a.jsonl"
    assert record.ev.session_id == "other-session"
    assert record.ev.payload["cwd"] == "/new/place"
    assert record.ev.payload["parent_session"] == "new-parent"
    # Positive control: the retained copy is untouched without the claim.
    (alone,) = next(iter(build_timelines([held]).sessions.values())).records
    assert alone.ev.payload["cwd"] == "/old/place"
    assert alone.ev.source_id == "claude:/z.jsonl"

    cleared = build_timelines(
        [
            held,
            _claim_for(
                held,
                source="claude:/a.jsonl",
                payload={"call_id": "one", "cwd": "/new/place"},
            ),
        ]
    )
    (cleared_record,) = next(iter(cleared.sessions.values())).records
    assert cleared_record.ev.payload["parent_session"] is None


def test_ownership_claim_without_a_string_target_record_cannot_replace_the_copy():
    held = _native_copy(
        "tool_call", "call:one", {"call_id": "one", "cwd": "/old/place"}, source="claude:/z.jsonl"
    )
    claim = _claim_for(
        held, source="claude:/a.jsonl", payload={"call_id": "one", "cwd": "/new/place"}
    )
    malformed = replace_evidence(claim, payload={**claim.payload, "target_record_id": 7})
    for candidate, expected in ((claim, "/new/place"), (malformed, "/old/place")):
        (record,) = next(iter(build_timelines([held, candidate]).sessions.values())).records
        assert record.ev.payload["cwd"] == expected


def test_conflicting_result_link_proofs_become_ambiguous_until_one_call_is_proven():
    def result(call: str, source: str) -> RawEvidence:
        payload = {"call_id": "c", "output": "same", "call_record_id": call}
        return _native_copy(
            "tool_result",
            "result:c",
            {**payload, "call_link_status": "unique_native_copy"},
            source=source,
            digest="shared-content-digest",
        )

    unique = build_timelines([result("call:one", "claude:/a.jsonl")])
    (single,) = next(iter(unique.sessions.values())).records
    assert single.ev.payload["call_record_id"] == "call:one"
    assert single.ev.payload["call_link_status"] == "unique_native_copy"

    both = build_timelines(
        [result("call:one", "claude:/a.jsonl"), result("call:two", "claude:/b.jsonl")]
    )
    (ambiguous,) = next(iter(both.sessions.values())).records
    assert ambiguous.ev.payload["call_record_id"] is None
    assert ambiguous.ev.payload["call_link_status"] == "ambiguous"


def test_ownership_claims_ignore_unaddressable_targets_and_conflict_evidence_needs_a_digest():
    from ssgrep.accounting.readers.native_identity import CLAIM, OwnershipClaims

    held = _native_copy("tool_call", "call:one", {"call_id": "one"})
    payload = {
        "target_native_key": list(held.native_key),
        "target_digest": held.digest,
        "original_kind": "tool_call",
        "origin_conflict": True,
    }
    claim = replace(held, kind=CLAIM, payload=payload, origin_status="metadata_claim")
    claims = OwnershipClaims()
    claims.note(claim, b"unused")
    assert len(claims.extra) == 1 and len(claims.conflicts) == 1
    evidence = OwnershipClaims.conflict_evidence(claim)
    assert evidence is not None and evidence.digest == held.digest and evidence.kind == "tool_call"

    for damaged in (
        {k: v for k, v in payload.items() if k != "target_native_key"},
        {**payload, "target_native_key": ["only", "two"]},
        {**payload, "target_digest": 17},
    ):
        broken = replace(claim, payload=damaged)
        assert OwnershipClaims.conflict_evidence(broken) is None
        untouched = OwnershipClaims()
        untouched.note(broken, b"unused")
        assert not untouched.conflicts
    targetless = OwnershipClaims()
    targetless.note(replace(claim, payload={"origin_conflict": True}), b"unused")
    assert not targetless.first and not targetless.extra and not targetless.conflicts


def test_native_scope_of_a_result_follows_the_call_cwd_when_the_result_has_none():
    from ssgrep.accounting.readers.native_identity import native_in_scope

    def view_for(cwd: str):
        call = _native_copy("tool_call", "call:one", {"call_id": "one", "cwd": cwd}, offset=0)
        result = _native_copy(
            "tool_result", "result:one", {"call_id": "one", "output": "x"}, offset=1
        )
        result = replace_evidence(
            result,
            payload={**result.payload, "call_record_id": "call:one", "call_link_status": "x"},
        )
        view = build_timelines([call, result])
        (line,) = view.sessions.values()
        return view, next(r for r in line.records if r.kind == "tool_result")

    from ssgrep.accounting.privacy.policy import PolicyFile

    org, personal = policy("/work/demo", profile="org"), policy()
    scoper = Scoper(personal, PolicyFile(profiles={"org": org, "personal-local": personal}))
    view, unowned = view_for("/home/me/app")
    assert "cwd" not in unowned.ev.payload
    assert native_in_scope(unowned, view, scoper)
    view, org_owned = view_for("/work/demo/app")
    assert not native_in_scope(org_owned, view, scoper)


def test_migration_keeps_native_and_unmigrated_runtime_records_and_gaps_unreadable_origins(
    tmp_path, monkeypatch
):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "a.jsonl"
    call = legacy_pair(path, "first")[0]
    assert migrate_original(call, env).evidence is not None  # positive control
    native = replace(call, origin_domain="first", origin_id="call:call_0")
    assert migrate_original(native, env).evidence is native
    codex = replace(call, runtime="codex", source_id=f"codex:{path}")
    assert migrate_original(codex, env).evidence is codex
    for broken in (
        replace(call, source_id=f"pi:{path}"),  # another runtime's namespace
        replace(call, locator="line:0"),  # no byte offset to verify
    ):
        outcome = migrate_original(broken, env)
        assert outcome.evidence is None and outcome.reason == "source_identity_mismatch"
        assert outcome.bytes_read == 0
    unknown_reader = migrate_record_without_reader(call)
    assert unknown_reader.evidence is None and unknown_reader.reason == "source_identity_mismatch"


def migrate_record_without_reader(item):
    from ssgrep.accounting.readers.native_identity import migrate_record

    return migrate_record(item, {}, Scoper(policy()), 16 * 1024 * 1024)


def test_defective_eof_pi_call_digest_without_cwd_migrates_only_for_its_own_stem(
    tmp_path, monkeypatch
):
    _, env = setup(tmp_path, monkeypatch)
    home = tmp_path / "pi" / "native-session.jsonl"
    originals = old_pi_history(home, "pi")
    call = next(item for item in originals if item.kind == "tool_call")
    defective_payload = {**call.payload, "cwd": None}
    defective = replace(
        call, payload=defective_payload, digest=evidence_digest("tool_call", defective_payload)
    )

    migrated = migrate_original(defective, env).evidence
    assert migrated is not None and migrated.origin_domain == "native-session"
    assert migrated.payload["cwd"] == CWD  # native context, not the defective baseline gap
    assert migrate_original(call, env).evidence is not None

    # The same defective digest is trusted only when the file stem is the session id.
    renamed = tmp_path / "pi" / "other-stem.jsonl"
    old_pi_history(renamed, "pi")
    foreign = replace(defective, source_id=f"pi:{renamed}")
    drift = migrate_original(foreign, env)
    assert drift.evidence is None and drift.reason == "source_digest_drift"


def test_native_context_scan_ignores_partial_or_malformed_lines_around_the_cursor(
    tmp_path, monkeypatch
):
    import os

    from ssgrep.accounting.readers.native_identity import _context

    _, env = setup(tmp_path, monkeypatch)
    reader = PiReader(env)
    header = jsonl([pi_header("native-session")])
    garbage = b"{not json\n"
    entry = jsonl([pi_user("turn-entry", "hello")])

    def scan(name: str, content: bytes, offset: int | None = None):
        path = tmp_path / "pi" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            return _context(
                reader,
                reader.source_for(path),
                descriptor,
                len(content) if offset is None else offset,
            )
        finally:
            os.close(descriptor)

    # Positive control: a terminated header, then a cursor-adjacent complete entry.
    context, consumed = scan("ok.jsonl", header + entry)
    assert context["session"] == "native-session"
    assert context["turn_id"] == "turn:turn-entry"
    assert consumed == 2 * len(header + entry)
    # A header without its terminating newline is not trusted metadata.
    partial_header, _ = scan("partial.jsonl", header.rstrip(b"\n"))
    assert "session" not in partial_header
    # Malformed rows are skipped; the later valid header is still found.
    skipped, _ = scan("skipped.jsonl", garbage + header)
    assert skipped["session"] == "native-session"
    # An unterminated tail row is ignored rather than decoded as native context.
    unterminated, _ = scan("tail.jsonl", header + entry.rstrip(b"\n"))
    assert unterminated["session"] == "native-session"
    assert unterminated["turn_id"] is None
    # A syntactically valid non-object row cannot carry native context either.
    non_object, _ = scan("array.jsonl", header + b"[1, 2]\n" + entry)
    assert non_object["turn_id"] == "turn:turn-entry"


def test_migration_reports_unparseable_original_record_as_unreadable_not_digest_drift(
    tmp_path, monkeypatch
):
    root, env = setup(tmp_path, monkeypatch)
    path = root / "a.jsonl"
    call = legacy_pair(path, "first")[0]
    assert migrate_original(call, env).evidence is not None
    path.write_bytes(b"{not json\n" + path.read_bytes())
    corrupted = replace(call, locator="offset:0")
    outcome = migrate_original(corrupted, env)
    assert outcome.evidence is None and outcome.reason == "source_unreadable"
