"""Partial copies and streamed usage converge independently of discovery/admission order."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ssgrep.accounting.collector_read import make_readers
from ssgrep.accounting.evidence_codec import evidence_from_line, evidence_line
from ssgrep.accounting.evidence_replay import EvidenceReplay
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.native_source import NativeSourceAdapter
from ssgrep.accounting.readers.base import EvidenceLedger
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.timeline import build_timelines
from tests.accounting.readers.test_native_identity import A, B, C, claude_pair
from tests.accounting.readers_fixtures import append_bytes, claude_tool_use, jsonl, write_records
from tests.accounting.reduce_builders import KEY, PROFILE, policy


def admitted_with_metadata(items):
    ledger = EvidenceLedger()
    accepted = []
    for item in items:
        if ledger.admit(item):
            accepted.append(item)
        else:
            claim = ledger.ownership_claim(item)
            if claim is not None:
                accepted.append(claim)
            revision = ledger.usage_revision(item)
            if revision is not None:
                accepted.append(revision)
    return accepted


def test_partial_result_copy_keeps_39_debit_and_unknown_task_in_both_orders_after_restart(
    tmp_path: Path,
):
    root = tmp_path / "claude" / "projects" / "fixture"
    env = {"HOME": str(tmp_path / "home"), "CLAUDE_CONFIG_DIR": str(tmp_path / "claude")}
    reader = ClaudeReader(env)
    original = write_records(root / "z-original.jsonl", claude_pair("original"))
    copy = write_records(root / "a-partial.jsonl", [claude_pair("resumed")[1]])
    full = reader.read(reader.source_for(original)).evidence
    partial = reader.read(reader.source_for(copy)).evidence
    assert partial[0].payload["call_record_id"] is None
    signatures = []
    for order in (full + partial, partial + full):
        retained = admitted_with_metadata(order)
        # Exercise actual compact retention and verified source hydration, not fixture text replay.
        compact = [evidence_from_line(evidence_line(e)) for e in retained]
        readers = make_readers(env)
        adapter = NativeSourceAdapter(
            {},
            readers,
            (tmp_path / "claude" / "projects",),
            Scoper(policy()),
            int(datetime(2026, 10, 14, tzinfo=UTC).timestamp() * 1000),
        )
        replay = EvidenceReplay(PROFILE, adapter=adapter)
        hydrated = [replay.hydrate(e) for e in compact]
        assert not replay.unavailable
        view = build_timelines(hydrated)
        line = view.sessions[("claude", "original")]
        call = next(r for r in line.records if r.kind == "tool_call")
        result = line.result_for(call)
        assert result is not None and result.seq > call.seq
        assert (
            result.ev.session_id == "resumed"
        )  # containing wrapper remains independently retained
        assert line.call_for(result) is call
        reduced = reduce_profile(
            PROFILE,
            hydrated,
            [],
            policy=policy(),
            frequency=FrequencyView.unknown(),
            fingerprint_key=KEY,
        )
        assert len(reduced.invocations) == 1
        assert sum(d.chars or 0 for b in reduced.bundles for d in b.debits) // 4 == 39
        task = reduced.invocations[0].task
        assert task is not None
        assert task.turn is None and task.branch == "unknown"
        signatures.append(
            (
                reduced.invocations[0].invocation_id,
                task,
                tuple((r.identity, r.ev.source_id, r.seq) for r in line.records),
            )
        )
    assert signatures[0] == signatures[1]


def usage_row(uuid: str, tokens: int, session: str = "native"):
    row = claude_tool_use(
        session,
        "unused",
        "Bash",
        {"command": "echo"},
        message_id="msg_0123456789abcdefghijkl",
        usage={"input_tokens": tokens},
    )
    row["uuid"] = uuid
    row["message"]["content"] = []
    return row


def test_incremental_usage_supersedes_once_and_stale_copy_never_sums(tmp_path: Path):
    reader = ClaudeReader()
    path = write_records(tmp_path / "native.jsonl", [usage_row(A, 10)])
    source = reader.source_for(path)
    first = reader.read(source)
    ledger = EvidenceLedger()
    [old] = first.evidence
    assert ledger.admit(old)
    append_bytes(path, jsonl([usage_row(C, 20)]))
    [new] = reader.read(source, first.state).evidence
    assert not ledger.admit(new)
    revision = ledger.usage_revision(new)
    assert revision is not None
    assert not ledger.admit(new) and ledger.usage_revision(new) is None
    copied = write_records(tmp_path / "copy.jsonl", [usage_row(B, 10, "resumed")])
    [stale] = reader.read(reader.source_for(copied)).evidence
    assert not ledger.admit(stale) and ledger.usage_revision(stale) is None
    claim = ledger.ownership_claim(stale)
    compact = [
        evidence_from_line(evidence_line(e)) for e in [old, revision, claim] if e is not None
    ]
    view = build_timelines(compact)
    usages = [r for line in view.sessions.values() for r in line.records if r.kind == "usage"]
    assert len(usages) == 1 and usages[0].ev.payload["tokens"]["input_tokens"] == 20
    assert view.conflicts == 0


def test_claude_turn_uuid_is_normalized_and_invalid_uuid_never_becomes_global(tmp_path: Path):
    from tests.accounting.readers_fixtures import claude_prompt

    reader = ClaudeReader()
    uuid = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    one = write_records(tmp_path / "a.jsonl", [claude_prompt("first", "prompt", uuid=uuid)])
    two = write_records(
        tmp_path / "b.jsonl", [claude_prompt("second", "prompt", uuid=uuid.upper())]
    )
    a = reader.read(reader.source_for(one)).evidence[0]
    b = reader.read(reader.source_for(two)).evidence[0]
    assert a.identity == b.identity
    invalid = write_records(
        tmp_path / "invalid.jsonl", [claude_prompt("third", "prompt", uuid="local-id")]
    )
    item = reader.read(reader.source_for(invalid)).evidence[0]
    assert item.origin_status == "session_scoped"
    assert item.origin_domain != "claude_record_uuid"


def test_pi_baseline_preserves_native_user_turn_identity_without_prompt_import(tmp_path: Path):
    from ssgrep.accounting.collector_baseline import baseline_record
    from ssgrep.accounting.lineage import build_lineage
    from ssgrep.accounting.readers.base import SourceState
    from ssgrep.accounting.readers.pi import PiReader
    from tests.accounting.readers_fixtures import pi_header, pi_tool_call, pi_user

    reader = PiReader()
    path = write_records(
        tmp_path / "different-filename.jsonl",
        [pi_header("native-session"), pi_user("native-turn", "prior-private-prompt")],
    )
    source = reader.source_for(path)
    state = SourceState.from_dict(baseline_record(reader, source).state)
    assert "prior-private-prompt" not in str(state.extra)
    call = pi_tool_call("native-call", "call_0", "bash", {"command": "ssgrep search x"})
    call["parentId"] = "native-turn"
    append_bytes(path, jsonl([call]))
    live = reader.read(source, state).evidence
    full = reader.read(source).evidence
    tasks = []
    for records in (live, full):
        view = build_timelines(records)
        line = view.sessions[("pi", "native-session")]
        record = next(r for r in line.records if r.kind == "tool_call")
        tasks.append(build_lineage(view).task_of(line.key, record.seq))
    assert tasks[0] == tasks[1]
    assert tasks[0].turn == "turn:native-turn"
    assert all(e.kind != "user_turn" for e in live)
