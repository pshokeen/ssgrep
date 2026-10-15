"""Native identity properties, independent of wrapper location and read mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.collector_baseline import baseline_record
from ssgrep.accounting.readers.base import EvidenceLedger, SourceState
from ssgrep.accounting.readers.claude import ClaudeReader
from ssgrep.accounting.readers.omp import OmpReader
from ssgrep.accounting.readers.pi import PiReader
from ssgrep.accounting.readers.prime import PrimeReader
from ssgrep.accounting.timeline import build_timelines
from tests.accounting.readers_fixtures import (
    append_bytes,
    claude_tool_result,
    claude_tool_use,
    jsonl,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    write_records,
)

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
    from ssgrep.accounting.frequency import FrequencyView
    from ssgrep.accounting.reduce import reduce_profile
    from tests.accounting.reduce_builders import KEY, PROFILE, policy

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
    from ssgrep.accounting.lineage import build_lineage

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
    from ssgrep.accounting.identity_claims import native_in_scope
    from ssgrep.accounting.scope_events import Scoper
    from tests.accounting.reduce_builders import policy

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
    from ssgrep.accounting.lineage import build_lineage

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
    from ssgrep.accounting.evidence_codec import evidence_from_line, evidence_line

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
