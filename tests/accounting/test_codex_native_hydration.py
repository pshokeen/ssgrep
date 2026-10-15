"""Codex compact replay uses validated native metadata without changing payloads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.evidence_codec import compact_payload
from ssgrep.accounting.evidence_replay import EvidenceReplay
from ssgrep.accounting.native_source import NativeSourceAdapter
from ssgrep.accounting.policy_rules import PolicyFile
from ssgrep.accounting.readers.codex import CodexReader
from ssgrep.accounting.scope_events import Scoper
from tests.accounting.readers_fixtures import (
    TS,
    codex_call,
    codex_meta,
    codex_output,
    codex_turn,
    write_records,
)
from tests.accounting.reduce_builders import policy

SESSION = "native-child"
CWD = "/work/team-a/project"
FOREIGN = "/work/foreign/project"


def scene(tmp_path: Path):
    header = codex_meta(SESSION, CWD)
    header["payload"]["source"] = {
        "subagent": {"thread_spawn": {"parent_thread_id": "native-parent"}}
    }
    rows = [header, codex_meta("embedded-parent", FOREIGN), codex_turn("t1", CWD)]
    for number in (1, 2):
        if number == 2:
            rows.append(codex_turn("t2", CWD))
        rows.extend(
            [
                {
                    "timestamp": TS,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"question {number}"}],
                    },
                },
                codex_call(f"c{number}", "exec_command", {"cmd": f"ssgrep search query{number}"}),
                codex_output(f"c{number}", f"answer {number}"),
            ]
        )
    path = write_records(tmp_path / ".codex/sessions/rollout.jsonl", rows)
    reader = CodexReader({"CODEX_HOME": str(tmp_path / ".codex")})
    originals = reader.read(reader.source_for(path)).evidence
    assert {item.kind for item in originals} == {"tool_call", "tool_result", "user_turn"}
    compact = [replace(item, payload=compact_payload(item.payload)) for item in originals]
    org = policy("/work/team-a", profile="org-a")
    other = policy("/work/foreign", profile="org-b")
    ownership = PolicyFile(profiles={"org-a": org, "org-b": other})
    adapter = NativeSourceAdapter(
        {},
        {"codex": reader},
        (tmp_path / ".codex/sessions",),
        Scoper(org, ownership),
        1_800_000_000_000,
    )
    return path, rows, originals, compact, adapter


@pytest.mark.parametrize("kind", ["tool_call", "tool_result", "user_turn"])
def test_codex_single_org_hydration_uses_native_header_and_preserves_digest(
    tmp_path: Path, kind: str
) -> None:
    _, _, originals, compact, adapter = scene(tmp_path)
    index = next(i for i, item in enumerate(originals) if item.kind == kind)
    expected = originals[index]
    if kind != "tool_call":
        assert "cwd" not in compact[index].payload
    replay = EvidenceReplay("org-a", adapter=adapter)
    hydrated = replay.hydrate(compact[index])
    assert hydrated == expected
    assert hydrated.digest == expected.digest
    assert hydrated.payload == expected.payload  # no derived cwd/native digest mutation
    assert not replay.failures


def test_codex_org_batch_replays_multiple_turns_and_missing_reference_cwd(tmp_path: Path) -> None:
    _, _, originals, compact, adapter = scene(tmp_path)
    assert compact[0].kind == "user_turn" and "cwd" not in compact[0].payload
    replay = EvidenceReplay("org-a", adapter=adapter)
    assert replay.hydrate_many(compact) == originals
    assert not replay.failures
    assert adapter.bytes_read <= adapter.byte_budget


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "corruption,reason",
    [
        ("wrong_owner", "source_identity_mismatch"),
        ("missing_owner", "source_identity_mismatch"),
        ("foreign_cwd", "source_out_of_scope"),
        ("missing_cwd", "source_scope_unknown"),
    ],
)
def test_codex_native_owner_and_scope_fail_closed(
    tmp_path: Path, batch: bool, corruption: str, reason: str
) -> None:
    path, rows, _, compact, adapter = scene(tmp_path)
    if corruption == "wrong_owner":
        rows[0]["payload"]["id"] = "native-rogue"  # same byte width retains selected offsets
    elif corruption == "missing_owner":
        rows[0]["payload"]["id"] = ""
    elif corruption == "foreign_cwd":
        rows[0]["payload"]["cwd"] = FOREIGN
    else:
        rows[0]["payload"].pop("cwd")
    write_records(path, rows)
    replay = EvidenceReplay("org-a", adapter=adapter)
    selected = compact[:3]
    hydrated = (
        replay.hydrate_many(selected) if batch else [replay.hydrate(item) for item in selected]
    )
    assert all(item.payload.get("_replay_unavailable") == reason for item in hydrated)
    assert replay.failures


def test_codex_embedded_parent_metadata_never_replaces_native_file_owner(tmp_path: Path) -> None:
    _, _, originals, compact, adapter = scene(tmp_path)
    replay = EvidenceReplay("org-a", adapter=adapter)
    hydrated = replay.hydrate_many(compact)
    assert hydrated == originals
    assert {item.session_id for item in hydrated} == {SESSION}
    assert all(item.payload.get("parent_session") == "native-parent" for item in hydrated)


@pytest.mark.parametrize("kind", ["tool_result", "user_turn"])
@pytest.mark.parametrize("cross_boundary", [False, True])
def test_codex_direct_never_authorizes_from_header_across_unknown_context(
    tmp_path: Path,
    kind: str,
    cross_boundary: bool,
) -> None:
    path, _, _, _, adapter = scene(tmp_path)

    def padding(size):
        return {
            "timestamp": TS,
            "type": "event_msg",
            "payload": {"type": "agent_reasoning", "text": "x" * size},
        }

    foreign = codex_turn("foreign-turn", FOREIGN)
    rows = [codex_meta(SESSION, CWD), padding(63_000 if cross_boundary else 80_000)]
    if cross_boundary:
        foreign["payload"]["native_metadata"] = "x" * 4096
        rows.append(foreign)  # valid native context straddles the old 64KiB split
    else:
        rows.extend([foreign, padding(80_000)])  # outside old head AND adjacent tail
    rows.extend(
        [
            codex_output("late-call", "late answer"),
            {
                "timestamp": TS,
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "id": "late-user",
                    "content": [{"type": "input_text", "text": "late question"}],
                },
            },
        ]
    )
    write_records(path, rows)
    reader = adapter.readers["codex"]
    original = next(
        item for item in reader.read(reader.source_for(path)).evidence if item.kind == kind
    )
    assert "cwd" not in original.payload
    compact = replace(original, payload=compact_payload(original.payload))
    hydrated = EvidenceReplay("org-a", adapter=adapter).hydrate(compact)
    expected = "source_out_of_scope" if cross_boundary else "source_incomplete"
    assert hydrated.payload.get("_replay_unavailable") == expected


def test_codex_batch_proves_foreign_context_between_selected_offsets(tmp_path: Path) -> None:
    path, _, originals, compact, adapter = scene(tmp_path)
    import json

    rows = [json.loads(raw) for raw in path.read_bytes().splitlines()]
    target = next(
        i
        for i, row in enumerate(rows)
        if row.get("type") == "turn_context" and row["payload"]["turn_id"] == "t2"
    )
    rows[target]["payload"]["cwd"] = CWD.replace("team-a", "roguex")  # retain byte offsets
    write_records(path, rows)
    replay = EvidenceReplay("org-a", adapter=adapter)
    hydrated = replay.hydrate_many(compact)
    assert hydrated[:3] == originals[:3]
    assert all(
        item.payload.get("_replay_unavailable") == "source_out_of_scope" for item in hydrated[3:]
    )
