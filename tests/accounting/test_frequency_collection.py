"""Frozen rarity inputs survive new lookups in the same root and collector restart."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ssgrep.accounting import collector_frequency, collector_state, fingerprint, index_frequency
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.contracts import canonical_bytes
from ssgrep.accounting.frequency_assignment import FrequencyAssignments
from ssgrep.accounting.policy import load_profile_policy
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.reduce_io import load_evidence
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import (
    MCP,
    PROFILE,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    results_json,
    source_session,
    ts,
)
from tests.accounting.test_frequency_generations import MetadataRepository, database

OLD = "review_initial_identifier"
LATER = "review_later_identifier"


def body(term: str) -> str:
    return results_json(card("claude:historical:ep:0", term))


def call(call_id: str, term: str, second: int) -> list[dict]:
    return [
        claude_tool_use(
            "same-root",
            call_id,
            MCP,
            {"query": "investigate deployment"},
            ts=ts(12, 9, 0, second),
            message_id=call_id + "-message",
        ),
        claude_tool_result("same-root", call_id, body(term), ts=ts(12, 9, 0, second + 2)),
    ]


def action(call_id: str, term: str, minute: int) -> list[dict]:
    return [
        claude_tool_use(
            "same-root",
            call_id,
            "Bash",
            {"command": f"grep {term} deploy.log"},
            ts=ts(12, 9, minute),
            message_id=call_id + "-message",
        ),
        claude_tool_result("same-root", call_id, "ok", ts=ts(12, 9, minute, 2)),
    ]


def state():
    checkpoint = vs.checkpoint()
    assert checkpoint is not None
    return collector_state.load_state(PROFILE, checkpoint, checkpoint.methodology)


def test_same_root_new_lookup_uses_new_generation_old_no_reuse_lookup_stays_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vs.enable()
    vs.start_capture()
    path = tmp_path / "index.sqlite"
    database(path, 511)
    repository = MetadataRepository(path)
    original = index_frequency.read_index_view

    def current(profile, **kwargs):
        return original(
            profile,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
            cached=kwargs.get("cached"),
        )

    monkeypatch.setattr(index_frequency, "read_index_view", current)
    scene.write_transcript(
        tmp_path,
        "historical",
        source_session("historical", f"{OLD} {LATER}", when_day=11),
        scene.at(hour=9),
    )
    transcript = scene.write_transcript(
        tmp_path,
        "same-root",
        [
            claude_prompt("same-root", "fix deployment", uuid="original-user", ts=ts(12, 9)),
            *call("initial-lookup", OLD, 10),
        ],
        scene.at(hour=9),
    )
    scene.journal_events(
        scene.execution(
            "n-initial", ts(12, 9, 0, 11), text="investigate deployment", transport="mcp"
        )
    )
    scene.collect(scene.at(hour=10))
    first = state()
    native_assignments = [
        row for row in first.frequency_assignments if not row.invocation_id.startswith("journal:")
    ]
    assert len(native_assignments) == 1  # frozen even without candidate or reuse
    old_assignment = native_assignments[0]
    old_bytes = canonical_bytes(old_assignment)
    old_view = collector_frequency.load_view(PROFILE, old_assignment.view_id)
    method = scene.current().methodology_id
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (
            json.dumps(
                {
                    "n_docs": 511,
                    "avgdl": 3.0,
                    "df": {OLD: 1, LATER: 1},
                }
            ),
        ),
    )
    writer.execute("UPDATE metadata SET value='generation-2' WHERE key='last_index_time'")
    writer.commit()
    writer.close()
    scene.append_transcript(
        transcript,
        [
            *action("old-action", OLD, 1),
            *call("later-lookup", LATER, 20),
            *action("later-action", LATER, 2),
        ],
        scene.at(hour=11),
    )
    scene.journal_events(
        scene.execution("n-later", ts(12, 9, 0, 21), text="investigate deployment", transport="mcp")
    )
    scene.collect(scene.at(hour=12))
    second = state()
    assert scene.current().methodology_id == method  # stats are inputs, not model namespaces
    assignments = {
        row.invocation_id: row
        for row in second.frequency_assignments
        if not row.invocation_id.startswith("journal:")
    }
    assert canonical_bytes(assignments[old_assignment.invocation_id]) == old_bytes
    assert len(assignments) == 2
    newer = next(
        row for row in assignments.values() if row.invocation_id != old_assignment.invocation_id
    )
    assert newer.view_id != old_assignment.view_id
    assert collector_frequency.load_view(PROFILE, old_assignment.view_id) == old_view
    assert any(
        dict(candidate.provenance)["frequency"] == newer.view_id
        and candidate.eligibility.value == "supported"
        for candidate in second.ledger.candidates
    )
    assert any(
        dict(candidate.provenance)["frequency"] == old_assignment.view_id
        and "rarity_unknown" in candidate.eligibility_reasons
        for candidate in second.ledger.candidates
    )
    before = vs.ledger_digest()
    fingerprint.forget_keys()  # fresh process-equivalent key/cache and persisted state reload
    scene.collect(scene.at(hour=12))
    assert vs.ledger_digest() == before
    replay = FrequencyAssignments(
        current(PROFILE),
        assignments=second.frequency_assignments,
        loader=lambda view_id: collector_frequency.load_view(PROFILE, view_id),
    )
    loaded = load_evidence()
    from ssgrep.accounting.journal_reader import read_journal

    full = reduce_profile(
        PROFILE,
        loaded.evidence,
        read_journal(PROFILE).events,
        policy=load_profile_policy(PROFILE),
        frequency=current(PROFILE),
        frequency_assignments=replay,
        fingerprint_key=fingerprint.load_key(PROFILE, create=False),
    )
    assert full.frequency_assignments == second.frequency_assignments
    assert {record.frequency_view_id for record in full.candidates} == {
        old_assignment.view_id,
        newer.view_id,
    }
    scene.collect(scene.at(hour=13), CollectOptions(recompute=True))
    replaced = state()
    assert all(row.view_id == newer.view_id for row in replaced.frequency_assignments)
    assert all(
        row.revision_name == f"frequency-recompute:{newer.view_id}"
        for row in replaced.frequency_assignments
    )
    assert scene.current().methodology_id == method
    repository.close()


def test_pending_frequency_completion_refines_receipts_without_counting_usage_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting import index_frequency_build

    vs.enable()
    vs.start_capture()
    path = tmp_path / "pending-index.sqlite"
    database(path, 511)
    writer = sqlite3.connect(path)
    writer.execute(
        "UPDATE metadata SET value=? WHERE key='lexical_stats'",
        (json.dumps({"n_docs": 511, "avgdl": 3.0, "df": {OLD: 1}}),),
    )
    writer.commit()
    writer.close()
    repository = MetadataRepository(path)
    original = index_frequency.read_index_view
    monkeypatch.setattr(index_frequency_build, "VISITS", 128)

    def current(profile, **kwargs):
        return original(
            profile,
            repository=repository,
            authority=lambda cwd: cwd == "/approved",
            authority_id="approved-v1",
            cached=kwargs.get("cached"),
        )

    monkeypatch.setattr(index_frequency, "read_index_view", current)
    scene.write_transcript(
        tmp_path, "historical", source_session("historical", OLD, when_day=11), scene.at(hour=9)
    )
    scene.write_transcript(
        tmp_path,
        "same-root",
        [
            claude_prompt("same-root", "fix deployment", uuid="pending-user", ts=ts(12, 9)),
            *call("pending-lookup", OLD, 10),
            *action("pending-action", OLD, 1),
        ],
        scene.at(hour=9),
    )
    scene.journal_events(
        scene.execution(
            "n-pending", ts(12, 9, 0, 11), text="investigate deployment", transport="mcp"
        )
    )
    scene.collect(scene.at(hour=10))
    pending = state()
    assert pending.frequency_assignments
    assert all(
        row.unavailable_reason == "frequency_build_pending" for row in pending.frequency_assignments
    )
    usage = scene.counter_total(scene.current())
    assert usage > 0
    assert not any(
        candidate.eligibility.value == "supported" for candidate in pending.ledger.candidates
    )
    for _ in range(16):
        fingerprint.forget_keys()
        scene.collect(scene.at(hour=11))
        completed = state()
        if completed.frequency_assignments and all(
            not row.unavailable_reason for row in completed.frequency_assignments
        ):
            break
    else:
        pytest.fail("pending collection inputs never acquired their completed generation")
    assert scene.counter_total(scene.current()) == usage
    assert {row.invocation_id for row in completed.frequency_assignments} == {
        row.invocation_id for row in pending.frequency_assignments
    }
    assert all(
        row.revision_name == f"frequency-build-complete:{row.view_id}"
        for row in completed.frequency_assignments
    )
    assert any(
        candidate.eligibility.value == "supported" for candidate in completed.ledger.candidates
    )
    digest = vs.ledger_digest()
    scene.collect(scene.at(hour=11))
    assert vs.ledger_digest() == digest
    assert scene.counter_total(scene.current()) == usage
    repository.close()
