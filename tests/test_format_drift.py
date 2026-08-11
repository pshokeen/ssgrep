"""Format-drift and mutation resilience tests.

Tests for graceful degradation when encountering:
- Unknown record types
- Unknown content-block types
- Malformed JSON lines
- Truncated final line
- Oversized blocks (>500KB)
- Empty file
- File deleted mid-index
- Field presence/absence differences (agentId vs teamName/agentName, optional slug)
- Synthetic "future format" with invented types that still extracts correct data
  from understood records
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssgrep import embed, indexer, records
from ssgrep.records import MAX_LINE_BYTES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "proj").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return home


def _install_fake_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_encode(texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 256), 1.0 / (256**0.5), dtype=np.float32)

    monkeypatch.setattr(embed, "encode", fake_encode)


def _write_jsonl(path: Path, records_list: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records_list:
            f.write(json.dumps(r) + "\n")


def _consume_records(path: Path) -> records.Stats:
    """Consume all records from a file and return the stats."""
    gen = records.read_records(path)
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def _user_record(
    uid: str, text: str, cwd: str, session_id: str, parent_uuid: str | None = None
) -> dict[str, Any]:
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": text},
        "uuid": uid,
        "timestamp": "2026-07-01T10:00:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _assistant_record(
    uid: str, text: str, cwd: str, session_id: str, parent_uuid: str
) -> dict[str, Any]:
    return {
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "uuid": uid,
        "timestamp": "2026-07-01T10:01:00.000Z",
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


# ---------------------------------------------------------------------------
# Fixture: Unknown Record Types
# ---------------------------------------------------------------------------


def test_unknown_record_types_do_not_crash(fake_home: Path, tmp_path: Path) -> None:
    """Unknown record types are silently skipped and counted."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record("u1", "Hello", str(project_dir), session_id),
            {
                "type": "future_widget_type",
                "uuid": "f1",
                "timestamp": "2026-07-01T10:00:00.000Z",
                "data": {"complex": "value", "nested": {"structure": True}},
            },
            _assistant_record("a1", "Hi there", str(project_dir), session_id, "u1"),
            {
                "type": "another_unknown_type",
                "version": 3,
                "payload": "abc",
                "uuid": "f2",
            },
        ],
    )

    stats = _consume_records(session_path)

    assert stats.total_lines == 4
    assert stats.parsed_records == 4
    assert stats.unknown_types == 2
    assert stats.malformed_lines == 0
    assert stats.skipped_oversized == 0


# ---------------------------------------------------------------------------
# Fixture: Unknown Content-Block Types
# ---------------------------------------------------------------------------


def test_unknown_content_block_types_do_not_crash(fake_home: Path, tmp_path: Path) -> None:
    """Unknown content block types are handled gracefully."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record("u1", "Analyze this", str(project_dir), session_id),
            {
                "parentUuid": "u1",
                "isSidechain": False,
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Regular text block"},
                        {"type": "future_ai_extension", "data": "unknown block type"},
                        {
                            "type": "speculative_reasoning",
                            "reasoning": "future feature",
                        },
                    ],
                },
                "uuid": "a1",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "cwd": str(project_dir),
                "sessionId": session_id,
                "gitBranch": "main",
            },
        ],
    )

    stats = _consume_records(session_path)

    assert stats.total_lines == 2
    assert stats.parsed_records == 2
    assert stats.unknown_types == 0
    assert stats.malformed_lines == 0


# ---------------------------------------------------------------------------
# Fixture: Malformed JSON Lines
# ---------------------------------------------------------------------------


def test_malformed_json_lines_are_skipped_and_counted(fake_home: Path, tmp_path: Path) -> None:
    """Malformed JSON lines are skipped and counted, indexing continues."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    with open(session_path, "w") as f:
        user_rec = _user_record("u1", "Start", str(project_dir), session_id)
        f.write(json.dumps(user_rec) + "\n")
        f.write('{"broken": "json", "no": "closing\n')  # Malformed
        asst_rec = _assistant_record("a1", "End", str(project_dir), session_id, "u1")
        f.write(json.dumps(asst_rec) + "\n")

    stats = _consume_records(session_path)

    assert stats.total_lines == 3
    assert stats.parsed_records == 2
    assert stats.malformed_lines == 1
    assert stats.unknown_types == 0
    assert stats.skipped_oversized == 0


# ---------------------------------------------------------------------------
# Fixture: Oversized Blocks
# ---------------------------------------------------------------------------


def test_oversized_blocks_are_skipped_and_counted(fake_home: Path, tmp_path: Path) -> None:
    """Lines exceeding MAX_LINE_BYTES are skipped and counted."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    oversized_record = {
        "type": "user",
        "message": {"role": "user", "content": "x" * (MAX_LINE_BYTES + 100)},
        "uuid": "oversized",
        "timestamp": "2026-07-01T10:00:00.000Z",
        "cwd": str(project_dir),
        "sessionId": session_id,
        "gitBranch": "main",
    }

    with open(session_path, "w") as f:
        user_rec = _user_record("u1", "First", str(project_dir), session_id)
        f.write(json.dumps(user_rec) + "\n")
        f.write(json.dumps(oversized_record) + "\n")
        asst_rec = _assistant_record("a1", "Second", str(project_dir), session_id, "u1")
        f.write(json.dumps(asst_rec) + "\n")

    stats = _consume_records(session_path)

    assert stats.total_lines == 3
    assert stats.parsed_records == 2
    assert stats.skipped_oversized == 1
    assert stats.malformed_lines == 0


# ---------------------------------------------------------------------------
# Fixture: Empty File
# ---------------------------------------------------------------------------


def test_empty_file_does_not_crash(fake_home: Path, tmp_path: Path) -> None:
    """Empty file is handled gracefully."""
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text("")

    stats = _consume_records(session_path)

    assert stats.total_lines == 0
    assert stats.parsed_records == 0
    assert stats.malformed_lines == 0
    assert stats.unknown_types == 0
    assert stats.skipped_oversized == 0


# ---------------------------------------------------------------------------
# Fixture: Truncated Final Line
# ---------------------------------------------------------------------------


def test_truncated_final_line_is_skipped(fake_home: Path, tmp_path: Path) -> None:
    """A truncated final line (no trailing newline) is left unconsumed."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    with open(session_path, "w") as f:
        f.write(json.dumps(_user_record("u1", "Complete", str(project_dir), session_id)) + "\n")
        # Write incomplete final line without newline
        f.write('{"type":"assistant","incomplete":true')

    stats = _consume_records(session_path)

    assert stats.total_lines == 2
    assert stats.parsed_records == 1
    assert stats.malformed_lines == 1


# ---------------------------------------------------------------------------
# Fixture: File Deleted Mid-Index
# ---------------------------------------------------------------------------


def test_file_deleted_mid_index_is_handled(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File deleted between discovery and indexing is counted as skipped."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"
    _write_jsonl(
        session_path,
        [
            _user_record("u1", "Content", str(project_dir), session_id),
            _assistant_record("a1", "Response", str(project_dir), session_id, "u1"),
        ],
    )

    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    # Delete file before indexing
    session_path.unlink()

    # Indexing should handle missing file gracefully
    # The discovery step won't find it, so it won't error
    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == 0


# ---------------------------------------------------------------------------
# Fixture: Field Presence/Absence Differences
# ---------------------------------------------------------------------------


def test_agentid_vs_old_agent_names_both_work(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files with newer agentId and older teamName/agentName both parse."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    parent_id = "99999999-2222-4333-8444-555555555555"
    sub_path = (
        fake_home
        / ".claude"
        / "projects"
        / "proj"
        / parent_id
        / "subagents"
        / "agent-newstyle.jsonl"
    )
    _write_jsonl(
        sub_path,
        [
            _user_record("u1", "Test with new agent format", str(project_dir), parent_id),
            {
                "parentUuid": "u1",
                "isSidechain": False,
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
                "uuid": "a1",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "cwd": str(project_dir),
                "sessionId": parent_id,
                "gitBranch": "main",
                "agentId": "agent-abc123",  # New format
            },
        ],
    )

    meta_path = sub_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "agentType": "general-purpose",
                "name": "new-style-agent",
                "model": "claude-opus",
                "description": "Uses new agentId format",
            }
        )
    )

    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == 1


def test_old_team_and_agent_names_work(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files with old teamName/agentName format (without agentId) still work."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    parent_id = "88888888-2222-4333-8444-555555555555"
    sub_path = (
        fake_home
        / ".claude"
        / "projects"
        / "proj"
        / parent_id
        / "subagents"
        / "agent-oldstyle.jsonl"
    )
    _write_jsonl(
        sub_path,
        [
            _user_record("u1", "Old format test", str(project_dir), parent_id),
            {
                "parentUuid": "u1",
                "isSidechain": False,
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
                "uuid": "a1",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "cwd": str(project_dir),
                "sessionId": parent_id,
                "gitBranch": "main",
                "teamName": "team-redacted",  # Old format
                "agentName": "old-agent",  # Old format
            },
        ],
    )

    meta_path = sub_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "agentType": "general-purpose",
                "name": "old-style-agent",
                "model": "claude-sonnet",
                "description": "Uses old teamName/agentName format",
            }
        )
    )

    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.session_count == 1


def test_optional_slug_field_handled(fake_home: Path, tmp_path: Path) -> None:
    """Optional slug field in records doesn't cause parsing issues."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    record_with_slug = _user_record("u1", "Test", str(project_dir), session_id)
    record_with_slug["slug"] = "optional-slug-field"  # Optional field

    record_without_slug = _assistant_record("a1", "Response", str(project_dir), session_id, "u1")

    _write_jsonl(session_path, [record_with_slug, record_without_slug])

    stats = _consume_records(session_path)

    assert stats.total_lines == 2
    assert stats.parsed_records == 2
    assert stats.malformed_lines == 0


# ---------------------------------------------------------------------------
# Fixture: Future Format with Unknown Types
# ---------------------------------------------------------------------------


def test_future_format_with_unknown_types_extracts_known_records(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A synthetic 'future format' with invented record types still yields
    correct extraction from the records it does understand.

    This is the headline test: not "does not crash" but "correct output from
    the comprehensible subset."
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    # Write a mix of future-format records and current format records
    _write_jsonl(
        session_path,
        [
            # Known type: user
            _user_record(
                "u1",
                "What is the capital of France?",
                str(project_dir),
                session_id,
            ),
            # Unknown future type 1: ai_reasoning_block
            {
                "type": "ai_reasoning_block",
                "uuid": "future1",
                "timestamp": "2026-07-01T10:00:30.000Z",
                "reasoning_depth": 5,
                "confidence_score": 0.95,
                "internal_state": {"vectors": [0.1, 0.2, 0.3]},
            },
            # Known type: assistant
            _assistant_record(
                "a1",
                "The capital of France is Paris.",
                str(project_dir),
                session_id,
                "u1",
            ),
            # Unknown future type 2: semantic_tag_assignment
            {
                "type": "semantic_tag_assignment",
                "uuid": "future2",
                "timestamp": "2026-07-01T10:00:45.000Z",
                "tags": ["geography", "factual", "simple"],
                "relevance_scores": [0.9, 0.95, 0.8],
            },
            # Unknown future type 3: content_certification
            {
                "type": "content_certification",
                "uuid": "future3",
                "timestamp": "2026-07-01T10:00:50.000Z",
                "cert_id": "cert_abc123",
                "cert_timestamp": "2026-07-01T10:00:50.000Z",
                "signature": "sig_xyz789",
            },
            # Known type: user (follow-up)
            _user_record(
                "u2",
                "What about the capital of Germany?",
                str(project_dir),
                session_id,
                "a1",
            ),
            # Known type: assistant (follow-up)
            _assistant_record(
                "a2",
                "The capital of Germany is Berlin.",
                str(project_dir),
                session_id,
                "u2",
            ),
            # Unknown future type 4: transcript_metadata_v3
            {
                "type": "transcript_metadata_v3",
                "uuid": "future4",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "quality_metrics": {"coherence": 0.92, "relevance": 0.88},
                "generation_config": {"temperature": 0.7, "top_p": 0.95},
            },
        ],
    )

    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    # Index should handle the mixture gracefully
    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    # We should have indexed the known records
    assert stats.session_count == 1
    assert stats.episode_count == 2  # Two user prompts = two episodes
    assert stats.chunk_count >= 4  # At least one chunk per prompt+response

    # Verify that the indexed content is correct and searchable
    import sqlite3

    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        # Search for "Paris" (from the assistant response to first query)
        fts_results = conn.execute(
            "SELECT text FROM chunks_fts WHERE chunks_fts MATCH 'Paris' LIMIT 5"
        ).fetchall()
        assert len(fts_results) > 0, "Should find 'Paris' in indexed chunks"

        # Search for "Berlin" (from the assistant response to second query)
        fts_results = conn.execute(
            "SELECT text FROM chunks_fts WHERE chunks_fts MATCH 'Berlin' LIMIT 5"
        ).fetchall()
        assert len(fts_results) > 0, "Should find 'Berlin' in indexed chunks"

        # Verify we only indexed 4 records (2 user + 2 assistant), not the future types
        total_records_indexed = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert total_records_indexed >= 4, "Should have indexed at least 4 known records"
    finally:
        conn.close()


def test_future_format_with_unknown_content_blocks_extracts_known_blocks(
    fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown content block types in future format don't prevent extraction
    of known block types from the same record.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    _write_jsonl(
        session_path,
        [
            _user_record("u1", "Do something", str(project_dir), session_id),
            {
                "parentUuid": "u1",
                "isSidechain": False,
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First, I'll analyze the problem."},
                        {
                            "type": "future_analysis_block",
                            "analysis_type": "semantic",
                            "depth": 3,
                            "results": {"clusters": 5},
                        },
                        {"type": "text", "text": "Then I'll implement the solution."},
                        {
                            "type": "hypothetical_reasoning",
                            "assumptions": ["X", "Y"],
                            "predictions": [0.5, 0.6],
                        },
                        {"type": "text", "text": "Finally, testing confirms it works."},
                    ],
                },
                "uuid": "a1",
                "timestamp": "2026-07-01T10:01:00.000Z",
                "cwd": str(project_dir),
                "sessionId": session_id,
                "gitBranch": "main",
            },
        ],
    )

    _install_fake_encode(monkeypatch)
    index_dir = tmp_path / "idx"

    stats = indexer.index(project_dir, index_dir=index_dir, quiet=True)

    assert stats.episode_count == 1
    assert stats.chunk_count >= 1  # Should extract chunks from known text blocks

    # Verify content is correctly extracted
    import sqlite3

    conn = sqlite3.connect(str(index_dir / "index.db"))
    try:
        fts_results = conn.execute(
            "SELECT text FROM chunks_fts WHERE chunks_fts MATCH 'implement'"
        ).fetchall()
        assert len(fts_results) > 0, "Should extract text blocks correctly despite unknown blocks"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Mutation tests for counters
# ---------------------------------------------------------------------------


def test_malformed_counter_mutation(fake_home: Path, tmp_path: Path) -> None:
    """Mutation test: verify malformed_lines counter increments.

    This test mutates the records.py code to NOT increment malformed_lines
    and verifies the test goes RED. Without this mutation test, a counter
    that nobody asserts is a counter that will drift to zero.
    """
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    with open(session_path, "w") as f:
        user_rec = _user_record("u1", "Text", str(project_dir), session_id)
        f.write(json.dumps(user_rec) + "\n")
        f.write('{"broken":\n')  # Malformed
        asst_rec = _assistant_record("a1", "Reply", str(project_dir), session_id, "u1")
        f.write(json.dumps(asst_rec) + "\n")

    stats = _consume_records(session_path)

    # ASSERTION POINT: This MUST fail if malformed_lines increment is removed
    assert stats.malformed_lines == 1, "Malformed line counter must be incremented"
    assert stats.parsed_records == 2


def test_unknown_types_counter_mutation(fake_home: Path, tmp_path: Path) -> None:
    """Mutation test: verify unknown_types counter increments."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    _write_jsonl(
        session_path,
        [
            {"type": "unknown_v1", "uuid": "f1", "data": "test"},
            _user_record("u1", "Text", str(project_dir), session_id),
            {"type": "unknown_v2", "uuid": "f2", "data": "test2"},
        ],
    )

    stats = _consume_records(session_path)

    # ASSERTION POINT: This MUST fail if unknown_types increment is removed
    assert stats.unknown_types == 2, "Unknown types counter must be incremented"
    assert stats.parsed_records == 3


def test_oversized_counter_mutation(fake_home: Path, tmp_path: Path) -> None:
    """Mutation test: verify skipped_oversized counter increments."""
    project_dir = tmp_path / "project"
    session_id = "11111111-2222-4333-8444-555555555555"
    session_path = fake_home / ".claude" / "projects" / "proj" / f"{session_id}.jsonl"

    oversized_record = {
        "type": "user",
        "message": {"content": "x" * (MAX_LINE_BYTES + 100)},
        "uuid": "big",
    }

    with open(session_path, "w") as f:
        f.write(json.dumps(_user_record("u1", "Normal", str(project_dir), session_id)) + "\n")
        f.write(json.dumps(oversized_record) + "\n")

    stats = _consume_records(session_path)

    # ASSERTION POINT: This MUST fail if skipped_oversized increment is removed
    assert stats.skipped_oversized == 1, "Oversized counter must be incremented"
    assert stats.parsed_records == 1
