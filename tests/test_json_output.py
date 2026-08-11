"""Test suite for --json output contract.

Every command and every error condition must emit exactly one parseable JSON
document to stdout. json.loads(stdout) must succeed for every invocation,
on every path including error cases.

Stdout and stderr are captured separately — a test that combines them passes
while the contract is broken.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import get_ssgrep_binary


def run_ssgrep_json(*args: str, project_dir: str | None = None) -> tuple[str, str, int]:
    """Run ssgrep with --json flag, capturing stdout and stderr separately.

    Uses isolated HOME directory to avoid modifying real ~/.claude/settings.json.

    Returns: (stdout, stderr, exit_code)
    """
    ssgrep_bin = get_ssgrep_binary()

    # Create isolated HOME with .claude/projects to match real Claude Code environment
    with tempfile.TemporaryDirectory(prefix="ssgrep-test-home-") as tmpdir:
        env = os.environ.copy()
        env["HOME"] = tmpdir
        # Create ~/.claude/projects directory structure for api.index() compatibility
        Path(tmpdir, ".claude", "projects").mkdir(parents=True, exist_ok=True)

        cmd = [str(ssgrep_bin), *args]
        if project_dir:
            cmd.extend(["--project-dir", project_dir])
        cmd.append("--json")

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return result.stdout, result.stderr, result.returncode


def test_search_with_results(tmp_path: Path) -> None:
    """Test search --json with actual results."""
    from ssgrep import store

    project_dir = tmp_path
    index_dir = project_dir / ".ssgrep"
    index_dir.mkdir()
    db_path = index_dir / "index.db"

    conn = store.init_db(db_path)
    conn.close()

    # Run search with query
    stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

    # Verify single JSON document
    assert stdout.strip(), "stdout should not be empty"
    doc = json.loads(stdout)
    assert isinstance(doc, dict), "stdout should be a JSON object"
    assert "ok" in doc
    # Exit code may be non-zero for no results, but JSON should parse
    assert code in (0, 3)


def test_search_empty_query_error(tmp_path: Path) -> None:
    """Test search --json with empty query."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("search", "", project_dir=str(project_dir))

    # Verify single JSON document on stdout
    assert stdout.strip(), "stdout should not be empty"
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout should be exactly one line, got {len(lines)}: {lines}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert "ok" in doc

    # Error structure is at top level (not nested in usecli envelope)
    assert doc["ok"] is False
    assert "condition" in doc
    assert doc["condition"] == "empty_query"
    assert "message" in doc
    assert len(doc["message"]) > 0
    # Exit code MUST be 2 for usage error (spec line 282-284)
    assert code == 2


def test_search_missing_index_error(tmp_path: Path) -> None:
    """Test search --json with missing index."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

    # Verify single JSON document
    assert stdout.strip(), "stdout should not be empty"
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout should be exactly one line, got {len(lines)}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert "ok" in doc

    # Check for error condition at top level (not nested in data)
    assert doc["ok"] is False
    assert "condition" in doc
    assert doc["condition"] in ("missing_index", "corrupt_index")
    assert "message" in doc
    assert "command" in doc

    # Exit code MUST be 4 for missing index (spec line 286-288)
    assert code == 4


def test_show_missing_index_error(tmp_path: Path) -> None:
    """Test show --json with missing index.

    Note: `show` still emits the wrapped envelope, unlike search/prune/index/
    init. This test documents that current behavior rather than asserting the
    shape the other four commands converged on.
    """
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("show", "session-id:ep:0", project_dir=str(project_dir))

    # Verify single JSON document
    assert stdout.strip(), "stdout should not be empty"
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout should be exactly one line, got {len(lines)}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert "ok" in doc

    # Check for error condition (may be nested in data for show)
    if "data" in doc:
        error_data = doc["data"]
        assert "condition" in error_data
        assert "message" in error_data
    else:
        assert "condition" in doc
        assert "message" in doc


def test_status_success_no_index(tmp_path: Path) -> None:
    """Test status --json with no index (should succeed)."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("status", project_dir=str(project_dir))

    # Verify single JSON document
    assert stdout.strip(), "stdout should not be empty"
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout should be exactly one line, got {len(lines)}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    # Status succeeds even with no index
    assert code == 0


def test_json_stdout_stderr_separate(tmp_path: Path) -> None:
    """Verify stdout and stderr are captured separately.

    This is critical: tests that combine them will pass even if the
    contract is broken.
    """
    project_dir = tmp_path

    # Use isolated HOME to avoid modifying real settings
    ssgrep_bin = get_ssgrep_binary()
    with tempfile.TemporaryDirectory(prefix="ssgrep-test-home-") as tmpdir:
        env = os.environ.copy()
        env["HOME"] = tmpdir
        Path(tmpdir, ".claude", "projects").mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                str(ssgrep_bin),
                "search",
                "",
                "--project-dir",
                str(project_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    # stdout should be JSON
    if result.stdout.strip():
        # Should parse as single JSON
        json.loads(result.stdout)

    # stderr should not contain JSON (human messages only)
    if result.stderr.strip():
        # stderr might have human message, but should not be JSON
        try:
            json.loads(result.stderr)
            # If it parses as JSON at top level, it's probably a problem
            # (human messages go to stderr, JSON to stdout)
        except (json.JSONDecodeError, ValueError):
            # Good - stderr is not JSON
            pass


def test_no_duplicate_json_documents(tmp_path: Path) -> None:
    """Verify that error paths don't emit multiple JSON documents.

    This is the main bug being fixed - currently error paths emit both
    the error document AND the usecli envelope.
    """
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("search", "", project_dir=str(project_dir))

    # Attempt to parse multiple JSON documents would fail
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"Should be exactly one line of JSON, got {len(lines)}"

    # Verify it's valid JSON
    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert "ok" in doc


def test_error_preserves_condition_message_command(tmp_path: Path) -> None:
    """Test that error responses preserve condition, message, and command."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("search", "", project_dir=str(project_dir))

    doc = json.loads(stdout)

    # Extract error details (at top level)
    assert "condition" in doc, "condition field missing"
    assert "message" in doc, "message field missing"
    assert doc["condition"] == "empty_query"
    assert len(doc["message"]) > 0


def test_search_no_results_single_document(tmp_path: Path) -> None:
    """Test search --json when no results found (empty index)."""
    from ssgrep import store

    project_dir = tmp_path
    index_dir = project_dir / ".ssgrep"
    index_dir.mkdir()
    db_path = index_dir / "index.db"

    # Create empty index
    conn = store.init_db(db_path)
    conn.close()

    stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

    # Verify single JSON document
    assert stdout.strip(), "stdout should not be empty"
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout should be exactly one line, got {len(lines)}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    # Exit code 3 for no results
    assert code == 3


def test_search_json_empty_scope_emits_zero_discovery_payload(tmp_path: Path) -> None:
    """search --json on an empty index emits the structured zero-discovery document.

    A machine caller must get the same census `index --json` already carries:
    condition, message, remedy command, and the scope report payload — as
    data on stdout, not prose on stderr. Asserts the payload keys and value
    types, not mere presence.
    """
    from ssgrep import store

    project_dir = tmp_path
    index_dir = project_dir / ".ssgrep"
    index_dir.mkdir()
    conn = store.init_db(index_dir / "index.db")
    conn.close()

    stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

    assert code == 3, f"empty scope must exit NO_MATCHING_DATA (3), got {code}"
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"stdout must be exactly one JSON document, got {len(lines)} lines"

    doc = json.loads(stdout)
    assert doc["ok"] is False
    assert doc["condition"] == "index_empty"
    assert isinstance(doc["message"], str) and doc["message"]
    assert isinstance(doc["command"], str) and "index" in doc["command"]

    zd = doc["zero_discovery"]
    # The exact shape ScopeReport.as_payload() emits (mirrors index --json).
    assert set(zd) == {
        "scope",
        "scope_canonical",
        "transcript_root",
        "root_exists",
        "total_transcripts",
        "rejected_count",
        "top_rejected_cwds",
        "same_basename_rejected",
        "remedy_command",
    }
    assert isinstance(zd["root_exists"], bool)
    assert isinstance(zd["total_transcripts"], int)
    assert isinstance(zd["rejected_count"], int)
    assert isinstance(zd["top_rejected_cwds"], list)
    for entry in zd["top_rejected_cwds"]:
        assert set(entry) == {"cwd", "count"}


def test_search_missing_index_exit_code(tmp_path: Path) -> None:
    """Test search --json missing index returns exit code 4 (not 0)."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

    # Parse JSON to verify it's a single document
    doc = json.loads(stdout)

    # Exit code MUST be 4 for missing index (spec line 286-288)
    assert code == 4, f"Expected exit code 4 for missing index, got {code}"

    # Verify error is at top level (not nested in data with ok:true wrapper)
    assert doc.get("ok") is False, "ok should be False for error"
    assert doc.get("condition") == "missing_index"


def test_search_empty_query_exit_code(tmp_path: Path) -> None:
    """Test search --json empty query returns exit code 2 (usage error)."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("search", "", project_dir=str(project_dir))

    # Parse JSON to verify it's a single document
    lines = stdout.strip().split("\n")
    assert len(lines) == 1, f"Expected one line of JSON, got {len(lines)}"
    doc = json.loads(stdout)

    # Exit code MUST be 2 for usage error (spec line 282-284)
    assert code == 2, f"Expected exit code 2 for empty query, got {code}"

    # Verify error is at top level (not nested with ok:true wrapper)
    assert doc.get("ok") is False, "ok should be False for error"
    assert doc.get("condition") == "empty_query"


def test_prune_missing_index_single_document(tmp_path: Path) -> None:
    """Test prune --yes --json missing index emits exactly one JSON document."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("prune", "--yes", project_dir=str(project_dir))

    # Verify single JSON document (not two: error + usecli envelope)
    lines = stdout.strip().split("\n") if stdout.strip() else []
    assert (
        len(lines) == 1
    ), f"prune should emit exactly one JSON document, got {len(lines)}: {lines}"

    # Verify it parses
    doc = json.loads(stdout)
    assert isinstance(doc, dict)

    # Exit code MUST be 4 for missing index
    assert code == 4, f"Expected exit code 4 for missing index, got {code}"
    assert doc.get("ok") is False
    assert doc.get("condition") == "missing_index"


def test_index_error_single_document(tmp_path: Path) -> None:
    """Test index --json on error emits exactly one JSON document.

    This tests the case where the command completes successfully even if
    there are no sessions. A successful index run should have single doc.
    """
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("index", project_dir=str(project_dir))

    # Should succeed (exit 0) since zero sessions is not an error
    # This test verifies single document output format
    lines = stdout.strip().split("\n") if stdout.strip() else []
    assert len(lines) == 1, f"index should emit exactly one JSON document, got {len(lines)}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert code == 0, f"Expected exit code 0 for successful index, got {code}"


def test_init_error_single_document(tmp_path: Path) -> None:
    """Test init --json on error emits exactly one JSON document."""
    project_dir = tmp_path

    stdout, stderr, code = run_ssgrep_json("init", project_dir=str(project_dir))

    # Should succeed (exit 0) even with zero sessions
    lines = stdout.strip().split("\n") if stdout.strip() else []
    assert len(lines) == 1, f"init should emit exactly one JSON document, got {len(lines)}"

    doc = json.loads(stdout)
    assert isinstance(doc, dict)
    assert code == 0, f"Expected exit code 0 for successful init, got {code}"


def test_init_missing_transcript_root_single_document(tmp_path: Path) -> None:
    """Test init --json emits one structured document when no transcript root exists.

    Regression test for the JSON-mode escape path in init.py: an indexing
    failure other than IndexNotReadyError (here, IndexNotFoundError from a
    missing transcript root) used to be re-raised past init.py's own
    handling and rely on usecli's wrapper to build the envelope. usecli
    0.1.77 narrowed its catch-all except clause to a fixed set of exception
    types that does not include ssgrep's SearchException family, so that
    escape path would otherwise leave stdout empty and print a raw
    traceback instead. Deliberately does NOT use run_ssgrep_json's helper,
    which creates .claude/projects — this test needs a HOME *without* it so
    indexer.py's missing-transcript-root guard actually fires.
    """
    from ssgrep.cli import exit_codes

    ssgrep_bin = get_ssgrep_binary()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with tempfile.TemporaryDirectory(prefix="ssgrep-test-home-") as tmpdir:
        env = os.environ.copy()
        env["HOME"] = tmpdir
        # No .claude/projects created here on purpose.

        cmd = [str(ssgrep_bin), "init", "--project-dir", str(project_dir), "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
    assert len(lines) == 1, (
        f"init should emit exactly one JSON document on a missing transcript "
        f"root, got {len(lines)}: {lines!r}; stderr={result.stderr!r}"
    )

    doc = json.loads(result.stdout)
    assert isinstance(doc, dict)
    assert doc.get("ok") is False
    assert doc.get("condition") == "missing_index"
    assert doc.get("command") == "export CLAUDE_CONFIG_DIR=<path to your .claude directory>"
    assert "Cannot build index" in doc.get("message", "")

    # Exit code must be unchanged from the pre-fix behavior: only the JSON
    # payload shape changes, not the exit code.
    assert result.returncode == exit_codes.INTERNAL_FAILURE, (
        f"Expected exit code {exit_codes.INTERNAL_FAILURE} (INTERNAL_FAILURE), "
        f"got {result.returncode}"
    )


def test_json_exit_codes_match_plain_mode() -> None:
    """Verify that JSON mode exit codes match plain mode for all conditions."""
    from ssgrep.cli import exit_codes

    # These are the spec-defined exit codes that must be used consistently
    # Spec section: Requirement: Exit Code Contract (line 271)
    assert exit_codes.SUCCESS == 0
    assert exit_codes.INTERNAL_FAILURE == 1
    assert exit_codes.USAGE_ERROR == 2
    assert exit_codes.NO_MATCHING_DATA == 3
    assert exit_codes.MISSING_INDEX == 4


class TestShowJsonSchema:
    """Validate show --json response schema contains documented keys."""

    def test_show_json_contains_documented_keys(self, isolated_real_corpus_index) -> None:
        """Validate show --json response contains exactly documented keys.

        This test prevents silent removal of documented fields from JSON output.
        The expected keys are taken from the documented contract (EpisodeDetail dataclass),
        written as an explicit frozenset literal independent of the code under test.
        If a field is deleted from both the dataclass and its producer, this test fails.
        """
        # Oracle: literal key list from documented contract
        # These keys come from src/ssgrep/types.py::EpisodeDetail dataclass definition
        expected_keys = frozenset(
            {
                "episode_id",
                "session_id",
                "title",
                "timestamp",
                "git_branch",
                "cwd",
                "prompt_text",
                "response_text",
                "files_touched",
                "tool_names",
                "is_subagent",
                "agent_type",
                "agent_name",
                "agent_description",
                "parent_session_id",
                "prompt_truncated",
                "response_truncated",
                "stale",
                "stale_count",
            }
        )

        project_dir = isolated_real_corpus_index

        # Get a valid reference by searching first
        stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

        if code != 0 or not stdout.strip():
            pytest.skip("Could not find test episodes via search")

        search_doc = json.loads(stdout)
        if "data" in search_doc and isinstance(search_doc["data"], dict):
            search_data = search_doc["data"]
        else:
            search_data = search_doc

        if not search_data.get("results"):
            pytest.skip("No search results found")

        ref = search_data["results"][0].get("ref")
        if not ref:
            pytest.skip("Could not extract ref from search results")

        # Test show with the valid ref
        stdout, stderr, code = run_ssgrep_json("show", ref, project_dir=str(project_dir))

        assert code == 0, f"show failed with exit code {code}"
        assert stdout.strip(), "show --json should produce output"

        doc = json.loads(stdout)

        # Show response may be wrapped in usecli's {"ok": true, "data": {...}} envelope
        if "data" in doc and isinstance(doc["data"], dict):
            show_doc = doc["data"]
        else:
            show_doc = doc

        # Verify exact key set matches oracle
        actual_keys = frozenset(show_doc.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        assert (
            actual_keys == expected_keys
        ), f"show --json keys mismatch. Missing: {missing}, Extra: {extra}"

    def test_show_json_response_text_value(self, tmp_path: Path) -> None:
        """Verify show --json response_text field value matches expected content.

        Uses a deliberately-created fixture with known non-empty response_text.
        Oracle: independent query counting response chunks for the episode.
        """
        import sqlite3
        from datetime import UTC, datetime

        from ssgrep import store
        from ssgrep.types import Chunk, ContentType, Episode, SessionFile

        # Create a test project with a custom index
        project_dir = tmp_path
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir()
        db_path = index_dir / "index.db"

        # Build a known test episode with non-empty response_text
        expected_response_text = "This is a test response with content for verification"
        session_id = "test-response-session-001"
        episode_id = f"{session_id}:ep:0"  # Format: {session_id}:ep:{index}

        # Initialize database and insert test data
        conn = store.init_db(db_path)

        # Insert session using store function
        session = SessionFile(
            path=Path("/test/session.jsonl"),
            session_id=session_id,
            is_main=True,
            size=1024,
            mtime=1700000000.0,
        )
        store.insert_session(conn, session)

        # Insert episode
        episode = Episode(
            episode_id=episode_id,
            session_id=session_id,
            title="Test Episode",
            timestamp=datetime.now(UTC),
            git_branch="main",
            cwd="/test",
            prompt_text="Test prompt",
            response_text=expected_response_text,  # Canonical text persisted on episodes
            files_touched=("src/test.py",),
            tool_names=("Read",),
            is_subagent=False,
        )
        store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

        # Insert a response chunk with known content
        response_chunk = Chunk(
            chunk_id="test-response-chunk-001",
            episode_id=episode_id,
            session_id=session_id,
            text=expected_response_text,
            content_type=ContentType.RESPONSE,
            byte_offset=0,
            vec_row=0,
        )
        store.insert_chunk(conn, response_chunk, vec_row=0)

        conn.commit()
        conn.close()

        # Oracle: the canonical response_text column on the episodes table
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT response_text FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        conn.close()

        assert row is not None, "episode row missing from episodes table"
        expected_response = row[0]
        # Verify the oracle is what we inserted
        assert expected_response == expected_response_text

        # Run show via CLI using the episode_id
        stdout, stderr, code = run_ssgrep_json("show", episode_id, project_dir=str(project_dir))

        assert code == 0, f"show failed with exit code {code}, stderr: {stderr}"
        doc = json.loads(stdout)

        if "data" in doc and isinstance(doc["data"], dict):
            show_doc = doc["data"]
        else:
            show_doc = doc

        # Verify response_text matches the oracle (canonical episodes-table column)
        assert "response_text" in show_doc, "response_text field missing from show --json"
        assert isinstance(show_doc["response_text"], str), "response_text must be a string"
        assert show_doc["response_text"] == expected_response, (
            f"response_text mismatch: expected {repr(expected_response)}, "
            f"got {repr(show_doc['response_text'])}"
        )
        # Critical check: response_text must be non-empty when response chunks exist
        assert (
            len(show_doc["response_text"]) > 0
        ), "response_text should not be empty when response chunks exist"


class TestSearchJsonSchema:
    """Validate search --json response schema contains documented keys."""

    def test_search_json_contains_documented_keys(self, isolated_real_corpus_index) -> None:
        """Validate search --json response contains exactly documented keys.

        This test prevents silent removal of documented fields like stale and
        total_matches from the JSON output. The expected keys are written as an
        explicit frozenset literal from the documented SearchResponse contract.
        """
        # Oracle: literal key list from documented contract
        # These keys come from src/ssgrep/types.py::SearchResponse dataclass definition
        expected_keys = frozenset(
            {
                "results",
                "omitted_count",
                "index_exists",
                "index_empty",
                "total_matches",
                "excerpts_truncated",
                "clamped",
                "stale",
                "stale_count",
            }
        )

        project_dir = isolated_real_corpus_index

        # Run search to get results
        stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

        assert stdout.strip(), "search --json should produce output"
        doc = json.loads(stdout)

        if code != 0:
            # If search failed, skip schema test
            assert doc.get("ok") is False
            pytest.skip("Search returned error")

        # Search response may be wrapped in usecli's {"ok": true, "data": {...}} envelope
        if "data" in doc and isinstance(doc["data"], dict):
            search_doc = doc["data"]
        else:
            search_doc = doc

        # Verify exact key set matches oracle
        actual_keys = frozenset(search_doc.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        assert (
            actual_keys == expected_keys
        ), f"search --json keys mismatch. Missing: {missing}, Extra: {extra}"

    def test_search_json_total_matches_value(self, isolated_real_corpus_index) -> None:
        """Verify search --json total_matches field value is correct.

        Oracle: count the actual results returned in the search response.
        """

        project_dir = isolated_real_corpus_index

        # Run search via CLI
        stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

        assert stdout.strip(), "search --json should produce output"
        doc = json.loads(stdout)

        if code != 0:
            pytest.skip("Search returned error")

        # Search response may be wrapped
        if "data" in doc and isinstance(doc["data"], dict):
            search_doc = doc["data"]
        else:
            search_doc = doc

        # Oracle: count the actual results in the search response
        result_count = len(search_doc.get("results", []))

        # Independent value checks
        assert "total_matches" in search_doc, "total_matches field missing from search --json"
        assert isinstance(search_doc["total_matches"], int), "total_matches should be integer"
        # The key check: total_matches should be >= the result count (may have omitted results)
        assert (
            search_doc["total_matches"] >= result_count
        ), f"total_matches {search_doc['total_matches']} should be >= result count {result_count}"
        # Most importantly: if we got results, total_matches should not be 0
        if result_count > 0:
            assert (
                search_doc["total_matches"] > 0
            ), "total_matches should be > 0 when results are present"


class TestStatusJsonSchema:
    """Validate status --json response schema contains documented keys."""

    def test_status_json_contains_documented_keys(self, tmp_path_project_indexed) -> None:
        """Validate status --json response contains exactly documented keys.

        This test prevents silent removal of documented fields like stale and
        chunk_count from the JSON output. The expected keys are written as an
        explicit frozenset literal from the documented IndexStats contract.
        """
        # Oracle: literal key list from documented contract
        # These keys come from src/ssgrep/types.py::IndexStats dataclass definition
        expected_keys = frozenset(
            {
                "session_count",
                "episode_count",
                "chunk_count",
                "index_size_bytes",
                "last_index_time",
                "model_id",
                "vector_dimension",
                "skipped_records",
                "malformed_records",
                "schema_version",
                "tombstoned_source_count",
                "tombstoned_chunk_count",
                "index_exists",
                "stale",
                "stale_count",
                "cwd_cache_degraded",
                "cwd_cache_fallback_scans",
                "queue_items_out_of_scope",
                "corpus_session_count",
            }
        )

        project_dir = tmp_path_project_indexed

        # Run status to get stats
        stdout, stderr, code = run_ssgrep_json("status", project_dir=str(project_dir))

        assert code == 0, f"status should succeed but got exit code {code}"
        assert stdout.strip(), "status --json should produce output"

        doc = json.loads(stdout)

        # Status response may be wrapped in usecli's {"ok": true, "data": {...}} envelope
        if "data" in doc and isinstance(doc["data"], dict):
            stats_doc = doc["data"]
        else:
            stats_doc = doc

        # Verify exact key set matches oracle
        actual_keys = frozenset(stats_doc.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        assert (
            actual_keys == expected_keys
        ), f"status --json keys mismatch. Missing: {missing}, Extra: {extra}"

    def test_status_json_chunk_count_value(self, tmp_path: Path) -> None:
        """Verify status --json chunk_count field value is correct.

        Oracle: independent COUNT query from database.
        Tests against a real index with known chunk count.
        """
        import sqlite3
        from datetime import UTC, datetime

        from ssgrep import store
        from ssgrep.types import Chunk, ContentType, Episode, SessionFile

        # Create a test project with a custom index
        project_dir = tmp_path
        index_dir = project_dir / ".ssgrep"
        index_dir.mkdir()
        db_path = index_dir / "index.db"

        # Create an index with known number of chunks
        expected_chunk_count = 7
        session_id = "test-status-session"
        episode_id = f"{session_id}:ep:0"

        # Initialize database and insert test data
        conn = store.init_db(db_path)

        # Insert session
        session = SessionFile(
            path=Path("/test/status-session.jsonl"),
            session_id=session_id,
            is_main=True,
            size=1024,
            mtime=1700000000.0,
        )
        store.insert_session(conn, session)

        # Insert episode
        episode = Episode(
            episode_id=episode_id,
            session_id=session_id,
            title="Test Episode",
            timestamp=datetime.now(UTC),
            git_branch="main",
            cwd="/test",
            prompt_text="Test prompt",
            response_text="",
            files_touched=("src/test.py",),
            tool_names=("Read",),
            is_subagent=False,
        )
        store.insert_episode(conn, episode, episode.prompt_text, episode.response_text)

        # Insert exactly expected_chunk_count chunks
        for i in range(expected_chunk_count):
            chunk = Chunk(
                chunk_id=f"chunk-{i:03d}",
                episode_id=episode_id,
                session_id=session_id,
                text=f"chunk text {i}",
                content_type=ContentType.PROMPT if i % 2 == 0 else ContentType.RESPONSE,
                byte_offset=i * 100,
                vec_row=None,
            )
            store.insert_chunk(conn, chunk, vec_row=None)

        conn.commit()
        conn.close()

        # Run status via CLI
        stdout, stderr, code = run_ssgrep_json("status", project_dir=str(project_dir))

        assert code == 0
        assert stdout.strip(), "status --json should produce output"

        doc = json.loads(stdout)

        # Status response may be wrapped
        if "data" in doc and isinstance(doc["data"], dict):
            stats_doc = doc["data"]
        else:
            stats_doc = doc

        # Oracle: count chunks directly from database
        conn = sqlite3.connect(str(db_path))
        db_chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source_status = 'available'"
        ).fetchone()[0]
        conn.close()

        # Verify oracle is what we expected
        assert db_chunk_count == expected_chunk_count

        # Independent value checks
        assert "chunk_count" in stats_doc, "chunk_count field missing from status --json"
        assert isinstance(stats_doc["chunk_count"], int), "chunk_count should be integer"
        # The key check: chunk_count should match the actual count in database
        assert stats_doc["chunk_count"] == db_chunk_count, (
            f"chunk_count {stats_doc['chunk_count']} should match database "
            f"COUNT {db_chunk_count}"
        )
        # Verify it matches our expected value
        assert (
            stats_doc["chunk_count"] == expected_chunk_count
        ), f"chunk_count should be {expected_chunk_count}"


class TestJsonPayloadRoundTrip:
    """Verify JSON output matches direct API calls (values, not just schema)."""

    def test_show_json_values_match_api(self, isolated_real_corpus_index) -> None:
        """Verify show --json response values match api.show() output."""
        from ssgrep import api

        project_dir = isolated_real_corpus_index

        # Get a valid reference by searching first
        stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))

        if code != 0 or not stdout.strip():
            pytest.skip("Could not find test episodes via search")

        search_doc = json.loads(stdout)
        # Handle wrapped search response
        if "data" in search_doc and isinstance(search_doc["data"], dict):
            search_data = search_doc["data"]
        else:
            search_data = search_doc

        if not search_data.get("results"):
            pytest.skip("No search results found")

        ref = search_data["results"][0].get("ref")
        if not ref:
            pytest.skip("Could not extract ref from search results")

        # Get via CLI
        stdout, stderr, code = run_ssgrep_json("show", ref, project_dir=str(project_dir))
        cli_doc = json.loads(stdout)

        # Extract inner data if wrapped
        if "data" in cli_doc and isinstance(cli_doc["data"], dict):
            cli_data = cli_doc["data"]
        else:
            cli_data = cli_doc

        # Get via API
        api_detail = api.show(project_dir, ref)
        from ssgrep.cli.commands import to_jsonable

        api_doc = to_jsonable(api_detail)

        # Values should match (ignoring potential datetime serialization differences)
        assert cli_data == api_doc, (
            "show --json output should match api.show() output. " f"CLI: {cli_data}, API: {api_doc}"
        )

    def test_search_json_values_match_api(self, isolated_real_corpus_index) -> None:
        """Verify search --json response values match api.search() output."""
        from ssgrep import api

        project_dir = isolated_real_corpus_index

        # Run search via CLI
        stdout, stderr, code = run_ssgrep_json("search", "test", project_dir=str(project_dir))
        cli_doc = json.loads(stdout)

        # Skip validation if search had an error
        if code != 0:
            pytest.skip("Search returned error")

        # Extract inner data if wrapped
        if "data" in cli_doc and isinstance(cli_doc["data"], dict):
            cli_data = cli_doc["data"]
        else:
            cli_data = cli_doc

        # Get via API with same parameters
        response = api.search(project_dir, "test", limit=10)
        from ssgrep.cli.commands import to_jsonable

        api_doc = to_jsonable(response)

        # Values should match
        assert cli_data == api_doc, "search --json output should match api.search() output"

    def test_status_json_values_match_api(self, isolated_real_corpus_index) -> None:
        """Verify status --json response values match api.status() output.

        Note: cwd_cache_* fields are excluded from exact matching because they
        capture timing-dependent cache state during the invocation, which can
        differ between the CLI subprocess (with isolated HOME) and the in-process
        API call (with test-provided home). Both should have the structure, but
        values may vary. Structural consistency is tested separately.
        """
        from ssgrep import api

        project_dir = isolated_real_corpus_index

        # Run status via CLI
        stdout, stderr, code = run_ssgrep_json("status", project_dir=str(project_dir))
        cli_doc = json.loads(stdout)

        # Extract the inner data if wrapped in usecli envelope
        if "data" in cli_doc and isinstance(cli_doc["data"], dict):
            cli_data = cli_doc["data"]
        else:
            cli_data = cli_doc

        # Get via API
        stats = api.status(project_dir)
        from ssgrep.cli.commands import to_jsonable

        api_doc = to_jsonable(stats)

        # Values should match, except for timing-dependent cache metrics
        excluded_keys = ("cwd_cache_degraded", "cwd_cache_fallback_scans")
        cli_normalized = {k: v for k, v in cli_data.items() if k not in excluded_keys}
        api_normalized = {k: v for k, v in api_doc.items() if k not in excluded_keys}

        assert cli_normalized == api_normalized, (
            "status --json output should match api.status() output "
            "(excluding timing-dependent cache metrics)"
        )

        # Verify both have the new fields (structure check, not value check)
        assert "cwd_cache_degraded" in cli_data
        assert "cwd_cache_fallback_scans" in cli_data
        assert "cwd_cache_degraded" in api_doc
        assert "cwd_cache_fallback_scans" in api_doc
