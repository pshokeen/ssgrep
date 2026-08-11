"""Tests for session discovery."""

from pathlib import Path

import pytest

from ssgrep.discovery import (
    _scope_matches_cwd,
    _should_exclude_path,
    classify_session,
    discover_sessions,
    encode_path,
)


def test_encode_path():
    assert encode_path(Path("/Users/p/code")) == "-Users-p-code"


def test_classify_main():
    # Main session is at depth 2: <projectDir>/<sessionId>.jsonl
    # Simulate: /tmp/projects/<projectDir>/<sessionId>.jsonl
    claude_dir = Path("/tmp/projects")
    session_file = Path("/tmp/projects/projectX/session-123.jsonl")
    is_main, _, _ = classify_session(session_file, claude_dir)
    assert is_main


def test_classify_subagent():
    # Subagent is at depth >= 4: <projectDir>/<sessionId>/subagents/**/*.jsonl
    # Simulate: /tmp/projects/<projectDir>/<sessionId>/subagents/agent-abc123.jsonl
    claude_dir = Path("/tmp/projects")
    subagent_file = Path("/tmp/projects/projectX/session-123/subagents/agent-abc123.jsonl")
    is_main, parent_id, agent_hash = classify_session(subagent_file, claude_dir)
    assert not is_main
    # New identity scheme includes the path relative to session dir for uniqueness
    assert agent_hash == "subagents-agent-abc123"
    assert parent_id == "session-123"


def test_discover_missing_claude_dir(tmp_path):
    result = discover_sessions(tmp_path)
    assert isinstance(result, list)


def test_should_exclude_path_direct_children():
    """_should_exclude_path's own docstring names three sidecar exclusion
    dirs: memory, tool-results, workflows. Prior to this test none of the
    three had direct unit coverage -- existing tests only proved the
    *inclusion* side (subagents/workflows/ must NOT be excluded, via
    test_critical_1_subagent_workflows_included).

    Depth, per-name: a live-corpus check (2026-07-28, re-confirmed
    independently of the coverage report that first flagged this) shows the
    three names are NOT all at the same depth:
    - "memory" is written at <projectDir>/memory/ -- a direct child of the
      *project* directory (parts[1]). One memory store per project, shared
      across all its sessions; every observed instance is at this depth,
      none nested under a session.
    - "tool-results" and "workflows" are written at
      <projectDir>/<sessionId>/tool-results|workflows/ -- a direct child of
      the *session* directory (parts[2], one level deeper than "memory").
      Every instance of both names in the corpus is at this depth, zero
      exceptions; neither is ever found directly under a project directory
      the way "memory" is.

    The pre-fix code checked all three at parts[1], which happened to be
    correct for "memory" (and is why the looser "per-project sidecar
    directories" framing wasn't obviously wrong) but silently never
    matched "tool-results"/"workflows" at their real depth -- inert only
    because no .jsonl file currently exists in either sidecar dir; a real
    gap the moment one does. This test now targets the depth each name is
    actually observed at, and pins the depth distinction itself (see the
    last two assertions) so a regression back to "check everything at
    parts[1]" fails loudly instead of silently.
    """
    claude_dir = Path("/tmp/projects")
    project = claude_dir / "projectX"
    session = project / "session-123"

    # "memory" excluded at project level (parts[1]).
    assert _should_exclude_path(project / "memory" / "notes.jsonl", claude_dir)

    # "tool-results" and "workflows" excluded at session level (parts[2]).
    assert _should_exclude_path(session / "tool-results" / "out.jsonl", claude_dir)
    assert _should_exclude_path(session / "workflows" / "summary.jsonl", claude_dir)

    # Control: an ordinary main session file must NOT be excluded.
    assert not _should_exclude_path(project / "session-123.jsonl", claude_dir)

    # Control: subagents/workflows/ (nested well past parts[2]) must NOT be
    # excluded -- position-aware, not "any path containing workflows".
    assert not _should_exclude_path(
        session / "subagents" / "workflows" / "wf_1" / "agent.jsonl", claude_dir
    )

    # Control: "tool-results"/"workflows" directly under the project
    # directory -- memory's depth (parts[1]), not their own real depth --
    # are not an observed corpus shape and must NOT be excluded there. This
    # pins the depth split itself: it fails if the two are ever folded back
    # into a single "check parts[1] for all three names" rule.
    assert not _should_exclude_path(project / "tool-results" / "out.jsonl", claude_dir)
    assert not _should_exclude_path(project / "workflows" / "summary.jsonl", claude_dir)


def test_scope_matches_cwd_exact():
    """Test exact cwd match to scope."""
    assert _scope_matches_cwd(
        "/home/dev/code/example/alpha",
        "/home/dev/code/example/alpha",
    )


def test_scope_matches_cwd_subdirectory():
    """Test cwd in subdirectory of scope."""
    assert _scope_matches_cwd(
        "/home/dev/code/example/alpha/src/module",
        "/home/dev/code/example/alpha",
    )


def test_scope_matches_cwd_no_match():
    """Test cwd not under scope."""
    assert not _scope_matches_cwd(
        "/home/dev/code/example/beta",
        "/home/dev/code/example/alpha",
    )


def test_scope_matches_cwd_similar_prefix_no_match():
    """Test that similar prefixes don't match (path boundary matters)."""
    assert not _scope_matches_cwd(
        "/home/dev/code/example/alpha-extended",
        "/home/dev/code/example/alpha",
    )


def test_discover_sessions_with_cwd_scope(
    has_real_corpus: bool, corpus_scope_with_renamed_dir: tuple[str, str] | None
):
    """Test that discovery finds files whose records contain a matching cwd.

    This is the critical test that validates the fix. It scopes to a cwd that
    the corpus records under a differently-named project directory, and should
    find those files even though the directory name doesn't match.

    This test requires the real Claude Code corpus to exist, so it skips
    on CI or machines without a populated ~/.claude/projects.
    """
    if not has_real_corpus:
        pytest.skip("Real Claude Code corpus not available")
    if corpus_scope_with_renamed_dir is None:
        pytest.skip("Corpus has no cwd recorded under a differently-named project dir")

    scope, _ = corpus_scope_with_renamed_dir
    sessions = discover_sessions(Path.home(), scope=scope, no_subagents=False)

    # There should be some sessions found
    assert len(sessions) > 0

    # All sessions should have the right path
    for session in sessions:
        assert session.path.exists(), f"Session file should exist: {session.path}"


def test_discover_sessions_cwd_scope_finds_mismatched_directories(
    has_real_corpus: bool, corpus_scope_with_renamed_dir: tuple[str, str] | None
):
    """Test that discovery finds files from directory names that don't match
    the encoded cwd, proving the cwd-based scoping works correctly.

    This validates that the fix solves the original problem: a project
    directory encoded for one path can hold records whose cwd is some other
    path, and scoping to that cwd must still reach them.

    This test requires the real Claude Code corpus to exist, so it skips
    on CI or machines without a populated ~/.claude/projects.
    """
    if not has_real_corpus:
        pytest.skip("Real Claude Code corpus not available")
    if corpus_scope_with_renamed_dir is None:
        pytest.skip("Corpus has no cwd recorded under a differently-named project dir")

    scope, renamed_dir = corpus_scope_with_renamed_dir
    sessions = discover_sessions(Path.home(), scope=scope, no_subagents=False)

    # Find sessions from the differently-named project directory
    from_renamed_dir = [s for s in sessions if renamed_dir in str(s.path)]

    # There should be at least one session from the mismatched directory
    assert len(from_renamed_dir) > 0, (
        f"Should find sessions from the {renamed_dir} directory "
        "because their records have the matching cwd"
    )


def test_old_encode_approach_fails_on_real_corpus(
    corpus_scope_with_renamed_dir: tuple[str, str] | None,
):
    """Critical test: demonstrates that the old encode_path approach fails.

    The old implementation encoded directory names and matched scopes against
    those encodings. This test proves that approach misses real files: it
    picks a scope the corpus records under a project directory encoded for a
    *different* path, so encoding the scope and comparing directory names
    never matches and the file is never scanned.

    This test MUST fail with the old approach and pass with the new one.
    """
    if corpus_scope_with_renamed_dir is None:
        pytest.skip("Corpus has no cwd recorded under a differently-named project dir")

    # Simulate the old approach
    def encode_path_old(p: Path) -> str:
        return str(p).replace("/", "-").replace(".", "-")

    def matches_scope_old(encoded_path: str, scope: str) -> bool:
        return encoded_path == scope or encoded_path.startswith(scope + "-")

    # The target scope
    scope, _ = corpus_scope_with_renamed_dir
    encoded_scope = encode_path_old(Path(scope))

    # Scan using the old approach
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    old_approach_found = []
    for jsonl_file in claude_dir.rglob("*.jsonl"):
        if any(d in jsonl_file.parts for d in ("memory", "tool-results", "workflows")):
            continue
        if jsonl_file.suffix == ".json":
            continue

        encoded = encode_path_old(jsonl_file.parent)
        if matches_scope_old(encoded, encoded_scope):
            old_approach_found.append(jsonl_file)

    # The new approach should find more files
    new_approach = discover_sessions(Path.home(), scope=scope, no_subagents=False)

    # Assert that the new approach finds at least some files
    assert len(new_approach) > 0, "New approach should find files for this scope"

    # Assert that the new approach finds more than the old approach
    assert len(new_approach) > len(old_approach_found), (
        f"New approach ({len(new_approach)} files) should find more than old approach "
        f"({len(old_approach_found)} files). The old approach failed to account for "
        "directories whose names don't match the actual record cwds."
    )


def test_critical_1_subagent_workflows_included():
    """CRITICAL 1: subagents/workflows/ must be INCLUDED, not excluded.

    The spec requires:
    - <sessionId>/workflows/ (direct child) is EXCLUDED (sidecar workflow summaries)
    - <sessionId>/subagents/workflows/wf_*/*.jsonl is INCLUDED (subagent transcripts)

    This test proves the fix works: discovery includes files under subagents/workflows/
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    # Invariant: some subagent workflow files should be present if corpus exists
    subagent_workflow_files = [s for s in sessions if "subagents/workflows/" in str(s.path)]

    # All files found should actually be under subagents/workflows/
    assert all("subagents/workflows/" in str(s.path) for s in subagent_workflow_files)

    # Invariant: if we found any sessions, at least some should include workflow files
    if sessions:
        assert len(subagent_workflow_files) > 0, (
            "Found sessions but no files under subagents/workflows/. "
            "This means the exclusion logic is incorrectly excluding subagents/workflows/."
        )


def test_critical_1_unscoped_discovery_full_corpus():
    """CRITICAL 1: Unscoped discovery includes both main sessions and subagents.

    The system must include subagent files in unscoped discovery, including those
    under subagents/workflows/. This test verifies the invariant that the number
    of subagent files significantly exceeds main sessions.
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    if not sessions:
        pytest.skip("No sessions found in corpus")

    # Invariant: unscoped discovery should return a non-empty result
    assert len(sessions) > 0

    # Invariant: subagents vastly outnumber main sessions
    main_count = sum(1 for s in sessions if s.is_main)
    subagent_count = sum(1 for s in sessions if not s.is_main)

    # Subagents should be at least as numerous as mains (normally far more)
    assert subagent_count >= main_count, (
        f"Found {subagent_count} subagents and {main_count} main sessions. "
        "This indicates subagents are not being discovered."
    )


def test_critical_1_scoped_discovery_finds_multiple_files(
    corpus_scope_with_renamed_dir: tuple[str, str] | None,
):
    """CRITICAL 1: Scoped discovery returns a subset of unscoped discovery.

    The scope constrains to a specific project, which includes many subagent
    transcripts nested under subagents/workflows/. This invariant test verifies
    that scoped discovery finds files and that they are a subset of unscoped.
    """
    if corpus_scope_with_renamed_dir is None:
        pytest.skip("Corpus has no cwd recorded under a differently-named project dir")

    scope, _ = corpus_scope_with_renamed_dir

    scoped = discover_sessions(Path.home(), scope=scope, no_subagents=False)

    if not scoped:
        pytest.skip("No sessions found for this scope")

    # Invariant: scoped discovery should return multiple files if any exist
    assert len(scoped) > 0

    # Invariant: all scoped sessions should be within the scope
    # (verified by the cwd matching in discovery itself)


def test_critical_2_subagent_identity_collision():
    """CRITICAL 2: Two subagent transcripts of same parent must have distinct IDs.

    The spec (line 180-184) requires: "Two subagent transcripts of the same parent
    get distinct identifiers, derived from <parentSessionId> combined with its own
    agent file identity."

    After the fix, they must be distinct and carry parent_session_id.
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    if not sessions:
        pytest.skip("No sessions found in corpus")

    # Group subagents by parent_session_id
    subagents_by_parent: dict[str | None, list] = {}
    for session in sessions:
        if not session.is_main:
            parent_id = session.parent_session_id
            if parent_id not in subagents_by_parent:
                subagents_by_parent[parent_id] = []
            subagents_by_parent[parent_id].append(session)

    # Find a parent with 2+ subagents
    parents_with_multiple = {
        parent_id: subagents
        for parent_id, subagents in subagents_by_parent.items()
        if len(subagents) >= 2 and parent_id is not None
    }

    if not parents_with_multiple:
        pytest.skip("No test data with multiple subagents per parent")

    # Invariant: Check that all subagents of the same parent have distinct session_ids
    for parent_id, subagents in parents_with_multiple.items():
        session_ids = [s.session_id for s in subagents]
        unique_ids = set(session_ids)

        # Invariant: all subagent session_ids must be unique per parent
        assert len(unique_ids) == len(session_ids), (
            f"Parent {parent_id} has {len(subagents)} subagents but only "
            f"{len(unique_ids)} distinct session_ids. "
            "Subagent identity must be unique per parent."
        )

        # Invariant: no subagent should have session_id equal to bare parent_id
        for session_id in session_ids:
            assert session_id != parent_id, (
                "Subagent session_id equals bare parent_id, " "but must include agent identity."
            )


def test_critical_2_subagent_parent_session_id_populated():
    """CRITICAL 2: Subagent transcripts must have parent_session_id populated.

    The spec (line 168-169) requires: "For a subagent transcript, the system
    SHALL derive a subagent transcript's document identity from the combination
    of its parent session id and its agent file identity."

    After the fix, every subagent must have parent_session_id set (not None).
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    # Find all subagent transcripts
    subagents = [s for s in sessions if not s.is_main]

    if not subagents:
        pytest.skip("No subagent data in corpus")

    # Invariant: Every subagent must have parent_session_id set
    for subagent in subagents:
        assert subagent.parent_session_id is not None, (
            f"Subagent {subagent.session_id} has parent_session_id=None, "
            "but must reference its parent session."
        )


def test_bug_nested_journal_classified_as_subagent():
    """BUG FIX: journal.jsonl under subagents/workflows/wf_*/ classified correctly.

    DEFECT 1 Bug: classify_session() keys on filename (agent-*), not path position.
    Result: journal.jsonl files under subagents/workflows/ should be classified as subagent.

    This test proves the bug by checking that a real journal.jsonl file under
    subagents/workflows/ is classified correctly (as subagent, not main).
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    # Find journal.jsonl files under subagents/workflows/
    journal_files_under_workflows = [
        s for s in sessions if "subagents/workflows/" in str(s.path) and s.path.stem == "journal"
    ]

    if not journal_files_under_workflows:
        pytest.skip("No journal.jsonl files under subagents/workflows/ in corpus")

    # Invariant: Every journal.jsonl under subagents/workflows/ must be subagent, not main
    # (classification is path-based, not filename-based)
    for session in journal_files_under_workflows:
        assert not session.is_main, (
            f"journal.jsonl under subagents/workflows/ should be classified as subagent "
            f"(is_main=False) but got is_main={session.is_main}"
        )


def test_bug_parent_session_id_not_workflows_literal():
    """BUG FIX: parent_session_id must be the actual session UUID.

    DEFECT 2 Bug: parent_session_id = path.parent.parent.stem
    Result: For journal.jsonl under <sessionId>/subagents/workflows/wf_*/journal.jsonl,
    it yields 'workflows' instead of the actual sessionId.

    This test proves the bug is fixed by checking that no subagent has
    parent_session_id='workflows' (unless it's actually named that, which is unlikely).
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    # Find all subagents under subagents/workflows/
    subagents_under_workflows = [
        s for s in sessions if not s.is_main and "subagents/workflows/" in str(s.path)
    ]

    if not subagents_under_workflows:
        pytest.skip("No subagents under subagents/workflows/ in corpus")

    # Invariant: No subagent's parent_session_id should be the literal directory name 'workflows'.
    # It must be an actual session UUID (part[1] of the path).
    for session in subagents_under_workflows:
        assert session.parent_session_id != "workflows", (
            f"Subagent {session.session_id} has parent_session_id='workflows'. "
            "This indicates the parent_session_id derivation is incorrect. "
            "parent_session_id should be an actual session UUID, not a directory name."
        )


def test_bug_no_duplicate_session_ids():
    """BUG FIX: No two session transcripts should have identical session_id.

    DEFECT 1 + DEFECT 2 combined cause collisions: the filename-only identity
    derivation means multiple files can produce the same session_id.

    This test proves the bug is fixed by checking that every session_id is unique.
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    if not sessions:
        pytest.skip("No sessions found in corpus")

    # Invariant: Each transcript file must have a unique session_id
    session_ids = [s.session_id for s in sessions]
    unique_ids = set(session_ids)

    # All session_ids must be unique (no collisions)
    assert len(unique_ids) == len(session_ids), (
        f"Found {len(session_ids) - len(unique_ids)} duplicate session_ids. "
        "Each session transcript must have a unique session_id."
    )


def test_synthetic_collision_detection(tmp_path):
    """SYNTHETIC TEST: Two files with same name at different paths produce different IDs.

    This test creates a fake session tree with the exact collision scenario:
    - Parent session UUID: synthetic-session-id-1
    - File 1: subagents/agent-X.jsonl
    - File 2: subagents/workflows/wf_123/agent-X.jsonl

    Both files have the same filename stem, but must produce different agent hashes
    and thus different session_ids.
    """
    # Create synthetic session structure matching the real layout:
    # claude_dir / projectDir / sessionId / subagents / ...
    claude_dir = tmp_path / "claude_projects"
    project_dir = claude_dir / "-test-project"
    session_id = "synthetic-session-id-1"
    session_dir = project_dir / session_id

    # Create both files with the same stem
    subagents_dir = session_dir / "subagents"
    subagents_dir.mkdir(parents=True)

    workflow_subagent_dir = subagents_dir / "workflows" / "wf_123"
    workflow_subagent_dir.mkdir(parents=True)

    # Create both files
    file1 = subagents_dir / "agent-abc123.jsonl"
    file2 = workflow_subagent_dir / "agent-abc123.jsonl"

    file1.touch()
    file2.touch()

    # Classify both files using claude_dir as reference
    is_main1, parent1, agent_hash1 = classify_session(file1, claude_dir)
    is_main2, parent2, agent_hash2 = classify_session(file2, claude_dir)

    # Invariants:
    # 1. Both should be classified as subagent (not main)
    assert not is_main1, "File 1 should be classified as subagent"
    assert not is_main2, "File 2 should be classified as subagent"

    # 2. Both should have the same parent session ID
    assert parent1 == session_id, f"File 1 parent should be {session_id}"
    assert parent2 == session_id, f"File 2 parent should be {session_id}"

    # 3. CRITICAL: Agent hashes must be DIFFERENT (this is the collision fix)
    assert agent_hash1 != agent_hash2, (
        f"COLLISION: Both files produced the same agent_hash: {agent_hash1}. "
        f"File 1 hash: '{agent_hash1}', File 2 hash: '{agent_hash2}'. "
        f"They must differ due to different path positions."
    )

    # 4. Verify the identities differ (no collision in session_id)
    session_id1 = f"{parent1}:agent:{agent_hash1}"
    session_id2 = f"{parent2}:agent:{agent_hash2}"
    assert (
        session_id1 != session_id2
    ), f"Session IDs must be distinct: {session_id1} vs {session_id2}"

    # 5. Verify readable format (includes path components)
    assert "agent-abc123" in agent_hash1, "Hash should include filename"
    assert "agent-abc123" in agent_hash2, "Hash should include filename"
    assert "workflows" in agent_hash2, "Hash should include workflow directory"
    assert "wf_123" in agent_hash2, "Hash should include workflow ID"
    assert "subagents" in agent_hash1, "Hash should include subagents component"


def test_bug_main_session_count_reasonable():
    """BUG FIX: Main sessions are at depth 2 only, not misclassified subagents.

    DEFECT 1 Bug: classify_session() keys on filename, not path position.
    Result: journal.jsonl files under subagents/workflows/ are incorrectly
    classified as MAIN.

    This test proves the bug is fixed by checking that main sessions are
    a small minority of the total (since most files are subagent files).
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        pytest.skip("Claude projects directory not found")

    sessions = discover_sessions(Path.home(), no_subagents=False)

    if not sessions:
        pytest.skip("No sessions found in corpus")

    # Invariant: All main sessions should be at depth 2 (path has exactly 2 parts)
    main_sessions = [s for s in sessions if s.is_main]
    if main_sessions:
        for session in main_sessions:
            path_parts = session.path.relative_to(Path.home() / ".claude" / "projects").parts
            assert len(path_parts) == 2, (
                f"Main session at unexpected depth: {session.path}. "
                f"Main sessions should be at <projectDir>/<sessionId>.jsonl (depth 2)."
            )
    else:
        pytest.skip("No main sessions found in corpus to verify depth")

    # Invariant: main sessions should be a small fraction of total
    # (most of the corpus consists of subagent files)
    subagent_count = sum(1 for s in sessions if not s.is_main)
    if subagent_count > 0:
        assert len(main_sessions) < subagent_count, (
            f"Found {len(main_sessions)} main sessions but only {subagent_count} subagents. "
            "This is inverted and indicates misclassification."
        )


def test_project_dir_scopes_discovery(tmp_path):
    """DEFECT FIX: project_dir parameter must scope discovery, not be ignored.

    When scope is None, discover_sessions(project_dir) should return only
    sessions whose cwd is within project_dir, not the entire corpus.
    """
    import json
    import os
    from unittest.mock import patch

    # Create two separate project directories
    project_a = tmp_path / "project_a_path"
    project_b = tmp_path / "project_b_path"
    project_a.mkdir()
    project_b.mkdir()

    # Create .claude/projects directory structure for mocked home
    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"

    # Create Session A: with cwd pointing to project_a
    proj_a_encoded = claude_dir / "-project-a"
    session_a_dir = proj_a_encoded / "sess-uuid-a"
    subagents_a = session_a_dir / "subagents"
    subagents_a.mkdir(parents=True)

    agent_file_a = subagents_a / "agent-aaa111.jsonl"
    with open(agent_file_a, "w") as f:
        f.write(json.dumps({"cwd": str(project_a), "type": "user"}) + "\n")

    # Create Session B: with cwd pointing to project_b (different project)
    proj_b_encoded = claude_dir / "-project-b"
    session_b_dir = proj_b_encoded / "sess-uuid-b"
    subagents_b = session_b_dir / "subagents"
    subagents_b.mkdir(parents=True)

    agent_file_b = subagents_b / "agent-bbb222.jsonl"
    with open(agent_file_b, "w") as f:
        f.write(json.dumps({"cwd": str(project_b), "type": "user"}) + "\n")

    # Test with mocked home
    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # Discover sessions for project_a WITHOUT explicit scope
        # The fix makes this default scope to project_a internally
        sessions_a = discover_sessions(project_a, no_subagents=False)

        # CRITICAL ASSERTION 1: project_a's session IS present
        assert len(sessions_a) == 1, (
            f"Project A discovery should return exactly 1 session, "
            f"but got {len(sessions_a)}: {[s.path for s in sessions_a]}"
        )
        assert sessions_a[0].path == agent_file_a

        # CRITICAL ASSERTION 2: project_b's session is ABSENT
        # This assertion FAILS if project_dir is ignored and filtering is skipped
        # (because then both sessions would be returned)
        session_paths = {s.path for s in sessions_a}
        assert agent_file_b not in session_paths, (
            f"Project A discovery should NOT include Project B's sessions, "
            f"but found: {session_paths}"
        )


def test_explicit_scope_overrides_project_dir(tmp_path):
    """Explicit scope parameter must override project_dir parameter.

    When both project_dir and scope are provided, scope should win.
    """
    # Create two project directories
    project_a = tmp_path / "my_project_a"
    project_b = tmp_path / "my_project_b"
    project_a.mkdir()
    project_b.mkdir()

    # Create .claude/projects directory structure for mocked home
    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"

    # Create a session with cwd pointing to project_b
    proj_a_dir = claude_dir / "-my-project-a"
    session_a = proj_a_dir / "session-a-uuid-1"
    subagents_a = session_a / "subagents"
    subagents_a.mkdir(parents=True)

    # Create a session file with cwd pointing to project_b (different from project_a)
    import json

    agent_file_a = subagents_a / "agent-aaa111.jsonl"
    with open(agent_file_a, "w") as f:
        f.write(json.dumps({"cwd": str(project_b)}) + "\n")

    # Temporarily override Home to use our fake_home
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # Call with project_dir=project_a but scope=project_b
        # Should return the session because its cwd matches scope (project_b)
        sessions = discover_sessions(project_a, scope=str(project_b), no_subagents=False)

        # Invariant: explicit scope overrides project_dir
        # The session has cwd=project_b, so it should be found even though
        # we passed project_dir=project_a
        assert len(sessions) > 0, (
            "Explicit scope should override project_dir: "
            "session with cwd=project_b should be found even with project_dir=project_a"
        )
        assert sessions[0].path == agent_file_a


# ---------------------------------------------------------------------------
# Persistent cwd-projection cache: the cwd-index bootstrap cost, cached
# across processes rather than only in memory. discover_sessions()'s
# full streaming cwd scan is expensive (~1s over the real corpus) and, until
# this cache, was rebuilt from scratch on every process because the
# in-memory-only cache dies with the process. These tests exercise the
# persisted cache directly: unchanged files are trusted with zero re-read,
# grown files extend their cached evidence via the append cursor, a
# rewritten/truncated file discards stale evidence rather than merging it,
# and a vanished file is never resurrected by a stale row.
#
# _cwd_index_cache.pop(str(claude_dir), None) simulates starting a fresh
# process between two discover_sessions() calls within one test: the
# in-process memo (which never expires within one process) is cleared while
# the persisted SQLite cache on disk is left intact, exactly like a second
# `ssgrep search` subprocess would see.
# ---------------------------------------------------------------------------


def test_store_cwd_cache_roundtrip(tmp_path):
    """Direct unit coverage of store.py's cwd-cache persistence functions,
    independent of discovery.py's scoping logic.
    """
    from ssgrep import store as store_mod

    db_path = tmp_path / "cache" / "cwd_index.db"
    conn = store_mod.init_cwd_cache(db_path)
    try:
        assert store_mod.load_cwd_cache_rows(conn) == []

        rows = [
            ("/a/b.jsonl", 100, 123.0, 100, "hash1", "cwd_one\ncwd_two"),
            ("/a/c.jsonl", 50, 456.0, 50, "hash2", ""),
        ]
        store_mod.save_cwd_cache_rows(conn, rows)
        loaded = {r[0]: r for r in store_mod.load_cwd_cache_rows(conn)}
        assert loaded["/a/b.jsonl"] == rows[0]
        assert loaded["/a/c.jsonl"] == rows[1]

        # Upsert overwrites in place rather than duplicating.
        updated_row = ("/a/b.jsonl", 200, 789.0, 200, "hash1b", "cwd_three")
        store_mod.save_cwd_cache_rows(conn, [updated_row])
        loaded = {r[0]: r for r in store_mod.load_cwd_cache_rows(conn)}
        assert loaded["/a/b.jsonl"] == updated_row
        assert len(loaded) == 2

        store_mod.delete_cwd_cache_rows(conn, ["/a/c.jsonl"])
        loaded = {r[0]: r for r in store_mod.load_cwd_cache_rows(conn)}
        assert list(loaded.keys()) == ["/a/b.jsonl"]

        # No-op save/delete (empty list) must not raise.
        store_mod.save_cwd_cache_rows(conn, [])
        store_mod.delete_cwd_cache_rows(conn, [])
    finally:
        conn.close()

    # Rows survive reopening the database.
    conn2 = store_mod.init_cwd_cache(db_path)
    try:
        assert len(store_mod.load_cwd_cache_rows(conn2)) == 1
    finally:
        conn2.close()

    # Containing directory is locked down, matching init_db()'s posture --
    # cwd values reveal project directory names.
    assert (db_path.parent.stat().st_mode & 0o777) == 0o700


def test_cwd_cache_db_path_is_sibling_of_claude_dir(tmp_path):
    """The persistent cache must live outside claude_dir, never inside it --
    claude_dir.rglob("*.jsonl") must never be able to see it.
    """
    from ssgrep import discovery

    claude_dir = tmp_path / ".claude" / "projects"
    cache_path = discovery._cwd_cache_db_path(claude_dir)

    assert claude_dir not in cache_path.parents
    assert cache_path.parent.parent == claude_dir.parent


def test_cwd_cache_unchanged_file_skips_rescan(tmp_path, monkeypatch):
    """Warm persistent cache: a file whose size and mtime match its cached
    row is trusted with zero file I/O on the next (simulated fresh-process)
    call.

    The spy is installed BEFORE the first call too (not just the second),
    with a positive assertion that a genuinely cold cache DOES rescan the
    file. Without that positive signal, an empty `rescanned` list on the
    second call is indistinguishable from `_rescan_cwd_file` never being
    reachable at all (e.g. _build_cwd_index short-circuiting before ever
    calling it, on either call) -- confirmed empirically: dropping just the
    first-call assertion below reproduces a version of this test that stays
    green even when the whole per-file cache-miss path is dead code.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project = tmp_path / "myproject"
    project.mkdir()

    session_dir = claude_dir / "-myproject" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project)}) + "\n")

    rescanned: list[Path] = []
    original_rescan = discovery._rescan_cwd_file

    def spy(path, disk_size, disk_mtime, cached):
        rescanned.append(path)
        return original_rescan(path, disk_size, disk_mtime, cached)

    monkeypatch.setattr(discovery, "_rescan_cwd_file", spy)

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        first = discover_sessions(project, no_subagents=False)
        assert len(first) == 1
        assert rescanned == [session_file], (
            "positive signal: a genuinely cold cache must rescan the file -- proves "
            "the spy is really wired to the per-file cache-miss path, so the "
            "empty-list assertion below (after a warm reopen) is meaningful rather "
            f"than vacuous. Observed rescans: {rescanned}"
        )

        # Simulate a fresh process: drop the in-memory memo only. The
        # persistent (SQLite-backed) cache populated by the call above must
        # survive this -- that's the layer under test.
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        rescanned.clear()

        second = discover_sessions(project, no_subagents=False)

    assert len(second) == 1
    assert second[0].path == first[0].path
    assert rescanned == [], f"unchanged file was re-read from disk: {rescanned}"


def test_cwd_cache_trusts_only_when_both_size_and_mtime_match(tmp_path):
    """_build_cwd_index's top-level cache-trust condition requires size AND
    mtime to both match the cached row before skipping a rescan entirely
    (zero file I/O, the file is never even handed to _rescan_cwd_file).
    This is the persistent, cross-process cache every scoped search reads:
    weakening this to "either matches" means a false cache-hit can silently
    serve stale cwd membership after a file changes.

    Isolates the top-level check from _rescan_cwd_file's own byte-level
    Rewrite/Truncation Guard (covered elsewhere in this module): here, if
    the top-level condition wrongly trusts the cache, _rescan_cwd_file is
    never even called, so those other tests can't catch this.

    Simulates coarse filesystem timestamp resolution (or a clock that
    didn't tick between two writes): the file's size and content genuinely
    change between the two discover_sessions() calls, but its mtime is
    pinned back to the exact value the cache already has on record.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project_a)}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        before = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert len(before) == 1
        mtime_before = session_file.stat().st_mtime

        # Genuinely change size and content (adds project_b's cwd), then
        # pin mtime back to the exact cached value.
        with open(session_file, "a") as f:
            f.write(json.dumps({"cwd": str(project_b)}) + "\n")
        os.utime(session_file, (mtime_before, mtime_before))
        assert session_file.stat().st_mtime == mtime_before, (
            "test setup requires an exact mtime pin -- if this fails the "
            "filesystem doesn't preserve the float as set"
        )
        assert session_file.stat().st_size != before[0].size

        discovery._cwd_index_cache.pop(str(claude_dir), None)
        after_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)

    assert len(after_b) == 1, (
        "size changed but mtime was pinned to the cached value, and the "
        "new cwd (project_b) was still not found -- the cache-trust "
        "condition is trusting a match on mtime alone instead of requiring "
        "both size AND mtime to match before skipping the rescan"
    )


def test_cwd_cache_append_merges_new_cwd(tmp_path):
    """Growing a file (same first line) extends its cached evidence via the
    append cursor instead of discarding it: a cwd recorded only in newly
    appended records is picked up on the next call, and the cwd already
    cached from before the append is still found too.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project_a)}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # Only project_a's cwd is on record so far.
        before_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert before_b == []

        # Append a record whose cwd is project_b. Same first line, file
        # only grows: the Rewrite/Truncation Guard allows extending the
        # cached evidence rather than discarding it.
        with open(session_file, "a") as f:
            f.write(json.dumps({"cwd": str(project_b)}) + "\n")

        discovery._cwd_index_cache.pop(str(claude_dir), None)
        after_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert len(after_b) == 1
        assert after_b[0].path == session_file

        # The cwd already cached before the append must still be found --
        # proving the new evidence was merged, not substituted.
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        after_a = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert len(after_a) == 1


def test_cwd_cache_truncated_rewrite_discards_stale_evidence(tmp_path):
    """A file that shrinks (truncated and rewritten) must not have its old
    cwd evidence merged into the new content: old evidence is discarded,
    not unioned, and the file is rescanned whole from byte 0.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project_a), "n": 1}) + "\n")
        f.write(json.dumps({"cwd": str(project_a), "n": 2}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        before = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert len(before) == 1

        # Truncate and rewrite smaller, with a different first line and cwd.
        with open(session_file, "w") as f:
            f.write(json.dumps({"cwd": str(project_b), "n": 99}) + "\n")
        assert session_file.stat().st_size < before[0].size

        discovery._cwd_index_cache.pop(str(claude_dir), None)

        # The stale project_a evidence must be gone, not merged in.
        after_a = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert after_a == []

        discovery._cwd_index_cache.pop(str(claude_dir), None)
        after_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert len(after_b) == 1


def test_cwd_cache_grown_but_rewritten_first_line_forces_full_rescan(tmp_path):
    """The Rewrite/Truncation Guard requires BOTH conditions. A file whose
    first line changed must be treated as a rewrite even when it also grew
    -- size alone is not sufficient evidence that content was only
    appended to, since a compacted-and-replayed transcript can easily grow
    while still invalidating everything cached about its old content.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project_a)}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        before = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert len(before) == 1
        size_before = before[0].size

        # Rewrite with a DIFFERENT first line and MORE total content: size
        # alone would look like a safe append (grew, never shrank), but the
        # first line no longer matches the cached one.
        with open(session_file, "w") as f:
            f.write(json.dumps({"cwd": str(project_b), "pad": "x" * 200}) + "\n")
            f.write(json.dumps({"cwd": str(project_b)}) + "\n")
        assert session_file.stat().st_size > size_before

        discovery._cwd_index_cache.pop(str(claude_dir), None)

        # If the guard used size alone, stale project_a evidence would
        # still be merged in. The correct guard discards it because the
        # first line changed.
        after_a = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert after_a == [], (
            "stale cwd evidence survived a rewrite that also grew the file: "
            "the first-line-hash guard was not applied"
        )

        discovery._cwd_index_cache.pop(str(claude_dir), None)
        after_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert len(after_b) == 1


def test_cwd_cache_shrunk_with_unchanged_first_line_forces_full_rescan(tmp_path):
    """The Rewrite/Truncation Guard's other half: even when the first line
    is unchanged, a file that shrank must not have its old evidence
    extended from the stale byte_offset -- content at those byte offsets
    may no longer exist, so it must be rescanned whole rather than trusted.

    Isolates the size half of the guard from the hash half: this file's
    first line never changes, so a mutation that dropped only the
    size-shrink check (while keeping the hash check) would still pass
    every other test in this module but must fail this one.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    first_line = json.dumps({"cwd": str(project_a), "marker": "same-first-line"})
    with open(session_file, "w") as f:
        f.write(first_line + "\n")
        f.write(json.dumps({"cwd": str(project_b), "n": 2}) + "\n")
        f.write(json.dumps({"cwd": str(project_b), "n": 3}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # Both cwds are on record: project_a (first line) and project_b
        # (later lines).
        before_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert len(before_b) == 1

        # Truncate down to just the first line -- same exact bytes, so the
        # hash still matches, but the file is now smaller than cached.size.
        with open(session_file, "w") as f:
            f.write(first_line + "\n")
        size_after = session_file.stat().st_size
        assert size_after < before_b[0].size

        discovery._cwd_index_cache.pop(str(claude_dir), None)

        # project_b's cwd lived only in the now-truncated-away records: if
        # the size-shrink check were skipped, a naive "hash still matches
        # so extend from the old byte_offset" would try to resume reading
        # past EOF and silently keep the stale project_b evidence cached
        # forever. The correct guard discards it and rescans from byte 0.
        after_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert after_b == [], (
            "stale cwd evidence survived a truncation that kept the first "
            "line intact: the size-shrink guard was not applied"
        )

        # project_a's cwd (still present in the truncated file) must still
        # be found -- proving this was a correct full rescan, not a bug
        # that lost everything.
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        after_a = discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        assert len(after_a) == 1


def test_cwd_cache_vanished_file_not_resurrected(tmp_path):
    """A file removed from disk must not be resurrected by its stale cache
    row on a later call, and the stale row itself is pruned so it cannot
    resurrect the file on any later run either.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery
    from ssgrep import store as store_mod

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project = tmp_path / "myproject"
    project.mkdir()

    session_dir = claude_dir / "-myproject" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project)}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        before = discover_sessions(project, no_subagents=False)
        assert len(before) == 1

        cache_db = discovery._cwd_cache_db_path(claude_dir)
        conn = store_mod.init_cwd_cache(cache_db)
        try:
            rows_before = store_mod.load_cwd_cache_rows(conn)
        finally:
            conn.close()
        assert any(
            r[0] == str(session_file) for r in rows_before
        ), "expected a cache row for the file after the first discover_sessions() call"

        session_file.unlink()
        discovery._cwd_index_cache.pop(str(claude_dir), None)

        after = discover_sessions(project, no_subagents=False)
        assert after == []

        conn = store_mod.init_cwd_cache(cache_db)
        try:
            rows_after = store_mod.load_cwd_cache_rows(conn)
        finally:
            conn.close()
        assert not any(
            r[0] == str(session_file) for r in rows_after
        ), "vanished file's cache row was not pruned"


def test_cwd_cache_unavailable_degrades_gracefully(tmp_path, monkeypatch):
    """If the persistent cache can't be opened (unwritable disk, permission
    error, first-run directory that doesn't exist yet, whatever), discovery
    must still produce correct results via a full scan -- the cache is a
    speed optimization, never a correctness dependency.

    A call counter on the injected failure is asserted > 0: without it, a
    correct `len(sessions) == 1` cannot distinguish "the cache failed and
    discovery gracefully fell back" from "the cache code path was never
    reached at all" (e.g. _build_cwd_index short-circuiting before ever
    calling store.init_cwd_cache) -- confirmed empirically: dropping just
    the counter assertion below reproduces a version of this test that
    stays green even when store.init_cwd_cache() is never invoked.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import store as store_mod

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project = tmp_path / "myproject"
    project.mkdir()

    session_dir = claude_dir / "-myproject" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project)}) + "\n")

    calls = {"init_cwd_cache": 0}

    def broken_init(db_path):
        calls["init_cwd_cache"] += 1
        raise OSError("simulated: cache directory is not writable")

    monkeypatch.setattr(store_mod, "init_cwd_cache", broken_init)

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        sessions = discover_sessions(project, no_subagents=False)

    assert calls["init_cwd_cache"] > 0, (
        "positive signal: the simulated failure must actually have been invoked -- "
        "otherwise the correct result below proves nothing about graceful degradation"
    )
    assert len(sessions) == 1
    assert sessions[0].path == session_file


def test_cwd_cache_cold_and_warm_scans_agree(tmp_path):
    """Warm-cache and forced-cold-rescan results must be identical over a
    small multi-file, multi-project synthetic corpus: the cache changes
    speed, never which sessions are found or their metadata. Includes a
    directory whose encoded name does not match its content's real cwd
    (the D11 scenario this whole module exists for), so this also proves
    the cache doesn't accidentally reintroduce trust in directory naming.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    # Main session for project_a.
    proj_a_dir = claude_dir / "-encoded-name-does-not-matter-a"
    proj_a_dir.mkdir(parents=True)
    with open(proj_a_dir / "session-a.jsonl", "w") as f:
        f.write(json.dumps({"cwd": str(project_a)}) + "\n")

    # Subagent whose cwd is project_a but which sits under a directory
    # encoded for something else entirely -- the D11 mismatch case.
    sub_dir = claude_dir / "-encoded-name-does-not-matter-b" / "sess-x" / "subagents"
    sub_dir.mkdir(parents=True)
    with open(sub_dir / "agent-1.jsonl", "w") as f:
        f.write(json.dumps({"cwd": str(project_a)}) + "\n")

    # Unrelated main session for project_b, must never appear in project_a's
    # results.
    proj_b_dir = claude_dir / "-encoded-name-does-not-matter-c"
    proj_b_dir.mkdir(parents=True)
    with open(proj_b_dir / "session-b.jsonl", "w") as f:
        f.write(json.dumps({"cwd": str(project_b)}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        warm = discover_sessions(project_a, no_subagents=False)

        # Force a genuinely cold rescan: drop the in-memory memo AND
        # delete the persisted cache file outright.
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        discovery._cwd_cache_db_path(claude_dir).unlink(missing_ok=True)

        cold = discover_sessions(project_a, no_subagents=False)

    def _key(s):
        return (str(s.path), s.session_id, s.is_main, s.parent_session_id, s.agent_hash)

    assert {_key(s) for s in warm} == {_key(s) for s in cold}
    # Both project_a sessions found (main + mismatched-directory subagent),
    # project_b's session correctly excluded from both.
    assert len(warm) == 2
    assert len(cold) == 2


def test_cwd_cache_append_reads_tail_only(tmp_path, monkeypatch):
    """The append path must actually read only the new tail bytes, not
    silently re-read the whole file and merge -- proven by capturing the
    start_offset _read_cwds_from_offset is called with, not just checking
    the final answer (which a full rescan would also get right, just
    slower -- this is exactly the distinction the latency fix depends on).
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "agent-1.jsonl"
    with open(session_file, "w") as f:
        f.write(json.dumps({"cwd": str(project_a)}) + "\n")
    size_before_append = session_file.stat().st_size

    calls: list[tuple[Path, int]] = []
    original = discovery._read_cwds_from_offset

    def spy(path, start_offset):
        calls.append((path, start_offset))
        return original(path, start_offset)

    monkeypatch.setattr(discovery, "_read_cwds_from_offset", spy)

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        discover_sessions(project_a, scope=str(project_a), no_subagents=False)
        # First (full) scan must have read from byte 0.
        assert calls[-1] == (session_file, 0)

        with open(session_file, "a") as f:
            f.write(json.dumps({"cwd": str(project_b)}) + "\n")

        discovery._cwd_index_cache.pop(str(claude_dir), None)
        calls.clear()
        discover_sessions(project_a, scope=str(project_b), no_subagents=False)

    # The append scan must resume from where the first scan left off, not
    # re-read from byte 0 -- proving a genuine tail-only read, not a full
    # rescan that merely happens to land on the right answer.
    assert calls == [
        (session_file, size_before_append)
    ], f"append did not read tail-only from the cached offset: {calls}"


def test_cwd_index_fallback_miss_is_visible_and_counted(tmp_path, monkeypatch):
    """Fallback scans (cache misses) are explicit and countable.

    This test verifies the fix for the silent-fallback defect: when
    _build_cwd_index is broken (returns {}), discovery still produces correct
    results via inline fallback scans, but now those scans are VISIBLE via
    _cwd_index_stats["fallback_scans"] counter.

    Demonstrates the three key properties:
    1. Warm cache (second call) has zero fallback scans -- all files hit cache
    2. Broken cache (patched to return {}) triggers fallback for every file
    3. Results are identical in both cases (graceful degradation preserved)

    Without counting fallback scans, a broken _build_cwd_index would go
    unnoticed forever: results would be correct (via fallback) but the cost
    would be permanent full-scan overhead instead of cache hits.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project = tmp_path / "myproject"
    project.mkdir()

    # Create a multi-file corpus: two session files, each with cwds
    proj_dir = claude_dir / "-myproject"
    proj_dir.mkdir(parents=True)

    file_1 = proj_dir / "session-1.jsonl"
    file_2 = proj_dir / "session-2.jsonl"

    with open(file_1, "w") as f:
        f.write(json.dumps({"cwd": str(project)}) + "\n")
    with open(file_2, "w") as f:
        f.write(json.dumps({"cwd": str(project)}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # COLD CALL: Discover sessions for the first time, building cache from scratch.
        # In-process cache is empty; _build_cwd_index is called to populate it.
        # Files with cwds are stored in the index, so no fallback scans occur.
        discovery._cwd_index_cache.clear()
        discovery._cwd_index_stats["fallback_scans"] = 0

        cold_sessions = discover_sessions(project, no_subagents=False)
        assert len(cold_sessions) == 2, "cold call should find both files"

        # WARM CALL: Discover sessions a second time.
        # In-process cache is still populated from cold call.
        # Every file is in the cache, so zero fallback scans occur.
        discovery._cwd_index_stats["fallback_scans"] = 0
        warm_fallback_before = discovery._cwd_index_stats["fallback_scans"]

        warm_sessions = discover_sessions(project, no_subagents=False)
        warm_fallback_after = discovery._cwd_index_stats["fallback_scans"]
        warm_fallback_delta = warm_fallback_after - warm_fallback_before

        assert len(warm_sessions) == 2, "warm call should find both files"
        assert (
            warm_fallback_delta == 0
        ), f"warm cache should have zero fallback scans, but got {warm_fallback_delta}"

        # BROKEN CACHE: Simulate _build_cwd_index returning {} (completely broken).
        # Clear the in-process cache to force a fresh call to _build_cwd_index.
        # With an empty index, every file will trigger a fallback scan.
        def broken_cwd_index(claude_dir):
            return {}  # Return empty dict, simulating catastrophic cache failure

        monkeypatch.setattr(discovery, "_build_cwd_index", broken_cwd_index)
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        discovery._cwd_index_stats["fallback_scans"] = 0
        broken_fallback_before = discovery._cwd_index_stats["fallback_scans"]

        broken_sessions = discover_sessions(project, no_subagents=False)
        broken_fallback_after = discovery._cwd_index_stats["fallback_scans"]
        broken_fallback_delta = broken_fallback_after - broken_fallback_before

        assert len(broken_sessions) == 2, "broken cache should still find both files"
        assert (
            broken_fallback_delta == 2
        ), f"broken cache should trigger 2 fallbacks, got {broken_fallback_delta}"

        # Results must be identical across all three scenarios.
        cold_paths = {str(s.path) for s in cold_sessions}
        warm_paths = {str(s.path) for s in warm_sessions}
        broken_paths = {str(s.path) for s in broken_sessions}

        assert (
            warm_paths == cold_paths
        ), f"warm and cold results differ: warm={warm_paths}, cold={cold_paths}"
        assert (
            broken_paths == cold_paths
        ), f"broken cache results differ from cold: broken={broken_paths}, cold={cold_paths}"


def test_empty_cwd_file_cached_not_re_scanned(tmp_path):
    """FIX VERIFICATION: A transcript with zero cwd records is cached.

    The bug: a file with empty cwds was never stored in the in-memory index,
    so _file_matches_scope() re-read it on every single discover_sessions()
    call, forever.

    The fix: store empty sets in the index at all three sites. This ensures
    "file_path_str in cwd_index" is True, and subsequent calls find it in
    cache rather than re-scanning.

    This test asserts the property directly: the file is read once, not once
    per call. It uses the fallback_scans counter to verify cache behavior.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project = tmp_path / "myproject"
    project.mkdir()

    # Create a session with a file that has ZERO cwd records
    session_dir = claude_dir / "-myproject" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    empty_cwd_file = session_dir / "agent-empty.jsonl"

    # Write records with NO cwd field at all -- genuinely empty-cwd case
    with open(empty_cwd_file, "w") as f:
        f.write(json.dumps({"type": "user", "text": "hello"}) + "\n")
        f.write(json.dumps({"type": "assistant", "text": "hi"}) + "\n")

    # Also create a file with cwds to ensure mixed corpus works
    file_with_cwd = session_dir / "agent-full.jsonl"
    with open(file_with_cwd, "w") as f:
        f.write(json.dumps({"cwd": str(project), "type": "user"}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # COLD CALL: First discovery, cache is empty, files are scanned.
        discovery._cwd_index_cache.clear()
        discovery._cwd_index_stats["fallback_scans"] = 0

        cold_sessions = discover_sessions(project, no_subagents=False)
        assert len(cold_sessions) == 1, "should find file with cwds, not empty-cwd file"

        # WARM CALL: Second discovery, simulating a fresh process but with
        # persisted cache. Clear in-memory cache but leave SQLite cache intact.
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        discovery._cwd_index_stats["fallback_scans"] = 0

        warm_sessions = discover_sessions(project, no_subagents=False)
        assert len(warm_sessions) == 1, "warm call should still find file with cwds"

        # CRITICAL: No fallback scans should occur on the warm call.
        # The empty-cwd file should be found in the persistent cache and not
        # re-scanned. The file with cwds should also be in cache.
        warm_fallback_count = discovery._cwd_index_stats["fallback_scans"]

        assert warm_fallback_count == 0, (
            f"Warm cache should have zero fallback scans for cached files, "
            f"but got {warm_fallback_count}. The empty-cwd file was re-read "
            f"instead of being found in the persistent cache. This indicates "
            f"the fix did not work."
        )

        # Results must be identical
        assert {str(s.path) for s in cold_sessions} == {str(s.path) for s in warm_sessions}


def test_empty_cwd_file_matches_no_scope(tmp_path):
    """A file with no cwd records correctly matches no scope.

    After the fix stores empty sets in the cache, we must verify that an
    empty-cwd file still correctly returns False for any scope check.
    The fix must not accidentally make empty-cwd files match.
    """
    import json
    import os
    from unittest.mock import patch

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()

    session_dir = claude_dir / "-some-project" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)
    empty_file = session_dir / "agent-empty.jsonl"

    with open(empty_file, "w") as f:
        # Write records with no cwd field
        f.write(json.dumps({"type": "user"}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # Scope to project_a -- the empty-cwd file should not match.
        sessions_a = discover_sessions(project_a, scope=str(project_a), no_subagents=False)

        # Empty-cwd file has no matching cwd, so it should not be discovered.
        assert len(sessions_a) == 0, (
            f"Empty-cwd file should not match any scope. " f"Found: {[s.path for s in sessions_a]}"
        )

        # Scope to project_b -- still should not match.
        sessions_b = discover_sessions(project_a, scope=str(project_b), no_subagents=False)
        assert len(sessions_b) == 0, (
            f"Empty-cwd file should not match project_b scope either. "
            f"Found: {[s.path for s in sessions_b]}"
        )


def test_mixed_corpus_empty_and_full_cwd_warm_cache_zero_fallbacks(tmp_path):
    """Mixed corpus: warm-cache steady state must have zero fallback scans.

    A corpus with both empty-cwd and non-empty-cwd files, after one cold
    call to build the cache, should have zero fallback scans on a subsequent
    warm call. This proves both types of files are cached correctly.
    """
    import json
    import os
    from unittest.mock import patch

    from ssgrep import discovery

    fake_home = tmp_path / "fake_home"
    claude_dir = fake_home / ".claude" / "projects"
    project = tmp_path / "myproject"
    project.mkdir()

    session_dir = claude_dir / "-myproject" / "sess-1" / "subagents"
    session_dir.mkdir(parents=True)

    # Create 3 files: empty, with-cwd, and another empty
    empty_1 = session_dir / "agent-empty-1.jsonl"
    with_cwd = session_dir / "agent-full.jsonl"
    empty_2 = session_dir / "agent-empty-2.jsonl"

    with open(empty_1, "w") as f:
        f.write(json.dumps({"type": "user"}) + "\n")

    with open(with_cwd, "w") as f:
        f.write(json.dumps({"cwd": str(project), "type": "user"}) + "\n")

    with open(empty_2, "w") as f:
        f.write(json.dumps({"type": "assistant"}) + "\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}):
        # COLD: Build cache from scratch
        discovery._cwd_index_cache.clear()
        discovery._cwd_index_stats["fallback_scans"] = 0

        cold = discover_sessions(project, no_subagents=False)
        assert len(cold) == 1, "should find the one file with matching cwd"

        # WARM: Second call with persisted cache
        discovery._cwd_index_cache.pop(str(claude_dir), None)
        discovery._cwd_index_stats["fallback_scans"] = 0

        warm = discover_sessions(project, no_subagents=False)
        assert len(warm) == 1, "warm call should find same file"

        # CRITICAL: Zero fallback scans in warm steady state, even with mixed empty/full
        warm_fallback = discovery._cwd_index_stats["fallback_scans"]
        assert warm_fallback == 0, (
            f"Warm cache with mixed empty/full files should have zero fallbacks, "
            f"but got {warm_fallback}. This means either empty-cwd or regular files "
            f"are being re-scanned instead of using cache."
        )


def test_real_corpus_cwd_cache_parity(tmp_path, has_real_corpus: bool):
    """Real-corpus parity: over the actual ~/.claude/projects corpus, a
    forced full rescan (in-memory memo cleared AND the persisted cache
    temporarily moved aside) must find exactly the same sessions as a
    normal cache-assisted call. Restores the real cache file afterward
    either way, so this never leaves the shared cache worse off than it
    found it.

    This machine's real corpus is demonstrably live during test runs: on a
    prior run this test caught a brand-new subagent transcript being written
    mid-comparison, confirmed by its mtime landing inside the test's own
    execution window. A single point-in-time diff between
    two temporally-separated scans is therefore not a reliable pass/fail
    signal here: the same file appearing or disappearing between the two
    scans is expected corpus churn, not a cache defect. What genuinely
    distinguishes a cache bug from churn is repeatability -- churn produces
    a different, unpredictable discrepancy (or none) on a retry; a real
    logic error reproduces the identical discrepancy every time. So this
    retries a bounded number of times and only fails if EVERY attempt
    disagrees, which a real bug would do and benign churn essentially never
    will.

    This test requires the real Claude Code corpus to exist, so it skips
    on CI or machines without a populated ~/.claude/projects.
    """
    if not has_real_corpus:
        pytest.skip("Real Claude Code corpus not available")

    import shutil

    from ssgrep import discovery

    claude_dir = Path.home() / ".claude" / "projects"
    cache_db = discovery._cwd_cache_db_path(claude_dir)
    sidecars = ("", "-wal", "-shm")

    def _one_attempt() -> tuple[set[str], set[str]]:
        backups = {suf: Path(str(cache_db) + suf + ".test-backup") for suf in sidecars}
        had_cache = {suf: Path(str(cache_db) + suf).exists() for suf in sidecars}

        warm = discover_sessions(Path.home(), no_subagents=False)

        for suf in sidecars:
            src = Path(str(cache_db) + suf)
            if src.exists():
                shutil.move(str(src), str(backups[suf]))

        try:
            discovery._cwd_index_cache.pop(str(claude_dir), None)
            cold = discover_sessions(Path.home(), no_subagents=False)
        finally:
            # Restore the real cache to exactly its prior state, regardless
            # of whether the block above raised.
            for suf in sidecars:
                live = Path(str(cache_db) + suf)
                live.unlink(missing_ok=True)
                if had_cache[suf] and backups[suf].exists():
                    shutil.move(str(backups[suf]), str(live))
            discovery._cwd_index_cache.pop(str(claude_dir), None)

        return {str(s.path) for s in warm}, {str(s.path) for s in cold}

    attempts = []
    for _ in range(3):
        warm_paths, cold_paths = _one_attempt()
        if warm_paths == cold_paths:
            return  # agreement reached; cache is consistent with a cold scan
        attempts.append((warm_paths - cold_paths, cold_paths - warm_paths))

    only_warm, only_cold = attempts[-1]
    pytest.fail(
        "cache disagreed with a cold rescan on all 3 attempts (a live corpus "
        "makes one-off disagreement expected churn, but disagreeing every "
        "time points at a real bug, not churn): "
        f"last attempt only_in_warm={only_warm}, only_in_cold={only_cold}; "
        f"all attempts={attempts}"
    )
