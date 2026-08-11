"""Tests verifying the type/API contract defect fixes.

Each test verifies one of the five high/medium severity defects was fixed.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.types import (
    Chunk,
    ContentType,
    EmptyQueryError,
    IndexNotFoundError,
    IndexNotReadyError,
    IndexStats,
    Record,
    SearchException,
    SearchResponse,
    SessionFile,
)
from tests.conftest import (
    build_chunk,
    build_episode,
    build_episode_detail,
    build_file_cursor,
    build_index_stats,
    build_result_card,
    build_search_filters,
    build_search_response,
    build_session_file,
)


def test_session_file_docstring_clarifies_subagent_identity() -> None:
    """SessionFile.session_id docstring must state it holds derived identity.

    Defect: The docstring was ambiguous about whether session_id holds the
    derived unique identity or the raw parent sessionId. Fix: added explicit
    statement that for subagents, session_id holds the derived unique value
    and parent_session_id holds the raw parent id.
    """
    # Verify docstring exists and mentions "derived"
    assert SessionFile.__doc__ is not None
    assert "derived unique" in SessionFile.__doc__
    assert "parent_session_id" in SessionFile.__doc__
    assert "MUST hold" in SessionFile.__doc__

    # Verify we can construct subagent SessionFile with distinct identities
    subagent1 = SessionFile(
        path=Path("/tmp/agent1.jsonl"),
        session_id="derived-unique-id-1",  # The derived identity
        is_main=False,
        size=1024,
        mtime=1700000000.0,
        parent_session_id="parent-session-id",  # Raw parent id
        agent_hash="agent-hash-1",
    )
    subagent2 = SessionFile(
        path=Path("/tmp/agent2.jsonl"),
        session_id="derived-unique-id-2",  # Different derived identity
        is_main=False,
        size=1024,
        mtime=1700000000.0,
        parent_session_id="parent-session-id",  # Same parent, different sibling
        agent_hash="agent-hash-2",
    )
    # The defect was that both would have identical session_id; now they differ
    assert subagent1.session_id != subagent2.session_id
    assert subagent1.parent_session_id == subagent2.parent_session_id


def test_search_response_wrapper_exists_with_degenerate_outcome_fields() -> None:
    """search() return type must be SearchResponse, not bare list[ResultCard].

    Defect: search() returned list[ResultCard] with no way to represent
    omitted count or index state. Fix: added SearchResponse wrapper with
    omitted_count and index_exists fields.
    """
    # Verify SearchResponse exists and has required fields
    response = SearchResponse(
        results=[],
        omitted_count=5,
        index_exists=True,
    )
    assert response.omitted_count == 5
    assert response.index_exists is True
    assert response.results == []

    # Verify it can represent no-index case
    no_index_response = SearchResponse(
        results=[],
        omitted_count=0,
        index_exists=False,
    )
    assert no_index_response.index_exists is False


def test_exception_types_for_degenerate_search_outcomes() -> None:
    """search() must raise distinct exceptions for degenerate outcomes.

    Defect: No exception types were defined for empty query, missing index,
    etc. Fix: added EmptyQueryError, IndexNotFoundError, IndexNotReadyError
    as SearchException subclasses.
    """
    # Verify exception types exist and are usable
    assert issubclass(EmptyQueryError, SearchException)
    assert issubclass(IndexNotFoundError, SearchException)
    assert issubclass(IndexNotReadyError, SearchException)

    # Verify they can be instantiated and caught
    try:
        raise EmptyQueryError("Query was empty")
    except EmptyQueryError as e:
        assert "empty" in str(e).lower()

    try:
        raise IndexNotFoundError("No index for this project")
    except SearchException as e:
        # Verify they're catchable as SearchException
        assert isinstance(e, IndexNotFoundError)


def test_index_stats_has_index_exists_field() -> None:
    """IndexStats must have index_exists field to distinguish no-index case.

    Defect: IndexStats had no field to signal whether an index exists,
    making it impossible to distinguish "no index" from "found nothing".
    Fix: added index_exists: bool = True field.
    """
    # Verify field exists on IndexStats
    stats_with_index = IndexStats(
        session_count=5,
        episode_count=20,
        chunk_count=150,
        index_size_bytes=1000000,
        last_index_time=datetime.now(UTC),
        model_id="test-model",
        vector_dimension=256,
        skipped_records=0,
        malformed_records=0,
        schema_version=1,
        index_exists=True,
    )
    assert stats_with_index.index_exists is True

    # Verify we can create stats for missing index
    stats_no_index = IndexStats(
        session_count=0,
        episode_count=0,
        chunk_count=0,
        index_size_bytes=0,
        last_index_time=None,
        model_id="test-model",
        vector_dimension=256,
        skipped_records=0,
        malformed_records=0,
        schema_version=1,
        index_exists=False,
    )
    assert stats_no_index.index_exists is False

    # Verify they're distinguishable
    assert stats_with_index.index_exists != stats_no_index.index_exists


def test_chunk_byte_offset_has_documentation() -> None:
    """Chunk.byte_offset must have documented meaning or be marked reserved.

    Defect: byte_offset field had no docstring and conflicted with the
    decision not to expose transcript offsets. Fix: added docstring explaining it's
    reserved for future use and not a transcript offset.
    """
    # Verify Chunk docstring mentions byte_offset
    assert Chunk.__doc__ is not None
    assert "byte_offset" in Chunk.__doc__
    assert "reserved" in Chunk.__doc__ or "legacy" in Chunk.__doc__.lower()
    assert "transcript file offset" in Chunk.__doc__

    # Verify the field can be constructed and used
    chunk = Chunk(
        chunk_id="chunk-001",
        episode_id="ep-001",
        session_id="session-001",
        text="Test text",
        content_type=ContentType.PROMPT,
        byte_offset=0,  # Currently expected to be ignored/reserved
    )
    # Default is 0, future use only
    assert chunk.byte_offset == 0


def test_fixture_files_renamed_to_match_sibling_convention(fixtures_dir) -> None:  # noqa: F811
    """Subagent fixture must use shared basename for agent-<hash> pair.

    Defect: subagent-session.jsonl had sibling meta.json with different basename,
    violating the spec's sibling-association rule. Fix: renamed to
    agent-aexample-agent-deadbeef01.jsonl to match agent-aexample-agent-deadbeef01.meta.json.
    """
    # Verify the renamed file exists
    transcript_path = fixtures_dir / "agent-aexample-agent-deadbeef01.jsonl"
    meta_path = fixtures_dir / "agent-aexample-agent-deadbeef01.meta.json"

    assert transcript_path.exists(), "Renamed transcript file should exist"
    assert meta_path.exists(), "Meta sidecar should exist"

    # Verify old name doesn't exist
    old_transcript_path = fixtures_dir / "subagent-session.jsonl"
    assert not old_transcript_path.exists(), "Old filename should not exist"

    # Verify they share the same base name (before extension)
    expected_base = "agent-aexample-agent-deadbeef01"
    assert transcript_path.stem == expected_base
    # For the .meta.json file, check it starts with the base name
    assert meta_path.name.startswith(expected_base)


def test_search_response_has_total_matches_field() -> None:
    """SearchResponse must report total_matches for complete result accounting.

    Defect (high severity): MCP-server spec line 144-147 requires 'Every
    response SHALL report the total number of matching episodes'. Design.md
    line 171 corroborates: 'an explicit total-match count so the model knows
    to refine rather than assuming it saw everything.' SearchResponse had no
    way to represent total matches, only omitted_count (dropped results).

    Fix: added total_matches: int field to SearchResponse.
    """
    # Verify the field exists and defaults to 0
    response_empty = SearchResponse(results=[])
    assert response_empty.total_matches == 0

    # Verify it can represent non-zero total matches
    response_with_matches = SearchResponse(
        results=[],  # But report 100 total matches (all dropped by budget)
        total_matches=100,
        omitted_count=100,
    )
    assert response_with_matches.total_matches == 100
    assert response_with_matches.omitted_count == 100

    # Verify it can represent partial results (some returned, some dropped)
    response_partial = SearchResponse(
        results=[],  # Empty results list for simplicity
        total_matches=50,
        omitted_count=30,  # 30 dropped, implying 20 were returned
    )
    assert response_partial.total_matches == 50
    assert response_partial.omitted_count == 30
    # The caller can compute: returned_count = total_matches - omitted_count
    assert response_partial.total_matches - response_partial.omitted_count == 20


def test_search_response_has_excerpts_truncated_flag() -> None:
    """SearchResponse must flag excerpt-only truncation distinct from dropped results.

    Defect (high severity): Session-search spec line 100-101 and mcp-server
    spec line 132 require 'the response explicitly indicates that excerpt
    truncation occurred'. Excerpt truncation (shortening text within a card)
    is distinct from omitted_count (dropping entire results). A response with
    omitted_count=0 and excerpts_truncated=True means "all results fit, but
    we shortened their text"; the two signals cannot be conflated.

    Fix: added excerpts_truncated: bool field to SearchResponse.
    """
    # Verify the field exists and defaults to False
    response_no_trunc = SearchResponse(results=[], omitted_count=0)
    assert response_no_trunc.excerpts_truncated is False

    # Verify it can signal excerpts were shortened
    response_excerpts_truncated = SearchResponse(
        results=[],  # Results omitted for test simplicity
        omitted_count=0,  # No results dropped
        excerpts_truncated=True,  # But excerpts were shortened
    )
    assert response_excerpts_truncated.excerpts_truncated is True
    assert response_excerpts_truncated.omitted_count == 0

    # Verify it can signal both truncation types
    response_both = SearchResponse(
        results=[],
        omitted_count=5,  # 5 results dropped entirely
        excerpts_truncated=True,  # And remaining ones had text shortened
    )
    assert response_both.excerpts_truncated is True
    assert response_both.omitted_count == 5


def test_search_response_has_clamped_flag() -> None:
    """SearchResponse must flag when requested limit was clamped to hard maximum.

    Defect (medium severity): Session-search spec line 202 and line 213-216
    require 'A requested count exceeding the hard maximum SHALL be clamped to
    that maximum, with the clamping stated explicitly in the response rather
    than applied silently.' The response type must distinguish "caller got
    what they asked for" from "we clamped their request."

    Fix: added clamped: bool field to SearchResponse.
    """
    # Verify the field exists and defaults to False
    response_no_clamp = SearchResponse(results=[])
    assert response_no_clamp.clamped is False

    # Verify it can signal clamping occurred
    response_clamped = SearchResponse(
        results=[],  # Results omitted for test simplicity
        clamped=True,  # Caller requested 100, we clamped to 8
    )
    assert response_clamped.clamped is True

    # Verify clamped flag is independent of other flags
    response_all_flags = SearchResponse(
        results=[],
        total_matches=100,
        omitted_count=50,
        excerpts_truncated=True,
        clamped=True,
    )
    assert response_all_flags.total_matches == 100
    assert response_all_flags.omitted_count == 50
    assert response_all_flags.excerpts_truncated is True
    assert response_all_flags.clamped is True


def test_record_type_documents_record_interchange_format() -> None:
    """Record type must define the interchange format for parsed transcript lines.

    Defect (critical severity): records.py (2.2) outputs classified records
    consumed by episodes.py (2.3), signal.py (2.4), and metadata.py (2.5),
    all built in parallel against the contract alone. Without a defined Record
    type, these three modules cannot know what shape they receive: raw dict,
    (dict, classification) pair, or dataclass. This forces either interface
    mismatch at wave-3 wiring or reimplementation of the classification logic.

    Fix: added Record TypedDict in types.py documenting that records are
    dicts from JSONL parsing with 'type' and optional 'subtype' fields.
    """
    # Verify Record type exists and is importable
    assert Record is not None

    # Verify Record docstring documents the interchange format
    assert Record.__doc__ is not None
    assert "parsed record" in Record.__doc__ or "JSONL" in Record.__doc__
    assert "type" in Record.__doc__
    assert "records.py" in Record.__doc__

    # Verify the type can be used to annotate record dicts
    record: Record = {
        "type": "user",
        "text": "Hello",
    }
    assert record["type"] == "user"

    # Verify system records can carry subtype
    system_record: Record = {
        "type": "system",
        "subtype": "away_summary",
        "recap": "Session recap text",
    }
    assert system_record["type"] == "system"
    assert system_record["subtype"] == "away_summary"


def test_search_response_has_index_empty_flag() -> None:
    """SearchResponse must distinguish empty index from no-match query.

    Defect (high severity): session-search spec requires 'the system returns
    a clear response distinguishing an empty index from a query that matched
    nothing'. SearchResponse had no way to express this: empty results with
    index_exists=True could mean either case. The spec explicitly says 'None
    of these conditions SHALL ... return an empty-success response
    indistinguishable from a normal no-matches result.'

    Fix: added index_empty: bool field to SearchResponse to distinguish
    SearchResponse(results=[], index_exists=True, index_empty=True) [empty
    index case] from SearchResponse(results=[], index_exists=True,
    index_empty=False) [query matched nothing case].
    """
    # Verify the field exists and defaults to False
    response_has_chunks = SearchResponse(results=[])
    assert response_has_chunks.index_empty is False

    # Verify it can signal index is empty
    response_empty_index = SearchResponse(
        results=[],  # No results
        index_exists=True,  # Index exists
        index_empty=True,  # But it has zero chunks
    )
    assert response_empty_index.index_empty is True
    assert response_empty_index.index_exists is True

    # Verify they're distinguishable
    response_no_match = SearchResponse(
        results=[],  # No results
        index_exists=True,  # Index exists
        index_empty=False,  # Index has chunks (but query matched none)
    )
    assert response_empty_index.index_empty != response_no_match.index_empty

    # Verify docstring documents the distinction
    assert SearchResponse.__doc__ is not None
    assert "index_empty" in SearchResponse.__doc__
    assert "empty index" in SearchResponse.__doc__.lower()


def test_show_docstring_documents_ref_scope() -> None:
    """show() docstring must document ref grammar and scope (episode-scoped).

    Defect (medium severity): CLI spec says show retrieves 'episode or session'
    but return type EpisodeDetail is single-episode. No docstring says what
    this means or what refs identify. Wave-3 show implementer and wave-4 CLI
    adapter cannot resolve this from the contract alone.

    Fix: updated show() docstring to document that refs identify episodes,
    that 'session' terminology refers to subagent-transcript case (one episode
    spanning a full subagent session), and that show() always returns exactly
    one episode.
    """
    from ssgrep.api import show

    # Verify docstring exists and addresses the ambiguity
    assert show.__doc__ is not None
    doc = show.__doc__

    # Verify it documents that refs are episode-scoped
    assert "episode" in doc.lower()
    assert "ref" in doc.lower()

    # Verify it mentions the subagent session case
    assert "subagent" in doc.lower()

    # Verify it clarifies the return type
    assert "EpisodeDetail" in doc or "single episode" in doc.lower()


def test_conftest_covers_all_contract_dataclasses() -> None:
    """conftest.py must define builders and fixtures for every contract dataclass.

    Defect (medium severity): the contract-helpers requirement is that they
    'import cleanly and cover every dataclass in the contract.' The
    SearchResponse fixture was missing, violating that, which pushes every
    test touching SearchResponse into inventing ad-hoc test doubles.

    Fix: added build_search_response() and sample_search_response fixture to
    conftest.py following the existing pattern.
    """
    import inspect

    from ssgrep import types
    from tests import conftest

    # Get all dataclasses from types module
    dataclass_names = set()
    for name, obj in inspect.getmembers(types):
        # Check if it's a dataclass (has __dataclass_fields__)
        if hasattr(obj, "__dataclass_fields__"):
            dataclass_names.add(name)

    # Get all builder functions from conftest
    builder_names = set()
    for name, obj in inspect.getmembers(conftest):
        if name.startswith("build_") and callable(obj):
            # Extract type name from builder name
            type_name = "".join(word.capitalize() for word in name[6:].split("_"))
            builder_names.add(type_name)

    # Verify every dataclass has a builder
    missing_builders = dataclass_names - builder_names
    assert not missing_builders, f"Missing builders for: {missing_builders}"

    # Verify SearchResponse specifically has builders
    assert "SearchResponse" in builder_names, "SearchResponse builder is missing"

    # Verify sample fixtures exist
    sample_fixture_names = {name for name in dir(conftest) if name.startswith("sample_")}

    # Check for sample_search_response fixture
    assert (
        "sample_search_response" in sample_fixture_names
    ), "sample_search_response fixture is missing"


def test_builder_overrides_apply_to_all_builders() -> None:
    """All build_* helpers must apply **overrides via dataclasses.replace().

    Defect (high severity): All 9 builders accept **overrides in their signature
    but silently discard it, making the parameter useless. This creates a silent
    failure mode: build_session_file(totally_bogus_kwarg='xyz') returns a default
    object with no error, leading parallel agents to chase phantom bugs.

    Fix: Apply overrides via dataclasses.replace(obj, **overrides) after
    construction. This raises TypeError on unknown field names, providing
    immediate feedback on typos.
    """
    # Test 1: Valid overrides take effect
    # SessionFile override
    sf = build_session_file(session_id="custom-session")
    assert sf.session_id == "custom-session"

    # Episode override
    ep = build_episode(episode_id="ep-custom", is_subagent=True)
    assert ep.episode_id == "ep-custom"
    assert ep.is_subagent is True

    # Chunk override
    chunk = build_chunk(text="Custom text")
    assert chunk.text == "Custom text"

    # FileCursor override
    fc = build_file_cursor(byte_offset=42)
    assert fc.byte_offset == 42

    # SearchFilters override
    sf_filt = build_search_filters(branch="feature-x")
    assert sf_filt.branch == "feature-x"

    # ResultCard override
    rc = build_result_card(score=0.99)
    assert rc.score == 0.99

    # SearchResponse override
    sr = build_search_response(total_matches=100)
    assert sr.total_matches == 100

    # EpisodeDetail override
    ed = build_episode_detail(agent_name="Agent Zero")
    assert ed.agent_name == "Agent Zero"

    # IndexStats override
    idx = build_index_stats(chunk_count=999)
    assert idx.chunk_count == 999

    # Test 2: Invalid kwargs raise TypeError
    # SessionFile invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_session_file(totally_bogus_kwarg="xyz")

    # Episode invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_episode(nonexistent_field=True)

    # Chunk invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_chunk(invalid_column=123)

    # FileCursor invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_file_cursor(bad_field="oops")

    # SearchFilters invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_search_filters(typo_field=None)

    # ResultCard invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_result_card(wrong_name="test")

    # SearchResponse invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_search_response(bogus_param=True)

    # EpisodeDetail invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_episode_detail(fake_field="data")

    # IndexStats invalid field
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_index_stats(missing_field=1)
