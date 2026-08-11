"""Tests for result rendering."""

import json

from ssgrep.render import (
    render_episode_detail,
    render_result_card,
    render_search_response,
)
from ssgrep.types import EpisodeDetail, ResultCard, SearchResponse


def test_render_card_text():
    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="Hello world",
        files_touched=("a.py",),
        is_subagent=False,
    )
    output = render_result_card(card, "text")
    assert "Test" in output
    assert "Hello world" in output


def test_render_card_json():
    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="Hello",
        files_touched=(),
        is_subagent=False,
    )
    output = render_result_card(card, "json")
    assert "ep-1" in output


def test_render_empty_response():
    response = SearchResponse(results=[], index_exists=True)
    output = render_search_response(response)
    assert "No matches" in output


def test_render_no_index():
    response = SearchResponse(results=[], index_exists=False)
    output = render_search_response(response)
    assert "Run" in output


# ============================================================================
# Staleness signal must reach rendered output, not just the dataclass.
#
# Regression tests: search.py computes SearchResponse.stale/stale_count, but
# render_search_response() silently discarded them -- a user searching a
# 3-hour-old index got confidently wrong results with no indication the
# index was behind. These assert the rendered string, not hasattr/dataclass
# field presence, since the defect was specifically that a populated field
# never reached the string a human or the JSON consumer actually sees.
# ============================================================================


def test_render_search_response_stale_warning_shown():
    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="Hello world",
        files_touched=(),
        is_subagent=False,
    )
    response = SearchResponse(results=[card], stale=True, stale_count=38)
    output = render_search_response(response)
    assert "stale" in output.lower()
    assert "38" in output
    assert "ssgrep index" in output


def test_render_search_response_no_stale_warning_when_fresh():
    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="Hello world",
        files_touched=(),
        is_subagent=False,
    )
    response = SearchResponse(results=[card], stale=False, stale_count=0)
    output = render_search_response(response)
    assert "stale" not in output.lower()


def test_render_search_response_stale_warning_with_no_results():
    """Staleness must be surfaced even on a 'no matches' response.

    Otherwise a user reasonably concludes "never solved before" when the
    real explanation is "the index hasn't seen the new content yet".
    """
    response = SearchResponse(results=[], index_exists=True, stale=True, stale_count=5)
    output = render_search_response(response)
    assert "stale" in output.lower()
    assert "5" in output
    assert "No matches found." in output


def test_render_search_response_stale_warning_precedes_results():
    """Must not be buried below the results, where it would be missed."""
    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="Hello world",
        files_touched=(),
        is_subagent=False,
    )
    response = SearchResponse(results=[card], stale=True, stale_count=2)
    output = render_search_response(response)
    assert output.lower().index("stale") < output.index("[ep-1]")


def test_render_search_response_json_includes_stale_fields():
    """JSON output is what the MCP server and scripts consume."""
    card = ResultCard(
        ref="ep-1",
        title="Test",
        timestamp=None,
        score=0.9,
        excerpt="Hello world",
        files_touched=(),
        is_subagent=False,
    )
    response = SearchResponse(results=[card], stale=True, stale_count=7)
    data = json.loads(render_search_response(response, format="json"))
    assert data["stale"] is True
    assert data["stale_count"] == 7


# ============================================================================
# render_episode_detail: same staleness signal, same defect shape.
#
# show() is "the only command that returns untruncated transcript content"
# (EpisodeDetail docstring) -- a user drilling into full episode detail from
# a stale index is at least as liable to be misled as one reading search
# results, so it carries the same stale/stale_count fields and must render
# them the same way.
# ============================================================================


def _make_episode_detail(**overrides: object) -> EpisodeDetail:
    defaults: dict[str, object] = dict(
        episode_id="s-1:ep:0",
        session_id="s-1",
        title="Test episode",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text="prompt text",
        response_text="response text",
        files_touched=(),
        tool_names=(),
        is_subagent=False,
    )
    defaults.update(overrides)
    return EpisodeDetail(**defaults)  # type: ignore[arg-type]


def test_render_episode_detail_stale_warning_shown():
    detail = _make_episode_detail(stale=True, stale_count=12)
    output = render_episode_detail(detail)
    assert "stale" in output.lower()
    assert "12" in output
    assert "ssgrep index" in output


def test_render_episode_detail_no_stale_warning_when_fresh():
    detail = _make_episode_detail(stale=False, stale_count=0)
    output = render_episode_detail(detail)
    assert "stale" not in output.lower()


def test_render_episode_detail_json_includes_stale_fields():
    detail = _make_episode_detail(stale=True, stale_count=4)
    data = json.loads(render_episode_detail(detail, format="json"))
    assert data["stale"] is True
    assert data["stale_count"] == 4


# ============================================================================
# Terminal-escape hardening: transcript-derived fields are attacker-influenced
# input. Text mode must escape control bytes (an embedded \x1b[2J could clear
# the screen or overwrite the genuine output); JSON must carry original bytes
# verbatim for machine consumers.
# ============================================================================


def test_render_card_text_escapes_terminal_control_sequences():
    evil = "before \x1b[2J after"
    card = ResultCard(
        ref="ep-1",
        title="Title \x1b[31mred",
        timestamp=None,
        score=0.9,
        excerpt=evil,
        files_touched=("a\x1b[2J.py",),
        is_subagent=True,
        agent_name="agent\x1b[0m",
        agent_description="desc\x07bell",
    )
    output = render_result_card(card, "text")
    assert "\x1b" not in output, "raw ESC bytes must never reach text output"
    assert "\x07" not in output
    # Escaped (visible) forms survive, so the reader sees something odd happened
    assert "\\x1b[2J" in output
    assert "before" in output and "after" in output


def test_render_card_json_keeps_control_sequences_verbatim():
    evil = "before \x1b[2J after"
    card = ResultCard(
        ref="ep-1",
        title="Title",
        timestamp=None,
        score=0.9,
        excerpt=evil,
        files_touched=(),
        is_subagent=False,
    )
    output = render_result_card(card, "json")
    doc = json.loads(output)
    assert doc["excerpt"] == evil, "JSON consumers must get the original bytes"


def test_render_episode_detail_text_escapes_but_keeps_newlines():
    detail = EpisodeDetail(
        episode_id="ep-1",
        session_id="s-1",
        title="Title\x1b[2J",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text="line one\nline two\x1b[2J",
        response_text="resp\ttabbed\x1b]0;owned\x07",
        files_touched=(),
        tool_names=(),
        is_subagent=False,
    )
    output = render_episode_detail(detail, "text")
    assert "\x1b" not in output
    assert "\x07" not in output
    assert "line one\nline two" in output, "genuine newlines in body text must survive"
    assert "resp\ttabbed" in output, "genuine tabs in body text must survive"
    # JSON stays verbatim
    doc = json.loads(render_episode_detail(detail, "json"))
    assert doc["prompt_text"] == "line one\nline two\x1b[2J"
    assert doc["response_text"] == "resp\ttabbed\x1b]0;owned\x07"


def test_render_card_text_marks_deleted_source():
    """Tombstoned (source-deleted) results are labeled in text output."""
    card = ResultCard(
        ref="ep-1",
        title="Archived work",
        timestamp=None,
        score=0.5,
        excerpt="old excerpt",
        files_touched=(),
        is_subagent=False,
        source_absent=True,
    )
    output = render_result_card(card, "text")
    assert "[source deleted]" in output.splitlines()[0]


def test_render_card_text_no_deleted_marker_for_live_source():
    card = ResultCard(
        ref="ep-1",
        title="Live work",
        timestamp=None,
        score=0.5,
        excerpt="excerpt",
        files_touched=(),
        is_subagent=False,
        source_absent=False,
    )
    output = render_result_card(card, "text")
    assert "[source deleted]" not in output


# ============================================================================
# Render-layer truncation pairing: text rendering AND JSON serialization
# must both handle the detail layer's truncation markers correctly.
#
# The detail layer (_apply_bound in detail.py) adds TRUNCATION_MARKER when
# text exceeds bounds and sets truncated flags. render_episode_detail() must:
# (1) include the marker in text output; (2) preserve the full bounded text
# in JSON; (3) carry the truncated flags through JSON serialization.
# ============================================================================


def test_render_episode_detail_text_includes_truncation_marker():
    """Text rendering must include the truncation marker when response_truncated=True."""
    from ssgrep.detail import TRUNCATION_MARKER

    long_response = "x" * 160000  # Exceeds MAX_RESPONSE_CHARS (150k)
    marker_text = TRUNCATION_MARKER.format(max_chars=150000)

    detail = EpisodeDetail(
        episode_id="s-1:ep:0",
        session_id="s-1",
        title="Truncated response",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text="What is the GIL?",
        response_text=long_response[:150000] + marker_text,  # As detail.py would prepare it
        files_touched=(),
        tool_names=(),
        is_subagent=False,
        response_truncated=True,  # Flag that response was truncated
    )

    output = render_episode_detail(detail, format="text")

    # Text output must contain the truncation marker
    assert "[... truncated" in output, "Text output must show truncation marker"
    assert marker_text in output, "Text output must contain the exact marker"
    # And the content before the marker
    assert "x" * 100 in output, "Text output must include truncated response content"


def test_render_episode_detail_json_preserves_truncated_flag_and_bounded_text():
    """JSON output must preserve response_truncated flag and full bounded text."""
    from ssgrep.cli.commands import to_jsonable
    from ssgrep.detail import TRUNCATION_MARKER

    long_response = "y" * 160000
    marker_text = TRUNCATION_MARKER.format(max_chars=150000)
    bounded_text = long_response[:150000] + marker_text

    detail = EpisodeDetail(
        episode_id="s-2:ep:0",
        session_id="s-2",
        title="JSON truncation test",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text="Short prompt",
        response_text=bounded_text,
        files_touched=(),
        tool_names=(),
        is_subagent=False,
        response_truncated=True,
    )

    jsonable = to_jsonable(detail)

    # JSON must carry the response_truncated flag as a boolean
    assert jsonable["response_truncated"] is True, "JSON must preserve response_truncated flag"
    # JSON must carry the full bounded text (including marker)
    assert jsonable["response_text"] == bounded_text, "JSON must preserve full bounded text"
    assert "[... truncated" in jsonable["response_text"], "JSON text must include marker"


def test_render_episode_detail_untruncated_has_no_marker():
    """Untruncated content must not have a truncation marker."""
    detail = EpisodeDetail(
        episode_id="s-3:ep:0",
        session_id="s-3",
        title="Untruncated",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text="Short prompt",
        response_text="Short response, well under 150k chars",
        files_touched=(),
        tool_names=(),
        is_subagent=False,
        response_truncated=False,
    )

    output = render_episode_detail(detail, format="text")

    # Text output must NOT contain truncation marker for untruncated content
    assert "[... truncated" not in output, "Untruncated content must not have marker in text"
    assert "Short response" in output, "Response content must be present"

    # JSON must show response_truncated=False
    from ssgrep.cli.commands import to_jsonable

    jsonable = to_jsonable(detail)
    assert jsonable["response_truncated"] is False, "JSON must show response_truncated=False"


def test_render_episode_detail_prompt_truncation_separate_from_response():
    """Prompt and response truncation are independent flags and markers."""
    from ssgrep.detail import MAX_PROMPT_CHARS, TRUNCATION_MARKER

    long_prompt = "a" * 60000  # Exceeds MAX_PROMPT_CHARS (50k)
    long_response = "b" * 10000  # Well within MAX_RESPONSE_CHARS

    prompt_marker = TRUNCATION_MARKER.format(max_chars=MAX_PROMPT_CHARS)
    bounded_prompt = long_prompt[:MAX_PROMPT_CHARS] + prompt_marker

    detail = EpisodeDetail(
        episode_id="s-4:ep:0",
        session_id="s-4",
        title="Separate truncation",
        timestamp=None,
        git_branch=None,
        cwd=None,
        prompt_text=bounded_prompt,
        response_text=long_response,
        files_touched=(),
        tool_names=(),
        is_subagent=False,
        prompt_truncated=True,
        response_truncated=False,
    )

    output = render_episode_detail(detail, format="text")

    # Both sections must be present
    assert "--- Prompt ---" in output
    assert "--- Response ---" in output
    # Only prompt should have marker
    assert prompt_marker in output
    assert "bbb" in output and "aaa" in output  # Both contents present
