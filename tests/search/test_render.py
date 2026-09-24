"""Tests for human-readable search rendering.

Every test renders through a recorded ``Console`` so the rich output can be
inspected as deterministic plain text, regardless of TTY or color support.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from rich.console import Console

from ssgrep.search.render import (
    render_episode_detail,
    render_result_card,
    render_search_response,
)
from ssgrep.utilities.types import SearchResponse


def test_render_result_card_renders_all_metadata_and_escapes_fields(sample_card) -> None:
    card = replace(
        sample_card,
        title="unsafe\x1b title",
        score=0.1256,
        project="proj\nforged",
        source_path="/tmp/evil\x00path",
        files_touched=("a.py", "b\n.py"),
        is_subagent=True,
        agent_name=None,
        agent_description="task\x1b" + "x" * 100,
        excerpt="excerpt\n" + "z" * 250,
        source_absent=True,
    )

    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()

    # Ref and title in header
    assert "[session-1:ep:0]" in rendered
    assert "unsafe\\x1b title" in rendered
    assert "[source deleted]" in rendered

    # Score rounded to 3 decimal places
    assert "0.126" in rendered

    # Escaped control bytes in all transcript fields
    assert "proj\\nforged" in rendered
    # Source path is shortened to trailing components, control bytes still escaped
    assert "evil\\x00path" in rendered
    assert "b\\n.py" in rendered

    # Agent metadata (agent_name is None, falls back to "unknown")
    assert "unknown" in rendered
    assert "task\\x1b" in rendered

    # Task display-bound: agent_description is "task\x1b" (4 raw bytes → 8
    # escaped) + 100 x's, sliced to 100 before escaping → max continuous x's is 96.
    # Ensure the truncation boundary is enforced.
    assert "x" * 97 not in rendered

    # Content is pretty-printed: real newlines preserved (not escaped to literal
    # \\n), so the excerpt reads as multi-line text. The full 250-z string wraps
    # across panel lines; verify a substantial chunk.
    assert "excerpt" in rendered
    assert "excerpt\\n" not in rendered
    assert "z" * 100 in rendered


def test_render_result_card_omits_empty_optional_fields(sample_card) -> None:
    card = replace(
        sample_card,
        score=0.0,
        project=None,
        source_path=None,
        files_touched=(),
        excerpt="",
        is_subagent=False,
    )

    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()

    # Header still shows ref + title
    assert "[session-1:ep:0]" in rendered
    assert "Testing" in rendered

    # No optional-field labels
    assert "Score:" not in rendered
    assert "Project:" not in rendered
    assert "Source:" not in rendered
    assert "Files:" not in rendered
    assert "Agent:" not in rendered
    assert "Task:" not in rendered
    assert "Runtime:" not in rendered
    assert "Branch:" not in rendered
    assert "Model:" not in rendered
    assert "Timestamp:" not in rendered


def test_render_result_card_shows_git_branch(sample_card) -> None:
    """Cover _tags_line git_branch branch."""
    card = replace(sample_card, git_branch="feature-x")
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Branch:" in rendered
    assert "feature-x" in rendered


def test_render_result_card_shows_agent_model(sample_card) -> None:
    """Cover _tags_line agent_model branch."""
    card = replace(sample_card, agent_model="claude-sonnet-4-20250514")
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Model:" in rendered
    assert "claude-sonnet-4" in rendered


def test_render_result_card_omits_tags_when_none_hit(sample_card) -> None:
    """Cover _tags_line return None when no tag fields are present."""
    card = replace(
        sample_card,
        project=None,
        timestamp=None,
        git_branch=None,
        agent_model=None,
    )
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Project:" not in rendered
    assert "Branch:" not in rendered
    assert "Model:" not in rendered


def test_render_result_card_caps_files_at_four(sample_card) -> None:
    """Cover _files_line overflow marker."""
    card = replace(
        sample_card,
        files_touched=("a.py", "b.py", "c.py", "d.py", "e.py", "f.py"),
    )
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "(+2 more)" in rendered


def test_render_result_card_shows_high_score_badge(sample_card) -> None:
    """Cover _score_badge score >= 0.8 branch."""
    card = replace(sample_card, score=0.95)
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Score: 0.950" in rendered


def test_render_result_card_shows_medium_score_badge(sample_card) -> None:
    """Cover _score_badge score >= 0.6 branch."""
    card = replace(sample_card, score=0.62)
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Score: 0.620" in rendered


def test_render_result_card_shows_low_score_badge(sample_card) -> None:
    """Cover _score_badge score < 0.6 branch."""
    card = replace(sample_card, score=0.12)
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Score: 0.120" in rendered


def test_render_search_response_summary_shows_plain_count(sample_card) -> None:
    """Cover _summary_line else branch (total_matches == shown)."""
    response = SearchResponse(results=[sample_card], total_matches=1)
    console = Console(record=True, width=120)
    console.print(render_search_response(response, query="test"))
    rendered = console.export_text()
    assert "1 results" in rendered
    assert "1 of 1" not in rendered


def test_render_result_card_shows_non_claude_runtime(sample_card) -> None:
    card = replace(sample_card, runtime="prime-agent")

    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "prime-agent" in rendered

    # Default sample_card has runtime="claude" — no runtime badge shown
    console2 = Console(record=True, width=120)
    console2.print(render_result_card(sample_card))
    assert "Runtime:" not in console2.export_text()


def test_render_search_response_empty_states() -> None:
    def _text(response: SearchResponse) -> str:
        console = Console(record=True, width=120)
        console.print(render_search_response(response))
        return console.export_text()

    assert "Index is empty." in _text(SearchResponse(results=[], index_empty=True))
    assert "3 matches found; none fit the result limit/token budget." in _text(
        SearchResponse(results=[], total_matches=3)
    )
    assert "No matches found." in _text(SearchResponse(results=[]))


def test_render_search_response_summary_line_shows_count_and_query(sample_card) -> None:
    response = SearchResponse(results=[sample_card], total_matches=8, omitted_count=7)

    console = Console(record=True, width=120)
    console.print(render_search_response(response, query="async errors"))
    rendered = console.export_text()

    assert "1 of 8 results" in rendered
    assert "async errors" in rendered


def test_render_result_card_highlights_fenced_code_and_keeps_prose(sample_card) -> None:
    card = replace(
        sample_card,
        excerpt=(
            "Before:\n\n"
            "```python\n"
            'import dspy\nlm = dspy.LM("openrouter/deepseek-v4-flash")\n'
            "dspy.configure(lm=lm)\n"
            "```\n\n"
            "After.\n"
        ),
    )

    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()

    # Prose is preserved verbatim
    assert "Before:" in rendered
    assert "After." in rendered
    # Code body is present and the fence markers are stripped
    assert "import dspy" in rendered
    assert "dspy.configure(lm=lm)" in rendered
    assert "```" not in rendered


def test_render_result_card_tolerates_unbalanced_fence(sample_card) -> None:
    card = replace(sample_card, excerpt="```python\ndef f():\n    return 1\n")

    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()

    assert "def f():" in rendered
    assert "return 1" in rendered


def test_render_search_response_cards_and_footer_markers(sample_card) -> None:
    response = SearchResponse(
        results=[sample_card],
        total_matches=3,
        omitted_count=2,
        excerpts_truncated=True,
        clamped=True,
    )

    console = Console(record=True, width=120)
    console.print(render_search_response(response))
    rendered = console.export_text()

    # Card header contains the ref
    assert "[session-1:ep:0]" in rendered

    # Footer markers
    assert "--- 2 results omitted (token budget) ---" in rendered
    assert "--- Results clamped to maximum ---" in rendered


def test_render_episode_detail_main_session_preserves_body_layout(sample_detail) -> None:
    detail = replace(
        sample_detail,
        title="Testing\x1b",
        prompt_text="line one\n\tline two\x00",
        response_text="answer\nnext",
        files_touched=(),
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    assert "Testing\\x1b" in rendered
    assert "Agent:" not in rendered
    assert "Files:" not in rendered
    assert "line one" in rendered
    assert "line two\\x00" in rendered
    # Response body lines survive as separate rendered lines inside the panel
    assert "answer" in rendered
    assert "next" in rendered
    assert "--- Prompt ---" in rendered
    assert "--- Response ---" in rendered


def test_render_episode_detail_subagent_with_metadata(sample_detail) -> None:
    detail = replace(
        sample_detail,
        is_subagent=True,
        agent_name=None,
        agent_description="inspect\nthings",
        files_touched=("a.py", "unsafe\x1b.py"),
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    assert "Agent: unknown" in rendered
    assert "Task: inspect\\nthings" in rendered
    assert "Files: a.py, unsafe\\x1b.py" in rendered


def test_render_episode_detail_placeholders_for_empty_bodies(sample_detail) -> None:
    detail = replace(
        sample_detail,
        prompt_text="",
        response_text="",
        tool_names=("Read", "Grep"),
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    assert "(no prompt)" in rendered
    assert "(no response \u2014 tool calls: Read, Grep)" in rendered


def test_render_episode_detail_placeholder_caps_tool_names(sample_detail) -> None:
    detail = replace(
        sample_detail,
        prompt_text="",
        response_text="",
        tool_names=("Read", "Grep", "Bash", "Edit", "Write", "Task", "Glob"),
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    assert "tool calls: Read, Grep, Bash, Edit, Write" in rendered
    assert "+2 more)" in rendered
    assert "Task" not in rendered
    assert "Glob" not in rendered


def test_render_episode_detail_placeholder_without_tools(sample_detail) -> None:
    detail = replace(
        sample_detail,
        prompt_text="",
        response_text="",
        tool_names=(),
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    assert "(no response)" in rendered
    assert "tool calls" not in rendered


def test_render_episode_detail_highlights_fenced_code(sample_detail) -> None:
    detail = replace(
        sample_detail,
        prompt_text="Fix:\n```python\nx = 1\n```",
        response_text="Plain prose\nsecond line",
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    # Prose outside fences keeps its line layout
    assert "Fix:" in rendered
    assert "Plain prose" in rendered
    assert "second line" in rendered
    # Fenced code blocks render their content
    assert "x = 1" in rendered


def test_render_episode_detail_renders_markdown_tables(sample_detail) -> None:
    detail = replace(
        sample_detail,
        response_text="| Finding | Evidence |\n|---|---|\n| A | B |",
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    # Table header and cells render as aligned columns, not raw pipe syntax
    assert "Finding" in rendered
    assert "Evidence" in rendered
    assert "A" in rendered
    assert "B" in rendered
    assert "| A | B |" not in rendered


def test_render_episode_detail_empty_bodies_render_labels(sample_detail) -> None:
    detail = replace(
        sample_detail,
        prompt_text="",
        response_text="",
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    # Section labels remain even when a body is empty
    assert "--- Prompt ---" in rendered
    assert "--- Response ---" in rendered


def test_render_episode_detail_truncation_markers(sample_detail) -> None:
    detail = replace(
        sample_detail,
        prompt_truncated=True,
        response_truncated=True,
    )

    console = Console(record=True, width=120)
    console.print(render_episode_detail(detail))
    rendered = console.export_text()

    assert "[prompt truncated]" in rendered
    assert "[response truncated]" in rendered


def test_content_block_returns_none_when_only_fence_lines(sample_card) -> None:
    """Cover _content_block line 202: _split_fenced yields no segments.

    When the excerpt contains only fence markers with no content between them,
    _split_fenced produces zero segments, leaving parts empty and hitting
    return None.
    """
    card = replace(sample_card, excerpt="```\n```")
    console = Console(record=True, width=120)
    console.print(render_result_card(card))
    rendered = console.export_text()
    assert "Testing" in rendered
    assert "Use pytest." not in rendered
    assert "────────────────────" not in rendered


def test_content_block_fallback_on_syntax_exception(sample_card) -> None:
    """Cover _content_block lines 197-200: Syntax() exception fallback.

    When Syntax() raises (e.g. from a malformed language tag), the exception
    handler falls back to a plain Text renderable with ReprHighlighter.
    """
    card = replace(sample_card, excerpt="```python\nx = 1\n```")
    with patch("ssgrep.search.render.Syntax", side_effect=RuntimeError("boom")):
        console = Console(record=True, width=120)
        console.print(render_result_card(card))
        rendered = console.export_text()
    assert "x = 1" in rendered
