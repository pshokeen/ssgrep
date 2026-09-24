"""Result formatting for search responses.

Every transcript-derived field is escaped: first through ``textsafe.printable()``
to neutralize control bytes, then through ``rich.markup.escape()`` to neutralize
brackets that rich would interpret as markup tags (markdown-rendered bodies rely
on the markdown parser instead, which never interprets brackets as rich markup).
Only literal labels and formatting in this module may contain rich markup
brackets.
"""

from __future__ import annotations

import re

from rich import box
from rich.console import Group, RenderableType
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from usecli.cli.config.colors import COLOR

from ssgrep.utilities import textsafe
from ssgrep.utilities.types import EpisodeDetail, ResultCard, SearchResponse

TOKEN_BUDGET_DEFAULT = 1500

#: Max touched-file basenames shown on a card before an overflow marker.
_FILES_CAP = 4
#: Max tool names shown in the no-response placeholder before an overflow marker.
_TOOLS_CAP = 5
#: Max characters of a subagent task description shown on a card.
_TASK_CAP = 100
#: Fence tags mapped to richer lexer names rich understands.
_LANG_ALIASES = {
    "py": "python",
    "sh": "shell",
    "shell": "shell",
    "js": "javascript",
    "ts": "typescript",
    "yml": "yaml",
    "md": "markdown",
    "markdown": "markdown",
    "c++": "cpp",
    "golang": "go",
}
#: Matches an opening/closing fenced-code line (``` or ~~~ with optional language).
_FENCE_RE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[ \t]*([\w.+-]*)[ \t]*$")


def _esc(value: str) -> str:
    """Escape transcript-derived text: control bytes first, then rich markup.

    This is a belt-and-suspenders approach: ``textsafe.printable`` replaces
    non-printable characters with their backslash representations, and
    ``rich.markup.escape`` escapes ``[`` and ``]`` so they are not interpreted
    as rich markup tags.
    """
    return escape(textsafe.printable(value))


def _base(path: str) -> str:
    """Return the final path component (basename) of a path string."""
    return path.rstrip("/").split("/")[-1] or path


def _short_path(path: str, parts: int = 2) -> str:
    """Return the trailing ``parts`` path components, elided with an ellipsis."""
    comps = path.rstrip("/").split("/")
    tail = "/".join(comps[-parts:])
    return tail if len(comps) <= parts else f"\u2026/{tail}"


def _tags_line(card: ResultCard) -> Text | None:
    """One compact line of identity tags (project, runtime, branch, model, time).

    Values use basenames and short labels so the strip stays on one or two
    lines and never dominates the card's content.
    """
    pairs: list[tuple[str, str]] = []
    if card.project:
        pairs.append(("Project", _base(card.project)))
    if card.runtime != "claude":
        pairs.append(("Runtime", card.runtime))
    if card.git_branch:
        pairs.append(("Branch", card.git_branch))
    if card.agent_model:
        pairs.append(("Model", _base(card.agent_model)))
    if card.timestamp:
        pairs.append(("When", card.timestamp.strftime("%Y-%m-%d %H:%M")))
    if not pairs:
        return None

    line = Text()
    for i, (label, value) in enumerate(pairs):
        if i:
            line.append("  \u00b7  ", style=COLOR.FOREGROUND_MUTED)
        line.append(f"{label}:", style=f"bold {COLOR.PRIMARY}")
        line.append(f" {_esc(value)}", style=COLOR.SECONDARY)
    return line


def _subagent_line(card: ResultCard) -> Text | None:
    """One compact line for subagent attribution (agent name plus task)."""
    if not card.is_subagent:
        return None
    line = Text()
    line.append("Agent:", style=f"bold {COLOR.PRIMARY}")
    line.append(f" {_esc(card.agent_name or 'unknown')}", style=COLOR.SECONDARY)
    if card.agent_description:
        line.append(" \u00b7 Task: ", style=COLOR.FOREGROUND_MUTED)
        line.append(_esc(card.agent_description[:_TASK_CAP]), style=COLOR.SECONDARY)
    return line


def _files_line(card: ResultCard) -> Text | None:
    """One compact line of touched-file basenames, capped with a +N marker."""
    if not card.files_touched:
        return None
    names = [_base(f) for f in card.files_touched]
    shown, rest = names[:_FILES_CAP], names[_FILES_CAP:]
    joined = ", ".join(shown)
    if rest:
        joined += f" (+{len(rest)} more)"
    line = Text()
    line.append("Files:", style=f"bold {COLOR.PRIMARY}")
    line.append(f" {_esc(joined)}", style=COLOR.SECONDARY)
    return line


def _split_fenced(safe: str) -> list[tuple[str, str | None, str]]:
    """Split sanitized text into ``(kind, lang, text)`` segments.

    Fenced code blocks (````` ``` ```` / ``~~~``) are extracted as ``code``
    segments carrying their optional language tag; everything else is ``prose``.
    Unbalanced fences are tolerated: the trailing run is treated as code rather
    than silently dropped.
    """
    segments: list[tuple[str, str | None, str]] = []
    buffer: list[str] = []
    in_code = False
    lang: str | None = None

    def flush(kind: str, current_lang: str | None) -> None:
        if buffer:
            segments.append((kind, current_lang, "\n".join(buffer)))
            buffer.clear()

    for line in safe.split("\n"):
        match = _FENCE_RE.match(line)
        if match:
            if not in_code:
                flush("prose", None)
                in_code = True
                tag = match.group(1).lower()
                lang = _LANG_ALIASES.get(tag) or tag or None
            else:
                flush("code", lang)
                in_code = False
                lang = None
        else:
            buffer.append(line)
    flush("code" if in_code else "prose", lang)
    return segments


def _highlighted_block(text: str) -> RenderableType | None:
    """Pretty-print transcript body text, highlighting fenced code.

    Control bytes are escaped (so transcript input can never inject ANSI or
    markup), while genuine newlines and tabs survive. Fenced code blocks are
    rendered with true syntax highlighting; prose gets a light
    ``ReprHighlighter`` touch and keeps its exact layout.
    """
    if not text:
        return None
    safe = textsafe.printable(text, keep="\n\t")
    parts: list[RenderableType] = []
    for kind, lang, text in _split_fenced(safe):
        if kind == "prose":
            line = Text(escape(text))
            ReprHighlighter().highlight(line)
            parts.append(line)
        else:
            try:
                parts.append(
                    Syntax(
                        text,
                        lang or "text",
                        theme="ansi_dark",
                        background_color="default",
                        word_wrap=True,
                        line_numbers=False,
                        padding=0,
                    )
                )
            except Exception:
                line = Text(escape(text))
                ReprHighlighter().highlight(line)
                parts.append(line)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else Group(*parts)


def _content_block(card: ResultCard) -> RenderableType | None:
    """The dominant excerpt block, pretty-printed via ``_highlighted_block``."""
    return _highlighted_block(card.excerpt)


def _markdown_hard_breaks(text: str) -> str:
    """Append hard-break spaces to prose lines so markdown keeps transcript layout.

    Rich's markdown parser reflows paragraph soft breaks onto a single line,
    which would destroy the line layout of transcript bodies. Appending two
    trailing spaces to every non-fence line turns each line into a hard break;
    fenced code blocks are left untouched.
    """
    lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence and line.strip():
            line = line.rstrip() + "  "
        lines.append(line)
    return "\n".join(lines)


def _markdown_body(text: str) -> RenderableType | None:
    """Pretty-print a prompt/response body as markdown.

    Control bytes are escaped (so transcript input can never inject ANSI or
    markup), and hard breaks preserve genuine newlines. Tables, lists,
    emphasis, inline code, and fenced code blocks all render as markdown.
    """
    if not text:
        return None
    safe = textsafe.printable(text, keep="\n\t")
    return RichMarkdown(_markdown_hard_breaks(safe), code_theme="ansi_dark")


def _response_placeholder(detail: EpisodeDetail) -> Text:
    if not detail.tool_names:
        return Text("  (no response)", style=f"italic {COLOR.FOREGROUND_MUTED}")
    shown = detail.tool_names[:_TOOLS_CAP]
    extra = len(detail.tool_names) - len(shown)
    suffix = f" \u00b7 +{extra} more" if extra else ""
    return Text(
        f"  (no response \u2014 tool calls: {_esc(', '.join(shown))}{suffix})",
        style=f"italic {COLOR.FOREGROUND_MUTED}",
    )


def _score_badge(score: float) -> Text:
    """A colored relevance badge, greener for higher scores."""
    if score >= 0.8:
        style = f"bold {COLOR.SUCCESS}"
    elif score >= 0.6:
        style = f"bold {COLOR.ACCENT}"
    else:
        style = f"bold {COLOR.ERROR}"
    return Text(f"Score: {score:.3f}", style=style)


def render_result_card(card: ResultCard, *, index: int | None = None) -> Panel:
    """Render one search result as a rich Panel card.

    The panel border carries the ref (with rank) as its title and a colored
    score badge as its right-aligned subtitle. Inside, the episode title leads,
    then compact metadata (basenames, inline tags, capped file list, one dim
    source line), then the dominant pretty-printed content block.
    """
    parts: list[RenderableType] = []

    # --- Episode title line (rendered as proper Markdown heading) ---
    parts.append(RichMarkdown(f"## {textsafe.printable(card.title)}"))
    parts.append(Text())
    if card.source_absent:
        parts.append(Text("  [source deleted]", style=f"bold {COLOR.ERROR}"))

    # --- Compact metadata: tags, subagent, files, dim source ---
    if tags := _tags_line(card):
        parts.append(tags)
    if sub := _subagent_line(card):
        parts.append(sub)
    if files := _files_line(card):
        parts.append(files)
    if card.source_path:
        line = Text()
        line.append("Source:", style=f"bold {COLOR.PRIMARY}")
        line.append(f" {_esc(_short_path(card.source_path))}", style=COLOR.SECONDARY)
        parts.append(line)
    parts.append(Text())

    # --- Content: the dominant, pretty-printed excerpt block ---
    if content := _content_block(card):
        parts.append(Rule(style=COLOR.FOREGROUND_MUTED))
        parts.append(Text())
        parts.append(content)

    # --- Panel border: ref (with rank) as title, score as right-aligned badge ---
    header = Text()
    if index is not None:
        header.append(f"{index:>2}  ", style=COLOR.SECONDARY)
    header.append(f"[{_esc(card.ref)}]", style=f"bold {COLOR.PRIMARY}")

    return Panel(
        Group(*parts),
        box=box.HEAVY,
        padding=(1, 2),
        title=header,
        title_align="left",
        subtitle=_score_badge(card.score) if card.score else None,
        subtitle_align="right",
        border_style=COLOR.BORDER,
        safe_box=True,
    )


def _summary_line(query: str, response: SearchResponse) -> Text:
    """A leading line naming how many results were found for the query."""
    shown = len(response.results)
    line = Text()
    if response.total_matches and response.total_matches != shown:
        line.append(f"{shown} of {response.total_matches} results", style="bold")
    else:
        line.append(f"{shown} results", style="bold")
    line.append(" for ", style=COLOR.FOREGROUND_MUTED)
    line.append(f'"{_esc(query)}"', style=f"italic {COLOR.PRIMARY}")
    return line


def render_search_response(response: SearchResponse, *, query: str | None = None) -> RenderableType:
    """Render the full search response as a Group of Panel cards plus footers.

    When ``query`` is given, a summary line naming the result count is prepended.
    Empty states (*index empty*, *no matches*, *all omitted*) return a single
    styled ``Text`` renderable instead.
    """
    if not response.results:
        if response.index_empty:
            return Text("Index is empty.", style=f"bold {COLOR.ERROR}")
        if response.total_matches:
            return Text(
                f"{response.total_matches} matches found; none fit the result limit/token budget.",
                style=COLOR.FOREGROUND_MUTED,
            )
        return Text("No matches found.", style=COLOR.FOREGROUND_MUTED)

    parts: list[RenderableType] = []
    if query:
        parts.append(_summary_line(query, response))
        parts.append(Text(""))

    parts.extend(render_result_card(card, index=i + 1) for i, card in enumerate(response.results))

    if response.omitted_count > 0:
        parts.append(
            Text(
                f"--- {response.omitted_count} results omitted (token budget) ---",
                style=COLOR.FOREGROUND_MUTED,
            ),
        )
    if response.clamped:
        parts.append(
            Text(
                "--- Results clamped to maximum ---",
                style=COLOR.FOREGROUND_MUTED,
            ),
        )

    return Group(*parts)


def render_episode_detail(detail: EpisodeDetail) -> RenderableType:
    """Render the full bounded prompt/response context for one episode as a Panel.

    The panel border carries the episode ref as its title. Inside, the episode
    title leads, then compact metadata (session/runtime, agent, files), then the
    prompt and response bodies in labeled sections. All transcript-derived
    fields are escaped; bodies render as markdown with hard breaks preserving
    line layout (tables, lists, emphasis, and fenced code).
    """
    parts: list[RenderableType] = []

    # --- Episode title line (rendered as proper Markdown heading) ---
    parts.append(RichMarkdown(f"## {textsafe.printable(detail.title)}"))
    parts.append(Text())

    # --- Compact metadata: session/runtime, agent, files ---
    meta = Text()
    meta.append("Session:", style=f"bold {COLOR.PRIMARY}")
    meta.append(f" {_esc(detail.session_id)}", style=COLOR.SECONDARY)
    meta.append("  \u00b7  ", style=COLOR.FOREGROUND_MUTED)
    meta.append("Runtime:", style=f"bold {COLOR.PRIMARY}")
    meta.append(f" {_esc(detail.runtime)}", style=COLOR.SECONDARY)
    parts.append(meta)

    if detail.is_subagent:
        sub = Text()
        sub.append("Agent:", style=f"bold {COLOR.PRIMARY}")
        sub.append(f" {_esc(detail.agent_name or 'unknown')}", style=COLOR.SECONDARY)
        if detail.agent_description:
            sub.append(" \u00b7 Task: ", style=COLOR.FOREGROUND_MUTED)
            sub.append(_esc(detail.agent_description), style=COLOR.SECONDARY)
        parts.append(sub)

    if detail.files_touched:
        files = Text()
        files.append("Files:", style=f"bold {COLOR.PRIMARY}")
        files.append(f" {_esc(', '.join(detail.files_touched))}", style=COLOR.SECONDARY)
        parts.append(files)

    # --- Prompt section ---
    parts.append(Text())
    parts.append(Rule(style=COLOR.FOREGROUND_MUTED))
    parts.append(Text("--- Prompt ---", style=f"bold {COLOR.ACCENT}"))
    if body := _markdown_body(detail.prompt_text):
        parts.append(body)
    else:
        parts.append(Text("  (no prompt)", style=f"italic {COLOR.FOREGROUND_MUTED}"))
    if detail.prompt_truncated:
        parts.append(Text("  [prompt truncated]", style=f"italic {COLOR.ERROR}"))

    # --- Response section ---
    parts.append(Text())
    parts.append(Rule(style=COLOR.FOREGROUND_MUTED))
    parts.append(Text("--- Response ---", style=f"bold {COLOR.ACCENT}"))
    if body := _markdown_body(detail.response_text):
        parts.append(body)
    else:
        parts.append(_response_placeholder(detail))
    if detail.response_truncated:
        parts.append(Text("  [response truncated]", style=f"italic {COLOR.ERROR}"))

    # --- Panel border: episode ref as title ---
    header = Text()
    header.append(f"[{_esc(detail.episode_id)}]", style=f"bold {COLOR.PRIMARY}")

    return Panel(
        Group(*parts),
        box=box.HEAVY,
        padding=(1, 2),
        title=header,
        title_align="left",
        border_style=COLOR.BORDER,
        safe_box=True,
    )
