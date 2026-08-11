"""Escaping for untrusted transcript-derived text bound for a terminal.

Transcript content (titles, excerpts, prompt/response bodies, agent fields,
recorded cwds) is attacker-influenced input that text-mode rendering
interpolates straight into output the buyer is asked to trust and copy from.
Raw ESC bytes let embedded sequences like ``\x1b[2J`` clear the screen or
overwrite the genuine text above them; newlines let a value forge whole extra
lines of UI.

Escaping rather than stripping keeps the line honest: the reader still sees
that the recorded text contains something odd, instead of a silently
different string. JSON and MCP output are never escaped -- machine consumers
get the original bytes.

This is the shared home of the escaping ``scope_report._display()``
established for recorded cwds; ``render.py`` applies it at every text-mode
interpolation point.
"""

from __future__ import annotations


def printable(value: str, *, keep: str = "") -> str:
    """``value`` with every non-printable character backslash-escaped.

    ``keep`` names characters to pass through verbatim even though
    ``str.isprintable()`` rejects them -- pass ``"\n\t"`` for multi-line
    body text whose genuine newlines and tabs must survive.
    """
    return "".join(c if c.isprintable() or c in keep else repr(c)[1:-1] for c in value)
