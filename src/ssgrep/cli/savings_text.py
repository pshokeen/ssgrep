"""Width-aware plain-text layout shared by the savings and metrics renderers.

Everything untrusted goes through ``safe`` (non-printables are escaped, never
dropped) and into ``rich.text.Text`` objects, which never parse markup. Color
is decoration only: every status is spelled out in words, and signs, units and
reasons are part of the text itself, so NO_COLOR and pipes lose nothing.
"""

from __future__ import annotations

import shlex
import textwrap

from rich.text import Text

from ssgrep.accounting.contracts import Quantity
from ssgrep.utilities.textsafe import printable

NARROW = 56
MAX_WIDTH = 100
MIN_WIDTH = 30
LABEL_WIDTH = 16


def safe(value: object) -> str:
    """Untrusted text with controls escaped and Rich-markup brackets left inert."""
    return printable(str(value))


def fmt_int(value: int, *, signed: bool = False) -> str:
    """Locale-independent integer; ``signed`` always shows ``+`` or ``-``."""
    return f"{value:+,}" if signed else f"{value:,}"


def fmt_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    for unit, shift in (("KiB", 10), ("MiB", 20), ("GiB", 30)):
        if value < 1 << (shift + 10) or unit == "GiB":
            tenths = value * 10 >> shift
            return f"{tenths // 10}.{tenths % 10} {unit}"
    raise AssertionError("unreachable")  # pragma: no cover


def fmt_age(seconds: int | None) -> str:
    if seconds is None:
        return "age unknown"
    if seconds < 120:
        return f"{seconds}s ago"
    if seconds < 7200:
        return f"{seconds // 60}m ago"
    if seconds < 172800:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


DISPLAY_UNITS = {"context_tokens": "tokens", "work_seconds": "s"}


def display_unit(unit: str) -> str:
    """Short, consistent unit label (``context_tokens`` -> ``tokens``)."""
    return DISPLAY_UNITS.get(unit, safe(unit))


def fmt_quantity(quantity: Quantity, *, signed: bool = False) -> str:
    """``value unit`` or ``unavailable (reason)``; never a bare zero for unknown."""
    if quantity.value is None:
        return f"unavailable ({safe(quantity.reason)})"
    if quantity.unit == "bytes":
        return fmt_bytes(quantity.value)
    return f"{fmt_int(quantity.value, signed=signed)} {display_unit(quantity.unit)}"


class Doc:
    """Accumulates wrapped lines for one terminal width."""

    def __init__(self, width: int) -> None:
        self.width = min(max(width, MIN_WIDTH), MAX_WIDTH)
        self.narrow = self.width < NARROW
        self._lines: list[Text] = []

    def blank(self) -> None:
        if self._lines and self._lines[-1].plain:
            self._lines.append(Text(""))

    def line(self, text: str, *, indent: int = 0, style: str = "") -> None:
        """Wrapped paragraph; continuation lines keep ``indent``."""
        pad = " " * indent
        wrapped = textwrap.wrap(
            text,
            self.width,
            initial_indent=pad,
            subsequent_indent=pad,
            break_long_words=True,
            break_on_hyphens=False,
        )
        for piece in wrapped or [pad.rstrip()]:
            self._lines.append(Text(piece, style=style))

    def command(self, label: str, command: str, *, indent: int = 0) -> None:
        """Keep a command shell-pasteable while fitting the terminal width.

        Long arguments concatenate independently quoted chunks across escaped newlines.
        Those continuations have no indentation: spaces there would change the argument.
        """
        head = " " * indent + (f"{label} " if label else "")
        pad = " " * len(head)
        lines: list[str] = []
        current = head
        has_argument = False
        for argument in shlex.split(command):
            word = shlex.quote(argument)
            separator = " " if has_argument else ""
            if len(current) + len(separator) + len(word) + 2 > self.width and has_argument:
                lines.append(current + " \\")
                current, separator = pad, ""
            current += separator
            remaining = argument
            while len(current) + len(shlex.quote(remaining)) + 2 > self.width:
                available = self.width - len(current) - 2
                end = min(len(remaining), available)
                while end > 0 and len(shlex.quote(remaining[:end])) > available:
                    end -= 1
                assert end > 0, "command label leaves no room for a shell-quoted character"
                lines.append(current + shlex.quote(remaining[:end]) + "\\")
                remaining = remaining[end:]
                current = ""
            current += shlex.quote(remaining)
            has_argument = True
        lines.append(current)
        self._lines.extend(Text(line) for line in lines)

    def bullet(self, text: str, *, indent: int = 2) -> None:
        """``- item`` whose wrapped lines hang under the text."""
        pad = " " * indent
        for number, piece in enumerate(
            textwrap.wrap(
                text, self.width - indent - 2, break_long_words=True, break_on_hyphens=False
            )
        ):
            self._lines.append(Text(f"{pad}{'- ' if number == 0 else '  '}{piece}"))

    def heading(self, text: str, note: str = "") -> None:
        """Section title (bold) with an optional explanatory line under it."""
        self.blank()
        self._lines.append(Text(text, style="bold"))
        if note:
            self.line(note, indent=2, style="dim")

    def pair(self, label: str, value: str, *, indent: int = 2, style: str = "") -> None:
        """``label  value``; narrow terminals stack the value under the label."""
        if self.narrow or len(label) + indent >= self.width // 2:
            inline = f"{label}: {value}"
            if indent + len(inline) <= self.width:
                self.line(inline, indent=indent, style=style)
                return
            self.line(label, indent=indent)
            self.line(value, indent=indent + 2, style=style)
            return
        pad = " " * indent
        head = f"{pad}{label:<{LABEL_WIDTH}} "
        body = textwrap.wrap(
            value,
            self.width - len(head),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        self._lines.append(Text(head + body[0], style=style))
        for piece in body[1:]:
            self._lines.append(Text(" " * len(head) + piece, style=style))

    def text(self) -> Text:
        while self._lines and not self._lines[-1].plain:
            self._lines.pop()
        return Text("\n").join(self._lines)
