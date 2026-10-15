"""Width-aware plain-text layout shared by the savings and metrics renderers.

Everything untrusted goes through ``safe`` (non-printables are escaped, never
dropped) and into ``rich.text.Text`` objects, which never parse markup. Color
is decoration only: every status is spelled out in words, and signs, units and
reasons are part of the text itself, so NO_COLOR and pipes lose nothing.
"""

from __future__ import annotations

import shlex
import textwrap

from rich.cells import cell_len
from rich.text import Text

from ssgrep.accounting.contracts import Quantity
from ssgrep.accounting.ledger_contracts import ScenarioEstimate
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


DISPLAY_UNITS = {"context_tokens": "tokens", "work_seconds": "s", "work_ms": "ms"}


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


def _fold_cells(text: str, width: int) -> list[str]:
    """Fold an already character-bounded line to its actual terminal cell width."""
    pieces = []
    while cell_len(text) > width:
        end = min(len(text), width)
        while end and cell_len(text[:end]) > width:
            end -= 1
        if not end:
            raise ValueError("indentation leaves no room for a terminal character")
        space = text.rfind(" ", 0, end + 1)
        if space > 0:
            end = space
        pieces.append(text[:end])
        text = text[end:].lstrip()
    return [*pieces, text]


class Doc:
    """Accumulates wrapped lines for one terminal width."""

    def __init__(self, width: int, *, compact: bool = False) -> None:
        self.width = min(max(width, MIN_WIDTH), MAX_WIDTH)
        self.narrow = self.width < NARROW
        self.compact = compact
        self._lines: list[Text] = []

    def blank(self) -> None:
        if not self.compact and self._lines and self._lines[-1].plain:
            self._lines.append(Text(""))

    def line(self, text: str, *, indent: int = 0, style: str = "") -> None:
        """Wrapped paragraph; continuation lines keep ``indent``."""
        text = safe(text)
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
            if cell_len(piece) <= self.width:
                self._lines.append(Text(piece, style=style))
            else:
                self._lines.extend(
                    Text(pad + part, style=style)
                    for part in _fold_cells(piece.strip(), self.width - indent)
                )

    def command(self, label: str, command: str, *, indent: int = 0) -> None:
        """Keep the complete labeled block shell-pasteable within the terminal width.

        Labels are comments, not command words. Long arguments concatenate independently
        quoted chunks across escaped newlines; those continuations have no indentation.
        """
        head = " " * indent
        if label:
            prefix = head + "# "
            comments = textwrap.wrap(
                safe(label), self.width, initial_indent=prefix, subsequent_indent=prefix
            )
            self._lines.extend(Text(line, style="dim") for line in comments)
        pad = head
        lines: list[str] = []
        current = head
        has_argument = False
        for argument in shlex.split(command):
            word = shlex.quote(safe(argument))
            separator = " " if has_argument else ""
            if (
                cell_len(current) + len(separator) + cell_len(word) + 2 > self.width
                and has_argument
            ):
                lines.append(current + " \\")
                current, separator = pad, ""
            current += separator
            remaining = safe(argument)
            while cell_len(current) + cell_len(shlex.quote(remaining)) + 2 > self.width:
                available = self.width - cell_len(current) - 2
                end = min(len(remaining), available)
                while end > 0 and cell_len(shlex.quote(remaining[:end])) > available:
                    end -= 1
                if end <= 0:
                    raise ValueError(
                        "command indentation leaves no room for a shell-quoted character"
                    )
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
        label, value = safe(label), safe(value)
        if self.narrow or cell_len(label) + indent >= self.width // 2:
            inline = f"{label}: {value}"
            if indent + cell_len(inline) <= self.width:
                self.line(inline, indent=indent, style=style)
                return
            self.line(label, indent=indent)
            self.line(value, indent=indent + 2, style=style)
            return
        pad = " " * indent
        head = f"{pad}{label:<{LABEL_WIDTH}} "
        body = textwrap.wrap(
            value,
            self.width - cell_len(head),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        body = [part for piece in body for part in _fold_cells(piece, self.width - cell_len(head))]
        self._lines.append(Text(head + body[0], style=style))
        for piece in body[1:]:
            self._lines.append(Text(" " * cell_len(head) + piece, style=style))

    def text(self) -> Text:
        while self._lines and not self._lines[-1].plain:
            self._lines.pop()
        return Text("\n").join(self._lines)


def fmt_estimate_net(quantity: Quantity, estimate: ScenarioEstimate | None) -> str:
    """Incomplete nets expose only an explicitly marked known subtotal."""
    if estimate is not None and not estimate.complete:
        if estimate.known_net is None:
            return "unavailable*"
        return f"{fmt_int(estimate.known_net, signed=True)} {display_unit(quantity.unit)}*"
    return fmt_quantity(quantity, signed=True) if quantity.value is not None else "unavailable"


def render_estimate_bounds(
    doc: Doc, estimates: tuple[ScenarioEstimate, ...], *, include_complete: bool = False
) -> None:
    incomplete = tuple(
        estimate
        for estimate in estimates
        if include_complete
        or not estimate.complete
        or (
            estimate.bounds is not None
            and estimate.bounds.net_lower.value != estimate.bounds.net_upper.value
        )
    )
    if not incomplete:
        return
    if any(estimate.complete for estimate in incomplete):
        doc.line("Point model floors components; bounds enclose raw quantities.", indent=2)
    if all(estimate.bounds is None for estimate in incomplete):
        doc.line("Lower bound unavailable; upper bound unavailable (not recorded).", indent=2)
        return
    for label, field in (("Lower bound", "net_lower"), ("Upper bound", "net_upper")):
        values = tuple(
            getattr(estimate.bounds, field) if estimate.bounds else None for estimate in incomplete
        )
        if all(value is None or value.value is None for value in values):
            reasons = sorted(
                {value.reason for value in values if value is not None and value.reason}
            )
            doc.line(
                f"{label} unavailable"
                + (
                    f": {', '.join(safe(reason) for reason in reasons)}"
                    if reasons
                    else ": bounds not recorded"
                ),
                indent=2,
            )
        else:
            cells = [
                fmt_quantity(value, signed=True) if value is not None else "unavailable"
                for value in values
            ]
            if len(cells) == 1:
                doc.line(f"{label} {cells[0]} ({incomplete[0].scenario.value})", indent=2)
            else:
                doc.line(
                    f"{label} "
                    f"({' / '.join(estimate.scenario.value for estimate in incomplete)}): "
                    f"{' / '.join(cells)}",
                    indent=2,
                )


def render_estimate_details(doc: Doc, estimates: tuple[ScenarioEstimate, ...]) -> None:
    """Detailed local views retain metadata without summing incompatible estimates."""
    for estimate in estimates:
        doc.heading(f"{estimate.scenario.value.upper()} ESTIMATE")
        doc.pair(
            "Complete net",
            fmt_quantity(estimate.net, signed=True) if estimate.complete else "unavailable",
        )
        known = (
            "unavailable"
            if estimate.known_net is None
            else f"{fmt_int(estimate.known_net, signed=True)} {display_unit(estimate.net.unit)}"
        )
        doc.pair("Known net subtotal", known)
        doc.pair("Credit subtotal", fmt_quantity(estimate.credit, signed=True))
        doc.pair("Debit subtotal", fmt_quantity(estimate.debit))
        render_estimate_bounds(doc, (estimate,), include_complete=True)
        doc.pair(
            "Coverage",
            f"{estimate.known_atoms} known atoms; {estimate.known_debits} known debits; "
            f"{len(estimate.credit_missing)} missing sources; "
            f"{len(estimate.debit_missing)} missing debits",
        )
        for kind, items in (
            ("Missing source", estimate.credit_missing),
            ("Missing debit", estimate.debit_missing),
            ("Eligibility", estimate.eligibility),
            ("Provenance", estimate.provenance),
        ):
            for identity, reason in items:
                doc.pair(kind, f"{safe(identity)}: {safe(reason)}")
        if estimate.bounds:
            doc.line("Bounds are deterministic, not confidence intervals.", indent=2)
            for assumption in estimate.bounds.assumptions:
                doc.line(f"Assumption: {safe(assumption)}", indent=2)
        if estimate.unit:
            doc.pair("Unit identity", safe(estimate.unit))
