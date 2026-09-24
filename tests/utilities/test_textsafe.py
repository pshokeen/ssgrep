"""Tests for honest terminal rendering of untrusted text."""

from __future__ import annotations

import pytest

from ssgrep.utilities.textsafe import printable


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ordinary ASCII", "ordinary ASCII"),
        ("café 😀", "café 😀"),
        ("line one\nline two\tindent", "line one\\nline two\\tindent"),
        ("before\x1b[2Jafter", "before\\x1b[2Jafter"),
        ("nul\x00delete\x7fzero-width\u200b", "nul\\x00delete\\x7fzero-width\\u200b"),
    ],
)
def test_printable_escapes_only_non_printable_characters(raw: str, expected: str) -> None:
    assert printable(raw) == expected


def test_printable_can_keep_selected_layout_characters() -> None:
    raw = "row one\n\trow two\r\x1b"
    assert printable(raw, keep="\n\t") == "row one\n\trow two\\r\\x1b"
    assert printable("") == ""
