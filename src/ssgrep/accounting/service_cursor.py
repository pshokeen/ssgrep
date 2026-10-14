"""Opaque page cursors bound to the data and selection they were issued for.

A cursor is ``<kind>1.<offset>.<tag>`` where ``tag`` is a SHA-256 prefix over
``(kind, offset, binding)``. ``binding`` is built by the caller from everything the page
depends on (snapshot identity, normalized selection, and a digest of the complete ordered
row set). Editing the offset, replaying a cursor on another selection, or following one after
the underlying rows changed all fail the check, so a page can never silently skip or repeat
rows. The tag is an integrity check, not a secret.
"""

from __future__ import annotations

import hashlib
import re

from ssgrep.accounting.policy_rules import PolicyError

_RE = re.compile(r"^([a-z]{1,8})1\.(\d{1,7})\.([0-9a-f]{16})$")


def _tag(kind: str, offset: int, binding: str) -> str:
    return hashlib.sha256(f"{kind}|{offset}|{binding}".encode()).hexdigest()[:16]


def issue(kind: str, offset: int, binding: str) -> str:
    return f"{kind}1.{offset}.{_tag(kind, offset, binding)}"


def redeem(kind: str, cursor: str | None, binding: str, command: str) -> int:
    """The row offset a valid cursor stands for (0 without a cursor); exit 2 otherwise."""
    if cursor is None:
        return 0
    match = _RE.fullmatch(cursor)
    if match is None or match.group(1) != kind:
        raise PolicyError("invalid_cursor", "That is not a cursor of this view.", command, 2)
    offset = int(match.group(2))
    if _tag(kind, offset, binding) != match.group(3):
        raise PolicyError(
            "invalid_cursor",
            "Cursor does not match this snapshot and selection (the data or the filters "
            "changed, or the cursor was edited). Start again without --cursor.",
            command,
            2,
        )
    return offset
