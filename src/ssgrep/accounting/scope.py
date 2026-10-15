"""Validation of savings selection arguments, done before any side effect.

Intervals are UTC half-open ``[since, until)``. Accepted spellings: RFC3339
with ``Z`` or a numeric offset, or a bare ``YYYY-MM-DD`` (midnight UTC).
Everything invalid raises ``PolicyError`` with exit code 2.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

from ssgrep.accounting.contracts import RUNTIMES
from ssgrep.accounting.ledger_contracts import valid_profile_id
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.service_types import Scope

MAX_LIMIT = 100
BY_CHOICES = ("task", "runtime", "project", "day")
_MAX_FILTER = 512
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,9})?)?(Z|[+-]\d{2}:\d{2})$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _usage(condition: str, message: str, command: str = "ssgrep savings --help") -> PolicyError:
    return PolicyError(condition, message, command, exit_code=2)


def parse_instant(text: str, name: str) -> str:
    """Normalize an RFC3339 or YYYY-MM-DD boundary to ``YYYY-MM-DDTHH:MM:SSZ``."""
    try:
        if _DAY.fullmatch(text):
            moment = datetime.combine(date.fromisoformat(text), datetime.min.time(), tzinfo=UTC)
        elif _RFC3339.fullmatch(text):
            moment = datetime.fromisoformat(text).astimezone(UTC)
        else:
            raise ValueError(text)
    except ValueError:
        raise _usage(
            "invalid_interval",
            f"{name} must be RFC3339 (2026-10-01T00:00:00Z) or YYYY-MM-DD, not {text[:40]!r}.",
        ) from None
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def check_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_LIMIT:
        raise _usage("invalid_limit", f"--limit must be between 1 and {MAX_LIMIT}, not {limit}.")
    return limit


def check_profile(profile: str) -> str:
    if not valid_profile_id(profile):
        raise _usage(
            "invalid_profile",
            "Profile ids are 1-32 characters of a-z, 0-9, '-' or '_' and never paths.",
            "ssgrep metrics status",
        )
    return profile


def _bounded(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not value or len(value) > _MAX_FILTER or "\0" in value:
        raise _usage(
            "invalid_filter", f"{name} must be a non-empty value under {_MAX_FILTER} bytes."
        )
    return value


def interval(since: str | None, until: str | None) -> tuple[str | None, str | None]:
    start = parse_instant(since, "--since") if since else None
    end = parse_instant(until, "--until") if until else None
    if start and end and start >= end:
        raise _usage("invalid_interval", "--since must be earlier than --until (half-open UTC).")
    return start, end


def build_scope(
    *,
    since: str | None = None,
    until: str | None = None,
    project: str | None = None,
    runtime: str | None = None,
    session: str | None = None,
    task: str | None = None,
    by: str | None = None,
    include_children: bool = False,
) -> Scope:
    start, end = interval(since, until)
    if runtime is not None and runtime not in RUNTIMES:
        raise _usage("invalid_filter", f"--runtime must be one of: {', '.join(RUNTIMES)}.")
    if by is not None and by not in BY_CHOICES:
        raise _usage("invalid_filter", f"--by must be one of: {', '.join(BY_CHOICES)}.")
    if include_children and not session:
        raise _usage("invalid_filter", "--include-children only applies together with --session.")
    return Scope(
        window="7d",
        since=start,
        until=end,
        project=_bounded(project, "--project"),
        runtime=runtime,
        session=_bounded(session, "--session"),
        task=_bounded(task, "--task"),
        by=by,
        include_children=include_children,
    )
