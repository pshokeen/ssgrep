"""Shared plumbing for the ``savings`` and ``metrics`` commands.

Lives outside ``cli/commands/`` because usecli imports every module in that
directory as a command module. Provides the JSON success envelope, the
``{ok, condition, message, command}`` error document, TTY detection, and the
terminal width used by renderers.
"""

from __future__ import annotations

import json
import os
import sys
from typing import NoReturn

from rich.console import Console
from usecli.cli.core.runtime import is_json_mode

from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views.types import NextAction
from ssgrep.cli import exit_codes

ENVELOPE_SCHEMA = 1


def terminal_width() -> int:
    return Console().width


def envelope(
    state: str,
    data: dict[str, object],
    *,
    warnings: tuple[str, ...] = (),
    next_actions: tuple[NextAction, ...] = (),
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """The success document body (usecli wraps it as ``{"ok": true, "data": ...}``)."""
    return {
        "schema_version": ENVELOPE_SCHEMA,
        "state": state,
        "data": data,
        "warnings": list(warnings),
        "next_actions": [{"command": a.command, "why": a.why} for a in next_actions],
        "provenance": provenance or {},
    }


def exit_with(condition: str, message: str, command: str, code: int) -> NoReturn:
    """Print the failure and exit; JSON mode bypasses usecli's exception wrapping."""
    if is_json_mode():
        document = {"ok": False, "condition": condition, "message": message, "command": command}
        stdout = sys.__stdout__ or sys.stdout
        stdout.write(json.dumps(document) + "\n")
        stdout.flush()
        os._exit(code)
    print(f"{message}\nTry: {command}" if command else message, file=sys.stderr)
    raise SystemExit(code)


def fail(error: PolicyError) -> NoReturn:
    exit_with(error.condition, str(error), error.command, error.exit_code)


def usage_error(condition: str, message: str, command: str) -> NoReturn:
    exit_with(condition, message, command, exit_codes.USAGE_ERROR)


def stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False
