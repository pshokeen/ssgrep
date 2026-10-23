"""Quiet exit when the reader of stdout goes away (``ssgrep ... | head``).

Python's documented recipe: on ``BrokenPipeError`` point the stdout descriptor at
``/dev/null`` so the interpreter's exit-time flush cannot fail again, then exit
non-zero without a traceback. usecli writes its JSON document to whatever
``sys.stdout`` was when the command started, so the guard wraps that stream once at
start-up and covers text, JSON and Rich output alike.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, TextIO, cast

BROKEN_PIPE_EXIT = 1


def detach_stdout(stream: TextIO) -> None:
    """Point ``stream``'s descriptor at the null device; later writes succeed silently."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stream.fileno())
    finally:
        os.close(devnull)


class PipeSafeStdout:
    """Delegating stdout whose write/flush turn a closed pipe into a quiet exit."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def _guard(self, call: Callable[..., Any], *args: object) -> Any:
        try:
            return call(*args)
        except BrokenPipeError:
            detach_stdout(self._stream)
            raise SystemExit(BROKEN_PIPE_EXIT) from None

    def write(self, text: str) -> int:
        return self._guard(self._stream.write, text)

    def flush(self) -> None:
        self._guard(self._stream.flush)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def install_pipe_guard() -> None:
    """Wrap ``sys.stdout`` once; a second call keeps the existing wrapper."""
    if not isinstance(sys.stdout, PipeSafeStdout):
        sys.stdout = cast(TextIO, PipeSafeStdout(sys.stdout))
