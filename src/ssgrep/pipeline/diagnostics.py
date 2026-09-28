"""Process-local parse diagnostics collected from traced components.

CocoIndex memoizes components by their declared inputs, so a component body
runs exactly when its source actually needs (re)processing. Recording parse
degradation there gives the same semantic as the old per-file counters: only
work done this run counts.
"""

from __future__ import annotations

import threading


class RunDiagnostics:
    """Thread-safe counters for one index run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._malformed = 0
        self._skipped = 0
        self._archived = 0

    def record(self, *, malformed: int = 0, skipped: int = 0, archived: int = 0) -> None:
        with self._lock:
            self._malformed += malformed
            self._skipped += skipped
            self._archived += archived

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self._malformed, self._skipped, self._archived

    def reset(self) -> None:
        with self._lock:
            self._malformed = 0
            self._skipped = 0
            self._archived = 0


#: One shared instance per CLI process; reset at the start of each run.
current = RunDiagnostics()

__all__ = ["RunDiagnostics", "current"]
