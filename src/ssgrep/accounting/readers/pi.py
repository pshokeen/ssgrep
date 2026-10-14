"""Pi: ``~/.pi/agent/sessions``; parents only through ``parentSessionId`` (when present)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.pi_family import PiFamilyReader


class PiReader(PiFamilyReader):
    runtime: ClassVar[str] = "pi"

    def roots(self) -> list[Path]:
        return [roots.pi_sessions(self.env)]
