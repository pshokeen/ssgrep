"""omp: ``~/.omp/agent/sessions``; children carry a raw ``parentSession`` file path.

Real omp transcripts mostly lack a useful ``rlmDepth`` yet carry ``parentSession``,
so the parent relation is resolved from that path (see ``PiFamilyReader.parent_link``)
independently of any retrieval-side parent metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ssgrep.accounting.readers import roots
from ssgrep.accounting.readers.pi_family import PiFamilyReader


class OmpReader(PiFamilyReader):
    runtime: ClassVar[str] = "omp"

    def roots(self) -> list[Path]:
        return [roots.omp_sessions(self.env)]
