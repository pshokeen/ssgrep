"""Read-only diagnostic DTOs; private messages are opt-in, identities are stable."""

from __future__ import annotations

from dataclasses import dataclass

from ssgrep.accounting.service_types import Scope


@dataclass(frozen=True)
class DiagnosticRecord:
    record_id: str
    code: str
    severity: str
    source: str
    message: str | None = None


@dataclass(frozen=True)
class DiagnosticGroup:
    code: str
    severity: str
    count: int


@dataclass(frozen=True)
class DiagnosticPage:
    profile: str
    scope: Scope
    snapshot_id: str | None
    generation: int | None
    records: tuple[DiagnosticRecord, ...]
    total: int
    alias_duplicates: int
    next_cursor: str | None
    limit: int = 20
    code: str | None = None
    include_text: bool = False
    reason: str | None = None
