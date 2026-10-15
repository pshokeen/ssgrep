"""Stable warning identities and bounded category summaries, without storage reads.

Repeated identical warnings are aliases, not extra records. Distinct native
subjects stay distinct even if they have the same diagnostic code. Messages
remain private unless the caller explicitly requests them.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable

from ssgrep.accounting.diagnostic_types import DiagnosticGroup, DiagnosticRecord

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEVERE = frozenset(
    {
        "corrupt",
        "storage_corrupt",
        "snapshot_unreadable",
        "unsupported_schema",
        "blocked",
        "credentials_expired",
        "unreadable",
        "policy_invalid",
        "identity_conflict",
        "replay_unavailable_sessions",
        "frequency_revision_unavailable",
    }
)
_RANK = {"error": 0, "warning": 1, "info": 2}


def warning_code(message: str) -> str:
    parts = message.split(":", 2)
    if len(parts) > 1 and parts[0] == "lineage" and _CODE.fullmatch(parts[1]):
        return f"lineage.{parts[1]}"
    return parts[0] if _CODE.fullmatch(parts[0]) else "diagnostic_note"


def severity(code: str) -> str:
    if code in _SEVERE or any(word in code for word in ("corrupt", "conflict", "cyclic", "denied")):
        return "error"
    return "warning"


def warning_records(
    warnings: Iterable[str], *, include_text: bool = False
) -> tuple[tuple[DiagnosticRecord, ...], int]:
    records: dict[str, DiagnosticRecord] = {}
    duplicates = 0
    for message in warnings:
        code = warning_code(message)
        identity = hashlib.sha256(f"warning\0{code}\0{message}".encode()).hexdigest()
        if identity in records:
            duplicates += 1
            continue
        records[identity] = DiagnosticRecord(
            identity, code, severity(code), "snapshot_warning", message if include_text else None
        )
    return tuple(
        sorted(records.values(), key=lambda r: (_RANK[r.severity], r.code, r.record_id))
    ), duplicates


def grouped_warnings(warnings: Iterable[str]) -> tuple[DiagnosticGroup, ...]:
    unique = set(warnings)
    counts = Counter(warning_code(message) for message in unique)
    return tuple(
        DiagnosticGroup(code, severity(code), count)
        for code, count in sorted(
            counts.items(), key=lambda item: (_RANK[severity(item[0])], item[0])
        )
    )
