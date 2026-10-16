"""Builders and guards for accounting tests.

Builders return *complete valid* instances (mirrors ``tests/conftest.py``) so
assertions run against real values, not field-existence proxies. Nothing here
reads the maintainer's data: every path is under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.ledger.contracts import (
    ExecutionEvent,
    Operation,
    Outcome,
    Phase,
    Quantity,
    RawEvidence,
    Transport,
    digest_of,
)

REAL_HOME = Path(os.path.expanduser("~"))


def build_event(**overrides: object) -> ExecutionEvent:
    """A complete valid ``completed`` search event."""
    fields: dict[str, Any] = {
        "execution_id": "n1-0001",
        "phase": Phase.COMPLETED,
        "operation": Operation.SEARCH,
        "transport": Transport.CLI,
        "profile": "personal-local",
        "ts_utc": "2026-11-01T07:00:00Z",
        "cwd": "/work/project",
        "fingerprint": "f" * 64,
        "version": "2.0.1",
        "outcome": Outcome.OK,
        "mono_ns": 2_000_000,
        "refs": ("claude:s1:ep:0",),
        "ranks_scores": ((1, 900),),
        "limit": 10,
        "token_budget": 1500,
        "omitted_count": 0,
        "truncated": False,
        "rendered_bytes": Quantity.observed(812, "bytes", "mcp_json"),
        "index_generation": "gen-1",
    }
    fields.update(overrides)
    return ExecutionEvent(**fields)


def build_evidence(**overrides: object) -> RawEvidence:
    """A complete valid raw tool-call record with a self-consistent digest."""
    payload = overrides.pop("payload", {"name": "bash", "arguments": {"command": "ls"}})
    fields: dict[str, Any] = {
        "runtime": "claude",
        "source_id": "claude:/p/s1.jsonl",
        "branch_id": "main",
        "session_id": "s1",
        "record_id": "toolu_1",
        "locator": "offset:0",
        "digest": digest_of(payload),
        "ts_utc": "2026-11-01T07:00:01Z",
        "kind": "tool_call",
        "payload": payload,
    }
    fields.update(overrides)
    return RawEvidence(**fields)


@pytest.fixture
def write_jsonl(tmp_path: Path) -> Callable[[str, list[dict]], Path]:
    """Write newline-complete JSONL beneath tmp_path and return its path."""

    def _write(relative: str, records: list[dict]) -> Path:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r) + "\n" for r in records))
        return target

    return _write


@pytest.fixture
def accounting_root(tmp_path: Path) -> Path:
    """The isolated accounting directory for the current test (not created)."""
    return tmp_path / "data" / "accounting"
