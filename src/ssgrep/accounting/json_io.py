"""Bounded, float-free JSON reads of existing accounting files (never creates)."""

from __future__ import annotations

import json
from typing import Any

from ssgrep.accounting import plainfile
from ssgrep.store.paths import data_dir

MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024


def _reject_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def load_json(profile: str, *parts: str, limit: int) -> Any | None:
    """Parsed JSON under ``accounting/<profile>/``; None when the file is absent."""
    raw = plainfile.read_private(data_dir(), "accounting", profile, *parts, max_bytes=limit)
    return None if raw is None else json.loads(raw, parse_float=_reject_float)
